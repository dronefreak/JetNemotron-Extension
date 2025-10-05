# jet_nemotron_nvidia/hybrid_model.py
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from jet_nemotron_nvidia.jetblock import JetBlockAttention

class PretrainedJetBackbone(nn.Module):
    """
    Generic module:
        Pretrained backbone → JetBlockAttention → output features
    """

    def __init__(
        self,
        backbone_name="resnet18",
        pretrained=True,
        jet_embed_dim=512,
        n_heads=8,
        qk_dim=64,
        v_dim=64,
        use_rope=True,
    ):
        super().__init__()

        # 1. Load pretrained backbone
        if backbone_name.startswith("resnet"):
            backbone = getattr(models, backbone_name)(pretrained=pretrained)
            self.backbone = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            )
            backbone_out_dim = backbone.fc.in_features  # typically 512 or 2048
        elif backbone_name.startswith("mobilenet"):
            backbone = getattr(models, backbone_name)(pretrained=pretrained)
            self.backbone = backbone.features
            backbone_out_dim = backbone.last_channel  # 1280 for MobileNetV2
        else:
            raise ValueError("Unsupported backbone")

        # 2. Project backbone features to JetBlock embedding dim
        self.proj = nn.Conv2d(backbone_out_dim, jet_embed_dim, kernel_size=1)

        # 3. JetBlockAttention
        self.jetblock = JetBlockAttention(
            hidden_size=jet_embed_dim,
            n_heads=n_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            use_rope=use_rope,
        )

        self.jet_embed_dim = jet_embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        # Backbone feature map
        feat = self.backbone(x)  # [B, Cb, Hb, Wb]
        feat = self.proj(feat)  # [B, jet_embed_dim, Hb, Wb]

        # Flatten to sequence for JetBlock
        B, D, Hf, Wf = feat.shape
        seq = feat.flatten(2).transpose(1, 2)  # [B, N_patches, D]

        # JetBlock processing
        seq = self.jetblock(seq)  # [B, N_patches, D]
        return seq, (Hf, Wf)  # Return sequence + spatial size


class TaskDecoder(nn.Module):
    def __init__(self, embed_dim, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, out_channels, 1),
        )

    def forward(self, seq, spatial_size):
        Hf, Wf = spatial_size
        B, N, D = seq.shape
        x = seq.transpose(1, 2).view(B, D, Hf, Wf)
        # Upsample to original image resolution if needed
        x = F.interpolate(x, scale_factor=1, mode="bilinear", align_corners=False)
        out = self.conv(x)
        return out


class HybridJetModel(nn.Module):
    def __init__(self,
                 backbone_name='resnet18',
                 pretrained_backbone=True,
                 jet_embed_dim=512,
                 n_heads=8,
                 qk_dim=64,
                 v_dim=64,
                 n_jetblocks=1,
                 use_rope=True,
                 out_channels=21):
        super().__init__()

        # -------- Pretrained Backbone --------
        if backbone_name.startswith('resnet'):
            backbone = getattr(models, backbone_name)(pretrained=pretrained_backbone)
            self.backbone = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4
            )
            backbone_out_dim = backbone.fc.in_features
        elif backbone_name.startswith('mobilenet'):
            backbone = getattr(models, backbone_name)(pretrained=pretrained_backbone)
            self.backbone = backbone.features
            backbone_out_dim = backbone.last_channel
        else:
            raise ValueError("Unsupported backbone")

        # -------- Projection to JetBlock embedding dim --------
        self.proj = nn.Conv2d(backbone_out_dim, jet_embed_dim, kernel_size=1)

        # -------- Stack of JetBlocks --------
        self.jetblocks = nn.ModuleList([
            JetBlockAttention(
                hidden_size=jet_embed_dim,
                n_heads=n_heads,
                qk_dim=qk_dim,
                v_dim=v_dim,
                use_rope=use_rope
            ) for _ in range(n_jetblocks)
        ])

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.Conv2d(jet_embed_dim, jet_embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(jet_embed_dim, out_channels, kernel_size=1)
        )

        self.jet_embed_dim = jet_embed_dim

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Backbone features
        feat = self.backbone(x)         # [B, Cb, Hf, Wf]
        feat = self.proj(feat)          # [B, D, Hf, Wf]
        Hf, Wf = feat.shape[2], feat.shape[3]

        # 2. Flatten to sequence for JetBlock
        seq = feat.flatten(2).transpose(1, 2)  # [B, N_patches, D]

        # 3. Pass through JetBlocks
        for jet in self.jetblocks:
            seq = jet(seq)

        # 4. Reshape sequence back to spatial grid
        x = seq.transpose(1, 2).view(B, self.jet_embed_dim, Hf, Wf)

        # 5. Decoder to task-specific output
        out = self.decoder(x)
        # Optional: upsample to original image size
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out
