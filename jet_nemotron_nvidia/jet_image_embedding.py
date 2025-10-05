import torch
import torch.nn as nn
import torch.nn.functional as F
from jet_nemotron_nvidia.jetblock import JetBlockAttention

class JetImageEmbed(nn.Module):
    """Converts images into a sequence of embeddings suitable for JetBlockAttention.

    Can be used as a backbone for classification, segmentation, depth estimation, etc.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        n_heads=8,
        qk_dim=64,
        v_dim=64,
        use_rope=True,
    ):
        super().__init__()

        # Patch embedding
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # JetBlockAttention
        self.jetblock = JetBlockAttention(
            hidden_size=embed_dim,
            n_heads=n_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            use_rope=use_rope,
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, N_patches, embed_dim] sequence
        """
        # Convert to patches: [B, embed_dim, H/ps, W/ps]
        x = self.patch_embed(x)
        # Flatten to sequence: [B, N_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        # Feed through JetBlock
        x = self.jetblock(x)
        return x


class PatchDecoder(nn.Module):
    """Converts patch embeddings from a JetBlock backbone back to dense image
    predictions.

    Can be used for semantic segmentation, depth estimation, etc.
    """

    def __init__(self, embed_dim, out_channels, patch_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Optional: a simple conv head to refine features per patch
        self.conv_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        )

    def forward(self, x, img_size):
        """
        x: [B, N_patches, embed_dim] sequence from JetBlock
        img_size: (H, W) of original image
        returns: [B, out_channels, H, W] dense output
        """
        B, N, D = x.shape
        H, W = img_size
        H_p, W_p = H // self.patch_size, W // self.patch_size

        # Reshape sequence to grid
        x = x.transpose(1, 2).contiguous()  # [B, D, N]
        x = x.view(B, D, H_p, W_p)  # [B, D, H_p, W_p]

        # Upsample to original resolution
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        # Apply refinement head
        out = self.conv_head(x)  # [B, out_channels, H, W]
        return out