import torch
import torch.nn as nn
import torch.nn.functional as F
from jet_nemotron_nvidia.jetblock import JetBlockAttention

class JetVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        n_heads=8,
        qk_dim=64,
        v_dim=64,
        num_classes=1000,
    ):
        super().__init__()
        # 1. Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Optional CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. JetBlockAttention
        self.jetblock = JetBlockAttention(
            hidden_size=embed_dim,
            n_heads=n_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            use_rope=True,
        )

        # 4. Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # [B, C, H, W] → [B, embed_dim, H/ps, W/ps]
        x = self.patch_embed(x)
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, embed_dim]

        # JetBlock
        x = self.jetblock(x)  # [B, 1+N, embed_dim]

        # Classification
        cls_out = x[:, 0]  # take CLS token
        cls_out = self.norm(cls_out)
        return self.fc(cls_out)