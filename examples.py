"""
example.py
-----------
Demo script showing different ways to use JetBlocks and JetNemotron-Extension.
Run this file directly to see output tensor shapes for various model combinations.
"""

from jet_nemotron_nvidia.hybrid_model import HybridJetModel
from jet_nemotron_nvidia.jet_image_embedding import JetImageEmbed, PatchDecoder
from jet_nemotron_nvidia.jetblock import JetBlockAttention
import torch


# ------------------------------------------------------
# A simple utility to get device (CPU or GPU)
# ------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------
# A simple utility to count model parameters
# ------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000


# ------------------------------------------------------
# 1. Raw JetBlock on sequence data
# ------------------------------------------------------
def example_jetblock():
    print("\n--- Example 1: Raw JetBlock on sequences ---")
    x = torch.randn(2, 16, 1536).to(get_device())  # [batch, seq_len, hidden_size]
    block = JetBlockAttention(
        hidden_size=1536, n_heads=12, qk_dim=96, v_dim=256, kernel_size=4, use_rope=True
    ).to(get_device())
    out = block(x).to(get_device())
    print("Input:", x.shape, "Output:", out.shape)
    print("JetBlock params (M):", count_parameters(block))


# ------------------------------------------------------
# 2. JetBlock with Image Patch Embedding
# ------------------------------------------------------
def example_patch_embedding():
    print("\n--- Example 2: JetBlock with patch embedding ---")
    images = torch.randn(2, 3, 224, 224).to(get_device())
    embed = JetImageEmbed(img_size=224, patch_size=16, embed_dim=1536).to(get_device())
    jetblock = JetBlockAttention(
        hidden_size=1536, n_heads=12, qk_dim=96, v_dim=256, kernel_size=4, use_rope=True
    ).to(get_device())

    patches = embed(images).to(get_device())
    out = jetblock(patches).to(get_device())
    print("Input images:", images.shape)
    print("Patch embeddings:", patches.shape)
    print("Output after JetBlock:", out.shape)
    print("PatchEmbed params (M):", count_parameters(embed))
    print("JetBlock params (M):", count_parameters(jetblock))


# ------------------------------------------------------
# 3. Patch embeddings + JetBlocks + Decoder
# ------------------------------------------------------
def example_with_decoder():
    print("\n--- Example 3: End-to-end with decoder ---")
    images = torch.randn(2, 3, 224, 224).to(get_device())

    # Patch encoder
    embed = JetImageEmbed(img_size=224, patch_size=16, embed_dim=1536).to(get_device())
    jetblock = JetBlockAttention(
        hidden_size=1536, n_heads=12, qk_dim=96, v_dim=256, kernel_size=4, use_rope=True
    ).to(get_device())
    decoder = PatchDecoder(embed_dim=1536, out_channels=21, patch_size=16).to(
        get_device()
    )  # segmentation

    patches = embed(images).to(get_device())
    features = jetblock(patches).to(get_device())
    preds = decoder(features, img_size=(224, 224)).to(get_device())

    print("Input images:", images.shape)
    print("Dense predictions:", preds.shape)  # [B, out_channels, H, W]
    print("PatchEmbed params (M):", count_parameters(embed))
    print("JetBlock params (M):", count_parameters(jetblock))
    print("Decoder params (M):", count_parameters(decoder))


# ------------------------------------------------------
# 4. Hybrid model with pretrained CNN backbone
# ------------------------------------------------------
def example_hybrid_model():
    print("\n--- Example 4: Pretrained CNN backbone + JetBlocks + Decoder ---")
    images = torch.randn(2, 3, 224, 224).to(get_device())

    # HybridJetModel wraps backbone + jetblocks + decoder
    model = HybridJetModel(
        backbone_name="resnet18",
        pretrained_backbone=True,
        jet_embed_dim=512,
        n_heads=8,
        qk_dim=64,
        v_dim=64,
        n_jetblocks=2,
        use_rope=True,
        out_channels=21,
    ).to(get_device())
    preds = model(images).to(get_device())

    print("Input images:", images.shape)
    print("Output from HybridJetModel:", preds.shape)
    print("HybridJetModel params (M):", count_parameters(model))


# ------------------------------------------------------
# Run all examples
# ------------------------------------------------------
if __name__ == "__main__":
    example_jetblock()
    example_patch_embedding()
    example_with_decoder()
    example_hybrid_model()
