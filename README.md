# JetNemotron-Extension

JetNemotron-Extension is a collection of PyTorch-based extensions of the JetBlock — an attention-like module designed for efficient sequence modeling from NVIDIA Research.
The library makes it easy to plug JetBlocks into existing **vision** pipelines and add task-specific decoders for semantic segmentation, depth estimation, and other dense prediction tasks.

I have attempted to make the following happen:

- Modular backbone + JetBlock + decoder design
- Clean, extensible PyTorch codebase
- Example script [examples.py](examples.py) that runs the different components with random images
- Ready for research prototyping and extension to real datasets

## 🌟 Inspiration

This work is inspired from [cityzen95/JetBlock-Attention](https://github.com/cityzen95/JetBlock-Attention/) and of course the original authors from NVIDIA [NVlabs/Jet-Nemotron](https://github.com/NVlabs/Jet-Nemotron)

## ⚙️ Installation

Tested on **Ubuntu 24.04** with Python 3.11 and PyTorch ≥ 2.0.

```bash
conda env create -f environment.yml
conda activate jet-nemotron
pip install -e .
```
And then simply run:
```python
python3 examples.py
```
## 💻 Explanation of Examples
Inside this package I have created the following implementations (all these examples can already be found inside the [examples.py](examples.py)):

1. First is of course the JetBlock-Attention which is simply a copy-paste from the original author [cityzen95/JetBlock-Attention](https://github.com/cityzen95/JetBlock-Attention/)
```
from jet_nemotron_nvidia.jetblock import JetBlockAttention
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
x = torch.randn(2, 16, 1536)  # [batch, seq_len, hidden_size]
block = JetBlockAttention(
  hidden_size=1536, n_heads=12, qk_dim=96, v_dim=256, kernel_size=4, use_rope=True
)
out = block(x)
print("Input:", x.shape, "Output:", out.shape)
print("JetBlock params (M):", count_parameters(block))
```
2. Second, is the patch embedding design, which basically takes an input image, breaks it down into smaller patches (to kind of simulate 'sequential information' for the JetBlock module). These image patches can then be used with the JetBlock Attention block.

```
from jet_nemotron_nvidia.jet_image_embedding import JetImageEmbed
from jet_nemotron_nvidia.jetblock import JetBlockAttention
images = torch.randn(2, 3, 224, 224)
embed = JetImageEmbed(img_size=224, patch_size=16, embed_dim=1536)
jetblock = JetBlockAttention(
  hidden_size=1536, n_heads=12, qk_dim=96, v_dim=256, kernel_size=4, use_rope=True
)
patches = embed(images)
out = jetblock(patches)
print("Input images:", images.shape)
print("Patch embeddings:", patches.shape)
print("Output after JetBlock:", out.shape)
print("PatchEmbed params (M):", count_parameters(embed))
print("JetBlock params (M):", count_parameters(jetblock))
```

3. Third is an extention to the second, where I simply add a decoder head to the jetblock attention block to create something like Patch embeddings -> JetBlocks -> Decoder pipeline. The patch decoder block is simply an implementation that converts the 'patched', sequential, attention-rich information from the JetBlock module to recognisable image data.

```
from jet_nemotron_nvidia.jetblock import JetBlockAttention
from jet_nemotron_nvidia.jet_image_embedding import JetImageEmbed, PatchDecoder

images = torch.randn(2, 3, 224, 224)

# Patch encoder
embed = JetImageEmbed(img_size=224, patch_size=16, embed_dim=1536)
jetblock = JetBlockAttention(
  hidden_size=1536, n_heads=12, qk_dim=96, v_dim=256, kernel_size=4, use_rope=True
)
decoder = PatchDecoder(embed_dim=1536, out_channels=19, patch_size=16).to(
  get_device()
)  # 19 class semantic segmentation for example, CityScapes

patches = embed(images)
features = jetblock(patches)
preds = decoder(features, img_size=(224, 224))

print("Input images:", images.shape)
print("Dense predictions:", preds.shape)  # [B, out_channels, H, W]
print("PatchEmbed params (M):", count_parameters(embed))
print("JetBlock params (M):", count_parameters(jetblock))
print("Decoder params (M):", count_parameters(decoder))

```

4. And lastly, I have created a full PyTorch module to mimic deep learning models for vision tasks which basically do Pretrained CNN backbone + Attention + Decoder, where the pre-trained backbone can be ResNet18 or MobileNetV3 etc.

```
from jet_nemotron_nvidia.hybrid_model import HybridJetModel

images = torch.randn(2, 3, 224, 224)

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
)
preds = model(images)

print("Input images:", images.shape)
print("Output from HybridJetModel:", preds.shape)
print("HybridJetModel params (M):", count_parameters(model))

```

## 🧩 Extending the Model

- Add more JetBlocks for deeper attention modeling.
- Replace the decoder with a UNet-style multi-scale decoder for segmentation.
- Experiment with different backbones (ResNet, MobileNet, EfficientNet, etc.).
- Swap JetBlockAttention with other attention blocks if desired.


## 🤝 Contribution

Contributions, bug reports, and feature requests are welcome!
Feel free to open an issue or submit a pull request.
As always, **Hare Krishna** and happy coding!

## 📚 References
- [cityzen95/JetBlock-Attention](https://github.com/cityzen95/JetBlock-Attention/)
- [NVlabs/Jet-Nemotron](https://github.com/NVlabs/Jet-Nemotron)