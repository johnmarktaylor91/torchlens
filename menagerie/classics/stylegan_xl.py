"""StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets.

Sauer et al., SIGGRAPH 2022.  arXiv:2202.00273.
Source: https://github.com/autonomousvision/stylegan-xl

StyleGAN-XL extends StyleGAN3 for large diverse datasets (ImageNet) with:

Distinctive primitives:
  - **Class-conditional generation via class embedding**: a one-hot class label is
    embedded and concatenated to z before the mapping network (similar to BigGAN);
    class embedding also modulates each synthesis layer.
  - **Progressive growing removed**: no staged training; single-stage training at full
    resolution using StyleGAN3 alias-free architecture as backbone.
  - **Projected discriminator** (shared with StyleGAN-T): frozen pretrained backbone
    features (DINOv2/EfficientNet) projected and classified. The discriminator does NOT
    need class conditioning since backbone features implicitly capture class structure.
  - **Mixing regularization** at the synthesis level.
  - Alias-free filtered lrelu inherited from StyleGAN3.

Compact: n_classes=10, z_dim=64, w_dim=64, base_ch=32, output 32x32.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared utilities
# ============================================================


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


class MappingNetworkConditional(nn.Module):
    """Mapping network: (z + class_emb) -> w."""

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        n_classes: int = 10,
        class_emb_dim: int = 32,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.pixel_norm = PixelNorm()
        self.class_emb = nn.Embedding(n_classes, class_emb_dim)
        in_dim = z_dim + class_emb_dim
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, w_dim), nn.LeakyReLU(0.2)]
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        z = self.pixel_norm(z)
        c = self.class_emb(class_idx)
        return self.net(torch.cat([z, c], dim=1))


# ============================================================
# Alias-free filtered activation (StyleGAN3 style)
# ============================================================


class FilteredLReLU(nn.Module):
    def __init__(self, channels: int, upsample: bool = True) -> None:
        super().__init__()
        self.do_upsample = upsample
        k = 5
        sigma = 1.5
        g = torch.tensor(
            [math.exp(-((i - k // 2) ** 2) / (2 * sigma**2)) for i in range(k)],
            dtype=torch.float32,
        )
        kernel = g[:, None] * g[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer("lpf", kernel.view(1, 1, k, k).expand(channels, 1, k, k).clone())
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.leaky_relu(x, 0.2)
        x = F.conv2d(x, self.lpf, padding=2, groups=self.channels)
        if self.do_upsample:
            x = F.avg_pool2d(x, 2)
        return x


# ============================================================
# StyleGAN-XL synthesis layer (modulated conv with class-conditional style)
# ============================================================


class StyleGANXLSynthesisLayer(nn.Module):
    """StyleGAN-XL synthesis layer: modulated conv + filtered lrelu.

    Class conditioning is already baked into w via the mapping network.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        w_dim: int,
        upsample: bool = True,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.affine = nn.Linear(w_dim, in_ch)
        nn.init.ones_(self.affine.bias)
        k = 3
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k) / math.sqrt(in_ch))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.filtered_act = FilteredLReLU(out_ch, upsample=upsample)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        s = self.affine(w)
        k = self.weight.size(-1)
        weight = self.weight.unsqueeze(0) * s.view(B, 1, self.in_ch, 1, 1)
        d = weight.pow(2).sum(dim=[2, 3, 4], keepdim=True).add(1e-8).sqrt()
        weight = weight / d
        x_in = x.reshape(1, B * self.in_ch, x.size(2), x.size(3))
        w_flat = weight.view(B * self.out_ch, self.in_ch, k, k)
        out = F.conv2d(x_in, w_flat, padding=k // 2, groups=B)
        out = out.view(B, self.out_ch, out.size(2), out.size(3))
        out = out + self.bias.view(1, -1, 1, 1)
        return self.filtered_act(out)


# ============================================================
# StyleGAN-XL Generator
# ============================================================


class StyleGANXLGenerator(nn.Module):
    """StyleGAN-XL class-conditional generator (compact).

    z + class_idx -> mapping -> w -> constant 4x4 input ->
    3 alias-free synthesis layers -> to-RGB.
    """

    def __init__(
        self,
        z_dim: int = 64,
        w_dim: int = 64,
        n_classes: int = 10,
        class_emb_dim: int = 32,
        base_ch: int = 32,
    ) -> None:
        super().__init__()
        self.mapping = MappingNetworkConditional(z_dim, w_dim, n_classes, class_emb_dim)
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 4, 4))
        ch = [base_ch * 4, base_ch * 2, base_ch, base_ch]
        self.layers = nn.ModuleList(
            [
                StyleGANXLSynthesisLayer(ch[0], ch[1], w_dim, upsample=True),
                StyleGANXLSynthesisLayer(ch[1], ch[2], w_dim, upsample=True),
                StyleGANXLSynthesisLayer(ch[2], ch[3], w_dim, upsample=True),
            ]
        )
        self.to_rgb = nn.Conv2d(ch[-1], 3, 1)

    def forward(self, z: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        w = self.mapping(z, class_idx)
        x = self.const.expand(B, -1, -1, -1)
        for layer in self.layers:
            x = layer(x, w)
        return torch.tanh(self.to_rgb(x))


def build_stylegan_xl_generator() -> nn.Module:
    return StyleGANXLGenerator()


def example_input_stylegan_xl_generator() -> tuple:
    z = torch.randn(1, 64)
    class_idx = torch.randint(0, 10, (1,))
    return (z, class_idx)


# ============================================================
# StyleGAN-XL Discriminator (projected discriminator)
# ============================================================


class StyleGANXLDiscriminator(nn.Module):
    """StyleGAN-XL projected discriminator.

    Uses a lightweight conv backbone (approximating frozen DINO/EfficientNet backbone
    in the paper). Features projected to a compact space and classified.
    No class conditioning in the discriminator (backbone implicitly handles it).
    """

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.proj = nn.Conv2d(base_ch * 4, base_ch, 1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch, 1),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(img)
        return self.head(self.proj(feats))


def build_stylegan_xl_discriminator() -> nn.Module:
    return StyleGANXLDiscriminator()


def example_input_stylegan_xl_discriminator() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


# ============================================================
# MENAGERIE_ENTRIES
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "stylegan_xl_generator",
        "build_stylegan_xl_generator",
        "example_input_stylegan_xl_generator",
        "2022",
        "DC",
    ),
    (
        "stylegan_xl_discriminator",
        "build_stylegan_xl_discriminator",
        "example_input_stylegan_xl_discriminator",
        "2022",
        "DC",
    ),
]
