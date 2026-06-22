"""TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation.

Chen et al., 2021.
Paper: https://arxiv.org/abs/2102.04306
Source: https://github.com/Beckschen/TransUNet

Distinctive architecture (the HYBRID encoder is the point):
  - A CNN (ResNet-50 style) encoder downsamples the image and produces
    multi-scale skip features (1/2, 1/4, 1/8) AND a 1/16 feature map.
  - The 1/16 feature map is patch-embedded (1x1 grid patches) into a token
    sequence and fed through a stack of ViT transformer-encoder blocks -- the
    transformer BOTTLENECK that gives the global receptive field.
  - The reshaped bottleneck tokens are decoded by a CASCADED UPSAMPLER
    (DecoderCup): a chain of conv+upsample blocks that fuse the CNN skip
    connections (U-Net style) back to full resolution, ending in a
    segmentation head.

This module provides two registered variants:
  - "R50-ViT-B_16": hybrid ResNet-CNN + ViT bottleneck + cascaded decoder
    (the flagship TransUNet).
  - "ViT-B_16": pure-ViT encoder (no ResNet stem; patchify the raw image)
    + cascaded decoder -- the non-hybrid ablation.

Faithful compact random-init reimplementation: hidden sizes / layer counts /
image size are shrunk so the unrolled trace draws quickly, but the hybrid
CNN->ViT->cascaded-upsampler structure and the skip fusion are reproduced.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# CNN (ResNet-style) encoder producing multi-scale skips + a 1/16 feature map.
# ----------------------------------------------------------------------------
class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (stride != 1 or in_ch != out_ch)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.down is None else self.down(x)
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + idt)


class ResNetEncoder(nn.Module):
    """ResNet-style hybrid stem: returns 1/16 feature + 3 skip features."""

    def __init__(
        self, in_ch: int = 3, widths: tuple[int, int, int, int] = (32, 64, 128, 256)
    ) -> None:
        super().__init__()
        w0, w1, w2, w3 = widths
        # stem: /2
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, w0, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(w0),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _ConvBlock(w0, w1, stride=2)  # /4
        self.layer2 = _ConvBlock(w1, w2, stride=2)  # /8
        self.layer3 = _ConvBlock(w2, w3, stride=2)  # /16

    def forward(self, x: torch.Tensor):
        s0 = self.stem(x)  # /2
        s1 = self.layer1(s0)  # /4
        s2 = self.layer2(s1)  # /8
        feat = self.layer3(s2)  # /16
        # skips ordered high->low res used by the decoder cup
        return feat, [s2, s1, s0]


# ----------------------------------------------------------------------------
# ViT transformer bottleneck.
# ----------------------------------------------------------------------------
class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBottleneck(nn.Module):
    """Patch-embed a feature map -> transformer-encoder stack -> reshape back."""

    def __init__(
        self, in_ch: int, dim: int = 192, depth: int = 4, heads: int = 4, n_tokens: int = 196
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.blocks = nn.ModuleList([_TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        t = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, HW, dim)
        t = t + self.pos_embed[:, : t.shape[1]]
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t.transpose(1, 2).reshape(b, self.dim, h, w)


# ----------------------------------------------------------------------------
# Cascaded upsampling decoder (DecoderCup) with U-Net skip fusion.
# ----------------------------------------------------------------------------
class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class DecoderCup(nn.Module):
    def __init__(
        self, dim: int, skip_chs: list[int], dec_chs: tuple[int, ...] = (128, 64, 32, 16)
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(dim, dec_chs[0], 3, padding=1)
        in_chs = [dec_chs[0]] + list(dec_chs[:-1])
        # first three blocks fuse skips; last block upsamples with no skip
        skips = skip_chs + [0]
        self.blocks = nn.ModuleList(
            [_DecoderBlock(in_chs[i], skips[i], dec_chs[i]) for i in range(len(dec_chs))]
        )

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = self.head(x)
        for i, blk in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = blk(x, skip)
        return x


# ----------------------------------------------------------------------------
# Full TransUNet (hybrid) and pure-ViT variant.
# ----------------------------------------------------------------------------
class TransUNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        num_classes: int = 9,
        dim: int = 192,
        depth: int = 4,
        heads: int = 4,
        img_size: int = 64,
        hybrid: bool = True,
    ) -> None:
        super().__init__()
        self.hybrid = hybrid
        if hybrid:
            widths = (32, 64, 128, 256)
            self.cnn = ResNetEncoder(in_ch, widths)
            bottleneck_in = widths[3]
            skip_chs = [widths[2], widths[1], widths[0]]  # /8, /4, /2
            grid = img_size // 16
        else:
            self.patchify = nn.Conv2d(in_ch, 256, kernel_size=16, stride=16)
            bottleneck_in = 256
            skip_chs = [0, 0, 0]
            grid = img_size // 16
        n_tokens = grid * grid
        self.vit = ViTBottleneck(
            bottleneck_in, dim=dim, depth=depth, heads=heads, n_tokens=n_tokens
        )
        self.decoder = DecoderCup(dim, skip_chs)
        self.seg_head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hybrid:
            feat, skips = self.cnn(x)
        else:
            feat = self.patchify(x)
            skips = [None, None, None]
        feat = self.vit(feat)
        dec = self.decoder(feat, skips)
        return self.seg_head(dec)


def build_transunet_r50() -> nn.Module:
    """Build the hybrid R50-ViT-B_16 TransUNet (CNN encoder + ViT bottleneck + cascaded decoder)."""
    return TransUNet(in_ch=3, num_classes=9, dim=192, depth=4, heads=4, img_size=64, hybrid=True)


def build_transunet_vit() -> nn.Module:
    """Build the pure-ViT-B_16 TransUNet variant (no ResNet stem, raw-image patchify)."""
    return TransUNet(in_ch=3, num_classes=9, dim=192, depth=4, heads=4, img_size=64, hybrid=False)


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 64, 64)`` for TransUNet."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "TransUNet R50-ViT-B_16 (hybrid CNN encoder + ViT bottleneck + cascaded decoder)",
        "build_transunet_r50",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "TransUNet ViT-B_16 (pure-ViT encoder + cascaded decoder)",
        "build_transunet_vit",
        "example_input",
        "2021",
        "DC",
    ),
]
