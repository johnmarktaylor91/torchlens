"""AdaBins: per-image adaptive depth-bins via a mini-ViT transformer head.

Bhat, Alhashim & Wonka, CVPR 2021.
Paper: https://arxiv.org/abs/2011.14141
Source: https://github.com/shariqfarooq123/AdaBins

Distinctive primitive -- the "AdaptiveBins" mini-ViT block:
  * An encoder-decoder (EfficientNet-B5 encoder + UNet-style decoder in the original;
    here a compact conv encoder-decoder) produces a dense feature map.
  * That feature map is patch-embedded and fed to a small Transformer encoder
    ("mViT", mini Vision Transformer). The FIRST output token regresses a vector of
    N bin-widths, which are normalized (sum to the depth range) -> the depth bin
    boundaries are predicted PER IMAGE (adaptive bins), not fixed.
  * The remaining tokens form 1x1 convolution kernels; convolving them with the decoder
    features and softmaxing gives per-pixel bin-probabilities.
  * Final depth = sum over bins of (bin-center * per-pixel bin-probability) -- a soft,
    linear-combination-over-adaptive-bin-centers depth regression.

The variants (adabins / efficientnet-b5 NYU / efficientnet-b5 KITTI) differ only in the
max depth (10m NYU vs 80m KITTI) and image size; the ARCHITECTURE is identical, so we
build one core and register the variants with different max_val/bins.

Compact random-init reimplementation: small image, shallow conv encoder-decoder, tiny mViT.
Reproduces the adaptive-bins mini-ViT head + soft bin-center depth regression faithfully.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EncoderDecoder(nn.Module):
    """Compact UNet-style encoder-decoder (stand-in for EfficientNet-B5 + UNet decoder)."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.e1 = _ConvBlock(3, base)
        self.e2 = _ConvBlock(base, base * 2)
        self.e3 = _ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.d2 = _ConvBlock(base * 4 + base * 2, base * 2)
        self.d1 = _ConvBlock(base * 2 + base, base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        u2 = F.interpolate(x3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, x2], 1))
        u1 = F.interpolate(d2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, x1], 1))
        return d1  # (B, base, H, W) decoder features


class _MiniViT(nn.Module):
    """Mini Vision Transformer ("mViT") that predicts adaptive bin-widths + bin kernels."""

    def __init__(
        self,
        in_c: int,
        embed: int = 32,
        n_bins: int = 16,
        patch: int = 4,
        depth: int = 2,
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.embed = embed
        self.patch_embed = nn.Conv2d(in_c, embed, patch, stride=patch)
        layer = nn.TransformerEncoderLayer(embed, heads, embed * 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth)
        # regressor head on the first token -> N bin widths
        self.bin_regressor = nn.Sequential(
            nn.Linear(embed, embed), nn.ReLU(True), nn.Linear(embed, n_bins)
        )
        # 1x1 conv producing per-pixel keys to dot against bin kernels
        self.pixel_key = nn.Conv2d(in_c, embed, 1)

    def forward(self, feats: torch.Tensor):
        b = feats.shape[0]
        tokens = self.patch_embed(feats).flatten(2).transpose(1, 2)  # (B, T, E)
        enc = self.transformer(tokens)  # (B, T, E)
        # first token -> adaptive bin widths
        bin_widths = self.bin_regressor(enc[:, 0])  # (B, n_bins)
        bin_widths = F.relu(bin_widths) + 0.1  # positive widths
        bin_widths = bin_widths / bin_widths.sum(1, keepdim=True)  # normalize (per image)
        # remaining tokens -> 1x1 "bin kernels"; dot with per-pixel keys -> bin logits
        kernels = enc[:, 1 : 1 + self.n_bins]  # (B, n_bins, E)
        keys = self.pixel_key(feats)  # (B, E, H, W)
        logits = torch.einsum("bke,behw->bkhw", kernels, keys)  # (B, n_bins, H, W)
        return bin_widths, logits


class _AdaBins(nn.Module):
    """AdaBins depth model: encoder-decoder + mViT adaptive bins + soft bin-center regression."""

    def __init__(
        self, n_bins: int = 16, min_val: float = 0.001, max_val: float = 10.0, base: int = 16
    ) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.backbone = _EncoderDecoder(base)
        self.mvit = _MiniViT(base, n_bins=n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        bin_widths, logits = self.mvit(feats)  # (B, n_bins), (B, n_bins, H, W)
        probs = F.softmax(logits, dim=1)  # per-pixel bin probabilities
        # adaptive bin centers from cumulative normalized widths
        widths = bin_widths * (self.max_val - self.min_val)
        edges = torch.cumsum(widths, dim=1) + self.min_val  # (B, n_bins)
        centers = edges - widths / 2  # bin centers
        # soft depth = sum_bins center * prob
        depth = torch.einsum("bk,bkhw->bhw", centers, probs).unsqueeze(1)
        return depth


def build_adabins() -> nn.Module:
    """AdaBins (NYU, max depth 10m)."""
    return _AdaBins(n_bins=16, max_val=10.0)


def build_adabins_efnet_b5() -> nn.Module:
    """AdaBins EfficientNet-B5 NYU recipe (max depth 10m)."""
    return _AdaBins(n_bins=16, max_val=10.0)


def build_adabins_efnet_b5_kitti() -> nn.Module:
    """AdaBins EfficientNet-B5 KITTI recipe (max depth 80m)."""
    return _AdaBins(n_bins=16, max_val=80.0)


def example_input() -> torch.Tensor:
    """Example RGB image tensor (1, 3, 96, 128)."""
    return torch.randn(1, 3, 96, 128)


MENAGERIE_ENTRIES = [
    ("AdaBins (adaptive depth-bins mini-ViT)", "build_adabins", "example_input", "2021", "DC"),
    (
        "AdaBins-EffNetB5 (NYU adaptive-bins depth)",
        "build_adabins_efnet_b5",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "AdaBins-EffNetB5-KITTI (80m adaptive-bins depth)",
        "build_adabins_efnet_b5_kitti",
        "example_input",
        "2021",
        "DC",
    ),
]
