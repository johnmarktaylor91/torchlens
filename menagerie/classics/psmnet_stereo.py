"""PSMNet: Pyramid Stereo Matching Network.

Chang & Chen, CVPR 2018.
Paper: https://arxiv.org/abs/1803.08669
Source: https://github.com/JiaRenChang/PSMNet

Distinctive primitives:
  - Spatial Pyramid Pooling (SPP) feature extractor with multi-scale pooling
  - Concatenation cost volume built across disparity shifts
  - 3D-convolutional hourglass aggregation (Basic or Stacked)
  - Soft-argmin disparity regression

Two aggregation heads:
  - psmnet_basic:        single 3D-conv basic aggregation
  - psmnet_stackhourglass: stacked 3D-conv hourglasses (the default PSMNet)

Random init; compact config: H=32, W=64, D=12 disparities, C=16 feature channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Feature extractor with SPP (Spatial Pyramid Pooling)
# ──────────────────────────────────────────────────────────────


class _ConvBnRelu(nn.Module):
    def __init__(
        self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, dilation: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _SPPModule(nn.Module):
    """Multi-scale average-pool + 1x1 conv branches, concatenated."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        # 4 branches with pool sizes 64,32,16,8 -> all upsample to feature map size
        self.branches = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool2d(s), nn.Conv2d(in_c, out_c, 1, bias=False))
                for s in (4, 2, 1, 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [
            F.interpolate(b(x), size=x.shape[2:], mode="bilinear", align_corners=False)
            for b in self.branches
        ]
        return torch.cat(parts, dim=1)


class _FeatureExtractor(nn.Module):
    """Shared 2-D CNN feature extractor with SPP module."""

    def __init__(self, C: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            _ConvBnRelu(3, C, 3, 2, 1),
            _ConvBnRelu(C, C, 3, 1, 1),
            _ConvBnRelu(C, C, 3, 1, 1),
        )
        self.layer1 = nn.Sequential(
            _ConvBnRelu(C, C * 2, 3, 2, 1),
            _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
        )
        self.spp = _SPPModule(C * 2, C)  # 4 branches -> 4*C channels
        self.fuse = _ConvBnRelu(C * 2 + 4 * C, C * 4, 3, 1, 1)
        self.out_channels = C * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.stem(x)
        l1 = self.layer1(s)
        spp = self.spp(l1)
        return self.fuse(torch.cat([l1, spp], dim=1))


# ──────────────────────────────────────────────────────────────
# Cost volume: concatenation across D disparity shifts
# ──────────────────────────────────────────────────────────────


def _build_concat_cost_volume(
    left: torch.Tensor, right: torch.Tensor, max_disp: int
) -> torch.Tensor:
    """Build (B, 2C, D, H, W) concatenation cost volume."""
    B, C, H, W = left.shape
    cost = torch.zeros(B, C * 2, max_disp, H, W, device=left.device, dtype=left.dtype)
    for d in range(max_disp):
        if d == 0:
            cost[:, :C, d, :, :] = left
            cost[:, C:, d, :, :] = right
        else:
            cost[:, :C, d, :, d:] = left[:, :, :, d:]
            cost[:, C:, d, :, d:] = right[:, :, :, :-d]
    return cost


# ──────────────────────────────────────────────────────────────
# 3D-conv Hourglass
# ──────────────────────────────────────────────────────────────


class _Conv3dBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _Hourglass3D(nn.Module):
    """Single 3D-conv hourglass for cost volume aggregation."""

    def __init__(self, C: int) -> None:
        super().__init__()
        self.enc1 = _Conv3dBnRelu(C, C)
        self.enc2 = _Conv3dBnRelu(C, C * 2)
        self.bot = _Conv3dBnRelu(C * 2, C * 2)
        self.dec2 = _Conv3dBnRelu(C * 2 + C * 2, C)
        self.dec1 = _Conv3dBnRelu(C + C, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e1p = F.avg_pool3d(e1, 2, stride=2)
        e2 = self.enc2(e1p)
        e2p = F.avg_pool3d(e2, 2, stride=2)
        bot = self.bot(e2p)
        up2 = F.interpolate(bot, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], 1))
        up1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], 1))
        return d1


# ──────────────────────────────────────────────────────────────
# Soft-argmin disparity regression
# ──────────────────────────────────────────────────────────────


def _soft_argmin(cost: torch.Tensor) -> torch.Tensor:
    """Soft-argmin over disparity dimension. cost: (B, D, H, W) -> (B, 1, H, W)."""
    prob = F.softmax(-cost, dim=1)
    d_idx = torch.arange(cost.shape[1], device=cost.device, dtype=cost.dtype)
    d_idx = d_idx.view(1, -1, 1, 1)
    return (prob * d_idx).sum(dim=1, keepdim=True)


# ──────────────────────────────────────────────────────────────
# PSMNet Basic
# ──────────────────────────────────────────────────────────────


class PSMNetBasic(nn.Module):
    """PSMNet with basic 3D-conv aggregation (no stacked hourglass)."""

    def __init__(self, max_disp: int = 12, C: int = 16) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.feat = _FeatureExtractor(C)
        fc = self.feat.out_channels  # 4*C
        # cost volume channels: 2*fc, aggregated down to 1 per disparity
        self.agg = nn.Sequential(
            _Conv3dBnRelu(fc * 2, fc),
            _Conv3dBnRelu(fc, fc),
            nn.Conv3d(fc, 1, 3, 1, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self.feat(left)
        fr = self.feat(right)
        # cost volume at 1/4 resolution
        cost = _build_concat_cost_volume(fl, fr, self.max_disp)
        agg = self.agg(cost).squeeze(1)  # (B, D, H, W)
        disp = _soft_argmin(agg)
        # upsample to original resolution
        disp = F.interpolate(disp, size=left.shape[2:], mode="bilinear", align_corners=False)
        return disp


# ──────────────────────────────────────────────────────────────
# PSMNet Stacked Hourglass
# ──────────────────────────────────────────────────────────────


class PSMNetStackHourglass(nn.Module):
    """PSMNet with stacked 3D-conv hourglasses (2 stacked)."""

    def __init__(self, max_disp: int = 12, C: int = 16, n_stacks: int = 2) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.feat = _FeatureExtractor(C)
        fc = self.feat.out_channels
        self.init_conv = _Conv3dBnRelu(fc * 2, fc)
        self.hourglasses = nn.ModuleList([_Hourglass3D(fc) for _ in range(n_stacks)])
        self.classifiers = nn.ModuleList([nn.Conv3d(fc, 1, 3, 1, 1) for _ in range(n_stacks)])

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self.feat(left)
        fr = self.feat(right)
        cost = _build_concat_cost_volume(fl, fr, self.max_disp)
        x = self.init_conv(cost)
        disp = None
        for hg, cls in zip(self.hourglasses, self.classifiers):
            x = hg(x)
            cost_out = cls(x).squeeze(1)  # (B, D, H, W)
            disp = _soft_argmin(cost_out)
        disp = F.interpolate(disp, size=left.shape[2:], mode="bilinear", align_corners=False)
        return disp


# ──────────────────────────────────────────────────────────────
# Wrappers (single stereo pair input -> single disparity output)
# ──────────────────────────────────────────────────────────────


class _StereoWrapper(nn.Module):
    """Wrap a stereo model so it takes a single 6-channel tensor (left||right)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = x[:, :3]
        right = x[:, 3:]
        return self.model(left, right)


# ──────────────────────────────────────────────────────────────
# Builders + example inputs + entries
# ──────────────────────────────────────────────────────────────


def build_psmnet_basic() -> nn.Module:
    """PSMNet Basic — single 3D-conv aggregation, compact (C=16, D=12)."""
    return _StereoWrapper(PSMNetBasic(max_disp=12, C=16))


def build_psmnet_stackhourglass() -> nn.Module:
    """PSMNet Stacked-Hourglass — 2-stack 3D-conv hourglass, compact (C=16, D=12)."""
    return _StereoWrapper(PSMNetStackHourglass(max_disp=12, C=16, n_stacks=2))


def example_input() -> torch.Tensor:
    """Stereo pair as 6-channel tensor (left||right), (1, 6, 32, 64)."""
    return torch.randn(1, 6, 32, 64)


MENAGERIE_ENTRIES = [
    (
        "PSMNet Basic (3D-conv cost-volume stereo matching)",
        "build_psmnet_basic",
        "example_input",
        "2018",
        "DC",
    ),
    (
        "PSMNet StackedHourglass (stacked 3D-conv hourglass stereo)",
        "build_psmnet_stackhourglass",
        "example_input",
        "2018",
        "DC",
    ),
]
