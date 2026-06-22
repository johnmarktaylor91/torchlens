"""GwcNet: Group-wise Correlation Stereo Network.

Guo et al., CVPR 2019.
Paper: https://arxiv.org/abs/1903.04025
Source: https://github.com/xy-guo/GwcNet

Distinctive primitive:
  - Group-Wise Correlation cost volume: split feature channels into G groups,
    compute inner-product (correlation) per group across disparity shifts,
    producing a (B, G, D, H, W) geometry volume.
  - _gc variant additionally concatenates raw left/right features alongside.
  - 3D-conv stacked hourglass aggregation (shared with PSMNet family).
  - Soft-argmin disparity regression.

Three model names:
  - gwcnet   : groupwise correlation only (alias for _g variant)
  - gwcnet_g : groupwise correlation only
  - gwcnet_gc: groupwise + concatenation (richer cost)

Random init; compact config: H=32, W=64, D=12, C=16 feat channels, G=4 groups.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Shared building blocks
# ──────────────────────────────────────────────────────────────


class _ConvBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _FeatureNet(nn.Module):
    """Simple 2D CNN feature extractor."""

    def __init__(self, C: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnRelu(3, C, 3, 2, 1),
            _ConvBnRelu(C, C, 3, 1, 1),
            _ConvBnRelu(C, C * 2, 3, 2, 1),
            _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
        )
        self.out_channels = C * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# Group-wise Correlation cost volume
# ──────────────────────────────────────────────────────────────


def _gwc_volume(
    left: torch.Tensor, right: torch.Tensor, max_disp: int, num_groups: int
) -> torch.Tensor:
    """Build group-wise correlation volume (B, G, D, H, W)."""
    B, C, H, W = left.shape
    assert C % num_groups == 0, "C must be divisible by num_groups"
    channels_per_group = C // num_groups
    cost = torch.zeros(B, num_groups, max_disp, H, W, device=left.device, dtype=left.dtype)
    for d in range(max_disp):
        if d == 0:
            r_shifted = right
        else:
            r_shifted = torch.zeros_like(right)
            r_shifted[:, :, :, d:] = right[:, :, :, :-d]
        # reshape to groups: (B, G, C//G, H, W)
        l_g = left.view(B, num_groups, channels_per_group, H, W)
        r_g = r_shifted.view(B, num_groups, channels_per_group, H, W)
        cost[:, :, d] = (l_g * r_g).mean(dim=2)  # (B, G, H, W)
    return cost


# ──────────────────────────────────────────────────────────────
# Concat cost volume (as in PSMNet) for the _gc variant
# ──────────────────────────────────────────────────────────────


def _concat_volume(left: torch.Tensor, right: torch.Tensor, max_disp: int) -> torch.Tensor:
    """Build (B, 2C, D, H, W) concatenation cost volume."""
    B, C, H, W = left.shape
    cost = torch.zeros(B, C * 2, max_disp, H, W, device=left.device, dtype=left.dtype)
    for d in range(max_disp):
        if d == 0:
            cost[:, :C, d] = left
            cost[:, C:, d] = right
        else:
            cost[:, :C, d, :, d:] = left[:, :, :, d:]
            cost[:, C:, d, :, d:] = right[:, :, :, :-d]
    return cost


# ──────────────────────────────────────────────────────────────
# 3D Hourglass (shared)
# ──────────────────────────────────────────────────────────────


class _Conv3dBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class _Hourglass3D(nn.Module):
    def __init__(self, C: int) -> None:
        super().__init__()
        self.e1 = _Conv3dBnRelu(C, C)
        self.e2 = _Conv3dBnRelu(C, C * 2)
        self.bot = _Conv3dBnRelu(C * 2, C * 2)
        self.d2 = _Conv3dBnRelu(C * 2 * 2, C)
        self.d1 = _Conv3dBnRelu(C * 2, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool3d(e1, 2, stride=2))
        bot = self.bot(F.avg_pool3d(e2, 2, stride=2))
        u2 = F.interpolate(bot, e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], 1))
        u1 = F.interpolate(d2, e1.shape[2:], mode="trilinear", align_corners=False)
        return self.d1(torch.cat([u1, e1], 1))


def _soft_argmin(cost: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(-cost, dim=1)
    idx = torch.arange(cost.shape[1], device=cost.device, dtype=cost.dtype).view(1, -1, 1, 1)
    return (prob * idx).sum(1, keepdim=True)


# ──────────────────────────────────────────────────────────────
# GwcNet_G: groupwise correlation only
# ──────────────────────────────────────────────────────────────


class GwcNetG(nn.Module):
    """GwcNet with group-wise correlation cost volume only."""

    def __init__(
        self, max_disp: int = 12, C: int = 16, num_groups: int = 4, n_stacks: int = 2
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.num_groups = num_groups
        self.feat = _FeatureNet(C)
        # cost volume is (B, G, D, H, W) -> flatten G into channel dim
        self.init_conv = _Conv3dBnRelu(num_groups, C)
        self.hourglasses = nn.ModuleList([_Hourglass3D(C) for _ in range(n_stacks)])
        self.classifiers = nn.ModuleList([nn.Conv3d(C, 1, 3, 1, 1) for _ in range(n_stacks)])

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self.feat(left)
        fr = self.feat(right)
        cost = _gwc_volume(fl, fr, self.max_disp, self.num_groups)
        x = self.init_conv(cost)
        disp = None
        for hg, cls in zip(self.hourglasses, self.classifiers):
            x = hg(x)
            disp = _soft_argmin(cls(x).squeeze(1))
        disp = F.interpolate(disp, left.shape[2:], mode="bilinear", align_corners=False)
        return disp


# ──────────────────────────────────────────────────────────────
# GwcNet_GC: groupwise + concatenation
# ──────────────────────────────────────────────────────────────


class GwcNetGC(nn.Module):
    """GwcNet with group-wise correlation + concatenation cost volume."""

    def __init__(
        self, max_disp: int = 12, C: int = 16, num_groups: int = 4, n_stacks: int = 2
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        self.num_groups = num_groups
        self.feat = _FeatureNet(C)
        fc = self.feat.out_channels
        # fuse groupwise + concat volumes
        fuse_in = num_groups + fc * 2
        self.init_conv = _Conv3dBnRelu(fuse_in, C)
        self.hourglasses = nn.ModuleList([_Hourglass3D(C) for _ in range(n_stacks)])
        self.classifiers = nn.ModuleList([nn.Conv3d(C, 1, 3, 1, 1) for _ in range(n_stacks)])

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self.feat(left)
        fr = self.feat(right)
        gwc = _gwc_volume(fl, fr, self.max_disp, self.num_groups)
        cat = _concat_volume(fl, fr, self.max_disp)
        cost = torch.cat([gwc, cat], dim=1)
        x = self.init_conv(cost)
        disp = None
        for hg, cls in zip(self.hourglasses, self.classifiers):
            x = hg(x)
            disp = _soft_argmin(cls(x).squeeze(1))
        disp = F.interpolate(disp, left.shape[2:], mode="bilinear", align_corners=False)
        return disp


# ──────────────────────────────────────────────────────────────
# Single-tensor wrapper
# ──────────────────────────────────────────────────────────────


class _StereoWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x[:, :3], x[:, 3:])


# ──────────────────────────────────────────────────────────────
# Builders
# ──────────────────────────────────────────────────────────────


def build_gwcnet() -> nn.Module:
    """GwcNet (groupwise correlation, alias for gwcnet_g), compact."""
    return _StereoWrapper(GwcNetG(max_disp=12, C=16, num_groups=4))


def build_gwcnet_g() -> nn.Module:
    """GwcNet-G (groupwise correlation only), compact."""
    return _StereoWrapper(GwcNetG(max_disp=12, C=16, num_groups=4))


def build_gwcnet_gc() -> nn.Module:
    """GwcNet-GC (groupwise + concat cost volume), compact."""
    return _StereoWrapper(GwcNetGC(max_disp=12, C=16, num_groups=4))


def example_input() -> torch.Tensor:
    """Stereo pair as 6-channel tensor (left||right), (1, 6, 32, 64)."""
    return torch.randn(1, 6, 32, 64)


MENAGERIE_ENTRIES = [
    (
        "GwcNet (group-wise correlation stereo matching)",
        "build_gwcnet",
        "example_input",
        "2019",
        "DC",
    ),
    (
        "GwcNet-G (groupwise correlation cost volume stereo)",
        "build_gwcnet_g",
        "example_input",
        "2019",
        "DC",
    ),
    (
        "GwcNet-GC (groupwise + concat cost volume stereo)",
        "build_gwcnet_gc",
        "example_input",
        "2019",
        "DC",
    ),
]
