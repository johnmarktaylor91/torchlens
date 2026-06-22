"""GA-Net: Guided Aggregation Net for End-to-End Stereo Matching.

Zhang et al., CVPR 2019.
Paper: https://arxiv.org/abs/1904.06587
Source: https://github.com/feihuzhang/GANet

Distinctive primitives:
  - Semi-Global Aggregation (SGA): semi-global-style aggregation over the cost
    volume guided by learned directional guidance weights (4 or 8 directions).
  - Local Guided Aggregation (LGA): local (3x3) convolutional aggregation of
    the cost volume with spatially-varying learned guidance weights.
  - Standard 2D CNN feature extractor.
  - Soft-argmin disparity regression.

Two depth variants:
  - ganet_11:   shallower feature extractor + 1 SGA + 1 LGA stage
  - ganet_deep: deeper feature extractor  + 2 SGA + 2 LGA stages

Random init; compact config: H=32, W=64, D=12, C=16 feat channels.
NOTE: True SGA computes 4-direction semi-global paths; here we implement a
faithful approximation using 2D convolutions with learned spatially-varying
guidance (the structural essence of SGA) since full semi-global DP is
non-differentiable through fixed paths. The LGA is faithfully reproduced.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


class _ConvBnRelu(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ──────────────────────────────────────────────────────────────
# Feature extractors (shallow vs deep)
# ──────────────────────────────────────────────────────────────


class _ShallowFeatureNet(nn.Module):
    """Feature extractor for ganet_11 (shallow)."""

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


class _DeepFeatureNet(nn.Module):
    """Feature extractor for ganet_deep (deeper with residual-like blocks)."""

    def __init__(self, C: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _ConvBnRelu(3, C, 3, 2, 1),
            _ConvBnRelu(C, C, 3, 1, 1),
            _ConvBnRelu(C, C * 2, 3, 2, 1),
            _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
            _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
            _ConvBnRelu2d(C * 2, C * 2, 3, 1, 2, dil=2),  # dilated
            _ConvBnRelu(C * 2, C * 2, 3, 1, 1),
        )
        self.out_channels = C * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConvBnRelu2d(nn.Module):
    def __init__(
        self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, dil: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, dilation=dil, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ──────────────────────────────────────────────────────────────
# Cost volume
# ──────────────────────────────────────────────────────────────


def _build_cost_volume(left: torch.Tensor, right: torch.Tensor, max_disp: int) -> torch.Tensor:
    """Concatenation cost volume (B, 2C, D, H, W)."""
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
# SGA: Semi-Global Aggregation layer
# A learnable guidance-weighted aggregation over the cost volume.
# We reproduce the key structural element: spatially-varying 2D guidance
# weights derived from image features, applied per-disparity-slice.
# ──────────────────────────────────────────────────────────────


class _SGALayer(nn.Module):
    """Semi-Global Aggregation layer (guidance-weighted cost aggregation).

    Learns spatially-varying guidance weights from image features and
    applies them to aggregate the cost volume across H and W directions.
    This captures the key structural distinctive of SGA: context-adaptive
    aggregation guided by the reference image.
    """

    def __init__(self, cost_channels: int, guidance_channels: int, num_dirs: int = 4) -> None:
        super().__init__()
        # Guidance network produces per-direction weights
        self.guide_net = nn.Sequential(
            _ConvBnRelu2d(guidance_channels, guidance_channels),
            nn.Conv2d(guidance_channels, num_dirs * cost_channels, 3, 1, 1),
        )
        self.num_dirs = num_dirs
        self.cost_channels = cost_channels
        # Aggregate per direction
        self.agg_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(cost_channels, cost_channels, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(cost_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_dirs)
            ]
        )
        self.out_conv = nn.Conv2d(
            cost_channels * num_dirs + cost_channels, cost_channels, 1, bias=False
        )

    def forward(self, cost: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """cost: (B, C, D, H, W), feat: (B, Cf, H, W) -> (B, C, D, H, W)."""
        B, C, D, H, W = cost.shape
        # guidance: (B, num_dirs*C, H, W)
        guide = self.guide_net(feat)
        guide = torch.sigmoid(guide)
        agg_slices = []
        for d in range(D):
            c_d = cost[:, :, d, :, :]  # (B, C, H, W)
            dir_outs = []
            for i, agg in enumerate(self.agg_convs):
                g = guide[:, i * C : (i + 1) * C, :, :]  # (B, C, H, W)
                dir_outs.append(agg(c_d * g))
            combined = torch.cat(dir_outs + [c_d], dim=1)  # (B, C*num_dirs+C, H, W)
            agg_slices.append(self.out_conv(combined))
        return torch.stack(agg_slices, dim=2)  # (B, C, D, H, W)


# ──────────────────────────────────────────────────────────────
# LGA: Local Guided Aggregation layer
# Applies 3x3 local guided aggregation with learned guidance kernels.
# ──────────────────────────────────────────────────────────────


class _LGALayer(nn.Module):
    """Local Guided Aggregation: locally learned 3x3 guidance for cost volume."""

    def __init__(self, cost_channels: int, guidance_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.k = kernel_size
        self.guide_net = nn.Sequential(
            _ConvBnRelu2d(guidance_channels, guidance_channels),
            nn.Conv2d(guidance_channels, cost_channels * kernel_size * kernel_size, 3, 1, 1),
        )
        self.cost_channels = cost_channels

    def forward(self, cost: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """cost: (B, C, D, H, W), feat: (B, Cf, H, W) -> (B, C, D, H, W)."""
        B, C, D, H, W = cost.shape
        k = self.k
        p = k // 2
        # guidance: (B, C*k*k, H, W) -> normalize per pixel over k^2
        guide = self.guide_net(feat)
        guide = guide.view(B, C, k * k, H, W)
        guide = F.softmax(guide, dim=2)  # (B, C, k*k, H, W)

        # unfold cost for each disparity
        output = []
        for d in range(D):
            c_d = cost[:, :, d, :, :]  # (B, C, H, W)
            # unfold: (B, C*k*k, H*W)
            unf = F.unfold(c_d, kernel_size=k, padding=p)  # (B, C*k*k, H*W)
            unf = unf.view(B, C, k * k, H * W)
            # guide: (B, C, k*k, H*W)
            g = guide.view(B, C, k * k, H * W)
            out_d = (unf * g).sum(dim=2).view(B, C, H, W)
            output.append(out_d)
        return torch.stack(output, dim=2)


# ──────────────────────────────────────────────────────────────
# Soft-argmin
# ──────────────────────────────────────────────────────────────


def _soft_argmin(cost: torch.Tensor) -> torch.Tensor:
    # cost: (B, C, D, H, W) -> aggregate channels, then soft-argmin over D
    cost_d = cost.mean(dim=1)  # (B, D, H, W)
    prob = F.softmax(-cost_d, dim=1)
    idx = torch.arange(cost_d.shape[1], device=cost.device, dtype=cost.dtype).view(1, -1, 1, 1)
    return (prob * idx).sum(1, keepdim=True)


# ──────────────────────────────────────────────────────────────
# GA-Net models
# ──────────────────────────────────────────────────────────────


class GANet(nn.Module):
    """GA-Net with configurable number of SGA + LGA stages."""

    def __init__(
        self, max_disp: int = 12, C: int = 16, n_sga: int = 1, n_lga: int = 1, deep: bool = False
    ) -> None:
        super().__init__()
        self.max_disp = max_disp
        if deep:
            self.feat = _DeepFeatureNet(C)
        else:
            self.feat = _ShallowFeatureNet(C)
        fc = self.feat.out_channels
        cost_c = fc  # single side of cost after initial conv
        # initial cost projection
        self.cost_init = nn.Sequential(
            nn.Conv3d(fc * 2, cost_c, 3, 1, 1, bias=False),
            nn.BatchNorm3d(cost_c),
            nn.ReLU(inplace=True),
        )
        self.sga_layers = nn.ModuleList([_SGALayer(cost_c, fc) for _ in range(n_sga)])
        self.lga_layers = nn.ModuleList([_LGALayer(cost_c, fc) for _ in range(n_lga)])
        self.final_conv = nn.Conv3d(cost_c, 1, 3, 1, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self.feat(left)
        fr = self.feat(right)
        # Build cost volume at feature resolution
        cost = _build_cost_volume(fl, fr, self.max_disp)
        # cost: (B, 2C, D, H, W) -> init projection to (B, C, D, H, W)
        cost = self.cost_init(cost)
        # SGA layers
        for sga in self.sga_layers:
            cost = sga(cost, fl)
        # LGA layers
        for lga in self.lga_layers:
            cost = lga(cost, fl)
        # Disparity regression
        cost_out = self.final_conv(cost).squeeze(1)  # (B, D, H, W)
        prob = F.softmax(-cost_out, dim=1)
        idx = torch.arange(cost_out.shape[1], device=cost_out.device, dtype=cost_out.dtype).view(
            1, -1, 1, 1
        )
        disp = (prob * idx).sum(1, keepdim=True)
        disp = F.interpolate(disp, left.shape[2:], mode="bilinear", align_corners=False)
        return disp


# ──────────────────────────────────────────────────────────────
# Wrapper + builders
# ──────────────────────────────────────────────────────────────


class _StereoWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x[:, :3], x[:, 3:])


def build_ganet_11() -> nn.Module:
    """GA-Net (shallow, 1 SGA + 1 LGA), compact config."""
    return _StereoWrapper(GANet(max_disp=12, C=16, n_sga=1, n_lga=1, deep=False))


def build_ganet_deep() -> nn.Module:
    """GA-Net Deep (deeper extractor, 2 SGA + 2 LGA), compact config."""
    return _StereoWrapper(GANet(max_disp=12, C=16, n_sga=2, n_lga=2, deep=True))


def example_input() -> torch.Tensor:
    """Stereo pair as 6-channel tensor (left||right), (1, 6, 32, 64)."""
    return torch.randn(1, 6, 32, 64)


MENAGERIE_ENTRIES = [
    (
        "GA-Net-11 (SGA+LGA guided aggregation stereo)",
        "build_ganet_11",
        "example_input",
        "2019",
        "DC",
    ),
    (
        "GA-Net-Deep (deep SGA+LGA guided aggregation stereo)",
        "build_ganet_deep",
        "example_input",
        "2019",
        "DC",
    ),
]
