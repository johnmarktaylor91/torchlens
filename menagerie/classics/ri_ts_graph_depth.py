"""Monocular depth, surface-normal, and stereo architectures (distinctive heads).

DiverseDepth / AdelaiDepth (RelDepthModel, ResNeXt-101 backbone)
  Yin et al., 2020.  Paper: https://arxiv.org/abs/2002.00569
  Source: https://github.com/aim-uofa/AdelaiDepth
  Distinctive primitive: a ResNeXt encoder (grouped/cardinality bottleneck) feeding a
  multi-scale feature-fusion decoder that predicts RELATIVE (scale-invariant) depth;
  trained with a scale-shift-invariant + pairwise-ranking loss so the SHAPE of depth is
  recovered without metric scale.

DORN (Deep Ordinal Regression Network, ResNet-101 backbone)
  Fu et al., CVPR 2018.  Paper: https://arxiv.org/abs/1806.02446
  Source: https://github.com/hufu6371/DORN
  Distinctive primitive: instead of regressing continuous depth, DORN DISCRETIZES depth
  into K ordinal bins (spacing-increasing discretization) and the head predicts, per
  pixel, 2K logits forming K independent ordinal classifiers; the predicted depth is the
  count of "active" ordinal levels.  Ordinal-regression head over a dilated ASPP-style
  scene-understanding module.

DLNR (Decoupled LSTM + Normalization Refinement stereo)
  Zhao et al., CVPR 2023.  Paper: https://arxiv.org/abs/2303.06615
  Source: https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network
  Distinctive primitive: a RAFT-Stereo style recurrent disparity refiner whose update
  operator is a DECOUPLED LSTM (separate hidden state for high/low-frequency content),
  iteratively updating disparity from a correlation volume, with a normalization
  refinement step to preserve high-frequency detail.

DSINE (surface-normal estimation, v02)
  Bae & Davison, CVPR 2024.  Paper: https://arxiv.org/abs/2403.00712
  Source: https://github.com/baegwangbin/DSINE
  Distinctive primitive: per-pixel RAY-DIRECTION conditioning (each pixel's viewing ray
  from camera intrinsics is concatenated to features) and a "rotation around the ray"
  parameterization, refined by a recurrent ConvGRU; a ray-ReLU enforces the predicted
  normal points toward the camera.  Encodes the inductive bias of perspective geometry.

All four are compact faithful random-init cores that reproduce the DISTINCTIVE head /
refinement primitive (ordinal bins, ray conditioning, decoupled-LSTM refinement,
scale-invariant fusion) on small inputs; the heavy ResNeXt/ResNet-101 backbones are
replaced by small representative conv stems so the unrolled draw finishes quickly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared small encoder primitives
# ============================================================


class _ResNeXtBlock(nn.Module):
    """Grouped (cardinality) bottleneck block -- the ResNeXt primitive."""

    def __init__(self, in_ch: int, out_ch: int, cardinality: int = 8, stride: int = 1) -> None:
        super().__init__()
        mid = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(
            mid, mid, 3, stride=stride, padding=1, groups=cardinality, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.down is None else self.down(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + identity)


class _ConvStem(nn.Module):
    """Small 4-stage downsampling conv stem producing multi-scale features."""

    def __init__(self, in_ch: int = 3, base: int = 32, grouped: bool = False) -> None:
        super().__init__()
        block = _ResNeXtBlock if grouped else _BasicBlock
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.layer1 = block(base, base * 2, stride=2)
        self.layer2 = block(base * 2, base * 4, stride=2)
        self.layer3 = block(base * 4, base * 4, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return c1, c2, c3


class _BasicBlock(nn.Module):
    """Plain residual block (ResNet primitive) used by DORN's ResNet stem."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, cardinality: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.down is None else self.down(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + identity)


# ============================================================
# DiverseDepth (ResNeXt encoder + scale-invariant fusion decoder)
# ============================================================


class DiverseDepth(nn.Module):
    """ResNeXt encoder + multi-scale fusion -> relative (scale-invariant) depth."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.encoder = _ConvStem(3, base, grouped=True)  # ResNeXt cardinality blocks
        self.lat3 = nn.Conv2d(base * 4, base * 2, 1)
        self.lat2 = nn.Conv2d(base * 4, base * 2, 1)
        self.lat1 = nn.Conv2d(base * 2, base * 2, 1)
        self.fuse = nn.Sequential(nn.Conv2d(base * 2, base, 3, padding=1), nn.ReLU(inplace=True))
        self.depth_head = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1, c2, c3 = self.encoder(x)
        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(
            p3, size=c2.shape[2:], mode="bilinear", align_corners=False
        )
        p1 = self.lat1(c1) + F.interpolate(
            p2, size=c1.shape[2:], mode="bilinear", align_corners=False
        )
        feat = self.fuse(p1)
        depth = self.depth_head(feat)
        # Scale-invariant relative depth: normalize by spatial mean (shape, not scale).
        depth = F.interpolate(depth, size=x.shape[2:], mode="bilinear", align_corners=False)
        return depth - depth.mean(dim=(2, 3), keepdim=True)


# ============================================================
# DORN (ordinal-regression discretized-depth head)
# ============================================================


class _ASPP(nn.Module):
    """Atrous spatial pyramid pooling scene-understanding module (DORN)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_ch, out_ch, 1),
                nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6),
                nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12),
                nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18),
            ]
        )
        self.project = nn.Conv2d(out_ch * 4, out_ch, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.relu(b(x)) for b in self.branches]
        return self.relu(self.project(torch.cat(feats, dim=1)))


class DORN(nn.Module):
    """DORN: ResNet stem + ASPP + ordinal-regression head (2K logits, K depth bins)."""

    def __init__(self, base: int = 32, num_bins: int = 40) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.encoder = _ConvStem(3, base, grouped=False)  # ResNet basic blocks
        self.aspp = _ASPP(base * 4, base * 2)
        # Ordinal head: 2*K channels = K independent binary ordinal classifiers.
        self.ord_head = nn.Conv2d(base * 2, 2 * num_bins, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, c3 = self.encoder(x)
        feat = self.aspp(c3)
        logits = self.ord_head(feat)  # (B, 2K, h, w)
        b, _, h, w = logits.shape
        logits = logits.view(b, self.num_bins, 2, h, w)
        # Per-bin ordinal probability P(depth > bin_k).
        prob = torch.softmax(logits, dim=2)[:, :, 1]  # (B, K, h, w)
        # Discretized depth = number of active ordinal levels.
        depth = (prob > 0.5).float().sum(dim=1, keepdim=True)
        depth = F.interpolate(depth, size=x.shape[2:], mode="bilinear", align_corners=False)
        return depth


# ============================================================
# DLNR (decoupled-LSTM stereo refinement)
# ============================================================


class _DecoupledLSTMCell(nn.Module):
    """ConvLSTM update operator with decoupled hi/lo-frequency hidden states (DLNR)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv_hi = nn.Conv2d(ch * 2, ch * 4, 3, padding=1)
        self.conv_lo = nn.Conv2d(ch * 2, ch * 4, 3, padding=1)
        self.ch = ch

    def _gate(self, conv: nn.Conv2d, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = conv(torch.cat([x, h], dim=1))
        i, f, g, o = z.chunk(4, dim=1)
        c = torch.sigmoid(f) * h + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c)

    def forward(self, x: torch.Tensor, h_hi: torch.Tensor, h_lo: torch.Tensor):
        # Decouple: high-frequency branch and low-frequency branch update separately.
        new_hi = self._gate(self.conv_hi, x, h_hi)
        new_lo = self._gate(self.conv_lo, x, h_lo)
        return new_hi, new_lo


class DLNR(nn.Module):
    """DLNR stereo: feature corr -> iterative decoupled-LSTM disparity refinement."""

    def __init__(self, ch: int = 32, iters: int = 6) -> None:
        super().__init__()
        self.iters = iters
        self.fnet = nn.Sequential(
            nn.Conv2d(3, ch, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.corr_proj = nn.Conv2d(ch * 2, ch, 3, padding=1)
        self.cell = _DecoupledLSTMCell(ch)
        self.disp_head = nn.Conv2d(ch, 1, 3, padding=1)
        self.norm_refine = nn.Conv2d(ch, ch, 3, padding=1)  # normalization refinement

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl = self.fnet(left)
        fr = self.fnet(right)
        x = self.corr_proj(torch.cat([fl, fr], dim=1))  # correlation context
        h_hi = torch.zeros_like(x)
        h_lo = torch.zeros_like(x)
        disp = None
        for _ in range(self.iters):
            h_hi, h_lo = self.cell(x, h_hi, h_lo)
            refined = h_hi + self.norm_refine(h_lo)  # fuse hi/lo with norm refinement
            disp = self.disp_head(refined)
            x = x + refined  # carry refined context forward
        disp = F.interpolate(disp, size=left.shape[2:], mode="bilinear", align_corners=False)
        return disp


# ============================================================
# DSINE (ray-conditioned surface-normal estimation)
# ============================================================


class _ConvGRUCell(nn.Module):
    """ConvGRU recurrent refinement cell (DSINE iterative refinement)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.convz = nn.Conv2d(ch * 2, ch, 3, padding=1)
        self.convr = nn.Conv2d(ch * 2, ch, 3, padding=1)
        self.convq = nn.Conv2d(ch * 2, ch, 3, padding=1)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class DSINE(nn.Module):
    """DSINE v02: ray-direction conditioning + ConvGRU refinement + ray-ReLU normals."""

    def __init__(self, ch: int = 32, iters: int = 4) -> None:
        super().__init__()
        self.iters = iters
        # Encoder takes image (3) + per-pixel ray direction (3) -> ray conditioning.
        self.encoder = nn.Sequential(
            nn.Conv2d(6, ch, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gru = _ConvGRUCell(ch)
        self.normal_head = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, img: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        x = self.encoder(torch.cat([img, rays], dim=1))
        h = torch.zeros_like(x)
        normal = None
        rays_ds = F.interpolate(rays, size=x.shape[2:], mode="bilinear", align_corners=False)
        for _ in range(self.iters):
            h = self.gru(h, x)
            normal = self.normal_head(h)
            # Ray-ReLU: enforce the normal points toward the camera (n . ray <= 0).
            dot = (normal * rays_ds).sum(dim=1, keepdim=True)
            normal = normal - F.relu(dot) * rays_ds
        normal = F.interpolate(normal, size=img.shape[2:], mode="bilinear", align_corners=False)
        return F.normalize(normal, dim=1)


# ============================================================
# Wrappers + menagerie wiring (single-tensor forward for the atlas)
# ============================================================


class _StereoWrapper(nn.Module):
    """Stereo wrapper: synthesizes the right view from a single left image."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, left: torch.Tensor) -> torch.Tensor:
        right = torch.roll(left, shifts=4, dims=3)  # horizontally-shifted right view
        return self.model(left, right)


class _DSINEWrapper(nn.Module):
    """DSINE wrapper: builds per-pixel camera rays internally from a single image."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        b, _, h, w = img.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        zz = torch.ones_like(xx)
        rays = torch.stack([xx, yy, zz], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        rays = F.normalize(rays, dim=1)
        return self.model(img, rays)


def build_diverse_depth() -> nn.Module:
    """Build DiverseDepth (ResNeXt encoder + scale-invariant relative-depth decoder)."""
    return DiverseDepth(base=24).eval()


def example_input_depth() -> torch.Tensor:
    """Example RGB image ``(1, 3, 128, 128)`` for monocular depth."""
    return torch.randn(1, 3, 128, 128)


def build_dorn() -> nn.Module:
    """Build DORN (ResNet stem + ASPP + ordinal-regression discretized-depth head)."""
    return DORN(base=24, num_bins=40).eval()


def build_dlnr() -> nn.Module:
    """Build DLNR stereo (decoupled-LSTM iterative disparity refinement)."""
    return _StereoWrapper(DLNR(ch=24, iters=6)).eval()


def example_input_stereo() -> torch.Tensor:
    """Example left image ``(1, 3, 96, 192)`` (right view synthesized internally)."""
    return torch.randn(1, 3, 96, 192)


def build_dsine() -> nn.Module:
    """Build DSINE v02 (ray-conditioned ConvGRU surface-normal estimation)."""
    return _DSINEWrapper(DSINE(ch=24, iters=4)).eval()


def example_input_normal() -> torch.Tensor:
    """Example RGB image ``(1, 3, 120, 160)`` for surface-normal estimation."""
    return torch.randn(1, 3, 120, 160)


MENAGERIE_ENTRIES = [
    (
        "DiverseDepth (ResNeXt encoder + scale-invariant relative-depth decoder)",
        "build_diverse_depth",
        "example_input_depth",
        "2020",
        "DC",
    ),
    (
        "DORN (ordinal-regression discretized-depth head)",
        "build_dorn",
        "example_input_depth",
        "2018",
        "DC",
    ),
    (
        "DLNR (decoupled-LSTM high-frequency stereo refinement)",
        "build_dlnr",
        "example_input_stereo",
        "2023",
        "DC",
    ),
    (
        "DSINE v02 (ray-conditioned ConvGRU surface-normal estimation)",
        "build_dsine",
        "example_input_normal",
        "2024",
        "DC",
    ),
]
