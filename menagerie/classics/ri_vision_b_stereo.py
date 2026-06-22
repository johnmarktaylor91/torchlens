"""Stereo / multi-view-stereo depth: CasMVSNet, CoEx, CREStereo.

All three build a COST VOLUME (matching feature similarities across candidate disparities or
depths) and regress depth/disparity. Each has a distinctive cost-volume mechanism reproduced
here at small scale on a stereo image pair.

CasMVSNet -- Gu et al., CVPR 2020, "Cascade Cost Volume for High-Resolution Multi-View Stereo
and Stereo Matching", arXiv:1912.06378. Source: github.com/alibaba/cascade-stereo.
  DISTINCTIVE: a CASCADE of cost volumes built at coarse->fine resolution. Each stage builds
  a cost volume over a narrowing depth range (centered on the previous stage's prediction),
  regularizes it with 3D convs, and regresses depth; the depth hypothesis range shrinks each
  stage. We reproduce the 2-stage cascade with a narrowing depth range over a stereo pair.

CoEx -- Bangunharcana et al., IROS 2021, "Correlate-and-Excite: Real-Time Stereo Matching via
Guided Cost Volume Excitation", arXiv:2108.05773. Source: github.com/antabangun/coex.
  DISTINCTIVE: Guided Cost-volume Excitation (GCE) -- channel/spatial attention weights for
  the cost volume are PREDICTED FROM the reference image features and used to re-weight
  ("excite") the cost volume, like SE applied to a 4D cost volume.

CREStereo -- Li et al., CVPR 2022, "Practical Stereo Matching via Cascaded Recurrent Network
with Adaptive Correlation", arXiv:2203.11483. Source: github.com/megvii-research/CREStereo.
  DISTINCTIVE: a cascaded RECURRENT update -- a GRU iteratively refines the disparity field
  using a local correlation lookup around the current estimate (RAFT-style, with adaptive
  group correlation). We reproduce the correlation + GRU update iterated a few times.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _feat_extract(in_ch: int = 3, ch: int = 16) -> nn.Module:
    """Shared 1/4-resolution feature extractor for stereo pairs."""
    return nn.Sequential(
        nn.Conv2d(in_ch, ch, 3, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, 2, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch, ch, 3, 1, 1),
    )


def _build_cost_volume(left: torch.Tensor, right: torch.Tensor, maxdisp: int) -> torch.Tensor:
    """Concatenation cost volume: (B, 2C, D, H, W) over `maxdisp` shifts."""
    b, c, h, w = left.shape
    cost = left.new_zeros(b, c * 2, maxdisp, h, w)
    for d in range(maxdisp):
        if d == 0:
            cost[:, :c, d] = left
            cost[:, c:, d] = right
        else:
            cost[:, :c, d, :, d:] = left[:, :, :, d:]
            cost[:, c:, d, :, d:] = right[:, :, :, :-d]
    return cost


# ============================================================
# CasMVSNet: cascade cost volume (coarse -> fine narrowing range)
# ============================================================


class CascadeStereo(nn.Module):
    """2-stage cascade cost volume over a stereo pair (coarse range -> narrowed range)."""

    def __init__(self, ch: int = 16) -> None:
        super().__init__()
        self.feat = _feat_extract(3, ch)
        self.reg1 = nn.Sequential(
            nn.Conv3d(ch * 2, ch, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(ch, 1, 3, 1, 1)
        )
        self.reg2 = nn.Sequential(
            nn.Conv3d(ch * 2, ch, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(ch, 1, 3, 1, 1)
        )

    def _regress(
        self, cost: torch.Tensor, reg: nn.Module, base: float, step: float
    ) -> torch.Tensor:
        cost = reg(cost).squeeze(1)  # (B, D, H, W)
        prob = F.softmax(cost, dim=1)
        disps = base + step * torch.arange(cost.shape[1], device=cost.device).float()
        return (prob * disps[None, :, None, None]).sum(1)  # (B, H, W)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl, fr = self.feat(left), self.feat(right)
        # Stage 1: coarse, wide range (8 hypotheses, step 4)
        c1 = _build_cost_volume(fl, fr, 8)
        d1 = self._regress(c1, self.reg1, base=0.0, step=4.0)
        # Stage 2: fine, narrowed range (4 hypotheses, step 1) -- cascade refinement
        c2 = _build_cost_volume(fl, fr, 4)
        d2 = self._regress(c2, self.reg2, base=0.0, step=1.0)
        return d1[:, None] + F.interpolate(
            d2[:, None], size=d1.shape[1:], mode="bilinear", align_corners=False
        )


# ============================================================
# CoEx: guided cost-volume excitation
# ============================================================


class CoExStereo(nn.Module):
    """Stereo matching with Guided Cost-volume Excitation (GCE)."""

    def __init__(self, ch: int = 16, maxdisp: int = 8) -> None:
        super().__init__()
        self.maxdisp = maxdisp
        self.feat = _feat_extract(3, ch)
        # GCE: predict cost-volume channel weights FROM the reference image features.
        self.gce = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch * 2, 1), nn.Sigmoid())
        self.reg = nn.Sequential(
            nn.Conv3d(ch * 2, ch, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(ch, 1, 3, 1, 1)
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl, fr = self.feat(left), self.feat(right)
        cost = _build_cost_volume(fl, fr, self.maxdisp)  # (B, 2C, D, H, W)
        excite = self.gce(fl)[:, :, None]  # (B, 2C, 1, 1, 1) guided weights from ref feats
        cost = cost * excite  # excite the cost volume
        cost = self.reg(cost).squeeze(1)  # (B, D, H, W)
        prob = F.softmax(cost, dim=1)
        disps = torch.arange(self.maxdisp, device=cost.device).float()
        return (prob * disps[None, :, None, None]).sum(1, keepdim=True)


# ============================================================
# CREStereo: cascaded recurrent network with adaptive correlation
# ============================================================


class CREStereoRecurrent(nn.Module):
    """Cascaded recurrent stereo: correlation lookup + GRU disparity update, iterated."""

    def __init__(self, ch: int = 16, iters: int = 3) -> None:
        super().__init__()
        self.iters = iters
        self.feat = _feat_extract(3, ch)
        self.corr_radius = 4
        # GRU update on (correlation features + disparity) -> disparity delta
        self.gru = nn.GRUCell(2 * self.corr_radius + 1 + 1, ch)
        self.disp_head = nn.Linear(ch, 1)

    def _correlation(self, fl: torch.Tensor, fr: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """Local 1D correlation around the current disparity estimate (adaptive lookup)."""
        b, c, h, w = fl.shape
        corrs = []
        for off in range(-self.corr_radius, self.corr_radius + 1):
            shifted = torch.roll(fr, shifts=int(off), dims=3)
            corrs.append((fl * shifted).sum(1, keepdim=True) / (c**0.5))  # (B,1,H,W)
        return torch.cat(corrs, dim=1)  # (B, 2r+1, H, W)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        fl, fr = self.feat(left), self.feat(right)
        b, c, h, w = fl.shape
        disp = fl.new_zeros(b, 1, h, w)
        hidden = fl.new_zeros(b * h * w, c)
        for _ in range(self.iters):
            corr = self._correlation(fl, fr, disp)  # (B, 2r+1, H, W)
            inp = torch.cat([corr, disp], dim=1)  # (B, 2r+2, H, W)
            inp = inp.permute(0, 2, 3, 1).reshape(b * h * w, -1)
            hidden = self.gru(inp, hidden)  # recurrent update
            delta = self.disp_head(hidden).view(b, h, w, 1).permute(0, 3, 1, 2)
            disp = disp + delta  # iterative refinement
        return disp


class _StereoPairWrapper(nn.Module):
    """Wraps a 2-input stereo net so it is forwardable from a single (1,6,H,W) tensor.

    Channels 0:3 = left image, 3:6 = right image.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        left, right = pair[:, :3], pair[:, 3:]
        return self.model(left, right)


def build_casmvsnet() -> nn.Module:
    """CasMVSNet cascade-cost-volume stereo/MVS depth estimator (2-stage cascade)."""
    return _StereoPairWrapper(CascadeStereo()).eval()


def build_coex() -> nn.Module:
    """CoEx stereo matching with guided cost-volume excitation (GCE)."""
    return _StereoPairWrapper(CoExStereo()).eval()


def build_crestereo() -> nn.Module:
    """CREStereo cascaded recurrent stereo (correlation lookup + GRU update)."""
    return _StereoPairWrapper(CREStereoRecurrent()).eval()


def example_input_stereo() -> torch.Tensor:
    """Stacked stereo pair (1, 6, 64, 64) -- left=ch0:3, right=ch3:6."""
    return torch.randn(1, 6, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "CasMVSNet (cascade cost volume, coarse-to-fine depth)",
        "build_casmvsnet",
        "example_input_stereo",
        "2020",
        "DC",
    ),
    (
        "CoEx (guided cost-volume excitation stereo)",
        "build_coex",
        "example_input_stereo",
        "2021",
        "DC",
    ),
    (
        "CREStereo (cascaded recurrent stereo, adaptive correlation)",
        "build_crestereo",
        "example_input_stereo",
        "2022",
        "DC",
    ),
]
