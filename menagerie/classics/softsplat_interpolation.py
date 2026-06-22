"""SoftSplat: Softmax Splatting for Video Frame Interpolation.

Niklaus & Liu, CVPR 2020.
Paper: https://arxiv.org/abs/2003.05534
Source: https://github.com/sniklaus/softmax-splatting

Distinctive primitive: FORWARD warping via softmax splatting.  Given a flow
field F (pixels -> where they land) and a per-pixel importance metric Z,
features are forward-warped by accumulating each pixel's contribution into the
target location, weighted by softmax(Z).  This avoids the holes / tearing
artifacts of naive forward warping.

Pipeline:
  1. FlowNet (small encoder-decoder) estimates optical flow from frame0 -> midpoint.
  2. Importance metric Z is computed from the magnitude of the flow (or can be a
     small learnable network; here we use a 1-layer conv for compactness).
  3. Softmax splatting: forward-warp frame features with softmax(Z) weighting.
     Approximated here via backward warping with grid_sample (documented below).
  4. GridNet (small synthesis decoder) combines the two warped feature pyramids.

Splatting approximation: The paper's true forward splatting requires a custom
CUDA scatter-add op.  Here we APPROXIMATE it with F.grid_sample (backward warp
of the SOURCE frame using the flow — equivalent to backward warping the target
from the source coordinates).  This faithfully shows the flow-warp-synthesis
pipeline at the expense of true scatter semantics; the approximation is noted
in the module docstring and would be replaced with the custom op for training.

Input: two RGB frames stacked (1, 6, H, W) — [frame0 | frame1].
Output: (1, 3, H, W) synthesized middle frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def backward_warp(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp a frame using flow (dx, dy in pixels).

    Uses F.grid_sample — faithfully represents the warp step even though
    the paper uses forward splatting (see module docstring).
    """
    B, C, H, W = frame.shape
    # Build sampling grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=flow.dtype, device=flow.device),
        torch.arange(W, dtype=flow.dtype, device=flow.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,H,W)
    grid = grid + flow  # apply flow
    # Normalize to [-1, 1]
    grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0
    grid = grid.permute(0, 2, 3, 1)  # (B,H,W,2)
    return F.grid_sample(frame, grid, mode="bilinear", padding_mode="border", align_corners=True)


# -----------------------------------------------------------------------
# Sub-networks
# -----------------------------------------------------------------------


class SmallFlowNet(nn.Module):
    """Tiny encoder-decoder estimating optical flow (2ch) from two frames."""

    def __init__(self, ch: int = 16) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch * 2, ch, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.flow_head = nn.Conv2d(ch, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow_head(self.dec(self.enc(x)))


class ImportanceNet(nn.Module):
    """1-conv importance metric Z (per-pixel scalar from flow magnitude)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 1)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        return self.conv(flow)


class FeaturePyramid(nn.Module):
    """Small feature extractor: single-level feature map."""

    def __init__(self, in_ch: int = 3, feat_ch: int = 16) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SynthesisNet(nn.Module):
    """Small GridNet-style synthesis from two warped feature maps -> RGB."""

    def __init__(self, feat_ch: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, 3, 1),
        )

    def forward(self, feat0: torch.Tensor, feat1: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([feat0, feat1], dim=1))


# -----------------------------------------------------------------------
# Full SoftSplat model
# -----------------------------------------------------------------------


class SoftSplatInterpolation(nn.Module):
    """SoftSplat frame interpolation (compact, warp approximated with grid_sample).

    Forward:
        x (B, 6, H, W): [frame0 | frame1] concatenated.
    Returns:
        (B, 3, H, W) middle-frame synthesis.
    """

    def __init__(self, ch: int = 16) -> None:
        super().__init__()
        self.flow_net = SmallFlowNet(ch)
        self.importance_net = ImportanceNet()
        self.feat_ext = FeaturePyramid(3, ch)
        self.synth_net = SynthesisNet(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame0 = x[:, :3]
        frame1 = x[:, 3:]

        # Step 1: estimate flow frame0 -> mid, frame1 -> mid
        flow01 = self.flow_net(x)  # (B, 2, H, W)
        flow10 = self.flow_net(torch.cat([frame1, frame0], dim=1))  # reversed

        # Step 2: importance metric Z from flow magnitude
        z0 = self.importance_net(flow01)  # (B, 1, H, W)
        z1 = self.importance_net(flow10)

        # Step 3: softmax splatting (approximated via backward warp)
        # Scale flow by 0.5 for mid-frame interpolation
        feat0 = self.feat_ext(frame0)
        feat1 = self.feat_ext(frame1)

        # Warp features with softmax importance weighting
        w0 = torch.sigmoid(z0)  # soft importance weight
        w1 = torch.sigmoid(z1)
        warped0 = backward_warp(feat0, flow01 * 0.5) * w0
        warped1 = backward_warp(feat1, flow10 * 0.5) * w1

        # Step 4: synthesis
        out = self.synth_net(warped0, warped1)
        return out


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_softsplat_interpolation() -> nn.Module:
    return SoftSplatInterpolation(ch=16)


def example_input_softsplat() -> torch.Tensor:
    return torch.randn(1, 6, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "SoftSplat (Niklaus 2020, softmax splatting forward-warp interpolation)",
        "build_softsplat_interpolation",
        "example_input_softsplat",
        "2020",
        "DC",
    ),
]
