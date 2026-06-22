"""XVFI: eXtreme Video Frame Interpolation.

Sim et al., ICCV 2021.
Paper: https://arxiv.org/abs/2103.16206
Source: https://github.com/JihyongOh/XVFI

Distinctive primitive: recursive multi-scale shared-parameter pyramid.
XVFI uses a coarse-to-fine architecture where the SAME set of parameters
(BiOF-I / BiOF-T shared subnetwork) is applied at multiple pyramid levels.
At each scale the subnetwork estimates complementary bidirectional optical
flows (BiOF) and the refined frame is propagated to the next finer scale.

Key components:
  - Shared BiOF backbone: a small conv-based flow estimator applied at every
    scale (parameter sharing across scales is the architectural innovation).
  - Complementary flow warping: both frames are warped to the midpoint using
    estimated flows, then combined.
  - Multi-scale recursive refinement: coarse estimate upsampled and used to
    initialize the next level.

Compact: 3-scale pyramid with a tiny shared backbone, 32x32 base resolution.
Input: two RGB frames stacked (1, 6, H, W) — [frame0 | frame1].
Output: (1, 3, H, W) synthesized middle frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Backward warp helper
# -----------------------------------------------------------------------


def backwarp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp img with flow (pixel-space dx, dy)."""
    B, C, H, W = img.shape
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=flow.dtype, device=flow.device),
        torch.arange(W, dtype=flow.dtype, device=flow.device),
        indexing="ij",
    )
    grid = torch.stack([gx, gy], dim=0).unsqueeze(0) + flow
    grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
    return F.grid_sample(
        img, grid.permute(0, 2, 3, 1), mode="bilinear", padding_mode="border", align_corners=True
    )


# -----------------------------------------------------------------------
# Shared BiOF sub-network (applied at every pyramid scale)
# -----------------------------------------------------------------------


class BiOFBackbone(nn.Module):
    """Shared bilateral optical flow estimator (BiOF-I compact version).

    Takes: [frame0 | frame1 | coarse_flow_upsampled] -> refined bidirectional
    flows (4ch: F_0t, F_1t each 2ch).
    """

    def __init__(self, ch: int = 16) -> None:
        super().__init__()
        # Input: 6 (two frames) + 4 (coarse bilateral flows) = 10ch
        self.conv1 = nn.Conv2d(10, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ch * 2, ch * 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.head = nn.Conv2d(ch, 4, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(
        self, frame0: torch.Tensor, frame1: torch.Tensor, coarse_flow: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([frame0, frame1, coarse_flow], dim=1)  # 10ch
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.up(x))
        return self.head(x)  # (B, 4, H, W)


# -----------------------------------------------------------------------
# XVFI model
# -----------------------------------------------------------------------


class XVFINet(nn.Module):
    """XVFI recursive multi-scale shared-parameter interpolation (compact).

    Scales: input is downsampled to [scale//4, scale//2, scale//1] and
    the shared BiOF backbone is applied coarse-to-fine.  This compactly
    demonstrates the recursive shared-parameter pyramid structure.

    Input: x (B, 6, H, W) = [frame0 | frame1].
    Output: (B, 3, H, W) synthesized middle frame.
    """

    def __init__(self, n_scales: int = 3, ch: int = 16) -> None:
        super().__init__()
        self.n_scales = n_scales
        # Single shared BiOF backbone used at ALL scales
        self.biof = BiOFBackbone(ch)
        # Final synthesis: blend using estimated flows
        self.synth = nn.Sequential(
            nn.Conv2d(6 + 4, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame0 = x[:, :3]
        frame1 = x[:, 3:]
        B, C, H, W = frame0.shape

        # Build image pyramids
        frames0 = [frame0]
        frames1 = [frame1]
        for _ in range(self.n_scales - 1):
            frames0.insert(0, F.avg_pool2d(frames0[0], 2))
            frames1.insert(0, F.avg_pool2d(frames1[0], 2))

        # Coarse-to-fine shared-parameter recursion
        flow = torch.zeros(
            B, 4, frames0[0].shape[2], frames0[0].shape[3], device=x.device, dtype=x.dtype
        )
        for s in range(self.n_scales):
            f0_s = frames0[s]
            f1_s = frames1[s]
            # Upsample coarse flow to current scale
            if s > 0:
                flow = (
                    F.interpolate(
                        flow,
                        size=(f0_s.shape[2], f0_s.shape[3]),
                        mode="bilinear",
                        align_corners=True,
                    )
                    * 2.0
                )
            # Shared backbone refines flows at this scale
            delta = self.biof(f0_s, f1_s, flow)  # (B, 4, h, w)
            flow = flow + delta

        # Final: warp both frames and synthesize
        Ft0 = flow[:, :2]
        Ft1 = flow[:, 2:]
        w0 = backwarp(frame0, Ft0)
        w1 = backwarp(frame1, Ft1)
        synth_in = torch.cat([w0, w1, Ft0, Ft1], dim=1)  # 10ch
        out = self.synth(synth_in)
        return out


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_xvfi_net() -> nn.Module:
    """Build XVFI (3-scale shared-parameter pyramid, compact)."""
    return XVFINet(n_scales=3, ch=16)


def example_input_xvfi() -> torch.Tensor:
    """Two frames (1, 6, 64, 64)."""
    return torch.randn(1, 6, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "XVFI (Sim 2021, recursive multi-scale shared-param pyramid interpolation)",
        "build_xvfi_net",
        "example_input_xvfi",
        "2021",
        "DC",
    ),
]
