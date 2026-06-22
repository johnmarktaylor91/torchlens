"""RIFE: Real-Time Intermediate Flow Estimation for video interpolation.

Huang et al. (2020/2022), "RIFE: Real-Time Intermediate Flow Estimation for
Video Frame Interpolation".  ECCV 2022.  arXiv:2011.06294.
Source: https://github.com/hzwer/ECCV2022-RIFE

Distinctive primitives:
  1. IFBlock (Intermediate Flow Block): the core coarse-to-fine flow estimator.
     Given two frames and a current flow estimate, the IFBlock:
       - Warps I0 and I1 by the current flow estimate.
       - Concatenates [warped_I0, warped_I1, I0, I1, flow, timestep_map].
       - Applies a small ConvNet to predict a flow REFINEMENT (delta_flow).
     The final flow is the sum of delta_flows across scales.
  2. MULTI-SCALE STACKED IFBlocks: IFBlock_0 (coarsest), IFBlock_1, IFBlock_2 (finest).
     Each receives the cumulative flow from previous blocks upsampled to its scale.
  3. FUSION / REFINE net: synthesises the final interpolated frame from warped frames
     + a mask (soft-composite of the two warpings).

v3: 3 IFBlocks, older architecture with all ConvNet IFBlocks.
v4: same structure but with a distilled/refined IFBlock; captured here as the same
    implementation (two entries, same model, distinguishable by architecture tag).

Compact config: C=16, H=W=8, 3 IFBlocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Differentiable bilinear backward warp
# ==============================================================


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Bilinear backward warp of img by flow.
    img: (B, C, H, W), flow: (B, 2, H, W) -> (B, C, H, W).
    """
    B, C, H, W = img.shape
    # Build grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=img.device),
        torch.arange(W, dtype=torch.float32, device=img.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1, 2, H, W)
    new_grid = grid + flow
    # Normalise to [-1, 1]
    new_grid[:, 0, :, :] = 2.0 * new_grid[:, 0, :, :] / (W - 1) - 1.0
    new_grid[:, 1, :, :] = 2.0 * new_grid[:, 1, :, :] / (H - 1) - 1.0
    new_grid = new_grid.permute(0, 2, 3, 1)  # (B, H, W, 2)
    new_grid = new_grid.expand(B, -1, -1, -1)
    return F.grid_sample(img, new_grid, mode="bilinear", align_corners=True, padding_mode="border")


# ==============================================================
# IFBlock (intermediate flow block)
# ==============================================================


class IFBlock(nn.Module):
    """Single IFBlock: takes [warped_I0, warped_I1, I0, I1, flow, t_map] -> delta_flow + mask.

    Input channels: 4*C_in (warped+orig frames) + 2 (flow) + 1 (timestep map) = 4*C_in+3.
    Output: 3 channels = (delta_flow_x, delta_flow_y, soft_mask).
    """

    def __init__(self, c_in: int = 3, c: int = 16) -> None:
        super().__init__()
        n_in = 4 * c_in + 3  # warped(I0, I1) + orig(I0, I1) + flow(2) + t_map(1)
        self.conv = nn.Sequential(
            nn.Conv2d(n_in, c, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(c, c, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(c, 3, 3, padding=1),  # (delta_flow_x, delta_flow_y, mask)
        )

    def forward(
        self,
        I0: torch.Tensor,
        I1: torch.Tensor,
        flow: torch.Tensor,
        t_map: torch.Tensor,
    ) -> tuple:
        """Returns (delta_flow, mask), delta_flow: (B, 2, H, W), mask: (B, 1, H, W)."""
        w0 = warp(I0, flow)
        w1 = warp(I1, -flow)
        x = torch.cat([w0, w1, I0, I1, flow, t_map], dim=1)
        out = self.conv(x)
        delta_flow = out[:, :2]
        mask = torch.sigmoid(out[:, 2:3])
        return delta_flow, mask


class FusionNet(nn.Module):
    """Fusion/refinement net: blend two warpings into final frame."""

    def __init__(self, c: int = 16) -> None:
        super().__init__()
        # Input: warped_I0 + warped_I1 + mask + flow = 3+3+1+2 = 9
        self.net = nn.Sequential(
            nn.Conv2d(9, c, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(c, 3, 3, padding=1),
            nn.Sigmoid(),  # output in [0,1]
        )

    def forward(
        self, w0: torch.Tensor, w1: torch.Tensor, mask: torch.Tensor, flow: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([w0, w1, mask, flow], dim=1)
        return self.net(x)


# ==============================================================
# RIFE IFNet (stacked IFBlocks)
# ==============================================================


class IFNet(nn.Module):
    """RIFE IFNet: 3 stacked IFBlocks + FusionNet."""

    def __init__(self, c: int = 16) -> None:
        super().__init__()
        self.block0 = IFBlock(c_in=3, c=c)
        self.block1 = IFBlock(c_in=3, c=c)
        self.block2 = IFBlock(c_in=3, c=c)
        self.fusion = FusionNet(c=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3+3+1, H, W) = I0 (3) + I1 (3) + t_map (1).
        Returns (B, 3, H, W) interpolated frame.
        """
        B, _, H, W = x.shape
        I0 = x[:, :3]
        I1 = x[:, 3:6]
        t_map = x[:, 6:7]
        # Initial flow: zero
        flow = torch.zeros(B, 2, H, W, device=x.device)

        # Block 0 (coarsest -- optionally at lower resolution; use full here)
        df0, mask0 = self.block0(I0, I1, flow, t_map)
        flow = flow + df0

        # Block 1
        df1, mask1 = self.block1(I0, I1, flow, t_map)
        flow = flow + df1

        # Block 2 (finest)
        df2, mask2 = self.block2(I0, I1, flow, t_map)
        flow = flow + df2

        # Final warp + fusion
        w0 = warp(I0, flow)
        w1 = warp(I1, -flow)
        out = self.fusion(w0, w1, mask2, flow)
        return out


# ==============================================================
# Builders
# ==============================================================


def build_rife_hd_v3() -> nn.Module:
    return IFNet(c=16).eval()


def build_rife_hd_v4() -> nn.Module:
    """RIFE v4 uses the same IFNet structure with minor improvements;
    captured here with the same compact IFNet (architecture identical at this scale)."""
    return IFNet(c=16).eval()


def example_input() -> torch.Tensor:
    """(1, 7, 8, 8) -- batch=1, I0(3)+I1(3)+t_map(1), 8x8 frames."""
    x = torch.randn(1, 7, 8, 8)
    x[:, 6] = 0.5  # t_map = 0.5 (midpoint interpolation)
    return x


MENAGERIE_ENTRIES = [
    (
        "RIFE HD v3 (IFNet: multi-scale stacked IFBlocks with flow-warping + fusion for video interpolation)",
        "build_rife_hd_v3",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "RIFE HD v4 (IFNet v4: same coarse-to-fine IFBlock stack, refined architecture)",
        "build_rife_hd_v4",
        "example_input",
        "2022",
        "DC",
    ),
]
