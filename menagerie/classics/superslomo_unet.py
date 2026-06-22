"""Super-SloMo: High Quality Estimation of Multiple Intermediate Frames.

Jiang et al., CVPR 2018.
Paper: https://arxiv.org/abs/1712.00080
Source: https://github.com/avinashpaliwal/Super-SloMo

Distinctive primitive: Two U-Nets + arbitrary-time interpolation.
  1. FlowComp U-Net: takes two frames -> predicts bidirectional flow
     F_{0->1} and F_{1->0} (4 channels total).
  2. ArbTime U-Net: at interpolation time t in [0,1], takes the two
     frames + backward-warped frames + initial flows -> predicts residual
     intermediate flows dF_t0, dF_t1 and a visibility/occlusion map V.
  3. Backward warp: both frames are backward-warped to time t using the
     refined intermediate flows; V blends them.

This faithfully reproduces the two-U-Net pipeline with backwarp (grid_sample)
and the blend/visibility output.  Compact: narrow U-Nets, small spatial size.
Input: two frames stacked (1, 6, H, W) + scalar t (hardcoded to 0.5 for trace).
       Forward signature: forward(self, x, t) — example_fn returns (x,) and
       t defaults to 0.5.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Backward warp
# -----------------------------------------------------------------------


def backwarp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp img with flow (dx, dy in pixel space)."""
    B, C, H, W = img.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=flow.dtype, device=flow.device),
        torch.arange(W, dtype=flow.dtype, device=flow.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,H,W)
    grid = grid + flow
    grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
    grid = grid.permute(0, 2, 3, 1)  # (B,H,W,2)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


# -----------------------------------------------------------------------
# U-Net components
# -----------------------------------------------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.AvgPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Compact U-Net for Super-SloMo (4-level)."""

    def __init__(self, in_ch: int, out_ch: int, base: int = 16) -> None:
        super().__init__()
        b = base
        self.enc0 = DoubleConv(in_ch, b)
        self.enc1 = Down(b, b * 2)
        self.enc2 = Down(b * 2, b * 4)
        self.enc3 = Down(b * 4, b * 8)
        self.bottleneck = Down(b * 8, b * 8)
        self.dec3 = Up(b * 8, b * 8, b * 4)
        self.dec2 = Up(b * 4, b * 4, b * 2)
        self.dec1 = Up(b * 2, b * 2, b)
        self.dec0 = Up(b, b, b)
        self.head = nn.Conv2d(b, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bn = self.bottleneck(e3)
        d3 = self.dec3(bn, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        return self.head(d0)


# -----------------------------------------------------------------------
# Super-SloMo model
# -----------------------------------------------------------------------


class SuperSloMoUNet(nn.Module):
    """Super-SloMo: two U-Nets + backwarp + occlusion blend.

    Forward:
        x: (B, 6, H, W) — [frame0 | frame1]
        t: float in (0,1), default 0.5
    Returns:
        (B, 3, H, W) interpolated frame at time t.
    """

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        # U-Net 1: FlowComp — 6ch input (two RGB frames) -> 4ch output (F01, F10)
        self.flow_comp = UNet(in_ch=6, out_ch=4, base=base)
        # U-Net 2: ArbTime — input = frame0 + frame1 + warped0 + warped1 + F_t0 + F_t1
        #   = 3+3+3+3+2+2 = 16ch -> 5ch (dF_t0[2] + dF_t1[2] + V[1])
        self.arb_time = UNet(in_ch=16, out_ch=5, base=base)

    def forward(self, x: torch.Tensor, t: float = 0.5) -> torch.Tensor:
        frame0 = x[:, :3]
        frame1 = x[:, 3:]

        # Step 1: compute bidirectional flow
        flows = self.flow_comp(x)  # (B, 4, H, W)
        F01 = flows[:, :2]  # frame0 -> frame1
        F10 = flows[:, 2:]  # frame1 -> frame0

        # Step 2: initial intermediate flows at time t
        Ft0 = -(1 - t) * t * F01 + t * t * F10  # (B,2,H,W)
        Ft1 = (1 - t) * (1 - t) * F01 - t * (1 - t) * F10

        # Step 3: backward-warp frames
        w0 = backwarp(frame0, Ft0)
        w1 = backwarp(frame1, Ft1)

        # Step 4: ArbTime U-Net refines flows + occlusion
        arb_in = torch.cat([frame0, frame1, w0, w1, Ft0, Ft1], dim=1)  # 16ch
        arb_out = self.arb_time(arb_in)  # (B, 5, H, W)
        dFt0 = arb_out[:, :2]
        dFt1 = arb_out[:, 2:4]
        V = torch.sigmoid(arb_out[:, 4:5])  # occlusion/visibility mask

        # Step 5: refine warps and blend
        Ft0_refined = Ft0 + dFt0
        Ft1_refined = Ft1 + dFt1
        g0 = backwarp(frame0, Ft0_refined)
        g1 = backwarp(frame1, Ft1_refined)

        # Blend: V gates the two contributions
        out = V * g0 + (1 - V) * g1
        return out


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_superslomo_unet() -> nn.Module:
    return SuperSloMoUNet(base=16)


def example_input_superslomo() -> torch.Tensor:
    """Two frames stacked (1, 6, 64, 64); t defaults to 0.5."""
    return torch.randn(1, 6, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Super-SloMo (Jiang 2018, two-UNet bidirectional flow frame interpolation)",
        "build_superslomo_unet",
        "example_input_superslomo",
        "2018",
        "DC",
    ),
]
