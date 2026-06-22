"""TOFlow / Task-Oriented Flow for Video Super-Resolution (x4).

Xue et al., IJCV 2019.
Paper: https://arxiv.org/abs/1711.09078
Source: https://github.com/anchen1011/toflow

Distinctive primitive: Task-Oriented Flow.
Unlike optical flow estimated purely for geometric accuracy, TOFlow learns
flows that are directly optimized for the downstream task (here: VSR).  The
flow estimation is inspired by SpyNet: a small spatial-pyramid network
estimates a coarse-to-fine flow, then the neighboring frames are warped
to align with the reference (centre) frame, and a fusion/reconstruction
network upsamples the aligned multi-frame stack.

Pipeline:
  1. FlowNet (SpyNet-style, 2-level): for each non-reference frame, estimate
     flow w.r.t. the reference frame.
  2. Warp each non-reference frame to the reference using the estimated flow.
  3. Concat all warped frames with the reference -> fusion conv net.
  4. Pixel-shuffle x4 upsample.

Compact: 3-frame clip, tiny SpyNet-style flow net, narrow fusion net.
Input: (1, T=3, C=3, H=32, W=32) clip — middle frame is the reference.
Output: (1, 3, H*4, W*4) super-resolved reference frame.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------
# Backward warp
# -----------------------------------------------------------------------


def backwarp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward-warp img (B,C,H,W) with flow (B,2,H,W)."""
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
# SpyNet-style task-oriented flow estimator (compact 2-level)
# -----------------------------------------------------------------------


class TOFlowNet(nn.Module):
    """Task-Oriented Flow: 2-level spatial pyramid estimator.

    Takes (frame, reference) concatenated -> 2ch flow.
    Learns task-oriented (not purely geometric) flow end-to-end.
    """

    def __init__(self, ch: int = 16) -> None:
        super().__init__()
        # Level 1 (coarse): at 1/2 resolution
        self.level1 = nn.Sequential(
            nn.Conv2d(6, ch, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2, 1),
        )
        # Level 2 (fine): at full resolution, takes [frame | ref | upsampled_flow]
        self.level2 = nn.Sequential(
            nn.Conv2d(6 + 2, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2, 1),
        )

    def forward(self, frame: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Estimate flow from frame to ref. Both (B,3,H,W)."""
        # Coarse level
        f_coarse = F.avg_pool2d(frame, 2)
        r_coarse = F.avg_pool2d(ref, 2)
        flow_c = self.level1(torch.cat([f_coarse, r_coarse], dim=1))  # (B,2,H/2,W/2)

        # Upsample flow to full resolution
        flow_up = F.interpolate(flow_c, scale_factor=2, mode="bilinear", align_corners=True) * 2.0

        # Fine level
        flow_f = self.level2(torch.cat([frame, ref, flow_up], dim=1))
        return flow_up + flow_f  # (B, 2, H, W)


# -----------------------------------------------------------------------
# Reconstruction / fusion net
# -----------------------------------------------------------------------


class FusionReconNet(nn.Module):
    """Conv fusion + pixel-shuffle x4."""

    def __init__(self, in_ch: int, ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3 * 16, 3, padding=1),  # 16 = 4*4
        )
        self.pix_shuffle = nn.PixelShuffle(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pix_shuffle(self.net(x))


# -----------------------------------------------------------------------
# TOFlow VSR model
# -----------------------------------------------------------------------


class TOFVSRx4(nn.Module):
    """Task-Oriented Flow video super-resolution x4 (compact).

    Forward:
        x: (B, T, 3, H, W) video clip. Middle frame (T//2) is reference.
    Returns:
        (B, 3, H*4, W*4) super-resolved reference frame.
    """

    def __init__(self, n_frames: int = 3, ch: int = 16, fuse_ch: int = 32) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.ref_idx = n_frames // 2

        # Shared task-oriented flow net (one per non-reference frame,
        # but parameter-shared across frames — same network instance)
        self.flow_net = TOFlowNet(ch)

        # Fusion: all warped frames + reference -> HR
        fusion_in = n_frames * 3
        self.fusion = FusionReconNet(fusion_in, fuse_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        ref = x[:, self.ref_idx]  # (B, 3, H, W)

        # Align each non-reference frame to the reference via task-oriented flow
        aligned = []
        for t in range(T):
            if t == self.ref_idx:
                aligned.append(ref)
            else:
                flow = self.flow_net(x[:, t], ref)
                aligned.append(backwarp(x[:, t], flow))

        # Fuse all aligned frames
        fused = torch.cat(aligned, dim=1)  # (B, T*3, H, W)
        return self.fusion(fused)  # (B, 3, H*4, W*4)


# -----------------------------------------------------------------------
# Menagerie wiring
# -----------------------------------------------------------------------


def build_tof_vsr_x4() -> nn.Module:
    """Build TOFlow VSR x4 (3-frame, compact)."""
    return TOFVSRx4(n_frames=3, ch=16, fuse_ch=32)


def example_input_tof() -> torch.Tensor:
    """3-frame clip (1, 3, 3, 32, 32) = (B, T, C, H, W)."""
    return torch.randn(1, 3, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "TOFlow-VSR x4 (Xue 2019, task-oriented flow warp-align-fuse video SR)",
        "build_tof_vsr_x4",
        "example_input_tof",
        "2019",
        "DC",
    ),
]
