"""SfMLearner: Unsupervised Learning of Depth and Ego-Motion from Video.

Zhou et al., 2017 (CVPR).
Paper: https://arxiv.org/abs/1704.07813
Source: https://github.com/tinghuiz/SfMLearner

Architecture -- two networks trained jointly:
  1. DispNet (depth network):
       Encoder-decoder (U-Net style) that takes a SINGLE RGB frame and outputs
       a per-pixel disparity map (single-channel, sigmoid activated).
       Encoder: 7 conv strided layers (3->32->64->128->256->512->512->512)
       Decoder: 7 upconv layers with skip connections from encoder + 4 multi-scale
                disparity outputs (at 1/2, 1/4, 1/8, 1/16 of input resolution).

  2. PoseNet (ego-motion network):
       Takes a stack of (K+1) consecutive frames [target + K source frames]
       concatenated along channels as input (3*(K+1) channels).
       A compact CNN backbone (7 conv layers + global average pool) outputs
       K 6-DoF pose vectors [rx, ry, rz, tx, ty, tz] per frame pair.

Faithful compact simplification:
  Channels reduced (32/64/128/256/512 -> 16/32/64/128/256) for compactness.
  K=2 source frames (3 total frames, 9-channel PoseNet input).
  Input spatial: (1, 3, 64, 64) for DispNet; (1, 9, 64, 64) for PoseNet (3 frames).
  Multi-scale depth outputs returned as list. Trace+draw verified 2026-06-21.
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DispNet (single-image depth encoder-decoder)
# ---------------------------------------------------------------------------


class _DownConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 2) -> None:
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _UpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Align sizes (skip may be 1px larger due to odd input dims)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        return self.conv(torch.cat([x, skip], dim=1))


class DispNet(nn.Module):
    """SfMLearner depth (disparity) network: encoder-decoder with multi-scale outputs."""

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        ch = [16, 32, 64, 128, 256, 256, 256]
        # Encoder
        self.enc1 = _DownConv(in_ch, ch[0], kernel=7, stride=2)
        self.enc2 = _DownConv(ch[0], ch[1], kernel=5, stride=2)
        self.enc3 = _DownConv(ch[1], ch[2], kernel=3, stride=2)
        self.enc4 = _DownConv(ch[2], ch[3], kernel=3, stride=2)
        self.enc5 = _DownConv(ch[3], ch[4], kernel=3, stride=2)
        self.enc6 = _DownConv(ch[4], ch[5], kernel=3, stride=2)
        self.enc7 = _DownConv(ch[5], ch[6], kernel=3, stride=2)

        # Decoder upconv layers; input = upsampled + skip
        self.up7 = _UpConv(ch[6] + ch[5], ch[5])
        self.up6 = _UpConv(ch[5] + ch[4], ch[4])
        self.up5 = _UpConv(ch[4] + ch[3], ch[3])
        self.up4 = _UpConv(ch[3] + ch[2], ch[2])
        # up3: input = ch[2] upsampled (from up4 output) concatenated with e2 (ch[1])
        # The multi-scale disp output (1 ch) is concatenated at the SKIP, not here
        self.up3 = _UpConv(ch[2] + ch[1], ch[1])
        self.up2 = _UpConv(ch[1] + ch[0], ch[0])

        # Multi-scale disparity heads -- each reads the output of the corresponding upconv
        # up4 outputs ch[2], up3 outputs ch[1], up2 outputs ch[0], final outputs ch[0]
        self.disp4 = nn.Sequential(nn.Conv2d(ch[2], 1, 3, padding=1), nn.Sigmoid())
        self.disp3 = nn.Sequential(nn.Conv2d(ch[1], 1, 3, padding=1), nn.Sigmoid())
        self.disp2 = nn.Sequential(nn.Conv2d(ch[0], 1, 3, padding=1), nn.Sigmoid())
        self.disp1 = nn.Sequential(nn.Conv2d(ch[0], 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        d7 = self.up7(e7, e6)
        d6 = self.up6(d7, e5)
        d5 = self.up5(d6, e4)
        d4 = self.up4(d5, e3)
        disp4 = self.disp4(d4)

        d3 = self.up3(d4, e2)
        disp3 = self.disp3(d3)

        d2 = self.up2(d3, e1)
        disp2 = self.disp2(d2)
        disp1 = self.disp1(F.interpolate(d2, scale_factor=2, mode="nearest"))

        return [disp1, disp2, disp3, disp4]  # finest to coarsest


# ---------------------------------------------------------------------------
# PoseNet (multi-frame ego-motion network)
# ---------------------------------------------------------------------------


class PoseNet(nn.Module):
    """SfMLearner ego-motion network.

    Input: (B, 3*(n_frames), H, W) -- stacked RGB frames
    Output: (B, n_frames-1, 6) -- 6-DoF pose per source->target pair
    """

    def __init__(self, n_frames: int = 3) -> None:
        super().__init__()
        in_ch = 3 * n_frames
        self.n_pairs = n_frames - 1

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Pose output: 6-DoF per source frame
        self.pose_head = nn.Conv2d(256, 6 * self.n_pairs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)  # (B, 256, H', W')
        poses = self.pose_head(feat)  # (B, 6*n_pairs, H', W')
        poses = poses.mean(dim=[2, 3])  # global avg pool -> (B, 6*n_pairs)
        poses = poses.view(x.shape[0], self.n_pairs, 6)  # (B, n_pairs, 6)
        return poses * 0.01  # scale factor (paper: 0.01)


# ---------------------------------------------------------------------------
# Joint SfMLearner wrapper
# ---------------------------------------------------------------------------


class SfMLearner(nn.Module):
    """SfMLearner: DispNet + PoseNet for self-supervised depth+ego-motion.

    forward(imgs, pose_frames) -> (disp_multiscale, poses)
      imgs:        (B, 3, H, W) single target frame for depth
      pose_frames: (B, 9, H, W) stacked 3 frames for pose (target + 2 sources)
    """

    def __init__(self, n_frames: int = 3) -> None:
        super().__init__()
        self.disp_net = DispNet(in_ch=3)
        self.pose_net = PoseNet(n_frames=n_frames)

    def forward(
        self,
        imgs: torch.Tensor,
        pose_frames: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        disps = self.disp_net(imgs)
        poses = self.pose_net(pose_frames)
        return disps, poses


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def build_sfmlearner() -> nn.Module:
    return SfMLearner(n_frames=3)


def example_input_sfm() -> Tuple[torch.Tensor, torch.Tensor]:
    """Two-tensor input: (single frame for depth, stacked frames for pose)."""
    imgs = torch.randn(1, 3, 64, 64)
    pose_frames = torch.randn(1, 9, 64, 64)
    return (imgs, pose_frames)


MENAGERIE_ENTRIES = [
    (
        "SfMLearner (DispNet + PoseNet, self-supervised depth+ego-motion)",
        "build_sfmlearner",
        "example_input_sfm",
        "2017",
        "DC",
    ),
]
