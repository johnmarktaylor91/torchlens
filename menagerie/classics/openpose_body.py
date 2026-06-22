"""OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields.

Cao et al., 2017 (CVPR); Cao et al., 2019 extended with Body25.
Paper: https://arxiv.org/abs/1812.08008 (Body25); https://arxiv.org/abs/1611.08050 (COCO)
Source: https://github.com/CMU-Perceptual-Computing-Lab/openpose

Architecture:
  VGG-19-derived feature extractor (first 10 conv layers from VGG-19) ->
  S sequential refinement stages, each with TWO branches:
    Branch 1: Part Confidence Maps  (PCM/heatmaps) for J keypoints
    Branch 2: Part Affinity Fields  (PAF)           for L limb-vector pairs
  Each stage receives [feature_map | prev_PCM | prev_PAF] as input.
  PCM has J+1 channels (J joints + background), PAF has 2*L channels (x/y per pair).

Two variants:
  COCO-18:  J=18 keypoints, L=19 limb pairs -> PCM=19ch, PAF=38ch
  Body25:   J=25 keypoints, L=26 limb pairs -> PCM=26ch, PAF=52ch

Faithful compact simplification:
  VGG-19 backbone replaced by a compact 10-conv CNN matching the published
  output stride (no pooling after pool-4, output 3x stride from input).
  Stage branch depth reduced to 5 conv layers (paper uses 7) to keep graph
  compact; channel widths match paper (128 -> 128 -> 128 -> 512 -> out_ch).
  Small input: (1, 3, 64, 64). Trace+draw verified 2026-06-21.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# VGG-19 feature extractor (first 10 convs, 2 max-pools)
# Paper uses vgg19 layers up to "pool4" giving 128-ch feature maps at stride 8
# Compact version: same topology but channels 32/64/128 instead of 64/128/512
# ---------------------------------------------------------------------------


class VGGFeatures(nn.Module):
    """Compact VGG-style feature extractor (mimics VGG-19 first 10 conv layers)."""

    def __init__(self, out_ch: int = 128) -> None:
        super().__init__()
        # Block 1: 2 convs, pool
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Block 2: 2 convs, pool
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Block 3: 4 convs, pool
        self.block3 = nn.Sequential(
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Extra 1x1 projection (paper uses additional 1x1 convs after pool-4)
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Stage branch: 5 conv layers producing either PCM or PAF output
# ---------------------------------------------------------------------------


class StageBranch(nn.Module):
    """One branch of one refinement stage (PCM or PAF)."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# One refinement stage (two branches; input = concat[features, prev_pcm, prev_paf])
# ---------------------------------------------------------------------------


class OpenPoseStage(nn.Module):
    def __init__(
        self,
        feat_ch: int,
        pcm_ch: int,  # J+1
        paf_ch: int,  # 2*L
        is_first: bool = False,
    ) -> None:
        super().__init__()
        in_ch = feat_ch + pcm_ch + paf_ch
        self.branch_pcm = StageBranch(in_ch, pcm_ch)
        self.branch_paf = StageBranch(in_ch, paf_ch)

    def forward(
        self,
        features: torch.Tensor,
        prev_pcm: torch.Tensor,
        prev_paf: torch.Tensor,
    ):
        x = torch.cat([features, prev_pcm, prev_paf], dim=1)
        return self.branch_pcm(x), self.branch_paf(x)


# ---------------------------------------------------------------------------
# Full OpenPose model
# ---------------------------------------------------------------------------


class OpenPose(nn.Module):
    """OpenPose multi-stage pose estimator.

    Args:
        n_joints:   Number of body joints (J).
        n_limbs:    Number of limb pairs (L).
        n_stages:   Number of refinement stages (paper uses 6).
        feat_ch:    Feature extractor output channels.
    """

    def __init__(
        self,
        n_joints: int = 18,
        n_limbs: int = 19,
        n_stages: int = 3,
        feat_ch: int = 128,
    ) -> None:
        super().__init__()
        self.feat_ch = feat_ch
        self.pcm_ch = n_joints + 1  # +1 background channel
        self.paf_ch = 2 * n_limbs  # x,y per limb

        self.backbone = VGGFeatures(feat_ch)

        stages = []
        for i in range(n_stages):
            stages.append(OpenPoseStage(feat_ch, self.pcm_ch, self.paf_ch, is_first=(i == 0)))
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        B, _, H, W = features.shape
        prev_pcm = torch.zeros(B, self.pcm_ch, H, W, device=x.device)
        prev_paf = torch.zeros(B, self.paf_ch, H, W, device=x.device)

        stage_pcms = []
        stage_pafs = []
        for stage in self.stages:
            pcm, paf = stage(features, prev_pcm, prev_paf)
            stage_pcms.append(pcm)
            stage_pafs.append(paf)
            prev_pcm = pcm
            prev_paf = paf

        # Return final stage predictions
        return stage_pcms[-1], stage_pafs[-1]


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def build_openpose_coco18() -> nn.Module:
    """Build OpenPose COCO-18 (18 joints, 19 PAF pairs)."""
    return OpenPose(n_joints=18, n_limbs=19, n_stages=3, feat_ch=128)


def build_openpose_body25() -> nn.Module:
    """Build OpenPose Body25 (25 joints, 26 PAF pairs)."""
    return OpenPose(n_joints=25, n_limbs=26, n_stages=3, feat_ch=128)


def example_input_openpose() -> torch.Tensor:
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "OpenPose-COCO18 (multi-stage PCM+PAF, 18 joints)",
        "build_openpose_coco18",
        "example_input_openpose",
        "2017",
        "DC",
    ),
    (
        "OpenPose-Body25 (multi-stage PCM+PAF, 25 joints)",
        "build_openpose_body25",
        "example_input_openpose",
        "2019",
        "DC",
    ),
]
