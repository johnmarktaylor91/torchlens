"""Selective-IGEV / Selective-Stereo: iterative GRU stereo with Selective Recurrent Unit.

Xu et al., CVPR 2024.
Paper: https://arxiv.org/abs/2401.08560
Source: https://github.com/Windsrain/Selective-Stereo

Selective-IGEV is in the RAFT-Stereo / IGEV-Stereo family: iterative GRU-based
stereo depth estimation. The key innovation is the Selective Recurrent Unit (SRU):
instead of a single GRU hidden state, multiple GRU branches at different frequencies
or feature resolutions process the cost-volume context independently, then a
Contextual Spatial Attention (CSA) module fuses them via learned gating — selecting
which branch's output to trust at each spatial location. This multi-frequency hidden
state prevents mode collapse and preserves fine structure at fine-grained disparity
boundaries.

Architecture:
  1. Feature Extractor: shared CNN -> (left, right) feature maps.
  2. Correlation / Cost Volume: build 1D correlation volume between left and right
     features (IGEV-style: all-pairs correlation along the disparity axis).
  3. Context Encoder: extracts context features from the left image only.
  4. GRU Iterations: iteratively update a disparity estimate using the SRU.
     Each SRU step: two GRU sub-cells (low-freq, high-freq) + CSA gated fusion.
  5. Disparity head: linear projection from hidden state -> disparity delta.

Compact faithfulness:
  - Small feature net (2 stride-2 convs), dim=32.
  - Cost volume: 1D correlation at D=8 disparity candidates, stride=1.
  - 2 SRU GRU iterations.
  - Input: stacked (left, right) as (1, 6, H, W); left=[:3], right=[3:].
  - Output: final disparity map (1, H, W).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


class FeatureExtractor(nn.Module):
    """Lightweight CNN: (B, 3, H, W) -> (B, C, H/4, W/4)."""

    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, out_channels // 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# 1D Correlation volume (along disparity axis)
# ---------------------------------------------------------------------------


def build_correlation_volume(
    left: torch.Tensor, right: torch.Tensor, max_disp: int = 8
) -> torch.Tensor:
    """Build 1D correlation volume.

    Args:
        left: (B, C, H, W)
        right: (B, C, H, W)
        max_disp: number of disparity candidates D.

    Returns:
        corr: (B, D, H, W)
    """
    B, C, H, W = left.shape
    corr_list = []
    for d in range(max_disp):
        if d == 0:
            corr_d = (left * right).sum(1, keepdim=True)  # (B, 1, H, W)
        else:
            # Shift right image leftward by d pixels
            right_shifted = F.pad(right[:, :, :, d:], (0, d))
            corr_d = (left * right_shifted).sum(1, keepdim=True)
        corr_list.append(corr_d)
    return torch.cat(corr_list, dim=1)  # (B, D, H, W)


# ---------------------------------------------------------------------------
# Contextual Spatial Attention (CSA) — the selective gating
# ---------------------------------------------------------------------------


class ContextualSpatialAttention(nn.Module):
    """CSA: learns per-spatial-location gates to select/blend two GRU branches.

    Takes the two hidden states h_lo (low-freq) and h_hi (high-freq) and
    outputs a fused hidden state via learned spatial attention weights.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 1),  # 2-way gate
        )

    def forward(self, h_lo: torch.Tensor, h_hi: torch.Tensor) -> torch.Tensor:
        # h_lo, h_hi: (B, C, H, W)
        combined = torch.cat([h_lo, h_hi], dim=1)
        gates = self.gate(combined).softmax(dim=1)  # (B, 2, H, W)
        g_lo = gates[:, 0:1]
        g_hi = gates[:, 1:2]
        return g_lo * h_lo + g_hi * h_hi


# ---------------------------------------------------------------------------
# Selective Recurrent Unit (SRU) — core innovation
# ---------------------------------------------------------------------------


class SelectiveRecurrentUnit(nn.Module):
    """SRU: two ConvGRU branches + CSA fusion.

    Low-frequency branch: sees downsampled context (coarse).
    High-frequency branch: sees full-resolution context (fine).
    CSA selects/blends them per spatial location.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # Low-frequency GRU sub-cell
        self.gru_lo = nn.GRUCell(input_dim, hidden_dim)
        # High-frequency GRU sub-cell
        self.gru_hi = nn.GRUCell(input_dim, hidden_dim)
        self.csa = ContextualSpatialAttention(hidden_dim)

    def forward(
        self,
        inp: torch.Tensor,  # (B*H*W, input_dim) — flattened spatial input
        h: torch.Tensor,  # (B*H*W, hidden_dim) — current hidden state
        h_shape: tuple,  # (B, hidden_dim, H, W) for CSA reshape
    ) -> torch.Tensor:
        B, C, H, W = h_shape

        # Low-frequency branch: small perturbation via gru_lo
        h_lo = self.gru_lo(inp, h)  # (B*H*W, hidden_dim)
        # High-frequency branch: gru_hi
        h_hi = self.gru_hi(inp, h)  # (B*H*W, hidden_dim)

        # Reshape to spatial for CSA gating
        h_lo_2d = h_lo.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        h_hi_2d = h_hi.view(B, H, W, C).permute(0, 3, 1, 2)

        fused_2d = self.csa(h_lo_2d, h_hi_2d)  # (B, C, H, W)
        # Flatten back
        fused = fused_2d.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return fused


# ---------------------------------------------------------------------------
# Motion encoder (cost volume + context -> GRU input features)
# ---------------------------------------------------------------------------


class MotionEncoder(nn.Module):
    """Encodes correlation volume + current disparity estimate -> GRU input."""

    def __init__(self, corr_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.corr_enc = nn.Sequential(
            nn.Conv2d(corr_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
        )
        self.disp_enc = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        out_dim = hidden_dim + hidden_dim // 4
        self.out = nn.Conv2d(out_dim, hidden_dim, 3, padding=1)

    def forward(self, corr: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        # corr: (B, D, H, W); disp: (B, 1, H, W)
        x_corr = self.corr_enc(corr)
        x_disp = self.disp_enc(disp)
        x = torch.cat([x_corr, x_disp], dim=1)
        return self.out(x)  # (B, hidden_dim, H, W)


# ---------------------------------------------------------------------------
# Full Selective-IGEV model
# ---------------------------------------------------------------------------


class SelectiveIGEV(nn.Module):
    """Selective-IGEV stereo network (compact random-init reimplementation).

    Input: (B, 6, H, W) — left image in [:3], right image in [3:].
    Output: (B, H, W) — disparity map.
    """

    def __init__(
        self,
        feat_dim: int = 32,
        hidden_dim: int = 32,
        max_disp: int = 8,
        num_iters: int = 2,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.max_disp = max_disp
        self.num_iters = num_iters

        # Feature + context extractors
        self.feature_net = FeatureExtractor(feat_dim)
        self.context_net = FeatureExtractor(hidden_dim)

        # Motion encoder: correlation + current disp -> GRU input
        self.motion_enc = MotionEncoder(max_disp, hidden_dim)

        # SRU
        self.sru = SelectiveRecurrentUnit(hidden_dim, hidden_dim)

        # Disparity update head
        self.disp_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left_img = x[:, :3]
        right_img = x[:, 3:]
        B, _, H, W = left_img.shape

        # Extract features (1/4 resolution)
        left_feat = self.feature_net(left_img)  # (B, feat_dim, H4, W4)
        right_feat = self.feature_net(right_img)

        _, _, H4, W4 = left_feat.shape

        # Context features = initial hidden state
        context = self.context_net(left_img)  # (B, hidden_dim, H4, W4)

        # Build correlation volume
        corr_vol = build_correlation_volume(left_feat, right_feat, self.max_disp)  # (B, D, H4, W4)

        # Initialize disparity
        disp = torch.zeros(B, 1, H4, W4, device=x.device)

        # Initial hidden state from context
        h = context.permute(0, 2, 3, 1).reshape(B * H4 * W4, self.hidden_dim)

        # Iterative SRU updates
        for _ in range(self.num_iters):
            # Motion features
            inp_2d = self.motion_enc(corr_vol, disp)  # (B, hidden_dim, H4, W4)
            inp_flat = inp_2d.permute(0, 2, 3, 1).reshape(B * H4 * W4, self.hidden_dim)

            # SRU step
            h = self.sru(inp_flat, h, (B, self.hidden_dim, H4, W4))

            # Predict disparity delta
            h_2d = h.view(B, H4, W4, self.hidden_dim).permute(0, 3, 1, 2)
            delta_disp = self.disp_head(h_2d)  # (B, 1, H4, W4)
            disp = disp + delta_disp

        # Upsample to full resolution
        disp_full = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=False)
        return disp_full.squeeze(1)  # (B, H, W)


# ---------------------------------------------------------------------------
# Builders and menagerie wiring
# ---------------------------------------------------------------------------


def build_selective_stereo() -> nn.Module:
    """Build Selective-IGEV / Selective-Stereo (compact, feat_dim=32, 2 iters)."""
    return SelectiveIGEV(feat_dim=32, hidden_dim=32, max_disp=8, num_iters=2)


def example_input() -> torch.Tensor:
    """Stacked left+right image pair: (1, 6, 64, 64)."""
    return torch.randn(1, 6, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Selective-IGEV (Selective Recurrent Unit stereo, CSA gated GRU fusion)",
        "build_selective_stereo",
        "example_input",
        "2024",
        "DC",
    ),
]
