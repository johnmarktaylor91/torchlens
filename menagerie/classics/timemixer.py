"""TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting.

Wang et al., ICLR 2024.
Paper: https://arxiv.org/abs/2405.14616
Source: https://github.com/kwuking/TimeMixer

TimeMixer decomposes a multivariate series into seasonal and trend components
at multiple downsampled scales via a moving-average (avg-pool) decomposition,
then mixes information across scales via two directions:

  1. Past-Decomposable-Mixing (PDM):
     - Bottom-up seasonal mixing: finer-scale seasonal -> coarser-scale
       (mixing detail into coarse, bottom-up direction).
     - Top-down trend mixing: coarser-scale trend -> finer-scale
       (mixing macro trend into fine, top-down direction).

  2. Future-Multipredictor-Mixing (FMM):
     - Aggregates multi-scale features via learned MLPs to produce the forecast.

Distinctive primitives shown:
  - Moving-average decomposition at multiple scales (avg_pool1d).
  - Multi-scale seasonal stack + bottom-up MLP mixing.
  - Multi-scale trend stack + top-down MLP mixing.
  - Multi-predictor MLP head aggregation.

Compact: 2 scales, small d_model, seq_len=32, pred_len=8, C=7 vars.
Input: (1, L, C) multivariate series. Output: (1, pred_len, C).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvgDecomp(nn.Module):
    """Moving-average seasonal/trend decomposition via avg_pool1d."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C)
        B, T, C = x.shape
        # Pad both ends for "same" output length
        pad = self.kernel_size // 2
        x_t = x.transpose(1, 2)  # (B, C, T)
        x_pad = F.pad(x_t, (pad, pad), mode="replicate")
        trend = self.pool(x_pad).transpose(1, 2)  # (B, T, C)
        # Ensure exact length match
        trend = trend[:, :T, :]
        seasonal = x - trend
        return seasonal, trend


class MixingLayer(nn.Module):
    """Simple MLP for cross-scale mixing of a sequence of tokens."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


class PDMBlock(nn.Module):
    """Past-Decomposable-Mixing block for one scale pair.

    bottom_up_seasonal: fine scale -> coarse scale via MLP.
    top_down_trend: coarse scale -> fine scale via MLP.
    """

    def __init__(self, fine_len: int, coarse_len: int, d_model: int) -> None:
        super().__init__()
        self.bottom_up = MixingLayer(fine_len, coarse_len)  # seasonal: fine -> coarse
        self.top_down = MixingLayer(coarse_len, fine_len)  # trend: coarse -> fine

    def forward(
        self,
        seasonal_fine: torch.Tensor,
        trend_fine: torch.Tensor,
        seasonal_coarse: torch.Tensor,
        trend_coarse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # seasonal_fine: (B, fine_len, d)
        # Bottom-up: project fine seasonal into coarse space and add
        seasonal_fine_t = seasonal_fine.transpose(1, 2)  # (B, d, fine_len)
        mixed_coarse_t = self.bottom_up(seasonal_fine_t)  # (B, d, coarse_len)
        seasonal_coarse = seasonal_coarse + mixed_coarse_t.transpose(1, 2)

        # Top-down: project coarse trend into fine space and add
        trend_coarse_t = trend_coarse.transpose(1, 2)  # (B, d, coarse_len)
        mixed_fine_t = self.top_down(trend_coarse_t)  # (B, d, fine_len)
        trend_fine = trend_fine + mixed_fine_t.transpose(1, 2)

        return seasonal_fine, trend_fine, seasonal_coarse, trend_coarse


class TimeMixer(nn.Module):
    """TimeMixer multiscale mixing forecaster (random-init reimpl).

    Two scales: original (seq_len) and downsampled (seq_len // 2).
    Compact: 1 PDM layer, small d_model.
    Input: (B, seq_len, n_vars). Output: (B, pred_len, n_vars).
    """

    def __init__(
        self,
        seq_len: int = 32,
        pred_len: int = 8,
        n_vars: int = 7,
        d_model: int = 16,
        down_factor: int = 2,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model
        coarse_len = seq_len // down_factor

        # Input embedding
        self.input_proj = nn.Linear(n_vars, d_model)

        # Moving-average decomp at both scales
        self.decomp_fine = MovingAvgDecomp(kernel_size=kernel_size)
        self.decomp_coarse = MovingAvgDecomp(kernel_size=kernel_size)

        # Downsampling for coarse scale
        self.downsample = nn.AvgPool1d(kernel_size=down_factor, stride=down_factor)

        # PDM mixing
        self.pdm = PDMBlock(seq_len, coarse_len, d_model)

        # Forecast heads (one per scale, multi-predictor mixing)
        self.head_fine = nn.Linear(seq_len, pred_len)
        self.head_coarse = nn.Linear(coarse_len, pred_len)
        self.mix_weights = nn.Linear(2, 1)  # combine two scale predictions

        # Output projection
        self.output_proj = nn.Linear(d_model, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_vars)
        B, L, C = x.shape

        # Embed input
        y = self.input_proj(x)  # (B, L, d_model)

        # ---- Multiscale decomposition ----
        # Fine scale
        s_fine, t_fine = self.decomp_fine(y)  # (B, L, d_model) each

        # Coarse scale: downsample then decompose
        y_t = y.transpose(1, 2)  # (B, d_model, L)
        y_coarse = self.downsample(y_t).transpose(1, 2)  # (B, L//2, d_model)
        s_coarse, t_coarse = self.decomp_coarse(y_coarse)

        # ---- PDM: cross-scale mixing ----
        s_fine, t_fine, s_coarse, t_coarse = self.pdm(s_fine, t_fine, s_coarse, t_coarse)

        # ---- FMM: multi-predictor mixing ----
        # Each scale: combine seasonal + trend, project to pred_len
        fine_feat = (s_fine + t_fine).transpose(1, 2)  # (B, d_model, L)
        coarse_feat = (s_coarse + t_coarse).transpose(1, 2)  # (B, d_model, L//2)

        pred_fine = self.head_fine(fine_feat).transpose(1, 2)  # (B, pred_len, d_model)
        pred_coarse = self.head_coarse(coarse_feat).transpose(1, 2)  # (B, pred_len, d_model)

        # Combine predictions from two scales
        combined = torch.stack([pred_fine, pred_coarse], dim=-1)  # (B, pred_len, d_model, 2)
        out = self.mix_weights(combined).squeeze(-1)  # (B, pred_len, d_model)

        return self.output_proj(out)  # (B, pred_len, n_vars)


def build_timemixer() -> nn.Module:
    return TimeMixer(seq_len=32, pred_len=8, n_vars=7, d_model=16, down_factor=2)


def example_input() -> torch.Tensor:
    """Multivariate series (1, 32, 7): batch=1, L=32 steps, C=7 vars."""
    return torch.randn(1, 32, 7)


MENAGERIE_ENTRIES = [
    (
        "TimeMixer (multiscale decomposable mixing, seasonal/trend + cross-scale PDM + FMM)",
        "build_timemixer",
        "example_input",
        "2024",
        "DC",
    ),
]
