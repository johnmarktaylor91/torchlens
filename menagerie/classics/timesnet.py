"""TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.

Wu et al., ICLR 2023.
Paper: https://arxiv.org/abs/2210.02186
Source: https://github.com/thuml/TimesNet

TimesNet discovers multiple periodicities in a 1-D time series via FFT,
then reshapes each period's 1-D sequence into a 2-D (period, frequency)
tensor and applies an Inception-style 2-D conv block (TimesBlock) that
mixes temporal patterns across time and frequency.  The block outputs are
aggregated by amplitude-weighted summation back to the original 1-D shape.

Distinctive primitives shown here:
  1. FFT period-finding: top-k periodicities by amplitude.
  2. 1-D -> 2-D reshape per period (pad + fold into (B, C, period, freq)).
  3. Inception-style 2-D conv: three parallel conv branches (kernels 1, 3, 5)
     whose outputs are summed.
  4. Amplitude-weighted aggregation: softmax over amplitudes, weighted sum
     of per-period outputs.

Simplified for compact graph: 1 TimesBlock, top_k=2 periods, small C/d_model.
Input: (1, L, C) multivariate series, L=32 steps, C=7 vars.
Output: (1, horizon, C) forecast.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock2D(nn.Module):
    """Three-branch Inception-style 2-D conv mixing temporal patterns."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, period, freq)
        y = self.conv1(x) + self.conv3(x) + self.conv5(x)
        return F.gelu(self.norm(y))


class TimesBlock(nn.Module):
    """One TimesBlock: FFT period-find -> 2-D reshape -> Inception 2-D conv -> aggregate."""

    def __init__(self, seq_len: int, d_model: int, top_k: int = 2) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.inception = InceptionBlock2D(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) where T = seq_len + horizon (or just seq_len)
        B, T, D = x.shape

        # --- 1. FFT period-finding (over time dim, averaged across B and D) ---
        # Use rfft over the time dim
        xf = torch.fft.rfft(x, dim=1)  # (B, T//2+1, D)
        amps = xf.abs().mean(dim=(0, 2))  # (T//2+1,) mean amplitude per freq
        # Top-k frequencies (skip DC at index 0)
        freq_list = amps[1:]
        _, top_idx = torch.topk(freq_list, min(self.top_k, len(freq_list)))
        top_idx = top_idx + 1  # shift back (skip DC)

        # --- 2. For each top-k period: reshape -> 2-D Inception -> reshape back ---
        outs = []
        amp_weights = []
        for i in range(top_idx.shape[0]):
            freq = top_idx[i].item()
            period = max(1, round(T / max(freq, 1)))
            # Pad T to next multiple of period
            pad_len = (period - T % period) % period
            xp = F.pad(x, (0, 0, 0, pad_len))  # (B, T+pad, D)
            T2 = T + pad_len
            n_periods = T2 // period
            # Reshape to 2-D: (B, D, period, n_periods)
            x2d = xp.reshape(B, n_periods, period, D).permute(
                0, 3, 2, 1
            )  # (B, D, period, n_periods)
            # Inception 2-D conv
            y2d = self.inception(x2d)  # (B, D, period, n_periods)
            # Reshape back to 1-D: (B, T+pad, D)
            y1d = y2d.permute(0, 3, 2, 1).reshape(B, T2, D)
            outs.append(y1d[:, :T, :])
            amp_weights.append(amps[top_idx[i]])

        # --- 3. Amplitude-weighted aggregation ---
        amp_weights_t = torch.stack(amp_weights)  # (k,)
        amp_weights_t = torch.softmax(amp_weights_t, dim=0)  # (k,)
        out = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        for i, o in enumerate(outs):
            out = out + amp_weights_t[i] * o

        # Residual + norm
        return self.norm(x + out)


class TimesNet(nn.Module):
    """TimesNet for multivariate time-series forecasting (random-init reimpl).

    Compact: 1 TimesBlock, top_k=2, small d_model.
    Input: (B, seq_len, n_vars). Output: (B, pred_len, n_vars).
    """

    def __init__(
        self,
        seq_len: int = 32,
        pred_len: int = 8,
        n_vars: int = 7,
        d_model: int = 16,
        top_k: int = 2,
        n_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model

        self.input_proj = nn.Linear(n_vars, d_model)
        self.blocks = nn.ModuleList(
            [TimesBlock(seq_len + pred_len, d_model, top_k) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(d_model, n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_vars)
        B, L, C = x.shape
        # Embed
        y = self.input_proj(x)  # (B, L, d_model)
        # Pad future horizon with zeros
        padding = torch.zeros(B, self.pred_len, self.d_model, device=x.device, dtype=x.dtype)
        y = torch.cat([y, padding], dim=1)  # (B, L+pred_len, d_model)
        # Apply blocks
        for blk in self.blocks:
            y = blk(y)
        # Project back to vars, take forecast slice
        y = self.output_proj(y)  # (B, L+pred_len, n_vars)
        return y[:, -self.pred_len :, :]  # (B, pred_len, n_vars)


def build_timesnet() -> nn.Module:
    return TimesNet(seq_len=32, pred_len=8, n_vars=7, d_model=16, top_k=2, n_blocks=1)


def example_input() -> torch.Tensor:
    """Multivariate series (1, 32, 7): batch=1, L=32 steps, C=7 vars."""
    return torch.randn(1, 32, 7)


MENAGERIE_ENTRIES = [
    (
        "TimesNet (temporal 2D-variation modeling, FFT period-find + Inception 2D conv)",
        "build_timesnet",
        "example_input",
        "2023",
        "DC",
    ),
]
