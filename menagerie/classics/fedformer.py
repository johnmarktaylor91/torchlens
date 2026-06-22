"""FEDformer: Frequency Enhanced Decomposed Transformer for time-series.

Zhou et al. (2022), "FEDformer: Frequency Enhanced Decomposed Transformer for
Long-term Series Forecasting".  ICML 2022.  arXiv:2201.12740.
Source: https://github.com/MAZiqing/FEDformer

Distinctive primitives:
  1. SERIES DECOMPOSITION: moving-average trend + residual seasonal (from Autoformer).
     Each block decomposes the series: trend = moving_avg(x), seasonal = x - trend.
  2. FREQUENCY ENHANCED BLOCK (FEB-f): Apply FFT to the seasonal component, keep a
     RANDOM SUBSET of modes (random Fourier modes selection), apply a complex-valued
     linear transform in frequency domain, iFFT back.  This replaces the standard
     attention mechanism with a frequency-domain mixing.
  3. Two decomposition sub-layers per encoder layer: one FEB-f attention path +
     one FeedForward path, each followed by decomposition + residual.

Compact config: d_model=32, seq_len=24, modes=8 (random modes selected from FFT).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    """Moving-average decomposition: trend (smoothed) + seasonal (residual)."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor):
        """x: (B, L, d) -> (seasonal, trend) both (B, L, d)"""
        # AvgPool1d operates on (B, d, L)
        xp = x.permute(0, 2, 1)  # (B, d, L)
        trend = self.avg(xp)
        # Trim if padding caused length mismatch
        L = x.shape[1]
        trend = trend[:, :, :L].permute(0, 2, 1)  # (B, L, d)
        seasonal = x - trend
        return seasonal, trend


class FEBf(nn.Module):
    """Frequency Enhanced Block (mode-f): random-modes FFT + complex linear + iFFT.

    Steps:
      1. FFT along time axis: (B, L, d) -> (B, L//2+1, d) complex
      2. Select `n_modes` random frequency indices
      3. Complex linear mixing: W_re, W_im applied to real/imag parts
      4. Place back in frequency bins, zero the rest
      5. iFFT -> (B, L, d) real output
    """

    def __init__(self, d_model: int = 32, seq_len: int = 24, n_modes: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_modes = min(n_modes, seq_len // 2 + 1)
        # Fixed random mode indices (registered buffer so they move with device)
        n_freq = seq_len // 2 + 1
        perm = torch.randperm(n_freq)[: self.n_modes].sort().values
        self.register_buffer("mode_idx", perm)
        # Complex linear: separate real/imag weights (d_model, d_model)
        self.W_re = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.W_im = nn.Parameter(torch.randn(d_model, d_model) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d) -> (B, L, d)"""
        B, L, d = x.shape
        # FFT along time
        xf = torch.fft.rfft(x, dim=1)  # (B, n_freq, d), complex
        n_freq = xf.shape[1]
        # Build output frequency tensor (start all zeros)
        out_f = torch.zeros_like(xf)
        # Apply complex linear at selected modes
        idx = self.mode_idx  # (n_modes,) - buffer, not a param
        xf_sel = xf[:, idx, :]  # (B, n_modes, d) complex
        # Complex multiply: (a+ib)(c+id) = (ac-bd) + i(ad+bc)
        re = xf_sel.real @ self.W_re - xf_sel.imag @ self.W_im  # (B, n_modes, d)
        im = xf_sel.real @ self.W_im + xf_sel.imag @ self.W_re
        out_f[:, idx, :] = torch.complex(re, im)
        # iFFT back
        out = torch.fft.irfft(out_f, n=L, dim=1)  # (B, L, d)
        return out


class FEDformerEncoderLayer(nn.Module):
    """One FEDformer encoder layer: FEB-f + FFN, each followed by decomposition."""

    def __init__(
        self,
        d_model: int = 32,
        seq_len: int = 24,
        n_modes: int = 8,
        d_ff: int = 64,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.feb = FEBf(d_model, seq_len, n_modes)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.decomp1 = SeriesDecomposition(kernel_size)
        self.decomp2 = SeriesDecomposition(kernel_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """x: (B, L, d) -> seasonal (B, L, d)  [trend accumulated externally]"""
        # FEB-f path
        y = self.feb(self.norm1(x))
        x = x + y
        seasonal, _ = self.decomp1(x)
        # FFN path
        y2 = self.ff2(F.gelu(self.ff1(self.norm2(seasonal))))
        seasonal = seasonal + y2
        seasonal, _ = self.decomp2(seasonal)
        return seasonal


class FEDformer(nn.Module):
    """Compact FEDformer: series decomp + frequency-enhanced encoder + forecast head."""

    def __init__(
        self,
        seq_len: int = 24,
        pred_len: int = 12,
        d_model: int = 32,
        n_modes: int = 8,
        n_layers: int = 2,
        d_ff: int = 64,
        n_features: int = 1,
    ) -> None:
        super().__init__()
        self.enc_embed = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList(
            [FEDformerEncoderLayer(d_model, seq_len, n_modes, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, n_features)
        # Forecast: project seq_len -> pred_len (simple learned projection)
        self.forecast_fc = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, n_features) -> (B, pred_len, n_features)"""
        x = self.enc_embed(x)  # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.proj(x)  # (B, L, n_features)
        # Forecast: (B, L, n_features) -> (B, pred_len, n_features)
        x = self.forecast_fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


def build_fedformer() -> nn.Module:
    return FEDformer(seq_len=24, pred_len=12, d_model=32, n_modes=8, n_layers=2).eval()


def example_input() -> torch.Tensor:
    """(1, 24, 1) -- batch=1, seq_len=24, n_features=1."""
    return torch.randn(1, 24, 1)


MENAGERIE_ENTRIES = [
    (
        "FEDformer (Frequency Enhanced Decomposed transformer: FFT random-mode mixing + series decomposition)",
        "build_fedformer",
        "example_input",
        "2022",
        "DC",
    ),
]
