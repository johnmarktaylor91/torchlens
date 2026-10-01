"""Time-series / sequence primitives: DLinear and DSS (diagonal state space).

DLinear: "Are Transformers Effective for Time Series Forecasting?"
  Zeng et al., AAAI 2023.  Paper: https://arxiv.org/abs/2205.13504
  Source: https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
  Distinctive primitive: a moving-average **series decomposition** splits the input
  into a trend component (avg-pool over the time axis) and a seasonal residual
  (input - trend); each component is forecast by its OWN single Linear layer mapping
  seq_len -> pred_len, and the two forecasts are summed.  A one-layer-per-component
  linear baseline that beats most Transformers on long-horizon forecasting.

DSS: "Diagonal State Spaces are as Effective as Structured State Spaces"
  Gupta et al., NeurIPS 2022.  Paper: https://arxiv.org/abs/2203.14343
  Source: https://github.com/ag1988/dss
  Distinctive primitive: a sequence layer whose long convolution **kernel** is computed
  analytically as a sum of complex exponentials from a DIAGONAL state matrix
  Lambda = -exp(Lambda_re) + i*Lambda_im.  Per state n the kernel element is
  w_n * Lambda_n^k for time step k; summing over the N diagonal modes yields the
  causal convolution kernel K[k] = Re(sum_n w_n * Lambda_n^k).  The layer applies this
  kernel as a depthwise causal convolution, adds a skip (D*u), then a GLU output gate.
  This reproduces the diagonal-SSM convolution that simplifies S4's DPLR/Cauchy kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DLinear
# ============================================================


class _MovingAvg(nn.Module):
    """Moving-average block: average-pool along time (with edge padding)."""

    def __init__(self, kernel_size: int = 25, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C). Pad the ends so the trend has the same length as the input.
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class _SeriesDecomp(nn.Module):
    """Series decomposition: split into (seasonal residual, trend)."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """DLinear: decomposition + two parallel Linear forecasters (seasonal + trend)."""

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        enc_in: int = 7,
        kernel_size: int = 25,
        individual: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        self.decomp = _SeriesDecomp(kernel_size)
        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)]
            )
            self.linear_trend = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(enc_in)])
        else:
            self.linear_seasonal = nn.Linear(seq_len, pred_len)
            self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        seasonal, trend = self.decomp(x)
        seasonal = seasonal.permute(0, 2, 1)  # (B, C, L)
        trend = trend.permute(0, 2, 1)
        if self.individual:
            s_out = torch.zeros(seasonal.size(0), seasonal.size(1), self.pred_len, dtype=x.dtype)
            t_out = torch.zeros_like(s_out)
            for i in range(self.channels):
                s_out[:, i, :] = self.linear_seasonal[i](seasonal[:, i, :])
                t_out[:, i, :] = self.linear_trend[i](trend[:, i, :])
        else:
            s_out = self.linear_seasonal(seasonal)
            t_out = self.linear_trend(trend)
        out = s_out + t_out
        return out.permute(0, 2, 1)  # (B, pred_len, C)


# ============================================================
# DSS layer (diagonal state space)
# ============================================================


class DSSKernel(nn.Module):
    """Diagonal SSM convolution kernel: K[k] = Re(sum_n w_n * Lambda_n^k).

    Lambda_n = -exp(log_dt) * exp(Lambda_re) + i * Lambda_im  (diagonal, stable).
    One independent diagonal SSM per model channel (depthwise).
    """

    def __init__(self, d_model: int = 64, d_state: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # Diagonal state matrix (real/imag parts), per channel x state.
        self.lambda_re = nn.Parameter(torch.randn(d_model, d_state) * 0.5 - 0.5)
        self.lambda_im = nn.Parameter(torch.randn(d_model, d_state))
        # Output mixing weights w_n (real + imag) and per-channel timescale + skip D.
        self.w_re = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.w_im = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.log_dt = nn.Parameter(torch.rand(d_model) * 0.5 - 3.0)

    def forward(self, length: int) -> torch.Tensor:
        # Discrete-time eigenvalues a_n = exp(dt * Lambda_n), with Lambda stable.
        dt = torch.exp(self.log_dt).unsqueeze(-1)  # (d_model, 1)
        real = -torch.exp(self.lambda_re) * dt  # (d_model, d_state), <0 -> decay
        imag = self.lambda_im * dt
        k = torch.arange(length, dtype=real.dtype).view(1, 1, length)  # (1,1,L)
        # Lambda_n^k = exp(real*k) * (cos(imag*k) + i sin(imag*k))
        decay = torch.exp(real.unsqueeze(-1) * k)  # (d_model, d_state, L)
        phase = imag.unsqueeze(-1) * k
        cos = torch.cos(phase)
        sin = torch.sin(phase)
        # Re(w * Lambda^k) = decay * (w_re*cos - w_im*sin)
        contrib = decay * (self.w_re.unsqueeze(-1) * cos - self.w_im.unsqueeze(-1) * sin)
        kernel = contrib.sum(dim=1)  # (d_model, L)
        return kernel


class DSS(nn.Module):
    """DSS sequence layer: diagonal-SSM causal conv + skip + GLU output gate."""

    def __init__(self, d_model: int = 64, d_state: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.kernel = DSSKernel(d_model, d_state)
        self.skip_d = nn.Parameter(torch.randn(d_model))
        self.out_gate = nn.Linear(d_model, 2 * d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, L, d_model)
        residual = u
        u = self.norm(u)
        b, length, d = u.shape
        k = self.kernel(length)  # (d_model, L)
        # Causal depthwise convolution of u by kernel k along time.
        u_t = u.transpose(1, 2)  # (B, d_model, L)
        # pad left by L-1 for causal conv
        u_pad = F.pad(u_t, (length - 1, 0))
        kernel = k.flip(-1).unsqueeze(1)  # (d_model, 1, L)
        y = F.conv1d(u_pad, kernel, groups=d)  # (B, d_model, L)
        y = y + u_t * self.skip_d.view(1, -1, 1)
        y = y.transpose(1, 2)  # (B, L, d_model)
        gate = self.out_gate(y)
        a, b2 = gate.chunk(2, dim=-1)
        y = a * torch.sigmoid(b2)  # GLU
        return residual + y


# ============================================================
# Menagerie wiring
# ============================================================


def build_dlinear() -> nn.Module:
    """Build DLinear (decomposition + dual-Linear long-horizon forecaster)."""
    return DLinear(seq_len=96, pred_len=24, enc_in=7, individual=False).eval()


def example_input_dlinear() -> torch.Tensor:
    """Example multivariate series ``(1, 96, 7)`` (B, seq_len, channels)."""
    return torch.randn(1, 96, 7)


def build_dss_layer() -> nn.Module:
    """Build a single DSS (diagonal state space) sequence layer."""
    return DSS(d_model=64, d_state=64).eval()


def example_input_dss() -> torch.Tensor:
    """Example sequence ``(1, 128, 64)`` (B, length, d_model)."""
    return torch.randn(1, 128, 64)


MENAGERIE_ENTRIES = [
    (
        "DLinear (series-decomposition + dual-Linear forecaster)",
        "build_dlinear",
        "example_input_dlinear",
        "2023",
        "DC",
    ),
    (
        "DSS layer (diagonal state-space sum-of-exponentials kernel)",
        "build_dss_layer",
        "example_input_dss",
        "2022",
        "DC",
    ),
]
