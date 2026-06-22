"""iTransformer: Inverted Transformer for multivariate time-series forecasting.

Liu et al. (2024), "iTransformer: Inverted Transformers Are Effective for
Time Series Forecasting".  ICLR 2024.  arXiv:2310.06625.
Source: https://github.com/thuml/iTransformer

Distinctive primitive: the INVERTED attention mechanism.
  - Standard transformer: each TIME-STEP is a token -> attention across time.
  - iTransformer:    each VARIATE (channel) is a token -> attention across variates.
    Input (B, T, N) is transposed to (B, N, T); each variate's time series is
    embedded as a single D-dim token via a linear projection; then standard
    multi-head self-attention operates ACROSS the N variates.
    Feed-forward operates on variate tokens (B, N, D).
    Final projection maps D -> T for forecasting.

This "variate-as-token" inversion is the hallmark: the model captures inter-variate
correlations rather than temporal patterns directly.

Compact config: N=8 variates, T_in=24 lookback, T_out=12 forecast, d_model=32, 2 heads.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariateAttention(nn.Module):
    """Multi-head self-attention over variates (B, N, d_model)."""

    def __init__(self, d_model: int = 32, n_heads: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, d = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, N, d_head)
        Q, K, V = qkv.unbind(0)  # each (B, h, N, d_head)
        scale = math.sqrt(self.d_head)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / scale, dim=-1)
        attn = self.drop(attn)
        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, N, d)
        return self.out(out)


class iTransformerLayer(nn.Module):
    """One iTransformer encoder layer: variate attention + FFN."""

    def __init__(self, d_model: int = 32, n_heads: int = 2, d_ff: int = 64) -> None:
        super().__init__()
        self.attn = VariateAttention(d_model, n_heads)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        x = x + self.attn(self.norm1(x))
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x


class ITransformer(nn.Module):
    """Compact iTransformer: embed each variate's time series -> attend -> forecast.

    Input (B, T_in, N) -> transpose to (B, N, T_in) -> embed to (B, N, d_model)
    -> L iTransformerLayers -> project to (B, N, T_out) -> transpose to (B, T_out, N).
    Wrapped to return (B, T_out, N) as a single tensor.
    """

    def __init__(
        self,
        n_variates: int = 8,
        t_in: int = 24,
        t_out: int = 12,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 64,
    ) -> None:
        super().__init__()
        # Variate embedding: project T_in -> d_model per variate
        self.embed = nn.Linear(t_in, d_model)
        self.layers = nn.ModuleList(
            [iTransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        # Forecast projection: d_model -> T_out
        self.fc_out = nn.Linear(d_model, t_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_in, N) -> (B, T_out, N)"""
        # Inversion: transpose so variates are tokens
        x = x.permute(0, 2, 1)  # (B, N, T_in)
        # Instance normalisation (simplified: just subtract mean per variate)
        mean = x.mean(dim=-1, keepdim=True)
        x = x - mean
        # Embed: each variate's T_in -> d_model
        x = self.embed(x)  # (B, N, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # Forecast
        x = self.fc_out(x)  # (B, N, T_out)
        x = x.permute(0, 2, 1)  # (B, T_out, N)
        return x


def build_itransformer() -> nn.Module:
    return ITransformer(n_variates=8, t_in=24, t_out=12, d_model=32, n_heads=2, n_layers=2).eval()


def example_input() -> torch.Tensor:
    """(1, 24, 8) -- batch=1, T_in=24, N=8 variates."""
    return torch.randn(1, 24, 8)


MENAGERIE_ENTRIES = [
    (
        "iTransformer (inverted transformer: variate-as-token attention for time series)",
        "build_itransformer",
        "example_input",
        "2024",
        "DC",
    ),
]
