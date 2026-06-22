"""PSE+TAE: Pixel-Set Encoder + Temporal Attention Encoder for Satellite Image Time Series.

Garnot et al., 2020 (CVPR).
Paper: https://arxiv.org/abs/1911.07757
Source: https://github.com/VSainteuf/pytorch-psetae

Architecture:
  For each satellite parcel (agricultural field), a TIME SERIES of satellite images
  is available. Each date provides a SET of pixels within the parcel's boundary.

  PSE (Pixel-Set Encoder):
    Input: (B, T, S, C)  -- T dates, S pixels per parcel (set), C spectral channels.
    For each date independently:
      1. Shared MLP over each pixel (C -> mlp_dims).
      2. Permutation-invariant pooling: concatenate [mean, std] across the S pixels.
      3. Optional extra MLP after pooling.
    Output: (B, T, D_pse)  -- per-date spatial embedding.

  TAE (Temporal Attention Encoder):
    Input: (B, T, D_pse + extra_date_info)  -- date-stamped sequence.
    Multi-head self-attention over the time dimension with positional encodings
    derived from the DOY (day-of-year) as sinusoidal features.
    Output: (B, D_tae)  -- aggregated temporal representation via master query.

  Classifier head: FC -> n_classes.

Faithful compact simplification:
  S=16 pixels, T=12 dates, C=10 spectral channels.
  PSE MLP: [10->32->64], D_pse=128.
  TAE: d_model=128, 2 heads, 1 layer.
  DOY positional encoding: 24-dim sinusoidal (paper: 128).
  Trace+draw verified 2026-06-21.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: sinusoidal DOY encoding
# ---------------------------------------------------------------------------


def _doy_encode(doy: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal positional encoding from Day-of-Year.

    Args:
        doy:     (B, T) integer days [1..365]
        d_model: encoding dimension (must be even)
    Returns:
        (B, T, d_model)
    """
    B, T = doy.shape
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=doy.device)
        * -(math.log(10000.0) / d_model)
    )
    doy_f = doy.float().unsqueeze(-1)  # (B, T, 1)
    pe = torch.zeros(B, T, d_model, device=doy.device)
    pe[:, :, 0::2] = torch.sin(doy_f * div_term)
    pe[:, :, 1::2] = torch.cos(doy_f * div_term)
    return pe


# ---------------------------------------------------------------------------
# PSE: Pixel-Set Encoder
# ---------------------------------------------------------------------------


class PixelSetEncoder(nn.Module):
    """PSE: Shared MLP over pixel set + mean/std pooling.

    Input:  (B, T, S, in_ch)  -- batch, time, pixels, spectral channels
    Output: (B, T, out_dim)
    """

    def __init__(
        self,
        in_ch: int = 10,
        mlp_dims: list = None,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [32, 64]

        # Shared pixel-wise MLP (applied to each pixel independently)
        layers = []
        d = in_ch
        for h in mlp_dims:
            layers.extend([nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)])
            d = h
        self.pixel_mlp = nn.Sequential(*layers)
        self.pixel_dim = d

        # Post-pooling MLP (input = mean+std = 2*d)
        self.pool_mlp = nn.Sequential(
            nn.Linear(2 * d, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, S, C)
        B, T, S, C = x.shape

        # Apply shared MLP to each pixel: reshape to (B*T*S, C)
        xr = x.reshape(B * T * S, C)
        # BatchNorm1d requires (N, C) -- OK here
        h = self.pixel_mlp(xr)  # (B*T*S, d)
        h = h.reshape(B, T, S, -1)  # (B, T, S, d)

        # Permutation-invariant pooling: mean + std
        mean = h.mean(dim=2)  # (B, T, d)
        std = h.std(dim=2).clamp(min=1e-5)  # (B, T, d)
        pooled = torch.cat([mean, std], dim=-1)  # (B, T, 2*d)

        # Post-pooling MLP: reshape to (B*T, 2*d)
        pr = pooled.reshape(B * T, -1)
        out = self.pool_mlp(pr)  # (B*T, out_dim)
        return out.reshape(B, T, self.out_dim)


# ---------------------------------------------------------------------------
# TAE: Temporal Attention Encoder
# ---------------------------------------------------------------------------


class TemporalAttentionEncoder(nn.Module):
    """TAE: Multi-head self-attention over time with DOY positional encoding.

    Uses a learned 'master query' to aggregate the temporal sequence into a
    single embedding (similar to CLS token pooling).

    Input:  (B, T, d_in)  features + (B, T) doy
    Output: (B, d_model)
    """

    def __init__(
        self,
        d_in: int = 128,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 1,
        doy_enc_dim: int = 24,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.doy_enc_dim = doy_enc_dim

        # Input projection: PSE features + DOY encoding -> d_model
        self.in_proj = nn.Linear(d_in + doy_enc_dim, d_model)

        # Transformer encoder layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Master query (learnable): aggregates temporal sequence
        self.master_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn_pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in);  doy: (B, T)
        B, T, _ = x.shape

        # Add DOY positional encoding
        pos = _doy_encode(doy, self.doy_enc_dim)  # (B, T, doy_enc_dim)
        x = torch.cat([x, pos], dim=-1)  # (B, T, d_in + doy_enc_dim)
        x = self.in_proj(x)  # (B, T, d_model)

        # Temporal self-attention
        mem = self.transformer(x)  # (B, T, d_model)

        # Master query attention pooling
        q = self.master_query.expand(B, -1, -1)  # (B, 1, d_model)
        out, _ = self.attn_pool(q, mem, mem)  # (B, 1, d_model)
        return out.squeeze(1)  # (B, d_model)


# ---------------------------------------------------------------------------
# Full PSE+TAE classifier
# ---------------------------------------------------------------------------


class PSETAEClassifier(nn.Module):
    """PSE + TAE + FC head for parcel-level crop-type classification.

    Input:  x    (B, T, S, C)  -- pixel sets over T dates
            doy  (B, T)         -- day-of-year per date
    Output: (B, n_classes) logits
    """

    def __init__(
        self,
        in_ch: int = 10,
        n_classes: int = 20,
        pse_out: int = 128,
        tae_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 1,
        doy_dim: int = 24,
    ) -> None:
        super().__init__()
        self.pse = PixelSetEncoder(in_ch=in_ch, mlp_dims=[32, 64], out_dim=pse_out)
        self.tae = TemporalAttentionEncoder(
            d_in=pse_out,
            d_model=tae_model,
            n_heads=n_heads,
            n_layers=n_layers,
            doy_enc_dim=doy_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(tae_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        spatial = self.pse(x)  # (B, T, pse_out)
        temporal = self.tae(spatial, doy)  # (B, tae_model)
        return self.classifier(temporal)


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def build_pse_tae() -> nn.Module:
    return PSETAEClassifier(
        in_ch=10,
        n_classes=20,
        pse_out=128,
        tae_model=128,
        n_heads=2,
        n_layers=1,
        doy_dim=24,
    )


def example_input_pse_tae():
    """Two-tensor input: pixel-set tensor + DOY tensor."""
    x = torch.randn(1, 12, 16, 10)  # (B=1, T=12, S=16, C=10)
    doy = torch.randint(1, 366, (1, 12))
    return (x, doy)


MENAGERIE_ENTRIES = [
    (
        "PSE+TAE (Pixel-Set Encoder + Temporal Attention Encoder, satellite time series)",
        "build_pse_tae",
        "example_input_pse_tae",
        "2020",
        "DC",
    ),
]
