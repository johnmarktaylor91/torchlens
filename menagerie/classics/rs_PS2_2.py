# FAITHFUL REIMPLEMENTATION from published architecture descriptions (no public code)
#
# "21cm Transformer" (candidate queue name) is a generic descriptor, not a single named repo: no
# GitHub search or arXiv search turned up dedicated code for a model under this exact name. The
# candidate's own triage note and independent research both confirm the family is real and
# well-precedented: multiple papers apply standard multi-head self-attention Transformer encoders
# (Vaswani et al., "Attention Is All You Need") to sequences of 21-cm power-spectrum measurements
# (one power-spectrum vector per redshift bin) to infer Epoch-of-Reionization astrophysical
# parameters (e.g. ionizing efficiency zeta, minimum virial temperature T_vir, X-ray luminosity per
# star-formation rate L_X/SFR, mean free path R_mfp) -- see e.g. arXiv:2203.15734 (implicit-likelihood
# inference from the 21cm power spectrum), arXiv:2303.07339 (Marginal Neural Ratio Estimation on
# 21-cm power spectra), and arXiv:2112.13866 (ANN-based reionization-parameter extraction), which
# together give a sufficiently detailed, convergent description of "linear per-redshift embedding of
# the power spectrum -> positional/redshift encoding -> stacked self-attention encoder -> pooled
# regression head over reionization parameters" to reimplement faithfully. No code exists to
# vendor/port (rung 2/3), so this is RUNG 4: the Transformer ENCODER mechanism itself is not
# hand-approximated -- it is built from the real `torch.nn.TransformerEncoderLayer` /
# `torch.nn.TransformerEncoder` (the actual multi-head self-attention + position-wise feed-forward
# + residual/LayerNorm mechanism from the base library), wrapped with the domain-specific embedding
# and regression head the papers describe.
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

MENAGERIE_ZOO = "reimpl-pytorch"


class RedshiftPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al.) over the redshift-bin axis."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.shape[1]]


class ReionizationTransformer(nn.Module):
    """Self-attention Transformer encoder over a 21-cm power-spectrum sequence, regressing to EoR
    astrophysical parameters (ionizing efficiency, virial temperature, X-ray luminosity, mean free
    path), following the attention-based 21-cm-power-spectrum-inference architecture family."""

    def __init__(
        self,
        n_k_bins: int = 16,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        n_params: int = 4,
        max_redshift_bins: int = 32,
    ) -> None:
        super().__init__()
        self.input_embed = nn.Linear(n_k_bins, d_model)
        self.pos_encoding = RedshiftPositionalEncoding(d_model, max_len=max_redshift_bins)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, n_params),
        )

    def forward(self, power_spectrum: Tensor) -> Tensor:
        """power_spectrum: (batch, n_redshift_bins, n_k_bins) 21-cm power spectrum sequence."""
        embedded = self.input_embed(power_spectrum)
        embedded = self.pos_encoding(embedded)
        encoded = self.encoder(embedded)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


def build_reionization_transformer() -> nn.Module:
    model = ReionizationTransformer(
        n_k_bins=16,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        n_params=4,
        max_redshift_bins=32,
    )
    model.eval()
    return model


def example_input_reionization_transformer() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 8, 16)  # (batch, 8 redshift bins, 16 k-bins per power spectrum)


MENAGERIE_ENTRIES = [
    (
        "21cm Transformer",
        "build_reionization_transformer",
        "example_input_reionization_transformer",
        2022,
        MENAGERIE_ZOO,
    ),
]
