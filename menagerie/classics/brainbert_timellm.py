"""BrainBERT and Time-LLM: neural-recording BERT and time-series LLM reprogramming.

BrainBERT:
  Wang et al., "BrainBERT: Self-supervised representation learning for
  intracranial recordings." arXiv:2302.14367 (ICLR 2023).
  Source: https://github.com/czlwang/BrainBERT

Time-LLM:
  Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming Large
  Language Models." arXiv:2310.01728 (ICLR 2024).
  Source: https://github.com/KimMeen/Time-LLM

------------------------------------------------------------------------------
BrainBERT distinctive primitive:
  BERT-style masked spectrogram modeling for intracranial neural recordings
  (SEEG). The model:
  1. Takes a multi-channel spectrogram: (n_channels, n_freq, n_time).
  2. Patchifies frequency bins (patch across frequency axis).
  3. Embeds each (channel, freq_patch) token -> token embedding.
  4. Runs a BERT transformer encoder with masked-spectrogram pretraining.
  5. Reconstructs masked frequency bins.

  Distinctive: the spectrogram-patch embed (frequency-patch tokens over
  multiple brain channels) + BERT transformer applied to neural data.

Faithful-compact simplifications:
  - 4 brain channels, 16 freq bins, 8 time frames.
  - Patch size: 4 freq bins (4 patches per channel).
  - d_model=32, 4 heads, 2 layers.
  - Output: per-patch reconstruction (n_patches*n_channels, n_freq_patch*n_time).
  - Random init, CPU, forward-only.

------------------------------------------------------------------------------
Time-LLM distinctive primitive:
  "Reprogramming" a frozen LLM for time series forecasting:
  1. Patch the time series: (B, T) -> (B, n_patches, patch_len).
  2. "Reprogram" patches via CROSS-ATTENTION against a small set of
     TEXT PROTOTYPE EMBEDDINGS (Ep: a learned embedding bank of size K).
     The cross-attention maps each patch to a weighted combination of the
     text prototype vocabulary. This is the key novelty: patches are
     "reprogrammed" into the LLM's text embedding space.
  3. "Prompt-as-Prefix" (PaP): prepend a text prompt token embedding to the
     reprogrammed patches (here: a learnable prefix).
  4. Feed to a transformer backbone (stand-in for the frozen LLM).
  5. Linear forecast head: pool last few tokens -> forecast horizon.

Faithful-compact simplifications:
  - LLM body = a 2-layer transformer (not GPT-2; documented stand-in).
  - K=8 text prototype embeddings.
  - Patch len=4, n_patches=4 from T=16.
  - d_model=32 (reprogramming cross-attn).
  - Forecast horizon H=4.
  - Random init, CPU, forward-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# BrainBERT
# =============================================================================


class SpectrogramPatchEmbed(nn.Module):
    """Patch embedding for multi-channel spectrogram.

    Input: (n_ch, n_freq, n_time)
    Patchify along frequency axis with stride=patch_size.
    Each (channel, freq_patch, time) -> flattened -> linear -> token embedding.
    """

    def __init__(
        self,
        n_channels: int,
        n_freq: int,
        n_time: int,
        patch_size: int,
        d_model: int,
    ) -> None:
        super().__init__()
        assert n_freq % patch_size == 0
        self.n_patches_per_ch = n_freq // patch_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_time = n_time
        # Flattened patch: patch_size * n_time values per token
        self.proj = nn.Linear(patch_size * n_time, d_model)
        # Channel embedding
        self.chan_embed = nn.Embedding(n_channels, d_model)
        # Patch-position embedding
        self.pos_embed = nn.Embedding(self.n_patches_per_ch, d_model)
        # Mask token (for BERT-style masking)
        self.mask_token = nn.Parameter(torch.randn(d_model))

    def forward(
        self,
        x: torch.Tensor,  # (n_ch, n_freq, n_time)
        mask: torch.Tensor | None = None,  # (n_ch * n_patches_per_ch,) bool
    ) -> tuple[torch.Tensor, int]:
        """Returns tokens (N_tok, d_model) and N_tok."""
        n_ch, n_freq, n_time = x.shape
        # Patchify: (n_ch, n_patches, patch_size * n_time)
        x_patches = x.view(n_ch, self.n_patches_per_ch, self.patch_size, n_time)
        x_patches = x_patches.reshape(n_ch, self.n_patches_per_ch, -1)  # (n_ch, P, D_in)
        # Project to d_model
        tok = self.proj(x_patches)  # (n_ch, P, d_model)

        # Add channel + position embeddings
        chan_ids = torch.arange(n_ch, device=x.device)
        pos_ids = torch.arange(self.n_patches_per_ch, device=x.device)
        tok = tok + self.chan_embed(chan_ids).unsqueeze(1) + self.pos_embed(pos_ids).unsqueeze(0)

        # Flatten to (n_ch*P, d_model)
        tok = tok.reshape(n_ch * self.n_patches_per_ch, -1)
        N_tok = tok.size(0)

        # Apply masking
        if mask is not None:
            tok = tok.clone()
            tok[mask] = self.mask_token

        return tok, N_tok


class BrainBERT(nn.Module):
    """BrainBERT: spectrogram-patch BERT for intracranial neural recordings."""

    def __init__(
        self,
        n_channels: int = 4,
        n_freq: int = 16,
        n_time: int = 8,
        patch_size: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.patch_embed = SpectrogramPatchEmbed(n_channels, n_freq, n_time, patch_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        # Reconstruction head: each token -> reconstructed freq-patch * time
        self.recon_head = nn.Linear(d_model, patch_size * n_time)

    def forward(
        self,
        x: torch.Tensor,  # (n_ch, n_freq, n_time)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns reconstruction (N_tok, patch_size*n_time)."""
        tok, _ = self.patch_embed(x, mask)  # (N_tok, d_model)
        tok = tok.unsqueeze(0)  # (1, N_tok, d_model) for batch-first
        tok = self.transformer(tok)
        tok = self.norm(tok.squeeze(0))
        return self.recon_head(tok)  # (N_tok, patch_size*n_time)


def build_brainbert() -> nn.Module:
    return BrainBERT(n_channels=4, n_freq=16, n_time=8, patch_size=4, d_model=32)


def example_input_brainbert() -> list[torch.Tensor]:
    torch.manual_seed(15)
    x = torch.randn(4, 16, 8)  # (n_channels, n_freq, n_time)
    n_tok = 4 * 4  # n_channels * (n_freq//patch_size)
    # Mask ~25% of tokens
    mask = torch.zeros(n_tok, dtype=torch.bool)
    mask[:4] = True
    return [x, mask]


# =============================================================================
# Time-LLM
# =============================================================================


class ReprogrammingCrossAttn(nn.Module):
    """Cross-attention that reprograms time-series patches to text-prototype space.

    Each time-series patch (query) attends over K text prototype embeddings (keys/values).
    Output: reprogrammed patch embedding in text prototype space.
    """

    def __init__(self, d_patch: int, d_proto: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Project patch to query
        self.q_proj = nn.Linear(d_patch, d_model, bias=False)
        # Keys and values from prototypes
        self.k_proj = nn.Linear(d_proto, d_model, bias=False)
        self.v_proj = nn.Linear(d_proto, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        patches: torch.Tensor,  # (n_patches, d_patch)
        prototypes: torch.Tensor,  # (K, d_proto) text prototype embeddings
    ) -> torch.Tensor:
        """Returns reprogrammed patches: (n_patches, d_model)"""
        N = patches.size(0)
        K = prototypes.size(0)

        q = self.q_proj(patches).view(N, self.n_heads, self.d_head)
        k = self.k_proj(prototypes).view(K, self.n_heads, self.d_head)
        v = self.v_proj(prototypes).view(K, self.n_heads, self.d_head)

        # Cross-attention: N queries, K keys
        attn = torch.einsum("nhd,khd->nkh", q, k) / math.sqrt(self.d_head)  # (N, K, n_heads)
        attn = F.softmax(attn, dim=1)  # softmax over K
        out = torch.einsum("nkh,khd->nhd", attn, v).reshape(N, -1)
        return self.out_proj(out)  # (N, d_model)


class TimeLLM(nn.Module):
    """Time-LLM: time series reprogrammed into text-prototype space for LLM forecasting.

    Components:
    1. Patch time series.
    2. Reprogram patches via cross-attention against text prototypes.
    3. Prepend prompt-as-prefix (learnable prefix tokens).
    4. Feed to transformer (LLM stand-in).
    5. Forecast head.
    """

    def __init__(
        self,
        patch_len: int = 4,
        n_patches: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        K_proto: int = 8,  # number of text prototypes
        d_proto: int = 32,  # prototype embedding dim
        n_prefix: int = 2,  # prompt-as-prefix tokens
        forecast_len: int = 4,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = n_patches
        self.n_prefix = n_prefix

        # Text prototype embeddings (learned)
        self.text_prototypes = nn.Embedding(K_proto, d_proto)

        # Reprogramming cross-attention
        self.reprogram = ReprogrammingCrossAttn(patch_len, d_proto, d_model, n_heads)

        # Prompt-as-prefix: learnable prefix tokens in d_model space
        self.prefix_tokens = nn.Parameter(torch.randn(n_prefix, d_model))

        # Position embedding for all tokens (prefix + patches)
        T_total = n_prefix + n_patches
        self.pos_embed = nn.Embedding(T_total, d_model)

        # Transformer backbone (LLM stand-in: 2-layer transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Forecast head: last n_patches tokens -> forecast_len
        self.forecast_head = nn.Linear(n_patches * d_model, forecast_len)

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        """ts: (T,) time series -> forecast: (forecast_len,)"""
        T = ts.size(0)
        # Patch the series: (n_patches, patch_len)
        patches = ts[: self.n_patches * self.patch_len].view(self.n_patches, self.patch_len)

        # Get text prototypes
        proto_ids = torch.arange(self.text_prototypes.num_embeddings, device=ts.device)
        protos = self.text_prototypes(proto_ids)  # (K, d_proto)

        # Reprogram patches
        reprog = self.reprogram(patches, protos)  # (n_patches, d_model)

        # Prepend prefix
        prefix = self.prefix_tokens  # (n_prefix, d_model)
        tokens = torch.cat([prefix, reprog], dim=0)  # (n_prefix+n_patches, d_model)

        # Add position embedding
        pos = torch.arange(tokens.size(0), device=ts.device)
        tokens = tokens + self.pos_embed(pos)

        # Transformer
        tokens = tokens.unsqueeze(0)  # (1, T, d_model)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens.squeeze(0))  # (T, d_model)

        # Forecast: use last n_patches tokens
        out = tokens[-self.n_patches :].reshape(-1)  # (n_patches * d_model,)
        return self.forecast_head(out)  # (forecast_len,)


def build_timellm() -> nn.Module:
    return TimeLLM(
        patch_len=4,
        n_patches=4,
        d_model=32,
        n_heads=4,
        n_layers=2,
        K_proto=8,
        d_proto=32,
        n_prefix=2,
        forecast_len=4,
    )


def example_input_timellm() -> torch.Tensor:
    torch.manual_seed(16)
    return torch.randn(16)  # T=16 time series


# =============================================================================
# Registry
# =============================================================================

MENAGERIE_ENTRIES = [
    (
        "BrainBERT",
        "build_brainbert",
        "example_input_brainbert",
        "2023",
        "DC",
    ),
    (
        "Time-LLM",
        "build_timellm",
        "example_input_timellm",
        "2024",
        "DC",
    ),
]
