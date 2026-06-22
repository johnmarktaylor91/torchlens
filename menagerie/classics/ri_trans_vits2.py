"""VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech.

Kong et al., INTERSPEECH 2023.
Paper: https://arxiv.org/abs/2307.16430
Source: https://github.com/daniilrobnikov/vits2

VITS2 keeps the VITS conditional-VAE + adversarial single-stage TTS skeleton
(text encoder -> normalizing-flow prior -> HiFiGAN decoder) but the DISTINCTIVE
DELTA over plain VITS is reproduced here:

  1. TRANSFORMER-AUGMENTED NORMALIZING FLOW: each affine coupling layer in the
     flow no longer uses only a WaveNet of dilated convs; VITS2 inserts a small
     self-ATTENTION (transformer) block into the coupling transform, improving
     the prior's expressiveness. (`TransformerCouplingLayer`.)
  2. SPEAKER-CONDITIONED STOCHASTIC DURATION PREDICTOR: the duration predictor
     is conditioned on the speaker embedding and uses a small transformer/conv
     stack rather than the original convolutional-only stochastic predictor.
  3. MONOTONIC ALIGNMENT: text frames are expanded to the spectrogram length
     via a (here, fixed/length-regulated) monotonic alignment instead of noisy
     attention.

This module reproduces (1) the transformer coupling flow and (2) the
speaker-conditioned duration predictor, wired with a small text encoder and a
length-regulated expansion to a tiny HiFiGAN-style decoder. Faithful compact
random-init reimplementation: tiny vocab/hidden/flow-depth and short sequences
so the unrolled trace draws quickly. Forward-able from a single token-id tensor
(speaker id + durations baked in). arXiv:2307.16430.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """Token embedding + small transformer encoder -> prior mean/logvar over frames."""

    def __init__(self, vocab: int, dim: int, heads: int, layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.proj = nn.Linear(dim, dim * 2)  # mean + logvar

    def forward(self, tokens: torch.Tensor):
        h = self.encoder(self.embed(tokens))
        stats = self.proj(h)
        m, logs = stats.chunk(2, dim=-1)
        return h, m, logs


class TransformerCouplingLayer(nn.Module):
    """VITS2 affine coupling with a self-attention block in the transform (the delta)."""

    def __init__(self, channels: int, heads: int = 2, gin_channels: int = 16) -> None:
        super().__init__()
        self.half = channels // 2
        self.pre = nn.Linear(self.half + gin_channels, channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels)
        )
        self.post = nn.Linear(channels, self.half)  # produces the shift (volume-preserving)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) ; g: (B, 1, gin) speaker embedding
        x0, x1 = x[..., : self.half], x[..., self.half :]
        gg = g.expand(-1, x0.shape[1], -1)
        h = self.pre(torch.cat([x0, gg], dim=-1))
        a, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm(h + a)
        h = h + self.ffn(h)
        shift = self.post(h)
        x1 = x1 + shift  # additive (invertible) coupling
        return torch.cat([x0, x1], dim=-1)


class TransformerFlow(nn.Module):
    """Stack of transformer coupling layers with channel-flip between them."""

    def __init__(self, channels: int, n_flows: int = 3, heads: int = 2, gin: int = 16) -> None:
        super().__init__()
        self.flows = nn.ModuleList(
            [TransformerCouplingLayer(channels, heads, gin) for _ in range(n_flows)]
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        for flow in self.flows:
            x = flow(x, g)
            x = torch.flip(x, dims=[-1])  # channel flip
        return x


class SpeakerDurationPredictor(nn.Module):
    """Speaker-conditioned duration predictor (transformer/conv stack)."""

    def __init__(self, dim: int, gin: int = 16) -> None:
        super().__init__()
        self.cond = nn.Linear(gin, dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # h: (B, T, dim) ; g: (B, 1, gin)
        h = h + self.cond(g)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return self.proj(h)  # (B, T, 1) log-durations


class _HiFiGANHead(nn.Module):
    """Tiny upsampling decoder standing in for the HiFiGAN waveform generator."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(dim, dim // 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(dim // 2, dim // 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.out = nn.Conv1d(dim // 4, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim) -> waveform (B, 1, 4T)
        return torch.tanh(self.out(self.up(x.transpose(1, 2))))


class VITS2(nn.Module):
    """VITS2 generator: text enc -> transformer-flow prior -> duration -> HiFiGAN head."""

    def __init__(
        self,
        vocab: int = 178,
        dim: int = 64,
        heads: int = 2,
        enc_layers: int = 2,
        n_flows: int = 3,
        n_speakers: int = 4,
        gin: int = 16,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.text_encoder = TextEncoder(vocab, dim, heads, enc_layers)
        self.speaker_embed = nn.Embedding(n_speakers, gin)
        self.dur_predictor = SpeakerDurationPredictor(dim, gin)
        self.flow = TransformerFlow(dim, n_flows, heads, gin)
        self.dec = _HiFiGANHead(dim)
        self.expand = expand

    def forward(self, tokens: torch.Tensor, speaker_id: torch.Tensor) -> torch.Tensor:
        g = self.speaker_embed(speaker_id).unsqueeze(1)  # (B, 1, gin)
        h, m, logs = self.text_encoder(tokens)
        _logdur = self.dur_predictor(h, g)  # speaker-conditioned durations
        # length-regulated monotonic expansion (fixed factor for the atlas)
        z = m + torch.randn_like(m) * torch.exp(logs)
        z = z.repeat_interleave(self.expand, dim=1)
        z = self.flow(z, g)  # transformer-augmented normalizing flow (VITS2 delta)
        return self.dec(z)


class _VITS2Wrapper(nn.Module):
    """Single-tensor wrapper: input is token ids; speaker id baked in."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        spk = torch.zeros(tokens.shape[0], dtype=torch.long, device=tokens.device)
        return self.model(tokens, spk)


def build_vits2() -> nn.Module:
    """Build a compact VITS2 (transformer-flow prior + speaker-conditioned duration TTS)."""
    return _VITS2Wrapper(
        VITS2(vocab=178, dim=64, heads=2, enc_layers=2, n_flows=3, n_speakers=4, gin=16, expand=2)
    )


def example_input() -> torch.Tensor:
    """Example phoneme token ids ``(1, 32)`` int64 for VITS2."""
    return torch.randint(0, 178, (1, 32), dtype=torch.int64)


MENAGERIE_ENTRIES = [
    (
        "VITS2 (transformer-flow prior + speaker-conditioned duration TTS)",
        "build_vits2",
        "example_input",
        "2023",
        "DC",
    ),
]
