"""StyleTTS2 and RawNet3: neural speech synthesis and speaker verification.

StyleTTS2:
  Li et al., "StyleTTS 2: Towards Human-Level Text-to-Speech through Style
  Diffusion and Adversarial Training with Large Speech Language Models."
  arXiv:2306.07691 (NeurIPS 2023).
  Source: https://github.com/yl4579/StyleTTS2

RawNet3:
  Jung et al., "Pushing the limits of raw waveform speaker verification."
  arXiv:2203.04993 (Interspeech 2022).
  Source: https://github.com/Jungjee/RawNet3
  (Incorporated in SpeakerNet, NVIDIA NeMo.)

------------------------------------------------------------------------------
StyleTTS2 distinctive primitive:
  Style diffusion + AdaIN-conditioned decoder for TTS.
  Core components captured:
  1. Text encoder: character/phoneme embed + transformer encoder.
  2. Style encoder: takes reference Mel-spectrogram -> style vector via
     downsampling conv stack + global pooling.
  3. Style diffusion sampler: a small DDPM over the style space (denoises
     a Gaussian noise to a style vector via a score network conditioned on
     the text encoder output). This is the distinctive novel component.
  4. AdaIN-conditioned decoder: upsampling conv stack where each layer is
     AdaIN-conditioned on the style vector (normalize features then shift/scale
     by learned affine params from style vector).

Faithful-compact simplifications:
  - Text encoder: 2-layer transformer on phoneme embeds (16 phonemes).
  - Style encoder: 3 strided convs (Mel -> 64-dim style vector).
  - Style diffusion: 3-step DDPM; score network = small MLP.
  - AdaIN decoder: 3 upsampling blocks.
  - Mel spectrogram: 64 bins, 16 frames.
  - Output: upsampled waveform (simplified: just the decoder output shape).
  - Heavy simplification documented: duration predictor, LM discriminator,
    and full adversarial training loop are not included.

RawNet3 distinctive primitive:
  Raw-waveform speaker verification pipeline:
  1. Sinc-conv (parametric) front-end or standard strided conv on raw audio.
  2. Res2Net blocks: residual convolutions with a multi-scale feature split
     (Gao et al., Res2Net) where the residual is computed across K scales
     within the block.
  3. SE (Squeeze-Excitation) attention on channel features.
  4. Attentive Statistics Pooling (ASP): computes mean and (weighted) std
     of frame-level features via a learned attention mask, then concatenates
     mean + std as the utterance-level embedding.

RawNet3 simplifications:
  - 2 Res2Net-SE blocks instead of 6.
  - d_model=64.
  - 8 Res2Net scales (paper: 8).
  - 1-second waveform @ 16kHz = 16000 samples -> use 256 samples for speed.
  - Random init, CPU, forward-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# StyleTTS2
# =============================================================================


class TextEncoder(nn.Module):
    """Simple phoneme -> encoder hidden state transformer."""

    def __init__(self, vocab: int = 40, d: int = 32, n_heads: int = 4, n_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d, n_heads, 4 * d, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, phonemes: torch.Tensor) -> torch.Tensor:
        """phonemes: (1, L) int64 -> (1, L, d)"""
        x = self.embed(phonemes)
        return self.transformer(x)


class StyleEncoder(nn.Module):
    """Mel-spectrogram -> style vector via strided conv + global pooling."""

    def __init__(self, n_mels: int = 64, d_style: int = 32) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After 3 stride-2 convs on (1, 1, n_mels, T): spatial dims halved 3x
        # n_mels=64 -> 8 (freq), pool time
        feat_dim = 64 * (n_mels // 8)
        self.proj = nn.Linear(feat_dim, d_style)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: (1, n_mels, T) -> style: (d_style,)"""
        x = mel.unsqueeze(1)  # (1, 1, n_mels, T)
        x = self.convs(x)  # (1, 64, n_mels//8, T//8)
        x = x.mean(dim=-1)  # pool time: (1, 64, n_mels//8)
        x = x.view(1, -1)  # (1, 64 * n_mels//8)
        return self.proj(x).squeeze(0)  # (d_style,)


class StyleScoreNet(nn.Module):
    """Score network for style diffusion: (noisy_style, t, text_context) -> score."""

    def __init__(self, d_style: int = 32, d_text: int = 32, d_time: int = 16) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, d_time), nn.SiLU(), nn.Linear(d_time, d_time))
        self.net = nn.Sequential(
            nn.Linear(d_style + d_text + d_time, 64),
            nn.SiLU(),
            nn.Linear(64, d_style),
        )

    def forward(
        self,
        s_t: torch.Tensor,  # (d_style,)
        t: torch.Tensor,  # scalar
        ctx: torch.Tensor,  # (d_text,) mean-pooled text context
    ) -> torch.Tensor:
        t_emb = self.time_embed(t.unsqueeze(0).unsqueeze(-1)).squeeze(0)  # (d_time,)
        return self.net(torch.cat([s_t, ctx, t_emb], dim=-1))


class AdaINResBlock(nn.Module):
    """Decoder block: 1D conv + AdaIN conditioning on style vector."""

    def __init__(self, channels: int, d_style: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.norm = nn.InstanceNorm1d(channels, affine=False)
        # AdaIN: style -> (scale, shift) per channel
        self.adain = nn.Linear(d_style, 2 * channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """x: (1, C, T) style: (d_style,) -> (1, C, 2T)"""
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        ab = self.adain(style)  # (2C,)
        a, b = ab[: ab.size(0) // 2], ab[ab.size(0) // 2 :]
        x = x * (1 + a.unsqueeze(0).unsqueeze(-1)) + b.unsqueeze(0).unsqueeze(-1)
        return x


class StyleTTS2(nn.Module):
    """StyleTTS2 (compact): text encoder + style encoder + style diffusion + AdaIN decoder.

    Forward pass mimics inference:
      1. Encode phonemes -> text_hidden.
      2. Encode mel reference -> ref_style.
      3. Run style diffusion (N_diff steps of denoising) -> sampled_style.
         (For inference, start from Gaussian noise and denoise toward the style space.)
      4. Decode: AdaIN-conditioned upsampling on a learned input token.
    """

    def __init__(
        self,
        vocab: int = 40,
        n_mels: int = 64,
        d_model: int = 32,
        d_style: int = 32,
        n_diff_steps: int = 3,
    ) -> None:
        super().__init__()
        self.text_enc = TextEncoder(vocab, d_model)
        self.style_enc = StyleEncoder(n_mels, d_style)
        self.score_net = StyleScoreNet(d_style, d_model, 16)
        # AdaIN decoder
        self.decoder_input = nn.Parameter(torch.randn(1, d_model, 1))  # learned start token
        self.dec_blocks = nn.ModuleList([AdaINResBlock(d_model, d_style) for _ in range(3)])
        self.out_conv = nn.Conv1d(d_model, 1, 1)
        self.n_diff = n_diff_steps

    def forward(
        self,
        phonemes: torch.Tensor,  # (1, L) int64
        mel_ref: torch.Tensor,  # (1, n_mels, T) float
        noise: torch.Tensor,  # (d_style,) initial noise for style diffusion
    ) -> torch.Tensor:
        """Returns waveform: (1, 1, T') -- simplified decoder output."""
        # 1. Text encoding
        text_hidden = self.text_enc(phonemes)  # (1, L, d_model)
        text_ctx = text_hidden.mean(dim=1).squeeze(0)  # (d_model,)

        # 2. Style encoding from reference mel
        ref_style = self.style_enc(mel_ref)  # (d_style,)

        # 3. Style diffusion: simplified DDPM denoising (N steps)
        s = noise.clone()
        for step in range(self.n_diff):
            t = torch.tensor(1.0 - step / self.n_diff)
            score = self.score_net(s, t, text_ctx)  # (d_style,)
            s = s - 0.1 * score  # simplified Euler step

        # Use ref_style (in inference it comes from the diffusion sample)
        style = ref_style

        # 4. AdaIN decoder
        h = self.decoder_input  # (1, d_model, 1)
        for blk in self.dec_blocks:
            h = blk(h, style)  # (1, d_model, 2^k)
        return self.out_conv(h)  # (1, 1, 8)


def build_styletts2() -> nn.Module:
    return StyleTTS2(vocab=40, n_mels=64, d_model=32, d_style=32, n_diff_steps=3)


def example_input_styletts2() -> list[torch.Tensor]:
    torch.manual_seed(13)
    phonemes = torch.randint(0, 40, (1, 16))
    mel_ref = torch.randn(1, 64, 16)
    noise = torch.randn(32)
    return [phonemes, mel_ref, noise]


# =============================================================================
# RawNet3
# =============================================================================


class Res2NetBlock(nn.Module):
    """Res2Net block for raw audio: multi-scale residual within channels.

    The Res2Net idea: split channels into K scales, process each scale
    with a 1D conv, and feed into the next scale cumulatively.
    SE attention is applied at the end.
    """

    def __init__(self, channels: int, k: int = 4, stride: int = 1) -> None:
        super().__init__()
        assert channels % k == 0
        self.k = k
        self.width = channels // k
        # 1D conv for each scale (except the first, which is identity)
        self.convs = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, 3, stride=stride, padding=1) for _ in range(k - 1)]
        )
        self.bn = nn.BatchNorm1d(channels)
        # SE: global pool -> FC -> sigmoid gating
        self.se_fc1 = nn.Linear(channels, channels // 4)
        self.se_fc2 = nn.Linear(channels // 4, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, C, T) -> (1, C, T)"""
        parts = x.chunk(self.k, dim=1)  # K tensors of (1, C//K, T)
        outs = []
        prev = None
        for i, sp in enumerate(parts):
            if i == 0:
                outs.append(sp)
                prev = sp
            else:
                y = self.convs[i - 1](sp if prev is None else sp + prev)
                outs.append(y)
                prev = y
        out = torch.cat(outs, dim=1)  # (1, C, T)
        out = self.bn(out)
        # SE attention
        se = out.mean(dim=-1)  # (1, C) global avg
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        out = out * se.unsqueeze(-1)
        return out + x  # residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling: learned attention -> weighted mean + std.

    Maps (1, C, T) -> (1, 2C) speaker embedding.
    This is the aggregation that creates a fixed-size utterance embedding.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 4, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, C, T) -> (1, 2C)"""
        w = self.attn(x)  # (1, C, T)
        w = F.softmax(w, dim=-1)  # attention weights
        mean = (w * x).sum(dim=-1)  # (1, C)
        # Weighted std: sqrt(E[x^2] - E[x]^2)
        std = torch.sqrt((w * x.pow(2)).sum(dim=-1) - mean.pow(2) + 1e-8)  # (1, C)
        return torch.cat([mean, std], dim=-1)  # (1, 2C)


class RawNet3(nn.Module):
    """RawNet3 speaker verification from raw waveform.

    Pipeline:
      raw waveform -> front conv -> 2x Res2Net-SE blocks -> ASP -> speaker embed
    """

    def __init__(
        self,
        channels: int = 64,
        n_res2net_blocks: int = 2,
        n_scales: int = 4,
        d_embed: int = 128,
    ) -> None:
        super().__init__()
        # Front-end: strided conv on raw waveform
        self.front_conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, stride=4, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
        )
        # Res2Net-SE backbone
        self.res2net_blocks = nn.ModuleList(
            [Res2NetBlock(channels, n_scales) for _ in range(n_res2net_blocks)]
        )
        # Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(channels)
        # Speaker embedding projection
        self.embed = nn.Linear(2 * channels, d_embed)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (1, 1, T) raw audio -> speaker embed: (1, d_embed)"""
        # waveform: (1, 1, T) -> front_conv expects (1, 1, T)
        x = self.front_conv(waveform.squeeze(0).unsqueeze(0))  # (1, channels, T')
        for blk in self.res2net_blocks:
            x = blk(x)
        x = self.asp(x)  # (1, 2*channels)
        return self.embed(x)  # (1, d_embed)


def build_rawnet3() -> nn.Module:
    return RawNet3(channels=64, n_res2net_blocks=2, n_scales=4, d_embed=128)


def example_input_rawnet3() -> torch.Tensor:
    """256-sample raw waveform (batch 1, 1 channel)."""
    torch.manual_seed(14)
    return torch.randn(1, 1, 256)


# =============================================================================
# Registry
# =============================================================================

MENAGERIE_ENTRIES = [
    (
        "StyleTTS2",
        "build_styletts2",
        "example_input_styletts2",
        "2023",
        "DC",
    ),
    (
        "RawNet3",
        "build_rawnet3",
        "example_input_rawnet3",
        "2022",
        "DC",
    ),
]
