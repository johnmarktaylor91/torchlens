# FAITHFUL REIMPLEMENTATION from arXiv:2210.14666 (no public code) -- A/B sonnet
"""XiaoiceSing2: singing-voice-synthesis GAN with a FastSpeech-style generator
and a multi-band/multi-length adversarial discriminator (arXiv:2210.14666).

Distinctive mechanisms (Section 2):
  1. Generator: encoder + length regulator + decoder, both encoder/decoder built
     from "ConvFFT" blocks -- each block fuses multi-head self-attention with
     PARALLEL residual convolutional sub-blocks (Section 2.1). Encoder has 6
     ConvFFT blocks with 2 residual convs each; decoder has 6 ConvFFT blocks
     with 5 residual convs each. LogF0 is predicted with a residual connection
     to the input pitch sequence.
  2. Discriminator: "multi-band" -- three parallel sub-discriminators over
     low/mid/high frequency slices of the 120-bin mel ([0:60],[30:90],[60:120],
     Section 2.2) -- each sub-discriminator is itself "multi-length": a
     Segment Discriminator (10-layer 1D CNN, kernel 3, hidden 128) evaluated at
     several temporal window lengths, plus a Detail Discriminator (2D
     PatchGAN-style CNN: 1 entry conv + 5 downsampling convs (3,3)/dilation
     (2,2) + 5 (1,3) output convs) examining local time-frequency patches
     (Section 2.2.1-2.2.2).
  3. Adversarial training uses an LSGAN objective (Eqs. 3-4) plus an L1 feature
     matching loss over discriminator intermediate activations (Eq. 6).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Generator ------------------------------------


class ResidualConvBlock(nn.Module):
    """A single residual 1D conv sub-block used inside a ConvFFT block."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=pad)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + F.relu(y))


class ConvFFTBlock(nn.Module):
    """Improved FFT block (Section 2.1): MHSA + N parallel residual conv
    sub-blocks, fused back together ("feature fusion to balance local and
    global information")."""

    def __init__(self, dim: int, n_heads: int, num_conv_blocks: int):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.mhsa_norm = nn.LayerNorm(dim)
        self.conv_blocks = nn.ModuleList([ResidualConvBlock(dim) for _ in range(num_conv_blocks)])
        self.fusion = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mhsa(x, x, x, need_weights=False)
        attn_out = self.mhsa_norm(x + attn_out)

        conv_out = x
        for block in self.conv_blocks:
            conv_out = block(conv_out)

        fused = self.fusion(torch.cat([attn_out, conv_out], dim=-1))
        return fused


class DurationPredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x.transpose(1, 2)))
        h = F.relu(self.conv2(h)).transpose(1, 2)
        return self.proj(h).squeeze(-1)  # (B, T) log-duration


def length_regulate(x: torch.Tensor, durations: torch.Tensor, max_len: int):
    """Expand each phoneme's hidden vector `durations[b,t]` times (rounded, clamped)."""
    batch, T, D = x.shape
    out = torch.zeros(batch, max_len, D, device=x.device, dtype=x.dtype)
    dur_int = durations.round().clamp(min=1, max=max_len).long()
    for b in range(batch):
        pos = 0
        for t in range(T):
            d = int(dur_int[b, t].item())
            if pos >= max_len:
                break
            d = min(d, max_len - pos)
            out[b, pos : pos + d] = x[b, t].unsqueeze(0)
            pos += d
    return out


class XiaoiceSing2Generator(nn.Module):
    def __init__(
        self,
        num_phonemes: int = 40,
        dim: int = 32,
        n_heads: int = 4,
        mel_dim: int = 120,
        max_frames: int = 48,
        enc_conv_blocks: int = 2,
        dec_conv_blocks: int = 5,
        enc_fft_blocks: int = 6,
        dec_fft_blocks: int = 6,
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.max_frames = max_frames

        self.phone_embed = nn.Embedding(num_phonemes, dim)
        self.pitch_proj = nn.Linear(1, dim)
        self.dur_proj = nn.Linear(1, dim)
        self.input_fuse = nn.Linear(dim * 3, dim)

        self.encoder = nn.ModuleList(
            [ConvFFTBlock(dim, n_heads, enc_conv_blocks) for _ in range(enc_fft_blocks)]
        )
        self.duration_predictor = DurationPredictor(dim)
        self.decoder = nn.ModuleList(
            [ConvFFTBlock(dim, n_heads, dec_conv_blocks) for _ in range(dec_fft_blocks)]
        )

        self.mel_head = nn.Linear(dim, mel_dim)
        self.logf0_head = nn.Linear(dim, 1)  # residual on top of the (upsampled) input pitch
        self.vuv_head = nn.Linear(dim, 1)

    def forward(self, phoneme_ids: torch.Tensor, pitch: torch.Tensor, score_duration: torch.Tensor):
        # phoneme_ids: (B, T) long; pitch: (B, T) float (note-level F0/MIDI); score_duration: (B, T) float
        ph = self.phone_embed(phoneme_ids)
        pi = self.pitch_proj(pitch.unsqueeze(-1))
        du = self.dur_proj(score_duration.unsqueeze(-1))
        h = self.input_fuse(torch.cat([ph, pi, du], dim=-1))

        for block in self.encoder:
            h = block(h)

        log_dur_pred = self.duration_predictor(h)
        dur_pred = torch.exp(log_dur_pred)
        expanded = length_regulate(h, dur_pred, self.max_frames)
        pitch_expanded = length_regulate(pitch.unsqueeze(-1), dur_pred, self.max_frames).squeeze(-1)

        d = expanded
        for block in self.decoder:
            d = block(d)

        mel = self.mel_head(d)
        logf0 = (
            self.logf0_head(d).squeeze(-1) + pitch_expanded
        )  # residual pitch connection (Section 2.1)
        vuv = torch.sigmoid(self.vuv_head(d)).squeeze(-1)
        return mel, logf0, vuv, log_dur_pred


# --------------------------- Discriminator -----------------------------------


class SegmentDiscriminator(nn.Module):
    """10-layer 1D CNN, kernel 3, hidden channels H, evaluated at several
    temporal window lengths ("multi-length", Section 2.2.1)."""

    def __init__(
        self, in_channels: int, hidden: int = 32, num_layers: int = 10, window_lengths=(8, 16, 24)
    ):
        super().__init__()
        self.window_lengths = window_lengths
        layers = []
        c_in = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv1d(c_in, hidden, kernel_size=3, padding=1))
            c_in = hidden
        self.convs = nn.ModuleList(layers)
        self.out_proj = nn.Conv1d(hidden, 1, kernel_size=1)

    def _run(self, x: torch.Tensor):
        feats = []
        h = x
        for conv in self.convs:
            h = F.leaky_relu(conv(h), 0.2)
            feats.append(h)
        score = self.out_proj(h)
        return score, feats

    def forward(self, band: torch.Tensor):
        # band: (B, F_band, T) -- frequency channels, time as the conv spatial axis.
        T = band.shape[-1]
        scores, all_feats = [], []
        for w in list(self.window_lengths) + [T]:
            w = min(w, T)
            seg = band[..., :w]
            score, feats = self._run(seg)
            scores.append(score.mean(dim=[1, 2]))
            all_feats.extend(feats)
        return torch.stack(scores, dim=-1), all_feats  # (B, num_windows)


class DetailDiscriminator(nn.Module):
    """2D PatchGAN-style discriminator over a (T, F_band) time-frequency patch
    (Section 2.2.2): 1 entry conv -> 5 downsampling convs (3,3)/dilation(2,2)
    -> 5 output convs (1,3)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.entry = nn.Conv2d(1, hidden * 2, kernel_size=(3, 3), padding=1)
        self.down = nn.ModuleList(
            [
                nn.Conv2d(hidden * 2, hidden * 2, kernel_size=(3, 3), padding=2, dilation=(2, 2))
                for _ in range(5)
            ]
        )
        self.out_convs = nn.ModuleList(
            [
                nn.Conv2d(hidden * 2, hidden * 2, kernel_size=(1, 3), padding=(0, 1))
                for _ in range(4)
            ]
            + [nn.Conv2d(hidden * 2, 1, kernel_size=(1, 3), padding=(0, 1))]
        )

    def forward(self, band: torch.Tensor):
        # band: (B, F_band, T) -> treat as single-channel 2D image (B, 1, F_band, T)
        x = band.unsqueeze(1)
        feats = []
        h = F.leaky_relu(self.entry(x), 0.2)
        feats.append(h)
        for conv in self.down:
            h = F.leaky_relu(conv(h), 0.2)
            feats.append(h)
        for i, conv in enumerate(self.out_convs):
            h = conv(h)
            if i < len(self.out_convs) - 1:
                h = F.leaky_relu(h, 0.2)
            feats.append(h)
        return h, feats


class MultiBandSubDiscriminator(nn.Module):
    def __init__(self, band_width: int):
        super().__init__()
        self.sd = SegmentDiscriminator(in_channels=band_width)
        self.dd = DetailDiscriminator()

    def forward(self, band: torch.Tensor):
        sd_score, sd_feats = self.sd(band)
        dd_score, dd_feats = self.dd(band)
        return sd_score, dd_score, sd_feats + dd_feats


class MultiBandDiscriminator(nn.Module):
    """3 sub-discriminators over low/mid/high mel bands (Section 2.2):
    [0:60], [30:90], [60:120] of a 120-bin mel-spectrogram."""

    def __init__(self, mel_dim: int = 120):
        super().__init__()
        assert mel_dim == 120, "band boundaries are defined for the paper's 120-bin mel"
        self.bands = [(0, 60), (30, 90), (60, 120)]
        self.sub_discs = nn.ModuleList(
            [MultiBandSubDiscriminator(hi - lo) for lo, hi in self.bands]
        )

    def forward(self, mel: torch.Tensor):
        # mel: (B, T, mel_dim) -> (B, mel_dim, T) channel-first per band.
        mel_t = mel.transpose(1, 2)
        sd_scores, dd_scores, all_feats = [], [], []
        for (lo, hi), sub in zip(self.bands, self.sub_discs):
            band = mel_t[:, lo:hi, :]
            sd_score, dd_score, feats = sub(band)
            sd_scores.append(sd_score)
            dd_scores.append(dd_score)
            all_feats.extend(feats)
        return sd_scores, dd_scores, all_feats


# ----------------------------- Full GAN --------------------------------------


class XiaoiceSing2(nn.Module):
    def __init__(
        self, num_phonemes: int = 40, dim: int = 32, mel_dim: int = 120, max_frames: int = 48
    ):
        super().__init__()
        self.generator = XiaoiceSing2Generator(
            num_phonemes=num_phonemes, dim=dim, mel_dim=mel_dim, max_frames=max_frames
        )
        self.discriminator = MultiBandDiscriminator(mel_dim=mel_dim)

    def forward(self, phoneme_ids: torch.Tensor, pitch: torch.Tensor, score_duration: torch.Tensor):
        mel, logf0, vuv, log_dur_pred = self.generator(phoneme_ids, pitch, score_duration)
        sd_scores, dd_scores, _feats = self.discriminator(mel)
        return mel, logf0, vuv, log_dur_pred, sd_scores, dd_scores


def build_xiaoicesing2() -> XiaoiceSing2:
    return XiaoiceSing2(num_phonemes=40, dim=32, mel_dim=120, max_frames=48)


def example_input_xiaoicesing2():
    batch, T = 2, 12
    phoneme_ids = torch.randint(0, 40, (batch, T))
    pitch = torch.rand(batch, T) * 2.0 + 3.0  # toy log-F0-ish values
    score_duration = torch.rand(batch, T) * 2.0 + 1.0
    return phoneme_ids, pitch, score_duration


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    ("XiaoiceSing2", "build_xiaoicesing2", "example_input_xiaoicesing2", 2022, "REIMPL"),
]
