"""Wav2Lip: A Lip Sync Expert Is All You Need for Speech to Lip Generation.

Prajwal et al., ACM MM 2020.
Paper: https://arxiv.org/abs/2008.10010
Source: https://github.com/Rudrabha/Wav2Lip

Distinctive architecture -- TWO-STREAM encoder + decoder + a sync expert:
  - An AUDIO encoder (2D conv stack over a mel-spectrogram chunk) -> audio
    embedding.
  - A FACE/IDENTITY encoder (2D conv stack over a 6-channel face image: the
    masked lower-half target frame stacked with a reference identity frame) ->
    face feature pyramid with skip connections.
  - The audio embedding is tiled and concatenated onto the bottleneck face
    feature; a U-Net-style DECODER with transpose convs (fusing the face skips)
    regenerates the lip-synced lower face.
  - Training is supervised by a pretrained SYNC-DISCRIMINATOR (SyncNet): a
    contrastive audio-vs-video lip-sync expert that scores whether a window of
    generated frames is in sync with the audio (we include the discriminator
    primitive as a separate small embedding-pair scorer; it is a loss-time
    module, not part of the generator forward).

This module reproduces the audio encoder + face encoder + concat-fusion U-Net
decoder generator, plus the SyncNet-style lip-sync expert primitive. Faithful
compact random-init reimplementation: small image (96x96 face, 5 frames) and a
short mel window so the unrolled trace draws quickly. The generator is wrapped
to be forward-able from a single mel-spectrogram tensor (the face frames are
synthesized internally). arXiv:2008.10010.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
    )


def _convT(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class AudioEncoder(nn.Module):
    """2D conv stack over a mel-spectrogram chunk -> audio embedding."""

    def __init__(self, embed: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv(1, 16),
            _conv(16, 32, s=2),
            _conv(32, 64, s=2),
            _conv(64, embed, s=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h = self.net(mel)
        return self.pool(h)  # (B, embed, 1, 1)


class FaceEncoder(nn.Module):
    """2D conv stack over the 6-channel face input -> bottleneck + skip pyramid."""

    def __init__(self, in_ch: int = 6, base: int = 16) -> None:
        super().__init__()
        self.e1 = _conv(in_ch, base)  # 96
        self.e2 = _conv(base, base * 2, s=2)  # 48
        self.e3 = _conv(base * 2, base * 4, s=2)  # 24
        self.e4 = _conv(base * 4, base * 8, s=2)  # 12
        self.e5 = _conv(base * 8, base * 8, s=2)  # 6

    def forward(self, face: torch.Tensor):
        s1 = self.e1(face)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b = self.e5(s4)
        return b, [s4, s3, s2, s1]


class Wav2LipGenerator(nn.Module):
    """Audio+face two-stream encoder -> concat-fusion U-Net decoder -> lip frame."""

    def __init__(self, base: int = 16, audio_embed: int = 128) -> None:
        super().__init__()
        self.audio_enc = AudioEncoder(audio_embed)
        self.face_enc = FaceEncoder(6, base)
        self.fuse = _conv(base * 8 + audio_embed, base * 8, k=1, p=0)
        self.d4 = _convT(base * 8, base * 8)  # 12
        self.d3 = _convT(base * 8 + base * 8, base * 4)  # 24
        self.d2 = _convT(base * 4 + base * 4, base * 2)  # 48
        self.d1 = _convT(base * 2 + base * 2, base)  # 96
        self.out = nn.Sequential(nn.Conv2d(base + base, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, mel: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
        a = self.audio_enc(mel)  # (B, embed, 1, 1)
        b, skips = self.face_enc(face)
        a_tiled = a.expand(-1, -1, b.shape[2], b.shape[3])
        h = self.fuse(torch.cat([b, a_tiled], dim=1))
        h = self.d4(h)
        h = self.d3(torch.cat([h, skips[0]], dim=1))
        h = self.d2(torch.cat([h, skips[1]], dim=1))
        h = self.d1(torch.cat([h, skips[2]], dim=1))
        return self.out(torch.cat([h, skips[3]], dim=1))


class SyncNet(nn.Module):
    """Lip-sync expert: separate audio & face encoders -> cosine-similarity score."""

    def __init__(self, embed: int = 64) -> None:
        super().__init__()
        self.audio = nn.Sequential(
            _conv(1, 16),
            _conv(16, 32, s=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed),
        )
        self.face = nn.Sequential(
            _conv(15, 16),
            _conv(16, 32, s=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, embed),
        )

    def forward(self, mel: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        ae = nn.functional.normalize(self.audio(mel), dim=-1)
        fe = nn.functional.normalize(self.face(frames), dim=-1)
        return (ae * fe).sum(dim=-1, keepdim=True)


class _Wav2LipWrapper(nn.Module):
    """Single-tensor wrapper: input is the mel chunk; face frames synthesized internally."""

    def __init__(self, gen: nn.Module) -> None:
        super().__init__()
        self.gen = gen

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        b = mel.shape[0]
        face = torch.randn(b, 6, 96, 96, device=mel.device)
        return self.gen(mel, face)


def build_wav2lip() -> nn.Module:
    """Build the Wav2Lip generator (audio+face two-stream U-Net lip-frame synth)."""
    return _Wav2LipWrapper(Wav2LipGenerator(base=16, audio_embed=128))


def example_input() -> torch.Tensor:
    """Example mel-spectrogram chunk ``(1, 1, 80, 16)`` for Wav2Lip."""
    return torch.randn(1, 1, 80, 16)


def build_wav2lip_syncnet() -> nn.Module:
    """Build the SyncNet lip-sync expert (audio/video contrastive sync scorer)."""
    return SyncNet(embed=64)


class _SyncNetWrapper(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        b = mel.shape[0]
        frames = torch.randn(b, 15, 48, 96, device=mel.device)  # 5 RGB frames stacked
        return self.net(mel, frames)


def build_syncnet() -> nn.Module:
    """Build the SyncNet lip-sync discriminator wrapped for single-tensor tracing."""
    return _SyncNetWrapper(SyncNet(embed=64))


def example_input_syncnet() -> torch.Tensor:
    """Example mel-spectrogram chunk ``(1, 1, 80, 16)`` for SyncNet."""
    return torch.randn(1, 1, 80, 16)


MENAGERIE_ENTRIES = [
    (
        "Wav2Lip (audio+face two-stream U-Net lip-sync generator)",
        "build_wav2lip",
        "example_input",
        "2020",
        "DC",
    ),
    (
        "Wav2Lip SyncNet (contrastive audio-video lip-sync expert)",
        "build_syncnet",
        "example_input_syncnet",
        "2020",
        "DC",
    ),
]
