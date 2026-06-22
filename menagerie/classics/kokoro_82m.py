"""Kokoro-82M compact text-to-speech reconstruction.

Paper/Model: Kokoro-82M, 2024; architecture derived from StyleTTS2 and
iSTFTNet.

Kokoro is a lightweight decoder-only TTS model.  This reconstruction keeps the
phoneme encoder, style/prosody conditioning, duration expansion, and iSTFTNet
vocoder primitive that predicts magnitude and phase before inverse-STFT-style
waveform synthesis.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class KokoroTTS(nn.Module):
    """Compact Kokoro-style non-autoregressive TTS decoder."""

    def __init__(self, vocab: int = 96, dim: int = 48, mel_bins: int = 16) -> None:
        """Initialize phoneme encoder, style adaptor, and iSTFTNet heads."""

        super().__init__()
        self.phoneme = nn.Embedding(vocab, dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True),
            num_layers=1,
        )
        self.style = nn.Linear(16, dim)
        self.duration = nn.Linear(dim, 1)
        self.prosody = nn.Linear(dim, dim)
        self.mag = nn.Conv1d(dim, mel_bins, 3, padding=1)
        self.phase = nn.Conv1d(dim, mel_bins, 3, padding=1)
        self.wave = nn.Conv1d(mel_bins, 1, 5, padding=2)

    def forward(self, phonemes: Tensor, style: Tensor) -> Tensor:
        """Synthesize a waveform proxy from phoneme ids and style vector."""

        x = self.phoneme(phonemes) + self.style(style).unsqueeze(1)
        x = self.encoder(x)
        durations = F.softplus(self.duration(x))
        expanded = (x * durations).repeat_interleave(2, dim=1)
        prosody = torch.tanh(self.prosody(expanded))
        feat = prosody.transpose(1, 2)
        magnitude = F.softplus(self.mag(feat))
        phase = torch.sin(self.phase(feat))
        spectrum = magnitude * phase
        return torch.tanh(self.wave(spectrum)).squeeze(1)


def build() -> nn.Module:
    """Build the compact Kokoro-82M TTS model."""

    return KokoroTTS().eval()


def example_input() -> tuple[Tensor, Tensor]:
    """Return phoneme ids and style vector."""

    return torch.randint(0, 96, (1, 12)), torch.randn(1, 16)


MENAGERIE_ENTRIES = [
    ("Kokoro-82M", "build", "example_input", "2024", "AUDIO"),
]
