"""Matcha-TTS compact conditional-flow-matching acoustic model.

Mehta et al., 2023/2024, "Matcha-TTS: A Fast TTS Architecture with Conditional
Flow Matching".  Matcha-TTS is a probabilistic non-autoregressive TTS model with
a text encoder, duration/alignment conditioning, and an ODE decoder trained with
optimal-transport conditional flow matching.  This compact reconstruction traces
the inference-time architecture: token encoder, length regulator, time/noise
conditioning, and a 1-D convolutional U-Net flow predictor.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class FlowBlock(nn.Module):
    """Residual temporal convolution block for Matcha's decoder."""

    def __init__(self, channels: int) -> None:
        """Initialize the block.

        Parameters
        ----------
        channels:
            Number of temporal channels.
        """
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 5, padding=2)
        self.norm = nn.GroupNorm(4, channels)
        self.gate = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply gated residual conditioning.

        Parameters
        ----------
        x:
            Decoder activations.
        cond:
            Broadcast conditioning activations.

        Returns
        -------
        Tensor
            Updated activations.
        """
        h = F.silu(self.norm(self.conv(x + cond)))
        return x + h * torch.sigmoid(self.gate(cond))


class MatchaTTS(nn.Module):
    """Compact Matcha-TTS acoustic flow predictor."""

    def __init__(self, vocab: int = 64, hidden: int = 32, mel_bins: int = 16) -> None:
        """Initialize text encoder and flow decoder.

        Parameters
        ----------
        vocab:
            Text-token vocabulary size.
        hidden:
            Hidden width.
        mel_bins:
            Mel-spectrogram channel count.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.encoder = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.duration = nn.Linear(hidden, 1)
        self.mel_in = nn.Conv1d(mel_bins, hidden, 1)
        self.cond = nn.Conv1d(hidden, hidden, 1)
        self.time = nn.Linear(1, hidden)
        self.down = FlowBlock(hidden)
        self.mid = FlowBlock(hidden)
        self.up = FlowBlock(hidden)
        self.out = nn.Conv1d(hidden, mel_bins, 1)

    def forward(self, tokens: Tensor, noise_mel: Tensor, time: Tensor) -> Tensor:
        """Predict the conditional flow vector field.

        Parameters
        ----------
        tokens:
            Text tokens with shape ``(batch, text_len)``.
        noise_mel:
            Noisy mel tensor with shape ``(batch, mel_bins, frames)``.
        time:
            Flow time tensor with shape ``(batch, 1)``.

        Returns
        -------
        Tensor
            Predicted mel velocity field.
        """
        text, _ = self.encoder(self.embed(tokens))
        durations = torch.sigmoid(self.duration(text)).transpose(1, 2)
        cond = F.interpolate(
            text.transpose(1, 2) * durations, size=noise_mel.shape[-1], mode="nearest"
        )
        cond = self.cond(cond) + self.time(time).unsqueeze(-1)
        x = self.mel_in(noise_mel)
        skip = self.down(x, cond)
        x = F.avg_pool1d(skip, 2)
        c = F.avg_pool1d(cond, 2)
        x = self.mid(x, c)
        x = F.interpolate(x, size=skip.shape[-1], mode="nearest") + skip
        return self.out(self.up(x, cond))


def build() -> nn.Module:
    """Build compact Matcha-TTS.

    Returns
    -------
    nn.Module
        Random-initialized Matcha-TTS reconstruction.
    """
    return MatchaTTS().eval()


def example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return text tokens, noised mel, and flow time.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Example model inputs.
    """
    return torch.randint(0, 64, (1, 10)), torch.randn(1, 16, 24), torch.rand(1, 1)


MENAGERIE_ENTRIES = [
    ("Matcha-TTS", "build", "example_input", "2023", "DE"),
]
