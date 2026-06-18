"""SampleRNN, 2016, Mehri et al., "SampleRNN".

Multi-rate recurrent tiers condition a bottom sample-level predictor for raw
audio. This compact module uses two frame GRU tiers and a categorical 8-bit
sample head.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SampleRNN(nn.Module):
    """Hierarchical raw-audio model with frame and sample tiers."""

    def __init__(
        self,
        quantization: int = 256,
        embed_size: int = 16,
        hidden_size: int = 48,
        frame_size: int = 16,
    ) -> None:
        """Initialize hierarchical recurrent and sample-level predictors.

        Parameters
        ----------
        quantization:
            Number of discrete sample values.
        embed_size:
            Sample embedding size.
        hidden_size:
            Frame-tier hidden size.
        frame_size:
            Samples per high-level frame.
        """
        super().__init__()
        self.frame_size = frame_size
        self.embedding = nn.Embedding(quantization, embed_size)
        self.frame_gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.sample_mlp = nn.Sequential(
            nn.Linear(embed_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, quantization),
        )

    def forward(self, audio: Tensor) -> Tensor:
        """Predict per-sample categorical logits from 8-bit audio.

        Parameters
        ----------
        audio:
            Unsigned integer samples with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Per-sample logits with shape ``(batch, time, quantization)``.
        """
        tokens = audio.long()
        embedded = self.embedding(tokens)
        batch, steps, features = embedded.shape
        n_frames = steps // self.frame_size
        framed = embedded[:, : n_frames * self.frame_size].reshape(
            batch, n_frames, self.frame_size, features
        )
        frame_inputs = framed.mean(dim=2)
        frame_hidden, _ = self.frame_gru(frame_inputs)
        conditioning = frame_hidden.unsqueeze(2).expand(-1, -1, self.frame_size, -1)
        conditioning = conditioning.reshape(batch, n_frames * self.frame_size, -1)
        conditioned = torch.cat((embedded[:, : n_frames * self.frame_size], conditioning), dim=-1)
        return self.sample_mlp(conditioned)


def build() -> nn.Module:
    """Build a compact SampleRNN.

    Returns
    -------
    nn.Module
        Random-initialized SampleRNN.
    """
    return SampleRNN()


def example_input() -> Tensor:
    """Return example 8-bit audio samples.

    Returns
    -------
    Tensor
        UInt8 tensor with shape ``(1, 4096)``.
    """
    return torch.randint(0, 256, (1, 4096), dtype=torch.uint8)


MENAGERIE_ENTRIES = [("SampleRNN Hierarchical Audio Model", "build", "example_input", "2016", "DE")]
