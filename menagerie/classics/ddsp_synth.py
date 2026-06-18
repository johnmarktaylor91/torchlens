"""DDSP differentiable audio synthesizer, 2020, Engel et al.

Paper: Engel 2020, "DDSP: Differentiable Digital Signal Processing." This
minimal harmonic-plus-noise synthesizer keeps the differentiable oscillator,
harmonic amplitudes, and filtered-noise path, omitting reverb and an audio
autoencoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DDSPSynth(nn.Module):
    """Differentiable harmonic oscillator bank plus filtered noise synthesizer."""

    def __init__(self, n_samples: int = 64, n_harmonics: int = 6, hidden: int = 32) -> None:
        """Initialize the small control network and filter parameters.

        Parameters
        ----------
        n_samples
            Number of audio samples per example.
        n_harmonics
            Number of harmonic oscillators.
        hidden
            Width of the control MLP.
        """
        super().__init__()
        self.n_samples = n_samples
        self.n_harmonics = n_harmonics
        self.encoder = nn.Sequential(
            nn.Linear(n_samples, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_harmonics + 2),
        )
        self.noise_filter = nn.Parameter(torch.randn(1, 1, 9) * 0.05)

    def forward(self, audio_features: Tensor) -> Tensor:
        """Synthesize a short waveform from conditioning features.

        Parameters
        ----------
        audio_features
            Conditioning tensor of shape ``(batch, n_samples)``.

        Returns
        -------
        Tensor
            Synthesized waveform with shape ``(batch, n_samples)``.
        """
        controls = self.encoder(audio_features)
        f0 = torch.sigmoid(controls[:, :1]) * 0.18 + 0.02
        loudness = torch.sigmoid(controls[:, 1:2])
        harmonic_weights = torch.softmax(controls[:, 2:], dim=-1)
        phase_step = 2.0 * torch.pi * f0
        base_phase = torch.cumsum(phase_step.expand(-1, self.n_samples), dim=-1)
        harmonic_numbers = torch.arange(
            1,
            self.n_harmonics + 1,
            device=audio_features.device,
            dtype=audio_features.dtype,
        )
        phases = base_phase.unsqueeze(-1) * harmonic_numbers
        harmonic = (torch.sin(phases) * harmonic_weights.unsqueeze(1)).sum(dim=-1)
        excitation = torch.randn(
            audio_features.shape[0], 1, self.n_samples, device=audio_features.device
        )
        kernel = torch.softmax(self.noise_filter, dim=-1)
        filtered_noise = F.conv1d(excitation, kernel, padding=4).squeeze(1)
        return loudness * harmonic + 0.05 * filtered_noise


def build() -> nn.Module:
    """Build a small DDSP synthesizer.

    Returns
    -------
    nn.Module
        Configured ``DDSPSynth`` instance.
    """
    return DDSPSynth()


def example_input() -> Tensor:
    """Create a conditioning feature example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 64)``.
    """
    return torch.randn(1, 64)


MENAGERIE_ENTRIES = [
    ("DDSP differentiable audio synthesizer", "build", "example_input", "2020", "CH-D")
]
