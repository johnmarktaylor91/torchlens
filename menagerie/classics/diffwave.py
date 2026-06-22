"""DiffWave: diffusion model for audio synthesis.

Kong et al., ICLR 2021, arXiv:2009.09761.  DiffWave is a non-autoregressive
WaveNet-like diffusion vocoder with sinusoidal diffusion-step embeddings,
mel-spectrogram conditioning, dilated gated residual blocks, skip aggregation,
and a waveform noise-prediction head.  This compact version keeps those blocks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionEmbedding(nn.Module):
    """Sinusoidal diffusion-step embedding followed by projections."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize embedding projections.

        Parameters
        ----------
        dim:
            Embedding width.
        """

        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.dim = dim

    def forward(self, step: torch.Tensor) -> torch.Tensor:
        """Embed diffusion steps.

        Parameters
        ----------
        step:
            Diffusion step tensor.

        Returns
        -------
        torch.Tensor
            Step embedding.
        """

        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=step.device) * -(math.log(10000.0) / max(1, half - 1))
        )
        args = step.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return self.proj(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))


class DiffWaveBlock(nn.Module):
    """Dilated gated residual block with conditioner projection."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize a residual block.

        Parameters
        ----------
        channels:
            Residual channel count.
        dilation:
            Dilated convolution factor.
        """

        super().__init__()
        self.diff_proj = nn.Linear(32, channels)
        self.cond_proj = nn.Conv1d(16, 2 * channels, 1)
        self.dilated = nn.Conv1d(channels, 2 * channels, 3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(channels, 2 * channels, 1)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, diff: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply gated residual update and return skip output.

        Parameters
        ----------
        x:
            Residual audio features.
        cond:
            Upsampled conditioner.
        diff:
            Diffusion embedding.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated residual features and skip features.
        """

        y = x + self.diff_proj(diff).unsqueeze(-1)
        gate, filt = (self.dilated(y) + self.cond_proj(cond)).chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        residual, skip = self.out(y).chunk(2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CompactDiffWave(nn.Module):
    """Compact conditional DiffWave noise predictor."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize compact DiffWave.

        Parameters
        ----------
        channels:
            Residual channel count.
        """

        super().__init__()
        self.input = nn.Conv1d(1, channels, 1)
        self.diffusion = DiffusionEmbedding()
        self.upsample = nn.Sequential(nn.ConvTranspose1d(8, 16, 4, stride=2, padding=1), nn.ReLU())
        self.blocks = nn.ModuleList(
            [DiffWaveBlock(channels, dilation) for dilation in (1, 2, 4, 8)]
        )
        self.output = nn.Sequential(
            nn.ReLU(), nn.Conv1d(channels, channels, 1), nn.ReLU(), nn.Conv1d(channels, 1, 1)
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict waveform noise.

        Parameters
        ----------
        inputs:
            Tuple ``(audio, mel_conditioner, diffusion_step)``.

        Returns
        -------
        torch.Tensor
            Predicted noise waveform.
        """

        audio, mel, step = inputs
        x = self.input(audio.unsqueeze(1))
        cond = F.interpolate(
            self.upsample(mel), size=x.shape[-1], mode="linear", align_corners=False
        )
        diff = self.diffusion(step)
        skips = []
        for block in self.blocks:
            x, skip = block(x, cond, diff)
            skips.append(skip)
        return self.output(torch.stack(skips, dim=0).sum(dim=0) / math.sqrt(len(skips))).squeeze(1)


def build() -> nn.Module:
    """Build compact DiffWave.

    Returns
    -------
    nn.Module
        Random-init DiffWave reconstruction.
    """

    return CompactDiffWave()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create audio, mel conditioner, and diffusion step.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        DiffWave inputs.
    """

    return torch.randn(1, 64), torch.randn(1, 8, 16), torch.tensor([7])


MENAGERIE_ENTRIES = [
    ("diffwave", "build", "example_input", "2021", "AUDIO"),
    ("DiffWave_lmnt", "build", "example_input", "2021", "AUDIO"),
]
