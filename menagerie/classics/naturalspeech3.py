"""NaturalSpeech 3: factorized codec and diffusion text-to-speech.

Paper: Ju et al. 2024, "NaturalSpeech 3: Zero-Shot Speech Synthesis with
Factorized Codec and Diffusion Models".

The compact reconstruction exposes the characteristic pipeline: FACodec-style
factorized vector quantization into content, prosody, timbre, and detail
subspaces, followed by factorized denoising heads and waveform reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedCodec(nn.Module):
    """Small FACodec-style factorized speech codec."""

    def __init__(self, dim: int = 32, codebooks: int = 4, code_dim: int = 8) -> None:
        """Initialize waveform encoder, subspace codebooks, and decoder.

        Parameters
        ----------
        dim:
            Hidden convolution width.
        codebooks:
            Number of factorized attribute subspaces.
        code_dim:
            Width of each subspace code.
        """

        super().__init__()
        self.encoder = nn.Conv1d(1, dim, kernel_size=5, padding=2)
        self.to_codes = nn.ModuleList([nn.Conv1d(dim, code_dim, 1) for _ in range(codebooks)])
        self.codebooks = nn.Parameter(torch.randn(codebooks, 8, code_dim) * 0.02)
        self.from_codes = nn.Conv1d(codebooks * code_dim, dim, 1)
        self.decoder = nn.Conv1d(dim, 1, kernel_size=5, padding=2)

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode waveform attributes and reconstruct audio.

        Parameters
        ----------
        wav:
            Waveform tensor with shape ``(batch, samples)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Reconstructed waveform and concatenated factor codes.
        """

        hidden = F.silu(self.encoder(wav.unsqueeze(1)))
        quantized = []
        for idx, head in enumerate(self.to_codes):
            logits = head(hidden).transpose(1, 2)
            dist = (
                (logits.unsqueeze(2) - self.codebooks[idx].view(1, 1, 8, -1)).square().sum(dim=-1)
            )
            weights = torch.softmax(-dist, dim=-1)
            quantized.append(torch.matmul(weights, self.codebooks[idx]).transpose(1, 2))
        codes = torch.cat(quantized, dim=1)
        recon = self.decoder(F.silu(self.from_codes(codes))).squeeze(1)
        return recon, codes


class NaturalSpeech3(nn.Module):
    """Compact NaturalSpeech 3 factorized codec-diffusion stack."""

    def __init__(self) -> None:
        """Initialize codec and per-factor diffusion denoisers."""

        super().__init__()
        self.codec = FactorizedCodec()
        self.text = nn.Embedding(32, 32)
        self.time = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.denoisers = nn.ModuleList([nn.Conv1d(8 + 64, 8, 3, padding=1) for _ in range(4)])

    def forward(
        self, wav: torch.Tensor, tokens: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Denoise factorized codec streams conditioned on text and diffusion time.

        Parameters
        ----------
        wav:
            Prompt waveform tensor.
        tokens:
            Text token ids.
        timestep:
            Diffusion timestep.

        Returns
        -------
        torch.Tensor
            Reconstructed waveform from denoised factor codes.
        """

        _, codes = self.codec(wav)
        text = self.text(tokens).mean(dim=1)
        cond_vec = torch.cat((text, self.time(timestep[:, None])), dim=-1)
        cond = cond_vec.unsqueeze(-1).expand(-1, -1, codes.shape[-1])
        chunks = torch.chunk(codes, 4, dim=1)
        denoised = [
            head(torch.cat([chunk, cond], dim=1))
            for head, chunk in zip(self.denoisers, chunks, strict=True)
        ]
        merged = torch.cat(denoised, dim=1)
        hidden = F.silu(self.codec.from_codes(merged))
        return self.codec.decoder(hidden).squeeze(1)


def build() -> nn.Module:
    """Build the compact NaturalSpeech 3 model.

    Returns
    -------
    nn.Module
        Random-initialized NaturalSpeech 3 module.
    """

    return NaturalSpeech3()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create waveform and text-token prompts.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Waveform, token, and timestep tensors.
    """

    return torch.randn(1, 64), torch.randint(0, 32, (1, 6)), torch.tensor([0.45])


MENAGERIE_ENTRIES = [
    ("NaturalSpeech3", "build", "example_input", "2024", "E6"),
]
