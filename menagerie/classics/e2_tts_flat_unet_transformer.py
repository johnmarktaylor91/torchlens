"""E2 TTS Flat-UNet Transformer text-to-speech backbone.

Paper: E2 TTS and the follow-on F5-TTS line use non-autoregressive flow matching for
speech generation.  The E2 architecture is commonly exposed as a Flat-UNet Transformer:
same-resolution encoder/bottleneck/decoder transformer blocks with skip fusion, text
conditioning, time conditioning, noised mel input, and masked reference-audio conditioning.

This compact random-init reconstruction preserves that flat U-Net Transformer topology.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from menagerie.classics.f5_tts_dit import DiTBlock, TimestepMLP


class FlatUNetTransformer(nn.Module):
    """Compact Flat-UNet Transformer for conditional speech flow matching."""

    def __init__(self, mel_bins: int = 16, vocab: int = 64, dim: int = 48) -> None:
        """Initialize the compact flat U-Net Transformer.

        Parameters
        ----------
        mel_bins:
            Number of compact mel channels.
        vocab:
            Text vocabulary size.
        dim:
            Transformer width.
        """

        super().__init__()
        self.text_embed = nn.Embedding(vocab, dim)
        self.in_proj = nn.Linear(mel_bins * 2 + 1, dim)
        self.time_mlp = TimestepMLP(dim)
        self.enc = nn.ModuleList([DiTBlock(dim) for _ in range(2)])
        self.mid = DiTBlock(dim)
        self.skip_fuse = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(2)])
        self.dec = nn.ModuleList([DiTBlock(dim) for _ in range(2)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, mel_bins)

    def forward(
        self,
        noisy_mel: torch.Tensor,
        text_tokens: torch.Tensor,
        cond_mel: torch.Tensor,
        cond_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict mel flow velocity with flat U-Net skip connections.

        Parameters
        ----------
        noisy_mel:
            Noised mel tensor of shape ``(batch, frames, mel_bins)``.
        text_tokens:
            Token IDs aligned to mel frames.
        cond_mel:
            Reference mel tensor.
        cond_mask:
            Reference mask of shape ``(batch, frames, 1)``.
        t:
            Flow-matching time tensor.

        Returns
        -------
        torch.Tensor
            Predicted velocity.
        """

        x = torch.cat([noisy_mel, cond_mel * cond_mask, cond_mask], dim=-1)
        x = self.in_proj(x) + self.text_embed(text_tokens)
        time = self.time_mlp(t)
        skips = []
        for block in self.enc:
            x = block(x, time)
            skips.append(x)
        x = self.mid(x, time)
        for block, fuse, skip in zip(self.dec, self.skip_fuse, reversed(skips), strict=True):
            x = fuse(torch.cat([x, skip], dim=-1))
            x = block(x, time)
        return self.out(self.norm(x))


def build_e2_tts_flat_unet_transformer() -> nn.Module:
    """Build compact E2 TTS Flat-UNet Transformer.

    Returns
    -------
    nn.Module
        Random-init compact E2 TTS model.
    """

    return FlatUNetTransformer()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact E2 TTS inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Noisy mel, text tokens, reference mel, mask, and time.
    """

    frames = 12
    noisy = torch.randn(1, frames, 16)
    text = torch.randint(0, 64, (1, frames))
    cond = torch.randn(1, frames, 16)
    mask = torch.zeros(1, frames, 1)
    mask[:, :5] = 1.0
    t = torch.tensor([0.45])
    return noisy, text, cond, mask, t


build = build_e2_tts_flat_unet_transformer

MENAGERIE_ENTRIES = [
    (
        "e2_tts_flat_unet_transformer",
        "build_e2_tts_flat_unet_transformer",
        "example_input",
        "2024",
        "E6",
    ),
]
