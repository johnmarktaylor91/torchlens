"""RFdiffusion compact protein-backbone diffusion model.

Paper: Watson et al. 2023, "De novo design of protein structure and function
with RFdiffusion." The full model adapts RoseTTAFold-style SE(3)-aware pair and
MSA reasoning to denoise protein backbone frames. This compact torch classic
keeps the faithful core: residue embeddings, pair-distance geometry, timestep
conditioning, pair-biased residue attention, and coordinate residual prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairBiasedResidueBlock(nn.Module):
    """Residue transformer block with pair-derived attention bias."""

    def __init__(self, dim: int, pair_dim: int, heads: int = 4) -> None:
        """Initialize pair-biased residue attention.

        Parameters
        ----------
        dim:
            Residue feature width.
        pair_dim:
            Pair feature width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.pair_bias = nn.Linear(pair_dim, heads)
        self.pair_update = nn.Sequential(nn.Linear(pair_dim + dim, pair_dim), nn.SiLU())
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 3), nn.SiLU(), nn.Linear(dim * 3, dim)
        )

    def forward(
        self, residue: torch.Tensor, pair: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply pair-biased attention and pair updates.

        Parameters
        ----------
        residue:
            Residue features ``(B, L, D)``.
        pair:
            Pair features ``(B, L, L, P)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated residue and pair features.
        """

        bias = (
            self.pair_bias(pair)
            .permute(0, 3, 1, 2)
            .reshape(residue.shape[0] * self.heads, residue.shape[1], residue.shape[1])
        )
        y, _ = self.attn(self.norm(residue), self.norm(residue), self.norm(residue), attn_mask=bias)
        residue = residue + y
        pair_context = residue[:, :, None, :] + residue[:, None, :, :]
        pair = pair + self.pair_update(torch.cat([pair, pair_context], dim=-1))
        return residue + self.ff(residue), pair


class RFDiffusionCompact(nn.Module):
    """Compact RFdiffusion-style protein backbone denoiser."""

    def __init__(self, length: int = 12, dim: int = 48, pair_dim: int = 24) -> None:
        """Initialize residue, pair, timestep, and coordinate heads.

        Parameters
        ----------
        length:
            Number of residues in the compact chain.
        dim:
            Residue feature width.
        pair_dim:
            Pair feature width.
        """

        super().__init__()
        self.length = length
        self.residue_embed = nn.Linear(3, dim)
        self.time_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.pair_embed = nn.Sequential(
            nn.Linear(4, pair_dim), nn.SiLU(), nn.Linear(pair_dim, pair_dim)
        )
        self.blocks = nn.ModuleList([PairBiasedResidueBlock(dim, pair_dim) for _ in range(2)])
        self.coord_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 3)
        )

    def forward(self, noisy_ca: torch.Tensor) -> torch.Tensor:
        """Predict backbone coordinate denoising residuals.

        Parameters
        ----------
        noisy_ca:
            Noisy CA coordinates of shape ``(B, L, 3)``.

        Returns
        -------
        torch.Tensor
            Denoised CA coordinates with shape ``(B, L, 3)``.
        """

        time = noisy_ca.mean(dim=(1, 2), keepdim=True)
        residue = self.residue_embed(noisy_ca) + self.time_embed(
            time.reshape(noisy_ca.shape[0], 1)
        ).unsqueeze(1)
        rel = noisy_ca[:, :, None, :] - noisy_ca[:, None, :, :]
        dist = rel.norm(dim=-1, keepdim=True)
        seq_sep = torch.arange(self.length, device=noisy_ca.device, dtype=noisy_ca.dtype)
        sep = (seq_sep[:, None] - seq_sep[None, :]).abs().view(1, self.length, self.length, 1)
        pair = self.pair_embed(torch.cat([rel[..., :3], dist + sep / self.length], dim=-1))
        for block in self.blocks:
            residue, pair = block(residue, pair)
        return noisy_ca + self.coord_head(residue)


def build() -> nn.Module:
    """Build compact RFdiffusion.

    Returns
    -------
    nn.Module
        Random-initialized RFdiffusion-style denoiser.
    """

    return RFDiffusionCompact()


def example_input() -> torch.Tensor:
    """Create a simple noisy backbone-coordinate tensor.

    Returns
    -------
    torch.Tensor
        Coordinates of shape ``(1, 12, 3)``.
    """

    return torch.randn(1, 12, 3)


MENAGERIE_ENTRIES = [("rfdiffusion", "build", "example_input", "2023", "DC")]
