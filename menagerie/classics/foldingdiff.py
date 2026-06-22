"""FoldingDiff: diffusion over protein backbone angles.

Paper: "Protein structure generation via folding diffusion", Wu et al., 2022.

The compact reconstruction denoises per-residue backbone angle features with a
BERT-style bidirectional Transformer encoder and timestep conditioning.  Inputs
are six angle/bond features per residue rather than 3-D coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FoldingDiffCompact(nn.Module):
    """Compact BERT denoiser for backbone angle diffusion."""

    def __init__(self, d_model: int = 32, layers: int = 2) -> None:
        """Initialize angle, position, and timestep-conditioned encoder."""

        super().__init__()
        self.angle_proj = nn.Linear(12, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.pos = nn.Parameter(torch.randn(1, 32, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=64, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, 6)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """Predict denoised angle residuals."""

        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        time = torch.full((angles.shape[0], 1), 0.5, device=angles.device, dtype=angles.dtype)
        x = self.angle_proj(sincos) + self.time_proj(time).unsqueeze(1)
        x = x + self.pos[:, : angles.shape[1]]
        return self.head(self.encoder(x))


def build() -> nn.Module:
    """Build compact FoldingDiff."""

    return FoldingDiffCompact()


def example_input() -> torch.Tensor:
    """Return noisy residue angle features."""

    return torch.randn(1, 16, 6)


MENAGERIE_ENTRIES = [("FoldingDiff", "build", "example_input", "2022", "E7")]
