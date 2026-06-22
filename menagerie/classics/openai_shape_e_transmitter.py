"""OpenAI Shap-E transmitter: asset encoder to implicit-function parameters.

Paper: Jun and Nichol 2023, "Shap-E: Generating Conditional 3D Implicit
Functions".

Shap-E first trains an encoder that maps 3D assets to the parameters of an
implicit function renderable as textured meshes and NeRFs.  This compact
transmitter encodes point samples plus colors into tri-plane-like parameters and
evaluates a small implicit field at query coordinates.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ShapETransmitter(nn.Module):
    """Compact Shap-E asset-to-implicit transmitter."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize asset encoder, parameter projector, and field head.

        Parameters
        ----------
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.asset = nn.Linear(6, dim)
        self.param = nn.Linear(dim, dim)
        self.query = nn.Linear(3, dim)
        self.field = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, 4)
        )

    def forward(self, samples: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        """Encode colored samples and evaluate an implicit field.

        Parameters
        ----------
        samples:
            Colored 3D samples with xyzrgb channels.
        queries:
            Query xyz coordinates.

        Returns
        -------
        torch.Tensor
            Density plus RGB predictions for each query.
        """

        asset_code = torch.tanh(self.asset(samples)).mean(dim=1, keepdim=True)
        params = self.param(asset_code)
        hidden = self.query(queries) + params
        return self.field(hidden)


def build() -> nn.Module:
    """Build the compact Shap-E transmitter.

    Returns
    -------
    nn.Module
        Random-initialized transmitter.
    """

    return ShapETransmitter()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create colored asset samples and query points.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Example transmitter inputs.
    """

    return torch.randn(1, 12, 6), torch.randn(1, 10, 3)


MENAGERIE_ENTRIES = [
    ("openai_shape_e_transmitter", "build", "example_input", "2023", "E6"),
]
