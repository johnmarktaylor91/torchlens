"""OpenAI Shap-E image300M: image-conditioned implicit-function diffusion.

Paper: Jun and Nichol 2023, "Shap-E: Generating Conditional 3D Implicit
Functions".

This compact version mirrors the image-conditional Shap-E model: an image
encoder conditions a transformer denoiser that predicts implicit-function
parameter residuals.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ShapEImageDiffusion(nn.Module):
    """Compact image-conditioned Shap-E parameter denoiser."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize image encoder and parameter transformer.

        Parameters
        ----------
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, dim, 3, stride=2, padding=1),
        )
        self.param_in = nn.Linear(1, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=96, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(dim, 1)

    def forward(self, noisy_params: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Denoise implicit parameters conditioned on an image.

        Parameters
        ----------
        noisy_params:
            Noisy implicit-function parameters.
        image:
            RGB conditioning image.

        Returns
        -------
        torch.Tensor
            Predicted parameter residuals.
        """

        cond = self.image(image).flatten(2).mean(dim=-1).unsqueeze(1)
        params = self.param_in(noisy_params.unsqueeze(-1))
        return self.out(self.blocks(params + cond)).squeeze(-1)


def build() -> nn.Module:
    """Build the compact Shap-E image diffusion model.

    Returns
    -------
    nn.Module
        Random-initialized Shap-E image model.
    """

    return ShapEImageDiffusion()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Create noisy implicit parameters and an RGB image.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Example image-conditioned Shap-E inputs.
    """

    return torch.randn(1, 32), torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("openai_shape_e_image300m", "build", "example_input", "2023", "E6"),
]
