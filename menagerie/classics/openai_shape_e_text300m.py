"""OpenAI Shap-E text300M: text-conditioned implicit-function diffusion.

Paper: Jun and Nichol 2023, "Shap-E: Generating Conditional 3D Implicit
Functions", arXiv:2305.02463.

The compact model keeps Shap-E's second stage: a conditional diffusion model
generates the parameters of an implicit 3D function from text conditioning.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ShapETextDiffusion(nn.Module):
    """Compact text-conditioned Shap-E parameter denoiser."""

    def __init__(self, params: int = 32, dim: int = 48) -> None:
        """Initialize text embedding and transformer denoiser.

        Parameters
        ----------
        params:
            Number of implicit-function parameter tokens.
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.param_in = nn.Linear(1, dim)
        self.text = nn.Embedding(96, dim)
        self.time = nn.Linear(1, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=96, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(dim, 1)
        self.params = params

    def forward(
        self, noisy_params: torch.Tensor, text_tokens: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Denoise implicit-function parameters from text.

        Parameters
        ----------
        noisy_params:
            Noisy parameter vector.
        text_tokens:
            Text token ids.
        t:
            Diffusion time.

        Returns
        -------
        torch.Tensor
            Predicted parameter residuals.
        """

        p = self.param_in(noisy_params.unsqueeze(-1))
        cond = self.text(text_tokens).mean(dim=1, keepdim=True) + self.time(t).unsqueeze(1)
        return self.out(self.blocks(p + cond)).squeeze(-1)


def build() -> nn.Module:
    """Build the compact Shap-E text diffusion model.

    Returns
    -------
    nn.Module
        Random-initialized Shap-E text model.
    """

    return ShapETextDiffusion()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create noisy implicit parameters, text tokens, and diffusion time.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Example Shap-E text inputs.
    """

    return torch.randn(1, 32), torch.randint(0, 96, (1, 6)), torch.ones(1, 1) * 0.25


MENAGERIE_ENTRIES = [
    ("openai_shape_e_text300m", "build", "example_input", "2023", "E6"),
]
