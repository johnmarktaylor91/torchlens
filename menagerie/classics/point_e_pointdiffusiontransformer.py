"""PointE_PointDiffusionTransformer: Point-E point-cloud diffusion transformer.

Nichol et al., 2022, "Point-E: A System for Generating 3D Point Clouds from
Complex Prompts".  Point-E generates point clouds with diffusion models that
denoise point-coordinate tokens while conditioning on text/image embeddings and
diffusion timesteps.  This compact version keeps the load-bearing primitive:
point tokens plus timestep and conditioning tokens processed by a transformer,
then projected to coordinate residuals.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointDiffusionTransformer(nn.Module):
    """Text/image/time-conditioned point denoising transformer."""

    def __init__(self, vocab_size: int = 128, dim: int = 48, image_dim: int = 16) -> None:
        """Initialize Point-E diffusion transformer layers.

        Parameters
        ----------
        vocab_size:
            Prompt vocabulary size.
        dim:
            Transformer width.
        image_dim:
            Input image-conditioning feature width.
        """

        super().__init__()
        self.point_in = nn.Linear(3, dim)
        self.text = nn.Embedding(vocab_size, dim)
        self.image = nn.Linear(image_dim, dim)
        self.time = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=96,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(dim, 3)

    def forward(
        self,
        noisy_points: torch.Tensor,
        text_tokens: torch.Tensor,
        image_embed: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Predict point-coordinate denoising residuals.

        Parameters
        ----------
        noisy_points:
            Noisy point coordinates, shape ``(batch, points, 3)``.
        text_tokens:
            Prompt token ids.
        image_embed:
            Image-conditioning vector from the synthetic-view stage.
        timestep:
            Diffusion noise level, shape ``(batch, 1)``.

        Returns
        -------
        torch.Tensor
            Coordinate residuals with shape ``(batch, points, 3)``.
        """

        points = self.point_in(noisy_points)
        text_cond = self.text(text_tokens).mean(dim=1, keepdim=True)
        image_cond = self.image(image_embed).unsqueeze(1)
        time_cond = self.time(timestep).unsqueeze(1)
        cond = torch.cat([time_cond, text_cond, image_cond], dim=1)
        hidden = self.transformer(torch.cat([cond, points], dim=1))
        return self.out(hidden[:, cond.shape[1] :])


def build() -> nn.Module:
    """Build a compact Point-E point diffusion transformer.

    Returns
    -------
    nn.Module
        Random-initialized conditioned point denoiser.
    """

    return PointDiffusionTransformer().eval()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create noisy points, prompt ids, image conditioning, and timestep.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Example inputs for Point-E denoising.
    """

    return (
        torch.randn(1, 16, 3),
        torch.randint(0, 128, (1, 6)),
        torch.randn(1, 16),
        torch.full((1, 1), 0.5),
    )


MENAGERIE_ENTRIES = [
    (
        "PointE_PointDiffusionTransformer",
        "build",
        "example_input",
        "2022",
        "E6",
    ),
]
