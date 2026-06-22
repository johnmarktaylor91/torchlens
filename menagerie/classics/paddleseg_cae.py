"""Context Autoencoder masked-image model.

Chen et al. (2022), "Context Autoencoder for Self-Supervised Representation
Learning."  CAE separates the representation encoder from prediction: visible
patches are encoded, a latent contextual regressor predicts masked-patch
representations, and a lightweight decoder reconstructs masked pixels.  This
compact model keeps the visible-token encoder, masked latent regressor,
alignment projection, and pixel decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CompactCAE(nn.Module):
    """Small CAE pretraining graph with fixed checkerboard masking."""

    def __init__(
        self,
        image_size: int = 32,
        patch: int = 8,
        dim: int = 64,
        decoder_dim: int = 32,
    ) -> None:
        """Initialize the compact CAE.

        Parameters
        ----------
        image_size:
            Square image size.
        patch:
            Square patch size.
        dim:
            Encoder and regressor width.
        decoder_dim:
            Pixel decoder width.
        """

        super().__init__()
        self.patch = patch
        self.num_patches = (image_size // patch) ** 2
        self.visible_idx = torch.arange(0, self.num_patches, 2)
        self.masked_idx = torch.arange(1, self.num_patches, 2)
        patch_dim = 3 * patch * patch
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        regressor_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=dim * 2,
            batch_first=True,
            activation="gelu",
        )
        self.mask_tokens = nn.Parameter(torch.zeros(1, len(self.masked_idx), dim))
        self.regressor = nn.TransformerEncoder(regressor_layer, num_layers=1)
        self.align = nn.Linear(dim, dim)
        self.decoder = nn.Sequential(
            nn.Linear(dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, patch_dim),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert images to flattened patch tokens.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Flattened patches.
        """

        patches = x.unfold(2, self.patch, self.patch).unfold(3, self.patch, self.patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5).flatten(1, 2).flatten(2)
        return patches

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict masked patch pixels and latent targets.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Reconstructed masked patches and aligned latent predictions.
        """

        patches = self._patchify(x)
        tokens = self.patch_embed(patches) + self.pos
        visible = tokens[:, self.visible_idx]
        encoded = self.encoder(visible)
        masked_pos = self.pos[:, self.masked_idx]
        masked_queries = self.mask_tokens + masked_pos
        context = torch.cat([encoded, masked_queries.expand(x.shape[0], -1, -1)], dim=1)
        predicted = self.regressor(context)[:, -len(self.masked_idx) :]
        return self.decoder(predicted), self.align(predicted)


def build() -> nn.Module:
    """Build the compact CAE model.

    Returns
    -------
    nn.Module
        Random-init CAE in evaluation mode.
    """

    return CompactCAE().eval()


def example_input() -> torch.Tensor:
    """Return a small RGB image.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("paddleseg_cae", "build", "example_input", "2022", "DC"),
]
