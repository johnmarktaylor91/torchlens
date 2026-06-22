"""GeoWizard and SSTN compact classics.

GeoWizard: Fu et al., ECCV 2024, "Unleashing the Diffusion Priors for 3D
Geometry Estimation from a Single Image."  The paper extends Stable Diffusion
for joint depth and surface-normal prediction by VAE-encoding RGB, depth, and
normal latents, routing two geometry groups through a U-Net under a geometry
switcher, and conditioning on scene-layout prompts.

SSTN: Zhong et al., IEEE TGRS 2021, "Spectral-Spatial Transformer Network for
Hyperspectral Image Classification: A Factorized Architecture Search Framework."
The model combines spatial attention with spectral association modules for HSI
classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Convolution, normalization, and SiLU activation block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize a convolution block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(4, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Parameters
        ----------
        x:
            Input tensor ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Activated feature map.
        """

        return F.silu(self.norm(self.conv(x)))


class _CrossAttention2d(nn.Module):
    """Cross-attention from spatial image tokens to conditioning tokens."""

    def __init__(self, channels: int, context_dim: int, num_heads: int = 2) -> None:
        """Initialize the attention layer.

        Parameters
        ----------
        channels:
            Spatial feature width.
        context_dim:
            Conditioning-token width.
        num_heads:
            Number of attention heads.
        """

        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.context_proj = nn.Linear(context_dim, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Attend from image features to context tokens.

        Parameters
        ----------
        x:
            Image features ``(B, C, H, W)``.
        context:
            Conditioning tokens ``(B, N, context_dim)``.

        Returns
        -------
        torch.Tensor
            Context-enriched image features.
        """

        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        query = self.norm(tokens)
        key_value = self.context_proj(context)
        attended, _ = self.attn(query, key_value, key_value)
        tokens = tokens + attended
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class GeoWizardDepthNormal(nn.Module):
    """Joint depth/normal latent-diffusion U-Net with geometry switch tokens."""

    def __init__(self, latent_channels: int = 4, width: int = 32, context_dim: int = 32) -> None:
        """Initialize GeoWizard compact core.

        Parameters
        ----------
        latent_channels:
            Channels per VAE latent group.
        width:
            Internal U-Net width.
        context_dim:
            Scene and geometry token width.
        """

        super().__init__()
        self.rgb_encoder = nn.Conv2d(3, latent_channels, 3, padding=1)
        self.depth_encoder = nn.Conv2d(1, latent_channels, 3, padding=1)
        self.normal_encoder = nn.Conv2d(3, latent_channels, 3, padding=1)
        self.scene_embed = nn.Embedding(3, context_dim)
        self.geometry_embed = nn.Embedding(2, context_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim)
        )

        self.down = _ConvBlock(latent_channels * 2, width)
        self.mid_attn = _CrossAttention2d(width, context_dim)
        self.mid = _ConvBlock(width, width)
        self.up = _ConvBlock(width + latent_channels * 2, width)
        self.depth_head = nn.Conv2d(width, 1, 3, padding=1)
        self.normal_head = nn.Conv2d(width, 3, 3, padding=1)

    def _geometry_group(
        self,
        rgb_latent: torch.Tensor,
        geom_latent: torch.Tensor,
        scene_id: torch.Tensor,
        timestep: torch.Tensor,
        geometry_id: int,
    ) -> torch.Tensor:
        """Process one GeoWizard geometry group.

        Parameters
        ----------
        rgb_latent:
            Encoded RGB latent.
        geom_latent:
            Encoded depth or normal latent.
        scene_id:
            Scene-layout index.
        timestep:
            Diffusion timestep tensor.
        geometry_id:
            ``0`` for depth, ``1`` for normal.

        Returns
        -------
        torch.Tensor
            Shared decoded feature map for the requested geometry.
        """

        batch = rgb_latent.shape[0]
        switch = torch.full((batch,), geometry_id, dtype=torch.long, device=rgb_latent.device)
        context = torch.stack(
            [
                self.scene_embed(scene_id),
                self.geometry_embed(switch),
                self.time_mlp(timestep[:, None].float()),
            ],
            dim=1,
        )
        group = torch.cat([rgb_latent, geom_latent], dim=1)
        down = self.down(group)
        mid = self.mid(self.mid_attn(down, context))
        return self.up(torch.cat([mid, group], dim=1))

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Predict depth and surface normal from RGB plus noisy geometry latents.

        Parameters
        ----------
        x:
            Tuple ``(rgb, noisy_depth, noisy_normal, timestep, scene_id)``.

        Returns
        -------
        torch.Tensor
            Concatenated ``(depth, normal)`` prediction with four channels.
        """

        rgb, noisy_depth, noisy_normal, timestep, scene_id = x
        rgb_latent = self.rgb_encoder(rgb)
        depth_latent = self.depth_encoder(noisy_depth)
        normal_latent = self.normal_encoder(noisy_normal)
        depth_features = self._geometry_group(rgb_latent, depth_latent, scene_id, timestep, 0)
        normal_features = self._geometry_group(rgb_latent, normal_latent, scene_id, timestep, 1)
        depth = self.depth_head(depth_features)
        normal = F.normalize(self.normal_head(normal_features), dim=1)
        return torch.cat([depth, normal], dim=1)


class _SpectralAssociation(nn.Module):
    """Self-attention over spectral bands for SSTN."""

    def __init__(self, bands: int, hidden: int) -> None:
        """Initialize the spectral association module.

        Parameters
        ----------
        bands:
            Number of hyperspectral bands.
        hidden:
            Hidden width for per-band tokens.
        """

        super().__init__()
        self.band_embed = nn.Linear(1, hidden)
        self.attn = nn.MultiheadAttention(hidden, 2, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Associate spectral bands at the central pixel.

        Parameters
        ----------
        x:
            Hyperspectral patch ``(B, bands, H, W)``.

        Returns
        -------
        torch.Tensor
            Spectral descriptor ``(B, bands)``.
        """

        center = x[:, :, x.shape[2] // 2, x.shape[3] // 2].unsqueeze(-1)
        tokens = self.band_embed(center)
        tokens = tokens + self.attn(self.norm(tokens), self.norm(tokens), self.norm(tokens))[0]
        return self.out(tokens.mean(dim=1))


class SSTNHSI(nn.Module):
    """Spectral-spatial transformer network for hyperspectral classification."""

    def __init__(self, bands: int = 16, hidden: int = 32, num_classes: int = 6) -> None:
        """Initialize SSTN compact core.

        Parameters
        ----------
        bands:
            Number of hyperspectral bands.
        hidden:
            Spatial/spectral hidden width.
        num_classes:
            Number of output classes.
        """

        super().__init__()
        self.spatial_proj = nn.Conv2d(bands, hidden, 3, padding=1)
        self.spatial_attn = nn.MultiheadAttention(hidden, 2, batch_first=True)
        self.spatial_norm = nn.LayerNorm(hidden)
        self.spectral = _SpectralAssociation(bands, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + bands, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a hyperspectral image patch.

        Parameters
        ----------
        x:
            Hyperspectral patch ``(B, bands, H, W)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        spatial = self.spatial_proj(x)
        batch, channels, height, width = spatial.shape
        tokens = spatial.flatten(2).transpose(1, 2)
        tokens = (
            tokens
            + self.spatial_attn(
                self.spatial_norm(tokens), self.spatial_norm(tokens), self.spatial_norm(tokens)
            )[0]
        )
        spatial_desc = tokens.mean(dim=1).reshape(batch, channels)
        spectral_desc = self.spectral(x)
        return self.head(torch.cat([spatial_desc, spectral_desc], dim=1))


def build_geowizard_depth_normal() -> nn.Module:
    """Build compact GeoWizard depth-normal model.

    Returns
    -------
    nn.Module
        Random-init GeoWizard core.
    """

    return GeoWizardDepthNormal()


def example_input_geowizard() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Return GeoWizard example inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        RGB, noisy depth, noisy normal, timestep, and scene-layout id.
    """

    return (
        torch.randn(1, 3, 16, 16),
        torch.randn(1, 1, 16, 16),
        torch.randn(1, 3, 16, 16),
        torch.tensor([10.0]),
        torch.tensor([1]),
    )


def build_sstn_hsi() -> nn.Module:
    """Build compact SSTN for HSI classification.

    Returns
    -------
    nn.Module
        Random-init SSTN core.
    """

    return SSTNHSI()


def example_input_sstn() -> torch.Tensor:
    """Return a small hyperspectral patch.

    Returns
    -------
    torch.Tensor
        Tensor ``(1, 16, 9, 9)``.
    """

    return torch.randn(1, 16, 9, 9)


MENAGERIE_ENTRIES = [
    (
        "GeoWizard Depth-Normal (joint latent diffusion geometry estimator)",
        "build_geowizard_depth_normal",
        "example_input_geowizard",
        "2024",
        "DC",
    ),
    (
        "SSTN HSI (spectral-spatial transformer network)",
        "build_sstn_hsi",
        "example_input_sstn",
        "2021",
        "CB",
    ),
]
