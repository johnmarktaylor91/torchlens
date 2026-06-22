"""Pi3-VisualGeometry: permutation-equivariant reference-free visual geometry.

Pi3 (2025), "Permutation-Equivariant Visual Geometry Learning", reconstructs
multi-view geometry without choosing a fixed reference view.  This compact
classic uses a shared image patch encoder, attention over unordered view tokens,
mean-set context, and per-view heads for affine pose and scale-invariant local
point/depth maps, preserving the audited load-bearing primitive:
permutation-equivariant multi-view geometry prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViewPatchEncoder(nn.Module):
    """Shared patch encoder applied independently to every view."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize the view encoder.

        Parameters
        ----------
        dim:
            Token width for each view.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images into view tokens and patch feature maps.

        Parameters
        ----------
        images:
            Batched views with shape ``(batch * views, 3, height, width)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Global view tokens and local patch features.
        """

        feat = self.local(self.patch(images))
        return feat.mean(dim=(2, 3)), feat


class Pi3VisualGeometry(nn.Module):
    """Reference-free permutation-equivariant multi-view geometry model."""

    def __init__(self, dim: int = 48, views: int = 3) -> None:
        """Initialize Pi3 compact model.

        Parameters
        ----------
        dim:
            Token width.
        views:
            Number of views in the compact example.
        """

        super().__init__()
        self.views = views
        self.encoder = ViewPatchEncoder(dim=dim)
        self.view_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.mix = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.pose = nn.Linear(dim, 7)
        self.depth = nn.Conv2d(dim * 2, 1, 1)
        self.point = nn.Conv2d(dim * 2, 3, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict per-view pose, depth, and local point maps.

        Parameters
        ----------
        images:
            Multi-view images with shape ``(batch, views, 3, height, width)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Pose parameters, depth maps, and point maps for each view.
        """

        batch, views, channels, height, width = images.shape
        flat = images.reshape(batch * views, channels, height, width)
        view_tokens, local = self.encoder(flat)
        view_tokens = view_tokens.reshape(batch, views, -1)
        attended = self.view_attention(view_tokens, view_tokens, view_tokens)[0]
        set_context = attended.mean(dim=1, keepdim=True)
        equivariant = self.mix(attended + set_context)
        pose = self.pose(equivariant)
        local = local.reshape(batch, views, local.shape[1], local.shape[2], local.shape[3])
        context_map = equivariant[:, :, :, None, None].expand_as(local)
        fused = torch.cat([local, context_map], dim=2).reshape(
            batch * views, local.shape[2] * 2, local.shape[3], local.shape[4]
        )
        depth = self.depth(fused).reshape(batch, views, 1, local.shape[3], local.shape[4])
        points = self.point(fused).reshape(batch, views, 3, local.shape[3], local.shape[4])
        return pose, depth, points


def build() -> nn.Module:
    """Build compact Pi3 visual geometry model.

    Returns
    -------
    nn.Module
        Random-initialized reference-free multi-view geometry network.
    """

    return Pi3VisualGeometry().eval()


def example_input() -> torch.Tensor:
    """Create compact unordered multi-view RGB images.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "Pi3-VisualGeometry",
        "build",
        "example_input",
        "2025",
        "E7",
    ),
]
