# FAITHFUL REIMPLEMENTATION from arXiv:2204.00325 (no public code) -- A/B codex
"""CAT-Det: contrastively augmented transformer for multi-modal 3D detection."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BasicPointTransformer(nn.Module):
    """Point transformer attention with relative positional encoding."""

    def __init__(self, channels: int) -> None:
        """Create a basic point transformer block.

        Parameters
        ----------
        channels:
            Feature width for the attention projections.
        """
        super().__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.position = nn.Sequential(
            nn.Linear(3, channels), nn.ReLU(), nn.Linear(channels, channels)
        )
        self.weight = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Aggregate point features with self-attention.

        Parameters
        ----------
        coords:
            Point coordinates with shape ``(batch, points, 3)``.
        features:
            Point features with shape ``(batch, points, channels)``.

        Returns
        -------
        torch.Tensor
            Aggregated point features.
        """
        rel_pos = coords.unsqueeze(2) - coords.unsqueeze(1)
        pos = self.position(rel_pos)
        logits = self.weight(
            self.query(features).unsqueeze(2) - self.key(features).unsqueeze(1) + pos
        )
        weights = F.softmax(logits, dim=2)
        values = self.value(features).unsqueeze(1) + pos
        return (weights * values).sum(dim=2)


class PointTransformerBlock(nn.Module):
    """CAT-Det point transformer block with local and global context."""

    def __init__(self, in_channels: int, out_channels: int, sampled_points: int) -> None:
        """Create a point transformer block.

        Parameters
        ----------
        in_channels:
            Input feature width.
        out_channels:
            Output feature width.
        sampled_points:
            Number of centroid points retained by the local layer.
        """
        super().__init__()
        self.sampled_points = sampled_points
        self.input_proj = nn.Linear(in_channels, out_channels)
        self.local = BasicPointTransformer(out_channels)
        self.global_block = BasicPointTransformer(out_channels)
        self.fuse = nn.Linear(out_channels * 2, out_channels)

    def forward(
        self, coords: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run local and global point transformer layers.

        Parameters
        ----------
        coords:
            Point coordinates.
        features:
            Point features.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Downsampled coordinates and fused point features.
        """
        projected = self.input_proj(features)
        sampled_coords = coords[:, : self.sampled_points, :]
        sampled_features = projected[:, : self.sampled_points, :]
        local_features = self.local(sampled_coords, sampled_features)
        global_features = self.global_block(sampled_coords, sampled_features)
        fused = self.fuse(torch.cat([local_features, global_features], dim=-1))
        return sampled_coords, fused


class ImageTransformerBlock(nn.Module):
    """CAT-Det image transformer block with convolutions before ViT attention."""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 4) -> None:
        """Create an image transformer block.

        Parameters
        ----------
        in_channels:
            Number of input image channels.
        out_channels:
            Number of output feature channels.
        heads:
            Number of self-attention heads.
        """
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(out_channels)
        self.attention = nn.MultiheadAttention(out_channels, heads, batch_first=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image features with local convolutions and global attention.

        Parameters
        ----------
        image:
            Image feature tensor.

        Returns
        -------
        torch.Tensor
            Image feature map.
        """
        features = self.local(image)
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        attended, _ = self.attention(self.norm(tokens), self.norm(tokens), self.norm(tokens))
        return attended.transpose(1, 2).reshape(batch, channels, height, width)


class CrossModalTransformer(nn.Module):
    """CAT-Det cross-modal transformer fusion module."""

    def __init__(self, point_channels: int, image_channels: int, fusion_dim: int) -> None:
        """Create cross-modal attention projections.

        Parameters
        ----------
        point_channels:
            Width of point features.
        image_channels:
            Width of sampled image features.
        fusion_dim:
            Shared attention dimension.
        """
        super().__init__()
        self.point_q = nn.Linear(point_channels, fusion_dim)
        self.point_k = nn.Linear(point_channels, fusion_dim)
        self.point_v = nn.Linear(point_channels, fusion_dim)
        self.image_q = nn.Linear(image_channels, fusion_dim)
        self.image_k = nn.Linear(image_channels, fusion_dim)
        self.image_v = nn.Linear(image_channels, fusion_dim)
        self.output = nn.Linear(point_channels + image_channels + (2 * fusion_dim), point_channels)

    def forward(
        self,
        coords: torch.Tensor,
        point_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse point and image features using cross-attention.

        Parameters
        ----------
        coords:
            Point coordinates normalized to ``[-1, 1]`` for image sampling.
        point_features:
            Point branch features.
        image_features:
            Image branch feature map.

        Returns
        -------
        torch.Tensor
            Fused point features.
        """
        grid = coords[:, :, :2].clamp(-1.0, 1.0).unsqueeze(2)
        sampled_image = (
            F.grid_sample(image_features, grid, align_corners=False).squeeze(-1).transpose(1, 2)
        )
        point_context = torch.softmax(
            self.point_q(point_features) @ self.image_k(sampled_image).transpose(1, 2), dim=-1
        )
        image_context = torch.softmax(
            self.image_q(sampled_image) @ self.point_k(point_features).transpose(1, 2), dim=-1
        )
        point_from_image = point_context @ self.image_v(sampled_image)
        image_from_point = image_context @ self.point_v(point_features)
        fused = torch.cat(
            [point_features, sampled_image, point_from_image, image_from_point], dim=-1
        )
        return self.output(fused)


class CATDet(nn.Module):
    """Tiny structural CAT-Det backbone and detection heads."""

    def __init__(self) -> None:
        """Create a compact CAT-Det model."""
        super().__init__()
        point_channels = [3, 16, 32, 48, 64]
        image_channels = [3, 16, 32, 48, 64]
        sampled = [32, 16, 8, 4]
        self.point_blocks = nn.ModuleList(
            PointTransformerBlock(point_channels[index], point_channels[index + 1], sampled[index])
            for index in range(4)
        )
        self.image_blocks = nn.ModuleList(
            ImageTransformerBlock(image_channels[index], image_channels[index + 1])
            for index in range(4)
        )
        self.cmt_blocks = nn.ModuleList(
            CrossModalTransformer(point_channels[index + 1], image_channels[index + 1], 32)
            for index in range(4)
        )
        self.segmentation_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.box_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8))
        self.point_projection = nn.Linear(64, 32)
        self.object_projection = nn.Linear(64, 32)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run two-stream transformer fusion and detection heads.

        Parameters
        ----------
        inputs:
            Tuple ``(points, image)`` where points are ``(batch, n, 3)`` and image is NCHW.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Point segmentation logits, box predictions, and contrastive embeddings.
        """
        coords, image = inputs
        point_features = coords
        image_features = image
        for point_block, image_block, cmt_block in zip(
            self.point_blocks,
            self.image_blocks,
            self.cmt_blocks,
            strict=True,
        ):
            image_features = F.avg_pool2d(image_features, kernel_size=2, stride=2)
            image_features = image_block(image_features)
            coords, point_features = point_block(coords, point_features)
            point_features = cmt_block(coords, point_features, image_features)
        pooled = point_features.mean(dim=1)
        segmentation = self.segmentation_head(point_features)
        boxes = self.box_head(pooled)
        contrastive = F.normalize(self.point_projection(point_features), dim=-1)
        object_embedding = F.normalize(self.object_projection(pooled), dim=-1).unsqueeze(1)
        return segmentation, boxes, torch.cat([contrastive, object_embedding], dim=1)


def build_cat_det() -> CATDet:
    """Build a compact CAT-Det model.

    Returns
    -------
    CATDet
        The reimplemented model.
    """
    return CATDet()


def example_input_cat_det() -> tuple[torch.Tensor, torch.Tensor]:
    """Create example LiDAR points and RGB image.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Point cloud and image tensors.
    """
    points = torch.rand(1, 64, 3) * 2.0 - 1.0
    image = torch.randn(1, 3, 64, 64)
    return points, image


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [("CAT-Det", "build_cat_det", "example_input_cat_det", 2022, "REIMPL")]
