"""UniFormer-XXS/4 for Kinetics-400 video classification.

Paper: UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning
and UniFormer: Unifying Convolution and Self-attention for Visual Recognition, 2022.

This is a compact random-init reconstruction of the official video UniFormer family:
four hierarchical 3D stages, Dynamic Position Embedding (depthwise 3D convolution),
local MHRA implemented as pointwise-depthwise-pointwise 3D convolution in shallow
stages, global MHRA implemented as spatiotemporal self-attention in deep stages, and
pointwise FFN heads.  The input is intentionally tiny while preserving the stage logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPositionEmbedding(nn.Module):
    """Depthwise 3D convolutional positional encoding used by UniFormer blocks."""

    def __init__(self, channels: int) -> None:
        """Initialize the depthwise 3D positional embedding.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """

        super().__init__()
        self.proj = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add dynamic 3D positional information to a video feature map.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, channels, time, height, width)``.

        Returns
        -------
        torch.Tensor
            Position-enhanced tensor with the same shape as ``x``.
        """

        return x + self.proj(x)


class LocalMHRA(nn.Module):
    """Local multi-head relation aggregation with 3D convolutional token affinity."""

    def __init__(self, channels: int) -> None:
        """Initialize local MHRA.

        Parameters
        ----------
        channels:
            Number of feature channels.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate local spatiotemporal relations.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, channels, time, height, width)``.

        Returns
        -------
        torch.Tensor
            Locally aggregated tensor.
        """

        return self.net(x)


class GlobalMHRA(nn.Module):
    """Global spatiotemporal self-attention relation aggregation."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        """Initialize global MHRA.

        Parameters
        ----------
        channels:
            Number of feature channels.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global attention over all time-space tokens.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, channels, time, height, width)``.

        Returns
        -------
        torch.Tensor
            Globally aggregated tensor with the same shape as ``x``.
        """

        batch, channels, time, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(tokens).view(batch, time * height * width, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = self.head_dim**-0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, time * height * width, channels)
        out = self.proj(out).transpose(1, 2).view(batch, channels, time, height, width)
        return out


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for 3D video feature maps."""

    def __init__(self, channels: int) -> None:
        """Initialize channel-wise layer normalization.

        Parameters
        ----------
        channels:
            Number of channels to normalize.
        """

        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize channels in ``(B, C, T, H, W)`` layout.

        Parameters
        ----------
        x:
            Video feature map.

        Returns
        -------
        torch.Tensor
            Normalized feature map.
        """

        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)


class UniFormerFFN(nn.Module):
    """Pointwise feed-forward network used inside UniFormer blocks."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        """Initialize the feed-forward branch.

        Parameters
        ----------
        channels:
            Number of input and output channels.
        expansion:
            Hidden-channel expansion ratio.
        """

        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the channel feed-forward branch.

        Parameters
        ----------
        x:
            Video feature map.

        Returns
        -------
        torch.Tensor
            Feature map after pointwise feed-forward mixing.
        """

        return self.net(x)


class UniFormerBlock(nn.Module):
    """UniFormer block with DPE, MHRA, and FFN residual branches."""

    def __init__(self, channels: int, global_relation: bool) -> None:
        """Initialize a local or global UniFormer block.

        Parameters
        ----------
        channels:
            Number of block channels.
        global_relation:
            If true, use global attention MHRA; otherwise use local convolutional MHRA.
        """

        super().__init__()
        self.dpe = DynamicPositionEmbedding(channels)
        self.norm1 = ChannelLayerNorm(channels)
        self.mhra = GlobalMHRA(channels) if global_relation else LocalMHRA(channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.ffn = UniFormerFFN(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one UniFormer residual block.

        Parameters
        ----------
        x:
            Video feature map.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        x = self.dpe(x)
        x = x + self.mhra(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class UniFormerStage(nn.Module):
    """Patch downsampling followed by one or more UniFormer blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        global_relation: bool,
        first: bool = False,
    ) -> None:
        """Initialize a hierarchical UniFormer stage.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        depth:
            Number of UniFormer blocks.
        global_relation:
            Whether stage blocks use global MHRA.
        first:
            Whether to use the first-stage patch embedding stride.
        """

        super().__init__()
        kernel = (1, 4, 4) if first else (1, 2, 2)
        stride = kernel
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride)
        self.blocks = nn.ModuleList(
            [UniFormerBlock(out_channels, global_relation) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample and process a video feature hierarchy stage.

        Parameters
        ----------
        x:
            Video feature map.

        Returns
        -------
        torch.Tensor
            Stage output.
        """

        x = self.down(x)
        for block in self.blocks:
            x = block(x)
        return x


class CompactVideoUniFormer(nn.Module):
    """Compact hierarchical video UniFormer classifier."""

    def __init__(self, depths: tuple[int, int, int, int], num_classes: int = 400) -> None:
        """Initialize a compact UniFormer video classifier.

        Parameters
        ----------
        depths:
            Number of blocks in the four UniFormer stages.
        num_classes:
            Number of action classes.
        """

        super().__init__()
        dims = (16, 24, 32, 48)
        self.stages = nn.ModuleList(
            [
                UniFormerStage(3, dims[0], depths[0], global_relation=False, first=True),
                UniFormerStage(dims[0], dims[1], depths[1], global_relation=False),
                UniFormerStage(dims[1], dims[2], depths[2], global_relation=True),
                UniFormerStage(dims[2], dims[3], depths[3], global_relation=True),
            ]
        )
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Classify a video clip.

        Parameters
        ----------
        video:
            Tensor of shape ``(batch, channels, time, height, width)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x = video
        for stage in self.stages:
            x = stage(x)
        pooled = x.mean(dim=(2, 3, 4))
        return self.head(pooled)


def build_uniformer_xxs4_160_k400() -> nn.Module:
    """Build a compact UniFormer-XXS/4 Kinetics-400 classifier.

    Returns
    -------
    nn.Module
        Random-init compact UniFormer model.
    """

    return CompactVideoUniFormer(depths=(1, 1, 1, 1))


def example_input() -> torch.Tensor:
    """Create a compact four-frame video input.

    Returns
    -------
    torch.Tensor
        Input tensor with shape ``(1, 3, 4, 32, 32)``.
    """

    return torch.randn(1, 3, 4, 32, 32)


build = build_uniformer_xxs4_160_k400

MENAGERIE_ENTRIES = [
    (
        "uniformer_xxs4_160_k400",
        "build_uniformer_xxs4_160_k400",
        "example_input",
        "2022",
        "E5",
    ),
]
