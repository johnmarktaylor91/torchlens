"""CrossFormer vision Transformer with cross-scale embeddings and L/SDA attention.

Paper: Wang et al. 2021/2023, "CrossFormer: A Versatile Vision Transformer
Hinging on Cross-scale Attention"; official JAX/Flax-style targets are
represented here as a compact PyTorch classic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossScaleEmbedding(nn.Module):
    """Cross-scale embedding layer using parallel patch projections."""

    def __init__(self, in_chans: int, dim: int) -> None:
        """Initialize multi-kernel patch projections.

        Parameters
        ----------
        in_chans:
            Number of input image channels.
        dim:
            Token width.
        """

        super().__init__()
        part = dim // 3
        self.proj3 = nn.Conv2d(in_chans, part, 3, stride=2, padding=1)
        self.proj5 = nn.Conv2d(in_chans, part, 5, stride=2, padding=2)
        self.proj7 = nn.Conv2d(in_chans, dim - 2 * part, 7, stride=2, padding=3)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed an image with multiple patch scales.

        Parameters
        ----------
        x:
            Image tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Token grid of shape ``(batch, height/2, width/2, dim)``.
        """

        y = torch.cat([self.proj3(x), self.proj5(x), self.proj7(x)], dim=1)
        return self.norm(y.permute(0, 2, 3, 1))


class LSDABlock(nn.Module):
    """Alternating short-distance and long-distance attention block."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize LSDA attention and MLP sublayers.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Attention heads.
        """

        super().__init__()
        self.short = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.long = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 3), nn.GELU(), nn.Linear(dim * 3, dim)
        )
        self.pos_mlp = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, heads))

    def _attend(self, attn: nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to flattened tokens.

        Parameters
        ----------
        attn:
            Attention module.
        x:
            Token tensor.

        Returns
        -------
        torch.Tensor
            Attention output.
        """

        y, _ = attn(x, x, x)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run short-window then long-stride attention over a token grid.

        Parameters
        ----------
        x:
            Token grid of shape ``(batch, height, width, dim)``.

        Returns
        -------
        torch.Tensor
            Updated token grid.
        """

        bsz, height, width, dim = x.shape
        y = self.norm1(x).reshape(bsz, height * width, dim)
        rel = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij"
            ),
            dim=-1,
        )
        _ = self.pos_mlp(rel.reshape(-1, 2).to(x.device))
        y = y + self._attend(self.short, y)
        long_tokens = (
            self.norm2(y).reshape(bsz, height, width, dim)[:, ::2, ::2].reshape(bsz, -1, dim)
        )
        long_out = self._attend(self.long, long_tokens)
        long_grid = F.interpolate(
            long_out.reshape(bsz, max(1, height // 2), max(1, width // 2), dim).permute(0, 3, 1, 2),
            size=(height, width),
            mode="nearest",
        ).permute(0, 2, 3, 1)
        y = y.reshape(bsz, height, width, dim) + long_grid
        return y + self.mlp(y)


class CrossFormerTiny(nn.Module):
    """Compact CrossFormer pyramid classifier."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize a two-stage CrossFormer.

        Parameters
        ----------
        num_classes:
            Number of classifier outputs.
        """

        super().__init__()
        self.stage1 = CrossScaleEmbedding(3, 24)
        self.block1 = LSDABlock(24)
        self.merge = nn.Conv2d(24, 48, 3, stride=2, padding=1)
        self.block2 = LSDABlock(48)
        self.head = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x = self.block1(self.stage1(x))
        x = self.merge(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.block2(x)
        return self.head(x.mean(dim=(1, 2)))


def build() -> nn.Module:
    """Build the compact CrossFormer model.

    Returns
    -------
    nn.Module
        CrossFormer model.
    """

    return CrossFormerTiny()


def example_input() -> torch.Tensor:
    """Create a small image input.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [("CrossFormer", "build", "example_input", "2021", "E7")]
