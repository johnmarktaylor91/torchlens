"""LambdaNetworks: modeling long-range interactions without attention maps.

Paper: "LambdaNetworks: Modeling Long-Range Interactions Without Attention",
Bello, 2021.

The compact reconstruction keeps the lambda layer: context is summarized into
content lambdas and local positional lambdas, then applied to queries without a
query-key attention matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    """Lambda layer with content and local positional interactions."""

    def __init__(self, channels: int = 24, heads: int = 4, key_dim: int = 8) -> None:
        """Initialize projections and positional convolution."""

        super().__init__()
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = channels // heads
        self.to_q = nn.Conv2d(channels, heads * key_dim, 1, bias=False)
        self.to_k = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.to_v = nn.Conv2d(channels, self.value_dim, 1, bias=False)
        self.pos = nn.Conv3d(self.value_dim, key_dim * self.value_dim, (1, 3, 3), padding=(0, 1, 1))
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lambda content and position summaries."""

        batch, _, height, width = x.shape
        q = self.to_q(x).view(batch, self.heads, self.key_dim, height * width)
        k = torch.softmax(self.to_k(x).view(batch, self.key_dim, height * width), dim=-1)
        v = self.to_v(x).view(batch, self.value_dim, height * width)
        content_lambda = torch.einsum("bkn,bvn->bkv", k, v)
        content = torch.einsum("bhkn,bkv->bhvn", q, content_lambda)
        pos_kernel = self.pos(v.view(batch, self.value_dim, 1, height, width))
        pos_kernel = pos_kernel.view(batch, self.key_dim, self.value_dim, height * width)
        position = torch.einsum("bhkn,bkvn->bhvn", q, pos_kernel)
        y = (content + position).reshape(batch, -1, height, width)
        return self.out(y)


class LambdaNetworkCompact(nn.Module):
    """Compact image classifier using a Lambda layer."""

    def __init__(self) -> None:
        """Initialize stem, lambda block, and classifier."""

        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU())
        self.lambda_layer = LambdaLayer(24)
        self.head = nn.Linear(24, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an image using lambda interactions."""

        x = self.stem(x)
        x = F.relu(x + self.lambda_layer(x))
        return self.head(x.mean(dim=(2, 3)))


def build() -> nn.Module:
    """Build compact LambdaNetworks model."""

    return LambdaNetworkCompact()


def example_input() -> torch.Tensor:
    """Return a small image."""

    return torch.randn(1, 3, 24, 24)


MENAGERIE_ENTRIES = [("LambdaNetworks", "build", "example_input", "2021", "E7")]
