"""DGCNN: Dynamic Graph CNN for Learning on Point Clouds.

Wang et al., ACM TOG 2019.
Paper: https://arxiv.org/abs/1801.07829
Source: https://github.com/WangYueFt/dgcnn

DGCNN's defining contribution is the EdgeConv operator over a *dynamically
recomputed* k-nearest-neighbor graph: at each layer the kNN graph is rebuilt in
the current feature space, edge features ``[x_i, x_j - x_i]`` are formed for each
neighbor, an MLP is applied, and a max over neighbors aggregates them.  Stacking
EdgeConv layers (each with its own dynamic graph) followed by a global pooling
head yields the classification network.

This is a faithful random-init reimplementation of the classification model
(``model.py`` ``DGCNN``):
  - 4 EdgeConv layers (64, 64, 128, 256), k=20 dynamic kNN
  - 1x1 conv fuses concatenated EdgeConv outputs -> emb_dims (1024)
  - global max+avg pool -> 2-layer MLP classifier (40 classes)
Input is channels-first ``(B, 3, N)`` per the published convention.  kNN is pure
torch (pairwise distance + topk), so no custom CUDA op is required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of the ``k`` nearest neighbors for each point.

    Parameters
    ----------
    x:
        Point features ``(B, C, N)``.
    k:
        Number of neighbors.

    Returns
    -------
    torch.Tensor
        Neighbor indices ``(B, N, k)``.
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x: torch.Tensor, k: int = 20) -> torch.Tensor:
    """Build edge features ``[x_i, x_j - x_i]`` over the dynamic kNN graph.

    Returns a ``(B, 2C, N, k)`` tensor ready for a 2D conv EdgeConv block.
    """
    batch_size, num_dims, num_points = x.size()
    idx = knn(x, k=k)  # (B, N, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    x_t = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x_t.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x_t = x_t.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x_t, x_t), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (B, 2C, N, k)


class EdgeConv(nn.Module):
    """A single EdgeConv block: dynamic-graph edge features -> conv -> max-pool."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 20) -> None:
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = get_graph_feature(x, k=self.k)
        x = self.conv(x)
        return x.max(dim=-1, keepdim=False)[0]  # (B, out_ch, N)


class DGCNN(nn.Module):
    """DGCNN classification network (4 EdgeConv layers + global head)."""

    def __init__(
        self,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
        output_channels: int = 40,
    ) -> None:
        super().__init__()
        self.k = k
        self.edge1 = EdgeConv(3, 64, k)
        self.edge2 = EdgeConv(64, 64, k)
        self.edge3 = EdgeConv(64, 128, k)
        self.edge4 = EdgeConv(128, 256, k)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x1 = self.edge1(x)
        x2 = self.edge2(x1)
        x3 = self.edge3(x2)
        x4 = self.edge4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), dim=1)

        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        return self.linear3(x)


def build() -> nn.Module:
    """Build DGCNN classification net (k=20, emb_dims=1024, 40 classes)."""
    return DGCNN(k=20, emb_dims=1024, dropout=0.5, output_channels=40)


def example_input() -> torch.Tensor:
    """Example channels-first point cloud ``(2, 3, 256)`` (B, xyz, num_points)."""
    return torch.randn(2, 3, 256)


MENAGERIE_ENTRIES = [
    (
        "DGCNN (Dynamic Graph CNN, EdgeConv + dynamic kNN)",
        "build",
        "example_input",
        "2019",
        "DC",
    ),
]
