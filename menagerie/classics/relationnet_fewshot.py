"""Relation Network for few-shot classification.

Sung et al., "Learning to Compare: Relation Network for Few-Shot Learning."
CVPR 2018. arXiv:1711.06025.
Source: https://github.com/floodsung/LearningToCompare_FSL

Distinctive primitive:
  The Relation Network splits learning into two modules:
  1. EMBEDDING MODULE (Conv4): 4-block conv encoder shared between support
     and query examples. Each block = Conv2d + BatchNorm + ReLU (+ MaxPool
     in first two blocks). Produces a spatial feature map.
  2. RELATION MODULE: takes the CONCATENATION of support and query embeddings
     (along channel dim) as input, and runs a 2-block conv followed by
     FC layers to predict a RELATION SCORE (similarity between 0 and 1).
     The key insight is learning the COMPARISON function explicitly.

  The few-shot protocol (N-way K-shot):
  - Support: N*K labeled examples -> embed -> average per class = prototype
  - Query: M query examples -> embed
  - Relation: concat(query_feat, proto_feat) for each pair -> relation_score
  - Predict: argmax of relation scores

Faithful-compact simplifications:
  - Conv4 embedding: 4 blocks (C=32), input 1x28x28 -> spatial 1x1 embedding.
    (For compactness, input is 1x16x16; 4 blocks with kernel=3, stride=1,
     first 2 with maxpool.)
  - Relation module: 2 conv blocks + 2 FC layers.
  - N=2 classes, K=1 shot, M=2 queries (2-way-1-shot).
  - Random init, CPU, forward-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard conv block: Conv2d + BN + ReLU [+ MaxPool]."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv4EmbeddingModule(nn.Module):
    """4-block conv embedding module (Conv4) for Relation Network.

    Input: (N, C_in, H, W)  ->  (N, 64, H', W') spatial features.
    First 2 blocks have MaxPool; last 2 do not.
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, base_ch, pool=True),  # -> base_ch, H/2, W/2
            ConvBlock(base_ch, base_ch, pool=True),  # -> base_ch, H/4, W/4
            ConvBlock(base_ch, base_ch, pool=False),  # -> base_ch, H/4, W/4
            ConvBlock(base_ch, base_ch, pool=False),  # -> base_ch, H/4, W/4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class RelationModule(nn.Module):
    """Relation module: takes concat(query_feat, support_feat) -> relation score.

    Input: (N*M, 2*base_ch, H', W')  -- doubled channels from concatenation
    Output: (N*M, 1) relation scores in [0, 1]
    """

    def __init__(self, base_ch: int = 32, spatial: int = 4) -> None:
        super().__init__()
        # 2 conv blocks on doubled channels
        self.conv_blocks = nn.Sequential(
            ConvBlock(2 * base_ch, base_ch, pool=True),  # H'//2
            ConvBlock(base_ch, base_ch, pool=True),  # H'//4
        )
        # After 2 pooling on spatial=4: 4->2->1 -> 1*1
        self.fc = nn.Sequential(
            nn.Linear(base_ch, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 2*base_ch, H, W) -> (N, 1) scores"""
        h = self.conv_blocks(x)  # (N, base_ch, 1, 1) after pooling
        h = h.view(h.size(0), -1)  # (N, base_ch)
        return self.fc(h)  # (N, 1)


class RelationNet(nn.Module):
    """Full Relation Network for few-shot classification.

    Forward: given support (N*K, C, H, W) and query (M, C, H, W) images,
    computes relation scores for all N*M (query, class) pairs.
    """

    def __init__(self, in_channels: int = 1, base_ch: int = 32, spatial: int = 4) -> None:
        super().__init__()
        self.embedding = Conv4EmbeddingModule(in_channels, base_ch)
        self.relation = RelationModule(base_ch, spatial)
        self.base_ch = base_ch

    def forward(
        self,
        support: torch.Tensor,  # (N*K, C, H, W) support images
        query: torch.Tensor,  # (M, C, H, W) query images
        n_way: int,
        k_shot: int,
    ) -> torch.Tensor:
        """Returns relation scores: (M, N) for each query x class pair."""
        NK = support.size(0)
        M = query.size(0)
        N = n_way

        # Embed support and query
        supp_feat = self.embedding(support)  # (N*K, base_ch, H', W')
        query_feat = self.embedding(query)  # (M, base_ch, H', W')

        H_, W_ = supp_feat.shape[-2], supp_feat.shape[-1]

        # Average support embeddings per class (prototype)
        # supp_feat shape: (N*K, C, H', W') -> (N, K, C, H', W') -> mean over K
        proto = supp_feat.view(N, k_shot, self.base_ch, H_, W_).mean(dim=1)  # (N, C, H', W')

        # Compute all (query, class) pairs: M*N pairs
        # Expand: query_feat (M, C, H, W) -> (M, N, C, H, W)
        qf = query_feat.unsqueeze(1).expand(M, N, self.base_ch, H_, W_)  # (M, N, C, H, W)
        # Expand: proto (N, C, H, W) -> (M, N, C, H, W)
        pf = proto.unsqueeze(0).expand(M, N, self.base_ch, H_, W_)

        # Concatenate along channel dim: (M*N, 2C, H, W)
        pairs = torch.cat([qf, pf], dim=2)  # (M, N, 2C, H, W)
        pairs = pairs.view(M * N, 2 * self.base_ch, H_, W_)

        # Compute relation scores
        scores = self.relation(pairs)  # (M*N, 1)
        return scores.view(M, N)  # (M, N)


def build_relationnet() -> nn.Module:
    return RelationNet(in_channels=1, base_ch=32, spatial=4)


def example_input_relationnet() -> list[torch.Tensor]:
    """2-way 1-shot: 2 support images + 2 query images, each 1x16x16."""
    torch.manual_seed(20)
    N, K, M = 2, 1, 2
    support = torch.randn(N * K, 1, 16, 16)
    query = torch.randn(M, 1, 16, 16)
    return [support, query]


class RelationNetWrapper(nn.Module):
    """Wrapper so tl.trace gets a single forward call (absorbs n_way/k_shot)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = RelationNet(in_channels=1, base_ch=32, spatial=4)

    def forward(self, support: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.net(support, query, n_way=2, k_shot=1)


def build_relationnet_wrapped() -> nn.Module:
    return RelationNetWrapper()


MENAGERIE_ENTRIES = [
    (
        "Relation Network (Conv4, Few-Shot)",
        "build_relationnet_wrapped",
        "example_input_relationnet",
        "2018",
        "DC",
    ),
]
