"""Compact gradient-based meta-learning classics.

Papers: Finn et al., 2017, "Model-Agnostic Meta-Learning"; Li et al., 2017,
"Meta-SGD"; Rusu et al., 2019, "Meta-Learning with Latent Embedding
Optimization".

The registered targets keep their distinctive primitives: MAML's one-step
support-set adaptation of a Conv4 classifier, Meta-SGD's learned per-parameter
update direction/rate, and LEO's relation-encoded latent parameter generator.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class Conv4Encoder(nn.Module):
    """Four-block few-shot ConvNet encoder."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize four convolution/batchnorm/pool blocks."""

        super().__init__()
        blocks = []
        in_channels = 3
        for _ in range(4):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = channels
        self.net = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Encode images into few-shot features."""

        return self.net(x).flatten(1)


class MAMLConv4(nn.Module):
    """Compact MAML/Conv4 model with differentiable one-step classifier update."""

    def __init__(self, ways: int = 5, feature_dim: int = 16) -> None:
        """Initialize Conv4 encoder and base classifier."""

        super().__init__()
        self.encoder = Conv4Encoder()
        self.weight = nn.Parameter(torch.randn(ways, feature_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(ways))
        self.inner_lr = nn.Parameter(torch.tensor(0.4))

    def forward(self, support_x: Tensor, support_y: Tensor, query_x: Tensor) -> Tensor:
        """Adapt classifier on support examples and classify query examples."""

        support = self.encoder(support_x)
        query = self.encoder(query_x)
        logits = support @ self.weight.t() + self.bias
        probs = torch.softmax(logits, dim=-1)
        targets = F.one_hot(support_y, num_classes=self.weight.shape[0]).to(probs.dtype)
        residual = (probs - targets) / support.shape[0]
        grad_w = residual.t() @ support
        grad_b = residual.sum(dim=0)
        fast_w = self.weight - F.softplus(self.inner_lr) * grad_w
        fast_b = self.bias - F.softplus(self.inner_lr) * grad_b
        return query @ fast_w.t() + fast_b


class MetaSGD(nn.Module):
    """Meta-SGD learner with learned per-weight learning-rate tensor."""

    def __init__(self, ways: int = 5, feature_dim: int = 16) -> None:
        """Initialize base parameters and learned alpha update rates."""

        super().__init__()
        self.encoder = Conv4Encoder()
        self.weight = nn.Parameter(torch.randn(ways, feature_dim) * 0.02)
        self.alpha_w = nn.Parameter(torch.full((ways, feature_dim), 0.1))
        self.bias = nn.Parameter(torch.zeros(ways))
        self.alpha_b = nn.Parameter(torch.full((ways,), 0.1))

    def forward(self, support_x: Tensor, support_y: Tensor, query_x: Tensor) -> Tensor:
        """Apply a learned-direction Meta-SGD inner update."""

        support = self.encoder(support_x)
        query = self.encoder(query_x)
        logits = support @ self.weight.t() + self.bias
        probs = torch.softmax(logits, dim=-1)
        targets = F.one_hot(support_y, num_classes=self.weight.shape[0]).to(probs.dtype)
        residual = (probs - targets) / support.shape[0]
        grad_w = residual.t() @ support
        grad_b = residual.sum(dim=0)
        fast_w = self.weight - self.alpha_w * grad_w
        fast_b = self.bias - self.alpha_b * grad_b
        return query @ fast_w.t() + fast_b


class LEO(nn.Module):
    """Latent Embedding Optimization with relation encoder and decoder."""

    def __init__(self, ways: int = 5, feature_dim: int = 16, latent_dim: int = 16) -> None:
        """Initialize encoder, relation network, and latent parameter decoder."""

        super().__init__()
        self.encoder = Conv4Encoder()
        self.relation = nn.Sequential(
            nn.Linear(feature_dim * 2, 64), nn.ReLU(), nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Linear(latent_dim, feature_dim)
        self.latent_lr = nn.Parameter(torch.tensor(0.3))
        self.ways = ways

    def forward(self, support_x: Tensor, support_y: Tensor, query_x: Tensor) -> Tensor:
        """Generate task-specific weights by optimizing in latent space."""

        support = self.encoder(support_x)
        query = self.encoder(query_x)
        prototypes = []
        for cls in range(self.ways):
            mask = (support_y == cls).float().unsqueeze(-1)
            prototypes.append((support * mask).sum(0) / mask.sum().clamp_min(1.0))
        proto = torch.stack(prototypes)
        pairs = torch.cat(
            [
                proto.unsqueeze(1).expand(-1, self.ways, -1),
                proto.unsqueeze(0).expand(self.ways, -1, -1),
            ],
            dim=-1,
        )
        latent = self.relation(pairs).mean(dim=1)
        weights = self.decoder(latent)
        logits = support @ weights.t()
        probs = torch.softmax(logits, dim=-1)
        targets = F.one_hot(support_y, num_classes=self.ways).to(probs.dtype)
        residual = (probs - targets) / support.shape[0]
        grad_weights = residual.t() @ support
        grad_z = grad_weights @ self.decoder.weight
        weights = self.decoder(latent - F.softplus(self.latent_lr) * grad_z)
        return query @ weights.t()


def build_maml() -> nn.Module:
    """Build compact MAML Conv4."""

    return MAMLConv4().eval()


def build_metasgd() -> nn.Module:
    """Build compact Meta-SGD Conv4."""

    return MetaSGD().eval()


def build_leo() -> nn.Module:
    """Build compact LEO."""

    return LEO().eval()


def example_episode() -> tuple[Tensor, Tensor, Tensor]:
    """Return a tiny 5-way support/query episode."""

    return torch.randn(5, 3, 16, 16), torch.arange(5), torch.randn(3, 3, 16, 16)


MENAGERIE_ENTRIES = [
    ("MAML-Conv4-C5W1", "build_maml", "example_episode", "2017", "META"),
    ("maml_conv4", "build_maml", "example_episode", "2017", "META"),
    ("Meta-SGD", "build_metasgd", "example_episode", "2017", "META"),
    ("LEO", "build_leo", "example_episode", "2019", "META"),
]
