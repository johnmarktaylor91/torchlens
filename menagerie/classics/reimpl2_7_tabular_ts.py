"""Compact tabular and time-series reimplementations for shard 7.

Sources checked: TabNet attentive sequential feature selection, RTDL
FT-Transformer and ResNet baselines for tabular data, lucidrains'
tab-transformer implementation notes, and Google's TSMixer all-MLP time/feature
mixing architecture.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GLUBlock(nn.Module):
    """Gated linear unit block used in TabNet feature transformers."""

    def __init__(self, dim: int) -> None:
        """Initialize GLU projection.

        Parameters
        ----------
        dim:
            Feature dimension.
        """
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim * 2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated feature transform.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        Tensor
            Gated output.
        """
        a, b = self.norm(self.fc(x)).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class TabNetCompact(nn.Module):
    """Compact TabNet with attentive masks over sequential decision steps."""

    def __init__(self, features: int = 12, hidden: int = 24, steps: int = 3) -> None:
        """Initialize compact TabNet.

        Parameters
        ----------
        features:
            Number of tabular features.
        hidden:
            Hidden decision width.
        steps:
            Number of decision steps.
        """
        super().__init__()
        self.steps = steps
        self.embed = nn.Linear(features, hidden)
        self.feature_blocks = nn.ModuleList([GLUBlock(hidden) for _ in range(steps)])
        self.attentive = nn.ModuleList([nn.Linear(hidden, features) for _ in range(steps)])
        self.proj = nn.Linear(features, hidden)
        self.head = nn.Linear(hidden, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Run sequential attentive feature selection.

        Parameters
        ----------
        x:
            Tabular feature matrix.

        Returns
        -------
        Tensor
            Prediction logits.
        """
        prior = torch.ones_like(x)
        state = self.embed(x)
        aggregate = torch.zeros_like(state)
        for block, attn in zip(self.feature_blocks, self.attentive, strict=True):
            mask = torch.softmax(attn(state) * prior, dim=-1)
            selected = self.proj(mask * x)
            decision = F.relu(block(state + selected))
            aggregate = aggregate + decision
            prior = prior * (1.5 - mask)
            state = decision
        return self.head(aggregate)


class TabResBlock(nn.Module):
    """Fully connected residual block for tabular data."""

    def __init__(self, dim: int) -> None:
        """Initialize residual MLP block.

        Parameters
        ----------
        dim:
            Feature dimension.
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual tabular MLP block.

        Parameters
        ----------
        x:
            Hidden features.

        Returns
        -------
        Tensor
            Updated hidden features.
        """
        return x + self.fc2(F.relu(self.fc1(self.norm(x))))


class TabResNet(nn.Module):
    """RTDL-style residual MLP for tabular prediction."""

    def __init__(self, features: int = 12, hidden: int = 32) -> None:
        """Initialize compact TabResNet.

        Parameters
        ----------
        features:
            Number of input features.
        hidden:
            Hidden width.
        """
        super().__init__()
        self.inp = nn.Linear(features, hidden)
        self.blocks = nn.Sequential(TabResBlock(hidden), TabResBlock(hidden), TabResBlock(hidden))
        self.head = nn.Linear(hidden, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Predict from tabular features.

        Parameters
        ----------
        x:
            Tabular feature matrix.

        Returns
        -------
        Tensor
            Prediction logits.
        """
        return self.head(F.relu(self.blocks(self.inp(x))))


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data."""

    def __init__(self, features: int = 12, dim: int = 32, heads: int = 4) -> None:
        """Initialize compact FT-Transformer.

        Parameters
        ----------
        features:
            Number of scalar input features.
        dim:
            Token dimension.
        heads:
            Attention heads.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(features, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(features, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(dim, 3)

    def forward(self, x: Tensor) -> Tensor:
        """Tokenize features and classify from the CLS token.

        Parameters
        ----------
        x:
            Numeric tabular features.

        Returns
        -------
        Tensor
            Prediction logits.
        """
        batch = x.shape[0]
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        tokens = torch.cat((self.cls.expand(batch, -1, -1), tokens), dim=1)
        return self.head(self.encoder(tokens)[:, 0])


class TSMixerBlock(nn.Module):
    """TSMixer block with temporal and feature MLP mixing."""

    def __init__(self, length: int, channels: int) -> None:
        """Initialize mixer block.

        Parameters
        ----------
        length:
            Time length.
        channels:
            Variable/channel count.
        """
        super().__init__()
        self.time_norm = nn.LayerNorm(channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(length, length * 2), nn.GELU(), nn.Linear(length * 2, length)
        )
        self.feature_norm = nn.LayerNorm(channels)
        self.feature_mlp = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply time and feature mixing.

        Parameters
        ----------
        x:
            Time series ``(batch, time, channels)``.

        Returns
        -------
        Tensor
            Mixed representation.
        """
        x = x + self.time_mlp(self.time_norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.feature_mlp(self.feature_norm(x))


class TSMixer(nn.Module):
    """All-MLP TSMixer forecaster."""

    def __init__(self, length: int = 16, channels: int = 5, horizon: int = 4) -> None:
        """Initialize compact TSMixer.

        Parameters
        ----------
        length:
            Input history length.
        channels:
            Number of variables.
        horizon:
            Forecast horizon.
        """
        super().__init__()
        self.blocks = nn.Sequential(TSMixerBlock(length, channels), TSMixerBlock(length, channels))
        self.temporal_head = nn.Linear(length, horizon)

    def forward(self, x: Tensor) -> Tensor:
        """Forecast future values.

        Parameters
        ----------
        x:
            Input time series.

        Returns
        -------
        Tensor
            Forecast ``(batch, horizon, channels)``.
        """
        y = self.blocks(x).transpose(1, 2)
        return self.temporal_head(y).transpose(1, 2)


def build_tabnet() -> nn.Module:
    """Build compact TabNet."""
    return TabNetCompact()


def build_tabresnet() -> nn.Module:
    """Build compact TabResNet."""
    return TabResNet()


def build_fttransformer() -> nn.Module:
    """Build compact FT-Transformer."""
    return FTTransformer()


def build_tsmixer() -> nn.Module:
    """Build compact TSMixer."""
    return TSMixer()


def example_tabular() -> Tensor:
    """Return a tabular feature batch."""
    return torch.randn(4, 12)


def example_ts() -> Tensor:
    """Return a multivariate time-series batch."""
    return torch.randn(2, 16, 5)


MENAGERIE_ENTRIES = [
    ("TabNet", "build_tabnet", "example_tabular", "2019", "DC"),
    ("TabResNet", "build_tabresnet", "example_tabular", "2021", "DC"),
    ("FTTransformer-lucidrains", "build_fttransformer", "example_tabular", "2021", "DC"),
    ("tsmixer", "build_tsmixer", "example_ts", "2023", "DC"),
]
