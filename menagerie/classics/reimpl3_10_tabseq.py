"""Compact tabular, memory, retention, and recommender classics.

This module covers dependency-gated models whose core primitives are
lightweight enough to reconstruct directly: TabPFN prior-data fitted in-context
tabular Transformer, Titans memory-as-context, RetNet multi-scale retention,
StockMixer indicator/time/stock MLP mixing, TSception temporal-spatial EEG
inception, TorchHD centroid hypervector classifier, YouTubeDNN candidate
retrieval tower, and basic FCN/HOFM/SRN components.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabPFNTiny(nn.Module):
    """Prior-data fitted tabular Transformer for classification/regression."""

    def __init__(self, regression: bool = False) -> None:
        """Initialize feature, target, and query-token encoder.

        Parameters
        ----------
        regression:
            Whether to output one regression value instead of class logits.
        """

        super().__init__()
        self.regression = regression
        self.feature = nn.Linear(6, 24)
        self.target = nn.Embedding(4, 24)
        layer = nn.TransformerEncoderLayer(24, 4, 48, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.query = nn.Parameter(torch.randn(1, 1, 24))
        self.head = nn.Linear(24, 1 if regression else 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Infer labels for query rows from a support table.

        Parameters
        ----------
        x:
            Tensor shaped ``(batch, rows, 7)`` with six features and one support
            target column; negative target denotes query rows.

        Returns
        -------
        torch.Tensor
            Query predictions.
        """

        feats = self.feature(x[..., :6])
        y = x[..., 6].round().long().clamp(0, 3)
        support = feats + self.target(y)
        query = self.query.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([support, query], dim=1))
        return self.head(encoded[:, -1])


class TitansMACTiny(nn.Module):
    """Titans memory-as-context block with surprise-gated neural memory."""

    def __init__(self) -> None:
        """Initialize memory, attention, and output layers."""

        super().__init__()
        self.embed = nn.Embedding(64, 24)
        self.surprise = nn.Linear(24, 1)
        self.memory_write = nn.Linear(24, 24)
        self.attn = nn.MultiheadAttention(24, 4, batch_first=True)
        self.persistent = nn.Parameter(torch.randn(1, 2, 24))
        self.head = nn.Linear(24, 64)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Run memory-as-context sequence modeling.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        x = self.embed(ids)
        mem = x.new_zeros(x.shape[0], 1, x.shape[-1])
        contexts: list[torch.Tensor] = []
        for step in range(x.shape[1]):
            token = x[:, step : step + 1]
            gate = torch.sigmoid(self.surprise(token))
            mem = 0.85 * mem + gate * torch.tanh(self.memory_write(token))
            context = torch.cat([self.persistent.expand(x.shape[0], -1, -1), mem, token], dim=1)
            y, _ = self.attn(token, context, context, need_weights=False)
            contexts.append(y.squeeze(1))
        return self.head(torch.stack(contexts, dim=1))


class RetNetTiny(nn.Module):
    """Retentive Network layer with multi-scale recurrent retention."""

    def __init__(self) -> None:
        """Initialize projections and retention decays."""

        super().__init__()
        self.embed = nn.Embedding(64, 24)
        self.q = nn.Linear(24, 24)
        self.k = nn.Linear(24, 24)
        self.v = nn.Linear(24, 24)
        self.decay = nn.Parameter(torch.tensor([0.70, 0.85, 0.95]))
        self.out = nn.Linear(24 * 3, 24)
        self.head = nn.Linear(24, 64)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Apply recurrent multi-scale retention.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        x = self.embed(ids)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        states = [x.new_zeros(x.shape[0], 24) for _ in range(3)]
        outs: list[torch.Tensor] = []
        for step in range(x.shape[1]):
            pieces = []
            for idx, decay in enumerate(torch.sigmoid(self.decay)):
                states[idx] = decay * states[idx] + k[:, step] * v[:, step]
                pieces.append(q[:, step] * states[idx])
            outs.append(torch.cat(pieces, dim=-1))
        return self.head(self.out(torch.stack(outs, dim=1)))


class StockMixerTiny(nn.Module):
    """StockMixer with indicator, temporal, and stock-axis MLP mixing."""

    def __init__(self) -> None:
        """Initialize axis-wise mixers."""

        super().__init__()
        self.indicator = nn.Linear(5, 5)
        self.time = nn.Linear(6, 6)
        self.stock = nn.Linear(4, 4)
        self.head = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast returns from stock/time/indicator tensor.

        Parameters
        ----------
        x:
            Tensor shaped ``(batch, stocks, time, indicators)``.

        Returns
        -------
        torch.Tensor
            Per-stock predictions.
        """

        y = x + F.gelu(self.indicator(x))
        y = y + F.gelu(self.time(y.transpose(2, 3)).transpose(2, 3))
        y = y + F.gelu(self.stock(y.transpose(1, 3)).transpose(1, 3))
        return self.head(y.mean(dim=2)).squeeze(-1)


class TSceptionTiny(nn.Module):
    """TSception temporal-spatial inception EEG classifier."""

    def __init__(self) -> None:
        """Initialize temporal and spatial convolution branches."""

        super().__init__()
        self.t1 = nn.Conv2d(1, 4, (1, 3), padding=(0, 1))
        self.t2 = nn.Conv2d(1, 4, (1, 5), padding=(0, 2))
        self.s1 = nn.Conv2d(8, 8, (4, 1))
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify EEG from temporal-spatial inception features.

        Parameters
        ----------
        x:
            EEG tensor shaped ``(batch, 1, channels, time)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        y = torch.cat([F.elu(self.t1(x)), F.elu(self.t2(x))], dim=1)
        y = F.elu(self.s1(y))
        return self.head(y.mean(dim=(2, 3)))


class HyperbolicHOFM(nn.Module):
    """Higher-order factorization machine with multiplicative interactions."""

    def __init__(self) -> None:
        """Initialize linear and factor parameters."""

        super().__init__()
        self.linear = nn.Linear(8, 1)
        self.factor = nn.Parameter(torch.randn(8, 6) * 0.1)
        self.third = nn.Parameter(torch.randn(8, 4) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate second- and third-order factor interactions.

        Parameters
        ----------
        x:
            Dense feature tensor.

        Returns
        -------
        torch.Tensor
            Prediction.
        """

        second = 0.5 * ((x @ self.factor).pow(2) - (x.pow(2) @ self.factor.pow(2))).sum(
            dim=-1, keepdim=True
        )
        third = (x @ self.third).pow(3).sum(dim=-1, keepdim=True) / 6.0
        return self.linear(x) + second + third


class TorchHDCentroid(nn.Module):
    """TorchHD centroid classifier using random hypervector binding."""

    def __init__(self) -> None:
        """Initialize item memory and class centroids."""

        super().__init__()
        self.item = nn.Parameter(torch.sign(torch.randn(8, 64)), requires_grad=False)
        self.centroids = nn.Parameter(torch.sign(torch.randn(3, 64)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features as a hypervector and compare to centroids.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Similarity logits.
        """

        hv = torch.tanh(x @ self.item)
        return hv @ self.centroids.t() / 64.0


class YouTubeDNNTiny(nn.Module):
    """YouTubeDNN two-tower candidate generation model."""

    def __init__(self) -> None:
        """Initialize user history tower and item tower."""

        super().__init__()
        self.video = nn.Embedding(128, 16)
        self.user = nn.Sequential(nn.Linear(16, 24), nn.ReLU(), nn.Linear(24, 16))
        self.item = nn.Embedding(32, 16)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Score candidate videos from watched-video history.

        Parameters
        ----------
        ids:
            Watched video ids shaped ``(batch, history)``.

        Returns
        -------
        torch.Tensor
            Candidate logits.
        """

        user = self.user(self.video(ids).mean(dim=1))
        return user @ self.item.weight.t()


class FCNTiny(nn.Module):
    """TorchPhysics-style fully connected PINN/FCN."""

    def __init__(self) -> None:
        """Initialize coordinate MLP."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map coordinates to scalar field values.

        Parameters
        ----------
        x:
            Coordinate tensor.

        Returns
        -------
        torch.Tensor
            Scalar field.
        """

        return self.net(x)


class SRNTiny(nn.Module):
    """Simple recurrent network with explicit context state."""

    def __init__(self) -> None:
        """Initialize input, recurrent, and output projections."""

        super().__init__()
        self.inp = nn.Linear(5, 10)
        self.rec = nn.Linear(10, 10, bias=False)
        self.out = nn.Linear(10, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run simple recurrent dynamics.

        Parameters
        ----------
        x:
            Sequence features.

        Returns
        -------
        torch.Tensor
            Final logits.
        """

        h = x.new_zeros(x.shape[0], 10)
        for step in range(x.shape[1]):
            h = torch.tanh(self.inp(x[:, step]) + self.rec(h))
        return self.out(h)


def example_tabpfn() -> torch.Tensor:
    """Return compact TabPFN support/query table.

    Returns
    -------
    torch.Tensor
        Tabular context tensor.
    """

    x = torch.randn(1, 5, 7)
    x[..., 6] = torch.tensor([[0.0, 1.0, 2.0, 1.0, 0.0]])
    return x


def example_tokens() -> torch.Tensor:
    """Return token sequence.

    Returns
    -------
    torch.Tensor
        Token ids.
    """

    return torch.randint(0, 64, (1, 8))


def example_stock() -> torch.Tensor:
    """Return stock/time/indicator tensor.

    Returns
    -------
    torch.Tensor
        StockMixer input.
    """

    return torch.randn(1, 4, 6, 5)


def example_eeg() -> torch.Tensor:
    """Return EEG tensor.

    Returns
    -------
    torch.Tensor
        EEG sample.
    """

    return torch.randn(1, 1, 4, 16)


def example_dense8() -> torch.Tensor:
    """Return dense feature tensor.

    Returns
    -------
    torch.Tensor
        Dense features.
    """

    return torch.randn(2, 8)


def example_youtube() -> torch.Tensor:
    """Return watched-video ids.

    Returns
    -------
    torch.Tensor
        Video id history.
    """

    return torch.randint(0, 128, (1, 6))


def example_coords() -> torch.Tensor:
    """Return coordinate tensor.

    Returns
    -------
    torch.Tensor
        Coordinate features.
    """

    return torch.randn(4, 3)


def example_srn() -> torch.Tensor:
    """Return SRN sequence.

    Returns
    -------
    torch.Tensor
        Sequence tensor.
    """

    return torch.randn(1, 6, 5)


def build_tabpfn_classifier_v2() -> nn.Module:
    """Build TabPFN v2 classifier.

    Returns
    -------
    nn.Module
        TabPFN classifier.
    """

    return TabPFNTiny(False)


def build_tabpfn_regressor_v2() -> nn.Module:
    """Build TabPFN v2 regressor.

    Returns
    -------
    nn.Module
        TabPFN regressor.
    """

    return TabPFNTiny(True)


def build_titans_mac() -> nn.Module:
    """Build Titans MAC model.

    Returns
    -------
    nn.Module
        Titans memory-as-context model.
    """

    return TitansMACTiny()


def build_retnet() -> nn.Module:
    """Build RetNet layer model.

    Returns
    -------
    nn.Module
        Retention model.
    """

    return RetNetTiny()


def build_stockmixer() -> nn.Module:
    """Build StockMixer model.

    Returns
    -------
    nn.Module
        StockMixer.
    """

    return StockMixerTiny()


def build_tsception() -> nn.Module:
    """Build TSception model.

    Returns
    -------
    nn.Module
        TSception classifier.
    """

    return TSceptionTiny()


def build_hofm() -> nn.Module:
    """Build higher-order factorization machine.

    Returns
    -------
    nn.Module
        HOFM model.
    """

    return HyperbolicHOFM()


def build_torchhd_centroid() -> nn.Module:
    """Build TorchHD centroid classifier.

    Returns
    -------
    nn.Module
        Centroid classifier.
    """

    return TorchHDCentroid()


def build_youtube_dnn() -> nn.Module:
    """Build YouTubeDNN candidate generator.

    Returns
    -------
    nn.Module
        YouTubeDNN model.
    """

    return YouTubeDNNTiny()


def build_torchphysics_fcn() -> nn.Module:
    """Build TorchPhysics FCN.

    Returns
    -------
    nn.Module
        Coordinate MLP.
    """

    return FCNTiny()


def build_srn() -> nn.Module:
    """Build simple recurrent network.

    Returns
    -------
    nn.Module
        SRN model.
    """

    return SRNTiny()


MENAGERIE_ENTRIES = [
    ("StockMixer", "build_stockmixer", "example_stock", "2024", "time-series/finance"),
    (
        "TabPFNClassifier v2",
        "build_tabpfn_classifier_v2",
        "example_tabpfn",
        "2025",
        "tabular/foundation",
    ),
    (
        "TabPFNv2-transformer",
        "build_tabpfn_classifier_v2",
        "example_tabpfn",
        "2025",
        "tabular/foundation",
    ),
    (
        "TabPFNRegressor v2",
        "build_tabpfn_regressor_v2",
        "example_tabpfn",
        "2025",
        "tabular/foundation",
    ),
    ("titans", "build_titans_mac", "example_tokens", "2025", "sequence/memory"),
    ("Titans-MAC", "build_titans_mac", "example_tokens", "2025", "sequence/memory"),
    ("lucidrains:Titans", "build_titans_mac", "example_tokens", "2025", "sequence/memory"),
    ("TorchPhysics FCN", "build_torchphysics_fcn", "example_coords", "2021", "pinn"),
    ("TSception", "build_tsception", "example_eeg", "2020", "eeg"),
    ("HOFM", "build_hofm", "example_dense8", "2016", "factorization"),
    ("TorchHD_Centroid", "build_torchhd_centroid", "example_dense8", "2022", "hyperdimensional"),
    ("YouTubeDNN", "build_youtube_dnn", "example_youtube", "2016", "recommender"),
    ("retnet_layer", "build_retnet", "example_tokens", "2023", "sequence/retention"),
    ("retnet_torchscale", "build_retnet", "example_tokens", "2023", "sequence/retention"),
    ("SRN", "build_srn", "example_srn", "1990", "sequence/rnn"),
]
