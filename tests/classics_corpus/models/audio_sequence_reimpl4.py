"""Compact faithful audio and memory-sequence classics.

Paper: SincNet, LEAF, Deep-FSMN, and MIMN.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv1d(nn.Module):
    """SincNet first layer with learnable low/high cutoff frequencies."""

    def __init__(
        self, out_channels: int = 8, kernel_size: int = 63, sample_rate: int = 16000
    ) -> None:
        """Initialize the parameterized sinc filterbank.

        Parameters
        ----------
        out_channels:
            Number of band-pass filters.
        kernel_size:
            Filter length.
        sample_rate:
            Audio sample rate.
        """

        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        low = torch.linspace(80.0, 3000.0, out_channels)
        band = torch.full((out_channels,), 400.0)
        self.low_hz = nn.Parameter(low)
        self.band_hz = nn.Parameter(band)
        n = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        self.register_buffer("n", n)
        self.register_buffer("window", torch.hamming_window(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Filter waveform with learned analytic band-pass filters.

        Parameters
        ----------
        x:
            Waveform tensor ``(batch, 1, time)``.

        Returns
        -------
        torch.Tensor
            Filterbank activations.
        """

        low = self.low_hz.abs().clamp(30.0, self.sample_rate / 2 - 100.0)
        high = (low + self.band_hz.abs()).clamp(80.0, self.sample_rate / 2 - 1.0)
        n = self.n[None, :]
        low_arg = 2 * math.pi * low[:, None] * n / self.sample_rate
        high_arg = 2 * math.pi * high[:, None] * n / self.sample_rate
        band = (torch.sin(high_arg) - torch.sin(low_arg)) / (n + 1e-6)
        center = (2 * (high - low) / self.sample_rate)[:, None]
        band = torch.where(n.abs() < 1e-6, center, band)
        filters = band * self.window[None, :]
        filters = filters / filters.abs().sum(dim=1, keepdim=True).clamp_min(1e-6)
        return F.conv1d(x, filters.unsqueeze(1), padding=self.kernel_size // 2)


class SincNetClassifier(nn.Module):
    """SincNet raw-waveform classifier."""

    def __init__(self, classes: int = 4) -> None:
        """Initialize SincNet.

        Parameters
        ----------
        classes:
            Number of output classes.
        """

        super().__init__()
        self.sinc = SincConv1d()
        self.pool = nn.MaxPool1d(4)
        self.conv = nn.Conv1d(8, 16, 5, padding=2)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(16, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify raw waveform.

        Parameters
        ----------
        x:
            Waveform tensor ``(batch, time)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        feat = self.sinc(x.unsqueeze(1)).abs()
        feat = self.pool(feat)
        feat = F.relu(self.conv(feat))
        return self.head(self.avg(feat).flatten(1))


class LeafFrontend(nn.Module):
    """LEAF frontend: learnable Gabor-like filters, Gaussian pooling, and PCEN."""

    def __init__(self, filters: int = 8, kernel_size: int = 63) -> None:
        """Initialize the LEAF frontend.

        Parameters
        ----------
        filters:
            Number of learnable filters.
        kernel_size:
            Filter length.
        """

        super().__init__()
        self.real = nn.Conv1d(1, filters, kernel_size, padding=kernel_size // 2, bias=False)
        self.imag = nn.Conv1d(1, filters, kernel_size, padding=kernel_size // 2, bias=False)
        self.pool = nn.Conv1d(filters, filters, 9, padding=4, groups=filters, bias=False)
        self.alpha = nn.Parameter(torch.full((filters,), 0.98))
        self.delta = nn.Parameter(torch.full((filters,), 2.0))
        self.root = nn.Parameter(torch.full((filters,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LEAF PCEN features.

        Parameters
        ----------
        x:
            Waveform tensor ``(batch, time)``.

        Returns
        -------
        torch.Tensor
            PCEN frontend features.
        """

        wave = x.unsqueeze(1)
        energy = self.real(wave).pow(2) + self.imag(wave).pow(2)
        smooth = F.softplus(self.pool(energy)).clamp_min(1e-4)
        alpha = self.alpha[None, :, None].sigmoid()
        delta = F.softplus(self.delta)[None, :, None]
        root = self.root[None, :, None].sigmoid()
        return (energy / smooth.pow(alpha) + delta).pow(root) - delta.pow(root)


class LeafClassifier(nn.Module):
    """Audio classifier with LEAF frontend."""

    def __init__(self, classes: int = 4) -> None:
        """Initialize LEAF classifier.

        Parameters
        ----------
        classes:
            Number of output classes.
        """

        super().__init__()
        self.frontend = LeafFrontend()
        self.head = nn.Linear(8, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify waveform through LEAF features.

        Parameters
        ----------
        x:
            Waveform tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.head(self.frontend(x).mean(dim=-1))


class DFSMNBlock(nn.Module):
    """Deep-FSMN memory block with projection memory and residual skip."""

    def __init__(self, dim: int = 32, memory: int = 4) -> None:
        """Initialize a DFSMN block.

        Parameters
        ----------
        dim:
            Feature width.
        memory:
            Number of left/right memory taps.
        """

        super().__init__()
        self.proj = nn.Linear(dim, dim // 2)
        self.expand = nn.Linear(dim // 2, dim)
        self.memory_conv = nn.Conv1d(
            dim // 2, dim // 2, memory * 2 + 1, padding=memory, groups=dim // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward sequential memory and residual skip.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        torch.Tensor
            Updated sequence.
        """

        p = F.relu(self.proj(x))
        mem = self.memory_conv(p.transpose(1, 2)).transpose(1, 2)
        return x + self.expand(p + mem)


class DFSMNNet(nn.Module):
    """Deep feedforward sequential memory network."""

    def __init__(self, features: int = 20, classes: int = 4, dim: int = 32) -> None:
        """Initialize DFSMN.

        Parameters
        ----------
        features:
            Input feature width.
        classes:
            Number of output classes.
        dim:
            Hidden width.
        """

        super().__init__()
        self.inp = nn.Linear(features, dim)
        self.blocks = nn.ModuleList([DFSMNBlock(dim) for _ in range(3)])
        self.head = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify sequence with stacked FSMN memory blocks.

        Parameters
        ----------
        x:
            Sequence features ``(batch, time, features)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        h = self.inp(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h.mean(dim=1))


class MIMNNet(nn.Module):
    """Multi-channel user-interest memory network."""

    def __init__(self, items: int = 64, dim: int = 32, slots: int = 4) -> None:
        """Initialize MIMN.

        Parameters
        ----------
        items:
            Item vocabulary size.
        dim:
            Embedding width.
        slots:
            External memory slot count.
        """

        super().__init__()
        self.embed = nn.Embedding(items, dim)
        self.memory_seed = nn.Parameter(torch.zeros(slots, dim))
        self.write_gate = nn.Linear(dim * 2, dim)
        self.erase = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)
        self.head = nn.Linear(dim * 2, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Induce multi-interest memory slots from a behavior sequence.

        Parameters
        ----------
        ids:
            Item id sequence.

        Returns
        -------
        torch.Tensor
            Click logit.
        """

        emb = self.embed(ids)
        memory = self.memory_seed.unsqueeze(0).expand(ids.shape[0], -1, -1)
        for t in range(ids.shape[1]):
            item = emb[:, t : t + 1]
            attn = torch.softmax(torch.matmul(item, memory.transpose(-1, -2)), dim=-1)
            read = torch.matmul(attn, memory)
            gate = torch.sigmoid(self.write_gate(torch.cat([item, read], dim=-1)))
            erase = torch.sigmoid(self.erase(item))
            memory = memory * (1 - attn.transpose(-1, -2) * erase) + attn.transpose(-1, -2) * gate
        weights = torch.softmax(self.score(memory), dim=1)
        user = (weights * memory).sum(dim=1)
        target = emb[:, -1]
        return self.head(torch.cat([user, target], dim=-1))


def build_sincnet() -> nn.Module:
    """Build SincNet.

    Returns
    -------
    nn.Module
        Compact SincNet model.
    """

    return SincNetClassifier()


def build_leaf() -> nn.Module:
    """Build LEAF.

    Returns
    -------
    nn.Module
        Compact LEAF model.
    """

    return LeafClassifier()


def build_dfsmn() -> nn.Module:
    """Build DFSMN.

    Returns
    -------
    nn.Module
        Compact DFSMN model.
    """

    return DFSMNNet()


def build_mimn() -> nn.Module:
    """Build MIMN.

    Returns
    -------
    nn.Module
        Compact MIMN model.
    """

    return MIMNNet()


def example_wave() -> torch.Tensor:
    """Return a compact waveform input.

    Returns
    -------
    torch.Tensor
        Waveform tensor.
    """

    return torch.randn(1, 512)


def example_sequence() -> torch.Tensor:
    """Return compact acoustic features.

    Returns
    -------
    torch.Tensor
        Sequence feature tensor.
    """

    return torch.randn(1, 16, 20)


def example_items() -> torch.Tensor:
    """Return compact behavior item ids.

    Returns
    -------
    torch.Tensor
        Item-id tensor.
    """

    return torch.randint(0, 64, (1, 12))


MENAGERIE_ENTRIES = [
    ("SincNet", "build_sincnet", "example_wave", "2018", "speech/audio"),
    ("LEAF", "build_leaf", "example_wave", "2021", "speech/audio"),
    ("DFSMN", "build_dfsmn", "example_sequence", "2018", "speech/audio"),
    ("MIMN", "build_mimn", "example_items", "2019", "recommender/sequence"),
]
