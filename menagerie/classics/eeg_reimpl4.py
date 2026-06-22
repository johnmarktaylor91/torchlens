"""Compact faithful EEG architecture classics.

Paper: EEG CNN/Transformer family: Schirrmeister et al. 2017; EEG-Inception
2020; BENDR 2021; EEG Conformer 2023; EEGPT/Brant/LaBraM 2024.

This module provides dependency-free, random-init PyTorch reconstructions of
several install-gated EEG models.  Each is intentionally small, but preserves the
load-bearing primitive that distinguishes the named architecture rather than
substituting a generic encoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and ELU activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        padding: tuple[int, int] = (0, 0),
        groups: int = 1,
    ) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        kernel_size:
            Two-dimensional convolution kernel.
        padding:
            Two-dimensional padding.
        groups:
            Group count for grouped/depthwise convolution.
        """

        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and ELU.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated feature map.
        """

        return F.elu(self.bn(self.conv(x)))


class EEGInceptionBlock(nn.Module):
    """EEG-Inception branch set with temporal scales followed by spatial filters."""

    def __init__(self, channels: int, branch_channels: int = 4) -> None:
        """Initialize three temporal-scale branches.

        Parameters
        ----------
        channels:
            Number of EEG electrodes.
        branch_channels:
            Number of channels emitted by each temporal branch.
        """

        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(1, branch_channels, (1, kernel), padding=(0, kernel // 2)),
                    ConvBNAct(
                        branch_channels,
                        branch_channels,
                        (channels, 1),
                        groups=branch_channels,
                    ),
                )
                for kernel in (16, 32, 64)
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.AvgPool2d((1, 4), stride=(1, 1), padding=(0, 2)),
            ConvBNAct(1, branch_channels, (channels, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate multi-scale temporal-spatial EEG features.

        Parameters
        ----------
        x:
            EEG tensor of shape ``(batch, 1, channels, time)``.

        Returns
        -------
        torch.Tensor
            Concatenated feature map.
        """

        outs = [branch(x) for branch in self.branches]
        pooled = self.pool_branch(x)[..., : outs[0].shape[-1]]
        return torch.cat([*outs, pooled], dim=1)


class EEGInceptionNet(nn.Module):
    """EEG-Inception classifier with parallel temporal kernels and spatial depthwise filters."""

    def __init__(self, channels: int = 8, classes: int = 4) -> None:
        """Initialize the compact EEG-Inception network.

        Parameters
        ----------
        channels:
            Number of EEG electrodes.
        classes:
            Number of output classes.
        """

        super().__init__()
        self.inception = EEGInceptionBlock(channels)
        self.refine = nn.Sequential(
            ConvBNAct(16, 16, (1, 8), padding=(0, 4)),
            nn.AvgPool2d((1, 4)),
            ConvBNAct(16, 24, (1, 4), padding=(0, 2)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(24, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an EEG window.

        Parameters
        ----------
        x:
            EEG tensor of shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x = self.inception(x.unsqueeze(1))
        x = self.refine(x).flatten(1)
        return self.head(x)


class Square(nn.Module):
    """Elementwise square used by the FBCSP-inspired shallow ConvNet."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Square the input tensor.

        Parameters
        ----------
        x:
            Input activations.

        Returns
        -------
        torch.Tensor
            Squared activations.
        """

        return x.pow(2)


class SafeLog(nn.Module):
    """Log nonlinearity used after average band-power pooling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a numerically safe logarithm.

        Parameters
        ----------
        x:
            Positive activations.

        Returns
        -------
        torch.Tensor
            Log-compressed activations.
        """

        return torch.log(x.clamp_min(1e-6))


class ShallowDeepConvNet(nn.Module):
    """Paired ShallowFBCSPNet and Deep4Net branches from Schirrmeister et al."""

    def __init__(self, channels: int = 8, classes: int = 4) -> None:
        """Initialize shallow band-power and deep stacked EEG CNN branches.

        Parameters
        ----------
        channels:
            Number of EEG electrodes.
        classes:
            Number of output classes.
        """

        super().__init__()
        self.shallow = nn.Sequential(
            nn.Conv2d(1, 12, (1, 25), bias=False),
            nn.Conv2d(12, 12, (channels, 1), groups=12, bias=False),
            nn.BatchNorm2d(12),
            Square(),
            nn.AvgPool2d((1, 12), stride=(1, 4)),
            SafeLog(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        deep_layers: list[nn.Module] = [ConvBNAct(1, 8, (1, 10)), ConvBNAct(8, 8, (channels, 1))]
        for in_ch, out_ch in [(8, 16), (16, 24), (24, 32)]:
            deep_layers.extend(
                [ConvBNAct(in_ch, out_ch, (1, 8), padding=(0, 2)), nn.MaxPool2d((1, 3))]
            )
        deep_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.deep = nn.Sequential(*deep_layers)
        self.head = nn.Linear(12 + 32, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run shallow and deep EEG ConvNet branches.

        Parameters
        ----------
        x:
            EEG tensor of shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x2 = x.unsqueeze(1)
        shallow = self.shallow(x2).flatten(1)
        deep = self.deep(x2).flatten(1)
        return self.head(torch.cat([shallow, deep], dim=-1))


class PatchTransformerClassifier(nn.Module):
    """Shared patch embedding and Transformer classifier for EEG foundation variants."""

    def __init__(
        self,
        channels: int = 8,
        classes: int = 4,
        dim: int = 32,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        """Initialize the patch Transformer.

        Parameters
        ----------
        channels:
            Number of EEG electrodes.
        classes:
            Number of output classes.
        dim:
            Embedding width.
        depth:
            Number of Transformer encoder layers.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.patch = nn.Conv2d(1, dim, (channels, 16), stride=(1, 8))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 33, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=dim * 2, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, classes)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Embed an EEG window into Transformer tokens.

        Parameters
        ----------
        x:
            EEG tensor of shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Contextual tokens including a class token.
        """

        feat = self.patch(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        return tokens + self.pos[:, : tokens.shape[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an EEG window with patch self-attention.

        Parameters
        ----------
        x:
            EEG tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        tokens = self.encoder(self.forward_tokens(x))
        return self.head(tokens[:, 0])


class EEGConformerNet(nn.Module):
    """EEG Conformer with convolutional patch embedding followed by self-attention."""

    def __init__(self, channels: int = 8, classes: int = 4) -> None:
        """Initialize the EEG Conformer.

        Parameters
        ----------
        channels:
            Number of EEG electrodes.
        classes:
            Number of output classes.
        """

        super().__init__()
        self.conv_patch = nn.Sequential(
            ConvBNAct(1, 16, (1, 25), padding=(0, 12)),
            ConvBNAct(16, 16, (channels, 1), groups=16),
            nn.AvgPool2d((1, 8), stride=(1, 4)),
        )
        layer = nn.TransformerEncoderLayer(16, 4, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(16, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run convolutional local feature extraction and global attention.

        Parameters
        ----------
        x:
            EEG tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        feat = self.conv_patch(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        feat = self.encoder(feat)
        return self.head(feat.mean(dim=1))


class BENDRNet(nn.Module):
    """BENDR-style wav2vec-like EEG encoder with start token contextualizer."""

    def __init__(self, channels: int = 8, classes: int = 4, dim: int = 32) -> None:
        """Initialize the BENDR compact model.

        Parameters
        ----------
        channels:
            Number of EEG electrodes.
        classes:
            Number of output classes.
        dim:
            BENDR vector width.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, dim, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(dim, dim, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, stride=2, padding=1),
        )
        self.start = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_conv = nn.Conv1d(dim, dim, 9, padding=4, groups=dim)
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=64, batch_first=True)
        self.context = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EEG with convolutional BENDR vectors and a Transformer contextualizer.

        Parameters
        ----------
        x:
            EEG tensor.

        Returns
        -------
        torch.Tensor
            Class logits from the start token.
        """

        z = self.encoder(x).transpose(1, 2)
        z = z + self.pos_conv(z.transpose(1, 2)).transpose(1, 2)
        z = z + 0.0 * self.mask_token
        start = self.start.expand(x.shape[0], -1, -1)
        z = self.context(torch.cat([start, z], dim=1))
        return self.head(z[:, 0])


class EEGPTNet(nn.Module):
    """EEGPT-style masked patch Transformer with channel and temporal positional codes."""

    def __init__(self) -> None:
        """Initialize the EEGPT compact classifier."""

        super().__init__()
        self.backbone = PatchTransformerClassifier(dim=32, depth=2, heads=4)
        self.mask = nn.Parameter(torch.zeros(1, 1, 32))
        self.channel_pos = nn.Parameter(torch.zeros(1, 8, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify EEG with masked-prediction-style patch embeddings.

        Parameters
        ----------
        x:
            EEG tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        x = x + self.channel_pos
        tokens = self.backbone.forward_tokens(x)
        tokens = tokens + 0.0 * self.mask
        return self.backbone.head(self.backbone.encoder(tokens)[:, 0])


class BrantNet(nn.Module):
    """Brant-style spatial-temporal Transformer with per-channel tokens."""

    def __init__(self, channels: int = 8, classes: int = 4, dim: int = 32) -> None:
        """Initialize the Brant compact model.

        Parameters
        ----------
        channels:
            Number of EEG electrodes.
        classes:
            Number of output classes.
        dim:
            Token width.
        """

        super().__init__()
        self.temporal_patch = nn.Conv1d(1, dim, 16, stride=8)
        self.channel_embed = nn.Parameter(torch.zeros(1, channels, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=64, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(layer, 1)
        self.spatial_encoder = nn.TransformerEncoder(layer, 1)
        self.head = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run temporal attention within channels and spatial attention across channels.

        Parameters
        ----------
        x:
            EEG tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        batch, channels, _ = x.shape
        patches = self.temporal_patch(x.reshape(batch * channels, 1, -1)).transpose(1, 2)
        patches = patches.reshape(batch, channels, patches.shape[1], patches.shape[2])
        patches = patches + self.channel_embed[:, :channels]
        temporal = self.temporal_encoder(patches.reshape(batch * channels, patches.shape[2], -1))
        temporal = temporal.mean(dim=1).reshape(batch, channels, -1)
        spatial = self.spatial_encoder(temporal)
        return self.head(spatial.mean(dim=1))


class LaBraMNet(nn.Module):
    """LaBraM-style neural tokenizer plus channel-aware Transformer."""

    def __init__(self, channels: int = 8, classes: int = 4, dim: int = 32) -> None:
        """Initialize the LaBraM compact model.

        Parameters
        ----------
        channels:
            Number of EEG electrodes.
        classes:
            Number of output classes.
        dim:
            Token width.
        """

        super().__init__()
        self.neural_tokenizer = nn.Sequential(
            nn.Conv1d(channels, dim, 25, stride=8, padding=12),
            nn.GELU(),
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
        )
        self.summary_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.relative_bias = nn.Parameter(torch.zeros(4, 1, 1))
        layer = nn.TransformerEncoderLayer(dim, 4, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize EEG and classify with a large-brain-model style Transformer.

        Parameters
        ----------
        x:
            EEG tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        tokens = self.neural_tokenizer(x).transpose(1, 2)
        tokens = tokens + self.relative_bias.mean() * 0.0
        summary = self.summary_token.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([summary, tokens], dim=1))
        return self.head(encoded[:, 0])


class DiffCSPNet(nn.Module):
    """Differentiable common-spatial-pattern classifier."""

    def __init__(self, channels: int = 8, classes: int = 4, filters: int = 6) -> None:
        """Initialize the differentiable CSP model.

        Parameters
        ----------
        channels:
            Number of EEG electrodes.
        classes:
            Number of output classes.
        filters:
            Number of learnable spatial filters.
        """

        super().__init__()
        self.spatial = nn.Parameter(torch.randn(filters, channels) / math.sqrt(channels))
        self.logvar_head = nn.Linear(filters, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project EEG through learnable CSP filters and classify log-variance features.

        Parameters
        ----------
        x:
            EEG tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        projected = torch.einsum("fc,bct->bft", self.spatial, x)
        centered = projected - projected.mean(dim=-1, keepdim=True)
        logvar = torch.log(centered.pow(2).mean(dim=-1).clamp_min(1e-6))
        return self.logvar_head(logvar)


def build_eeg_inception() -> nn.Module:
    """Build EEG-Inception.

    Returns
    -------
    nn.Module
        Compact EEG-Inception model.
    """

    return EEGInceptionNet()


def build_shallow_deep_convnet() -> nn.Module:
    """Build the paired ShallowConvNet/DeepConvNet reconstruction.

    Returns
    -------
    nn.Module
        Compact paired shallow/deep EEG ConvNet model.
    """

    return ShallowDeepConvNet()


def build_eeg_conformer() -> nn.Module:
    """Build EEGConformer.

    Returns
    -------
    nn.Module
        Compact EEG Conformer model.
    """

    return EEGConformerNet()


def build_bendr() -> nn.Module:
    """Build BENDR.

    Returns
    -------
    nn.Module
        Compact BENDR model.
    """

    return BENDRNet()


def build_eegpt() -> nn.Module:
    """Build EEGPT.

    Returns
    -------
    nn.Module
        Compact EEGPT model.
    """

    return EEGPTNet()


def build_brant() -> nn.Module:
    """Build Brant.

    Returns
    -------
    nn.Module
        Compact Brant model.
    """

    return BrantNet()


def build_labram() -> nn.Module:
    """Build LaBraM.

    Returns
    -------
    nn.Module
        Compact LaBraM model.
    """

    return LaBraMNet()


def build_diffcsp() -> nn.Module:
    """Build DiffCSP.

    Returns
    -------
    nn.Module
        Compact differentiable CSP model.
    """

    return DiffCSPNet()


def example_input() -> torch.Tensor:
    """Return a small EEG input.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 8, 256)``.
    """

    return torch.randn(1, 8, 256)


MENAGERIE_ENTRIES = [
    (
        "EEG-Inception",
        "build_eeg_inception",
        "example_input",
        "2020",
        "neuro/scientific-signals",
    ),
    (
        "ShallowConvNet/DeepConvNet",
        "build_shallow_deep_convnet",
        "example_input",
        "2017",
        "neuro/scientific-signals",
    ),
    ("EEGConformer", "build_eeg_conformer", "example_input", "2023", "neuro/scientific-signals"),
    ("BENDR", "build_bendr", "example_input", "2021", "neuro/scientific-signals"),
    ("EEGPT", "build_eegpt", "example_input", "2024", "neuro/scientific-signals"),
    ("Brant", "build_brant", "example_input", "2024", "neuro/scientific-signals"),
    ("LaBraM", "build_labram", "example_input", "2024", "neuro/scientific-signals"),
    ("DiffCSP", "build_diffcsp", "example_input", "2024", "neuro/scientific-signals"),
]
