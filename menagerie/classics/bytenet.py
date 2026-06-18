"""ByteNet, 2016, Kalchbrenner et al., "Neural Machine Translation in Linear Time".

Dilated convolutional encoder and masked causal dilated decoder blocks implement
a fully convolutional sequence-to-sequence model. This small version keeps the
residual dilated blocks and teacher-forced decoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualConvBlock(nn.Module):
    """Residual one-dimensional dilated convolution block."""

    def __init__(self, channels: int, dilation: int, causal: bool) -> None:
        """Initialize a dilated residual convolution block.

        Parameters
        ----------
        channels:
            Number of sequence channels.
        dilation:
            Convolution dilation factor.
        causal:
            Whether to use left-only causal padding.
        """
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual dilated convolution.

        Parameters
        ----------
        x:
            Tensor with shape ``(batch, channels, time)``.

        Returns
        -------
        Tensor
            Tensor with the same shape as ``x``.
        """
        padding = (2 * self.dilation, 0) if self.causal else (self.dilation, self.dilation)
        out = self.conv(F.pad(x, padding))
        return x + torch.relu(self.norm(out))


class ByteNet(nn.Module):
    """Compact ByteNet encoder-decoder."""

    def __init__(self, vocab_size: int = 64, channels: int = 48, n_layers: int = 4) -> None:
        """Initialize embeddings and dilated convolution stacks.

        Parameters
        ----------
        vocab_size:
            Source and target vocabulary size.
        channels:
            Hidden channel count.
        n_layers:
            Number of residual blocks per stack.
        """
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, channels)
        self.tgt_embed = nn.Embedding(vocab_size, channels)
        dilations = [2**i for i in range(n_layers)]
        self.encoder = nn.ModuleList(
            [ResidualConvBlock(channels, d, causal=False) for d in dilations]
        )
        self.decoder = nn.ModuleList(
            [ResidualConvBlock(channels, d, causal=True) for d in dilations]
        )
        self.classifier = nn.Conv1d(channels, vocab_size, kernel_size=1)

    def forward(self, src_ids: Tensor, tgt_ids: Tensor | None = None) -> Tensor:
        """Compute teacher-forced target logits from source and target ids.

        Parameters
        ----------
        src_ids:
            Source token ids with shape ``(batch, src_time)``.
        tgt_ids:
            Optional target token ids with shape ``(batch, tgt_time)``. If omitted,
            source ids are reused as a trace-friendly teacher-forced target.

        Returns
        -------
        Tensor
            Target logits with shape ``(batch, tgt_time, vocab_size)``.
        """
        if tgt_ids is None:
            tgt_ids = src_ids
        enc = self.src_embed(src_ids).transpose(1, 2)
        for block in self.encoder:
            enc = block(enc)
        context = enc.mean(dim=-1, keepdim=True)
        dec = self.tgt_embed(tgt_ids).transpose(1, 2) + context
        for block in self.decoder:
            dec = block(dec)
        return self.classifier(dec).transpose(1, 2)


def build() -> nn.Module:
    """Build a compact ByteNet.

    Returns
    -------
    nn.Module
        Random-initialized ByteNet.
    """
    return ByteNet()


def example_input() -> Tensor:
    """Return example source token ids.

    Returns
    -------
    Tensor
        Long tensor with shape ``(1, 32)``.
    """
    return torch.randint(0, 64, (1, 32), dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("ByteNet Dilated Convolutional Seq2Seq", "build", "example_input", "2016", "DE")
]
