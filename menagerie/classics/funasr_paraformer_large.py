"""FunASR Paraformer large compact reconstruction.

Paper: Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive
End-to-End Speech Recognition, 2022.  FunASR Paraformer combines a speech encoder,
a continuous integrate-and-fire style acoustic predictor, sampled acoustic embeddings,
and a parallel Transformer decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModule(nn.Module):
    """Conformer-style feed-forward module."""

    def __init__(self, dim: int) -> None:
        """Initialize feed-forward layers.

        Parameters
        ----------
        dim:
            Model dimension.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        torch.Tensor
            Transformed sequence.
        """

        return self.net(x)


class SANMBlock(nn.Module):
    """Self-attention with a memory convolution branch, as used in FunASR encoders."""

    def __init__(self, dim: int, heads: int = 4, causal: bool = False) -> None:
        """Initialize the SAN-M block.

        Parameters
        ----------
        dim:
            Model dimension.
        heads:
            Number of attention heads.
        causal:
            Whether to apply a causal attention mask.
        """

        super().__init__()
        self.causal = causal
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.memory = nn.Conv1d(dim, dim, kernel_size=5, padding=4 if causal else 2, groups=dim)
        self.ffn = FeedForwardModule(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run self-attention and memory-convolution updates.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, time, dim)``.

        Returns
        -------
        torch.Tensor
            Updated sequence.
        """

        time = x.shape[1]
        mask = (
            torch.full((time, time), float("-inf"), device=x.device).triu(1)
            if self.causal
            else None
        )
        h = self.norm(x)
        attn, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        mem = self.memory(h.transpose(1, 2))
        if self.causal:
            mem = mem[..., :time]
        mem = mem.transpose(1, 2)
        x = x + attn + mem
        return x + self.ffn(x)


class AcousticPredictor(nn.Module):
    """Compact CIF-like acoustic predictor for Paraformer."""

    def __init__(self, dim: int) -> None:
        """Initialize predictor layers.

        Parameters
        ----------
        dim:
            Encoder dimension.
        """

        super().__init__()
        self.weight = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1), nn.ReLU(), nn.Conv1d(dim, 1, 1), nn.Sigmoid()
        )
        self.query = nn.Linear(dim, dim)

    def forward(self, enc: torch.Tensor, token_count: int) -> torch.Tensor:
        """Produce acoustic embeddings for decoder positions.

        Parameters
        ----------
        enc:
            Encoder sequence.
        token_count:
            Number of compact output positions.

        Returns
        -------
        torch.Tensor
            Predictor embeddings.
        """

        weights = self.weight(enc.transpose(1, 2)).transpose(1, 2)
        weighted = enc * weights
        chunks = torch.chunk(weighted, token_count, dim=1)
        pooled = [chunk.mean(dim=1) for chunk in chunks]
        return self.query(torch.stack(pooled, dim=1))


class CompactParaformer(nn.Module):
    """Compact non-autoregressive Paraformer ASR model."""

    def __init__(
        self, feat_dim: int = 32, dim: int = 48, vocab: int = 128, causal: bool = False
    ) -> None:
        """Initialize compact Paraformer.

        Parameters
        ----------
        feat_dim:
            Acoustic input feature dimension.
        dim:
            Model dimension.
        vocab:
            Output vocabulary size.
        causal:
            Whether to use streaming-style causal encoder blocks.
        """

        super().__init__()
        self.subsample = nn.Conv1d(feat_dim, dim, kernel_size=3, stride=2, padding=1)
        self.encoder = nn.ModuleList([SANMBlock(dim, causal=causal) for _ in range(2)])
        self.predictor = AcousticPredictor(dim)
        self.decoder = nn.ModuleList([SANMBlock(dim, causal=False) for _ in range(2)])
        self.head = nn.Linear(dim, vocab)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Recognize speech features in parallel.

        Parameters
        ----------
        feats:
            Acoustic features of shape ``(batch, time, feat_dim)``.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        x = self.subsample(feats.transpose(1, 2)).transpose(1, 2)
        for block in self.encoder:
            x = block(x)
        y = self.predictor(x, token_count=6)
        for block in self.decoder:
            y = block(y)
        return self.head(y)


def build_funasr_paraformer_large() -> nn.Module:
    """Build compact FunASR Paraformer large.

    Returns
    -------
    nn.Module
        Random-init compact Paraformer.
    """

    return CompactParaformer()


def example_input() -> torch.Tensor:
    """Create compact speech features.

    Returns
    -------
    torch.Tensor
        Feature tensor of shape ``(1, 24, 32)``.
    """

    return torch.randn(1, 24, 32)


build = build_funasr_paraformer_large

MENAGERIE_ENTRIES = [
    ("funasr_paraformer_large", "build_funasr_paraformer_large", "example_input", "2022", "E5"),
]
