"""Perceiver and Performer compact transformer classics.

Perceiver: Jaegle et al. 2021, arXiv:2103.03206.
Performer: Choromanski et al. 2020, arXiv:2009.14794.

The Perceiver model uses learned latent queries, input-to-latent cross-attention,
latent self-attention, and a decoder head.  The Performer language model keeps the
Transformer block layout but replaces quadratic softmax attention with positive
random-feature FAVOR-style linear attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Two-layer feed-forward network used in transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Initialize the feed-forward network.

        Parameters
        ----------
        dim:
            Input and output feature dimension.
        hidden_dim:
            Hidden expansion dimension.
        """

        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with the same shape as ``x``.
        """

        return self.fc2(F.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    """Single-head latent-query cross-attention."""

    def __init__(self, latent_dim: int, input_dim: int) -> None:
        """Initialize projections for latent-to-input attention.

        Parameters
        ----------
        latent_dim:
            Latent token feature dimension.
        input_dim:
            Input token feature dimension.
        """

        super().__init__()
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(input_dim, latent_dim)
        self.v = nn.Linear(input_dim, latent_dim)
        self.out = nn.Linear(latent_dim, latent_dim)

    def forward(self, latents: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Attend from learned latents into input tokens.

        Parameters
        ----------
        latents:
            Latent tensor ``(B, L, D)``.
        inputs:
            Input token tensor ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Updated latent tensor.
        """

        q = self.q(latents)
        k = self.k(inputs)
        v = self.v(inputs)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5), dim=-1)
        return self.out(torch.matmul(attn, v))


class PerceiverBlock(nn.Module):
    """Perceiver latent self-attention block."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize latent self-attention and MLP.

        Parameters
        ----------
        dim:
            Latent feature dimension.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process latent tokens.

        Parameters
        ----------
        x:
            Latent token tensor.

        Returns
        -------
        torch.Tensor
            Updated latent token tensor.
        """

        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class PerceiverClassifier(nn.Module):
    """Compact Perceiver classifier with a learned latent bottleneck."""

    def __init__(self, input_dim: int = 16, latent_dim: int = 32, classes: int = 10) -> None:
        """Initialize the compact Perceiver.

        Parameters
        ----------
        input_dim:
            Input token feature dimension.
        latent_dim:
            Latent token feature dimension.
        classes:
            Number of classifier outputs.
        """

        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, 12, latent_dim) * 0.02)
        self.cross_norm = nn.LayerNorm(latent_dim)
        self.cross = CrossAttention(latent_dim, input_dim)
        self.blocks = nn.ModuleList([PerceiverBlock(latent_dim) for _ in range(2)])
        self.head = nn.Linear(latent_dim, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify an unordered token set through a latent bottleneck.

        Parameters
        ----------
        x:
            Input tokens ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        latents = self.latents.expand(x.shape[0], -1, -1)
        latents = latents + self.cross(self.cross_norm(latents), x)
        for block in self.blocks:
            latents = block(latents)
        return self.head(latents.mean(dim=1))


class FavorAttention(nn.Module):
    """Positive random-feature linear attention used by Performers."""

    def __init__(self, dim: int, heads: int = 4, features: int = 32) -> None:
        """Initialize FAVOR-style projections.

        Parameters
        ----------
        dim:
            Model width.
        heads:
            Number of attention heads.
        features:
            Random feature dimension per head.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.features = features
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        omega = torch.randn(heads, self.head_dim, features)
        self.register_buffer("omega", omega)

    def _positive_features(self, x: torch.Tensor) -> torch.Tensor:
        """Project inputs into positive random features.

        Parameters
        ----------
        x:
            Tensor ``(B, H, T, D)``.

        Returns
        -------
        torch.Tensor
            Positive feature tensor ``(B, H, T, R)``.
        """

        projected = torch.einsum("bhtd,hdr->bhtr", x, self.omega)
        scale = torch.exp(-0.5 * x.pow(2).sum(dim=-1, keepdim=True))
        return F.elu(projected) + 1.0 + scale * 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linearized self-attention.

        Parameters
        ----------
        x:
            Token tensor ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Attended token tensor.
        """

        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        qp = self._positive_features(q)
        kp = self._positive_features(k)
        kv = torch.einsum("bhtr,bhtd->bhrd", kp, v)
        denom = torch.einsum("bhtr,bhr->bht", qp, kp.sum(dim=2)).clamp_min(1e-6)
        out = torch.einsum("bhtr,bhrd->bhtd", qp, kv) / denom.unsqueeze(-1)
        out = out.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(out)


class PerformerBlock(nn.Module):
    """Transformer block with FAVOR linear attention."""

    def __init__(self, dim: int) -> None:
        """Initialize the Performer block.

        Parameters
        ----------
        dim:
            Model feature dimension.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FavorAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a token sequence.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        torch.Tensor
            Updated token tensor.
        """

        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PerformerLM(nn.Module):
    """Compact Performer language model."""

    def __init__(self, vocab: int = 128, dim: int = 32, layers: int = 2) -> None:
        """Initialize token embeddings, Performer blocks, and LM head.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Model width.
        layers:
            Number of Performer blocks.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 24, dim) * 0.02)
        self.blocks = nn.ModuleList([PerformerBlock(dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Run the language model.

        Parameters
        ----------
        ids:
            Token ids ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Vocabulary logits ``(B, T, vocab)``.
        """

        x = self.embed(ids) + self.pos[:, : ids.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def build_perceiver() -> nn.Module:
    """Build a compact Perceiver classifier.

    Returns
    -------
    nn.Module
        Random-init Perceiver classifier.
    """

    return PerceiverClassifier()


def build_performer_lm() -> nn.Module:
    """Build a compact Performer language model.

    Returns
    -------
    nn.Module
        Random-init Performer LM.
    """

    return PerformerLM()


def example_perceiver_input() -> torch.Tensor:
    """Create compact Perceiver tokens.

    Returns
    -------
    torch.Tensor
        Input tensor ``(1, 32, 16)``.
    """

    return torch.randn(1, 32, 16)


def example_token_ids() -> torch.Tensor:
    """Create compact token ids.

    Returns
    -------
    torch.Tensor
        Integer token tensor ``(1, 16)``.
    """

    return torch.randint(0, 128, (1, 16))


MENAGERIE_ENTRIES = [
    ("Perceiver", "build_perceiver", "example_perceiver_input", "2021", "ATTN"),
    ("performer_lm", "build_performer_lm", "example_token_ids", "2020", "ATTN"),
]
