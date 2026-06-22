"""Compact faithful language/sequence classics.

Paper: Charformer (GBST), Compressive Transformer, cosFormer,
Infini-attention, Quiet-STaR, and Coconut.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformerLM(nn.Module):
    """Small reusable Transformer language-model trunk."""

    def __init__(self, vocab: int = 128, dim: int = 32, depth: int = 2, heads: int = 4) -> None:
        """Initialize the language-model trunk.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Embedding width.
        depth:
            Number of Transformer layers.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 2, batch_first=True)
        self.layers = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, vocab)

    def encode_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Encode already embedded tokens.

        Parameters
        ----------
        x:
            Embedded sequence.

        Returns
        -------
        torch.Tensor
            Contextual sequence.
        """

        return self.layers(x + self.pos[:, : x.shape[1]])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Compute token logits.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        return self.head(self.encode_embeddings(self.embed(ids)))


class GBST(nn.Module):
    """Gradient-based subword tokenization with soft block selection."""

    def __init__(self, vocab: int = 128, dim: int = 32, max_block: int = 4) -> None:
        """Initialize the GBST module.

        Parameters
        ----------
        vocab:
            Byte/character vocabulary.
        dim:
            Embedding width.
        max_block:
            Maximum candidate block length.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.max_block = max_block
        self.score = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Construct soft latent subword embeddings from character ids.

        Parameters
        ----------
        ids:
            Character ids.

        Returns
        -------
        torch.Tensor
            Soft block embeddings.
        """

        chars = self.embed(ids)
        candidates = []
        scores = []
        for block in range(1, self.max_block + 1):
            padded = F.pad(chars, (0, 0, 0, block - 1))
            windows = padded.unfold(1, block, 1).mean(dim=-1)
            candidates.append(windows[:, : ids.shape[1]])
            scores.append(self.score(candidates[-1]))
        stacked = torch.stack(candidates, dim=2)
        weights = torch.softmax(torch.cat(scores, dim=-1), dim=-1).unsqueeze(-1)
        return self.proj((stacked * weights).sum(dim=2))


class CharformerLM(nn.Module):
    """Charformer with GBST before the Transformer encoder."""

    def __init__(self) -> None:
        """Initialize Charformer."""

        super().__init__()
        self.gbst = GBST()
        self.lm = TinyTransformerLM()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Compute byte-level token logits.

        Parameters
        ----------
        ids:
            Character ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        return self.lm.head(self.lm.encode_embeddings(self.gbst(ids)))


class CompressiveTransformerLM(nn.Module):
    """Transformer-XL-style memory plus downsampled compressed memory."""

    def __init__(self, vocab: int = 128, dim: int = 32) -> None:
        """Initialize the compressive Transformer.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Embedding width.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.compress = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Attend to current tokens, memory, and compressed memory.

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
        memory = x.detach()
        compressed = self.compress(memory.transpose(1, 2)).transpose(1, 2)
        context = torch.cat([compressed, memory, x], dim=1)
        q, k, v = self.qkv(torch.cat([x, context], dim=1)).chunk(3, dim=-1)
        q = q[:, : x.shape[1]]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        attended = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.head(x + attended + self.ff(attended))


class CosFormerAttention(nn.Module):
    """cosFormer linear attention: ReLU features plus cosine distance reweighting."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize cosFormer attention.

        Parameters
        ----------
        dim:
            Feature width.
        """

        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply nonnegative cosine-reweighted linear attention.

        Parameters
        ----------
        x:
            Sequence features.

        Returns
        -------
        torch.Tensor
            Attended sequence.
        """

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = F.relu(q) + 1e-4
        k = F.relu(k) + 1e-4
        pos = torch.linspace(0, math.pi / 2, x.shape[1], device=x.device)
        q = torch.cat([q * pos.cos()[None, :, None], q * pos.sin()[None, :, None]], dim=-1)
        k = torch.cat([k * pos.cos()[None, :, None], k * pos.sin()[None, :, None]], dim=-1)
        kv = torch.einsum("btd,bte->bde", k, v)
        denom = torch.einsum("btd,bd->bt", q, k.sum(dim=1)).clamp_min(1e-6)
        out = torch.einsum("btd,bde->bte", q, kv) / denom.unsqueeze(-1)
        return self.out(out)


class CosFormerLM(nn.Module):
    """Language model using cosFormer attention."""

    def __init__(self, vocab: int = 128, dim: int = 32) -> None:
        """Initialize cosFormer LM.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Feature width.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.attn = CosFormerAttention(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Compute logits with linear cosFormer attention.

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
        x = x + self.attn(x)
        return self.head(x + self.ff(x))


class InfiniAttentionLM(nn.Module):
    """Infini-attention block combining local masked attention and compressive memory."""

    def __init__(self, vocab: int = 128, dim: int = 32) -> None:
        """Initialize Infini-attention LM.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Feature width.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.memory = nn.Parameter(torch.zeros(dim, dim))
        self.memory_norm = nn.Parameter(torch.ones(dim))
        self.gate = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Mix local causal attention with long-term linear memory retrieval.

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
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        mask = torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device).triu(1)
        local = torch.matmul(torch.softmax(scores + mask, dim=-1), v)
        mem_read = torch.einsum("btd,de->bte", F.elu(q) + 1.0, self.memory)
        mem_read = mem_read / self.memory_norm.abs().mean().clamp_min(1e-6)
        gate = torch.sigmoid(self.gate(x))
        return self.head(x + gate * local + (1.0 - gate) * mem_read)


class QuietStarLM(nn.Module):
    """Quiet-STaR-style rationale generator and mixer."""

    def __init__(self, vocab: int = 128, dim: int = 32) -> None:
        """Initialize Quiet-STaR LM.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Feature width.
        """

        super().__init__()
        self.base = TinyTransformerLM(vocab=vocab, dim=dim)
        self.thought_start = nn.Parameter(torch.zeros(1, 1, dim))
        self.thought_end = nn.Parameter(torch.zeros(1, 1, dim))
        self.rationale = nn.GRU(dim, dim, batch_first=True)
        self.mix = nn.Linear(dim * 2, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Generate token-wise hidden rationales and mix them into predictions.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        emb = self.base.embed(ids)
        base_hidden = self.base.encode_embeddings(emb)
        thought_in = emb + self.thought_start + self.thought_end
        thought_hidden, _ = self.rationale(thought_in)
        mixed = torch.tanh(self.mix(torch.cat([base_hidden, thought_hidden], dim=-1)))
        return self.base.head(mixed)


class CoconutLM(nn.Module):
    """Coconut chain-of-continuous-thought model."""

    def __init__(self, vocab: int = 128, dim: int = 32, thoughts: int = 3) -> None:
        """Initialize Coconut LM.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Feature width.
        thoughts:
            Number of continuous thought feedback steps.
        """

        super().__init__()
        self.base = TinyTransformerLM(vocab=vocab, dim=dim)
        self.thoughts = thoughts
        self.project = nn.Linear(dim, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Feed last hidden states back as continuous thought embeddings.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        emb = self.base.embed(ids)
        hidden = self.base.encode_embeddings(emb)
        thought = hidden[:, -1:, :]
        for _ in range(self.thoughts):
            thought = self.base.encode_embeddings(self.project(thought))[:, -1:, :]
        augmented = torch.cat([hidden, thought.expand(-1, hidden.shape[1], -1)], dim=-1)
        return self.base.head(augmented[..., : hidden.shape[-1]])


def build_charformer() -> nn.Module:
    """Build Charformer.

    Returns
    -------
    nn.Module
        Compact Charformer model.
    """

    return CharformerLM()


def build_compressive_transformer() -> nn.Module:
    """Build Compressive Transformer.

    Returns
    -------
    nn.Module
        Compact Compressive Transformer model.
    """

    return CompressiveTransformerLM()


def build_cosformer() -> nn.Module:
    """Build cosFormer.

    Returns
    -------
    nn.Module
        Compact cosFormer model.
    """

    return CosFormerLM()


def build_infini_attention() -> nn.Module:
    """Build Infini-attention.

    Returns
    -------
    nn.Module
        Compact Infini-attention model.
    """

    return InfiniAttentionLM()


def build_quiet_star() -> nn.Module:
    """Build Quiet-STaR.

    Returns
    -------
    nn.Module
        Compact Quiet-STaR model.
    """

    return QuietStarLM()


def build_coconut() -> nn.Module:
    """Build Coconut.

    Returns
    -------
    nn.Module
        Compact Coconut model.
    """

    return CoconutLM()


def example_tokens() -> torch.Tensor:
    """Return token ids for compact language models.

    Returns
    -------
    torch.Tensor
        Token id tensor.
    """

    return torch.randint(0, 128, (1, 16))


MENAGERIE_ENTRIES = [
    ("Charformer", "build_charformer", "example_tokens", "2021", "language/sequence"),
    (
        "Compressive-Transformer",
        "build_compressive_transformer",
        "example_tokens",
        "2019",
        "language/sequence",
    ),
    ("CosFormer", "build_cosformer", "example_tokens", "2022", "language/sequence"),
    ("Infini-attention", "build_infini_attention", "example_tokens", "2024", "language/sequence"),
    ("Quiet-STaR", "build_quiet_star", "example_tokens", "2024", "language/sequence"),
    ("Coconut", "build_coconut", "example_tokens", "2025", "language/sequence"),
]
