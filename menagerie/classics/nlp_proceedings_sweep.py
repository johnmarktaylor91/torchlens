"""Notable NLP architectures from proceedings that were absent from the catalog.

This module groups compact, random-initialized reimplementations found during a
2026 NLP proceedings sweep.  Each model preserves a distinctive forward
primitive while using small widths and sequence lengths for TorchLens tracing.

Papers represented:
  * Yang et al., "Hierarchical Attention Networks for Document Classification",
    NAACL 2016.
  * Guo et al., "A Deep Relevance Matching Model for Ad-hoc Retrieval",
    CIKM 2016.
  * Xiong et al., "End-to-End Neural Ad-hoc Ranking with Kernel Pooling",
    SIGIR 2017.
  * Gehring et al., "Convolutional Sequence to Sequence Learning", ICML 2017.
  * Yu et al., "QANet: Combining Local Convolution with Global Self-Attention
    for Reading Comprehension", ICLR 2018.
  * Dehghani et al., "Universal Transformers", ICLR 2019.
  * See et al., "Get To The Point: Summarization with Pointer-Generator
    Networks", ACL 2017.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AdditiveAttention(nn.Module):
    """Additive attention pooling over a sequence."""

    def __init__(self, dim: int) -> None:
        """Initialize query and projection parameters.

        Parameters
        ----------
        dim:
            Sequence feature dimension.
        """
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.query = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Pool sequence states with learned additive attention.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, dim)``.

        Returns
        -------
        Tensor
            Pooled tensor with shape ``(batch, dim)``.
        """
        scores = torch.matmul(torch.tanh(self.proj(x)), self.query)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class HierarchicalAttentionNetwork(nn.Module):
    """Hierarchical word-then-sentence attention document classifier."""

    def __init__(
        self,
        vocab_size: int = 128,
        emb_dim: int = 24,
        hidden_dim: int = 16,
        n_classes: int = 4,
    ) -> None:
        """Initialize HAN embeddings, BiGRUs, and attention pools.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        emb_dim:
            Token embedding dimension.
        hidden_dim:
            Per-direction GRU hidden size.
        n_classes:
            Number of document classes.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.word_gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.word_attn = AdditiveAttention(2 * hidden_dim)
        self.sent_gru = nn.GRU(2 * hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.sent_attn = AdditiveAttention(2 * hidden_dim)
        self.classifier = nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, doc_ids: Tensor) -> Tensor:
        """Classify a batch of tokenized documents.

        Parameters
        ----------
        doc_ids:
            Token ids with shape ``(batch, sentences, words)``.

        Returns
        -------
        Tensor
            Class logits with shape ``(batch, n_classes)``.
        """
        batch, n_sent, n_word = doc_ids.shape
        words = doc_ids.reshape(batch * n_sent, n_word)
        word_states, _ = self.word_gru(self.embedding(words))
        sent_vecs = self.word_attn(word_states).reshape(batch, n_sent, -1)
        sent_states, _ = self.sent_gru(sent_vecs)
        doc_vec = self.sent_attn(sent_states)
        return self.classifier(doc_vec)


class DRMMRanker(nn.Module):
    """Deep Relevance Matching Model with histograms and term gating."""

    def __init__(self, vocab_size: int = 128, emb_dim: int = 24, bins: int = 6) -> None:
        """Initialize embeddings, histogram MLP, and query gate.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        emb_dim:
            Token embedding dimension.
        bins:
            Number of soft histogram bins.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.register_buffer("centers", torch.linspace(-1.0, 1.0, bins))
        self.mlp = nn.Sequential(nn.Linear(bins, 12), nn.Tanh(), nn.Linear(12, 1))
        self.gate = nn.Linear(emb_dim, 1)

    def forward(self, pair_ids: Tensor) -> Tensor:
        """Score query-document relevance.

        Parameters
        ----------
        pair_ids:
            Concatenated ids with shape ``(batch, query_len + doc_len)``.

        Returns
        -------
        Tensor
            Relevance scores with shape ``(batch, 1)``.
        """
        query_ids = pair_ids[:, :4]
        doc_ids = pair_ids[:, 4:]
        query = F.normalize(self.embedding(query_ids), dim=-1)
        doc = F.normalize(self.embedding(doc_ids), dim=-1)
        sim = torch.matmul(query, doc.transpose(1, 2))
        diff = sim.unsqueeze(-1) - self.centers.view(1, 1, 1, -1)
        hist = torch.exp(-12.0 * diff.pow(2)).sum(dim=2)
        term_scores = self.mlp(torch.log1p(hist)).squeeze(-1)
        gates = torch.softmax(self.gate(query).squeeze(-1), dim=-1)
        return torch.sum(gates * term_scores, dim=-1, keepdim=True)


class KNRMRanker(nn.Module):
    """Kernel-pooling neural ranker for ad-hoc retrieval."""

    def __init__(self, vocab_size: int = 128, emb_dim: int = 24, kernels: int = 7) -> None:
        """Initialize embeddings, Gaussian kernels, and rank layer.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        emb_dim:
            Token embedding dimension.
        kernels:
            Number of RBF kernels over cosine similarity.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        mus = torch.linspace(-1.0, 1.0, kernels)
        sigmas = torch.full((kernels,), 0.25)
        sigmas[-1] = 0.05
        self.register_buffer("mus", mus)
        self.register_buffer("sigmas", sigmas)
        self.rank = nn.Linear(kernels, 1)

    def forward(self, pair_ids: Tensor) -> Tensor:
        """Score query-document relevance with kernel pooling.

        Parameters
        ----------
        pair_ids:
            Concatenated ids with shape ``(batch, query_len + doc_len)``.

        Returns
        -------
        Tensor
            Relevance scores with shape ``(batch, 1)``.
        """
        query = F.normalize(self.embedding(pair_ids[:, :4]), dim=-1)
        doc = F.normalize(self.embedding(pair_ids[:, 4:]), dim=-1)
        sim = torch.matmul(query, doc.transpose(1, 2)).unsqueeze(-1)
        kernels = torch.exp(-0.5 * ((sim - self.mus) / self.sigmas).pow(2))
        pooled = torch.log1p(kernels.sum(dim=2)).sum(dim=1)
        return self.rank(pooled)


class GLUConvBlock(nn.Module):
    """Residual gated convolution block used by ConvS2S."""

    def __init__(self, channels: int, kernel_size: int = 3, causal: bool = False) -> None:
        """Initialize a gated convolution block.

        Parameters
        ----------
        channels:
            Hidden channel count.
        kernel_size:
            Temporal convolution kernel size.
        causal:
            Whether to use left-only causal padding.
        """
        super().__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual GLU convolution.

        Parameters
        ----------
        x:
            Hidden states with shape ``(batch, time, channels)``.

        Returns
        -------
        Tensor
            Hidden states with the same shape as ``x``.
        """
        xt = x.transpose(1, 2)
        if self.causal:
            xt = F.pad(xt, (self.kernel_size - 1, 0))
        else:
            left = self.kernel_size // 2
            xt = F.pad(xt, (left, self.kernel_size - 1 - left))
        out = F.glu(self.conv(xt), dim=1).transpose(1, 2)
        return (x + out) * math.sqrt(0.5)


class ConvS2S(nn.Module):
    """Convolutional encoder-decoder with GLU blocks and decoder attention."""

    def __init__(self, vocab_size: int = 128, channels: int = 32, layers: int = 3) -> None:
        """Initialize ConvS2S embeddings and convolution stacks.

        Parameters
        ----------
        vocab_size:
            Source and target vocabulary size.
        channels:
            Hidden channel count.
        layers:
            Number of encoder and decoder convolution blocks.
        """
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, channels)
        self.tgt_embed = nn.Embedding(vocab_size, channels)
        self.encoder = nn.ModuleList([GLUConvBlock(channels) for _ in range(layers)])
        self.decoder = nn.ModuleList([GLUConvBlock(channels, causal=True) for _ in range(layers)])
        self.out = nn.Linear(channels, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Run teacher-forced convolutional seq2seq logits.

        Parameters
        ----------
        token_ids:
            Token ids with shape ``(batch, src_len + tgt_len)``.

        Returns
        -------
        Tensor
            Target logits with shape ``(batch, tgt_len, vocab_size)``.
        """
        src = token_ids[:, :8]
        tgt = token_ids[:, 8:]
        enc = self.src_embed(src)
        for block in self.encoder:
            enc = block(enc)
        dec = self.tgt_embed(tgt)
        for block in self.decoder:
            dec = block(dec)
            attn = torch.softmax(
                torch.matmul(dec, enc.transpose(1, 2)) / math.sqrt(dec.shape[-1]), -1
            )
            dec = dec + torch.matmul(attn, enc)
        return self.out(dec)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable temporal convolution."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        """Initialize depthwise and pointwise convolutions.

        Parameters
        ----------
        channels:
            Hidden channel count.
        kernel_size:
            Temporal kernel size.
        """
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply depthwise separable convolution.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, time, channels)``.

        Returns
        -------
        Tensor
            Convolved tensor with shape ``(batch, time, channels)``.
        """
        return self.pointwise(self.depthwise(x.transpose(1, 2))).transpose(1, 2)


class EncoderBlock(nn.Module):
    """QANet-style conv plus self-attention encoder block."""

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        """Initialize convolution, self-attention, and feed-forward layers.

        Parameters
        ----------
        channels:
            Hidden channel count.
        n_heads:
            Number of attention heads.
        """
        super().__init__()
        self.conv = DepthwiseSeparableConv(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(channels, 2 * channels), nn.ReLU(), nn.Linear(2 * channels, channels)
        )
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(3)])

    def forward(self, x: Tensor) -> Tensor:
        """Apply QANet encoder sublayers.

        Parameters
        ----------
        x:
            Hidden states with shape ``(batch, time, channels)``.

        Returns
        -------
        Tensor
            Hidden states with the same shape as ``x``.
        """
        x = self.norms[0](x + F.relu(self.conv(x)))
        attn, _ = self.attn(x, x, x, need_weights=False)
        x = self.norms[1](x + attn)
        return self.norms[2](x + self.ff(x))


class QANetReader(nn.Module):
    """QANet reading-comprehension model with context-query attention."""

    def __init__(self, vocab_size: int = 128, channels: int = 32) -> None:
        """Initialize compact QANet layers.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        channels:
            Hidden channel count.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, channels)
        self.context_enc = EncoderBlock(channels)
        self.question_enc = EncoderBlock(channels)
        self.fuse = nn.Linear(4 * channels, channels)
        self.model_blocks = nn.ModuleList([EncoderBlock(channels) for _ in range(2)])
        self.start = nn.Linear(2 * channels, 1)
        self.end = nn.Linear(2 * channels, 1)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Predict start and end logits for a compact QA example.

        Parameters
        ----------
        token_ids:
            Concatenated context and question ids with shape ``(batch, 18)``.

        Returns
        -------
        Tensor
            Span logits with shape ``(batch, context_len, 2)``.
        """
        context = self.context_enc(self.embedding(token_ids[:, :12]))
        question = self.question_enc(self.embedding(token_ids[:, 12:]))
        sim = torch.matmul(context, question.transpose(1, 2))
        c2q = torch.matmul(torch.softmax(sim, dim=-1), question)
        q2c_weights = torch.softmax(sim.max(dim=2).values, dim=-1)
        q2c = torch.matmul(q2c_weights.unsqueeze(1), context).expand_as(context)
        x = self.fuse(torch.cat([context, c2q, context * c2q, context * q2c], dim=-1))
        states = []
        for block in self.model_blocks:
            x = block(x)
            states.append(x)
        start = self.start(torch.cat([states[0], states[1]], dim=-1))
        end = self.end(torch.cat([states[1], x], dim=-1))
        return torch.cat([start, end], dim=-1)


class UniversalTransformer(nn.Module):
    """Recurrent shared-block Transformer with ACT-style halting weights."""

    def __init__(
        self,
        vocab_size: int = 128,
        channels: int = 32,
        steps: int = 3,
        n_heads: int = 4,
    ) -> None:
        """Initialize shared recurrent Transformer block.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        channels:
            Hidden channel count.
        steps:
            Number of recurrent refinement steps.
        n_heads:
            Number of attention heads.
        """
        super().__init__()
        self.steps = steps
        self.embedding = nn.Embedding(vocab_size, channels)
        self.step_embedding = nn.Embedding(steps, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(channels, 2 * channels), nn.ReLU(), nn.Linear(2 * channels, channels)
        )
        self.halt = nn.Linear(channels, 1)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.out = nn.Linear(channels, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Run recurrent self-attention updates and emit token logits.

        Parameters
        ----------
        token_ids:
            Token ids with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Token logits with shape ``(batch, time, vocab_size)``.
        """
        x = self.embedding(token_ids)
        accumulated = torch.zeros_like(x)
        remainder = torch.ones(token_ids.shape[0], token_ids.shape[1], 1, device=token_ids.device)
        for step in range(self.steps):
            step_ids = torch.full_like(token_ids, step)
            h = x + self.step_embedding(step_ids)
            attn, _ = self.attn(h, h, h, need_weights=False)
            h = self.norm1(h + attn)
            h = self.norm2(h + self.ff(h))
            p = torch.sigmoid(self.halt(h))
            weight = remainder * p if step < self.steps - 1 else remainder
            accumulated = accumulated + weight * h
            remainder = remainder * (1.0 - p)
            x = h
        return self.out(accumulated)


class PointerGenerator(nn.Module):
    """Attention seq2seq summarizer with generator-copy mixture."""

    def __init__(self, vocab_size: int = 64, emb_dim: int = 24, hidden_dim: int = 24) -> None:
        """Initialize encoder, decoder, and pointer-generator projections.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        emb_dim:
            Token embedding dimension.
        hidden_dim:
            Decoder hidden size.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.enc_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.decoder = nn.GRUCell(emb_dim + 2 * hidden_dim, hidden_dim)
        self.vocab = nn.Linear(3 * hidden_dim, vocab_size)
        self.p_gen = nn.Linear(3 * hidden_dim + emb_dim, 1)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Generate logits with source-copy probability mass.

        Parameters
        ----------
        token_ids:
            Concatenated source and teacher decoder ids with shape ``(batch, 15)``.

        Returns
        -------
        Tensor
            Mixed generation/copy distributions with shape ``(batch, tgt_len, vocab_size)``.
        """
        source = token_ids[:, :10]
        target = token_ids[:, 10:]
        enc, h = self.encoder(self.embedding(source))
        dec_state = self.enc_proj(torch.cat([h[-2], h[-1]], dim=-1))
        context = enc.mean(dim=1)
        outputs = []
        for step in range(target.shape[1]):
            emb = self.embedding(target[:, step])
            dec_state = self.decoder(torch.cat([emb, context], dim=-1), dec_state)
            scores = torch.matmul(enc, self.attn_proj(dec_state).unsqueeze(-1)).squeeze(-1)
            attn = torch.softmax(scores, dim=-1)
            context = torch.bmm(attn.unsqueeze(1), enc).squeeze(1)
            vocab_dist = torch.softmax(self.vocab(torch.cat([dec_state, context], dim=-1)), dim=-1)
            p_gen = torch.sigmoid(self.p_gen(torch.cat([context, dec_state, emb], dim=-1)))
            copy_dist = torch.zeros_like(vocab_dist).scatter_add(1, source, attn)
            outputs.append(p_gen * vocab_dist + (1.0 - p_gen) * copy_dist)
        return torch.stack(outputs, dim=1)


def build_han_document_classifier() -> nn.Module:
    """Build a compact Hierarchical Attention Network.

    Returns
    -------
    nn.Module
        Random-initialized HAN document classifier.
    """
    return HierarchicalAttentionNetwork().eval()


def example_input_han() -> Tensor:
    """Return a small document tensor.

    Returns
    -------
    Tensor
        Token ids with shape ``(1, 3, 5)``.
    """
    return torch.randint(0, 128, (1, 3, 5), dtype=torch.long)


def build_drmm_ranker() -> nn.Module:
    """Build a compact DRMM ranker.

    Returns
    -------
    nn.Module
        Random-initialized DRMM ranker.
    """
    return DRMMRanker().eval()


def build_knrm_ranker() -> nn.Module:
    """Build a compact K-NRM ranker.

    Returns
    -------
    nn.Module
        Random-initialized K-NRM ranker.
    """
    return KNRMRanker().eval()


def example_input_ranker() -> Tensor:
    """Return concatenated query and document ids.

    Returns
    -------
    Tensor
        Token ids with shape ``(1, 14)``.
    """
    return torch.randint(0, 128, (1, 14), dtype=torch.long)


def build_convs2s() -> nn.Module:
    """Build a compact convolutional sequence-to-sequence model.

    Returns
    -------
    nn.Module
        Random-initialized ConvS2S model.
    """
    return ConvS2S().eval()


def example_input_convs2s() -> Tensor:
    """Return concatenated source and target ids.

    Returns
    -------
    Tensor
        Token ids with shape ``(1, 14)``.
    """
    return torch.randint(0, 128, (1, 14), dtype=torch.long)


def build_qanet_reader() -> nn.Module:
    """Build a compact QANet reader.

    Returns
    -------
    nn.Module
        Random-initialized QANet reader.
    """
    return QANetReader().eval()


def example_input_qanet() -> Tensor:
    """Return concatenated context and question ids.

    Returns
    -------
    Tensor
        Token ids with shape ``(1, 18)``.
    """
    return torch.randint(0, 128, (1, 18), dtype=torch.long)


def build_universal_transformer() -> nn.Module:
    """Build a compact Universal Transformer.

    Returns
    -------
    nn.Module
        Random-initialized Universal Transformer.
    """
    return UniversalTransformer().eval()


def example_input_universal_transformer() -> Tensor:
    """Return token ids for Universal Transformer.

    Returns
    -------
    Tensor
        Token ids with shape ``(1, 10)``.
    """
    return torch.randint(0, 128, (1, 10), dtype=torch.long)


def build_pointer_generator() -> nn.Module:
    """Build a compact pointer-generator summarizer.

    Returns
    -------
    nn.Module
        Random-initialized pointer-generator model.
    """
    return PointerGenerator().eval()


def example_input_pointer_generator() -> Tensor:
    """Return concatenated source and teacher decoder ids.

    Returns
    -------
    Tensor
        Token ids with shape ``(1, 15)``.
    """
    return torch.randint(0, 64, (1, 15), dtype=torch.long)


MENAGERIE_ENTRIES = [
    (
        "Hierarchical Attention Network (HAN document classifier)",
        "build_han_document_classifier",
        "example_input_han",
        "2016",
        "DC",
    ),
    (
        "DRMM (Deep Relevance Matching Model)",
        "build_drmm_ranker",
        "example_input_ranker",
        "2016",
        "DC",
    ),
    (
        "K-NRM (kernel-pooling neural ranker)",
        "build_knrm_ranker",
        "example_input_ranker",
        "2017",
        "DC",
    ),
    (
        "ConvS2S (Convolutional Sequence-to-Sequence Learning)",
        "build_convs2s",
        "example_input_convs2s",
        "2017",
        "DC",
    ),
    (
        "QANet (conv + self-attention reading comprehension)",
        "build_qanet_reader",
        "example_input_qanet",
        "2018",
        "DC",
    ),
    (
        "Universal Transformer (recurrent self-attention with ACT)",
        "build_universal_transformer",
        "example_input_universal_transformer",
        "2019",
        "DC",
    ),
    (
        "Pointer-Generator Network (copying summarizer)",
        "build_pointer_generator",
        "example_input_pointer_generator",
        "2017",
        "DC",
    ),
]
