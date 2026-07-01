"""Menagerie batch w6a6: single-cell/genomics language and regulatory models.

Sources checked (reference only; no cloning, no pip installs):
  - LangCell (cand_00739): Zhao, Zhang, Wu, Luo & Nie, ICML 2024, "LangCell:
    Language-Cell Pre-training for Cell Identity Understanding", arXiv
    2405.06708, official code https://github.com/PharMolix/LangCell
    (``LangCell-CE-annotation/utils.py``). The distinctive mechanism is a
    BLIP/ALBEF-style dual-encoder-plus-fusion design: a Geneformer-style
    cell encoder (a BERT stack over rank-order-encoded gene tokens) and a
    text encoder (a BERT stack over natural-language cell-identity
    descriptions) whose upper layers carry an added cross-attention block
    that attends the text stream onto the cell encoder's hidden states
    (``BertLayer`` with ``add_cross_attention=True`` in ``utils.py``, mirror-
    ing BLIP's ``BertLMHeadModel`` fusion encoder). Contrastive (ITC) and
    matching (ITM) heads sit on top of, respectively, the pooled uni-modal
    embeddings and the fused cross-attended representation. This module
    reimplements the dual-tower-plus-cross-attention-fusion topology
    compactly: a tiny cell-token transformer encoder, a tiny text-token
    transformer encoder, one cross-attention fusion layer that lets text
    queries attend to cell keys/values, and both an ITC (cosine-similarity)
    projection head and an ITM (binary match) classification head on the
    fused ``[CLS]`` token.
  - LDARNet (cand_00740): darlednik, ICML 2026 (arXiv 2606.04552),
    "Hierarchical Latent Dynamic Adaptive Routing Network" (H-Net-style
    tokenizer-free genomic LM), official code
    https://github.com/darlednik/ICML-LDARNet (``ldar/models/ldar.py``,
    ``ldar/modules/dc.py``). The distinctive mechanism is *learned dynamic
    chunking*: a ``RoutingModule`` scores every adjacent-token pair by the
    cosine dissimilarity between a query projection of position ``j`` and a
    key projection of position ``j+1`` (``p_boundary = (1 - cos_sim) / 2``);
    positions with high boundary probability are kept by a ``ChunkLayer``
    that compresses the sequence to only its predicted chunk-boundary
    tokens, a coarser main network runs over the compressed sequence, and a
    ``DeChunkLayer`` re-expands back to full resolution via a bidirectional
    EMA-style scan (``z_j = p_j z_j + (1-p_j) z_{j-1}``, forward and
    backward passes averaged) gated back into the encoder residual stream
    through a straight-through-estimator boundary probability. The official
    code depends on ``mamba_ssm``/``flash_attn`` Triton kernels
    (``mamba_chunk_scan_combined``) which are not present in the base env;
    this module reproduces the routing/chunk/dechunk boundary-prediction
    topology exactly (cosine-similarity boundary scorer, hard top-boundary
    compression, bidirectional log-space EMA re-expansion, STE-gated
    residual) with plain-PyTorch bidirectional GRU isotropic mixer blocks
    standing in for the BiMamba-2 encoder/decoder/main-network stages.
  - LINGER (cand_00741): Yuan & Duren (Duren Lab), Nature Biotechnology
    2024, "Linking regulatory elements to target genes by integrating
    single-cell multi-omics with deep learning", official code
    https://github.com/Durenlab/LINGER (``code/lingergrn-1.106/LingerGRN/
    LINGER_tr.py``, class ``Net`` + function ``sc_nn``). The distinctive
    mechanism is a per-gene regression MLP (``Linear(in, 64) -> Linear(64,
    16) -> Linear(16, 1)`` over concatenated TF-expression and regulatory-
    element (RE) accessibility features) trained with a *manifold
    (graph-Laplacian) regularizer* on the first-layer weight matrix,
    ``alpha * trace(W1 @ L @ W1^T)`` where ``L = D^-1/2 (D - A) D^-1/2`` is
    the normalized graph Laplacian of a TF-RE binding adjacency matrix
    (so co-regulating TF/RE inputs are pulled toward similar first-layer
    weight vectors), plus a lifelong-learning bulk-to-single-cell knowledge
    transfer term that dot-products the current ``fc1`` weights against a
    frozen bulk-pretrained weight matrix. This module reproduces the
    per-gene MLP-over-[TF;RE]-features head and exposes the graph-Laplacian
    manifold-regularization matrix as a registered buffer plus a
    ``manifold_penalty()`` method computing the same trace form, faithfully
    carrying the architecture's defining regularization structure into a
    traceable forward pass (the forward pass itself is the plain MLP
    regression; the Laplacian penalty is exposed as an auxiliary method
    consistent with how the reference code adds it to the training loss).
  - MetagenBERT (cand_00742): CorvusVaine, arXiv 2601.03295, "metagenomic
    representation learning via frozen genomic-LLM read embeddings",
    official code https://github.com/CorvusVaine/MetagenBERT
    (``DeepSets.py``, classes ``Phi``/``Rho``/``DeepSets``). Despite the
    "BERT" name the trainable network is not itself a transformer -- the
    genomic-LLM (DNABERT-2/DNABERT-S) backbone is used frozen purely to
    embed reads/read-clusters -- so the distinctive trainable architecture
    is a **Deep Sets / attention-MIL bag classifier** over those frozen
    per-read (or per-cluster-centroid) embeddings: a per-instance encoder
    ``phi`` (small MLP) maps each read embedding to a latent vector, a
    learned gated-attention pooling layer (``Linear -> Tanh -> Linear ->
    softmax`` over the bag dimension, as in Ilse et al. attention-MIL)
    aggregates the bag into one representation, and a bag-level classifier
    ``rho`` (small MLP) predicts the sample-level phenotype label (e.g.
    disease status) from the pooled representation. This module reproduces
    the ``Phi`` / attention-pooling / ``Rho`` topology exactly, operating on
    a synthetic bag of per-read embedding vectors in place of the frozen
    DNABERT-2 embeddings.
  - MethylBERT (cand_00743): Jeong et al. (CompEpigen), Nature
    Communications 2025, "MethylBERT: a Transformer-based model for read-
    level DNA methylation pattern identification and tumour deconvolution",
    official code https://github.com/CompEpigen/methylbert
    (``src/methylbert/network.py``, class ``MethylBertEmbeddedDMR``). The
    distinctive mechanism is DMR-conditioned read classification: a BERT
    encoder runs over a tokenized methylation-call sequence (per-CpG
    methylated/unmethylated/no-CpG tokens) for one sequencing read, and the
    *differentially-methylated-region (DMR) identity* that the read was
    sampled from is embedded (``nn.Embedding(num_dmrs, seq_len+1)``) and
    concatenated as one extra feature channel onto every position of the
    BERT sequence output before flattening through a read-classifier MLP
    that predicts whether the read originated from tumour or normal tissue
    -- i.e. the same transformer backbone is reused across all DMRs, with
    DMR identity injected as a conditioning signal rather than a separate
    per-region model. This module reproduces the BERT-encoder +
    DMR-embedding-concatenation + flattened-MLP-read-classifier topology
    exactly, with tiny transformer dims.
  - MMSplice (cand_00745): Cheng, Celik, Kundaje & Gagneur, Genome Biology
    2019, "MMSplice: predicting the effect of mutations on splicing using
    adaptive gradient boosting" [sic tool name only, model is modular NN],
    official code https://github.com/gagneurlab/MMSplice_MTSplice
    (``mmsplice/mmsplice.py`` class ``MMSplice``, ``mmsplice/layers.py``).
    Distinct from the already-registered ``MTSplice`` tissue-extension
    (which only has donor/acceptor towers plus a per-tissue head), the base
    MMSplice model is a **five-module** modular scorer: independent small
    1D-conv networks over five different sequence windows around a
    variant/exon -- ``acceptor_intronM`` (upstream intron), ``acceptorM``
    (acceptor splice site), ``exonM`` (exon body, with 0-masked global
    average pooling via ``GlobalAveragePooling1D_Mask0`` so padded exon
    positions don't contribute), ``donorM`` (donor splice site), and
    ``donor_intronM`` (downstream intron) -- whose five scalar module scores
    are concatenated into one modular-score vector
    (``predict_modular_scores_on_batch``) and linearly combined into a
    single delta-logit-PSI splicing-effect prediction. This module
    reproduces the five-tower modular topology (masked-average-pool conv
    scorer per window) plus the linear combiner exactly, using the same
    ``SpliceSiteTower``-style conv-and-pool primitive already established
    for ``MTSplice`` in ``gen_w5a7.py`` but rebuilt independently here (five
    towers, masked pooling, no tissue head) since it is architecturally the
    base model MTSplice extends, not a duplicate of it.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# LangCell: dual-tower (cell BERT + text BERT) with cross-attention fusion.
# ---------------------------------------------------------------------------


class _TinyTransformerEncoder(nn.Module):
    """A minimal pre-norm transformer encoder stack (uni-modal tower)."""

    def __init__(self, vocab_size: int, hidden: int, n_layers: int, n_heads: int) -> None:
        """Initialize the tiny transformer encoder.

        Parameters
        ----------
        vocab_size : int
            Token vocabulary size.
        hidden : int
            Hidden/embedding dimension.
        n_layers : int
            Number of transformer encoder layers.
        n_heads : int
            Number of self-attention heads.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of token-id sequences.

        Parameters
        ----------
        token_ids : torch.Tensor
            Shape ``(batch, seq_len)`` integer token ids.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, hidden)`` contextual embeddings.
        """
        return self.encoder(self.embed(token_ids))


class LangCell(nn.Module):
    """LangCell dual-encoder cell-language model with cross-attention fusion.

    A Geneformer-style cell-token encoder and a text-token encoder each
    produce contextual embeddings; a single cross-attention fusion layer
    lets the text stream attend onto the cell stream's keys/values, and the
    fused ``[CLS]`` representation feeds both an ITC (contrastive
    projection) head and an ITM (binary match) head, matching LangCell's
    BLIP-style pretraining topology.
    """

    def __init__(
        self,
        gene_vocab: int = 256,
        text_vocab: int = 512,
        hidden: int = 32,
        n_heads: int = 4,
        n_uni_layers: int = 2,
        proj_dim: int = 16,
    ) -> None:
        """Initialize LangCell.

        Parameters
        ----------
        gene_vocab : int
            Number of distinct gene-rank tokens for the cell encoder.
        text_vocab : int
            Text token vocabulary size.
        hidden : int
            Shared hidden dimension for both towers and the fusion layer.
        n_heads : int
            Number of attention heads (self- and cross-attention).
        n_uni_layers : int
            Number of uni-modal transformer layers per tower.
        proj_dim : int
            Dimension of the ITC contrastive projection space.
        """
        super().__init__()
        self.cell_encoder = _TinyTransformerEncoder(gene_vocab, hidden, n_uni_layers, n_heads)
        self.text_encoder = _TinyTransformerEncoder(text_vocab, hidden, n_uni_layers, n_heads)

        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.fusion_ffn_norm = nn.LayerNorm(hidden)

        self.cell_proj = nn.Linear(hidden, proj_dim)
        self.text_proj = nn.Linear(hidden, proj_dim)
        self.itm_head = nn.Linear(hidden, 2)

    def forward(
        self, gene_tokens: torch.Tensor, text_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the dual-encoder-plus-fusion forward pass.

        Parameters
        ----------
        gene_tokens : torch.Tensor
            Shape ``(batch, gene_seq_len)`` rank-ordered gene token ids.
        text_tokens : torch.Tensor
            Shape ``(batch, text_seq_len)`` text token ids.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(cell_embed, text_embed, itm_logits)``: the ITC projection of
            the pooled cell embedding (``(batch, proj_dim)``), the ITC
            projection of the pooled text embedding (``(batch, proj_dim)``),
            and the ITM match logits from the fused ``[CLS]`` token
            (``(batch, 2)``).
        """
        cell_hidden = self.cell_encoder(gene_tokens)
        text_hidden = self.text_encoder(text_tokens)

        fused, _ = self.cross_attn(text_hidden, cell_hidden, cell_hidden)
        fused = self.fusion_norm(text_hidden + fused)
        fused = self.fusion_ffn_norm(fused + self.fusion_ffn(fused))

        cell_cls = cell_hidden[:, 0]
        text_cls = text_hidden[:, 0]
        fused_cls = fused[:, 0]

        cell_embed = F.normalize(self.cell_proj(cell_cls), dim=-1)
        text_embed = F.normalize(self.text_proj(text_cls), dim=-1)
        itm_logits = self.itm_head(fused_cls)
        return cell_embed, text_embed, itm_logits


def build_langcell() -> nn.Module:
    """Build a compact LangCell dual-encoder cell-language model.

    Returns
    -------
    nn.Module
        LangCell reconstruction in evaluation mode.
    """
    return LangCell(
        gene_vocab=256, text_vocab=512, hidden=32, n_heads=4, n_uni_layers=2, proj_dim=16
    ).eval()


def example_input_langcell() -> tuple[torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_langcell`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(gene_tokens, text_tokens)`` of shape ``(2, 24)`` and ``(2, 16)``.
    """
    gene_tokens = torch.randint(0, 256, (2, 24))
    text_tokens = torch.randint(0, 512, (2, 16))
    return gene_tokens, text_tokens


# ---------------------------------------------------------------------------
# LDARNet: learned dynamic chunking (H-Net-style routing/chunk/dechunk).
# ---------------------------------------------------------------------------


class _BiGRUMixer(nn.Module):
    """Bidirectional-GRU isotropic mixer standing in for a BiMamba-2 stage."""

    def __init__(self, d_model: int, n_layers: int = 1) -> None:
        """Initialize the isotropic mixer.

        Parameters
        ----------
        d_model : int
            Feature dimension (kept the same on input and output).
        n_layers : int
            Number of stacked bidirectional-GRU layers.
        """
        super().__init__()
        assert d_model % 2 == 0
        self.gru = nn.GRU(
            d_model, d_model // 2, num_layers=n_layers, batch_first=True, bidirectional=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix a (possibly compressed) token sequence.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, d_model)``, residual-mixed and normed.
        """
        out, _ = self.gru(x)
        return self.norm(x + out)


class _RoutingModule(nn.Module):
    """Cosine-dissimilarity boundary scorer (learned dynamic chunking)."""

    def __init__(self, d_model: int) -> None:
        """Initialize the routing module.

        Parameters
        ----------
        d_model : int
            Feature dimension of the incoming hidden states.
        """
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d_model))
            self.k_proj.weight.copy_(torch.eye(d_model))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Score every position's boundary probability.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len)`` boundary probabilities in ``[0, 1]``;
            position 0 is always a forced boundary (probability 1).
        """
        q = F.normalize(self.q_proj(hidden_states), dim=-1)
        k = F.normalize(self.k_proj(hidden_states), dim=-1)
        cos_fwd = torch.einsum("bld,bld->bl", q[:, :-1], k[:, 1:])
        cos_bwd = torch.einsum("bld,bld->bl", q[:, 1:], k[:, :-1])
        cos_sim = 0.5 * (cos_fwd + cos_bwd)
        p_boundary = torch.clamp((1.0 - cos_sim) / 2.0, 0.0, 1.0)
        return F.pad(p_boundary, (1, 0), value=1.0)


class _STEBoundaryGate(torch.autograd.Function):
    """Straight-through estimator: forward returns 1s, backward passes grad."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Return an all-ones tensor shaped like ``x``."""
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass the incoming gradient straight through."""
        return grad_output


def _bidirectional_ema_expand(
    compressed: torch.Tensor, boundary_prob: torch.Tensor
) -> torch.Tensor:
    """Re-expand a compressed sequence to full resolution via bidirectional EMA.

    Every kept (boundary) position is repeated forward until the next
    boundary, then a forward EMA (``z_j = p_j z_j + (1-p_j) z_{j-1}``) and a
    backward EMA are each applied over the full-resolution repeated stream
    and averaged, matching the reference ``DeChunkLayer``.

    Parameters
    ----------
    compressed : torch.Tensor
        Shape ``(batch, n_boundaries, d_model)`` compressed hidden states,
        one row per predicted boundary token (in sequence order).
    boundary_prob : torch.Tensor
        Shape ``(batch, seq_len)`` full-resolution boundary probabilities
        (position 0 is always 1.0).

    Returns
    -------
    torch.Tensor
        Shape ``(batch, seq_len, d_model)`` full-resolution re-expanded
        hidden states.
    """
    batch, seq_len = boundary_prob.shape
    n_boundaries = compressed.shape[1]
    plug_back_idx = (torch.cumsum((boundary_prob > 0.5).long(), dim=1) - 1).clamp(
        min=0, max=n_boundaries - 1
    )
    repeated = torch.gather(
        compressed, 1, plug_back_idx.unsqueeze(-1).expand(-1, -1, compressed.shape[-1])
    )
    p = boundary_prob.clamp(1e-4, 1.0 - 1e-4).unsqueeze(-1)

    fwd = torch.empty_like(repeated)
    fwd[:, 0] = repeated[:, 0]
    for t in range(1, seq_len):
        fwd[:, t] = p[:, t] * repeated[:, t] + (1.0 - p[:, t]) * fwd[:, t - 1]

    bwd = torch.empty_like(repeated)
    bwd[:, -1] = repeated[:, -1]
    for t in range(seq_len - 2, -1, -1):
        bwd[:, t] = p[:, t] * repeated[:, t] + (1.0 - p[:, t]) * bwd[:, t + 1]

    return 0.5 * (fwd + bwd)


class LDARNet(nn.Module):
    """LDARNet hierarchical dynamic-chunking genomic language model.

    An encoder mixer runs at full token resolution, a learned routing
    module predicts chunk boundaries from local cosine dissimilarity, a
    coarser main-network mixer runs over the compressed boundary tokens,
    and a bidirectional-EMA dechunk layer re-expands back to full
    resolution before an STE-gated residual and a decoder mixer.
    """

    def __init__(self, vocab_size: int = 8, d_model: int = 32, n_boundaries: int = 8) -> None:
        """Initialize LDARNet.

        Parameters
        ----------
        vocab_size : int
            Nucleotide-level token vocabulary (4 bases + special tokens).
        d_model : int
            Hidden dimension shared across all stages.
        n_boundaries : int
            Number of boundary tokens kept by the compression step (fixed
            here for a traceable static-shape compression, standing in for
            the reference's data-dependent boundary count).
        """
        super().__init__()
        self.n_boundaries = n_boundaries
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = _BiGRUMixer(d_model)
        self.routing_module = _RoutingModule(d_model)
        self.main_network = _BiGRUMixer(d_model)
        self.decoder = _BiGRUMixer(d_model)
        self.residual_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.residual_proj.weight)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run the routing/chunk/main/dechunk/decode pipeline.

        Parameters
        ----------
        token_ids : torch.Tensor
            Shape ``(batch, seq_len)`` integer nucleotide token ids.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, vocab_size)`` per-position logits.
        """
        hidden = self.embed(token_ids)
        hidden = self.encoder(hidden)
        residual = self.residual_proj(hidden)

        boundary_prob = self.routing_module(hidden)
        top_idx = torch.topk(boundary_prob, k=self.n_boundaries, dim=1, sorted=True).indices
        top_idx, _ = torch.sort(top_idx, dim=1)
        compressed = torch.gather(hidden, 1, top_idx.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))

        compressed = self.main_network(compressed)
        expanded = _bidirectional_ema_expand(compressed, boundary_prob)

        gate = _STEBoundaryGate.apply(boundary_prob).unsqueeze(-1)
        hidden = expanded * gate + residual

        hidden = self.decoder(hidden)
        return self.lm_head(hidden)


def build_ldarnet() -> nn.Module:
    """Build a compact LDARNet dynamic-chunking genomic language model.

    Returns
    -------
    nn.Module
        LDARNet reconstruction in evaluation mode.
    """
    return LDARNet(vocab_size=8, d_model=32, n_boundaries=8).eval()


def example_input_ldarnet() -> torch.Tensor:
    """Create example input for :func:`build_ldarnet`.

    Returns
    -------
    torch.Tensor
        Shape ``(2, 32)`` random nucleotide token ids.
    """
    return torch.randint(0, 8, (2, 32))


# ---------------------------------------------------------------------------
# LINGER: manifold-regularized per-gene TF/RE regression MLP.
# ---------------------------------------------------------------------------


class LINGER(nn.Module):
    """LINGER per-gene regulatory-element regression net with manifold reg.

    A small MLP predicts one target gene's expression from concatenated
    transcription-factor (TF) expression and regulatory-element (RE)
    accessibility features. A normalized graph Laplacian of the TF-RE
    binding adjacency is stored as a buffer, and :meth:`manifold_penalty`
    computes the ``trace(W1 @ L @ W1^T)`` smoothness regularizer that the
    reference training loop adds alongside the MSE loss, reproducing
    LINGER's defining manifold-regularization structure.
    """

    def __init__(self, n_tf: int = 6, n_re: int = 10) -> None:
        """Initialize LINGER.

        Parameters
        ----------
        n_tf : int
            Number of transcription-factor input features.
        n_re : int
            Number of regulatory-element (chromatin accessibility) input
            features.
        """
        super().__init__()
        n_in = n_tf + n_re
        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

        adjacency = torch.zeros(n_in, n_in)
        adjacency[:n_tf, n_tf:] = torch.rand(n_tf, n_re) > 0.7
        adjacency[n_tf:, :n_tf] = adjacency[:n_tf, n_tf:].t()
        degree = adjacency.sum(dim=1)
        d_sqrt_inv = torch.diag(1.0 / (degree.sqrt() + 1e-12))
        laplacian = d_sqrt_inv @ (torch.diag(degree) - adjacency) @ d_sqrt_inv
        self.register_buffer("laplacian", laplacian)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict target-gene expression from TF/RE features.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(batch, n_tf + n_re)`` standardized TF-expression and
            RE-accessibility features for one gene.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` predicted target-gene expression.
        """
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def manifold_penalty(self) -> torch.Tensor:
        """Compute the graph-Laplacian manifold-regularization term.

        Returns
        -------
        torch.Tensor
            Scalar ``trace(W1 @ L @ W1^T)``, the same smoothness penalty
            the reference training loop adds to the MSE loss.
        """
        weight = cast(torch.Tensor, self.fc1.weight)
        laplacian = cast(torch.Tensor, self.laplacian)
        return torch.trace(weight @ laplacian @ weight.t())


def build_linger() -> nn.Module:
    """Build a compact LINGER manifold-regularized regulatory regression net.

    Returns
    -------
    nn.Module
        LINGER reconstruction in evaluation mode.
    """
    return LINGER(n_tf=6, n_re=10).eval()


def example_input_linger() -> torch.Tensor:
    """Create example input for :func:`build_linger`.

    Returns
    -------
    torch.Tensor
        Shape ``(4, 16)`` standardized TF/RE feature vectors.
    """
    return torch.randn(4, 16)


# ---------------------------------------------------------------------------
# MetagenBERT: Deep Sets / attention-MIL bag classifier over read embeddings.
# ---------------------------------------------------------------------------


class _Phi(nn.Module):
    """Per-instance encoder mapping a read embedding to a latent vector."""

    def __init__(self, embed_size: int, hidden: int = 32) -> None:
        """Initialize the per-instance encoder.

        Parameters
        ----------
        embed_size : int
            Dimension of each frozen-genomic-LLM read embedding.
        hidden : int
            Output latent dimension.
        """
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_size, hidden), nn.LeakyReLU(), nn.Dropout(0.2))
        self.last_hidden_size = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode every instance in a bag."""
        return self.net(x)


class _Rho(nn.Module):
    """Bag-level classifier mapping a pooled representation to a label."""

    def __init__(self, in_size: int, hidden: int = 16, output_size: int = 1) -> None:
        """Initialize the bag-level classifier.

        Parameters
        ----------
        in_size : int
            Dimension of the pooled bag representation.
        hidden : int
            Hidden layer width.
        output_size : int
            Number of output classes/logits.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify the pooled bag representation."""
        return self.net(x)


class MetagenBERT(nn.Module):
    """MetagenBERT Deep-Sets attention-MIL classifier over read embeddings.

    Each read (or read-cluster) embedding in a metagenomic sample is
    independently mapped through ``phi``, gated-attention pooling
    aggregates the bag into a single sample representation, and ``rho``
    predicts the sample-level phenotype -- reproducing the reference
    ``DeepSets`` module's attention-MIL topology over frozen DNABERT
    embeddings.
    """

    def __init__(self, embed_size: int = 48, phi_hidden: int = 32, rho_hidden: int = 16) -> None:
        """Initialize MetagenBERT.

        Parameters
        ----------
        embed_size : int
            Dimension of each (frozen genomic-LLM) per-read embedding.
        phi_hidden : int
            Latent dimension produced by the per-instance encoder ``phi``.
        rho_hidden : int
            Hidden width of the bag-level classifier ``rho``.
        """
        super().__init__()
        self.phi = _Phi(embed_size, phi_hidden)
        self.attention = nn.Sequential(
            nn.Linear(phi_hidden, phi_hidden // 3 + 1), nn.Tanh(), nn.Linear(phi_hidden // 3 + 1, 1)
        )
        self.rho = _Rho(phi_hidden, rho_hidden, output_size=1)

    def forward(self, read_embeddings: torch.Tensor) -> torch.Tensor:
        """Classify a bag (sample) of per-read embeddings.

        Parameters
        ----------
        read_embeddings : torch.Tensor
            Shape ``(batch, n_reads, embed_size)`` per-read (or per-
            cluster-centroid) embedding vectors for one sample.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` sample-level phenotype logit.
        """
        instance_repr = self.phi(read_embeddings)
        attn_scores = self.attention(instance_repr)
        attn_weights = F.softmax(attn_scores, dim=1)
        pooled = torch.bmm(attn_weights.transpose(1, 2), instance_repr).squeeze(1)
        return self.rho(pooled)


def build_metagenbert() -> nn.Module:
    """Build a compact MetagenBERT attention-MIL bag classifier.

    Returns
    -------
    nn.Module
        MetagenBERT reconstruction in evaluation mode.
    """
    return MetagenBERT(embed_size=48, phi_hidden=32, rho_hidden=16).eval()


def example_input_metagenbert() -> torch.Tensor:
    """Create example input for :func:`build_metagenbert`.

    Returns
    -------
    torch.Tensor
        Shape ``(2, 20, 48)`` synthetic per-read embeddings for two samples.
    """
    return torch.randn(2, 20, 48)


# ---------------------------------------------------------------------------
# MethylBERT: DMR-conditioned BERT read classifier for methylation calls.
# ---------------------------------------------------------------------------


class MethylBERT(nn.Module):
    """MethylBERT DMR-conditioned read-level methylation classifier.

    A small BERT-style transformer encodes a per-CpG methylation-call
    token sequence for one read; the differentially-methylated-region
    (DMR) the read was drawn from is embedded and concatenated as an
    extra feature channel onto every position of the sequence output,
    and the flattened, DMR-conditioned representation is classified by an
    MLP into tumour-vs-normal read origin, matching the reference
    ``MethylBertEmbeddedDMR`` topology.
    """

    def __init__(
        self,
        vocab_size: int = 5,
        seq_len: int = 20,
        hidden: int = 16,
        n_layers: int = 2,
        n_heads: int = 4,
        n_dmrs: int = 8,
    ) -> None:
        """Initialize MethylBERT.

        Parameters
        ----------
        vocab_size : int
            Methylation-call token vocabulary (methylated / unmethylated /
            no-CpG / CLS / PAD).
        seq_len : int
            Number of methylation-call positions per read (excluding CLS).
        hidden : int
            BERT hidden dimension.
        n_layers : int
            Number of BERT encoder layers.
        n_heads : int
            Number of self-attention heads.
        n_dmrs : int
            Number of distinct DMR identities the DMR-embedding table
            covers.
        """
        super().__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos_embed = nn.Embedding(seq_len + 1, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            batch_first=True,
            norm_first=True,
        )
        self.bert = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.dmr_encoder = nn.Embedding(n_dmrs, seq_len + 1)
        self.read_classifier = nn.Sequential(
            nn.Linear((hidden + 1) * (seq_len + 1), seq_len + 1),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.LayerNorm(seq_len + 1),
            nn.Linear(seq_len + 1, 2),
        )

    def forward(self, methylation_tokens: torch.Tensor, dmr_id: torch.Tensor) -> torch.Tensor:
        """Classify reads as tumour- or normal-derived, conditioned on DMR.

        Parameters
        ----------
        methylation_tokens : torch.Tensor
            Shape ``(batch, seq_len + 1)`` per-CpG methylation-call token
            ids (position 0 is a CLS token).
        dmr_id : torch.Tensor
            Shape ``(batch,)`` integer id of the DMR each read was sampled
            from.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 2)`` tumour-vs-normal read-origin logits.
        """
        positions = torch.arange(methylation_tokens.shape[1], device=methylation_tokens.device)
        hidden = self.embed(methylation_tokens) + self.pos_embed(positions).unsqueeze(0)
        sequence_output = self.bert(hidden)

        encoded_dmr = self.dmr_encoder(dmr_id)
        conditioned = torch.cat([sequence_output, encoded_dmr.unsqueeze(-1)], dim=-1)

        flat = conditioned.reshape(conditioned.shape[0], -1)
        return self.read_classifier(flat)


def build_methylbert() -> nn.Module:
    """Build a compact MethylBERT DMR-conditioned read classifier.

    Returns
    -------
    nn.Module
        MethylBERT reconstruction in evaluation mode.
    """
    return MethylBERT(vocab_size=5, seq_len=20, hidden=16, n_layers=2, n_heads=4, n_dmrs=8).eval()


def example_input_methylbert() -> tuple[torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_methylbert`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(methylation_tokens, dmr_id)`` of shape ``(3, 21)`` and ``(3,)``.
    """
    methylation_tokens = torch.randint(0, 5, (3, 21))
    dmr_id = torch.randint(0, 8, (3,))
    return methylation_tokens, dmr_id


# ---------------------------------------------------------------------------
# MMSplice: five-tower modular splicing-effect scorer.
# ---------------------------------------------------------------------------


class _MaskedAvgPoolConvTower(nn.Module):
    """1D-conv scorer over a one-hot DNA window with 0-masked average pooling.

    Reproduces ``ConvDNA`` + ``GlobalAveragePooling1D_Mask0``: positions
    that are all-zero in the one-hot encoding (padding) are excluded from
    the pooling average.
    """

    def __init__(self, out_dim: int, channels: int = 16) -> None:
        """Initialize the masked-average-pool conv tower.

        Parameters
        ----------
        out_dim : int
            Dimension of the module's output score vector.
        channels : int
            Number of convolutional output channels.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(4, channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.out_proj = nn.Linear(channels, out_dim)

    def forward(self, one_hot_dna: torch.Tensor) -> torch.Tensor:
        """Score a one-hot DNA window with 0-masked average pooling.

        Parameters
        ----------
        one_hot_dna : torch.Tensor
            Shape ``(batch, 4, seq_len)`` one-hot ACGT encoding; an
            all-zero column marks a padded position.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, out_dim)`` module score.
        """
        h = torch.relu(self.conv1(one_hot_dna))
        h = torch.relu(self.conv2(h))
        mask = one_hot_dna.amax(dim=1, keepdim=True)  # (batch, 1, seq_len), 1 where valid
        h = h * mask
        pooled = h.sum(dim=2) / mask.sum(dim=2).clamp(min=1e-12)
        return self.out_proj(pooled)


class MMSplice(nn.Module):
    """MMSplice five-module splicing-variant-effect scorer.

    Five independent conv towers score the acceptor-intron, acceptor,
    exon, donor, and donor-intron sequence windows around a splice site;
    their scores are concatenated into a modular-score vector and linearly
    combined into one delta-logit-PSI splicing-effect prediction,
    reproducing the reference ``MMSplice.predict_modular_scores_on_batch``
    plus its linear combiner.
    """

    def __init__(self, intron_len: int = 30, exon_len: int = 40, site_len: int = 20) -> None:
        """Initialize MMSplice.

        Parameters
        ----------
        intron_len : int
            Length of each intronic (acceptor-intron / donor-intron)
            window.
        exon_len : int
            Length of the exon-body window.
        site_len : int
            Length of each splice-site (acceptor / donor) window.
        """
        super().__init__()
        self.acceptor_intron_tower = _MaskedAvgPoolConvTower(1)
        self.acceptor_tower = _MaskedAvgPoolConvTower(1)
        self.exon_tower = _MaskedAvgPoolConvTower(1)
        self.donor_tower = _MaskedAvgPoolConvTower(1)
        self.donor_intron_tower = _MaskedAvgPoolConvTower(1)
        self.combiner = nn.Linear(5, 1)

    def forward(
        self,
        acceptor_intron: torch.Tensor,
        acceptor: torch.Tensor,
        exon: torch.Tensor,
        donor: torch.Tensor,
        donor_intron: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a delta-logit-PSI splicing-effect score.

        Parameters
        ----------
        acceptor_intron : torch.Tensor
            Shape ``(batch, 4, intron_len)`` one-hot upstream-intron window.
        acceptor : torch.Tensor
            Shape ``(batch, 4, site_len)`` one-hot acceptor splice-site
            window.
        exon : torch.Tensor
            Shape ``(batch, 4, exon_len)`` one-hot exon-body window.
        donor : torch.Tensor
            Shape ``(batch, 4, site_len)`` one-hot donor splice-site window.
        donor_intron : torch.Tensor
            Shape ``(batch, 4, intron_len)`` one-hot downstream-intron
            window.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` delta-logit-PSI splicing-effect score.
        """
        scores = torch.cat(
            [
                self.acceptor_intron_tower(acceptor_intron),
                self.acceptor_tower(acceptor),
                self.exon_tower(exon),
                self.donor_tower(donor),
                self.donor_intron_tower(donor_intron),
            ],
            dim=1,
        )
        return self.combiner(scores)


def build_mmsplice() -> nn.Module:
    """Build a compact MMSplice five-module splicing-effect scorer.

    Returns
    -------
    nn.Module
        MMSplice reconstruction in evaluation mode.
    """
    return MMSplice(intron_len=30, exon_len=40, site_len=20).eval()


def example_input_mmsplice() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Create example input for :func:`build_mmsplice`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        One-hot ``(acceptor_intron, acceptor, exon, donor, donor_intron)``
        windows of shape ``(2, 4, 30)``, ``(2, 4, 20)``, ``(2, 4, 40)``,
        ``(2, 4, 20)``, and ``(2, 4, 30)`` respectively.
    """

    def one_hot_dna(batch: int, length: int) -> torch.Tensor:
        idx = torch.randint(0, 4, (batch, length))
        return F.one_hot(idx, num_classes=4).permute(0, 2, 1).float()

    acceptor_intron = one_hot_dna(2, 30)
    acceptor = one_hot_dna(2, 20)
    exon = one_hot_dna(2, 40)
    donor = one_hot_dna(2, 20)
    donor_intron = one_hot_dna(2, 30)
    return acceptor_intron, acceptor, exon, donor, donor_intron


MENAGERIE_ENTRIES = [
    ("LangCell", "build_langcell", "example_input_langcell", "2024", "BIO"),
    ("LDARNet", "build_ldarnet", "example_input_ldarnet", "2026", "BIO"),
    ("LINGER", "build_linger", "example_input_linger", "2024", "BIO"),
    ("MetagenBERT", "build_metagenbert", "example_input_metagenbert", "2026", "BIO"),
    ("MethylBERT", "build_methylbert", "example_input_methylbert", "2025", "BIO"),
    ("MMSplice", "build_mmsplice", "example_input_mmsplice", "2019", "BIO"),
]
