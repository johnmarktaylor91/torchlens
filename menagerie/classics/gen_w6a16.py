"""Menagerie batch w6a16: two-phase regulatory-region CNN cascade, hierarchical
ALiBi cross-attention gene-expression transformer, Enformer-backbone personal-
genome fine-tuning head adapter, siamese forward/reverse-strand viral-contig
CNN classifier, and E(n)-equivariant graph neural network CDR-loop modeler.

Sources checked (reference only; no cloning, no pip installs):
  - TREDNet (cand_00810): Hudaiberdiev, Taylor, et al., okurman/TREDNet,
    https://github.com/okurman/TREDNet (``lib/v1/models.py``, function
    ``define_model``; ``kipoi/TREDNet/phase_one/model.yaml`` +
    ``kipoi/TREDNet/phase_two_HepG2/model.py``, class ``PhaseTwoModel``).
    The defining mechanism: TREDNet is a **two-phase cascade** of 1D CNNs
    over one-hot DNA -- phase one is a multi-task 1D-conv classifier
    (``Conv1D(k=4) -> BN -> pool -> Conv1D(k=2) -> Dense`` stack) trained to
    predict 1,924 DNase/histone-mark/TF-binding tracks from a 2 kb one-hot
    sequence window; phase two takes that 1,924-dim "epigenomic-track"
    vector as its *input* (reshaped as a length-1924 1-channel sequence, not
    raw DNA) and pushes it through a second, smaller 1D-conv stack that
    regresses a single scalar tissue-specific regulatory/enhancer score
    (``PhaseTwoModel.predict_on_batch`` explicitly chains
    ``phase_one_model.predict_on_batch(x) -> phase_two model``) -- i.e.
    "one CNN predicts an intermediate multi-track epigenomic profile from
    DNA, and a second CNN consumes that profile (not the sequence) to score
    tissue-specific regulatory activity" is TREDNet's two-phase contribution
    over a single end-to-end regressor. Reimplemented as one differentiable
    module chaining a compact phase-one 1D-CNN (one-hot DNA -> track vector)
    into a compact phase-two 1D-CNN (track vector -> scalar enhancer score),
    at reduced sequence length, filter counts, and track count.
  - VariantFormer (cand_00812): CZI Science, czi-ai/variantformer,
    https://github.com/czi-ai/variantformer (``seq2gene/model.py``, class
    ``Seq2GenePredictor``; ``seq2gene/modules/layers.py``, classes
    ``EpigeneticsModulator``, ``GeneModulator``, ``TissueExpressionHeads``,
    ``get_alibi_slopes``/``ContextFlashAttentionEncoderLayer``). The
    defining mechanism: a **hierarchical two-stage ALiBi-biased transformer**
    -- stage one (``EpigeneticsModulator``) is a stack of ALiBi self-
    attention encoder layers that let cis-regulatory-element (CRE) token
    embeddings for a locus attend to each other and refine a
    "regulatory-context" representation; stage two (``GeneModulator``) is a
    stack of ALiBi cross-attention layers where gene-embedding query tokens
    attend to that refined CRE context (cross-attention only, gene tokens
    never self-attend to each other) to produce a gene representation
    conditioned on its surrounding regulatory landscape; per-tissue linear
    heads (``TissueExpressionHeads``) then regress expression from the
    gene's contextualized embedding -- i.e. "ALiBi self-attention over
    regulatory elements, then ALiBi cross-attention pulling genes into that
    regulatory context, then per-tissue expression heads" is VariantFormer's
    hierarchical CRE-to-gene contribution over a flat single-stream
    sequence transformer (the real model additionally reads raw DNA through
    separate frozen CRE/gene tokenizer sub-networks and uses FlashAttention
    kernels; both are training-infrastructure/pretrained-encoder details
    outside the scope of this from-scratch reimplementation). Reimplemented
    with plain ``nn.MultiheadAttention`` plus a manual ALiBi additive bias
    for both the self-attention CRE stage and the cross-attention gene
    stage, followed by per-tissue linear heads, at reduced embedding width,
    CRE-token count, gene-token count, and tissue count.
  - Variformer (cand_00813): Rastogi, Reyna, et al. (also released as
    ni-lab/finetuning-enformer), shirondru/enformer_fine_tuning, Genome
    Biology 2025, https://github.com/shirondru/enformer_fine_tuning
    (``code/pl_models.py``, class ``LitModelHeadAdapterWrapper``, which
    wraps ``enformer_pytorch.finetune.HeadAdapterWrapper`` around
    ``enformer_pytorch.Enformer.from_pretrained``). The defining mechanism:
    Variformer takes the **Enformer** backbone -- a 1D-conv stem, a
    dilated/pooled conv tower that downsamples a long one-hot DNA window,
    and a transformer stack with relative positional attention over the
    pooled sequence bins -- and swaps its frozen pretrained reference-genome
    output head for a **new trainable "head adapter"** (a small linear/MLP
    projection from the final transformer bin embeddings to tissue-specific
    expression) that is fine-tuned end-to-end on *personal-genome* one-hot
    sequences (donor-specific variants baked into the input encoding, not a
    separate variant-effect side-channel) paired with GTEx expression --
    i.e. "reuse Enformer's conv+attention DNA encoder, replace only the
    output head, and fine-tune on individualized reference+variant
    sequences" is Variformer's head-adapter-on-Enformer contribution over
    training a new encoder from scratch. Reimplemented as a compact
    from-scratch Enformer-style backbone (conv stem -> residual pooled conv
    tower -> a relative-position-aware transformer block -> pointwise
    projection) followed by a fresh trainable per-tissue linear head
    adapter, at drastically reduced sequence length, channel width, tower
    depth, and transformer depth.
  - VirHunter (cand_00814): Sukhorukov, Khalili, et al., cbib/virhunter,
    Frontiers in Bioinformatics 2022, https://github.com/cbib/virhunter
    (``virhunter/models/model_10.py`` / ``model_7.py``, function ``model``).
    The defining mechanism: VirHunter classifies an assembled contig into
    virus / plant-host / bacteria by running a **weight-shared 1D CNN
    independently over the forward-strand and reverse-complement-strand**
    one-hot encodings of the same contig (``Conv1D -> LeakyReLU ->
    GlobalMaxPooling1D`` applied identically to both strands via the same
    ``hidden_layers`` list), concatenating the two pooled strand embeddings,
    and feeding the concatenation through a small dense classifier head --
    i.e. "one siamese (weight-tied) conv encoder scanning both DNA strands,
    fused before classification" is VirHunter's strand-symmetric contribution
    over a single-strand-only contig classifier (the real tool ensembles
    three such siamese CNNs at different context lengths/kernel sizes;
    that ensembling is a training/inference-time detail, not a distinct
    architecture). Reimplemented as one siamese weight-shared 1D-CNN over
    explicit forward- and reverse-complement-strand one-hot inputs,
    concatenated into a 3-class (virus/host/bacteria) softmax head, at
    reduced contig length and filter count.
  - ABlooper (cand_00816): Abanades, Georges, et al., oxpig/ABlooper,
    Bioinformatics 2022, https://github.com/oxpig/ABlooper
    (``ABlooper/models.py``, classes ``EGNN``, ``ResEGNN``, ``DecoyGen``).
    The defining mechanism: ABlooper predicts antibody CDR-loop backbone
    coordinates with a stack of **E(n)-equivariant graph neural network**
    (EGNN, Satorras et al. 2021) layers -- each layer computes pairwise
    relative-coordinate and relative-distance features between every residue
    pair, passes the concatenation of the two residues' scalar (rotation/
    translation-invariant) features plus that relative distance through an
    edge MLP, uses a second small MLP on the aggregated edge messages to
    predict a *scalar* per-neighbor weight that rescales (never rotates) the
    normalized relative-coordinate vectors before summing them back onto
    each residue's 3D coordinates (guaranteeing exact E(3) equivariance: the
    coordinate update is a data-dependent linear combination of relative
    vectors, never a learned rotation), while the scalar node features are
    updated by an ordinary residual node MLP -- i.e. "coordinates are only
    ever displaced along relative-position directions with learned scalar
    weights, so rotating/translating the input rotates/translates the
    output identically" is EGNN's namesake equivariance contribution over a
    plain coordinate-regression MLP/transformer. Reimplemented essentially
    verbatim (the reference file is small, self-contained, and already
    minimal): the same ``EGNN`` edge-MLP/coords-MLP/node-MLP layer, stacked
    into a residual ``ResEGNN`` correction block, at reduced residue count,
    feature width, and message dimension (the real ``DecoyGen`` wraps
    several such blocks to produce an ensemble of decoys; one block is kept
    here since decoy-ensembling is a sampling-time detail, not a distinct
    per-decoy architecture).

Skipped:
  - trVAE (cand_00811): theislab/trVAE is architecturally identical to the
    scArches base conditional VAE already reimplemented faithfully as
    ``build_scarches``/``TrvaeModel`` in ``menagerie/classics/gen_w6a9.py``
    (that module's own docstring and class names explicitly identify it as
    "scArches/trVAE" -- trVAE is scArches's reference-mapping conditional
    VAE with additive one-hot-condition ``CondLayers`` in both encoder and
    decoder). Building a second, differently named copy of the exact same
    mechanism would not add a distinct architecture to the catalog.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# TREDNet: two-phase 1D-CNN cascade -- phase one maps one-hot DNA to a
# multi-track epigenomic profile; phase two maps that profile to a scalar
# tissue-specific regulatory/enhancer score.
# ---------------------------------------------------------------------------


class TredNetPhaseOne(nn.Module):
    """Phase-one 1D-CNN: one-hot DNA window -> multi-track epigenomic profile."""

    def __init__(self, seq_len: int = 256, n_tracks: int = 48) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2)
        pooled_len = (seq_len - 3) // 2
        flat_len = (pooled_len - 1) * 64
        self.fc1 = nn.Linear(flat_len, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, n_tracks)

    def forward(self, x_onehot: Tensor) -> Tensor:
        """Map ``(batch, 4, seq_len)`` one-hot DNA to ``(batch, n_tracks)`` scores."""
        h = F.relu(self.bn1(self.conv1(x_onehot)))
        h = self.pool(h)
        h = F.relu(self.conv2(h))
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))


class TredNetPhaseTwo(nn.Module):
    """Phase-two 1D-CNN: multi-track profile -> scalar enhancer score."""

    def __init__(self, n_tracks: int = 48) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=4)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2)
        pooled_len = (n_tracks - 3) // 2
        flat_len = (pooled_len - 1) * 32
        self.fc1 = nn.Linear(flat_len, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, tracks: Tensor) -> Tensor:
        """Map ``(batch, n_tracks)`` epigenomic tracks to ``(batch, 1)`` score."""
        h = tracks.unsqueeze(1)
        h = F.relu(self.bn1(self.conv1(h)))
        h = self.pool(h)
        h = F.relu(self.conv2(h))
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class TredNet(nn.Module):
    """TREDNet two-phase cascade: DNA -> epigenomic tracks -> enhancer score."""

    def __init__(self, seq_len: int = 256, n_tracks: int = 48) -> None:
        super().__init__()
        self.phase_one = TredNetPhaseOne(seq_len=seq_len, n_tracks=n_tracks)
        self.phase_two = TredNetPhaseTwo(n_tracks=n_tracks)

    def forward(self, x_onehot: Tensor) -> Tensor:
        """Chain phase one and phase two on a one-hot DNA window."""
        tracks = self.phase_one(x_onehot)
        return self.phase_two(tracks)


def build_trednet() -> nn.Module:
    """Build a compact two-phase TREDNet regulatory-region scorer."""
    return TredNet(seq_len=256, n_tracks=48).eval()


def example_input_trednet() -> Tensor:
    """Return a one-hot-encoded DNA window batch for TREDNet."""
    batch, seq_len = 4, 256
    idx = torch.randint(0, 4, (batch, seq_len))
    return F.one_hot(idx, 4).permute(0, 2, 1).float()


# ---------------------------------------------------------------------------
# VariantFormer: ALiBi self-attention over cis-regulatory-element (CRE)
# tokens, then ALiBi cross-attention where gene tokens attend to that CRE
# context, then per-tissue expression heads.
# ---------------------------------------------------------------------------


def _alibi_bias(n_heads: int, n_query: int, n_key: int, device: torch.device) -> Tensor:
    """Build an additive ALiBi bias of shape ``(n_heads, n_query, n_key)``."""
    start = 2.0 ** (-8.0 / n_heads)
    slopes = torch.tensor([start ** (i + 1) for i in range(n_heads)], device=device)
    rel = (
        torch.arange(n_key, device=device)[None, :] - torch.arange(n_query, device=device)[:, None]
    )
    return -slopes[:, None, None] * rel[None].abs().float()


class AlibiSelfAttentionLayer(nn.Module):
    """ALiBi-biased self-attention encoder layer with a GEGLU-style MLP."""

    def __init__(self, emb_dim: int, n_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Refine CRE token embeddings with ALiBi self-attention + MLP."""
        h = self.norm1(x)
        bias = _alibi_bias(self.n_heads, h.shape[1], h.shape[1], h.device)
        bias = bias.repeat(x.shape[0], 1, 1)
        attn_out, _ = self.attn(h, h, h, attn_mask=bias)
        x = x + attn_out
        h = self.norm2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class AlibiCrossAttentionLayer(nn.Module):
    """ALiBi-biased cross-attention layer: gene tokens attend to CRE context."""

    def __init__(self, emb_dim: int, n_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.norm_q = nn.LayerNorm(emb_dim)
        self.norm_kv = nn.LayerNorm(emb_dim)
        self.cross_attn = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, gene: Tensor, cre_context: Tensor) -> Tensor:
        """Pull gene tokens into the refined CRE regulatory context."""
        q = self.norm_q(gene)
        kv = self.norm_kv(cre_context)
        bias = _alibi_bias(self.n_heads, q.shape[1], kv.shape[1], gene.device)
        bias = bias.repeat(gene.shape[0], 1, 1)
        attn_out, _ = self.cross_attn(q, kv, kv, attn_mask=bias)
        gene = gene + attn_out
        h = self.norm2(gene)
        return gene + self.fc2(F.gelu(self.fc1(h)))


class TissueExpressionHeads(nn.Module):
    """Per-tissue linear heads regressing expression from gene embeddings."""

    def __init__(self, emb_dim: int, n_tissues: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, n_tissues)
        )

    def forward(self, gene_emb: Tensor) -> Tensor:
        """Map ``(batch, n_genes, emb_dim)`` embeddings to per-tissue expression."""
        return self.head(gene_emb)


class VariantFormer(nn.Module):
    """Hierarchical CRE-self-attention -> gene-cross-attention -> tissue heads."""

    def __init__(
        self,
        emb_dim: int = 32,
        n_heads: int = 4,
        n_cre_layers: int = 2,
        n_gene_layers: int = 2,
        n_tissues: int = 6,
    ) -> None:
        super().__init__()
        self.epigenetics_modulator = nn.ModuleList(
            [AlibiSelfAttentionLayer(emb_dim, n_heads, emb_dim * 2) for _ in range(n_cre_layers)]
        )
        self.gene_modulator = nn.ModuleList(
            [AlibiCrossAttentionLayer(emb_dim, n_heads, emb_dim * 2) for _ in range(n_gene_layers)]
        )
        self.tissue_heads = TissueExpressionHeads(emb_dim, n_tissues)

    def forward(self, cre_tokens: Tensor, gene_tokens: Tensor) -> Tensor:
        """Refine CRE context, cross-attend genes into it, regress expression."""
        cre_ctx = cre_tokens
        for layer in self.epigenetics_modulator:
            cre_ctx = layer(cre_ctx)
        gene = gene_tokens
        for layer in self.gene_modulator:
            gene = layer(gene, cre_ctx)
        return self.tissue_heads(gene)


def build_variantformer() -> nn.Module:
    """Build a compact hierarchical VariantFormer CRE-to-gene transformer."""
    return VariantFormer(emb_dim=32, n_heads=4, n_cre_layers=2, n_gene_layers=2, n_tissues=6).eval()


def example_input_variantformer() -> tuple[Tensor, Tensor]:
    """Return (CRE-token embeddings, gene-token embeddings) for VariantFormer."""
    batch, n_cre, n_genes, emb_dim = 3, 12, 5, 32
    cre_tokens = torch.randn(batch, n_cre, emb_dim)
    gene_tokens = torch.randn(batch, n_genes, emb_dim)
    return cre_tokens, gene_tokens


# ---------------------------------------------------------------------------
# Variformer: Enformer-style conv-tower + relative-position transformer
# backbone with a fresh trainable per-tissue head adapter, fine-tuned on
# personal-genome (variant-baked-in) one-hot sequences.
# ---------------------------------------------------------------------------


class EnformerConvTowerBlock(nn.Module):
    """One residual pooled conv block of Enformer's dilated conv tower."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: Tensor) -> Tensor:
        """Residual conv followed by 2x max-pool downsampling."""
        h = F.gelu(self.bn(x))
        h = self.conv(h)
        return self.pool(x + h)


class RelativePositionAttention(nn.Module):
    """Self-attention with a learned relative-position additive bias."""

    def __init__(self, dim: int, n_heads: int, max_len: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, 2 * max_len - 1))
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        """Self-attend over pooled sequence bins with relative-position bias."""
        seq_len = x.shape[1]
        idx = torch.arange(seq_len, device=x.device)
        rel_idx = (idx[None, :] - idx[:, None]) + (self.max_len - 1)
        rel_idx = rel_idx.clamp(0, 2 * self.max_len - 2)
        bias = self.rel_bias[:, rel_idx]
        bias = bias.repeat(x.shape[0], 1, 1)
        out, _ = self.attn(x, x, x, attn_mask=bias)
        return out


class EnformerBackbone(nn.Module):
    """Compact Enformer-style conv stem + pooled tower + relative-pos attention."""

    def __init__(
        self, channels: int = 32, n_tower_blocks: int = 2, n_heads: int = 4, max_len: int = 16
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(4, channels, kernel_size=15, padding=7)
        self.tower = nn.ModuleList(
            [EnformerConvTowerBlock(channels) for _ in range(n_tower_blocks)]
        )
        self.norm = nn.LayerNorm(channels)
        self.rel_attn = RelativePositionAttention(channels, n_heads, max_len)
        self.pointwise = nn.Linear(channels, channels)

    def forward(self, x_onehot: Tensor) -> Tensor:
        """Encode ``(batch, 4, seq_len)`` one-hot DNA into per-bin embeddings."""
        h = self.stem(x_onehot)
        for block in self.tower:
            h = block(h)
        h = h.transpose(1, 2)
        h = h + self.rel_attn(self.norm(h))
        return F.relu(self.pointwise(h))


class Variformer(nn.Module):
    """Enformer backbone plus a fresh trainable per-tissue head adapter."""

    def __init__(self, channels: int = 32, n_tissues: int = 4, max_len: int = 16) -> None:
        super().__init__()
        self.backbone = EnformerBackbone(channels=channels, max_len=max_len)
        self.head_adapter = nn.Linear(channels, n_tissues)

    def forward(self, x_onehot: Tensor) -> Tensor:
        """Predict per-bin, per-tissue expression from personal-genome DNA."""
        bin_embeddings = self.backbone(x_onehot)
        pooled = bin_embeddings.mean(dim=1)
        return self.head_adapter(pooled)


def build_variformer() -> nn.Module:
    """Build a compact Enformer-backbone Variformer head-adapter model."""
    return Variformer(channels=32, n_tissues=4, max_len=16).eval()


def example_input_variformer() -> Tensor:
    """Return a one-hot personal-genome DNA window for Variformer."""
    batch, seq_len = 3, 256
    idx = torch.randint(0, 4, (batch, seq_len))
    return F.one_hot(idx, 4).permute(0, 2, 1).float()


# ---------------------------------------------------------------------------
# VirHunter: siamese (weight-shared) 1D CNN independently scanning the
# forward and reverse-complement strands of a contig, fused before a
# 3-class virus/host/bacteria softmax head.
# ---------------------------------------------------------------------------


class VirHunterStrandEncoder(nn.Module):
    """Shared conv encoder applied identically to either DNA strand."""

    def __init__(self, kernel_size: int = 7, filters: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv1d(4, filters, kernel_size=kernel_size)

    def forward(self, x_onehot: Tensor) -> Tensor:
        """Conv -> LeakyReLU -> global max pool over one strand."""
        h = F.leaky_relu(self.conv(x_onehot), negative_slope=0.1)
        return h.amax(dim=-1)


class VirHunter(nn.Module):
    """Siamese forward/reverse-strand CNN classifier for viral contigs."""

    def __init__(self, kernel_size: int = 7, filters: int = 32, dense_ns: int = 32) -> None:
        super().__init__()
        self.strand_encoder = VirHunterStrandEncoder(kernel_size=kernel_size, filters=filters)
        self.fc1 = nn.Linear(filters * 2, dense_ns)
        self.fc2 = nn.Linear(dense_ns, 3)

    def forward(self, forward_strand: Tensor, reverse_strand: Tensor) -> Tensor:
        """Encode both strands with the same weights, fuse, and classify."""
        fwd = self.strand_encoder(forward_strand)
        rev = self.strand_encoder(reverse_strand)
        fused = torch.cat([fwd, rev], dim=-1)
        h = F.relu(self.fc1(fused))
        return self.fc2(h)


def build_virhunter() -> nn.Module:
    """Build a compact siamese-strand VirHunter viral-contig classifier."""
    return VirHunter(kernel_size=7, filters=32, dense_ns=32).eval()


def example_input_virhunter() -> tuple[Tensor, Tensor]:
    """Return (forward-strand, reverse-complement-strand) one-hot contigs."""
    batch, seq_len = 6, 200
    idx = torch.randint(0, 4, (batch, seq_len))
    forward_strand = F.one_hot(idx, 4).permute(0, 2, 1).float()
    reverse_strand = forward_strand.flip(dims=(2,)).flip(dims=(1,))
    return forward_strand, reverse_strand


# ---------------------------------------------------------------------------
# ABlooper: E(n)-equivariant graph neural network (EGNN) -- residue-pair
# edge messages from relative coordinates rescale relative-position vectors
# by a learned scalar weight, guaranteeing exact rotation/translation
# equivariance of the predicted CDR-loop backbone coordinates.
# ---------------------------------------------------------------------------


class CoorsNorm(nn.Module):
    """Normalize relative coordinate vectors, keeping only their direction."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.fn = nn.LayerNorm(1)

    def forward(self, coors: Tensor) -> Tensor:
        """Rescale each relative-coordinate vector to unit norm, phase-gated."""
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        phase = self.fn(norm)
        return phase * normed_coors


class Egnn(nn.Module):
    """One E(n)-equivariant graph layer: scalar features + 3D coordinates."""

    def __init__(self, dim: int, m_dim: int = 16) -> None:
        super().__init__()
        edge_input_dim = (dim * 2) + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            nn.SiLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            nn.SiLU(),
        )
        self.coors_norm = CoorsNorm()
        self.node_mlp = nn.Sequential(
            nn.Linear(dim + m_dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim)
        )
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4), nn.SiLU(), nn.Linear(m_dim * 4, 1)
        )

    def forward(self, feats: Tensor, coors: Tensor) -> tuple[Tensor, Tensor]:
        """Update per-residue scalar features and 3D coordinates equivariantly."""
        rel_coors = rearrange(coors, "b i d -> b i () d") - rearrange(coors, "b j d -> b () j d")
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)

        feats_j = rearrange(feats, "b j d -> b () j d")
        feats_i = rearrange(feats, "b i d -> b i () d")
        feats_i, feats_j = torch.broadcast_tensors(feats_i, feats_j)

        edge_input = torch.cat((feats_i, feats_j, rel_dist), dim=-1)
        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)
        coor_weights = rearrange(coor_weights, "b i j () -> b i j")

        rel_coors = self.coors_norm(rel_coors)
        coors_out = torch.einsum("b i j, b i j c -> b i c", coor_weights, rel_coors) + coors
        m_i = m_ij.sum(dim=-2)

        node_mlp_input = torch.cat((feats, m_i), dim=-1)
        node_out = self.node_mlp(node_mlp_input) + feats

        return node_out, coors_out


class ResEgnn(nn.Module):
    """Stack of EGNN correction layers refining CDR-loop backbone coordinates."""

    def __init__(self, corrections: int = 3, dims_in: int = 20, m_dim: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Egnn(dim=dims_in, m_dim=m_dim) for _ in range(corrections)])

    def forward(self, amino: Tensor, geom: Tensor) -> Tensor:
        """Repeatedly refine backbone coordinates through EGNN layers."""
        for layer in self.layers:
            amino, geom = layer(amino, geom)
        return geom


def build_ablooper() -> nn.Module:
    """Build a compact ABlooper EGNN CDR-loop coordinate refiner."""
    return ResEgnn(corrections=3, dims_in=20, m_dim=16).eval()


def example_input_ablooper() -> tuple[Tensor, Tensor]:
    """Return (per-residue scalar features, initial 3D coordinates) for ABlooper."""
    batch, n_residues, dims_in = 2, 14, 20
    amino = torch.randn(batch, n_residues, dims_in)
    geom = torch.randn(batch, n_residues, 3)
    return amino, geom


MENAGERIE_ENTRIES = [
    ("TREDNet", "build_trednet", "example_input_trednet", "2022", "BIO"),
    ("VariantFormer", "build_variantformer", "example_input_variantformer", "2025", "BIO"),
    ("Variformer", "build_variformer", "example_input_variformer", "2025", "BIO"),
    ("VirHunter", "build_virhunter", "example_input_virhunter", "2022", "BIO"),
    ("ABlooper", "build_ablooper", "example_input_ablooper", "2022", "BIO"),
]
