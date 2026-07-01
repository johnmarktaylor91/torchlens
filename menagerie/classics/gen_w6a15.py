"""Menagerie batch w6a15: spatial-transcriptomics morphology-graph smoothing,
optimal-transport-flavored single-cell-to-spatial mapping, rank-value gene-
language-model pretraining, CRISPR-Cas13d guide-efficacy CNN, per-cell
free-parameter RNA-velocity fitting, and multi-transformer conservation-
aware splicing-outcome prediction.

Sources checked (reference only; no cloning, no pip installs):
  - stLearn (cand_00803): Pham, Tan, et al., BiomedicalMachineLearning/stLearn,
    https://github.com/BiomedicalMachineLearning/stLearn (``stlearn/
    image_preprocessing/model_zoo.py``, class ``Model``; ``stlearn/
    image_preprocessing/feature_extractor.py``, function ``extract_feature``;
    ``stlearn/spatial/sme/sme_normalize.py``, function ``sme_normalize``;
    ``stlearn/spatial/sme/_weighting_matrix.py``). The defining mechanism:
    stLearn's namesake "SME" (Spatial, Morphological, Expression) normalization
    is **not** a single end-to-end trainable network but a three-signal
    neighbor-weighted smoothing pipeline built on top of a **pretrained
    ImageNet CNN backbone** (``model_zoo.Model`` -- ResNet50/VGG16/Inception-
    v3/ConvNeXt with the classification head stripped, used purely as a
    frozen morphology-feature extractor over per-spot H&E tissue tiles) that
    is combined with a **physical-distance kernel** and a **gene-expression-
    correlation kernel** into one composite spot-spot similarity/weight
    matrix; that weight matrix is used to compute a weighted-neighbor
    "imputed" expression estimate per spot, which is then averaged with the
    spot's own raw expression -- i.e. "CNN-derived tissue-morphology
    similarity + spatial proximity + expression correlation, fused into one
    neighbor-weighting kernel used to smooth/denoise every spot's expression"
    is stLearn's SME contribution over spatial-only or expression-only
    smoothing. Reimplemented as one differentiable forward pass: a small
    random-init CNN tile encoder (standing in for the frozen ImageNet
    backbone, since only the "CNN morphology embedding as one smoothing
    signal" mechanism -- not ImageNet pretrained weights -- is in scope for
    this architecture catalog) produces per-spot morphology embeddings;
    those are combined with spot coordinates and raw expression into the
    same tri-modal similarity-weighted neighbor-averaging smoother, at
    reduced tile size, spot count, and gene count.
  - Tangram (cand_00805): Biancalani, Scalia, et al., Broad Institute /
    Klarman Cell Observatory, Nature Methods 2021,
    https://github.com/broadinstitute/Tangram (``tangram/
    mapping_optimizer.py``, class ``Mapper``, method ``_loss_fn``). The
    defining mechanism: a single **learnable dense cell-by-spot mapping
    matrix** ``M`` (shape ``n_cells x n_spots``) is **row-softmaxed** into a
    doubly-stochastic-like probability matrix and used to project single-
    cell profiles onto space via ``G_pred = softmax(M, dim=1).T @ S`` (an
    optimal-transport-flavored soft assignment, not a parametric encoder/
    decoder) -- the mapping matrix itself, trained by gradient descent
    against a cosine-similarity gene-expression-matching loss (plus optional
    density/entropy/spatial regularizers), *is* the whole model: no other
    network layers -- i.e. "one free (cells x spots) softmax coupling
    matrix, learned end-to-end to align single-cell and spatial expression"
    is Tangram's namesake mapping contribution over cluster-label-transfer
    or marker-gene correlation methods. Reimplemented with the exact same
    learnable coupling matrix, row-softmax, and inner-product spatial
    projection (the forward-pass-traceable mapping mechanism; the iterative
    training loop and optional regularizer terms are training-time-only) at
    reduced cell/spot/gene counts.
  - tGPT (cand_00806): Lixiang Chun et al., iScience 2022,
    https://github.com/lixiangchun/tGPT (``train.py``, using
    ``transformers.AutoModelForCausalLM``/``GPT2Config``; README: "Generative
    pretraining on rankings of top-expressing genes"). The defining
    mechanism: raw continuous scRNA-seq expression is **not** fed to the
    network directly -- each cell's genes are rank-ordered by expression and
    the top-``k`` gene identifiers become a token sequence (a "sentence" of
    ranked gene symbols), which a **standard GPT-2 causal transformer**
    (``AutoModelForCausalLM`` over ``GPT2Config``) is pretrained to model
    autoregressively -- i.e. "expression-rank-to-token-sequence encoding,
    modeled by an off-the-shelf GPT-2 causal LM" is tGPT's namesake
    generative-pretraining-on-gene-rankings contribution over expression-
    value regression networks. The trainable architecture itself is exactly
    GPT-2 (confirmed via the reference training script's direct
    ``transformers`` import); reimplemented via ``transformers.GPT2Config``
    + ``GPT2LMHeadModel`` at tiny width/depth/vocab, with a helper that
    performs the reference's rank-token encoding of a raw expression vector
    into the input ID sequence the causal LM consumes.
  - TIGER (cand_00807): Wessels, Stirn, et al. (New York Genome Center /
    DaSilva lab), Nature Biotechnology 2023, https://github.com/daklab/tiger
    (``models.py``, class ``Tiger1D``, using ``layers.py``'s
    ``SequenceSequentialWithNonSequenceBypass``; TensorFlow/Keras source
    ported to torch since the declared framework is TF and no PyTorch
    checkpoint exists). The defining mechanism: guide-target (and optional
    guide-RNA) nucleotide sequences are **one-hot encoded and concatenated**
    (5' context + target + 3' context, optionally + guide sequence), passed
    through a **stacked 1D-convolutional feature extractor**
    (``Conv1D(64,k=4)`` x2 + ``MaxPool1D``), flattened, and fed through a
    dense sigmoid-activated regression trunk down to a single scalar --
    predicted Cas13d knockdown efficacy (log-fold-change); scalar
    biochemical/positional features are concatenated in via a "bypass" path
    alongside the flattened conv features before the dense trunk -- i.e.
    "1D CNN over one-hot guide+target(+context) sequence, bypass-fused with
    scalar biochemical features, dense sigmoid regression head" is TIGER's
    namesake sequence-to-efficacy CNN contribution over hand-engineered
    thermodynamic scoring rules. Reimplemented with the same one-hot-
    sequence conv-stack + scalar-feature bypass + sigmoid dense trunk
    topology (a faithful torch port of the Keras ``Tiger1D`` model, since
    only the reference implementation's TensorFlow layers -- not weights --
    are in scope) at reduced sequence length and channel widths.
  - TIVelo (cand_00808): Cui, Lu, Wang, Lin, et al. (CUHK), Nature
    Communications 2025, https://github.com/cuhklinlab/TIVelo (``tivelo/
    velocity/model.py``, class ``model_velo``). The defining mechanism:
    unlike encoder-based RNA-velocity models (scVelo's ODE fit, VeloVAE's
    variational encoder), TIVelo's velocity step optimizes **free per-cell
    parameter tensors** ``v_u``/``v_s`` (one 2-vector per cell x gene, shape
    ``n_obs x n_vars``, directly ``nn.Parameter`` -- no shared weights, no
    encoder network) so that each cell's unspliced/spliced counts plus its
    own velocity vector regress toward its **directed-neighbor-graph
    (DTI-corrected) target expression** (``d_u``/``d_s``, a KNN-weighted
    neighbor average precomputed from a lineage-direction-corrected KNN
    graph), while a second loss term pulls each cell's velocity vector into
    **cosine agreement with its undirected KNN neighbors' velocity vectors**
    (a local-smoothness/consistency regularizer computed via a full
    pairwise cosine-similarity matrix masked by the KNN adjacency) -- i.e.
    "free per-cell velocity parameters fit by neighbor-target regression
    plus KNN-graph cosine-consistency regularization" is TIVelo's namesake
    per-cell velocity-inference contribution over parametric encoder-based
    velocity models. Reimplemented with the same free per-cell ``v_u``/
    ``v_s`` parameters, neighbor-target MSE loss, and full pairwise KNN-
    masked cosine-consistency loss, returned as the forward pass's scalar
    training-objective output (the model *is* its loss landscape -- there is
    no separate inference-only path in the reference) at reduced cell/gene
    counts and a small dense synthetic KNN/directed-neighbor adjacency.
  - TRASPr (cand_00809): Qazi, Barash, et al. (BioCiphers, UPenn), eLife 2024,
    https://bitbucket.org/biociphers/traspr (``src/transformers/
    modeling_bert.py``, class
    ``BertForSequenceMultiClassificationMultiTransformer``; confirmed a
    DNABERT fork -- ``src/transformers/tokenization_dna.py``,
    ``config/bert-config-{3,4,5,6}`` k-mer configs -- with a splicing-
    specific head added in ``bos/tasks/oracle.py``). The defining mechanism:
    a cassette-exon splicing event is split into **four flanking sequence
    segments** (exon1+intron1, intron1+acceptor, acceptor+intron2,
    intron2+exon2) and each segment is independently encoded by its **own
    weight-independent DNABERT transformer** (``nn.ModuleList`` of 4 full
    ``BertModel`` instances, k-mer DNA tokenization); each segment's BERT
    additionally consumes a **per-token conservation-value embedding**
    (``consval_embeddings``, an 11-bucket lookup table summed into the
    token+position+segment embedding, encoding per-nucleotide evolutionary
    conservation) that is concatenated to the standard embedding sum before
    the transformer stack; the four segment BERTs' pooled ``[CLS]`` outputs
    are concatenated and passed through a 2-layer dense head (optionally
    fused with tissue-identity and scalar biochemical features) to a PSI
    (percent-spliced-in) sigmoid/regression output -- i.e. "four
    weight-independent per-segment DNABERT transformers, each conservation-
    value-conditioned via an extra embedding channel, pooled-and-fused by a
    dense splicing-outcome head" is TRASPr's namesake multi-transformer
    tissue-splicing-prediction contribution over single-sequence splice-site
    scoring. Reimplemented with the same four independent tiny BERT-style
    transformer encoders (torch ``nn.TransformerEncoder`` stacks, since only
    the DNABERT *k-mer-tokenized encoder* mechanism -- not the pretrained
    DNABERT checkpoint -- is in scope here), each with its own token +
    position + conservation-value embedding sum, pooled-``[CLS]``
    concatenation, and dense PSI regression head, at drastically reduced
    k-mer vocabulary, sequence length, hidden width, and layer depth.

All six models are reimplemented from scratch in base-env torch (+
``transformers`` for tGPT's GPT-2 causal LM, already a declared base-env
dependency); no repo cloning, no pip installs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GPT2Config, GPT2LMHeadModel

# ============================================================
# stLearn -- CNN-morphology + spatial + expression neighbor-
# weighted smoothing (BiomedicalMachineLearning/stLearn)
# ============================================================


class _TileEncoder(nn.Module):
    """Small conv-stack standing in for stLearn's pretrained ImageNet CNN
    backbone (``model_zoo.Model``), used purely as a per-spot tissue-tile
    morphology-feature extractor (frozen in the reference; random-init here
    since only the CNN-embedding-as-a-smoothing-signal mechanism is in
    scope, not ImageNet weights).
    """

    def __init__(self, out_dim: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, out_dim)

    def forward(self, tiles: Tensor) -> Tensor:
        """Encode ``(n_spots, 3, H, W)`` H&E tissue tiles to morphology embeddings."""
        h = F.relu(self.conv1(tiles))
        h = F.relu(self.conv2(h))
        h = self.pool(h).flatten(1)
        return self.fc(h)


class StLearnSME(nn.Module):
    """SME (Spatial, Morphological, Expression) neighbor-weighted smoother.

    Ports ``stlearn.spatial.sme.sme_normalize``'s composite weighting: a CNN
    tile encoder produces per-spot morphology embeddings; spatial-distance,
    morphology-similarity, and expression-correlation kernels are fused
    (multiplicatively, as in ``weights_matrix_all``) into one spot-spot
    weight matrix; a weighted-neighbor "imputed" expression estimate is
    averaged with each spot's own raw expression.
    """

    def __init__(self, n_genes: int = 32, morph_dim: int = 16) -> None:
        super().__init__()
        self.tile_encoder = _TileEncoder(out_dim=morph_dim)

    def forward(self, tiles: Tensor, coords: Tensor, expr: Tensor) -> Tensor:
        """Smooth per-spot expression via the tri-modal SME weight matrix.

        Parameters
        ----------
        tiles : Tensor
            Shape ``(n_spots, 3, H, W)`` H&E tissue tiles, one per spot.
        coords : Tensor
            Shape ``(n_spots, 2)`` spatial (x, y) spot coordinates.
        expr : Tensor
            Shape ``(n_spots, n_genes)`` raw expression matrix.
        """
        morph = self.tile_encoder(tiles)
        morph_sim = F.normalize(morph, dim=1) @ F.normalize(morph, dim=1).t()

        dist = torch.cdist(coords, coords)
        spatial_sim = torch.exp(-dist / (dist.mean() + 1e-6))

        expr_sim = F.normalize(expr, dim=1) @ F.normalize(expr, dim=1).t()

        weight = (
            (spatial_sim.clamp(min=0) + 1e-3)
            * (morph_sim.clamp(min=0) + 1e-3)
            * (expr_sim.clamp(min=0) + 1e-3)
        )
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-6)

        imputed = weight @ expr
        return 0.5 * (expr + imputed)


def build_stlearn_sme() -> nn.Module:
    """Build a small stLearn SME morphology-graph smoothing model."""
    return StLearnSME(n_genes=32, morph_dim=16).eval()


def example_input_stlearn_sme() -> tuple[Tensor, Tensor, Tensor]:
    """Return (tissue tiles, spot coordinates, expression matrix) for stLearn SME."""
    n_spots = 12
    tiles = torch.rand(n_spots, 3, 16, 16)
    coords = torch.rand(n_spots, 2) * 100.0
    expr = torch.rand(n_spots, 32)
    return tiles, coords, expr


# ============================================================
# Tangram -- learnable softmax coupling matrix mapping single
# cells onto spatial expression (broadinstitute/Tangram)
# ============================================================


class Tangram(nn.Module):
    """Optimal-transport-flavored single-cell-to-spatial mapping.

    Ports ``mapping_optimizer.Mapper``: a single learnable
    ``(n_cells, n_spots)`` matrix, row-softmaxed into a soft coupling, is
    used to project single-cell expression onto spatial expression via
    ``softmax(M, dim=1).T @ S``.
    """

    def __init__(self, n_cells: int = 20, n_spots: int = 14) -> None:
        super().__init__()
        self.mapping = nn.Parameter(torch.randn(n_cells, n_spots) * 0.1)

    def forward(self, single_cell_expr: Tensor) -> Tensor:
        """Map single-cell expression onto spatial spots.

        Parameters
        ----------
        single_cell_expr : Tensor
            Shape ``(n_cells, n_genes)`` single-cell expression matrix
            (``S`` in the reference).
        """
        m_probs = F.softmax(self.mapping, dim=1)
        return m_probs.t() @ single_cell_expr


def build_tangram() -> nn.Module:
    """Build a small Tangram single-cell-to-spatial mapping model."""
    return Tangram(n_cells=20, n_spots=14).eval()


def example_input_tangram() -> Tensor:
    """Return a single-cell expression matrix for Tangram."""
    return torch.rand(20, 26)


# ============================================================
# tGPT -- GPT-2 causal LM pretrained on rank-ordered gene-
# identity token sequences (lixiangchun/tGPT)
# ============================================================


class TGPT(nn.Module):
    """Rank-value gene-expression GPT-2.

    Ports ``train.py``'s ``AutoModelForCausalLM(GPT2Config(...))``: expression
    is not modeled directly -- callers rank-encode each cell's top-expressed
    genes into a token-ID sequence (see ``rank_encode``); this module is the
    causal-LM architecture that consumes that sequence.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        n_positions: int = 32,
        n_embd: int = 32,
        n_layer: int = 2,
        n_head: int = 4,
    ) -> None:
        super().__init__()
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        self.gpt2 = GPT2LMHeadModel(config)

    @staticmethod
    def rank_encode(expr: Tensor, vocab_size: int, seq_len: int) -> Tensor:
        """Rank-order a ``(batch, n_genes)`` expression matrix into gene-rank token IDs.

        Ports the reference's "generative pretraining on rankings of
        top-expressing genes" preprocessing: each cell's genes are sorted
        by descending expression and the top-``seq_len`` gene indices
        become the input token sequence (clipped into the model vocabulary).
        """
        order = torch.argsort(expr, dim=1, descending=True)[:, :seq_len]
        return (order % vocab_size).long()

    def forward(self, gene_rank_ids: Tensor) -> Tensor:
        """Compute next-gene-rank-token logits for a rank-encoded gene sequence.

        Parameters
        ----------
        gene_rank_ids : Tensor
            Shape ``(batch, seq_len)`` integer gene-rank token IDs, as
            produced by :meth:`rank_encode`.
        """
        return self.gpt2(input_ids=gene_rank_ids).logits


def build_tgpt() -> nn.Module:
    """Build a small tGPT rank-value gene-token causal LM."""
    return TGPT(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=4).eval()


def example_input_tgpt() -> Tensor:
    """Return a rank-encoded gene-token-ID sequence for tGPT."""
    expr = torch.rand(4, 100)
    return TGPT.rank_encode(expr, vocab_size=64, seq_len=16)


# ============================================================
# TIGER -- 1D-CNN + scalar-feature-bypass CRISPR-Cas13d guide-
# efficacy regressor (daklab/tiger, TF source ported to torch)
# ============================================================


class Tiger1D(nn.Module):
    """CRISPR-Cas13d guide-efficacy predictor.

    Ports ``models.py``'s ``Tiger1D`` (Keras): one-hot-encoded target
    (+ flanking context, + optional guide) sequence is convolved by a
    stacked 1D-CNN, flattened, bypass-concatenated with scalar biochemical
    features, and regressed to a scalar knockdown log-fold-change via a
    sigmoid-activated dense trunk.
    """

    def __init__(self, seq_len: int = 30, n_scalar_feats: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=4, padding="same")
        self.conv2 = nn.Conv1d(64, 64, kernel_size=4, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        conv_out_dim = 64 * (seq_len // 2)
        self.fc1 = nn.Linear(conv_out_dim + n_scalar_feats, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 32)
        self.dropout3 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, seq_onehot: Tensor, scalar_feats: Tensor) -> Tensor:
        """Predict guide-efficacy log-fold-change.

        Parameters
        ----------
        seq_onehot : Tensor
            Shape ``(batch, 4, seq_len)`` one-hot-encoded nucleotide
            sequence (target + flanking context, channels-first).
        scalar_feats : Tensor
            Shape ``(batch, n_scalar_feats)`` non-sequence biochemical /
            positional features, bypass-concatenated before the dense trunk.
        """
        h = F.relu(self.conv1(seq_onehot))
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        h = self.dropout1(h).flatten(1)
        h = torch.cat([h, scalar_feats], dim=1)
        h = torch.sigmoid(self.fc1(h))
        h = self.dropout2(h)
        h = torch.sigmoid(self.fc2(h))
        h = self.dropout3(h)
        return self.fc3(h)


def build_tiger() -> nn.Module:
    """Build a small TIGER Cas13d guide-efficacy CNN."""
    return Tiger1D(seq_len=30, n_scalar_feats=8).eval()


def example_input_tiger() -> tuple[Tensor, Tensor]:
    """Return (one-hot sequence, scalar features) for TIGER."""
    batch = 6
    seq_onehot = F.one_hot(torch.randint(0, 4, (batch, 30)), num_classes=4).permute(0, 2, 1).float()
    scalar_feats = torch.randn(batch, 8)
    return seq_onehot, scalar_feats


# ============================================================
# TIVelo -- free per-cell velocity parameters fit by neighbor-
# target regression + KNN cosine-consistency (cuhklinlab/TIVelo)
# ============================================================


class TIVeloVelocity(nn.Module):
    """Per-cell free-parameter RNA-velocity fitting model.

    Ports ``tivelo/velocity/model.py``'s ``model_velo``: unspliced/spliced
    velocity vectors ``v_u``/``v_s`` are free per-cell parameters (no shared
    encoder weights), fit so that ``x + v`` regresses toward a directed-
    neighbor-graph target ``d`` while ``v`` stays cosine-consistent with its
    undirected-KNN neighbors' velocity vectors. Returns the reference's
    training objective (this model *is* its loss landscape; there is no
    separate inference path in the source).
    """

    def __init__(self, n_cells: int = 24, n_genes: int = 20) -> None:
        super().__init__()
        self.v_u = nn.Parameter(torch.zeros(n_cells, n_genes))
        self.v_s = nn.Parameter(torch.zeros(n_cells, n_genes))

    def forward(
        self,
        x_u: Tensor,
        x_s: Tensor,
        d_u: Tensor,
        d_s: Tensor,
        knn: Tensor,
        alpha_1: float = 1.0,
        alpha_2: float = 0.1,
    ) -> Tensor:
        """Compute the neighbor-target + KNN-cosine-consistency velocity loss.

        Parameters
        ----------
        x_u, x_s : Tensor
            Shape ``(n_cells, n_genes)`` unspliced / spliced counts.
        d_u, d_s : Tensor
            Shape ``(n_cells, n_genes)`` directed-neighbor-graph (DTI-
            corrected) target unspliced / spliced expression.
        knn : Tensor
            Shape ``(n_cells, n_cells)`` undirected KNN adjacency used for
            the velocity-vector cosine-consistency regularizer.
        alpha_1 : float
            Weight on the spliced-count regression term.
        alpha_2 : float
            Weight on the KNN cosine-consistency regularizer.
        """
        mse_u = F.mse_loss(x_u + self.v_u, d_u)
        mse_s = F.mse_loss(x_s + self.v_s, d_s)

        norm_v_u = self.v_u.norm(p=2, dim=1, keepdim=True)
        norm_v_s = self.v_s.norm(p=2, dim=1, keepdim=True)
        cos_u = (self.v_u @ self.v_u.t()) / (norm_v_u @ norm_v_u.t() + 1e-6)
        cos_s = (self.v_s @ self.v_s.t()) / (norm_v_s @ norm_v_s.t() + 1e-6)

        n_neighs = knn.sum(dim=1)
        cos_u = torch.mean((cos_u * knn).sum(dim=1) / (n_neighs + 1e-6))
        cos_s = torch.mean((cos_s * knn).sum(dim=1) / (n_neighs + 1e-6))

        return mse_u + alpha_1 * mse_s - alpha_2 * cos_u - alpha_2 * cos_s


def build_tivelo() -> nn.Module:
    """Build a small TIVelo per-cell free-parameter velocity model."""
    return TIVeloVelocity(n_cells=24, n_genes=20).eval()


def example_input_tivelo() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (x_u, x_s, d_u, d_s, knn adjacency) for TIVelo."""
    n_cells, n_genes = 24, 20
    x_u = torch.rand(n_cells, n_genes)
    x_s = torch.rand(n_cells, n_genes)
    d_u = torch.rand(n_cells, n_genes)
    d_s = torch.rand(n_cells, n_genes)
    knn = (torch.rand(n_cells, n_cells) > 0.7).float()
    knn.fill_diagonal_(0)
    return x_u, x_s, d_u, d_s, knn


# ============================================================
# TRASPr -- four weight-independent conservation-value-
# conditioned DNABERT transformers -> PSI head
# (biociphers/traspr, DNABERT fork)
# ============================================================


class _ConsvalBertEncoder(nn.Module):
    """One of TRASPr's four weight-independent per-segment BERT encoders.

    Ports the reference's per-segment ``BertModel`` with the added
    ``consval_embeddings`` channel: token + position + conservation-value
    bucket embeddings are summed before a standard transformer encoder
    stack; the pooled first-token (``[CLS]``) representation is returned.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        hidden: int,
        n_layers: int,
        n_heads: int,
        n_consval_buckets: int = 11,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.consval_emb = nn.Embedding(n_consval_buckets, hidden, padding_idx=0)
        self.emb_norm = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pooler = nn.Linear(hidden, hidden)

    def forward(self, input_ids: Tensor, consval: Tensor) -> Tensor:
        """Encode one k-mer-tokenized DNA segment to its pooled ``[CLS]`` embedding.

        Parameters
        ----------
        input_ids : Tensor
            Shape ``(batch, seq_len)`` integer k-mer token IDs.
        consval : Tensor
            Shape ``(batch, seq_len)`` integer per-token conservation-value
            bucket IDs.
        """
        b, n = input_ids.shape
        pos = torch.arange(n, device=input_ids.device).unsqueeze(0).expand(b, -1)
        emb = self.token_emb(input_ids) + self.pos_emb(pos) + self.consval_emb(consval)
        emb = self.emb_norm(emb)
        hidden = self.encoder(emb)
        return torch.tanh(self.pooler(hidden[:, 0]))


class TRASPr(nn.Module):
    """Four-transformer, conservation-aware splicing-outcome (PSI) predictor.

    Ports ``modeling_bert.BertForSequenceMultiClassificationMultiTransformer``:
    exon1+intron1, intron1+acceptor, acceptor+intron2, intron2+exon2 segments
    are each encoded by their own weight-independent conservation-value-
    conditioned BERT encoder; the four pooled ``[CLS]`` embeddings are
    concatenated and passed through a 2-layer dense head to a PSI logit.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        seq_len: int = 24,
        hidden: int = 24,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.segments = nn.ModuleList(
            [_ConsvalBertEncoder(vocab_size, seq_len, hidden, n_layers, n_heads) for _ in range(4)],
        )
        self.dropout = nn.Dropout(0.1)
        self.act = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(hidden * 4, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, input_ids: Tensor, consval: Tensor) -> Tensor:
        """Predict a PSI (percent-spliced-in) logit from four flanking segments.

        Parameters
        ----------
        input_ids : Tensor
            Shape ``(batch, 4, seq_len)`` k-mer token IDs for the four
            segments (E1+I1, I1+A, A+I2, I2+E2).
        consval : Tensor
            Shape ``(batch, 4, seq_len)`` per-token conservation-value
            bucket IDs, matching ``input_ids``.
        """
        pooled = [
            encoder(input_ids[:, i], consval[:, i]) for i, encoder in enumerate(self.segments)
        ]
        pooled_output = torch.cat(pooled, dim=-1)
        hidden = self.act(self.fc1(self.dropout(pooled_output)))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


def build_traspr() -> nn.Module:
    """Build a small TRASPr four-transformer conservation-aware PSI predictor."""
    return TRASPr(vocab_size=32, seq_len=24, hidden=24, n_layers=2, n_heads=4).eval()


def example_input_traspr() -> tuple[Tensor, Tensor]:
    """Return (segment k-mer token IDs, conservation-value buckets) for TRASPr."""
    batch, seq_len = 5, 24
    input_ids = torch.randint(0, 32, (batch, 4, seq_len))
    consval = torch.randint(0, 11, (batch, 4, seq_len))
    return input_ids, consval


MENAGERIE_ENTRIES = [
    ("stLearn", "build_stlearn_sme", "example_input_stlearn_sme", "2020", "BIO"),
    ("Tangram", "build_tangram", "example_input_tangram", "2021", "BIO"),
    ("tGPT", "build_tgpt", "example_input_tgpt", "2022", "BIO"),
    ("TIGER", "build_tiger", "example_input_tiger", "2023", "BIO"),
    ("TIVelo", "build_tivelo", "example_input_tivelo", "2025", "BIO"),
    ("TRASPr", "build_traspr", "example_input_traspr", "2024", "BIO"),
]
