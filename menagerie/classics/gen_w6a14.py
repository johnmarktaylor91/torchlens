"""Menagerie batch w6a14: genomics/transcriptomics deep-learning classics for
paired donor/acceptor splice-site residual-CNN prediction, single-nucleotide
RNA-language-model pretraining, dual dilated-CNN gene-encoder + long-context
attention tissue-specific splicing, negative-binomial spatial cell-type
deconvolution, and graph-neighbor-aggregation spatial-transcriptomics gene
imputation.

Sources checked (reference only; no cloning, no pip installs):
  - SpaOTsc (cand_00795): Cang & Nie, Nature Communications 2020,
    https://github.com/zcang/SpaOTsc. Inspected ``spaotsc/SpaOTsc.py`` and
    ``requirements.txt`` directly via the GitHub API: the entire package is
    built on ``scipy``/``POT`` (optimal transport), ``networkx``/``igraph``
    (graph clustering), and ``sklearn.ensemble.RandomForestRegressor`` /
    ``GradientBoostingRegressor`` for its "infer_signal_range_ml" step. There
    is no ``torch`` import anywhere in the repository and no ``nn.Module``
    of any kind -- the entire method is a classical optimal-transport +
    random-forest pipeline, not a deep neural network. SKIPPED: not a
    trainable nn.Module architecture (genuinely no neural-network component
    to reimplement; the transport-plan/random-forest machinery it wraps is
    a scipy/sklearn algorithm, not a distinctive NN mechanism).
  - Splam (cand_00797): Chao, Mao, et al., Genome Biology 2024,
    https://github.com/Kuanhao-Chao/splam (``test/SPLAM.py``, class
    ``SPLAM``; ``ResidualUnit``/``Skip``). The defining mechanism: a
    **SpliceAI-style dilated 1D-residual-CNN** over a one-hot-encoded
    400+context-nt donor+acceptor window, but with **grouped ("cardinality")
    convolutions** inside each residual unit (``groups=bot_channels //
    CARDINALITY_ITEM``, a ResNeXt-style cardinality split absent from plain
    SpliceAI) and dilation rates that grow across four residual-block groups
    (``AR`` = 1,1,1,1 / 4,4,4,4 / 10,10,10,10 / 25,25,25,25) with a skip
    accumulator (``Skip``) summed back in every 4 blocks -- i.e. "grouped-
    convolution dilated residual stack with periodic skip-accumulation over
    a paired donor/acceptor 400nt window" is Splam's namesake contribution
    over the ungrouped SpliceAI residual stack. Reimplemented with the same
    grouped-conv residual-unit structure, growing-dilation block groups, and
    periodic skip accumulation at a reduced channel width and sequence
    length, ending in the reference's per-position 3-way (neither/donor/
    acceptor) softmax.
  - SpliceBERT (cand_00798): Chen, Xu, et al., Briefings in Bioinformatics
    2024, https://github.com/biomed-AI/SpliceBERT (``src/splicebert_model.py``,
    a near-verbatim fork of HuggingFace's ``modeling_bert.py`` with an
    optional FlashAttention path; README confirms a **6-layer BERT** loaded
    via the stock ``transformers.BertModel``/``BertConfig`` API with
    **single-nucleotide tokenization** (vocab: N/A/C/G/T + BERT specials,
    ``AutoTokenizer.encode`` on whitespace-separated bases) pretrained on
    vertebrate primary RNA sequences 64-1024nt long. The defining mechanism
    is squarely "a config of an installed library model" per the base-env
    allowance: this is built directly via ``transformers.BertConfig`` +
    ``transformers.BertModel`` at tiny dims with the same nucleotide-level
    vocabulary and 6-layer depth (random-init weights, since only the
    architecture -- not the pretrained checkpoint -- is in scope for this
    catalog).
  - SpliceTransformer / SpTransformer (cand_00800): Chao, Ye, Shen, et al.,
    Nature Communications 2024, https://github.com/ShenLab-Genomics/
    SpliceTransformer (``model/model.py``, classes ``SpTransformer``,
    ``SpEncoder_L``/``SpEncoder2_L``, ``ResBlock``, ``AttnBlock``). The
    defining mechanism: **two separately-pretrained dilated-residual-CNN
    gene encoders** (``SpEncoder_L``/``SpEncoder2_L``, SpliceAI-style
    ``ResBlock`` stacks with growing dilation and periodic dense-skip
    accumulation, run in ``feature=True`` mode to emit intermediate
    per-position feature maps rather than final logits) whose concatenated
    features are fused with a **fresh small conv-embedding of the raw
    one-hot sequence** (``conv1``: two 1x1 convs), projected to a working
    width (``conv2``), and passed through a **long-context self-attention
    block** (``AttnBlock``: the reference uses ``SinkhornTransformer`` +
    ``AxialPositionalEmbedding`` for linear-memory long-range attention over
    up to 8192 positions; this port uses a standard ``nn.TransformerEncoder``
    over the same conv-fused token sequence as the "long-context attention
    over dual-CNN gene features" mechanism, since the Sinkhorn/axial
    packages are not in the base env) -- ending in a dual splice-type
    (3-way) / tissue-usage (per-tissue sigmoid) prediction head, matching
    the reference's ``self.splice``/``self.usage`` output convs. I.e. "two
    frozen dilated-residual-CNN gene encoders feeding a long-context
    self-attention block for tissue-specific splice prediction" is
    SpTransformer's namesake contribution over a single-encoder, tissue-
    agnostic SpliceAI-style model. Reimplemented with the same dual-encoder
    concat-then-attend topology and dual output head at drastically reduced
    channel widths, encoder depth, attention depth/length, and tissue count.
  - Stereoscope (cand_00801): Andersson, Bergenstrahle, et al., Communications
    Biology 2020, https://github.com/almaan/stereoscope (``stsc/models.py``,
    classes ``ScModel``/``STModel``). The defining mechanism: a **two-stage
    negative-binomial generative deconvolution model** -- ``ScModel`` learns
    per-gene, per-cell-type NB rate parameters (``theta`` -> softplus ->
    per-cell-type rate ``R``) and a shared per-gene NB "success probability"
    logit (``o``) from single-cell reference data; ``STModel`` then holds
    those single-cell-derived rates **fixed** and learns per-spot,
    per-cell-type **mixing proportions** (``theta`` -> softplus -> ``v``,
    one unconstrained "unknown" type ``Z = K+1`` slot included) plus a
    per-gene multiplicative bias (``beta``) and additive technical-noise
    term (``eta``), combining them via ``r = einsum('gz,zs->gs', R_hat, v)``
    into a spot-by-gene NB rate matrix that is the sufficient statistic for
    the spatial-spot NB log-likelihood -- i.e. "single-cell-reference NB
    rate estimation feeding a per-spot proportion-mixture NB generative
    model, with an explicit free 'unknown cell type' slot and per-gene
    bias/noise" is Stereoscope's namesake spatial cell-type deconvolution
    contribution over regression-based or marker-gene deconvolution.
    Reimplemented as a single ``nn.Module`` composing both stages' forward
    computation into one traceable inference pass (single-cell rate
    parameters -> per-spot proportion mixture -> per-gene bias/noise ->
    combined NB rate matrix) at reduced gene/cell-type/spot counts, since
    the reference trains the two stages with separate optimizers but the
    forward-pass composition (its literal ``r = einsum(...)`` combination)
    is the architecture's defining mechanism.
  - stImpute (cand_00802): (no confirmed peer-reviewed venue at build time;
    method paper on the topic of ESM-2-conditioned spatial-transcriptomics
    gene imputation), https://github.com/cquzys/stImpute (``model.py``,
    classes ``AutoEncoder``, ``GS_block``, ``Trans``). The defining
    mechanism: a **GraphSAGE-style mean-aggregation graph layer**
    (``GS_block``: normalize a gene-gene cosine-similarity adjacency by
    row-sum, aggregate neighbor gene profiles, concatenate with the gene's
    own profile, project+normalize -- stacked ``gnnlayers`` deep) transforms
    a gene-by-cell expression block (genes as graph nodes, built from either
    raw expression or an optional ESM-2-embedding-derived similarity graph)
    into graph-refined gene features, which are then passed through a dense
    **``trans`` MLP head** to predict expression for unmeasured genes,
    jointly trained (EM-style, alternating ``Trans``/``AutoEncoder`` steps)
    with a bottleneck ``AutoEncoder`` used for nearest-neighbor cell
    matching between spatial and reference scRNA-seq data, plus a separate
    **``reliable`` MLP head** that regresses a per-gene prediction-
    reliability score from the same graph-refined-then-transformed features
    -- i.e. "gene-graph GraphSAGE aggregation feeding a dense imputation
    head, paired with a reliability-scoring head over the same
    representation" is stImpute's namesake contribution over plain kNN or
    autoencoder-only spatial gene imputation. Reimplemented with the same
    ``GS_block`` mean-aggregation graph-conv stack, dense ``trans``
    imputation head, and ``reliable`` scoring head (the single-pass
    ``Trans.forward``/``reliable_predict`` computation graph; the EM
    training loop, the separate ``AutoEncoder`` nearest-neighbor-matching
    stage, and the optional frozen ESM-2 embedding lookup are training-
    /preprocessing-time-only and out of scope for a single forward trace of
    the *distinctive* graph-imputation mechanism) at reduced gene/cell
    counts and hidden width.

Five of six candidates are reimplemented from scratch in base-env torch (+
``transformers`` for SpliceBERT's BERT config, already a declared base-env
dependency); SpaOTsc is skipped as it has no PyTorch or nn.Module component
whatsoever. No repo cloning, no pip installs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import BertConfig, BertModel

# ============================================================
# Splam -- grouped-convolution ("cardinality") dilated residual
# stack with periodic skip accumulation over a paired
# donor/acceptor window (Kuanhao-Chao/splam)
# ============================================================


class _SplamResidualUnit(nn.Module):
    """Port of ``test/SPLAM.py``'s ``ResidualUnit``: a grouped-convolution
    (ResNeXt-style "cardinality") dilated residual block.
    """

    def __init__(self, length: int, width: int, dilation: int, cardinality_item: int = 4) -> None:
        super().__init__()
        groups = max(1, length // cardinality_item)
        self.bn1 = nn.BatchNorm1d(length)
        self.relu = nn.LeakyReLU(0.1)
        self.bn2 = nn.BatchNorm1d(length)
        pad = (width - 1) * dilation // 2
        self.conv1 = nn.Conv1d(length, length, width, dilation=dilation, padding=pad, groups=groups)
        self.conv2 = nn.Conv1d(length, length, width, dilation=dilation, padding=pad, groups=groups)

    def forward(self, x: Tensor, skip: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one grouped-conv dilated residual unit, passing the running skip through."""
        h1 = self.relu(self.bn1(self.conv1(x)))
        h2 = self.relu(self.bn2(self.conv2(h1)))
        return x + h2, skip


class _SplamSkip(nn.Module):
    """Port of ``test/SPLAM.py``'s ``Skip``: a 1x1 conv accumulated into the running skip sum."""

    def __init__(self, length: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(length, length, 1)

    def forward(self, x: Tensor, skip: Tensor) -> tuple[Tensor, Tensor]:
        """Accumulate a 1x1-projected copy of ``x`` into the running skip sum."""
        return x, self.conv(x) + skip


class Splam(nn.Module):
    """Paired donor/acceptor splice-site residual-CNN predictor.

    Grouped-convolution ("cardinality") dilated residual units, arranged in
    four groups of growing dilation rate with a skip-accumulator inserted
    every four blocks, over a one-hot-encoded paired donor+acceptor nucleotide
    window; ends in a per-position 3-way (neither/donor/acceptor) softmax.
    """

    def __init__(
        self,
        length: int = 32,
        widths: tuple[int, ...] = (11, 11, 11, 11, 21, 21, 21, 21),
        dilations: tuple[int, ...] = (1, 1, 1, 1, 4, 4, 4, 4),
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, length, 1)
        self.skip1 = _SplamSkip(length)
        blocks: list[nn.Module] = []
        for i, (w, d) in enumerate(zip(widths, dilations)):
            blocks.append(_SplamResidualUnit(length, w, d))
            if (i + 1) % 4 == 0:
                blocks.append(_SplamSkip(length))
        self.residual_blocks = nn.ModuleList(blocks)
        self.last_conv = nn.Conv1d(length, 3, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict per-position splice-site class probabilities.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, 4, seq_len)`` one-hot-encoded nucleotide window.
        """
        h, skip = self.skip1(
            self.conv1(x),
            torch.zeros(x.shape[0], self.conv1.out_channels, x.shape[2], device=x.device),
        )
        for block in self.residual_blocks:
            h, skip = block(h, skip)
        return F.softmax(self.last_conv(skip), dim=1)


def build_splam() -> nn.Module:
    """Build a small Splam grouped-conv dilated-residual splice-site predictor."""
    return Splam(
        length=32, widths=(11, 11, 11, 11, 21, 21, 21, 21), dilations=(1, 1, 1, 1, 4, 4, 4, 4)
    ).eval()


def example_input_splam() -> Tensor:
    """Return a one-hot-encoded paired donor/acceptor nucleotide window for Splam."""
    onehot = torch.zeros(2, 4, 200)
    idx = torch.randint(0, 4, (2, 200))
    onehot.scatter_(1, idx.unsqueeze(1), 1.0)
    return onehot


# ============================================================
# SpliceBERT -- 6-layer BERT over single-nucleotide tokens
# (biomed-AI/SpliceBERT; built via transformers.BertModel)
# ============================================================


def build_splicebert() -> nn.Module:
    """Build a small SpliceBERT: 6-layer BERT over a single-nucleotide vocabulary.

    Ports the reference's use of the stock ``transformers.BertModel`` /
    ``BertConfig`` API with single-nucleotide tokenization (vocab: N/A/C/G/T
    plus BERT specials) and 6 transformer-encoder layers, at reduced hidden
    width, head count, and max sequence length (random-init weights; the
    reference model is pretrained on 2M+ vertebrate RNA sequences, out of
    scope here).
    """
    cfg = BertConfig(
        vocab_size=10,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        pad_token_id=0,
    )
    return BertModel(cfg).eval()


def example_input_splicebert() -> Tensor:
    """Return a batch of single-nucleotide token-id sequences for SpliceBERT."""
    return torch.randint(4, 10, (2, 48))


# ============================================================
# SpliceTransformer (SpTransformer) -- dual dilated-residual-CNN
# gene encoders feeding a long-context self-attention block for
# tissue-specific splice-site + usage prediction
# (ShenLab-Genomics/SpliceTransformer)
# ============================================================


class _SpResBlock(nn.Module):
    """Port of ``model/model.py``'s ``ResBlock``: a dilated pre-activation residual unit."""

    def __init__(self, length: int, width: int, dilation: int) -> None:
        super().__init__()
        pad = (width - 1) * dilation // 2
        self.bn1 = nn.BatchNorm1d(length)
        self.conv1 = nn.Conv1d(length, length, width, dilation=dilation, padding=pad)
        self.bn2 = nn.BatchNorm1d(length)
        self.conv2 = nn.Conv1d(length, length, width, dilation=dilation, padding=pad)

    def forward(self, x: Tensor) -> Tensor:
        """Apply one dilated pre-activation residual unit."""
        h = self.conv1(torch.relu(self.bn1(x)))
        h = self.conv2(torch.relu(self.bn2(h)))
        return x + h


class _SpGeneEncoder(nn.Module):
    """Port of ``model/model.py``'s ``SpEncoder_L``: a dilated-residual-CNN
    gene encoder that emits intermediate per-position skip features
    (``feature=True``) for downstream fusion.
    """

    def __init__(
        self,
        length: int,
        widths: tuple[int, ...] = (11, 11, 11, 11),
        dilations: tuple[int, ...] = (1, 1, 4, 4),
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, length, 1)
        self.skip = nn.Conv1d(length, length, 1)
        self.resblocks = nn.ModuleList(
            [_SpResBlock(length, w, d) for w, d in zip(widths, dilations)]
        )
        self.dense_convs = nn.ModuleList(
            [
                nn.Conv1d(length, length, 1)
                for i in range(len(widths))
                if (i + 1) % 4 == 0 or (i + 1) == len(widths)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the accumulated skip feature map (pretrained-encoder feature output)."""
        conv = self.conv1(x)
        skip = self.skip(conv)
        j = 0
        for i, block in enumerate(self.resblocks):
            conv = block(conv)
            if (i + 1) % 4 == 0 or (i + 1) == len(self.resblocks):
                skip = skip + self.dense_convs[j](conv)
                j += 1
        return skip


class SpliceTransformer(nn.Module):
    """Dual dilated-residual-CNN gene encoders fused with a long-context
    self-attention block for tissue-specific splice-site + usage prediction.

    Ports ``model/model.py``'s ``SpTransformer``: two gene encoders (here of
    matched size, standing in for the reference's two differently-sized
    "pretrained" encoders) emit concatenated skip-feature maps; a small
    fresh conv-embedding of the raw one-hot sequence is fused in; the
    combined per-position features are self-attended by a Transformer
    encoder (standing in for the reference's Sinkhorn/axial-positional
    long-context attention) and split into a splice-type softmax head and a
    per-tissue usage sigmoid head.
    """

    def __init__(
        self,
        dim: int = 16,
        encoder_len: int = 16,
        tissue_num: int = 6,
        attn_depth: int = 2,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder_a = _SpGeneEncoder(encoder_len)
        self.encoder_b = _SpGeneEncoder(encoder_len)
        self.conv1 = nn.Sequential(nn.Conv1d(4, dim, 1), nn.Conv1d(dim, dim, 1))
        fused_dim = dim + 2 * encoder_len
        self.conv2 = nn.Conv1d(fused_dim, dim * 2, 1)
        layer = nn.TransformerEncoderLayer(d_model=dim * 2, nhead=attn_heads, batch_first=True)
        self.attn = nn.TransformerEncoder(layer, num_layers=attn_depth)
        self.splice_head = nn.Conv1d(dim * 2, 3, 1)
        self.usage_head = nn.Conv1d(dim * 2, tissue_num, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict per-position splice-type and per-tissue usage scores.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, 4, seq_len)`` one-hot-encoded nucleotide window.
        """
        with torch.no_grad():
            feat_a = self.encoder_a(x)
            feat_b = self.encoder_b(x)
        feat = torch.cat([feat_a, feat_b], dim=1)
        fresh = self.conv1(x)
        fused = torch.cat([feat, fresh], dim=1)
        emb = self.conv2(fused)
        emb = emb.transpose(1, 2)
        attn_out = self.attn(emb).transpose(1, 2)
        splice_out = F.softmax(self.splice_head(attn_out), dim=1)
        usage_out = torch.sigmoid(self.usage_head(attn_out))
        return torch.cat([splice_out, usage_out], dim=1)


def build_splicetransformer() -> nn.Module:
    """Build a small SpliceTransformer dual-encoder + attention splice/usage predictor."""
    return SpliceTransformer(
        dim=16, encoder_len=16, tissue_num=6, attn_depth=2, attn_heads=4
    ).eval()


def example_input_splicetransformer() -> Tensor:
    """Return a one-hot-encoded nucleotide window for SpliceTransformer."""
    onehot = torch.zeros(2, 4, 64)
    idx = torch.randint(0, 4, (2, 64))
    onehot.scatter_(1, idx.unsqueeze(1), 1.0)
    return onehot


# ============================================================
# Stereoscope -- negative-binomial two-stage generative
# spatial cell-type deconvolution (almaan/stereoscope)
# ============================================================


class Stereoscope(nn.Module):
    """Negative-binomial spatial cell-type deconvolution.

    Ports ``stsc/models.py``'s ``ScModel``/``STModel`` forward computation
    fused into one traceable module: single-cell-reference-derived
    per-gene-per-cell-type NB rate parameters (``theta`` -> softplus ->
    ``R``) are combined with per-spot cell-type mixing proportions
    (``theta_st`` -> softplus -> ``v``, including one free "unknown cell
    type" slot) via ``einsum('gz,zs->gs', ...)``, plus a per-gene
    multiplicative bias and additive noise term, into a spot-by-gene NB rate
    matrix -- Stereoscope's namesake deconvolution mechanism.
    """

    def __init__(self, n_genes: int = 40, n_celltypes: int = 6, n_spots: int = 10) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_celltypes = n_celltypes
        self.n_spots = n_spots
        z = n_celltypes + 1

        self.sc_theta = nn.Parameter(torch.randn(n_genes, n_celltypes) * 0.1)
        self.sc_o = nn.Parameter(torch.randn(n_genes, 1) * 0.1)

        self.st_theta = nn.Parameter(torch.randn(z, n_spots) * 0.1)
        self.beta = nn.Parameter(torch.randn(n_genes, 1) * 0.1)
        self.eta = nn.Parameter(torch.randn(n_genes, 1) * 0.1)

    def forward(self) -> tuple[Tensor, Tensor]:
        """Return the combined spot-by-gene NB rate matrix and the shared NB logit.

        Takes no data tensor: Stereoscope's forward computation combines only
        its own learned rate/proportion/bias parameters into the NB
        sufficient-statistic rate matrix used by the (training-time-only) NB
        log-likelihood loss.
        """
        rate_per_celltype = F.softplus(self.sc_theta)
        proportions = F.softplus(self.st_theta)
        gene_bias = F.softplus(self.beta)
        noise = F.softplus(self.eta)
        rate_hat = torch.cat([gene_bias * rate_per_celltype, noise], dim=1)
        spot_gene_rate = torch.einsum("gz,zs->gs", rate_hat, proportions)
        return spot_gene_rate, self.sc_o


def build_stereoscope() -> nn.Module:
    """Build a small Stereoscope NB spatial cell-type deconvolution model."""
    return Stereoscope(n_genes=40, n_celltypes=6, n_spots=10).eval()


def example_input_stereoscope() -> tuple[()]:
    """Return the (empty) input for Stereoscope: it forwards over its own learned parameters only."""
    return ()


# ============================================================
# stImpute -- GraphSAGE-style gene-graph aggregation feeding a
# dense imputation head + reliability-scoring head
# (cquzys/stImpute)
# ============================================================


class _GsBlock(nn.Module):
    """Port of ``model.py``'s ``GS_block``: a GraphSAGE-style mean-aggregation
    graph-convolution layer over a gene-gene similarity adjacency.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.empty(input_dim * 2, output_dim))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Aggregate neighbor gene features (row-normalized adjacency) and project.

        Parameters
        ----------
        x : Tensor
            Shape ``(n_genes, n_cells)`` gene-by-cell expression block.
        adj : Tensor
            Shape ``(n_genes, n_genes)`` gene-gene similarity adjacency
            (self-loop included, as in the reference's ``build_graph_by_gene``).
        """
        n = adj.shape[0]
        adj_no_self = adj - torch.eye(n, device=adj.device)
        adj_norm = adj_no_self / (adj_no_self.sum(1, keepdim=True) + 1e-12)
        neigh_feats = adj_norm.mm(x)
        combined = torch.cat(
            [x.reshape(-1, self.input_dim), neigh_feats.reshape(-1, self.input_dim)], dim=1
        )
        combined = F.relu(combined @ self.weight)
        return F.normalize(combined, p=2, dim=1).reshape(x.shape[0], -1)


class StImpute(nn.Module):
    """Graph-neighbor-aggregation spatial-transcriptomics gene imputer.

    Ports ``model.py``'s ``Trans`` module: a stack of ``GS_block`` GraphSAGE
    mean-aggregation layers over a gene-gene similarity graph (``adj``,
    shape ``n_genes x n_genes``, per the reference's ``build_graph_by_gene``
    over gene rows) refines a gene-by-neighbor-cell expression block (each
    spatial cell mapped to its ``n_neighbors`` nearest reference cells, per
    the reference's ``find_neighbors`` KNN step, so the neighbor-cell axis
    has length ``n_spatial_cells * n_neighbors``), which a dense MLP head
    (``trans``) transforms into imputed spatial-cell-by-gene predictions; a
    parallel dense head (``reliable``) regresses a per-gene reliability
    score from the same imputed representation.
    """

    def __init__(
        self,
        n_genes: int = 12,
        n_spatial_cells: int = 6,
        n_neighbors: int = 4,
        hidden_dim: int = 32,
        gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_neighbors = n_neighbors
        self.graph_layers = nn.ModuleList([_GsBlock(n_genes, n_genes) for _ in range(gnn_layers)])
        self.trans = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reliable = nn.Sequential(
            nn.Linear(n_spatial_cells, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor, graph: Tensor) -> tuple[Tensor, Tensor]:
        """Impute per-gene expression and predict a per-gene reliability score.

        Parameters
        ----------
        x : Tensor
            Shape ``(n_spatial_cells * n_neighbors, n_genes)`` reference-cell
            expression block (each spatial cell's ``n_neighbors`` nearest
            reference cells, stacked along the leading dimension).
        graph : Tensor
            Shape ``(n_genes, n_genes)`` gene-gene similarity adjacency used
            by the ``GS_block`` layers.
        """
        x_hat = x.t()
        for layer in self.graph_layers:
            x_hat = layer(x_hat, graph)
        x_hat = x_hat.reshape(-1, self.n_genes)
        y_hat = self.trans(x_hat).reshape(self.n_neighbors, -1).t()
        reliable_score = self.reliable(y_hat.t()).squeeze(-1)
        return y_hat, reliable_score


def build_stimpute() -> nn.Module:
    """Build a small stImpute GraphSAGE-aggregation gene-imputation model."""
    return StImpute(
        n_genes=12, n_spatial_cells=6, n_neighbors=4, hidden_dim=32, gnn_layers=2
    ).eval()


def example_input_stimpute() -> tuple[Tensor, Tensor]:
    """Return (neighbor-cell expression block, gene-gene similarity adjacency) inputs for stImpute."""
    n_genes, n_spatial_cells, n_neighbors = 12, 6, 4
    x = torch.rand(n_spatial_cells * n_neighbors, n_genes)
    sim = torch.rand(n_genes, n_genes)
    graph = (sim + sim.t()) / 2 + torch.eye(n_genes)
    return x, graph


MENAGERIE_ENTRIES = [
    ("Splam", "build_splam", "example_input_splam", "2024", "BIO"),
    ("SpliceBERT", "build_splicebert", "example_input_splicebert", "2024", "BIO"),
    (
        "SpliceTransformer (SpTransformer)",
        "build_splicetransformer",
        "example_input_splicetransformer",
        "2024",
        "BIO",
    ),
    ("Stereoscope", "build_stereoscope", "example_input_stereoscope", "2020", "BIO"),
    ("stImpute", "build_stimpute", "example_input_stimpute", "2024", "BIO"),
]
