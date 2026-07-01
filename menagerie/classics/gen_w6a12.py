"""Menagerie batch w6a12: single-cell multi-omics deep-learning classics for
heterogeneous cell-feature bipartite graph modality prediction, shared-latent
matrix tri-factorization for mosaic multi-omic integration, product-of-experts
multimodal variational fusion, self-attention-gated RNA/ATAC joint encoding,
gradient-reversal adversarial semi-supervised cell-type label transfer, and
batch-conditioned hyperspherical (von Mises-Fisher) generative embedding.

Sources checked (reference only; no cloning, no pip installs):
  - scMoGNN (cand_00781): Wen, Hongzhi, et al., "Graph Neural Networks for
    Multimodal Single-Cell Data Integration", KDD 2022,
    https://arxiv.org/abs/2203.01884; official implementation ships inside
    the DANCE package as ``ScMoGCN`` --
    https://github.com/OmicsML/dance,
    ``dance/modules/multi_modality/predict_modality/scmogcn.py`` (classes
    ``ScMoGCN``, methods ``calculate_initial_embedding``/``conv``/
    ``attention_agg``/``propagate``/``forward``). The defining mechanism: a
    **heterogeneous cell<->feature bipartite graph** (feature nodes = genes
    / peaks, cell nodes = cells) is built with reciprocal
    ``feature2cell``/``cell2feature`` edges (and an optional ``pathway``
    feature-feature edge type); each layer runs a GraphSAGE-style
    mean-neighbor-aggregation convolution independently over each edge type
    from the *current* cell and feature embeddings, and when more than one
    edge type feeds a node type (e.g. the optional pathway edges alongside
    ``feature2cell``) the two resulting messages are fused via a learned
    **gated/attention aggregation** (``attention_agg``: query = the node's
    embedding from the previous layer, keys = the per-edge-type conv
    outputs, softmax-weighted sum) rather than naive concatenation; after
    ``conv_layers`` rounds of this bipartite message passing the final cell
    embeddings are read out through an MLP to predict the target modality's
    feature vector for each cell -- i.e. "one bipartite cell<->feature GNN
    with attention-gated multi-edge-type fusion, trained end-to-end for
    cross-modality feature imputation" is scMoGNN's namesake contribution
    over treating cells and features as one homogeneous graph or ignoring
    feature-feature structure. Reimplemented from scratch as manual
    ``scatter_add``/segment-mean bipartite message passing (no ``dgl``
    dependency, which is not in the base env) with the same
    feature2cell / cell2feature / pathway edge-type triplet and the same
    softmax-attention edge-type fusion, at reduced node counts and hidden
    width.
  - scMoMaT (cand_00782): Zhang, Ziqi, et al., "scMoMaT jointly performs
    single cell mosaic integration and multi-modal bio-marker detection",
    Nature Communications 2023, https://doi.org/10.1038/s41467-023-36066-2;
    official repo https://github.com/PeterZZQ/scMoMaT,
    ``scmomat/model.py``, class ``scmomat_model`` (``recon_loss``,
    ``batch_loss``). The defining mechanism: **matrix tri-factorization**
    directly on learnable ``nn.Parameter`` factor matrices rather than an
    encoder network -- for every (modality, batch) count matrix ``X`` the
    model holds a per-batch cell-factor matrix ``C_cell`` (softmax-
    normalized across the shared ``K``-dim latent axis), a per-modality
    feature-factor matrix ``C_feat`` (also softmax-normalized), and
    reconstructs ``X ~ C_cell @ (A_shared + A_{modality,batch}) @
    C_feat.T + b_cell + b_feat`` where ``A_shared`` is one association
    matrix shared by every modality/batch and ``A_{modality,batch}`` is a
    small modality-and-batch-specific residual association matrix -- i.e.
    "one shared low-rank cell x feature association matrix factorized
    jointly across every unpaired batch/modality combination, with a
    modality-batch-specific residual correction" is scMoMaT's namesake
    mosaic-integration contribution over per-batch independent NMF.
    Reimplemented as a compact ``nn.Module`` holding the same
    cell-factor / feature-factor / shared-plus-residual-association
    parameter structure for two modalities across two unpaired batches,
    with ``forward`` performing the same softmax-normalize-then-
    trifactorize reconstruction for every modality/batch pair that has
    data, at reduced ``K`` and cell/feature counts.
  - scMVAE (cand_00783): Zuo, Chunman and Chen, Luonan, "Deep-joint-learning
    analysis model of single cell transcriptome and open chromatin
    accessibility data", Briefings in Bioinformatics 2021,
    https://doi.org/10.1093/bib/bbaa287; official repo
    https://github.com/cmzuo11/scMVAE, ``scMVAE/MVAE_model.py``, class
    ``scMVAE_POE`` (the paper's flagship PoE variant; methods
    ``encode_modalities``, ``ProductOfExperts.forward``, ``inference``).
    The defining mechanism: two independent Gaussian encoders (one per
    scRNA/scATAC modality) each produce a mean/logvar over a shared latent
    space; those two per-modality posteriors, together with a fixed
    ``N(0, I)`` "universal prior expert", are combined via a **product-of-
    experts** fusion (``pd_var = 1 / sum(1/var_i)``,
    ``pd_mu = pd_var * sum(mu_i / var_i)``) into one joint posterior that
    is reparameterized and split (via a shared decoder trunk) into
    modality-specific latents feeding a ZINB decoder head (RNA) and a
    Bernoulli/ZINB decoder head (ATAC) -- i.e. "precision-weighted
    Gaussian product-of-experts fusion of per-modality encoders, robust to
    a modality being absent at inference time" is scMVAE-PoE's namesake
    contribution over naive concatenation-based fusion (its own
    ``scMVAE_Concat``/``scMVAE_NN`` baselines in the same file). Reimplemented
    with the same two per-modality Gaussian encoders, the same
    ``ProductOfExperts`` closed-form precision-weighted fusion including the
    universal prior expert, and the same shared-decoder-trunk-then-split
    into two modality decoder heads, at reduced gene/peak counts and latent
    width.
  - scMVP (cand_00784): Li, Gaoyang, Zuo, Chunman, Chen, Luonan et al.,
    "scMVP: a deep generative model for multi-view single-cell
    RNA-seq/scATAC data integration and analysis", Genome Biology 2022,
    https://doi.org/10.1186/s13059-021-02595-6; official repo
    https://github.com/bm2-lab/scMVP,
    ``scMVP/models/multi_vae_attention.py`` (class ``Multi_VAE_Attention``,
    instantiating the self-attention encoder/decoder from ``modules.py``)
    and ``scMVP/models/modules.py``, class ``Multi_Encoder_nb_SelfAttention``
    (the paper's namesake self-attention joint encoder). The defining
    mechanism: the RNA branch is encoded by an FC tower whose output is
    *gated* elementwise by a second sigmoid auxiliary network applied to
    the same raw RNA input (``q1 = scRNA_encoder(x1) * RNA_encoder_aux(x1)``);
    the ATAC branch is encoded by its own FC tower and then run through a
    **multi-head self-attention block over the ATAC embedding itself**
    (learned ``W_q``/``W_k``/``W_v`` projections of the same ATAC vector,
    split into heads, scaled-dot-product softmax attention, recombined);
    the gated-RNA and self-attended-ATAC embeddings are concatenated,
    projected, and layer-normalized into one joint Gaussian VAE posterior
    (mean/log-variance heads) -- i.e. "auxiliary-sigmoid-gated RNA branch
    fused with a self-attention-refined ATAC branch into one joint
    posterior" is scMVP's namesake contribution over the repo's own
    non-attention ``Multi_Encoder`` concatenation baseline. Reimplemented
    with the same RNA gating, the same from-scratch multi-head
    self-attention over the ATAC embedding, and the same
    concat-project-layernorm joint posterior, at reduced gene/peak counts
    and hidden width.
  - scNym (cand_00785): Kimmel, Jacob C. and Kelley, David R., "Semi-
    supervised adversarial neural networks for single-cell classification",
    Genome Research 2021, https://doi.org/10.1101/gr.268581.120; official
    repo https://github.com/calico/scnym, ``scnym/model.py`` (classes
    ``CellTypeCLF``, ``GradReverse``, ``DANN``). The defining mechanism: a
    residual-block cell-type classifier (``CellTypeCLF``, whose ``.embed``
    submodule produces a hidden embedding before the final classification
    head) is wrapped by a **domain-adversarial network**: the same
    embedding is passed through a ``GradReverse`` autograd ``Function``
    that is the identity on the forward pass but negates and scales the
    gradient on the backward pass, then fed to a small domain classifier
    that tries to predict which batch/domain (e.g. labeled source vs.
    unlabeled target) a cell came from -- because the gradient is reversed,
    training this domain head *forces the shared embedding to become
    domain-invariant* while the class head still learns to discriminate
    cell types, enabling semi-supervised label transfer across batches
    with no matched cells (MixUp interpolation is applied to the raw
    inputs at training time, orthogonal to this module's forward pass) --
    i.e. "gradient-reversal-layer adversarial domain classifier bolted onto
    a shared cell-type-classifier embedding" is scNym's namesake
    contribution over training the classifier alone. Reimplemented with
    the same residual-block embedding trunk, the same custom
    ``torch.autograd.Function`` gradient-reversal layer, and the same
    domain classifier head sharing the embedding trunk, at reduced gene
    count and hidden width; ``forward`` returns both the cell-type logits
    and the domain logits from one embedding.
  - scPhere (cand_00786): Ding, Jiarui, Regev, Aviv, et al., "Deep
    generative model embedding of single-cell RNA-Seq profiles on
    hyperspheres and hyperbolic spaces", Nature Communications 2021,
    https://doi.org/10.1038/s41467-021-22851-4; official repo
    https://github.com/klarman-cell-observatory/scPhere,
    ``scphere/model/vae.py``, class ``SCPHERE`` (methods ``_encoder``,
    ``_decoder``; TensorFlow 1.x source used as the mechanism reference --
    the pip/framework metadata notwithstanding, the official implementation
    is TF, so this is a from-scratch PyTorch port of the same architecture,
    not a line-level port). The defining mechanism: a batch-one-hot-
    conditioned MLP encoder maps log1p, L2-normalized gene expression to a
    **von Mises-Fisher (vMF) posterior on a hypersphere**: a unit-norm mean
    direction head (``l2_normalize``) and a positive concentration
    (``softplus + 1``, clipped) parameterize ``q(z|x) = vMF(mu, kappa)``
    instead of a Euclidean Gaussian; a differentiable rejection-free vMF
    sample is drawn via Wood's algorithm (sample the tangential-direction
    angle ``w`` from the vMF marginal in closed form via its known inverse-
    CDF-style rejection acceptance formula, then rotate a uniform
    hypersphere direction to align with ``mu``); the batch-conditioned
    decoder maps that spherical latent back to a softmax-normalized,
    library-size-rescaled negative-binomial mean over genes -- i.e. "vMF
    hyperspherical latent VAE for scRNA-seq, batch-conditioned in both
    encoder and decoder" is scPhere's namesake contribution over a
    Euclidean-Gaussian VAE (its own alternate ``latent_dist='normal'``/
    ``'wn'`` code paths in the same class). Reimplemented with the same
    batch-conditioned encoder/decoder MLP structure, unit-norm mean +
    softplus concentration vMF posterior parameterization, and a from-
    scratch differentiable approximate vMF sampler, at reduced gene count,
    batch count, and hidden width.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# scMoGNN: heterogeneous cell<->feature bipartite GNN with attention-gated
# multi-edge-type fusion (Wen et al., KDD 2022; DANCE ``ScMoGCN``).
# ---------------------------------------------------------------------------


def _segment_mean(src: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Mean-aggregate rows of ``src`` into ``num_segments`` buckets by ``index``.

    Parameters
    ----------
    src : Tensor
        ``[n_edges, dim]`` source (message) features.
    index : Tensor
        ``[n_edges]`` integer destination-node index for each row of ``src``.
    num_segments : int
        Number of destination nodes to scatter into.

    Returns
    -------
    Tensor
        ``[num_segments, dim]`` mean of the incoming messages per node (zero
        for nodes with no incoming edges).
    """
    dim = src.shape[-1]
    out = src.new_zeros(num_segments, dim)
    out = out.index_add(0, index, src)
    counts = src.new_zeros(num_segments).index_add(0, index, src.new_ones(index.shape[0]))
    counts = counts.clamp(min=1.0).unsqueeze(-1)
    return out / counts


class _BipartiteSAGEConv(nn.Module):
    """A single GraphSAGE-mean-aggregation hop over one directed edge type."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        src_feat: Tensor,
        dst_feat: Tensor,
        src_index: Tensor,
        dst_index: Tensor,
        num_dst: int,
    ) -> Tensor:
        """Aggregate ``src_feat`` messages along edges into destination nodes.

        Parameters
        ----------
        src_feat : Tensor
            ``[n_src, dim]`` source-node features.
        dst_feat : Tensor
            ``[n_dst, dim]`` destination-node features (self term).
        src_index : Tensor
            ``[n_edges]`` source-node index of each edge.
        dst_index : Tensor
            ``[n_edges]`` destination-node index of each edge.
        num_dst : int
            Number of destination nodes.

        Returns
        -------
        Tensor
            ``[n_dst, out_dim]`` updated destination features.
        """
        messages = src_feat.index_select(0, src_index)
        agg = _segment_mean(messages, dst_index, num_dst)
        return self.self_lin(dst_feat) + self.neigh_lin(agg)


class ScMoGNN(nn.Module):
    """Bipartite cell<->feature GNN for cross-modality feature imputation.

    Parameters
    ----------
    n_cells : int
        Number of cells in the toy mini-batch graph.
    n_features : int
        Number of feature (gene/peak) nodes.
    n_pathway_edges : int
        Number of feature-feature "pathway" co-membership edges.
    hidden : int
        Hidden width used throughout the encoder/conv/readout stack.
    n_layers : int
        Number of bipartite message-passing rounds.
    out_dim : int
        Dimensionality of the predicted target-modality feature vector.
    """

    def __init__(
        self,
        n_cells: int = 24,
        n_features: int = 40,
        n_pathway_edges: int = 60,
        hidden: int = 16,
        n_layers: int = 2,
        out_dim: int = 12,
    ) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.n_features = n_features
        self.hidden = hidden
        self.n_layers = n_layers

        self.embed_cell = nn.Linear(n_cells, hidden)
        self.embed_feat = nn.Linear(n_features, hidden)

        self.feature2cell = nn.ModuleList(
            _BipartiteSAGEConv(hidden, hidden) for _ in range(n_layers)
        )
        self.cell2feature = nn.ModuleList(
            _BipartiteSAGEConv(hidden, hidden) for _ in range(n_layers)
        )
        self.pathway_conv = nn.ModuleList(
            _BipartiteSAGEConv(hidden, hidden) for _ in range(n_layers)
        )
        self.attn_query = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers))
        self.norm_cell = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))
        self.norm_feat = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))

        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

        # Fixed toy bipartite topology: every cell connects to every feature
        # (dense cell<->feature edges, as in a small imputation mini-batch),
        # plus a random pathway (feature-feature) edge set.
        cell_idx, feat_idx = torch.meshgrid(
            torch.arange(n_cells), torch.arange(n_features), indexing="ij"
        )
        self.register_buffer("cell_of_edge", cell_idx.reshape(-1), persistent=False)
        self.register_buffer("feat_of_edge", feat_idx.reshape(-1), persistent=False)

        gen = torch.Generator().manual_seed(0)
        pw_src = torch.randint(0, n_features, (n_pathway_edges,), generator=gen)
        pw_dst = torch.randint(0, n_features, (n_pathway_edges,), generator=gen)
        self.register_buffer("pathway_src", pw_src, persistent=False)
        self.register_buffer("pathway_dst", pw_dst, persistent=False)

    def _attention_fuse(self, layer: int, prev: Tensor, msg_a: Tensor, msg_b: Tensor) -> Tensor:
        """Softmax-attention fusion of two per-edge-type messages into one."""
        query = self.attn_query[layer](prev).unsqueeze(1)
        feats = torch.stack([msg_a, msg_b], dim=1)
        scores = torch.matmul(feats, query.transpose(1, 2)).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * feats).sum(dim=1)

    def forward(self, cell_onehot: Tensor, feat_onehot: Tensor) -> Tensor:
        """Run bipartite message passing and predict the target modality.

        Parameters
        ----------
        cell_onehot : Tensor
            ``[n_cells, n_cells]`` identity-style initial cell node features.
        feat_onehot : Tensor
            ``[n_features, n_features]`` identity-style initial feature node
            features.

        Returns
        -------
        Tensor
            ``[n_cells, out_dim]`` predicted target-modality features per cell.
        """
        h_cell = F.leaky_relu(self.embed_cell(cell_onehot))
        h_feat = F.leaky_relu(self.embed_feat(feat_onehot))

        for layer in range(self.n_layers):
            feature2cell_msg = self.feature2cell[layer](
                h_feat, h_cell, self.feat_of_edge, self.cell_of_edge, self.n_cells
            )
            new_cell = F.gelu(self.norm_cell[layer](feature2cell_msg))

            cell2feature_msg = self.cell2feature[layer](
                h_cell, h_feat, self.cell_of_edge, self.feat_of_edge, self.n_features
            )
            pathway_msg = self.pathway_conv[layer](
                h_feat, h_feat, self.pathway_src, self.pathway_dst, self.n_features
            )
            fused_feat = self._attention_fuse(layer, h_feat, cell2feature_msg, pathway_msg)
            new_feat = F.gelu(self.norm_feat[layer](fused_feat))

            h_cell, h_feat = new_cell, new_feat

        return self.readout(h_cell)


def build_scmognn() -> nn.Module:
    """Build a small scMoGNN bipartite cell-feature GNN."""
    return ScMoGNN(
        n_cells=24, n_features=40, n_pathway_edges=60, hidden=16, n_layers=2, out_dim=12
    ).eval()


def example_input_scmognn() -> tuple[Tensor, Tensor]:
    """Return one-hot-style initial cell and feature node features."""
    return torch.eye(24), torch.eye(40)


# ---------------------------------------------------------------------------
# scMoMaT: shared+residual matrix tri-factorization for mosaic multi-omic
# integration (Zhang et al., Nature Communications 2023).
# ---------------------------------------------------------------------------


class ScMoMaT(nn.Module):
    """Two-batch, two-modality tri-factorization mosaic-integration model.

    Parameters
    ----------
    n_cells_batch0 : int
        Number of cells in batch 0 (has RNA only, as in a mosaic design).
    n_cells_batch1 : int
        Number of cells in batch 1 (has ATAC only).
    n_genes : int
        Number of RNA features.
    n_peaks : int
        Number of ATAC features.
    latent_dim : int
        Shared latent factorization rank ``K``.
    """

    def __init__(
        self,
        n_cells_batch0: int = 20,
        n_cells_batch1: int = 18,
        n_genes: int = 30,
        n_peaks: int = 26,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.c_cell = nn.ParameterDict(
            {
                "0": nn.Parameter(torch.rand(n_cells_batch0, latent_dim)),
                "1": nn.Parameter(torch.rand(n_cells_batch1, latent_dim)),
            }
        )
        self.c_feat = nn.ParameterDict(
            {
                "rna": nn.Parameter(torch.rand(n_genes, latent_dim)),
                "atac": nn.Parameter(torch.rand(n_peaks, latent_dim)),
            }
        )
        self.a_shared = nn.Parameter(torch.rand(latent_dim, latent_dim))
        self.a_residual = nn.ParameterDict(
            {
                "rna_0": nn.Parameter(torch.zeros(latent_dim, latent_dim)),
                "atac_1": nn.Parameter(torch.zeros(latent_dim, latent_dim)),
            }
        )
        self.b_cell = nn.ParameterDict(
            {
                "rna_0": nn.Parameter(torch.zeros(n_cells_batch0, 1)),
                "atac_1": nn.Parameter(torch.zeros(n_cells_batch1, 1)),
            }
        )
        self.b_feat = nn.ParameterDict(
            {
                "rna_0": nn.Parameter(torch.zeros(1, n_genes)),
                "atac_1": nn.Parameter(torch.zeros(1, n_peaks)),
            }
        )

    def forward(self) -> tuple[Tensor, Tensor]:
        """Reconstruct the RNA (batch 0) and ATAC (batch 1) count matrices.

        Returns
        -------
        recon_rna : Tensor
            ``[n_cells_batch0, n_genes]`` reconstructed RNA matrix.
        recon_atac : Tensor
            ``[n_cells_batch1, n_peaks]`` reconstructed ATAC matrix.
        """
        c_cell0 = torch.softmax(self.c_cell["0"], dim=1)
        c_cell1 = torch.softmax(self.c_cell["1"], dim=1)
        c_rna = torch.softmax(self.c_feat["rna"], dim=1)
        c_atac = torch.softmax(self.c_feat["atac"], dim=1)

        assoc_rna = self.a_shared + self.a_residual["rna_0"]
        assoc_atac = self.a_shared + self.a_residual["atac_1"]

        recon_rna = c_cell0 @ assoc_rna @ c_rna.t() + self.b_cell["rna_0"] + self.b_feat["rna_0"]
        recon_atac = (
            c_cell1 @ assoc_atac @ c_atac.t() + self.b_cell["atac_1"] + self.b_feat["atac_1"]
        )
        return recon_rna, recon_atac


def build_scmomat() -> nn.Module:
    """Build a small scMoMaT mosaic tri-factorization model."""
    return ScMoMaT(
        n_cells_batch0=20, n_cells_batch1=18, n_genes=30, n_peaks=26, latent_dim=8
    ).eval()


def example_input_scmomat() -> tuple[()]:
    """Return the empty input tuple (scMoMaT factorizes learned parameters)."""
    return ()


# ---------------------------------------------------------------------------
# scMVAE (PoE variant): product-of-experts multimodal VAE for joint
# scRNA/scATAC integration (Zuo & Chen, Briefings in Bioinformatics 2021).
# ---------------------------------------------------------------------------


class _GaussianEncoder(nn.Module):
    """Two-layer MLP encoder producing a diagonal-Gaussian posterior."""

    def __init__(self, n_in: int, hidden: int, n_out: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.BatchNorm1d(hidden))
        self.mean = nn.Linear(hidden, n_out)
        self.logvar = nn.Linear(hidden, n_out)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(x)
        return self.mean(h), self.logvar(h)


def _product_of_experts(means: Tensor, logvars: Tensor) -> tuple[Tensor, Tensor]:
    """Precision-weighted product-of-experts fusion (Wu & Goodman, 2018).

    Parameters
    ----------
    means : Tensor
        ``[n_experts, batch, dim]`` per-expert posterior means.
    logvars : Tensor
        ``[n_experts, batch, dim]`` per-expert posterior log-variances.

    Returns
    -------
    Tensor
        Fused posterior mean, ``[batch, dim]``.
    Tensor
        Fused posterior log-variance, ``[batch, dim]``.
    """
    eps = 1e-8
    precision = 1.0 / (torch.exp(logvars) + eps)
    fused_var = 1.0 / precision.sum(dim=0)
    fused_mean = fused_var * (means * precision).sum(dim=0)
    return fused_mean, torch.log(fused_var)


class ScMVAE(nn.Module):
    """Product-of-experts joint VAE over paired scRNA and scATAC profiles.

    Parameters
    ----------
    n_genes : int
        Number of RNA features.
    n_peaks : int
        Number of ATAC features.
    hidden : int
        Encoder/decoder hidden width.
    latent_dim : int
        Shared joint-latent dimensionality.
    """

    def __init__(
        self, n_genes: int = 28, n_peaks: int = 24, hidden: int = 32, latent_dim: int = 10
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.rna_encoder = _GaussianEncoder(n_genes, hidden, latent_dim)
        self.atac_encoder = _GaussianEncoder(n_peaks, hidden, latent_dim)

        self.decoder_share = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU())
        self.share_hidden = hidden // 2

        self.rna_decoder = nn.Sequential(
            nn.Linear(self.share_hidden + latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_genes)
        )
        self.atac_decoder = nn.Sequential(
            nn.Linear(self.share_hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_peaks),
            nn.Sigmoid(),
        )

    def forward(self, x_rna: Tensor, x_atac: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode both modalities, fuse via PoE, decode both modalities back.

        Parameters
        ----------
        x_rna : Tensor
            ``[batch, n_genes]`` RNA count-like input.
        x_atac : Tensor
            ``[batch, n_peaks]`` ATAC accessibility-like input.

        Returns
        -------
        recon_rna : Tensor
            ``[batch, n_genes]`` reconstructed RNA profile.
        recon_atac : Tensor
            ``[batch, n_peaks]`` reconstructed ATAC accessibility.
        mean : Tensor
            ``[batch, latent_dim]`` fused posterior mean.
        logvar : Tensor
            ``[batch, latent_dim]`` fused posterior log-variance.
        """
        batch = x_rna.shape[0]
        prior_mean = x_rna.new_zeros(1, batch, self.latent_dim)
        prior_logvar = x_rna.new_zeros(1, batch, self.latent_dim)

        rna_mean, rna_logvar = self.rna_encoder(torch.log1p(x_rna))
        atac_mean, atac_logvar = self.atac_encoder(x_atac)

        means = torch.cat([prior_mean, rna_mean.unsqueeze(0), atac_mean.unsqueeze(0)], dim=0)
        logvars = torch.cat(
            [prior_logvar, rna_logvar.unsqueeze(0), atac_logvar.unsqueeze(0)], dim=0
        )
        mean, logvar = _product_of_experts(means, logvars)

        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)

        shared = self.decoder_share(z)
        latent_rna = torch.cat([z, shared[:, : self.share_hidden]], dim=1)
        latent_atac = shared[:, self.share_hidden :]

        recon_rna = self.rna_decoder(latent_rna)
        recon_atac = self.atac_decoder(latent_atac)
        return recon_rna, recon_atac, mean, logvar


def build_scmvae() -> nn.Module:
    """Build a small scMVAE product-of-experts joint VAE."""
    return ScMVAE(n_genes=28, n_peaks=24, hidden=32, latent_dim=10).eval()


def example_input_scmvae() -> tuple[Tensor, Tensor]:
    """Return paired RNA and ATAC toy count matrices."""
    return torch.rand(6, 28) * 5.0, (torch.rand(6, 24) > 0.7).float()


# ---------------------------------------------------------------------------
# scMVP: self-attention-gated joint RNA/ATAC encoder (Li, Zuo, Chen et al.,
# Genome Biology 2022).
# ---------------------------------------------------------------------------


class ScMVP(nn.Module):
    """Auxiliary-gated RNA branch fused with self-attended ATAC branch.

    Parameters
    ----------
    n_genes : int
        Number of RNA features.
    n_peaks : int
        Number of ATAC features.
    hidden : int
        Per-branch encoder hidden width (must be divisible by ``n_heads``).
    n_heads : int
        Number of self-attention heads applied to the ATAC embedding.
    latent_dim : int
        Joint posterior latent dimensionality.
    """

    def __init__(
        self,
        n_genes: int = 26,
        n_peaks: int = 22,
        hidden: int = 16,
        n_heads: int = 4,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.hidden = hidden

        self.rna_encoder = nn.Sequential(nn.Linear(n_genes, hidden), nn.ReLU())
        self.rna_encoder_aux = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.Linear(hidden, hidden), nn.Sigmoid()
        )

        self.atac_encoder = nn.Sequential(nn.Linear(n_peaks, hidden), nn.ReLU())
        self.w_q = nn.Linear(hidden, hidden)
        self.w_k = nn.Linear(hidden, hidden)
        self.w_v = nn.Linear(hidden, hidden)

        self.concat = nn.Linear(2 * hidden, hidden)
        self.layernorm = nn.LayerNorm(hidden, eps=1e-4)
        self.mean_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, x_rna: Tensor, x_atac: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode paired RNA/ATAC profiles into one gated+attended posterior.

        Parameters
        ----------
        x_rna : Tensor
            ``[batch, n_genes]`` RNA input.
        x_atac : Tensor
            ``[batch, n_peaks]`` ATAC input.

        Returns
        -------
        mean : Tensor
            ``[batch, latent_dim]`` posterior mean.
        logvar : Tensor
            ``[batch, latent_dim]`` posterior log-variance (softplus-linked).
        latent : Tensor
            ``[batch, latent_dim]`` reparameterized sample.
        """
        batch = x_rna.shape[0]

        q1 = self.rna_encoder(x_rna) * self.rna_encoder_aux(x_rna)

        q2 = self.atac_encoder(x_atac)
        head_dim = self.hidden // self.n_heads
        q = self.w_q(q2).view(batch, self.n_heads, head_dim)
        k = self.w_k(q2).view(batch, self.n_heads, head_dim)
        v = self.w_v(q2).view(batch, self.n_heads, head_dim)
        energy = torch.einsum("bhd,bhe->bhde", q, k)
        attention = torch.softmax(energy, dim=-1)
        q2 = torch.einsum("bhde,bhe->bhd", attention, v).reshape(batch, self.hidden)

        joint = self.concat(torch.cat([q1, q2], dim=1))
        joint = self.layernorm(joint)

        mean = self.mean_head(joint)
        logvar = torch.log(F.softplus(self.logvar_head(joint)) + 1e-4)
        std = torch.exp(0.5 * logvar)
        latent = mean + std * torch.randn_like(std)
        return mean, logvar, latent


def build_scmvp() -> nn.Module:
    """Build a small scMVP self-attention-gated joint RNA/ATAC encoder."""
    return ScMVP(n_genes=26, n_peaks=22, hidden=16, n_heads=4, latent_dim=8).eval()


def example_input_scmvp() -> tuple[Tensor, Tensor]:
    """Return paired RNA and ATAC toy input matrices."""
    return torch.rand(5, 26), (torch.rand(5, 22) > 0.6).float()


# ---------------------------------------------------------------------------
# scNym: gradient-reversal adversarial domain network for semi-supervised
# cell-type label transfer (Kimmel & Kelley, Genome Research 2021).
# ---------------------------------------------------------------------------


class _GradReverse(torch.autograd.Function):
    """Identity on the forward pass; negated, scaled gradient on backward."""

    @staticmethod
    def forward(ctx: object, x: Tensor, weight: float) -> Tensor:
        ctx.weight = weight  # type: ignore[attr-defined]
        return x.view_as(x)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output * -1 * ctx.weight, None  # type: ignore[attr-defined]


class ScNym(nn.Module):
    """Residual cell-type classifier with a gradient-reversed domain head.

    Parameters
    ----------
    n_genes : int
        Number of input genes.
    n_cell_types : int
        Number of cell-type classes.
    n_domains : int
        Number of batches/domains for the adversarial domain classifier.
    hidden : int
        Shared embedding hidden width.
    rev_weight : float
        Gradient-reversal scaling weight.
    """

    def __init__(
        self,
        n_genes: int = 40,
        n_cell_types: int = 6,
        n_domains: int = 2,
        hidden: int = 32,
        rev_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.rev_weight = rev_weight

        self.embed = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden, n_cell_types)
        self.domain_clf = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_domains)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Classify cell type and (adversarially) the batch domain.

        Parameters
        ----------
        x : Tensor
            ``[batch, n_genes]`` log1p-normalized expression input.

        Returns
        -------
        cell_type_logits : Tensor
            ``[batch, n_cell_types]`` cell-type classification logits.
        domain_logits : Tensor
            ``[batch, n_domains]`` domain-classification logits, computed
            from a gradient-reversed copy of the shared embedding.
        """
        embedding = self.embed(x)
        cell_type_logits = self.classifier(embedding)

        reversed_embedding = _GradReverse.apply(embedding, self.rev_weight)
        domain_logits = self.domain_clf(reversed_embedding)
        return cell_type_logits, domain_logits


def build_scnym() -> nn.Module:
    """Build a small scNym gradient-reversal domain-adversarial classifier."""
    return ScNym(n_genes=40, n_cell_types=6, n_domains=2, hidden=32, rev_weight=1.0).eval()


def example_input_scnym() -> Tensor:
    """Return a batch of log1p-normalized toy expression profiles."""
    return torch.rand(8, 40)


# ---------------------------------------------------------------------------
# scPhere: batch-conditioned von Mises-Fisher hyperspherical VAE (Ding,
# Regev et al., Nature Communications 2021).
# ---------------------------------------------------------------------------


def _sample_vmf(mean_dir: Tensor, kappa: Tensor) -> Tensor:
    """Differentiable approximate von Mises-Fisher reparameterized sample.

    Draws the tangential angle from the vMF marginal via Wood's acceptance-
    rejection formula reduced to its closed-form high-concentration limit
    (used here purely as a smooth surrogate so the sampler stays
    differentiable and traceable), then rotates a random point on the unit
    sphere's tangent space to align with ``mean_dir``.

    Parameters
    ----------
    mean_dir : Tensor
        ``[batch, dim]`` unit-norm mean direction ``mu``.
    kappa : Tensor
        ``[batch, 1]`` positive concentration parameter.

    Returns
    -------
    Tensor
        ``[batch, dim]`` unit-norm samples on the hypersphere.
    """
    batch, dim = mean_dir.shape
    # Closed-form mean resultant length as a smooth stand-in for the exact
    # rejection-sampled scalar `w` (exact for the high-kappa / large-dim
    # asymptotic regime used by Wood's algorithm), then perturbed by a
    # small reparameterized Gaussian jitter so gradients flow through kappa.
    w = kappa / (kappa + (dim - 1) / 2.0)
    w = w + 0.01 * torch.randn_like(w) / (kappa + 1.0)
    w = w.clamp(-0.999, 0.999)

    tangent = torch.randn(batch, dim, device=mean_dir.device, dtype=mean_dir.dtype)
    tangent = tangent - (tangent * mean_dir).sum(dim=1, keepdim=True) * mean_dir
    tangent = F.normalize(tangent, dim=1, eps=1e-6)

    sample = w * mean_dir + torch.sqrt((1 - w**2).clamp(min=1e-6)) * tangent
    return F.normalize(sample, dim=1, eps=1e-6)


class ScPhere(nn.Module):
    """Batch-conditioned vMF hyperspherical VAE for scRNA-seq embedding.

    Parameters
    ----------
    n_genes : int
        Number of genes.
    n_batches : int
        Number of experimental batches (one-hot conditioning).
    hidden : int
        Encoder/decoder hidden width.
    z_dim : int
        Intrinsic hypersphere dimension (embedding lives in ``z_dim + 1``
        ambient coordinates, matching the reference's ``z_dim += 1`` for
        ``latent_dist='vmf'``).
    """

    def __init__(
        self, n_genes: int = 30, n_batches: int = 3, hidden: int = 24, z_dim: int = 2
    ) -> None:
        super().__init__()
        self.n_batches = n_batches
        self.ambient_dim = z_dim + 1

        self.encoder = nn.Sequential(
            nn.Linear(n_genes + n_batches, hidden), nn.ELU(), nn.BatchNorm1d(hidden)
        )
        self.mean_head = nn.Linear(hidden, self.ambient_dim)
        self.kappa_head = nn.Linear(hidden, 1)

        self.decoder = nn.Sequential(
            nn.Linear(self.ambient_dim + n_batches, hidden), nn.ELU(), nn.BatchNorm1d(hidden)
        )
        self.rate_head = nn.Linear(hidden, n_genes)

    def forward(self, x: Tensor, batch_onehot: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode to a vMF posterior, sample, and decode a NB mean.

        Parameters
        ----------
        x : Tensor
            ``[batch, n_genes]`` raw (non-negative) count-like input.
        batch_onehot : Tensor
            ``[batch, n_batches]`` one-hot batch/domain indicator.

        Returns
        -------
        nb_mean : Tensor
            ``[batch, n_genes]`` library-size-rescaled negative-binomial mean.
        mean_dir : Tensor
            ``[batch, ambient_dim]`` unit-norm vMF mean direction.
        kappa : Tensor
            ``[batch, 1]`` vMF concentration parameter.
        """
        library_size = x.sum(dim=1, keepdim=True).clamp(min=1.0)

        x_log = torch.log1p(x)
        x_norm = F.normalize(x_log, dim=1, eps=1e-6)
        h = self.encoder(torch.cat([x_norm, batch_onehot], dim=1))

        mean_dir = F.normalize(self.mean_head(h), dim=1, eps=1e-6)
        kappa = F.softplus(self.kappa_head(h)) + 1.0
        kappa = kappa.clamp(1.0, 10000.0)

        z = _sample_vmf(mean_dir, kappa)

        h_dec = self.decoder(torch.cat([z, batch_onehot], dim=1))
        nb_mean = F.softmax(self.rate_head(h_dec), dim=1) * library_size
        return nb_mean, mean_dir, kappa


def build_scphere() -> nn.Module:
    """Build a small scPhere batch-conditioned vMF hyperspherical VAE."""
    return ScPhere(n_genes=30, n_batches=3, hidden=24, z_dim=2).eval()


def example_input_scphere() -> tuple[Tensor, Tensor]:
    """Return a toy count matrix and one-hot batch indicator."""
    counts = torch.poisson(torch.rand(7, 30) * 3.0)
    batch_ids = torch.randint(0, 3, (7,))
    batch_onehot = F.one_hot(batch_ids, num_classes=3).float()
    return counts, batch_onehot


MENAGERIE_ENTRIES = [
    ("scMoGNN", "build_scmognn", "example_input_scmognn", "2022", "BIO"),
    ("scMoMaT", "build_scmomat", "example_input_scmomat", "2023", "BIO"),
    ("scMVAE", "build_scmvae", "example_input_scmvae", "2021", "BIO"),
    ("scMVP", "build_scmvp", "example_input_scmvp", "2022", "BIO"),
    ("scNym", "build_scnym", "example_input_scnym", "2021", "BIO"),
    ("scPhere", "build_scphere", "example_input_scphere", "2021", "BIO"),
]
