"""Single-cell genomics classics (batch w5a17).

Sources checked (paper + official repo source where available; no clone, no
pip install -- reimplemented from scratch in base-env torch):

- cell2location: Kleshchevnikov, Shmatko, Dann, Aivazidis, King et al.,
  Nature Biotechnology 2022, doi:10.1038/s41587-021-01139-4.
  https://github.com/BayraktarLab/cell2location
  (cell2location/models/_cell2location_module.py: class
  ``LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevel
  GeneAlphaPyroModel``). The distinguishing mechanism is a hierarchical
  Bayesian negative-binomial factorization of spatial gene expression: the
  per-location, per-gene expression rate is modelled as a *linear*
  combination of fixed reference cell-type expression signatures
  ``mu_{s,g} = m_g * sum_f w_{s,f} * g_{f,g} + s_{e,g}``, where the cell
  abundance weights ``w_{s,f}`` are further factorized through a second
  layer of ``R`` cellular-compartment/tissue-zone groups, and the whole
  observation is drawn from a Negative Binomial with a gene- and
  batch-specific overdispersion. Reimplemented deterministically (no Pyro
  dependency) as an amortized encoder that outputs the two-level
  cell-abundance factorization plus the NB rate/dispersion parameterization
  from fixed reference signatures, keeping the linear-decomposition +
  hierarchical-grouping + NB-rate structure that defines the model.

- CellLM: Zhao, Zhang, He, Xu, Fu, Nie & Ding, "Large-Scale Cell
  Representation Learning via Divide-and-Conquer Contrastive Learning",
  arXiv:2306.04371 (2023). https://ar5iv.labs.arxiv.org/html/2306.04371.
  The distinguishing mechanism is a >50M-parameter cell-language-model
  architecture combining (1) a gene-expression *binning* embedder
  ``phi_E`` and a gene-identity embedder ``phi_G`` fused additively
  ``c_k = phi_G(p_k) + phi_E(y_k)``, (2) a linear-attention (Performer
  -style) transformer encoder with a ``[CLS]`` token, and (3) three joint
  pretraining objectives: masked (binned) expression reconstruction (MLM),
  ``[CLS]``-token normal/tumor discrimination, and a
  divide-and-conquer InfoNCE contrastive loss between two independently
  -dropout perturbed views of the same cell. The GitHub repo referenced in
  the build queue (MeiHou0204/CellLM) hosts only a narrower decoder-only
  demo notebook that does not match the paper; the paper itself is the
  authoritative description used here. Reimplemented compactly: binned
  -expression + gene-id embedder, a small linear-attention transformer,
  and all three heads (MLM logits, CLS discrimination logit, contrastive
  projection of the dropout-twin forward pass).

- CellPLM: Wen, Tang, Bai, Zhao, Zhang, Hong, Xie, Walid & Fu, "CellPLM:
  Pre-training of Cell Language Model Beyond Single Cells", ICLR 2024,
  https://openreview.net/forum?id=BKXvPDekud.
  https://github.com/OmicsML/CellPLM (CellPLM/model/cellformer.py: class
  ``OmicsFormer``; CellPLM/encoder/transformer.py: ``TransformerEncoder``
  with ``model_type='performer'`` treating every cell in a batch as one
  attention token, i.e. inter-cell attention; CellPLM/latent/
  autoencoders.py: ``GMVAELatentLayer``/``InferenceNet``, ported from
  jariasf/GMVAE). The distinguishing mechanism is cells-as-tokens: gene
  expression per cell is embedded and every cell in a mini-batch attends
  to every other cell through a shared linear-attention transformer
  (batch/spatial context, not a single-cell-only encoder), and the
  resulting per-cell hidden state is passed through a Gaussian-Mixture VAE
  latent (Gumbel-Softmax categorical cluster assignment ``q(y|x)`` +
  cluster-conditional Gaussian ``q(z|x,y)``) rather than a plain Gaussian
  VAE. Reimplemented compactly: gene-expression embedder, one
  inter-cell self-attention transformer layer, and a GMVAE latent with
  Gumbel-Softmax cluster inference + reparameterized Gaussian sampling.

- CellVQ: (A4Bio), "Learning the fundamental language of single cells with
  a discrete vector-quantized codebook", Nature Communications 2026,
  doi:10.1038/s41467-026-70071-5. https://github.com/A4Bio/CellVQ
  (modules/vq_modules.py: class ``VQLayer`` -- cosine-distance codebook
  quantization with EMA codebook updates; model/pretrainmodels/model.py:
  class ``Model`` -- auto-discretization gene-expression binning embedder
  + positional + gene-ontology embeddings feeding a transformer encoder,
  a per-cell VQ layer, and a decoder head predicting per-gene Zero
  -Inflated Negative Binomial (ZINB) parameters ``mean``/``disp``/``pi``).
  The distinguishing mechanism is a *discrete cell codebook*: continuous
  per-cell transformer embeddings are L2-normalized and vector-quantized
  against a learned codebook (straight-through estimator, cosine/Euclidean
  distance on the unit sphere) before being decoded back into ZINB
  expression parameters, giving every cell state a discrete "cell word".
  Reimplemented compactly: binned-expression + positional embedder, a
  small transformer encoder, an L2-normalized VQ codebook bottleneck, and
  a ZINB decoder head.

- Chiron3D: (BoevaLab), "An interpretable deep learning framework for
  understanding the DNA code of chromatin looping", bioRxiv 2026,
  doi:10.64898/2026.03.20.713211. https://github.com/BoevaLab/Chiron3D
  (src/models/model/chiron_model.py: class ``Chiron3D``; src/models/model/
  blocks.py: ``AttnModuleSmall``, ``Decoder``, ``ResBlockDilated``). The
  distinguishing mechanism is a frozen-sequence-backbone-to-3D-contact-map
  pipeline: (1) a long-range genomic-sequence backbone (Borzoi in the
  official code, frozen) produces per-position embeddings; (2) a 1x1 conv
  projects and an adaptive pool downsamples them to a fixed track length;
  (3) a small Transformer (``AttnModuleSmall``) mixes information along
  the track; (4) the 1D track is *diagonalized* into a 2D map by
  broadcasting position ``i`` and position ``j`` features and
  concatenating them along the channel axis (an outer-product-style
  pairwise feature map, ``diagonalize_small``); (5) a dilated 2D residual
  -conv decoder (``Decoder``/``ResBlockDilated``, exponentially growing
  dilation) regresses a symmetric Hi-C-style contact map. Since Borzoi
  needs external pretrained weights and is not available in the base
  environment, the frozen sequence backbone is stood in with a small
  random-init 1D convolutional stack over one-hot DNA (architecturally
  equivalent role: sequence -> per-position embedding track), preserving
  every distinctive downstream mechanism (projector, pooling, attention,
  pairwise diagonalization, dilated 2D decoder).

- ChromatinHD: Wouters, Kalkan, Hulselmans, Spanier, Poovathingal &
  Aerts/Deplancke labs, "ChromatinHD: scalable and interpretable modeling
  of the chromatin accessibility landscape", bioRxiv 2023,
  doi:10.1101/2023.07.21.549899.
  https://github.com/DeplanckeLab/ChromatinHD
  (src/chromatinhd/models/diff/model/cutnf.py: class ``Model`` and
  ``Decoder``; src/chromatinhd/models/diff/model/spline.py: classes
  ``QuadraticSplineTransform``, ``DifferentialQuadraticSplineStack``,
  ``TransformedDistribution``). The distinguishing mechanism is a
  cluster-conditional *normalizing flow over fragment cut-site position*:
  raw ATAC fragment cut sites within a genomic region are rescaled to
  ``[0, 1]`` and modelled as the pushforward of a uniform base density
  through a stack of monotonic piecewise-quadratic splines; the spline bin
  -width/height deltas at each stack level are produced by a small decoder
  network conditioned on the cell's cluster/state embedding, so different
  clusters warp the base density differently and the log-probability of an
  observed cut site is the flow's forward log-abs-det-Jacobian. Reimplemented
  compactly with a single-level monotonic cumulative-softmax spline
  (`n_bins` learned bin masses, warped by a small cluster-conditioned MLP)
  that keeps the defining "cluster-conditioned flow density over cut-site
  position, forward transform + log-abs-det Jacobian" mechanism, at a tiny
  region length / bin count.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# cell2location -- hierarchical linear NB factorization of spatial expression
# ---------------------------------------------------------------------------


class Cell2Location(nn.Module):
    """cell2location: two-level linear factorization + Negative-Binomial rate.

    Compact deterministic stand-in for the Pyro hierarchical Bayesian model:
    an amortized encoder maps each spatial location's observed expression to
    (1) group-level abundances over ``n_groups`` cellular
    compartments/tissue zones and (2) a within-group factor mixing matrix,
    whose product gives the cell-type abundance weights ``w_{s,f}``. These
    weights linearly combine fixed reference cell-type signatures into a
    location- and gene-specific Negative-Binomial rate, with a learned
    per-gene platform-effect scale, per-location detection efficiency, and
    additive background term.

    Parameters
    ----------
    n_genes:
        Number of genes.
    n_factors:
        Number of reference cell-type signatures.
    n_groups:
        Number of cellular-compartment/tissue-zone groups (default 8,
        vs. 50 in the paper's full-scale setting).
    """

    def __init__(self, n_genes: int = 40, n_factors: int = 10, n_groups: int = 8) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.n_groups = n_groups

        self.register_buffer("reference_signatures", F.softplus(torch.randn(n_factors, n_genes)))

        self.group_encoder = nn.Sequential(nn.Linear(n_genes, n_groups), nn.Softplus())
        self.factor_per_group = nn.Parameter(torch.randn(n_groups, n_factors) * 0.1)

        self.gene_scale_m_g = nn.Parameter(torch.ones(n_genes))
        self.gene_background_s_eg = nn.Parameter(torch.zeros(n_genes) - 3.0)
        self.detection_encoder = nn.Sequential(nn.Linear(n_genes, 1), nn.Softplus())
        self.log_alpha_eg = nn.Parameter(torch.zeros(n_genes))

    def forward(self, spatial_expression: Tensor) -> tuple[Tensor, Tensor]:
        """Predict the Negative-Binomial rate and dispersion per location.

        Parameters
        ----------
        spatial_expression:
            Observed spot-by-gene counts, shape ``(n_locations, n_genes)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Rate ``mu`` and dispersion ``alpha``, each shape
            ``(n_locations, n_genes)``.
        """

        group_abundance = self.group_encoder(spatial_expression)  # (S, n_groups)
        factor_mix = F.softmax(self.factor_per_group, dim=-1)  # (n_groups, n_factors)
        w_sf = group_abundance @ factor_mix  # (S, n_factors)

        signal = w_sf @ self.reference_signatures  # (S, n_genes)
        y_s = self.detection_encoder(spatial_expression)  # (S, 1)
        background = F.softplus(self.gene_background_s_eg).unsqueeze(0)

        mu = (self.gene_scale_m_g.unsqueeze(0) * signal + background) * y_s
        alpha = F.softplus(self.log_alpha_eg).unsqueeze(0).expand_as(mu)
        return mu, alpha


def build_cell2location() -> nn.Module:
    """Build a compact cell2location hierarchical NB factorization model.

    Returns
    -------
    nn.Module
        Random-initialized ``Cell2Location`` in eval mode.
    """

    return Cell2Location().eval()


def example_input_cell2location() -> Tensor:
    """Create example spatial spot-by-gene expression counts.

    Returns
    -------
    torch.Tensor
        Shape ``(12, 40)``.
    """

    return torch.poisson(torch.rand(12, 40) * 5.0)


# ---------------------------------------------------------------------------
# CellLM -- binned-expression + gene-id transformer with MLM/CLS/contrastive
# heads (divide-and-conquer contrastive learning)
# ---------------------------------------------------------------------------


class _LinearAttention(nn.Module):
    """Performer-style linear attention via an elu(x)+1 feature map."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    @staticmethod
    def _feature_map(x: Tensor) -> Tensor:
        """Positive random-feature stand-in: elu(x) + 1."""

        return F.elu(x) + 1.0

    def forward(self, x: Tensor) -> Tensor:
        """Run linear (Performer-style) self-attention.

        Parameters
        ----------
        x:
            Input sequence, shape ``(batch, seq, dim)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """

        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (b, heads, t, head_dim)

        q = self._feature_map(q)
        k = self._feature_map(k)

        kv = torch.einsum("bhtd,bhte->bhde", k, v)
        z = 1.0 / (torch.einsum("bhtd,bhd->bht", q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhtd,bhde,bht->bhte", q, kv, z)
        out = out.transpose(1, 2).reshape(b, t, d)
        return self.out(out)


class _CellLMBlock(nn.Module):
    """Pre-norm transformer block: linear attention + MLP."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _LinearAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Run one transformer block."""

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CellLM(nn.Module):
    """CellLM: binned-expression cell-language model with 3 pretraining heads.

    Fuses a gene-identity embedding and a binned gene-expression embedding
    additively, encodes the resulting sequence (with a prepended ``[CLS]``
    token) through a small linear-attention transformer, and exposes the
    three paper's pretraining heads: masked-expression-bin logits, a
    ``[CLS]``-token tumor/normal discrimination logit, and an L2
    -normalized contrastive projection (evaluated twice under independent
    dropout to realize the divide-and-conquer contrastive positive pair).

    Parameters
    ----------
    n_genes:
        Number of gene identities.
    n_bins:
        Number of expression-level bins.
    dim:
        Hidden dimension.
    num_layers:
        Number of linear-attention transformer blocks.
    num_heads:
        Number of attention heads.
    """

    def __init__(
        self,
        n_genes: int = 64,
        n_bins: int = 8,
        dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.gene_embed = nn.Embedding(n_genes, dim)  # phi_G
        self.expr_embed = nn.Embedding(n_bins, dim)  # phi_E
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([_CellLMBlock(dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)

        self.mlm_head = nn.Linear(dim, n_bins)
        self.cls_head = nn.Linear(dim, 1)
        self.contrastive_proj = nn.Linear(dim, dim)

    def _encode(self, gene_ids: Tensor, expr_bins: Tensor) -> Tensor:
        c = self.gene_embed(gene_ids) + self.expr_embed(expr_bins)
        c = self.dropout(c)
        cls = self.cls_token.expand(c.shape[0], -1, -1)
        x = torch.cat([cls, c], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, gene_ids: Tensor, expr_bins: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run CellLM pretraining forward pass (all three objectives).

        Parameters
        ----------
        gene_ids:
            Gene identity ids, shape ``(batch, n_genes)``.
        expr_bins:
            Binned expression levels, shape ``(batch, n_genes)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            MLM bin logits ``(batch, n_genes, n_bins)``, CLS discrimination
            logit ``(batch, 1)``, and two independently-dropout contrastive
            projections ``(batch, dim)`` each (positive pair for the
            divide-and-conquer contrastive loss).
        """

        hidden = self._encode(gene_ids, expr_bins)
        cls_hidden, gene_hidden = hidden[:, 0], hidden[:, 1:]

        mlm_logits = self.mlm_head(gene_hidden)
        cls_logit = self.cls_head(cls_hidden)

        hidden_view2 = self._encode(gene_ids, expr_bins)
        proj1 = F.normalize(self.contrastive_proj(hidden[:, 0]), dim=-1)
        proj2 = F.normalize(self.contrastive_proj(hidden_view2[:, 0]), dim=-1)
        return mlm_logits, cls_logit, proj1, proj2


def build_celllm() -> nn.Module:
    """Build a compact CellLM binned-expression cell-language model.

    Returns
    -------
    nn.Module
        Random-initialized ``CellLM`` in train mode (dropout must be active
        for the two contrastive views to differ, matching the paper's
        divide-and-conquer contrastive positive-pair construction).
    """

    return CellLM().train()


def example_input_celllm() -> tuple[Tensor, Tensor]:
    """Create example gene-id / expression-bin sequences for CellLM.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Gene ids and expression bins, each shape ``(2, 64)``.
    """

    gene_ids = torch.arange(64).unsqueeze(0).expand(2, -1).clone()
    expr_bins = torch.randint(0, 8, (2, 64))
    return gene_ids, expr_bins


# ---------------------------------------------------------------------------
# CellPLM -- cells-as-tokens inter-cell attention + Gaussian-mixture VAE
# ---------------------------------------------------------------------------


class _InterCellAttention(nn.Module):
    """Self-attention over the *batch* dimension (every cell attends to
    every other cell in the mini-batch, not to its own genes)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, cell_embeddings: Tensor) -> Tensor:
        """Cross-cell self-attention.

        Parameters
        ----------
        cell_embeddings:
            Per-cell embeddings, shape ``(1, n_cells, dim)`` (cells form the
            attention sequence).

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """

        out, _ = self.attn(cell_embeddings, cell_embeddings, cell_embeddings)
        return out


class _GumbelSoftmax(nn.Module):
    """Categorical cluster-assignment head via the Gumbel-Softmax trick."""

    def __init__(self, dim: int, num_clusters: int) -> None:
        super().__init__()
        self.logits = nn.Linear(dim, num_clusters)

    def forward(self, x: Tensor, temperature: float = 1.0) -> Tensor:
        """Sample a (soft) one-hot cluster assignment.

        Parameters
        ----------
        x:
            Input features, shape ``(n_cells, dim)``.
        temperature:
            Gumbel-Softmax temperature.

        Returns
        -------
        torch.Tensor
            Soft cluster assignment, shape ``(n_cells, num_clusters)``.
        """

        logits = self.logits(x)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        return F.softmax((logits + gumbel_noise) / temperature, dim=-1)


class CellPLM(nn.Module):
    """CellPLM: inter-cell attention transformer + Gaussian-mixture VAE latent.

    Every cell's gene-expression embedding attends to every other cell in
    the batch (cells-as-tokens), then the fused per-cell representation is
    passed through a Gaussian-mixture VAE latent: a Gumbel-Softmax head
    infers a soft cluster assignment ``q(y|x)``, and a cluster-conditional
    Gaussian head infers ``q(z|x,y)`` via the reparameterization trick.

    Parameters
    ----------
    n_genes:
        Number of genes in the (small) input panel.
    dim:
        Hidden / encoder dimension.
    latent_dim:
        GMVAE latent dimension.
    num_clusters:
        Number of Gaussian-mixture components.
    """

    def __init__(
        self, n_genes: int = 48, dim: int = 32, latent_dim: int = 16, num_clusters: int = 6
    ) -> None:
        super().__init__()
        self.embedder = nn.Sequential(nn.Linear(n_genes, dim), nn.LayerNorm(dim), nn.GELU())
        self.inter_cell_attn = _InterCellAttention(dim, num_heads=4)
        self.norm = nn.LayerNorm(dim)

        self.qyx = _GumbelSoftmax(dim, num_clusters)
        self.qzxy = nn.Sequential(nn.Linear(dim + num_clusters, dim), nn.ReLU())
        self.z_mu = nn.Linear(dim, latent_dim)
        self.z_var = nn.Linear(dim, latent_dim)

        self.decoder = nn.Linear(latent_dim, n_genes)

    def forward(self, expression: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a batch of cells (inter-cell attention) and reconstruct.

        Parameters
        ----------
        expression:
            Per-cell gene-expression vectors, shape ``(n_cells, n_genes)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Reconstructed expression ``(n_cells, n_genes)``, soft cluster
            assignment ``(n_cells, num_clusters)``, and latent sample
            ``(n_cells, latent_dim)``.
        """

        h = self.embedder(expression).unsqueeze(0)  # (1, n_cells, dim)
        h = self.norm(h + self.inter_cell_attn(h)).squeeze(0)  # (n_cells, dim)

        y = self.qyx(h)
        zy_input = torch.cat([h, y], dim=-1)
        zy_hidden = self.qzxy(zy_input)
        mu = self.z_mu(zy_hidden)
        var = F.softplus(self.z_var(zy_hidden))
        z = mu + torch.randn_like(mu) * torch.sqrt(var + 1e-10)

        recon = self.decoder(z)
        return recon, y, z


def build_cellplm() -> nn.Module:
    """Build a compact CellPLM inter-cell-attention GMVAE model.

    Returns
    -------
    nn.Module
        Random-initialized ``CellPLM`` in eval mode.
    """

    return CellPLM().eval()


def example_input_cellplm() -> Tensor:
    """Create an example mini-batch of cell expression vectors.

    Returns
    -------
    torch.Tensor
        Shape ``(16, 48)``.
    """

    return torch.randn(16, 48)


# ---------------------------------------------------------------------------
# CellVQ -- transformer encoder + L2-normalized VQ codebook + ZINB decoder
# ---------------------------------------------------------------------------


class _CosineVQLayer(nn.Module):
    """Straight-through VQ codebook on the unit hypersphere (CellVQ ``VQLayer``)."""

    def __init__(self, dim: int, vq_dim: int, num_embeddings: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, dim)
        self.codebook = nn.Embedding(num_embeddings, vq_dim)

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize per-cell features against the learned codebook.

        Parameters
        ----------
        h:
            Per-cell features, shape ``(n_cells, dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Quantized (straight-through) features ``(n_cells, dim)`` and the
            selected discrete code indices ``(n_cells,)``.
        """

        z = F.normalize(self.proj(h), dim=-1)
        codebook = F.normalize(self.codebook.weight, dim=-1)
        distances = torch.cdist(z, codebook)
        code = distances.argmin(dim=-1)
        quantized = codebook[code]
        quantized_st = z + (quantized - z).detach()
        return self.proj_inv(quantized_st), code


class CellVQ(nn.Module):
    """CellVQ: transformer encoder -> discrete cell codebook -> ZINB decoder.

    Binned gene expression is embedded with gene-identity and positional
    embeddings, encoded by a small transformer, bottlenecked through an
    L2-normalized vector-quantized codebook (giving every cell a discrete
    "cell word"), and decoded into per-gene Zero-Inflated Negative-Binomial
    parameters.

    Parameters
    ----------
    n_genes:
        Number of genes in the (small) input panel.
    n_bins:
        Number of expression-level bins.
    dim:
        Transformer hidden dimension.
    vq_dim:
        Codebook embedding dimension.
    num_embeddings:
        Codebook size.
    """

    def __init__(
        self,
        n_genes: int = 48,
        n_bins: int = 8,
        dim: int = 32,
        vq_dim: int = 16,
        num_embeddings: int = 64,
    ) -> None:
        super().__init__()
        self.gene_embed = nn.Embedding(n_genes, dim)
        self.expr_embed = nn.Embedding(n_bins, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.vq = _CosineVQLayer(dim, vq_dim, num_embeddings)

        self.mean_head = nn.Linear(dim, n_genes)
        self.disp_head = nn.Linear(dim, n_genes)
        self.pi_head = nn.Linear(dim, n_genes)

    def forward(self, gene_ids: Tensor, expr_bins: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode, quantize, then decode ZINB parameters.

        Parameters
        ----------
        gene_ids:
            Gene identity ids, shape ``(batch, n_genes)``.
        expr_bins:
            Binned expression levels, shape ``(batch, n_genes)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ZINB mean, dispersion, and dropout-logit each shape
            ``(batch, n_genes)``, plus the discrete VQ code indices
            ``(batch,)``.
        """

        x = self.gene_embed(gene_ids) + self.expr_embed(expr_bins)
        hidden = self.encoder(x)
        cell_repr = hidden.mean(dim=1)  # pool genes -> one code per cell

        quantized, code = self.vq(cell_repr)

        mean = F.softplus(self.mean_head(quantized))
        disp = F.softplus(self.disp_head(quantized))
        pi = self.pi_head(quantized)
        return mean, disp, pi, code


def build_cellvq() -> nn.Module:
    """Build a compact CellVQ discrete-codebook cell model.

    Returns
    -------
    nn.Module
        Random-initialized ``CellVQ`` in eval mode.
    """

    return CellVQ().eval()


def example_input_cellvq() -> tuple[Tensor, Tensor]:
    """Create example gene-id / expression-bin sequences for CellVQ.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Gene ids and expression bins, each shape ``(3, 48)``.
    """

    gene_ids = torch.arange(48).unsqueeze(0).expand(3, -1).clone()
    expr_bins = torch.randint(0, 8, (3, 48))
    return gene_ids, expr_bins


# ---------------------------------------------------------------------------
# Chiron3D -- sequence backbone -> pairwise diagonalization -> dilated 2D
# residual decoder (3D chromatin contact map from DNA sequence)
# ---------------------------------------------------------------------------


class _SequenceBackbone(nn.Module):
    """Compact stand-in for the frozen Borzoi backbone: a 1D conv stack
    mapping one-hot DNA to a per-position embedding track."""

    def __init__(self, out_channels: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, out_channels, kernel_size=7, padding=3),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, one_hot_seq: Tensor) -> Tensor:
        """Encode one-hot DNA into a per-position embedding track.

        Parameters
        ----------
        one_hot_seq:
            Shape ``(batch, 4, length)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, out_channels, length // 2)``.
        """

        return self.net(one_hot_seq)


class _DilatedResBlock2d(nn.Module):
    """Exponentially-dilated 2D residual block (Chiron3D ``ResBlockDilated``)."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Run one dilated residual block."""

        return self.relu(x + self.res(x))


class Chiron3D(nn.Module):
    """Chiron3D: sequence backbone + pairwise diagonalization + 2D decoder.

    A (stand-in) sequence backbone produces a per-position embedding track,
    which is projected, pooled to a fixed length, mixed by a small
    self-attention transformer, then *diagonalized* into a 2D pairwise map
    by broadcasting position ``i`` and ``j`` features and concatenating
    along channels; a dilated 2D residual-conv decoder regresses a
    symmetric Hi-C-style contact map from that pairwise map.

    Parameters
    ----------
    mid_hidden:
        Feature width after the projector / through the pairwise map.
    map_len:
        Fixed track length used for the pairwise diagonalization (bins).
    num_dilated_blocks:
        Number of dilated residual blocks in the 2D decoder.
    """

    def __init__(
        self, mid_hidden: int = 16, map_len: int = 12, num_dilated_blocks: int = 2
    ) -> None:
        super().__init__()
        self.map_len = map_len
        self.backbone = _SequenceBackbone(out_channels=48)
        self.projector = nn.Conv1d(48, mid_hidden, kernel_size=1)
        self.length_reducer = nn.AdaptiveAvgPool1d(map_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=mid_hidden, nhead=4, dim_feedforward=mid_hidden * 2, batch_first=True
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=1)

        decoder_hidden = mid_hidden * 2
        self.decoder_start = nn.Sequential(
            nn.Conv2d(mid_hidden * 2, decoder_hidden, 3, padding=1),
            nn.BatchNorm2d(decoder_hidden),
            nn.ReLU(),
        )
        self.decoder_blocks = nn.Sequential(
            *[
                _DilatedResBlock2d(decoder_hidden, dilation=2 ** (i + 1))
                for i in range(num_dilated_blocks)
            ]
        )
        self.decoder_out = nn.Conv2d(decoder_hidden, 1, kernel_size=1)

    @staticmethod
    def _diagonalize(x: Tensor) -> Tensor:
        """Broadcast a 1D track into a pairwise 2D feature map.

        Parameters
        ----------
        x:
            Shape ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 2 * channels, length, length)``.
        """

        length = x.shape[-1]
        x_i = x.unsqueeze(2).expand(-1, -1, length, -1)
        x_j = x.unsqueeze(3).expand(-1, -1, -1, length)
        return torch.cat([x_i, x_j], dim=1)

    def forward(self, one_hot_seq: Tensor) -> Tensor:
        """Predict a symmetric contact map from a one-hot DNA window.

        Parameters
        ----------
        one_hot_seq:
            Shape ``(batch, 4, length)``.

        Returns
        -------
        torch.Tensor
            Predicted contact map, shape ``(batch, map_len, map_len)``.
        """

        track = self.backbone(one_hot_seq)
        track = self.projector(track)
        track = self.length_reducer(track)

        track = track.transpose(1, 2)  # (batch, map_len, mid_hidden)
        track = self.attn(track)
        track = track.transpose(1, 2)  # (batch, mid_hidden, map_len)

        pair_map = self._diagonalize(track)
        h = self.decoder_start(pair_map)
        h = self.decoder_blocks(h)
        return self.decoder_out(h).squeeze(1)


def build_chiron3d() -> nn.Module:
    """Build a compact Chiron3D sequence-to-contact-map model.

    Returns
    -------
    nn.Module
        Random-initialized ``Chiron3D`` in eval mode.
    """

    return Chiron3D().eval()


def example_input_chiron3d() -> Tensor:
    """Create an example one-hot DNA window for Chiron3D.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 4, 96)``.
    """

    idx = torch.randint(0, 4, (1, 96))
    return F.one_hot(idx, num_classes=4).permute(0, 2, 1).float()


# ---------------------------------------------------------------------------
# ChromatinHD -- cluster-conditioned normalizing flow over fragment cut-site
# position (monotonic cumulative-softmax spline)
# ---------------------------------------------------------------------------


class ChromatinHD(nn.Module):
    """ChromatinHD: cluster-conditional flow density over ATAC cut-site position.

    A small decoder maps a one-hot cluster/state vector to unnormalized bin
    logits; ``softmax`` turns those into bin probability masses, whose
    cumulative sum defines a monotonic, piecewise-linear CDF over
    ``[0, 1]`` (the base measure for cut-site position within a genomic
    region). The forward transform maps a normalized cut-site position to
    its image under that per-cluster CDF and returns the flow's
    log-probability (log of the local bin density) -- the defining
    "cluster-conditioned normalizing flow over fragment position" mechanism
    of the paper's ``DifferentialQuadraticSplineStack`` /
    ``TransformedDistribution``, simplified to a single monotonic spline
    level for compactness.

    Parameters
    ----------
    n_clusters:
        Number of cell clusters/states.
    n_bins:
        Number of spline bins partitioning ``[0, 1]``.
    hidden:
        Hidden width of the cluster-conditioning decoder.
    """

    def __init__(self, n_clusters: int = 5, n_bins: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.decoder = nn.Sequential(
            nn.Linear(n_clusters, hidden), nn.ReLU(), nn.Linear(hidden, n_bins)
        )

    def forward(self, cut_site_position: Tensor, cluster_onehot: Tensor) -> Tensor:
        """Compute the flow log-probability of observed cut-site positions.

        Parameters
        ----------
        cut_site_position:
            Cut-site positions normalized to ``[0, 1]``, shape
            ``(n_fragments,)``.
        cluster_onehot:
            One-hot cluster/state assignment per fragment, shape
            ``(n_fragments, n_clusters)``.

        Returns
        -------
        torch.Tensor
            Flow log-probability per fragment, shape ``(n_fragments,)``.
        """

        bin_logits = self.decoder(cluster_onehot)  # (n_fragments, n_bins)
        bin_probs = F.softmax(bin_logits, dim=-1)
        bin_density = bin_probs * self.n_bins  # density under a uniform-width partition

        bin_idx = (cut_site_position.clamp(0.0, 1.0 - 1e-6) * self.n_bins).long()
        local_density = bin_density.gather(1, bin_idx.unsqueeze(-1)).squeeze(-1)
        return torch.log(local_density + 1e-8)


def build_chromatinhd() -> nn.Module:
    """Build a compact ChromatinHD cluster-conditioned cut-site flow model.

    Returns
    -------
    nn.Module
        Random-initialized ``ChromatinHD`` in eval mode.
    """

    return ChromatinHD().eval()


def example_input_chromatinhd() -> tuple[Tensor, Tensor]:
    """Create example fragment cut-site positions and cluster assignments.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Cut-site positions in ``[0, 1]`` shape ``(20,)`` and one-hot cluster
        assignments shape ``(20, 5)``.
    """

    cut_site_position = torch.rand(20)
    cluster_idx = torch.randint(0, 5, (20,))
    cluster_onehot = F.one_hot(cluster_idx, num_classes=5).float()
    return cut_site_position, cluster_onehot


MENAGERIE_ENTRIES = [
    ("cell2location", "build_cell2location", "example_input_cell2location", "2022", "BIO"),
    ("CellLM", "build_celllm", "example_input_celllm", "2023", "BIO"),
    ("CellPLM", "build_cellplm", "example_input_cellplm", "2024", "BIO"),
    ("CellVQ", "build_cellvq", "example_input_cellvq", "2026", "BIO"),
    ("Chiron3D", "build_chiron3d", "example_input_chiron3d", "2026", "BIO"),
    ("ChromatinHD", "build_chromatinhd", "example_input_chromatinhd", "2023", "BIO"),
]
