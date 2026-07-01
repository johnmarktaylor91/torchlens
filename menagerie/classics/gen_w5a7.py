"""Menagerie batch w5a7: spatial-omics, pathway-visible, splicing, and DTI models.

Sources checked (reference only; no cloning, no pip installs):
  - MISTy (cand_00586): Tanevski, Ramirez Flores, Gabor, Schapiro & Saez-
    Rodriguez, Genome Biology 2022 "Explainable multiview framework for
    dissecting spatial relationships from highly multiplexed data", official
    R package https://github.com/saezlab/mistyR (``R/model.R``,
    ``R/model-functions.R``). MISTy's per-view sub-models are classical ML
    (random forest / xgboost / bagged MARS via ``ranger``/``xgboost``/
    ``earth``) -- not neural networks -- so there is no single "MISTy
    architecture" to port verbatim. Its distinctive, architecture-level idea
    (independent of the choice of per-view learner) is the **multi-view
    stacking topology**: for each target marker, every view (intraview,
    juxtaview/paraview spatial-context views, ...) is modeled independently
    from the *other* markers in that view, the per-view out-of-bag/held-out
    predictions are collected, and a final linear/ridge **meta-model**
    (``stats::lm`` / ``ridge::linearRidge`` in ``build_model()``) combines
    the per-view predictions into the final estimate for that marker. This
    module reproduces that topology faithfully as a differentiable stand-in:
    one small per-view MLP predictor per view (replacing the RF/xgboost/MARS
    view models, which are not differentiable and cannot be traced), whose
    scalar outputs are concatenated and passed through a learned linear
    combiner (replacing the ``lm``/``linearRidge`` meta-model), applied
    per-target-marker exactly as ``build_model()`` does.
  - MPNN-MD (NequIP / PaiNN) (cand_00587): both named implementations
    (NequIP, https://github.com/mir-group/nequip; PaiNN via schnetpack,
    https://github.com/atomistic-machine-learning/schnetpack) are already
    faithfully reimplemented and registered in this catalog under
    ``menagerie/classics/nequip.py`` (``build_nequip``) and
    ``menagerie/classics/reimpl3_10_atomistic.py`` /
    ``reimpl2_7_atomistic.py`` (``build_painn``) -- see the module-level
    grep confirming both names and their distinctive E(3)-equivariant
    tensor-product / scalar-vector message-passing mechanisms are already
    captured. This candidate is therefore SKIPPED as already_in_catalog
    rather than re-built as a third duplicate.
  - MPVNN / MPVNN cancer survival (cand_00588, cand_00589; the queue itself
    flags these as POTENTIAL_DEDUP -- same paper arXiv:2202.00882, same repo,
    built once): Ghosh Roy & collaborators, Bioinformatics 2022 "MPVNN:
    mutated pathway visible neural network architecture for interpretable
    cancer-specific survival risk prediction", official code
    https://github.com/gourabghoshroy/MPVNN (``Code/mpvnn.py``). The
    distinctive mechanism is ``CustomConnected``: a ``Dense`` layer whose
    kernel is elementwise-masked by a fixed pathway-topology connectivity
    matrix ``W`` (an identity matrix seeded with 1s at gene-gene edges taken
    from the PI3K-Akt KEGG pathway, so each hidden "gene" unit only receives
    input from its topological neighbors plus itself -- a sparse "visible"
    layer, not a generic dense layer), followed by a small dense output head
    that predicts a scalar survival risk score, originally trained with a
    concordance-index-bound Cox-style loss (``cibound_loss``). This module
    reproduces the masked-dense visible layer exactly (register the mask as
    a buffer and multiply it into the weight at every forward call) plus the
    dense output head, with a small synthetic gene-gene adjacency in place
    of the real PI3K-Akt topology.
  - MTSplice (cand_00590): Cheng, Celik, Kundaje & Gagneur, Genome Biology
    2021 (MMSplice) / bioRxiv 2020.06.07.138453 (MTSplice extension),
    official code https://github.com/gagneurlab/MMSplice_MTSplice
    (``mmsplice/layers.py``, ``mmsplice/mtsplice.py``, ``mmsplice/
    mmsplice.py``). MTSplice extends MMSplice's modular exon/intron/donor/
    acceptor DNA-sequence CNN scoring modules with a tissue-specific head:
    acceptor- and donor-side 1D-conv towers over one-hot DNA score each
    position, then the repo's own ``SplineWeight1D`` layer up/down-weights
    every position of the resulting activation map by ``1 + f_S(position)``,
    where ``f_S`` is a smooth position-dependent gain built from a fixed
    cubic B-spline basis (``get_X_spline``/``get_knots``) dotted with a
    learned per-channel spline-coefficient kernel -- i.e. a learned smooth
    positional-gain reweighting, not a generic attention or pooling op. The
    spline-weighted acceptor and donor activations are globally pooled,
    concatenated, and passed through a final dense layer that branches into
    one scalar delta-PSI prediction per GTEx tissue (56 tissues in the
    shipped model). This module reproduces the B-spline-basis positional
    gain (``SplineBasisGain``, using the identical open-uniform-knot cubic
    B-spline construction) applied to acceptor/donor conv towers, pooled and
    projected to a multi-tissue output head.
  - NeoDTI (cand_00591): Wan, Zeng et al., Bioinformatics 2019 "NeoDTI:
    neural integration of neighbor information from a heterogeneous network
    for discovering new drug-target interactions", official code
    https://github.com/FangpingWan/NeoDTI (``src/NeoDTI_cv.py``,
    ``src/utils.py``). The distinctive mechanism is a one-hop heterogeneous-
    network neighborhood-aggregation ("message passing") step per node type
    (drug / protein / disease / side-effect): for each node type, every
    incident relation's row-normalized adjacency matrix is multiplied by a
    *relation-specific* linear+ReLU projection (``a_layer``) of the
    neighboring node type's embedding, the resulting messages are summed,
    concatenated with the node's own embedding, projected through a single
    shared weight matrix ``W0``, ReLU'd and L2-normalized to obtain the new
    node representation. Edges are then reconstructed via ``bi_layer``: a
    low-rank bilinear form ``(x0 W0p)(x1 W1p)^T`` (shared ``W0p`` for
    symmetric relations such as drug-drug/protein-protein, separate
    ``W0p``/``W1p`` for asymmetric relations such as drug-disease), trained
    to match the observed heterogeneous-network adjacency matrices. This
    module reproduces the one-hop multi-relation aggregation plus bilinear
    reconstruction for a compact 4-node-type / 7-relation heterogeneous
    network (drug-drug, drug-disease, drug-sideeffect, drug-protein,
    protein-protein, protein-disease, and the drug-protein interaction
    matrix being reconstructed).
"""

from __future__ import annotations

import torch
from torch import nn


# ---------------------------------------------------------------------------
# MISTy: multi-view stacking with a linear meta-combiner.
# ---------------------------------------------------------------------------
class MistyViewPredictor(nn.Module):
    """Small MLP standing in for one MISTy per-view sub-model.

    MISTy's real per-view models are non-differentiable classical learners
    (random forest / xgboost / bagged MARS); this MLP is the differentiable
    stand-in used to preserve the *stacking topology* (per-view sub-model ->
    scalar prediction per target marker) so the overall MISTy architecture
    can be traced.
    """

    def __init__(self, n_markers: int, hidden: int = 16) -> None:
        """Initialize the per-view predictor.

        Parameters
        ----------
        n_markers : int
            Number of marker channels visible to this view.
        hidden : int
            Hidden width of the small MLP.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_markers, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, view_data: torch.Tensor) -> torch.Tensor:
        """Predict one scalar (this view's estimate of the target marker).

        Parameters
        ----------
        view_data : torch.Tensor
            Shape ``(n_cells, n_markers)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_cells, 1)``.
        """
        return self.net(view_data)


class MistyMultiView(nn.Module):
    """MISTy-style multi-view stacking model for one target marker.

    Each view (intraview + spatial-context views such as juxtaview and
    paraview) is modeled independently by its own :class:`MistyViewPredictor`
    over the *other* markers; the per-view scalar predictions are
    concatenated and combined by a single learned linear meta-model, exactly
    mirroring ``build_model()`` in mistyR's ``R/model.R``.
    """

    def __init__(self, n_views: int = 3, n_markers: int = 8) -> None:
        """Initialize the multi-view stacking model.

        Parameters
        ----------
        n_views : int
            Number of MISTy views (e.g. intraview, juxtaview, paraview).
        n_markers : int
            Number of marker channels per view.
        """
        super().__init__()
        self.view_models = nn.ModuleList([MistyViewPredictor(n_markers) for _ in range(n_views)])
        self.meta_model = nn.Linear(n_views, 1)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """Predict the target marker by stacking per-view predictions.

        Parameters
        ----------
        views : torch.Tensor
            Shape ``(n_views, n_cells, n_markers)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_cells, 1)`` final target-marker prediction.
        """
        per_view_preds = [model(views[i]) for i, model in enumerate(self.view_models)]
        stacked = torch.cat(per_view_preds, dim=1)
        return self.meta_model(stacked)


def build_misty() -> nn.Module:
    """Build a compact MISTy multi-view stacking model.

    Returns
    -------
    nn.Module
        MISTy reconstruction in evaluation mode.
    """
    return MistyMultiView(n_views=3, n_markers=8).eval()


def example_input_misty() -> torch.Tensor:
    """Create example input for :func:`build_misty`.

    Returns
    -------
    torch.Tensor
        Shape ``(3, 32, 8)``: 3 views, 32 cells, 8 markers each.
    """
    return torch.randn(3, 32, 8)


# ---------------------------------------------------------------------------
# MPVNN: pathway-topology masked "visible" dense layer + survival head.
# ---------------------------------------------------------------------------
class CustomConnected(nn.Module):
    """Dense layer masked by a fixed gene-gene connectivity matrix.

    Direct port of MPVNN's ``CustomConnected`` Keras layer: the kernel is
    elementwise-multiplied by a fixed binary ``connections`` mask (pathway
    topology + self-loops) before the matmul, so unit ``j`` only receives
    input from genes that are its topological neighbors (or itself).
    """

    def __init__(self, connections: torch.Tensor) -> None:
        """Initialize the pathway-masked visible layer.

        Parameters
        ----------
        connections : torch.Tensor
            Square binary connectivity matrix, shape ``(n_genes, n_genes)``.
        """
        super().__init__()
        n_genes = connections.shape[0]
        self.weight = nn.Parameter(torch.randn(n_genes, n_genes) * 0.05)
        self.bias = nn.Parameter(torch.zeros(n_genes))
        self.connections: torch.Tensor
        self.register_buffer("connections", connections)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pathway-masked dense transform.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_genes)`` standardized gene-expression input.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_genes)`` masked-dense activations.
        """
        masked_weight = self.weight * self.connections
        out = x @ masked_weight + self.bias
        return self.activation(out)


class MPVNN(nn.Module):
    """Mutated Pathway Visible Neural Network for survival risk prediction.

    ``CustomConnected`` visible layer (masked by PI3K-Akt pathway topology)
    followed by a scalar output head, exactly mirroring the two-layer
    ``Sequential`` model built in MPVNN's ``mpvnn.py``.
    """

    def __init__(self, connections: torch.Tensor) -> None:
        """Initialize MPVNN.

        Parameters
        ----------
        connections : torch.Tensor
            Square binary pathway connectivity matrix.
        """
        super().__init__()
        self.visible = CustomConnected(connections)
        n_genes = connections.shape[0]
        self.output_head = nn.Linear(n_genes, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a scalar survival risk score.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_genes)`` standardized gene-expression input.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` risk score.
        """
        hidden = self.visible(x)
        return self.output_activation(self.output_head(hidden))


def _pi3k_akt_toy_connectivity(n_genes: int = 24, seed: int = 0) -> torch.Tensor:
    """Build a small synthetic pathway-topology connectivity matrix.

    Stands in for the real PI3K-Akt KEGG edge list used by
    ``createW()`` in the official MPVNN repo: identity (self-loops) plus a
    handful of symmetric gene-gene edges.

    Parameters
    ----------
    n_genes : int
        Number of genes (nodes) in the toy pathway.
    seed : int
        RNG seed for reproducible edge sampling.

    Returns
    -------
    torch.Tensor
        Shape ``(n_genes, n_genes)`` symmetric binary connectivity matrix.
    """
    generator = torch.Generator().manual_seed(seed)
    connections = torch.eye(n_genes)
    n_edges = n_genes * 2
    src = torch.randint(0, n_genes, (n_edges,), generator=generator)
    dst = torch.randint(0, n_genes, (n_edges,), generator=generator)
    connections[src, dst] = 1.0
    connections[dst, src] = 1.0
    return connections


def build_mpvnn() -> nn.Module:
    """Build a compact MPVNN pathway-visible survival model.

    Returns
    -------
    nn.Module
        MPVNN reconstruction in evaluation mode.
    """
    connections = _pi3k_akt_toy_connectivity(n_genes=24)
    return MPVNN(connections).eval()


def example_input_mpvnn() -> torch.Tensor:
    """Create example input for :func:`build_mpvnn`.

    Returns
    -------
    torch.Tensor
        Shape ``(16, 24)``: 16 samples, 24 standardized gene-expression
        features.
    """
    return torch.randn(16, 24)


# ---------------------------------------------------------------------------
# MTSplice: B-spline positional-gain reweighting over DNA conv towers.
# ---------------------------------------------------------------------------
def _cubic_bspline_basis(n_positions: int, n_bases: int = 6) -> torch.Tensor:
    """Build an open-uniform-knot cubic B-spline design matrix.

    Faithful (dependency-free) port of the knot construction and recursive
    Cox-de Boor evaluation used by ``get_knots``/``get_X_spline`` in the
    official ``mmsplice/layers.py``.

    Parameters
    ----------
    n_positions : int
        Number of sequence positions to evaluate the spline basis at.
    n_bases : int
        Number of B-spline basis functions.

    Returns
    -------
    torch.Tensor
        Shape ``(n_positions, n_bases)`` spline design matrix.
    """
    degree = 3
    start, end = 0.0, float(n_positions - 1)
    x_range = end - start
    pad_start = start - x_range * 0.001
    pad_end = end + x_range * 0.001
    n_interior = n_bases - (degree - 1)
    step = (pad_end - pad_start) / max(n_interior - 1, 1)
    n_knots = n_interior + 2 * degree
    knots = torch.tensor(
        [pad_start + step * (i - degree) for i in range(n_knots)],
        dtype=torch.float32,
    )
    positions = torch.linspace(start, end, n_positions)

    def basis(i: int, k: int, t: torch.Tensor) -> torch.Tensor:
        if k == 0:
            lo, hi = knots[i], knots[i + 1]
            return ((t >= lo) & (t < hi)).float()
        denom_a = knots[i + k] - knots[i]
        denom_b = knots[i + k + 1] - knots[i + 1]
        term_a = torch.zeros_like(t)
        term_b = torch.zeros_like(t)
        if denom_a.abs() > 1e-8:
            term_a = (t - knots[i]) / denom_a * basis(i, k - 1, t)
        if denom_b.abs() > 1e-8:
            term_b = (knots[i + k + 1] - t) / denom_b * basis(i + 1, k - 1, t)
        return term_a + term_b

    columns = [basis(i, degree, positions) for i in range(n_bases)]
    return torch.stack(columns, dim=1)


class SplineBasisGain(nn.Module):
    """Learned smooth positional gain via a fixed cubic B-spline basis.

    Direct port of ``SplineWeight1D`` from the official
    ``mmsplice/layers.py``: ``x_out[:, j, k] = x_in[:, j, k] * (1 + f(j))``
    where ``f`` is a per-channel smooth function of position ``j`` built
    from a fixed B-spline design matrix dotted with a learned coefficient
    kernel.
    """

    def __init__(self, seq_len: int, channels: int, n_bases: int = 6) -> None:
        """Initialize the spline positional gain.

        Parameters
        ----------
        seq_len : int
            Sequence length (number of positions) of the conv activations.
        channels : int
            Number of channels (per-channel independent spline gain).
        n_bases : int
            Number of B-spline basis functions.
        """
        super().__init__()
        design = _cubic_bspline_basis(seq_len, n_bases=n_bases)
        self.register_buffer("design", design)
        self.kernel = nn.Parameter(torch.zeros(n_bases, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the position-dependent gain.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, channels)`` conv activations.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, seq_len, channels)`` reweighted activations.
        """
        spline_track = self.design @ self.kernel
        return x * (1.0 + spline_track.unsqueeze(0))


class SpliceSiteTower(nn.Module):
    """1D-conv tower scoring a one-hot DNA window (donor or acceptor side)."""

    def __init__(self, seq_len: int, channels: int = 16) -> None:
        """Initialize the splice-site conv tower.

        Parameters
        ----------
        seq_len : int
            Length of the input DNA window.
        channels : int
            Number of conv output channels.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(4, channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.spline_gain = SplineBasisGain(seq_len, channels)

    def forward(self, one_hot_dna: torch.Tensor) -> torch.Tensor:
        """Score a one-hot DNA window.

        Parameters
        ----------
        one_hot_dna : torch.Tensor
            Shape ``(batch, 4, seq_len)`` one-hot ACGT encoding.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, channels)`` globally-pooled, spline-reweighted
            conv activations.
        """
        h = torch.relu(self.conv1(one_hot_dna))
        h = torch.relu(self.conv2(h))
        h = h.transpose(1, 2)  # (batch, seq_len, channels)
        h = self.spline_gain(h)
        return h.mean(dim=1)


class MTSplice(nn.Module):
    """MTSplice tissue-specific splicing-variant-effect model.

    Acceptor- and donor-side DNA conv towers (each with a B-spline
    positional gain) are pooled, concatenated, and projected to one
    delta-PSI prediction per tissue.
    """

    def __init__(self, seq_len: int = 40, n_tissues: int = 12) -> None:
        """Initialize MTSplice.

        Parameters
        ----------
        seq_len : int
            Length of each acceptor/donor DNA window.
        n_tissues : int
            Number of GTEx-style tissue output channels.
        """
        super().__init__()
        self.acceptor_tower = SpliceSiteTower(seq_len)
        self.donor_tower = SpliceSiteTower(seq_len)
        self.tissue_head = nn.Linear(32, n_tissues)

    def forward(self, acceptor: torch.Tensor, donor: torch.Tensor) -> torch.Tensor:
        """Predict per-tissue delta-PSI splicing effect.

        Parameters
        ----------
        acceptor : torch.Tensor
            Shape ``(batch, 4, seq_len)`` one-hot acceptor-side DNA window.
        donor : torch.Tensor
            Shape ``(batch, 4, seq_len)`` one-hot donor-side DNA window.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_tissues)`` per-tissue delta-PSI predictions.
        """
        acc_repr = self.acceptor_tower(acceptor)
        don_repr = self.donor_tower(donor)
        combined = torch.cat([acc_repr, don_repr], dim=1)
        return self.tissue_head(combined)


def build_mtsplice() -> nn.Module:
    """Build a compact MTSplice tissue-specific splicing model.

    Returns
    -------
    nn.Module
        MTSplice reconstruction in evaluation mode.
    """
    return MTSplice(seq_len=40, n_tissues=12).eval()


def example_input_mtsplice() -> tuple[torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_mtsplice`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(acceptor, donor)`` one-hot DNA windows, each shape ``(4, 4, 40)``.
    """
    acceptor = torch.eye(4)[torch.randint(0, 4, (4, 40))].transpose(1, 2)
    donor = torch.eye(4)[torch.randint(0, 4, (4, 40))].transpose(1, 2)
    return acceptor, donor


# ---------------------------------------------------------------------------
# NeoDTI: heterogeneous-network neighbor aggregation + bilinear reconstruction.
# ---------------------------------------------------------------------------
def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    """Row-normalize an adjacency matrix, matching NeoDTI's ``row_normalize``.

    Parameters
    ----------
    matrix : torch.Tensor
        Shape ``(n, m)`` non-negative adjacency/similarity matrix.

    Returns
    -------
    torch.Tensor
        Row-normalized matrix (rows summing to 1, with zero-safe division).
    """
    row_sums = matrix.sum(dim=1, keepdim=True) + 1e-12
    return matrix / row_sums


class RelationAggregator(nn.Module):
    """One relation's contribution to a neighbor-aggregation message.

    Direct port of NeoDTI's ``a_layer``: a shared per-relation linear+ReLU
    projection of the neighboring node type's embedding, whose output is
    then weighted by the (row-normalized) adjacency and summed into the
    aggregating node type's message.
    """

    def __init__(self, in_dim: int, pass_dim: int) -> None:
        """Initialize the relation-specific projection.

        Parameters
        ----------
        in_dim : int
            Embedding dimension of the neighboring node type.
        pass_dim : int
            Message-passing dimension.
        """
        super().__init__()
        self.proj = nn.Linear(in_dim, pass_dim)

    def forward(
        self, adjacency_normalized: torch.Tensor, neighbor_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute this relation's message contribution.

        Parameters
        ----------
        adjacency_normalized : torch.Tensor
            Shape ``(n_self, n_neighbor)`` row-normalized adjacency.
        neighbor_embedding : torch.Tensor
            Shape ``(n_neighbor, in_dim)`` neighboring node-type embedding.

        Returns
        -------
        torch.Tensor
            Shape ``(n_self, pass_dim)`` message contribution.
        """
        projected = torch.relu(self.proj(neighbor_embedding))
        return adjacency_normalized @ projected


def bi_layer(
    x0: torch.Tensor, x1: torch.Tensor, w0: nn.Linear, w1: nn.Linear | None
) -> torch.Tensor:
    """Low-rank bilinear edge reconstruction, matching NeoDTI's ``bi_layer``.

    Parameters
    ----------
    x0 : torch.Tensor
        Shape ``(n0, dim)`` first node-type representation.
    x1 : torch.Tensor
        Shape ``(n1, dim)`` second node-type representation.
    w0 : nn.Linear
        Projection applied to ``x0``.
    w1 : nn.Linear or None
        Projection applied to ``x1``; if ``None``, ``w0`` is reused
        (symmetric relation, e.g. drug-drug or protein-protein).

    Returns
    -------
    torch.Tensor
        Shape ``(n0, n1)`` reconstructed relation matrix.
    """
    left = w0(x0)
    right = (w1 or w0)(x1)
    return left @ right.transpose(0, 1)


class NeoDTI(nn.Module):
    """Heterogeneous drug-target-disease-sideeffect network embedding model.

    One-hop relation-specific neighbor aggregation per node type (drug,
    protein, disease, side-effect), followed by bilinear low-rank
    reconstruction of the drug-protein interaction matrix, mirroring
    ``Model._build_model`` in the official ``NeoDTI_cv.py``.
    """

    def __init__(
        self,
        n_drug: int = 12,
        n_protein: int = 10,
        n_disease: int = 6,
        n_sideeffect: int = 5,
        embed_dim: int = 16,
        pass_dim: int = 16,
        pred_dim: int = 8,
    ) -> None:
        """Initialize NeoDTI.

        Parameters
        ----------
        n_drug : int
            Number of drug nodes.
        n_protein : int
            Number of protein nodes.
        n_disease : int
            Number of disease nodes.
        n_sideeffect : int
            Number of side-effect nodes.
        embed_dim : int
            Node embedding dimension.
        pass_dim : int
            Message-passing dimension.
        pred_dim : int
            Bilinear reconstruction rank.
        """
        super().__init__()
        self.n_drug = n_drug
        self.n_protein = n_protein
        self.n_disease = n_disease
        self.n_sideeffect = n_sideeffect

        self.drug_embedding = nn.Parameter(torch.randn(n_drug, embed_dim) * 0.1)
        self.protein_embedding = nn.Parameter(torch.randn(n_protein, embed_dim) * 0.1)
        self.disease_embedding = nn.Parameter(torch.randn(n_disease, embed_dim) * 0.1)
        self.sideeffect_embedding = nn.Parameter(torch.randn(n_sideeffect, embed_dim) * 0.1)

        # Relation-specific aggregators feeding into the drug representation.
        self.drug_drug_agg = RelationAggregator(embed_dim, pass_dim)
        self.drug_disease_agg = RelationAggregator(embed_dim, pass_dim)
        self.drug_sideeffect_agg = RelationAggregator(embed_dim, pass_dim)
        self.drug_protein_agg = RelationAggregator(embed_dim, pass_dim)

        # Relation-specific aggregators feeding into the protein representation.
        self.protein_protein_agg = RelationAggregator(embed_dim, pass_dim)
        self.protein_disease_agg = RelationAggregator(embed_dim, pass_dim)
        self.protein_drug_agg = RelationAggregator(embed_dim, pass_dim)

        # Shared feature-passing weight (single ``W0`` in the official repo).
        self.combine_drug = nn.Linear(pass_dim + embed_dim, embed_dim)
        self.combine_protein = nn.Linear(pass_dim + embed_dim, embed_dim)

        self.drug_protein_w0 = nn.Linear(embed_dim, pred_dim, bias=False)
        self.drug_protein_w1 = nn.Linear(embed_dim, pred_dim, bias=False)

    def forward(
        self,
        drug_drug: torch.Tensor,
        drug_disease: torch.Tensor,
        drug_sideeffect: torch.Tensor,
        drug_protein: torch.Tensor,
        protein_protein: torch.Tensor,
        protein_disease: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct the drug-protein interaction matrix.

        Parameters
        ----------
        drug_drug : torch.Tensor
            Shape ``(n_drug, n_drug)`` drug-drug interaction matrix.
        drug_disease : torch.Tensor
            Shape ``(n_drug, n_disease)`` drug-disease association matrix.
        drug_sideeffect : torch.Tensor
            Shape ``(n_drug, n_sideeffect)`` drug-side-effect matrix.
        drug_protein : torch.Tensor
            Shape ``(n_drug, n_protein)`` drug-protein interaction matrix.
        protein_protein : torch.Tensor
            Shape ``(n_protein, n_protein)`` protein-protein interaction
            matrix.
        protein_disease : torch.Tensor
            Shape ``(n_protein, n_disease)`` protein-disease association
            matrix.

        Returns
        -------
        torch.Tensor
            Shape ``(n_drug, n_protein)`` reconstructed drug-protein
            interaction matrix.
        """
        protein_drug = drug_protein.transpose(0, 1)

        drug_message = (
            self.drug_drug_agg(_row_normalize(drug_drug), self.drug_embedding)
            + self.drug_disease_agg(_row_normalize(drug_disease), self.disease_embedding)
            + self.drug_sideeffect_agg(_row_normalize(drug_sideeffect), self.sideeffect_embedding)
            + self.drug_protein_agg(_row_normalize(drug_protein), self.protein_embedding)
        )
        drug_repr = torch.relu(
            self.combine_drug(torch.cat([drug_message, self.drug_embedding], dim=1))
        )
        drug_repr = nn.functional.normalize(drug_repr, dim=1)

        protein_message = (
            self.protein_protein_agg(_row_normalize(protein_protein), self.protein_embedding)
            + self.protein_disease_agg(_row_normalize(protein_disease), self.disease_embedding)
            + self.protein_drug_agg(_row_normalize(protein_drug), self.drug_embedding)
        )
        protein_repr = torch.relu(
            self.combine_protein(torch.cat([protein_message, self.protein_embedding], dim=1))
        )
        protein_repr = nn.functional.normalize(protein_repr, dim=1)

        return bi_layer(drug_repr, protein_repr, self.drug_protein_w0, self.drug_protein_w1)


def build_neodti() -> nn.Module:
    """Build a compact NeoDTI heterogeneous-network embedding model.

    Returns
    -------
    nn.Module
        NeoDTI reconstruction in evaluation mode.
    """
    return NeoDTI().eval()


def example_input_neodti() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create example input for :func:`build_neodti`.

    Returns
    -------
    tuple of torch.Tensor
        ``(drug_drug, drug_disease, drug_sideeffect, drug_protein,
        protein_protein, protein_disease)`` heterogeneous-network adjacency
        matrices, matching the dims used in :func:`build_neodti`.
    """
    n_drug, n_protein, n_disease, n_sideeffect = 12, 10, 6, 5
    drug_drug = (torch.rand(n_drug, n_drug) > 0.7).float()
    drug_disease = (torch.rand(n_drug, n_disease) > 0.7).float()
    drug_sideeffect = (torch.rand(n_drug, n_sideeffect) > 0.7).float()
    drug_protein = (torch.rand(n_drug, n_protein) > 0.7).float()
    protein_protein = (torch.rand(n_protein, n_protein) > 0.7).float()
    protein_disease = (torch.rand(n_protein, n_disease) > 0.7).float()
    return (
        drug_drug,
        drug_disease,
        drug_sideeffect,
        drug_protein,
        protein_protein,
        protein_disease,
    )


MENAGERIE_ENTRIES = [
    ("MISTy", "build_misty", "example_input_misty", "2022", "BIO"),
    ("MPVNN", "build_mpvnn", "example_input_mpvnn", "2022", "BIO"),
    ("MPVNN cancer survival", "build_mpvnn", "example_input_mpvnn", "2022", "BIO"),
    ("MTSplice", "build_mtsplice", "example_input_mtsplice", "2021", "BIO"),
    ("NeoDTI", "build_neodti", "example_input_neodti", "2019", "BIO"),
]
