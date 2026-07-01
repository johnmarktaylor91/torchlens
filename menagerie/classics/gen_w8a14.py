"""Compact faithful reimplementations for build_queue rows 85-90 (W8A14).

Sources checked (repo browsed via ``gh api`` / web, no clone/pip-install):
  - TorchANI (cand_01145) is SKIPPED: the build queue row resolves to
    ``https://github.com/aiqm/torchani``, the exact repository already built
    as ``build_ani2x`` / ``ANI-2x`` in ``gen_w6a17.py`` (see that module's
    docstring and ``gen_w7a13.py``'s explicit skip note). Same repo, same
    Behler-Parrinello AEV-featurized atomic-potential family -- no new
    architecture to add.
  - ViSNet-LSRM (cand_01146): Li, Simon, Hu, Chuang, Yao, Wu, Wang, Wang,
    Liu, "Long-Short-Range Message-Passing: A Physics-Informed Framework to
    Capture Non-Local Interaction for Scalable Molecular Dynamics
    Simulation", ICLR 2024, arXiv:2304.13542. The paper's own repository is
    https://github.com/liyy2/LSR-MP (not ``microsoft/ViSNet``, which only
    ships vanilla ViSNet -- confirmed by browsing both repos: the
    ``microsoft/ViSNet`` tree has no LSRM-specific module, while
    ``liyy2/LSR-MP`` has ``lightnp/LSRM/models/lsrm_modules.py`` defining
    ``Visnorm_shared_LSRMNorm2_2branchSerial``, the LSR-MP-augmented ViSNet).
    Distinctive mechanism: short-range branch is a ViSNet-style equivariant
    scalar/vector message-passing GNN over a small-cutoff atom graph
    (radial-basis edge features, scalar-vector "EquivariantMultiHeadAttention"
    updates). The long-range branch coarse-grains atoms into groups
    ("fragments", e.g. via a group-center-of-mass rule) and passes messages
    over a *bipartite* atom<->group graph with a larger cutoff, so long-range
    interactions are captured through a small number of coarse group nodes
    instead of a dense all-pairs graph. Group (long-range) and atom
    (short-range) branches run in parallel per layer and are fused: the
    updated group vector/scalar features are broadcast back down to their
    member atoms and added into the atom embedding before the final
    two-branch output head. Reproduced here with a compact ViSNet-style
    short-range block, a coarse group-assignment (fixed group id per atom,
    computed once from position order) + bipartite long-range block, and a
    concatenated two-branch scalar output head, matching the paper's
    "short-range + long-range, serially fused" design.
  - 3D Infomax (cand_01147): Stark, Beaini, Corso, Tossou, Dallago, Gunnemann,
    Lio, "3D Infomax improves GNNs for Molecular Property Prediction", ICML
    2022, arXiv:2110.04126. Repo https://github.com/HannesStark/3DInfomax,
    ``models/net3d.py`` (``Net3D``/``Net3DLayer``, the 3D geometric encoder)
    + ``models/pna.py`` (the 2D topology-only encoder) + the BYOL/InfoNCE
    trainers in ``trainer/`` that maximize mutual information between the
    two encoders' pooled representations. Distinctive mechanism: two
    parallel graph encoders read the *same* molecule from different views --
    a 2D encoder sees only bond connectivity (no coordinates), a 3D encoder
    sees Euclidean inter-atom distances on every edge (soft-edge-gated
    message passing, ``edge_weight = sigmoid(...)`` multiplying each
    message, exactly as in ``Net3DLayer.message_function``) -- both are
    pooled to graph-level vectors and projected through small MLP heads;
    the training objective (reproduced structurally, not the loss itself)
    is a symmetric InfoNCE/normalized-temperature contrastive objective
    between the two views' projections. Reproduced here as the two encoders
    plus their projection heads returning ``(z_2d, z_3d)``, the artifact
    that is actually maximized against each other; the distance-gated
    ``Net3DLayer`` soft-edge mechanism is preserved verbatim in spirit.
  - AR (autoregressive 3D-SBDD) (cand_01148 / cand_01149): Luo, Guan, Ma, Peng,
    "A 3D Generative Model for Structure-Based Drug Design", NeurIPS 2021,
    arXiv:2203.17003 (build-queue arXiv id; the NeurIPS paper is the same
    work). Repo https://github.com/luost26/3D-Generative-SBDD,
    ``models/maskfill.py`` (``MaskFillModel``) + ``models/fields/
    classifier.py`` (``SpatialClassifier``). cand_01148 ("3DSBDD AR") and
    cand_01149 ("AR (autoregressive 3D-SBDD)") are build-queue-flagged
    POTENTIAL_DEDUP resolving to the exact same repository and paper (the
    "AR" model *is* 3D-SBDD's autoregressive generator, there is no second
    variant) -- built once here (``build_ar_sbdd``) and registered under
    both canonical names per the established dedup convention (see e.g.
    ``gen_w8a9.py`` Ewald GNN / Ewald Message Passing / EwaldMP). Distinctive
    mechanism: a context encoder embeds protein-pocket and partially-built
    ligand atoms together into one point cloud and runs a compact
    equivariant message-passing GNN over it; a ``SpatialClassifier``
    "field" then, for arbitrary 3D query points, gathers the k nearest
    context atoms, expands the query-to-context distance through Gaussian
    radial basis functions, applies a smooth cosine cutoff envelope
    (``0.5*(cos(d*pi/cutoff)+1)``, verbatim from ``classifier.py``), and
    scatter-aggregates the resulting messages into per-query logits over
    atom-element classes plus a scalar "keep growing" indicator -- exactly
    the density field used to autoregressively pick where and what atom to
    place next. Reproduced here with the same context encoder + spatial
    density-field classifier design (KNN implemented via ``torch.cdist``
    instead of ``torch_geometric.nn.knn`` to avoid a ``torch_cluster``
    dependency).
  - ASKCOS Condition Recommender (cand_01150): Gao, Struble, Coley, Wang,
    Green, Jensen, "Using Machine Learning to Predict Suitable Conditions
    for Organic Reactions", ACS Cent. Sci. 2018 (the model shipped inside
    the ASKCOS platform, arxiv 2501.01835 is the 2025 ASKCOS-suite overview
    paper the build-queue row cites). Repo
    https://github.com/Coughy1991/Reaction_condition_recommendation,
    ``scripts/train_model_c_s_r_deploy.py`` (function ``build``), a Keras/
    Theano model reimplemented here natively in ``torch.nn``. Distinctive
    mechanism: NOT a generic multi-task MLP -- reaction/product Morgan
    fingerprints are transformed by a shared trunk, then a *sequential
    conditioning cascade* predicts one reaction-condition slot at a time
    (catalyst c1 -> solvent s1 -> solvent s2 -> reagent r1 -> reagent r2 ->
    temperature T), where each stage's small dense-embedded prediction is
    concatenated onto the running context before the next stage's own
    two-layer (relu then tanh) MLP head runs, exactly mirroring the
    ``concat_fp_c1``, ``concat_fp_c1_s1``, ... chain in ``build()``.
    Reproduced here as an explicit six-stage cascade of dense heads with
    growing concatenated context, condition-slot softmax outputs, and a
    final linear temperature regression head.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# ViSNet-LSRM
# ---------------------------------------------------------------------------


def _rbf(dist: Tensor, num_rbf: int, cutoff: float) -> Tensor:
    """Expand a distance tensor into a Gaussian radial-basis feature bank.

    Parameters
    ----------
    dist : Tensor
        Pairwise distances, any shape.
    num_rbf : int
        Number of radial-basis centers.
    cutoff : float
        Largest distance covered by the basis.

    Returns
    -------
    Tensor
        ``dist.shape + (num_rbf,)`` Gaussian features.
    """

    centers = torch.linspace(0.0, cutoff, num_rbf, device=dist.device, dtype=dist.dtype)
    width = cutoff / num_rbf
    return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (2 * width**2))


def _cosine_envelope(dist: Tensor, cutoff: float) -> Tensor:
    """Smooth cosine cutoff envelope, zero at and beyond ``cutoff``."""

    env = 0.5 * (torch.cos(dist * math.pi / cutoff) + 1.0)
    return env * (dist < cutoff).to(dist.dtype)


class _ShortRangeBlock(nn.Module):
    """ViSNet-style scalar/vector equivariant message-passing layer."""

    def __init__(self, hidden: int, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.msg_scalar = nn.Sequential(
            nn.Linear(hidden * 2 + num_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.msg_vec_gate = nn.Linear(hidden, hidden)
        self.update_scalar = nn.Linear(hidden, hidden)
        self.update_vec_scale = nn.Linear(hidden, hidden, bias=False)

    def forward(self, s: Tensor, v: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Update scalar features ``s`` (N,H) and vector features ``v`` (N,3,H)."""

        n = s.shape[0]
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # (N, N, 3) : i - j
        dist = diff.norm(dim=-1).clamp_min(1e-6)
        env = _cosine_envelope(dist, self.cutoff)
        rbf = _rbf(dist, self.num_rbf, self.cutoff)

        si = s.unsqueeze(1).expand(n, n, -1)
        sj = s.unsqueeze(0).expand(n, n, -1)
        m = self.msg_scalar(torch.cat([si, sj, rbf], dim=-1)) * env.unsqueeze(-1)
        s_agg = m.sum(dim=1)

        gate = self.msg_vec_gate(m)  # (N, N, H)
        unit = diff / dist.unsqueeze(-1)  # (N, N, 3)
        v_msg = unit.unsqueeze(-1) * gate.unsqueeze(2)  # (N, N, 3, H)

        # Fold the current vector channel's norm (an invariant scalar) back into
        # the scalar update, and use it to gate the vector channel's own scale --
        # a compact scalar<->vector coupling in the spirit of ViSNet's runtime
        # geometric vector/scalar interaction.
        v_norm = v.norm(dim=1)  # (N, H)
        s_out = s + s_agg + self.update_scalar(v_norm)
        v_out = v + v_msg.sum(dim=1) + v * self.update_vec_scale(v_norm).unsqueeze(1)
        return s_out, v_out


class _LongRangeBlock(nn.Module):
    """Bipartite atom<->group long-range message passing over coarse groups."""

    def __init__(self, hidden: int, num_rbf: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.to_group = nn.Sequential(
            nn.Linear(hidden * 2 + num_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.to_atom = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, s: Tensor, pos: Tensor, group_id: Tensor, num_groups: int) -> Tensor:
        """Scatter atoms into groups, message-pass, broadcast back to atoms."""

        group_pos = torch.zeros(num_groups, 3, device=pos.device, dtype=pos.dtype)
        counts = torch.zeros(num_groups, device=pos.device, dtype=pos.dtype)
        group_pos.index_add_(0, group_id, pos)
        counts.index_add_(0, group_id, torch.ones_like(counts[group_id]))
        group_pos = group_pos / counts.clamp_min(1.0).unsqueeze(-1)

        group_s = torch.zeros(num_groups, s.shape[-1], device=s.device, dtype=s.dtype)
        group_s.index_add_(0, group_id, s)
        group_s = group_s / counts.clamp_min(1.0).unsqueeze(-1)

        diff = pos.unsqueeze(1) - group_pos.unsqueeze(0)  # (N, G, 3)
        dist = diff.norm(dim=-1).clamp_min(1e-6)
        env = _cosine_envelope(dist, self.cutoff)
        rbf = _rbf(dist, self.num_rbf, self.cutoff)

        n, g = s.shape[0], num_groups
        si = s.unsqueeze(1).expand(n, g, -1)
        gj = group_s.unsqueeze(0).expand(n, g, -1)
        atom_to_group_msg = self.to_group(torch.cat([si, gj, rbf], dim=-1)) * env.unsqueeze(-1)
        group_update = torch.zeros_like(group_s)
        group_update.index_add_(0, group_id, atom_to_group_msg.mean(dim=1))

        broadcast = group_update[group_id]  # (N, H)
        return self.to_atom(torch.cat([s, broadcast], dim=-1))


class ViSNetLSRM(nn.Module):
    """Compact ViSNet-LSRM: short-range equivariant GNN fused with a
    coarse-grained (grouped) long-range bipartite message-passing branch."""

    def __init__(
        self,
        hidden: int = 32,
        num_layers: int = 2,
        num_rbf: int = 12,
        short_cutoff: float = 4.0,
        long_cutoff: float = 10.0,
        group_size: int = 3,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.group_size = group_size
        self.embedding = nn.Embedding(30, hidden)
        self.short_blocks = nn.ModuleList(
            [_ShortRangeBlock(hidden, num_rbf, short_cutoff) for _ in range(num_layers)]
        )
        self.long_blocks = nn.ModuleList(
            [_LongRangeBlock(hidden, num_rbf, long_cutoff) for _ in range(num_layers)]
        )
        self.norm_short = nn.LayerNorm(hidden)
        self.norm_long = nn.LayerNorm(hidden)
        self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, z: Tensor, pos: Tensor) -> Tensor:
        """Predict a scalar per-molecule property from species ``z`` and coords ``pos``."""

        n = z.shape[0]
        s = self.embedding(z)
        v = torch.zeros(n, 3, self.hidden, device=pos.device, dtype=s.dtype)
        order = torch.arange(n, device=pos.device)
        group_id = order // self.group_size
        num_groups = int(group_id.max().item()) + 1

        s_long = s
        for short_block, long_block in zip(self.short_blocks, self.long_blocks):
            s, v = short_block(s, v, pos)
            s_long = long_block(s_long, pos, group_id, num_groups)

        s_short = self.norm_short(s)
        s_long = self.norm_long(s_long)
        fused = torch.cat([s_short, s_long], dim=-1)
        return self.out(fused).sum(dim=0)


def build_visnet_lsrm() -> nn.Module:
    """Build a compact ViSNet-LSRM.

    Returns
    -------
    nn.Module
        ``ViSNetLSRM`` in eval mode.
    """

    return ViSNetLSRM().eval()


def example_input_visnet_lsrm() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_visnet_lsrm`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(z, pos)`` -- 9 atoms with species indices and 3D coordinates.
    """

    torch.manual_seed(0)
    z = torch.randint(0, 20, (9,))
    pos = torch.randn(9, 3) * 3.0
    return z, pos


# ---------------------------------------------------------------------------
# 3D Infomax
# ---------------------------------------------------------------------------


class _Net3DLayer(nn.Module):
    """Distance-gated 3D message-passing layer (soft-edge weighting)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(hidden * 2 + hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.soft_edge = nn.Linear(hidden, 1)

    def forward(self, feat: Tensor, edge_feat: Tensor) -> Tensor:
        """``feat``: (N, H) node features; ``edge_feat``: (N, N, H) edge distance features."""

        n = feat.shape[0]
        fi = feat.unsqueeze(1).expand(n, n, -1)
        fj = feat.unsqueeze(0).expand(n, n, -1)
        message = self.message_net(torch.cat([fi, fj, edge_feat], dim=-1))
        edge_weight = torch.sigmoid(self.soft_edge(message))
        gated = message * edge_weight
        agg = gated.sum(dim=1)
        return self.update_net(agg)


class _Net3DEncoder(nn.Module):
    """3D geometric encoder: distance-gated message passing + pooled projection head."""

    def __init__(self, hidden: int = 32, num_layers: int = 2, proj_dim: int = 16) -> None:
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(hidden) * 0.1)
        self.edge_input = nn.Sequential(nn.Linear(1, hidden), nn.SiLU())
        self.layers = nn.ModuleList([_Net3DLayer(hidden) for _ in range(num_layers)])
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, proj_dim))

    def forward(self, pos: Tensor) -> Tensor:
        n = pos.shape[0]
        feat = self.node_embedding.unsqueeze(0).expand(n, -1).clone()
        dist = torch.cdist(pos, pos).unsqueeze(-1)
        edge_feat = self.edge_input(dist)
        for layer in self.layers:
            feat = feat + layer(feat, edge_feat)
        pooled = feat.mean(dim=0)
        return self.proj(pooled)


class _Net2DEncoder(nn.Module):
    """2D topology-only encoder: bond-connectivity message passing + projection head."""

    def __init__(self, hidden: int = 32, num_layers: int = 2, proj_dim: int = 16) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(16, hidden)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, proj_dim))

    def forward(self, atom_type: Tensor, adjacency: Tensor) -> Tensor:
        feat = self.atom_embedding(atom_type)
        n = feat.shape[0]
        for layer in self.layers:
            fi = feat.unsqueeze(1).expand(n, n, -1)
            fj = feat.unsqueeze(0).expand(n, n, -1)
            msg = layer(torch.cat([fi, fj], dim=-1)) * adjacency.unsqueeze(-1)
            feat = feat + msg.sum(dim=1)
        return self.proj(feat.mean(dim=0))


class ThreeDInfomax(nn.Module):
    """Dual 2D/3D molecular encoder pair whose pooled projections are trained
    to maximize mutual information (contrastive 2D<->3D pretraining)."""

    def __init__(self, hidden: int = 32, proj_dim: int = 16) -> None:
        super().__init__()
        self.encoder_2d = _Net2DEncoder(hidden, proj_dim=proj_dim)
        self.encoder_3d = _Net3DEncoder(hidden, proj_dim=proj_dim)

    def forward(self, atom_type: Tensor, adjacency: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Return the ``(2D projection, 3D projection)`` pair to be contrasted."""

        z_2d = self.encoder_2d(atom_type, adjacency)
        z_3d = self.encoder_3d(pos)
        return z_2d, z_3d


def build_3dinfomax() -> nn.Module:
    """Build a compact 3D Infomax dual-encoder model.

    Returns
    -------
    nn.Module
        ``ThreeDInfomax`` in eval mode.
    """

    return ThreeDInfomax().eval()


def example_input_3dinfomax() -> tuple[Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_3dinfomax`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(atom_type, adjacency, pos)`` for an 8-atom toy molecule.
    """

    torch.manual_seed(1)
    n = 8
    atom_type = torch.randint(0, 8, (n,))
    idx = torch.arange(n)
    adjacency = ((idx.unsqueeze(0) - idx.unsqueeze(1)).abs() == 1).float()
    pos = torch.randn(n, 3)
    return atom_type, adjacency, pos


# ---------------------------------------------------------------------------
# AR (autoregressive 3D-SBDD)
# ---------------------------------------------------------------------------


class _ContextEncoder(nn.Module):
    """Compact equivariant message-passing encoder over a protein+ligand point cloud."""

    def __init__(
        self, hidden: int, num_layers: int = 2, cutoff: float = 6.0, num_rbf: int = 12
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 2 + num_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, h: Tensor, pos: Tensor) -> Tensor:
        n = h.shape[0]
        dist = torch.cdist(pos, pos)
        env = _cosine_envelope(dist, self.cutoff)
        rbf = _rbf(dist, self.num_rbf, self.cutoff)
        for layer in self.layers:
            hi = h.unsqueeze(1).expand(n, n, -1)
            hj = h.unsqueeze(0).expand(n, n, -1)
            msg = layer(torch.cat([hi, hj, rbf], dim=-1)) * env.unsqueeze(-1)
            h = h + msg.sum(dim=1)
        return h


class _SpatialClassifier(nn.Module):
    """KNN density field: predicts per-query atom-class logits + growth indicator."""

    def __init__(
        self, hidden: int, num_classes: int, num_filters: int = 32, k: int = 8, cutoff: float = 8.0
    ) -> None:
        super().__init__()
        self.k = k
        self.cutoff = cutoff
        self.lin1 = nn.Linear(hidden, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, num_filters)
        self.filter_net = nn.Sequential(
            nn.Linear(num_filters, num_filters), nn.Softplus(), nn.Linear(num_filters, num_filters)
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_filters), nn.Softplus(), nn.Linear(num_filters, num_classes)
        )
        self.indicator = nn.Sequential(
            nn.Linear(num_filters, num_filters), nn.Softplus(), nn.Linear(num_filters, 1)
        )
        self.num_filters = num_filters

    def forward(self, pos_query: Tensor, pos_ctx: Tensor, h_ctx: Tensor) -> tuple[Tensor, Tensor]:
        dist = torch.cdist(pos_query, pos_ctx)  # (Q, C)
        k = min(self.k, dist.shape[1])
        topk_dist, topk_idx = dist.topk(k, dim=-1, largest=False)  # (Q, k)
        h_ctx_j = h_ctx[topk_idx]  # (Q, k, H)

        rbf = _rbf(topk_dist, self.num_filters, self.cutoff)  # reuse rbf width as radial expansion
        w = self.filter_net(rbf)
        h = self.lin2(w * self.lin1(h_ctx_j))

        env = _cosine_envelope(topk_dist, self.cutoff).unsqueeze(-1)
        h = (h * env).sum(dim=1)

        return self.classifier(h), self.indicator(h)


class ARStructureBasedDrugDesign(nn.Module):
    """Autoregressive 3D structure-based drug design: encode protein+ligand
    context, then query a spatial density field for the next atom's class
    and location-suitability score."""

    def __init__(
        self,
        hidden: int = 32,
        num_atom_classes: int = 8,
        protein_feat_dim: int = 10,
        ligand_feat_dim: int = 8,
    ) -> None:
        super().__init__()
        self.protein_emb = nn.Linear(protein_feat_dim, hidden)
        self.ligand_emb = nn.Linear(ligand_feat_dim, hidden)
        self.encoder = _ContextEncoder(hidden)
        self.field = _SpatialClassifier(hidden, num_atom_classes)

    def forward(
        self,
        pos_query: Tensor,
        protein_pos: Tensor,
        protein_feat: Tensor,
        ligand_pos: Tensor,
        ligand_feat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(class_logits, growth_indicator)`` for each query point."""

        h_protein = self.protein_emb(protein_feat)
        h_ligand = self.ligand_emb(ligand_feat)
        h_ctx = torch.cat([h_protein, h_ligand], dim=0)
        pos_ctx = torch.cat([protein_pos, ligand_pos], dim=0)

        h_ctx = self.encoder(h_ctx, pos_ctx)
        return self.field(pos_query, pos_ctx, h_ctx)


def build_ar_sbdd() -> nn.Module:
    """Build a compact autoregressive 3D-SBDD generator.

    Returns
    -------
    nn.Module
        ``ARStructureBasedDrugDesign`` in eval mode.
    """

    return ARStructureBasedDrugDesign().eval()


def example_input_ar_sbdd() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_ar_sbdd`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        ``(pos_query, protein_pos, protein_feat, ligand_pos, ligand_feat)``.
    """

    torch.manual_seed(2)
    pos_query = torch.randn(5, 3)
    protein_pos = torch.randn(10, 3) * 4.0
    protein_feat = torch.rand(10, 10)
    ligand_pos = torch.randn(6, 3)
    ligand_feat = torch.rand(6, 8)
    return pos_query, protein_pos, protein_feat, ligand_pos, ligand_feat


# ---------------------------------------------------------------------------
# ASKCOS Condition Recommender
# ---------------------------------------------------------------------------


class _CascadeStage(nn.Module):
    """One conditioning stage: two-layer (relu -> tanh) head + softmax logits,
    plus a dense re-embedding of the (teacher-forced) chosen slot value."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.h1 = nn.Linear(in_dim, hidden)
        self.h2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.embed = nn.Linear(out_dim, embed_dim)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(slot_logits, dense_embedding_of_slot_prediction)``."""

        h = torch.relu(self.h1(context))
        h = torch.tanh(self.h2(h))
        logits = self.out(h)
        probs = torch.softmax(logits, dim=-1)
        embedded = torch.relu(self.embed(probs))
        return logits, embedded


class AskcosConditionRecommender(nn.Module):
    """Reaction-condition recommender: a shared fingerprint trunk feeding a
    sequential conditioning cascade (catalyst -> solvent1 -> solvent2 ->
    reagent1 -> reagent2 -> temperature), each stage conditioned on every
    prior stage's dense-embedded prediction."""

    def __init__(
        self,
        fp_dim: int = 64,
        trunk_hidden: int = 48,
        stage_hidden: int = 32,
        embed_dim: int = 16,
        n_catalyst: int = 10,
        n_solvent: int = 12,
        n_reagent: int = 12,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(fp_dim * 2, trunk_hidden),
            nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.ReLU(),
        )

        self.c1 = _CascadeStage(trunk_hidden, stage_hidden, n_catalyst, embed_dim)
        self.s1 = _CascadeStage(trunk_hidden + embed_dim, stage_hidden, n_solvent, embed_dim)
        self.s2 = _CascadeStage(trunk_hidden + 2 * embed_dim, stage_hidden, n_solvent, embed_dim)
        self.r1 = _CascadeStage(trunk_hidden + 3 * embed_dim, stage_hidden, n_reagent, embed_dim)
        self.r2 = _CascadeStage(trunk_hidden + 4 * embed_dim, stage_hidden, n_reagent, embed_dim)
        self.temp_head = nn.Sequential(
            nn.Linear(trunk_hidden + 5 * embed_dim, stage_hidden),
            nn.ReLU(),
            nn.Linear(stage_hidden, 1),
        )

    def forward(
        self, product_fp: Tensor, reaction_fp: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return ``(c1, s1, s2, r1, r2, temperature)`` logits/values."""

        h = self.trunk(torch.cat([product_fp, reaction_fp], dim=-1))

        c1_logits, c1_emb = self.c1(h)
        ctx = torch.cat([h, c1_emb], dim=-1)

        s1_logits, s1_emb = self.s1(ctx)
        ctx = torch.cat([ctx, s1_emb], dim=-1)

        s2_logits, s2_emb = self.s2(ctx)
        ctx = torch.cat([ctx, s2_emb], dim=-1)

        r1_logits, r1_emb = self.r1(ctx)
        ctx = torch.cat([ctx, r1_emb], dim=-1)

        r2_logits, r2_emb = self.r2(ctx)
        ctx = torch.cat([ctx, r2_emb], dim=-1)

        temperature = self.temp_head(ctx)
        return c1_logits, s1_logits, s2_logits, r1_logits, r2_logits, temperature


def build_askcos_condition_recommender() -> nn.Module:
    """Build a compact ASKCOS reaction-condition recommender.

    Returns
    -------
    nn.Module
        ``AskcosConditionRecommender`` in eval mode.
    """

    return AskcosConditionRecommender().eval()


def example_input_askcos_condition_recommender() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_askcos_condition_recommender`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(product_fp, reaction_fp)`` -- batch-of-1 Morgan-fingerprint-like
        binary feature vectors.
    """

    torch.manual_seed(4)
    product_fp = torch.randint(0, 2, (1, 64)).float()
    reaction_fp = torch.randint(0, 2, (1, 64)).float()
    return product_fp, reaction_fp


MENAGERIE_ENTRIES = [
    ("ViSNet-LSRM", "build_visnet_lsrm", "example_input_visnet_lsrm", "2024", "BIO"),
    ("3D Infomax", "build_3dinfomax", "example_input_3dinfomax", "2022", "BIO"),
    ("3DSBDD AR", "build_ar_sbdd", "example_input_ar_sbdd", "2021", "BIO"),
    ("AR (autoregressive 3D-SBDD)", "build_ar_sbdd", "example_input_ar_sbdd", "2021", "BIO"),
    (
        "ASKCOS (Condition Recommender)",
        "build_askcos_condition_recommender",
        "example_input_askcos_condition_recommender",
        "2018",
        "BIO",
    ),
]
