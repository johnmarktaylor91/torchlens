"""Menagerie batch w7a2: protein-sequence diffusion and SE(3) structure/docking models.

Sources checked (reference only; no cloning, no pip installs):
  - EvoDiff (cand_00871): Alamdari, Thakkar, van den Berg, Lu, Fusi, Amini &
    Yang, bioRxiv 2023.09.11.556673 "Protein generation with evolutionary
    diffusion: sequence is all you need", official repo
    https://github.com/microsoft/evodiff (README + ``evodiff/model.py``
    fetched via GitHub API). EvoDiff's sequence models (EvoDiff-Seq) use a
    **dilated-convolutional ByteNet backbone** (from the CARP protein
    masked-LM, itself a stack of residual 1D convolutions whose dilation
    cycles through powers of two to grow the receptive field without
    attention) with a **sinusoidal diffusion-timestep embedding added into
    the token embedding** before the conv tower, trained under two forward
    corruption processes: order-agnostic autoregressive masking (OADM,
    tokens progressively replaced by ``[MASK]``) and discrete D3PM
    (uniform/BLOSUM transition-matrix corruption). This module reproduces
    the defining mechanism -- token embedding + timestep embedding ->
    dilated residual 1D-conv (ByteNet-style) tower -> per-position
    vocabulary logits -- at compact scale, with an OADM-style masked input
    as the example (the outer iterative unmasking sampler is not itself a
    traceable ``nn.Module`` and is intentionally not reproduced).
  - FABind (cand_00873): Pei, Gao, Wu, Zhu, Xia, Xie, Qin, He, Liu & Yan,
    NeurIPS 2023 "FABind: Fast and Accurate Protein-Ligand Binding",
    official repo https://github.com/QizhiPei/FABind (README + the
    ``FABind/fabind/models/att_model.py`` ``EfficientMCAttModel``/
    ``ComplexGraph`` source fetched via GitHub API). FABind's defining
    mechanism is **end-to-end joint pocket-prediction + docking on one
    protein-ligand complex graph**: protein residues and ligand atoms are
    nodes of a single heterogeneous graph (with intra-protein, intra-ligand
    and cross inter-molecular radius-graph edges), a pocket-prediction head
    scores/selects the binding-pocket residues from the protein-only node
    embeddings, and an **iterative E(3)-equivariant multi-channel attention
    message-passing** stack (``MCAttEGNN``-style: coordinate updates driven
    by per-edge attention-gated relative-position vectors) then refines
    ligand-atom coordinates directly, entirely avoiding the outer
    search/sampling loop of classical or diffusion docking. This module
    reproduces the joint hetero-graph embedding + pocket-selection head +
    iterative equivariant coordinate-refinement message passing at compact
    scale.
  - FABFlex (cand_00872): Zhang, Wu, Gao, Yao & Han, ICLR 2025 "Fast and
    Accurate Blind Flexible Docking", official repo
    https://github.com/resistzzz/FABFlex (README fetched via GitHub API,
    including the ``main_fatwo_joint.py``/``inference.py`` CLI signature
    revealing ``--use_iterative --total_iterative 6``, ``--mean_layers 5``,
    ``--pocket_pred_hidden_size``, ``--pocket_radius`` flags). FABFlex
    extends FABind's rigid pocket+ligand docking to **flexible** docking by
    adding a second, symmetric **pocket-structure refinement module**: the
    architecture alternates, across ``total_iterative`` outer rounds,
    between (1) a pocket-prediction/radius-regression head that re-selects
    candidate pocket residues from the current protein conformation, (2) an
    equivariant "mean-layers" message-passing block that updates *pocket
    residue* coordinates (the flexible-protein half FABind lacks), and (3)
    the ligand-docking equivariant block that updates ligand-atom
    coordinates conditioned on the just-updated pocket -- i.e. the
    architecture-defining novelty vs. FABind is this **joint two-sided
    (protein-pocket + ligand) iterative equivariant refinement loop with an
    explicit pocket-radius head**, rather than only refining the ligand
    against a fixed pocket. This module reproduces that two-sided iterative
    refinement (pocket-radius head, pocket-coordinate EGNN block,
    ligand-coordinate EGNN block, repeated for ``total_iterative`` rounds)
    at compact scale, distinct from the FABind module above (which only
    refines the ligand side).
  - FoldFlow (cand_00874): Bose, Akhound-Sadegh, Huguet, Fatras, Rector-
    Brooks, Liu, Nica, Korablyov, Bronstein & Tong, ICLR 2024
    "SE(3)-Stochastic Flow Matching for Protein Backbone Generation",
    arXiv:2310.02391, official repo https://github.com/DreamFold/FoldFlow
    (README fetched via GitHub API; builds on the FrameDiff/openfold IPA
    structure-module lineage per the repo's stated acknowledgment). FoldFlow
    represents each residue as an **SE(3) frame** (rotation + translation)
    and learns a **flow-matching vector field on the SE(3)^N manifold**:
    a shared trunk of **Invariant Point Attention (IPA)** layers -- which
    build geometry-aware attention logits/values from frame-local 3D query
    /key/value *points* projected into and back out of each residue's local
    frame, on top of the usual scalar attention -- consumes the noisy
    per-residue frames at a continuous flow-time ``t`` and predicts a
    tangent (rotation-rate + translation-rate) update field for each frame,
    with FoldFlow's OT/stochastic coupling defining *how* frames are
    interpolated during training (a training-time-only detail, not part of
    the traceable network). This module reproduces the IPA-based per-residue
    SE(3)-frame trunk (invariant point attention conditioned on a flow-time
    embedding) with a vector-field head predicting rotation and translation
    updates, at compact scale.
  - FrameFlow (cand_00875): Yim, Campbell, Mathieu, Foong, Gastegger,
    Jimenez-Luna, Lewis, Satorras, Veeling, Noe, Barzilay & Jaakkola, ICLR
    2024 (arXiv:2310.05297 / TMLR 2401.04082) "Fast protein backbone
    generation with SE(3) flow matching" / "Improved motif-scaffolding with
    SE(3) flow matching", official repo
    https://github.com/microsoft/protein-frame-flow (README fetched via
    GitHub API). FrameFlow shares FoldFlow's IPA-trunk-on-SE(3)-frames flow-
    matching design but its architecture-defining contribution is the
    **motif-scaffolding conditioning path**: alongside the noisy frame at
    flow-time ``t``, the network additionally consumes a per-residue
    **binary motif mask** and the *fixed* motif frames, concatenating a
    motif-conditioning embedding (mask indicator + motif-frame features
    gated by the mask) into every residue's input features before the IPA
    trunk, so scaffolded (non-motif) residues are generated conditioned on
    the frozen motif residues in the same forward pass -- a conditioning
    mechanism FoldFlow's unconditional trunk above does not have. This
    module reproduces the motif-mask-conditioned IPA trunk (motif-frame
    features gated by a binary mask, concatenated into the per-residue
    input before a shared IPA stack) with a rotation/translation
    vector-field head, at compact scale, distinct from the unconditional
    FoldFlow module above.
  - GCPNet (cand_00876): Morehead & Cheng, Bioinformatics 2024 "Geometry-
    complete diffusion for 3D molecule generation and optimization" /
    "Geometry-Complete Perceptron Networks for 3D Molecular Graphs",
    official repo https://github.com/BioinfoMachineLearning/GCPNet
    (``src/models/components/gcpnet.py`` ``GCP``/``GCPMessagePassing``
    source fetched via GitHub API). GCPNet's defining mechanism is the
    **Geometry-Complete Perceptron (GCP)** layer: scalar and vector
    per-node features are updated jointly, but critically the vector
    branch is *scalarized* against a local geometric reference frame (in
    the paper, derived from each residue's/atom's neighborhood geometry)
    -- i.e. vector features are projected into frame-relative invariant
    scalar quantities and re-merged with the scalar stream -- which the
    authors prove makes the layer "geometry-complete" (information-
    preserving) where plain vector-gating GNNs (e.g. GVP) are not, while
    remaining SE(3)-equivariant on the vector output. This module
    reproduces the GCP layer's core scalar/vector duality with frame-based
    scalarization (local frame built from each node's neighbor
    coordinates, vector features scalarized into that frame and
    concatenated back into the scalar update, vector output gated by the
    scalar stream) stacked into a compact message-passing network over a
    small point cloud.
"""

from __future__ import annotations

import math

import torch
from torch import nn


# ---------------------------------------------------------------------------
# EvoDiff: timestep-conditioned dilated-conv (ByteNet) diffusion denoiser.
# ---------------------------------------------------------------------------
class ByteNetBlock(nn.Module):
    """One dilated-residual 1D-convolution block of the ByteNet tower.

    Two convolutions (kernel size ``k``, dilation ``d``) with an inverted-
    bottleneck channel schedule and a residual skip, mirroring the repeated
    unit of EvoDiff's CARP-derived ``ByteNetTime`` backbone.
    """

    def __init__(self, d_model: int, kernel_size: int = 5, dilation: int = 1) -> None:
        """Initialize a dilated ByteNet residual block.

        Parameters
        ----------
        d_model : int
            Channel width of the block (input and output).
        kernel_size : int
            Convolution kernel width.
        dilation : int
            Dilation factor for the depthwise convolution.
        """
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, dilation=dilation, padding=padding)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one dilated-conv residual block.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, length, d_model)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, length, d_model)``.
        """
        h = self.norm1(x)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        x = x + h
        return x + self.ffn(self.norm2(x))


class EvoDiffByteNet(nn.Module):
    """EvoDiff-Seq: timestep-conditioned dilated-conv diffusion denoiser.

    Amino-acid tokens (with some positions replaced by ``[MASK]`` under the
    OADM forward-corruption process) are embedded, summed with a sinusoidal
    diffusion-timestep embedding, and passed through a stack of dilated
    ``ByteNetBlock``s whose dilation cycles through powers of two -- the
    CARP/ByteNet backbone EvoDiff-Seq uses in place of a transformer.
    """

    def __init__(
        self,
        vocab_size: int = 31,
        d_model: int = 64,
        n_layers: int = 6,
        max_dilation_pow2: int = 3,
    ) -> None:
        """Initialize the EvoDiff ByteNet denoiser.

        Parameters
        ----------
        vocab_size : int
            Amino-acid (+ special-token) vocabulary size.
        d_model : int
            Channel width of the ByteNet tower.
        n_layers : int
            Number of dilated residual blocks.
        max_dilation_pow2 : int
            Dilation cycles through ``2**(i % (max_dilation_pow2 + 1))``.
        """
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.blocks = nn.ModuleList(
            [
                ByteNetBlock(d_model, kernel_size=5, dilation=2 ** (i % (max_dilation_pow2 + 1)))
                for i in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Denoise a (partially masked) amino-acid sequence.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape ``(batch, length)`` amino-acid token ids (some positions
            set to the ``[MASK]`` id under OADM corruption).
        timestep : torch.Tensor
            Shape ``(batch,)`` diffusion timestep (fraction of sequence
            masked so far, normalized to ``[0, 1]``).

        Returns
        -------
        torch.Tensor
            Shape ``(batch, length, vocab_size)`` denoised per-position
            amino-acid logits.
        """
        h = self.token_embed(tokens)
        t_embed = self.time_embed(timestep.view(-1, 1).float()).unsqueeze(1)
        h = h + t_embed
        for block in self.blocks:
            h = block(h)
        return self.head(self.out_norm(h))


def build_evodiff() -> nn.Module:
    """Build a compact EvoDiff-Seq ByteNet diffusion denoiser.

    Returns
    -------
    nn.Module
        EvoDiff reconstruction in evaluation mode.
    """
    return EvoDiffByteNet(vocab_size=31, d_model=64, n_layers=6, max_dilation_pow2=3).eval()


def example_input_evodiff() -> tuple[torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_evodiff`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(tokens, timestep)``: a batch of 2 partially ``[MASK]``-token
        (id 30) sequences of length 48, and a timestep fraction per
        sequence.
    """
    mask_id = 30
    tokens = torch.randint(0, 20, (2, 48))
    mask_positions = torch.rand(2, 48) < 0.4
    tokens = torch.where(mask_positions, torch.full_like(tokens, mask_id), tokens)
    timestep = torch.tensor([0.4, 0.7])
    return tokens, timestep


# ---------------------------------------------------------------------------
# FABind: joint protein-ligand hetero-graph, pocket head + equivariant EGNN.
# ---------------------------------------------------------------------------
class EquivariantCoordLayer(nn.Module):
    """One E(3)-equivariant coordinate-update message-passing layer.

    Messages are computed from sender/receiver scalar embeddings and their
    pairwise distance; attention-gated messages update scalar embeddings,
    and a separate scalar gate scales the (rotation/translation-invariant)
    relative-position vector used to update coordinates -- the standard
    EGNN-style update FABind's ``MCAttEGNN`` specializes with multi-channel
    attention.
    """

    def __init__(self, dim: int) -> None:
        """Initialize one equivariant coordinate-update layer.

        Parameters
        ----------
        dim : int
            Node scalar-embedding dimension.
        """
        super().__init__()
        self.msg_mlp = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.attn_gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.coord_gate = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.update_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh node scalar embeddings and coordinates for a fully-connected node set.

        Parameters
        ----------
        h : torch.Tensor
            Shape ``(n_nodes, dim)`` scalar embeddings.
        coords : torch.Tensor
            Shape ``(n_nodes, 3)`` node coordinates.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(h, coords)`` with the same shapes.
        """
        rel = coords.unsqueeze(1) - coords.unsqueeze(0)  # (n, n, 3)
        dist2 = (rel**2).sum(-1, keepdim=True)  # (n, n, 1)
        n = h.shape[0]
        h_i = h.unsqueeze(1).expand(n, n, -1)
        h_j = h.unsqueeze(0).expand(n, n, -1)
        msg_in = torch.cat([h_i, h_j, dist2], dim=-1)
        msg = self.msg_mlp(msg_in)
        gate = self.attn_gate(msg)
        weighted_msg = (msg * gate).sum(dim=1)

        coord_weight = self.coord_gate(msg)  # (n, n, 1)
        coord_update = (rel * coord_weight).mean(dim=1)
        new_coords = coords + coord_update

        h_new = h + self.update_mlp(torch.cat([h, weighted_msg], dim=-1))
        return h_new, new_coords


class FABind(nn.Module):
    """FABind: joint pocket-prediction + iterative equivariant docking.

    Protein residues and ligand atoms share one embedding space; a
    pocket-prediction head scores each protein residue's probability of
    belonging to the binding pocket from protein-only embeddings, and a
    stack of ``EquivariantCoordLayer``s jointly refines protein+ligand
    node embeddings and *ligand* atom coordinates via full cross-node
    message passing (protein coordinates are held fixed, as in FABind's
    rigid-receptor docking setting).
    """

    def __init__(self, dim: int = 32, n_iter: int = 3) -> None:
        """Initialize FABind.

        Parameters
        ----------
        dim : int
            Shared node scalar-embedding dimension.
        n_iter : int
            Number of iterative equivariant refinement layers.
        """
        super().__init__()
        self.protein_embed = nn.Linear(20, dim)
        self.ligand_embed = nn.Linear(12, dim)
        self.segment_embed = nn.Embedding(2, dim)
        self.pocket_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.layers = nn.ModuleList([EquivariantCoordLayer(dim) for _ in range(n_iter)])

    def forward(
        self,
        protein_feat: torch.Tensor,
        protein_coords: torch.Tensor,
        ligand_feat: torch.Tensor,
        ligand_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the binding pocket and refine ligand-atom coordinates.

        Parameters
        ----------
        protein_feat : torch.Tensor
            Shape ``(n_res, 20)`` per-residue (one-hot amino-acid) features.
        protein_coords : torch.Tensor
            Shape ``(n_res, 3)`` protein C-alpha coordinates (fixed/rigid).
        ligand_feat : torch.Tensor
            Shape ``(n_atoms, 12)`` per-atom ligand features.
        ligand_coords : torch.Tensor
            Shape ``(n_atoms, 3)`` initial ligand-atom coordinates.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(pocket_logits, refined_ligand_coords)`` of shape
            ``(n_res,)`` and ``(n_atoms, 3)``.
        """
        n_res = protein_feat.shape[0]
        n_atoms = ligand_feat.shape[0]
        h_p = self.protein_embed(protein_feat) + self.segment_embed(
            torch.zeros(n_res, dtype=torch.long, device=protein_feat.device)
        )
        h_l = self.ligand_embed(ligand_feat) + self.segment_embed(
            torch.ones(n_atoms, dtype=torch.long, device=ligand_feat.device)
        )
        pocket_logits = self.pocket_head(h_p).squeeze(-1)

        h = torch.cat([h_p, h_l], dim=0)
        coords = torch.cat([protein_coords, ligand_coords], dim=0)
        for layer in self.layers:
            h, coords = layer(h, coords)
            coords = torch.cat([protein_coords, coords[n_res:]], dim=0)  # keep receptor rigid

        return pocket_logits, coords[n_res:]


def build_fabind() -> nn.Module:
    """Build a compact FABind joint pocket-prediction + docking model.

    Returns
    -------
    nn.Module
        FABind reconstruction in evaluation mode.
    """
    return FABind(dim=32, n_iter=3).eval()


def example_input_fabind() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_fabind`.

    Returns
    -------
    tuple of torch.Tensor
        ``(protein_feat, protein_coords, ligand_feat, ligand_coords)`` for
        a toy 15-residue protein and 10-atom ligand.
    """
    n_res, n_atoms = 15, 10
    protein_feat = torch.eye(20)[torch.randint(0, 20, (n_res,))]
    protein_coords = torch.randn(n_res, 3) * 5
    ligand_feat = torch.randn(n_atoms, 12)
    ligand_coords = torch.randn(n_atoms, 3) * 2
    return protein_feat, protein_coords, ligand_feat, ligand_coords


# ---------------------------------------------------------------------------
# FABFlex: two-sided iterative pocket + ligand equivariant flexible docking.
# ---------------------------------------------------------------------------
class FABFlex(nn.Module):
    """FABFlex: iterative pocket-radius head + two-sided equivariant docking.

    Extends the FABind design (above) with a **flexible** receptor: each of
    ``total_iterative`` outer rounds re-predicts a pocket radius/selection
    score from the current pocket-residue embeddings, then runs an
    equivariant coordinate-update layer on the *pocket residues themselves*
    (their coordinates move, unlike FABind's rigid receptor) before running
    a second equivariant layer that updates the ligand against the
    just-moved pocket -- the joint two-sided iterative loop that is
    FABFlex's defining addition over FABind.
    """

    def __init__(self, dim: int = 32, total_iterative: int = 4) -> None:
        """Initialize FABFlex.

        Parameters
        ----------
        dim : int
            Shared node scalar-embedding dimension.
        total_iterative : int
            Number of outer pocket/ligand refinement rounds.
        """
        super().__init__()
        self.total_iterative = total_iterative
        self.pocket_embed = nn.Linear(20, dim)
        self.ligand_embed = nn.Linear(12, dim)
        self.radius_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.pocket_layer = EquivariantCoordLayer(dim)
        self.ligand_layer = EquivariantCoordLayer(dim)

    def forward(
        self,
        pocket_feat: torch.Tensor,
        pocket_coords: torch.Tensor,
        ligand_feat: torch.Tensor,
        ligand_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Jointly refine pocket-residue and ligand-atom coordinates.

        Parameters
        ----------
        pocket_feat : torch.Tensor
            Shape ``(n_pocket, 20)`` pocket-residue (one-hot amino-acid)
            features.
        pocket_coords : torch.Tensor
            Shape ``(n_pocket, 3)`` initial (apo) pocket coordinates.
        ligand_feat : torch.Tensor
            Shape ``(n_atoms, 12)`` per-atom ligand features.
        ligand_coords : torch.Tensor
            Shape ``(n_atoms, 3)`` initial ligand-atom coordinates.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(pocket_radius, refined_pocket_coords, refined_ligand_coords)``.
        """
        h_pocket = self.pocket_embed(pocket_feat)
        h_ligand = self.ligand_embed(ligand_feat)
        n_pocket = pocket_feat.shape[0]

        for _ in range(self.total_iterative):
            radius = self.radius_head(h_pocket.mean(dim=0, keepdim=True)).squeeze()
            h_pocket, pocket_coords = self.pocket_layer(h_pocket, pocket_coords)

            h_joint = torch.cat([h_pocket, h_ligand], dim=0)
            coords_joint = torch.cat([pocket_coords, ligand_coords], dim=0)
            h_joint, coords_joint = self.ligand_layer(h_joint, coords_joint)
            h_ligand = h_joint[n_pocket:]
            ligand_coords = coords_joint[n_pocket:]

        return radius, pocket_coords, ligand_coords


def build_fabflex() -> nn.Module:
    """Build a compact FABFlex two-sided iterative flexible-docking model.

    Returns
    -------
    nn.Module
        FABFlex reconstruction in evaluation mode.
    """
    return FABFlex(dim=32, total_iterative=4).eval()


def example_input_fabflex() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_fabflex`.

    Returns
    -------
    tuple of torch.Tensor
        ``(pocket_feat, pocket_coords, ligand_feat, ligand_coords)`` for a
        toy 8-residue apo pocket and 10-atom ligand.
    """
    n_pocket, n_atoms = 8, 10
    pocket_feat = torch.eye(20)[torch.randint(0, 20, (n_pocket,))]
    pocket_coords = torch.randn(n_pocket, 3) * 4
    ligand_feat = torch.randn(n_atoms, 12)
    ligand_coords = torch.randn(n_atoms, 3) * 2
    return pocket_feat, pocket_coords, ligand_feat, ligand_coords


# ---------------------------------------------------------------------------
# FoldFlow / FrameFlow: Invariant Point Attention trunks on SE(3) frames.
# ---------------------------------------------------------------------------
def _rotation_from_6d(rot6d: torch.Tensor) -> torch.Tensor:
    """Recover an orthonormal rotation matrix from a continuous 6D representation.

    Parameters
    ----------
    rot6d : torch.Tensor
        Shape ``(..., 6)`` raw 6D rotation representation.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3, 3)`` orthonormal rotation matrices (Gram-Schmidt).
    """
    a1, a2 = rot6d[..., :3], rot6d[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2_proj = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2_proj, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class InvariantPointAttention(nn.Module):
    """A compact Invariant Point Attention (IPA) layer.

    Per residue frame (rotation + translation), scalar queries/keys/values
    are augmented with 3D query/key/value *points* that are transformed
    into the global frame before computing attention, and back into each
    receiving residue's local frame afterward -- the mechanism that makes
    IPA's attention SE(3)-invariant, following AlphaFold2's structure
    module (reused by the FrameDiff/FoldFlow/FrameFlow lineage).
    """

    def __init__(self, dim: int, n_heads: int = 4, n_points: int = 4) -> None:
        """Initialize an IPA layer.

        Parameters
        ----------
        dim : int
            Per-residue scalar embedding dimension.
        n_heads : int
            Number of attention heads.
        n_points : int
            Number of 3D query/key points per head.
        """
        super().__init__()
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = dim // n_heads
        self.q_scalar = nn.Linear(dim, dim)
        self.k_scalar = nn.Linear(dim, dim)
        self.v_scalar = nn.Linear(dim, dim)
        self.q_points = nn.Linear(dim, n_heads * n_points * 3)
        self.k_points = nn.Linear(dim, n_heads * n_points * 3)
        self.v_points = nn.Linear(dim, n_heads * n_points * 3)
        self.point_weight = nn.Parameter(torch.zeros(n_heads))
        self.out_proj = nn.Linear(dim + n_heads * n_points * 4, dim)

    def forward(self, h: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        """Apply invariant point attention over all residue pairs.

        Parameters
        ----------
        h : torch.Tensor
            Shape ``(n_res, dim)`` per-residue scalar embeddings.
        rot : torch.Tensor
            Shape ``(n_res, 3, 3)`` per-residue frame rotation matrices.
        trans : torch.Tensor
            Shape ``(n_res, 3)`` per-residue frame translations.

        Returns
        -------
        torch.Tensor
            Shape ``(n_res, dim)`` updated scalar embeddings.
        """
        n = h.shape[0]
        q = self.q_scalar(h).view(n, self.n_heads, self.head_dim)
        k = self.k_scalar(h).view(n, self.n_heads, self.head_dim)
        v = self.v_scalar(h).view(n, self.n_heads, self.head_dim)
        scalar_logits = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(self.head_dim)

        qp = self.q_points(h).view(n, self.n_heads, self.n_points, 3)
        kp = self.k_points(h).view(n, self.n_heads, self.n_points, 3)
        vp = self.v_points(h).view(n, self.n_heads, self.n_points, 3)
        # local frame -> global frame
        qp_g = torch.einsum("ixy,ihpy->ihpx", rot, qp) + trans.view(n, 1, 1, 3)
        kp_g = torch.einsum("ixy,ihpy->ihpx", rot, kp) + trans.view(n, 1, 1, 3)
        vp_g = torch.einsum("ixy,ihpy->ihpx", rot, vp) + trans.view(n, 1, 1, 3)

        point_dist2 = ((qp_g.unsqueeze(1) - kp_g.unsqueeze(0)) ** 2).sum(-1).sum(-1)  # (n, n, h)
        gamma = torch.nn.functional.softplus(self.point_weight)
        point_logits = -0.5 * gamma * point_dist2

        attn = torch.softmax(scalar_logits + point_logits, dim=1)

        scalar_out = torch.einsum("ijh,jhd->ihd", attn, v).reshape(n, -1)
        point_out_g = torch.einsum("ijh,jhpx->ihpx", attn, vp_g)
        # global frame -> local (receiving) frame
        point_out_local = torch.einsum(
            "ixy,ihpx->ihpy", rot.transpose(-1, -2), point_out_g - trans.view(n, 1, 1, 3)
        )
        point_norm = torch.linalg.norm(point_out_local, dim=-1)
        combined = torch.cat(
            [scalar_out, point_out_local.reshape(n, -1), point_norm.reshape(n, -1)], dim=-1
        )
        return self.out_proj(combined)


class FoldFlowTrunk(nn.Module):
    """FoldFlow: flow-time-conditioned IPA trunk over unconditional SE(3) frames.

    A stack of :class:`InvariantPointAttention` layers, conditioned on a
    continuous flow-matching time embedding, consumes per-residue noisy
    frames (rotation + translation) and predicts a tangent vector field
    (rotation-update 6D representation + translation-update vector) for
    each residue -- the unconditional backbone-generation trunk shared by
    FoldFlow's SE(3) flow-matching objective.
    """

    def __init__(
        self, dim: int = 48, n_layers: int = 3, n_heads: int = 4, n_points: int = 4
    ) -> None:
        """Initialize the FoldFlow IPA trunk.

        Parameters
        ----------
        dim : int
            Per-residue scalar embedding dimension.
        n_layers : int
            Number of stacked IPA layers.
        n_heads : int
            Attention heads per IPA layer.
        n_points : int
            Query/key/value points per head.
        """
        super().__init__()
        self.node_embed = nn.Linear(1, dim)
        self.time_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(
            [InvariantPointAttention(dim, n_heads, n_points) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.rot_head = nn.Linear(dim, 6)
        self.trans_head = nn.Linear(dim, 3)

    def forward(
        self, rot: torch.Tensor, trans: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the SE(3) flow-matching vector field for noisy frames.

        Parameters
        ----------
        rot : torch.Tensor
            Shape ``(n_res, 3, 3)`` noisy per-residue frame rotations.
        trans : torch.Tensor
            Shape ``(n_res, 3)`` noisy per-residue frame translations.
        t : torch.Tensor
            Scalar flow-matching time in ``[0, 1]``, shape ``(1,)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(rot_update, trans_update)`` of shape ``(n_res, 3, 3)`` and
            ``(n_res, 3)``: predicted rotation-update matrices and
            translation-update vectors.
        """
        n = rot.shape[0]
        h = self.node_embed(torch.ones(n, 1, device=rot.device))
        h = h + self.time_embed(t.view(1, 1).float())
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h, rot, trans))
        rot_update = _rotation_from_6d(self.rot_head(h))
        trans_update = self.trans_head(h)
        return rot_update, trans_update


def build_foldflow() -> nn.Module:
    """Build a compact FoldFlow unconditional SE(3) IPA flow-matching trunk.

    Returns
    -------
    nn.Module
        FoldFlow reconstruction in evaluation mode.
    """
    return FoldFlowTrunk(dim=48, n_layers=3, n_heads=4, n_points=4).eval()


def example_input_foldflow() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_foldflow`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(rot, trans, t)`` random orthonormal frames for a 10-residue toy
        backbone plus a scalar flow-time.
    """
    n_res = 10
    rot = _rotation_from_6d(torch.randn(n_res, 6))
    trans = torch.randn(n_res, 3) * 3
    t = torch.tensor([0.3])
    return rot, trans, t


class FrameFlowTrunk(nn.Module):
    """FrameFlow: motif-mask-conditioned IPA trunk for scaffolded SE(3) frames.

    Shares :class:`FoldFlowTrunk`'s IPA-on-SE(3)-frames flow-matching
    design, but additionally consumes a per-residue binary motif mask and
    the fixed motif frames: motif-frame translations/rotation-features are
    gated by the mask and concatenated into every residue's input scalar
    embedding before the IPA stack, letting scaffold (non-motif) residues
    condition on the frozen motif residues within a single forward pass --
    the mechanism FrameFlow adds on top of the unconditional trunk.
    """

    def __init__(
        self, dim: int = 48, n_layers: int = 3, n_heads: int = 4, n_points: int = 4
    ) -> None:
        """Initialize the FrameFlow motif-conditioned IPA trunk.

        Parameters
        ----------
        dim : int
            Per-residue scalar embedding dimension.
        n_layers : int
            Number of stacked IPA layers.
        n_heads : int
            Attention heads per IPA layer.
        n_points : int
            Query/key/value points per head.
        """
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.motif_frame_embed = nn.Linear(12, dim)  # flattened 3x3 rot + 3 trans
        self.node_in = nn.Linear(dim * 2 + 1, dim)
        self.layers = nn.ModuleList(
            [InvariantPointAttention(dim, n_heads, n_points) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_layers)])
        self.rot_head = nn.Linear(dim, 6)
        self.trans_head = nn.Linear(dim, 3)

    def forward(
        self,
        rot: torch.Tensor,
        trans: torch.Tensor,
        t: torch.Tensor,
        motif_mask: torch.Tensor,
        motif_rot: torch.Tensor,
        motif_trans: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the motif-conditioned SE(3) flow-matching vector field.

        Parameters
        ----------
        rot : torch.Tensor
            Shape ``(n_res, 3, 3)`` noisy per-residue frame rotations.
        trans : torch.Tensor
            Shape ``(n_res, 3)`` noisy per-residue frame translations.
        t : torch.Tensor
            Scalar flow-matching time in ``[0, 1]``, shape ``(1,)``.
        motif_mask : torch.Tensor
            Shape ``(n_res,)`` binary indicator of fixed motif residues.
        motif_rot : torch.Tensor
            Shape ``(n_res, 3, 3)`` fixed motif frame rotations (ignored
            where ``motif_mask`` is 0).
        motif_trans : torch.Tensor
            Shape ``(n_res, 3)`` fixed motif frame translations (ignored
            where ``motif_mask`` is 0).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(rot_update, trans_update)`` of shape ``(n_res, 3, 3)`` and
            ``(n_res, 3)``.
        """
        n = rot.shape[0]
        motif_feat = torch.cat([motif_rot.reshape(n, 9), motif_trans], dim=-1)
        motif_embed = self.motif_frame_embed(motif_feat) * motif_mask.view(n, 1)
        t_embed = self.time_embed(t.view(1, 1).float()).expand(n, -1)
        h = self.node_in(torch.cat([t_embed, motif_embed, motif_mask.view(n, 1)], dim=-1))
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h, rot, trans))
        rot_update = _rotation_from_6d(self.rot_head(h))
        trans_update = self.trans_head(h)
        return rot_update, trans_update


def build_frameflow() -> nn.Module:
    """Build a compact FrameFlow motif-conditioned SE(3) IPA flow-matching trunk.

    Returns
    -------
    nn.Module
        FrameFlow reconstruction in evaluation mode.
    """
    return FrameFlowTrunk(dim=48, n_layers=3, n_heads=4, n_points=4).eval()


def example_input_frameflow() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Create example input for :func:`build_frameflow`.

    Returns
    -------
    tuple of torch.Tensor
        ``(rot, trans, t, motif_mask, motif_rot, motif_trans)`` for a
        10-residue toy backbone where the first 3 residues are a fixed
        motif.
    """
    n_res = 10
    rot = _rotation_from_6d(torch.randn(n_res, 6))
    trans = torch.randn(n_res, 3) * 3
    t = torch.tensor([0.5])
    motif_mask = torch.zeros(n_res)
    motif_mask[:3] = 1.0
    motif_rot = _rotation_from_6d(torch.randn(n_res, 6))
    motif_trans = torch.randn(n_res, 3) * 3
    return rot, trans, t, motif_mask, motif_rot, motif_trans


# ---------------------------------------------------------------------------
# GCPNet: Geometry-Complete Perceptron -- frame-scalarized scalar/vector GNN.
# ---------------------------------------------------------------------------
class GeometryCompletePerceptron(nn.Module):
    """One Geometry-Complete Perceptron (GCP) layer.

    Scalar and vector per-node features are updated jointly: the vector
    stream is first bottlenecked and its norm is merged into the scalar
    update (as in a GVP-style layer), but GCPNet additionally builds a
    **local geometric frame** for each node from its neighbor coordinates
    and *scalarizes* the updated vector features against that frame
    (projecting them onto the frame axes), feeding those frame-relative
    invariant scalars back into a second scalar update -- the extra step
    that makes the layer information-complete ("geometry-complete") rather
    than merely norm-gated. The vector output is gated by the (frame-aware)
    scalar stream, preserving SE(3) equivariance.
    """

    def __init__(self, scalar_dim: int, vector_dim: int, hidden_dim: int | None = None) -> None:
        """Initialize a GCP layer.

        Parameters
        ----------
        scalar_dim : int
            Scalar feature dimension (input and output).
        vector_dim : int
            Vector-channel count per node (each a 3-vector).
        hidden_dim : int, optional
            Vector bottleneck width; defaults to ``vector_dim``.
        """
        super().__init__()
        hidden_dim = hidden_dim or vector_dim
        self.vector_down = nn.Linear(vector_dim, hidden_dim, bias=False)
        self.scalar_out = nn.Linear(scalar_dim + hidden_dim, scalar_dim)
        self.vector_up = nn.Linear(hidden_dim, vector_dim, bias=False)
        self.vector_gate = nn.Linear(scalar_dim, vector_dim)

        # frame-scalarization stage: 3 frame axes -> 3 extra invariant scalars per channel
        self.scalar_out_frame = nn.Linear(scalar_dim + vector_dim * 3, scalar_dim)

    def _local_frame(self, coords: torch.Tensor) -> torch.Tensor:
        """Build an orthonormal local frame per node from neighbor geometry.

        Parameters
        ----------
        coords : torch.Tensor
            Shape ``(n_nodes, 3)`` node coordinates.

        Returns
        -------
        torch.Tensor
            Shape ``(n_nodes, 3, 3)`` per-node orthonormal frame axes,
            built from the centroid-relative and nearest-neighbor-relative
            directions (Gram-Schmidt).
        """
        centroid = coords.mean(dim=0, keepdim=True)
        e1 = torch.nn.functional.normalize(coords - centroid, dim=-1)
        rel = coords.unsqueeze(1) - coords.unsqueeze(0)  # (n, n, 3)
        dist = (
            torch.linalg.norm(rel, dim=-1) + torch.eye(coords.shape[0], device=coords.device) * 1e6
        )
        nearest = torch.argmin(dist, dim=1)
        raw_e2 = coords[nearest] - coords
        e2_proj = raw_e2 - (e1 * raw_e2).sum(-1, keepdim=True) * e1
        e2 = torch.nn.functional.normalize(e2_proj, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=1)  # (n, 3, 3)

    def forward(
        self, scalar_rep: torch.Tensor, vector_rep: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Jointly update scalar and vector node features.

        Parameters
        ----------
        scalar_rep : torch.Tensor
            Shape ``(n_nodes, scalar_dim)`` scalar node features.
        vector_rep : torch.Tensor
            Shape ``(n_nodes, vector_dim, 3)`` vector node features.
        coords : torch.Tensor
            Shape ``(n_nodes, 3)`` node coordinates (used to build the
            local geometric frame; not itself updated).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(scalar_rep, vector_rep)`` with the same shapes.
        """
        v_pre = vector_rep.transpose(-1, -2)  # (n, 3, vector_dim)
        vector_hidden = self.vector_down(v_pre)  # (n, 3, hidden_dim)
        vector_norm = torch.linalg.norm(vector_hidden, dim=1)  # (n, hidden_dim)

        merged = torch.cat([scalar_rep, vector_norm], dim=-1)
        scalar_mid = torch.relu(self.scalar_out(merged))

        vector_out = self.vector_up(vector_hidden).transpose(-1, -2)  # (n, vector_dim, 3)

        # Geometry-completeness step: scalarize the updated vectors against a
        # local frame built from neighbor geometry.
        frame = self._local_frame(coords)  # (n, 3, 3)
        scalarized = torch.einsum("nfc,nvc->nvf", frame, vector_out)  # (n, vector_dim, 3)
        scalar_final = self.scalar_out_frame(
            torch.cat([scalar_mid, scalarized.reshape(scalarized.shape[0], -1)], dim=-1)
        )

        gate = torch.sigmoid(self.vector_gate(scalar_final)).unsqueeze(-1)
        vector_final = vector_out * gate
        return torch.relu(scalar_final), vector_final


class GCPNet(nn.Module):
    """GCPNet: stacked Geometry-Complete Perceptron message-passing network.

    Embeds scalar (e.g. atom-type) and vector (e.g. bond-direction) node
    features, refines them through a stack of
    :class:`GeometryCompletePerceptron` layers conditioned on the node
    point cloud's geometry, and reads out a per-node scalar property.
    """

    def __init__(
        self,
        scalar_in: int = 16,
        vector_in: int = 3,
        scalar_dim: int = 32,
        vector_dim: int = 8,
        n_layers: int = 3,
    ) -> None:
        """Initialize GCPNet.

        Parameters
        ----------
        scalar_in : int
            Input scalar feature dimension (e.g. one-hot atom type).
        vector_in : int
            Input vector-channel count per node.
        scalar_dim : int
            Hidden scalar feature dimension.
        vector_dim : int
            Hidden vector-channel count per node.
        n_layers : int
            Number of stacked GCP layers.
        """
        super().__init__()
        self.scalar_embed = nn.Linear(scalar_in, scalar_dim)
        self.vector_embed = nn.Linear(vector_in, vector_dim, bias=False)
        self.layers = nn.ModuleList(
            [GeometryCompletePerceptron(scalar_dim, vector_dim) for _ in range(n_layers)]
        )
        self.readout = nn.Linear(scalar_dim, 1)

    def forward(
        self, scalar_feat: torch.Tensor, vector_feat: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Predict a per-node scalar property from geometry-complete features.

        Parameters
        ----------
        scalar_feat : torch.Tensor
            Shape ``(n_nodes, scalar_in)`` input scalar node features.
        vector_feat : torch.Tensor
            Shape ``(n_nodes, vector_in, 3)`` input vector node features.
        coords : torch.Tensor
            Shape ``(n_nodes, 3)`` node coordinates.

        Returns
        -------
        torch.Tensor
            Shape ``(n_nodes,)`` predicted per-node scalar property.
        """
        scalar_rep = self.scalar_embed(scalar_feat)
        vector_rep = self.vector_embed(vector_feat.transpose(-1, -2)).transpose(-1, -2)
        for layer in self.layers:
            scalar_rep, vector_rep = layer(scalar_rep, vector_rep, coords)
        return self.readout(scalar_rep).squeeze(-1)


def build_gcpnet() -> nn.Module:
    """Build a compact GCPNet geometry-complete message-passing network.

    Returns
    -------
    nn.Module
        GCPNet reconstruction in evaluation mode.
    """
    return GCPNet(scalar_in=16, vector_in=3, scalar_dim=32, vector_dim=8, n_layers=3).eval()


def example_input_gcpnet() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create example input for :func:`build_gcpnet`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(scalar_feat, vector_feat, coords)`` for a 14-node toy point
        cloud (e.g. a small protein-residue graph).
    """
    n_nodes = 14
    scalar_feat = torch.eye(16)[torch.randint(0, 16, (n_nodes,))]
    vector_feat = torch.randn(n_nodes, 3, 3)
    coords = torch.randn(n_nodes, 3) * 5
    return scalar_feat, vector_feat, coords


MENAGERIE_ENTRIES = [
    ("EvoDiff", "build_evodiff", "example_input_evodiff", "2023", "BIO"),
    ("FABind", "build_fabind", "example_input_fabind", "2023", "BIO"),
    ("FABFlex", "build_fabflex", "example_input_fabflex", "2025", "BIO"),
    ("FoldFlow", "build_foldflow", "example_input_foldflow", "2023", "BIO"),
    ("FrameFlow", "build_frameflow", "example_input_frameflow", "2023", "BIO"),
    ("GCPNet", "build_gcpnet", "example_input_gcpnet", "2024", "BIO"),
]
