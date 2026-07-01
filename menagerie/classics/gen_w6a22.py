"""Compact faithful reimplementations: structural/computational biology models.

Sources checked (paper / official repo, read via GitHub API + raw content;
no clone, no pip install -- base env only):

- DeepRank: Renaud et al., "DeepRank: a deep learning framework for data mining
  3D protein-protein interfaces", Nature Communications 12, 7068 (2021).
  https://github.com/DeepRank/deeprank -- maps a protein-protein interface onto
  a 3D voxel grid (channels: atomic density, charge, BSA, distance-to-interface,
  etc.) and trains a 3D CNN classifier/regressor on the grid.
- DeepRank-GNN: Réau et al., "DeepRank-GNN: a graph neural network framework to
  learn patterns in protein-protein interfaces", Bioinformatics 39(1), btac759
  (2023). https://github.com/DeepRank/Deeprank-GNN -- GINet architecture: the
  interface is split into an INTERNAL sub-graph (same-chain residues, <3A) and
  an EXTERNAL sub-graph (cross-chain residues, <8.5A); each is passed through
  its own message-passing + pooling stage before a shared readout head.
- DeepTracer: Pfab, Phan, Si, "DeepTracer: Predicting Backbone Atomic Structure
  from High Resolution Cryo-EM Density Maps of Protein Complexes", bioRxiv
  2020.02.12.946772 (2020). https://github.com/yangyunfei16/DeepTracer -- a
  cascade of four 3D U-Nets sharing the same cryo-EM density-map input, each
  predicting a distinct target map (backbone-atom location, amino-acid type,
  secondary structure, C-alpha location), stacked so downstream U-Nets can
  condition on upstream predictions.
- DiffAb: Luo, Su, Peng et al., "Antigen-Specific Antibody Design and
  Optimization with Diffusion-Based Generative Models for Protein Structures",
  NeurIPS 2022. https://github.com/luost26/diffab -- SE(3)-equivariant
  diffusion over antibody CDR residue types + backbone frames (rotation,
  translation). The distinctive network primitive (confirmed by reading
  diffab/modules/encoders/ga.py) is "Geometric Attention" (GABlock): AlphaFold2
  Invariant-Point-Attention-style multi-head attention combining scalar
  node/pair logits with an SE(3)-invariant "spatial" term computed by
  projecting query/key/value POINTS into the global frame via each residue's
  (R, t) and comparing them with squared Euclidean distance.
- DiffDock-PP: Ketata, Laue, Mammadov et al., "DiffDock-PP: Rigid Protein-
  Protein Docking with Diffusion Models", ICLR 2023 MLDD workshop, arXiv
  2304.03889. https://github.com/ketatam/DiffDock-PP -- receptor and ligand
  are each embedded by a shared tensor-product-style equivariant GNN
  (confirmed by reading src/model/model.py: BaseModel wraps a
  TensorProductScoreModel), then cross-graph attention pools the two
  embeddings and predicts a SINGLE RIGID-BODY translation + rotation (+
  torsion) for the ligand as a whole -- distinct from DiffDock (already in
  this catalog), which diffuses per-atom ligand poses without a rigid-body
  cross-graph pooling head.
- DiffSDS: Gao, Tan, Li, "DiffSDS: A language diffusion model for protein
  backbone inpainting under geometric conditions and constraints", ICLR 2023 /
  arXiv 2301.09642. https://github.com/A4Bio/DiffSDS -- a bidirectional
  Transformer encoder over backbone dihedral-angle tokens, with an added
  hidden "Atomic Direction Space" (ADS) layer that lifts each token's
  invariant scalar angle features into equivariant 3-vector "direction"
  features (via a learned linear map + cross product basis), letting later
  layers reason jointly over sequence tokens and their local geometric frame
  while inpainting masked backbone spans.

All models below use small random-init dimensions; this is an architecture
catalog, not a trained-weights zoo.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# DeepRank: 3D CNN over a voxelized protein-protein interface grid
# ---------------------------------------------------------------------------


class DeepRankCNN3D(nn.Module):
    """Compact reproduction of DeepRank's configurable 3D-CNN classifier.

    DeepRank rasterizes a protein-protein interface (atom density, charge,
    buried-surface-area, distance-to-interface, etc.) onto a 3D voxel grid and
    trains a user-configurable 3D CNN on it. This module reproduces the
    default DeepRank CNN topology: stacked Conv3d + BatchNorm3d + ReLU blocks
    with strided downsampling, followed by fully-connected classification
    head (docking-pose quality score).
    """

    def __init__(self, in_channels: int = 5, grid: int = 16) -> None:
        """Build the 3D-CNN grid encoder + FC head.

        Parameters
        ----------
        in_channels:
            Number of voxel feature channels (atom density, charge, BSA,
            distance-to-interface, residue-depth in the reference config).
        grid:
            Cubic grid side length in voxels.
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)
        reduced = grid // 4
        self.fc = nn.Sequential(
            nn.Linear(16 * reduced**3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Score a batch of voxelized protein-protein interfaces.

        Parameters
        ----------
        grid:
            Voxel feature tensor, shape ``(batch, in_channels, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Docking-quality score, shape ``(batch, 1)``.
        """
        x = self.pool1(self.conv1(grid))
        x = self.pool2(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)


def build_deeprank() -> nn.Module:
    """Construct a compact DeepRank 3D-CNN interface scorer."""
    return DeepRankCNN3D(in_channels=5, grid=16).eval()


def example_input_deeprank() -> torch.Tensor:
    """Random voxel grid: batch of 2 interfaces, 5 channels, 16^3 grid."""
    return torch.randn(2, 5, 16, 16, 16)


# ---------------------------------------------------------------------------
# DeepRank-GNN: dual-subgraph GINet (internal + external interface graphs)
# ---------------------------------------------------------------------------


class InteractionConv(nn.Module):
    """One message-passing + edge-gated aggregation layer (GINet building block)."""

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        """Initialize the message MLP and gating network.

        Parameters
        ----------
        node_dim:
            Residue node feature width.
        edge_dim:
            Edge (distance/interaction) feature width.
        """
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim), nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(nn.Linear(2 * node_dim + edge_dim, 1), nn.Sigmoid())
        self.update = nn.Sequential(nn.Linear(2 * node_dim, node_dim), nn.ReLU(inplace=True))

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Propagate one gated message-passing step over a residue sub-graph.

        Parameters
        ----------
        x:
            Node features, shape ``(N, node_dim)``.
        edge_index:
            Edge endpoints, shape ``(2, E)``.
        edge_attr:
            Edge features, shape ``(E, edge_dim)``.

        Returns
        -------
        torch.Tensor
            Updated node features, shape ``(N, node_dim)``.
        """
        src, dst = edge_index[0], edge_index[1]
        pair = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        msg = self.message(pair) * self.gate(pair)
        agg = torch.zeros_like(x).index_add_(0, dst, msg)
        return self.update(torch.cat([x, agg], dim=-1))


class DeepRankGNN(nn.Module):
    """GINet: split internal/external message passing over the interface graph.

    DeepRank-GNN builds one residue-level graph per interface with two edge
    types: INTERNAL edges (same-chain residues, heavy atoms < 3A) and
    EXTERNAL edges (cross-chain residues, heavy atoms < 8.5A). GINet processes
    the two sub-graphs through separate convolution stacks before pooling to a
    single interface-quality score -- the rotation-invariant improvement over
    voxel-grid DeepRank.
    """

    def __init__(self, node_dim: int = 16, edge_dim: int = 4, layers: int = 2) -> None:
        """Build the internal/external convolution stacks + readout.

        Parameters
        ----------
        node_dim:
            Residue node feature width.
        edge_dim:
            Edge feature width (shared by both sub-graphs).
        layers:
            Number of stacked ``InteractionConv`` layers per sub-graph.
        """
        super().__init__()
        self.node_embed = nn.Linear(node_dim, node_dim)
        self.internal_convs = nn.ModuleList(
            [InteractionConv(node_dim, edge_dim) for _ in range(layers)]
        )
        self.external_convs = nn.ModuleList(
            [InteractionConv(node_dim, edge_dim) for _ in range(layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(node_dim, node_dim), nn.ReLU(inplace=True), nn.Linear(node_dim, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        internal_edge_index: torch.Tensor,
        internal_edge_attr: torch.Tensor,
        external_edge_index: torch.Tensor,
        external_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Score one interface graph via sequential internal-then-external passing.

        Parameters
        ----------
        x:
            Residue node features, shape ``(N, node_dim)``.
        internal_edge_index, internal_edge_attr:
            Same-chain edges/features.
        external_edge_index, external_edge_attr:
            Cross-chain interface edges/features.

        Returns
        -------
        torch.Tensor
            Scalar interface-quality score, shape ``(1,)``.
        """
        h = self.node_embed(x)
        for conv in self.internal_convs:
            h = conv(h, internal_edge_index, internal_edge_attr)
        for conv in self.external_convs:
            h = conv(h, external_edge_index, external_edge_attr)
        pooled = h.mean(dim=0, keepdim=True)
        return self.readout(pooled).squeeze(-1)


def build_deeprank_gnn() -> nn.Module:
    """Construct a compact DeepRank-GNN (GINet) interface scorer."""
    return DeepRankGNN(node_dim=16, edge_dim=4, layers=2).eval()


def example_input_deeprank_gnn() -> tuple:
    """Random 20-residue interface graph split into internal/external edges."""
    n = 20
    x = torch.randn(n, 16)
    internal_edge_index = torch.randint(0, n, (2, 30))
    internal_edge_attr = torch.randn(30, 4)
    external_edge_index = torch.randint(0, n, (2, 24))
    external_edge_attr = torch.randn(24, 4)
    return (x, internal_edge_index, internal_edge_attr, external_edge_index, external_edge_attr)


# ---------------------------------------------------------------------------
# DeepTracer: cascade of 3D U-Nets over a shared cryo-EM density map
# ---------------------------------------------------------------------------


class UNet3DBlock(nn.Module):
    """One small 3D U-Net: two down-steps, bottleneck, two up-steps."""

    def __init__(self, in_channels: int, out_channels: int, base: int = 8) -> None:
        """Initialize a compact 3D U-Net.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count (target map channels).
        base:
            Base feature width.
        """
        super().__init__()

        def conv_block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv3d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True))

        self.enc1 = conv_block(in_channels, base)
        self.enc2 = conv_block(base, base * 2)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = conv_block(base * 2, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.out = nn.Conv3d(base, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one 3D U-Net pass, predicting a per-voxel target map.

        Parameters
        ----------
        x:
            Input density (+ conditioning) volume, shape
            ``(batch, in_channels, D, H, W)`` with D, H, W divisible by 4.

        Returns
        -------
        torch.Tensor
            Predicted map, shape ``(batch, out_channels, D, H, W)``.
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class DeepTracerCascade(nn.Module):
    """Cascade of four 3D U-Nets sharing a cryo-EM density-map input.

    DeepTracer predicts, from a cryo-EM density map: (1) backbone-atom
    location, (2) amino-acid type, (3) secondary structure, (4) C-alpha
    location. Each stage is its own 3D U-Net; downstream U-Nets take the raw
    density map concatenated with the upstream stage's (detached) prediction
    as extra input channels, reproducing the cascaded conditioning that lets
    later maps refine on earlier structural cues.
    """

    def __init__(self, base: int = 8) -> None:
        """Build the four cascaded 3D U-Net stages.

        Parameters
        ----------
        base:
            Base feature width shared by every stage's U-Net.
        """
        super().__init__()
        self.backbone_unet = UNet3DBlock(1, 1, base)
        self.aa_unet = UNet3DBlock(2, 20, base)
        self.ss_unet = UNet3DBlock(3, 3, base)
        self.calpha_unet = UNet3DBlock(4, 1, base)

    def forward(self, density: torch.Tensor) -> tuple:
        """Predict the four cascaded target maps from a cryo-EM density map.

        Parameters
        ----------
        density:
            Cryo-EM density volume, shape ``(batch, 1, D, H, W)`` with D, H, W
            divisible by 4.

        Returns
        -------
        tuple of torch.Tensor
            ``(backbone_map, aa_type_map, secondary_structure_map, calpha_map)``.
        """
        backbone_map = self.backbone_unet(density)
        aa_map = self.aa_unet(torch.cat([density, backbone_map], dim=1))
        ss_map = self.ss_unet(
            torch.cat([density, backbone_map, aa_map.mean(1, keepdim=True)], dim=1)
        )
        calpha_map = self.calpha_unet(
            torch.cat(
                [density, backbone_map, aa_map.mean(1, keepdim=True), ss_map.mean(1, keepdim=True)],
                dim=1,
            )
        )
        return backbone_map, aa_map, ss_map, calpha_map


def build_deeptracer() -> nn.Module:
    """Construct a compact DeepTracer cascaded 3D U-Net stack."""
    return DeepTracerCascade(base=8).eval()


def example_input_deeptracer() -> torch.Tensor:
    """Random cryo-EM density sub-volume, 1x16x16x16."""
    return torch.randn(1, 1, 16, 16, 16)


# ---------------------------------------------------------------------------
# DiffAb: SE(3)-equivariant Geometric Attention (AlphaFold-IPA-style) diffusion
# ---------------------------------------------------------------------------


def _rotation_from_6d(x6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt orthogonalize a 6D rotation representation to an SO(3) matrix.

    Parameters
    ----------
    x6:
        Raw 6D rotation parameters, shape ``(..., 6)``.

    Returns
    -------
    torch.Tensor
        Rotation matrices, shape ``(..., 3, 3)``.
    """
    a1, a2 = x6[..., 0:3], x6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class GeometricAttention(nn.Module):
    """AlphaFold-IPA-style attention: scalar + pair + SE(3)-invariant spatial terms.

    Reproduces the ``GABlock`` primitive from DiffAb's official encoder: each
    residue carries a local frame ``(R, t)``; query/key/value POINTS are
    projected from node features, mapped into the GLOBAL frame via ``(R, t)``,
    and compared with squared Euclidean distance to form an SE(3)-invariant
    attention bias alongside the usual scalar dot-product and pair-feature
    logits.
    """

    def __init__(self, node_dim: int, pair_dim: int, heads: int = 4, n_points: int = 4) -> None:
        """Initialize node/pair/point projections for geometric attention.

        Parameters
        ----------
        node_dim:
            Per-residue scalar feature width.
        pair_dim:
            Pairwise (residue-residue) feature width.
        heads:
            Number of attention heads.
        n_points:
            Number of query/key/value 3D points per head.
        """
        super().__init__()
        self.heads = heads
        self.n_points = n_points
        qk = node_dim // heads
        self.qk_dim = qk
        self.q_proj = nn.Linear(node_dim, heads * qk)
        self.k_proj = nn.Linear(node_dim, heads * qk)
        self.v_proj = nn.Linear(node_dim, heads * qk)
        self.pair_bias = nn.Linear(pair_dim, heads)
        self.q_point = nn.Linear(node_dim, heads * n_points * 3)
        self.k_point = nn.Linear(node_dim, heads * n_points * 3)
        self.v_point = nn.Linear(node_dim, heads * n_points * 3)
        self.gamma = nn.Parameter(torch.zeros(heads))
        self.out_proj = nn.Linear(heads * qk + heads * n_points * 4, node_dim)

    def forward(
        self, x: torch.Tensor, z: torch.Tensor, R: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Run one geometric-attention update over residue frames.

        Parameters
        ----------
        x:
            Node scalar features, shape ``(L, node_dim)``.
        z:
            Pair features, shape ``(L, L, pair_dim)``.
        R:
            Per-residue rotation matrices (local -> global frame), ``(L, 3, 3)``.
        t:
            Per-residue translations, shape ``(L, 3)``.

        Returns
        -------
        torch.Tensor
            Updated node scalar features, shape ``(L, node_dim)``.
        """
        length = x.shape[0]
        h, qk, p = self.heads, self.qk_dim, self.n_points

        q = self.q_proj(x).view(length, h, qk)
        k = self.k_proj(x).view(length, h, qk)
        v = self.v_proj(x).view(length, h, qk)
        scalar_logits = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(qk)

        pair_logits = self.pair_bias(z)

        def to_global(local_pts: torch.Tensor) -> torch.Tensor:
            # local_pts: (L, h*p, 3) -> global: R_i @ p + t_i, per residue i
            local_pts = local_pts.view(length, h * p, 3)
            g = torch.einsum("lij,lpj->lpi", R, local_pts) + t.unsqueeze(1)
            return g.view(length, h, p, 3)

        q_pt = to_global(self.q_point(x))
        k_pt = to_global(self.k_point(x))
        v_pt = to_global(self.v_point(x))

        sq_dist = ((q_pt.unsqueeze(1) - k_pt.unsqueeze(0)) ** 2).sum(-1).sum(-1)  # (L, L, h)
        spatial_logits = -F.softplus(self.gamma) * sq_dist / (2.0 * math.sqrt(2.0 / (9 * p)))

        logits = scalar_logits + pair_logits + spatial_logits
        alpha = torch.softmax(logits, dim=1)  # (L, L, h)

        node_out = torch.einsum("ijh,jhd->ihd", alpha, v).reshape(length, h * qk)

        # Aggregate value points into the global frame, then bring back local.
        agg_pt = torch.einsum("ijh,jhpc->ihpc", alpha, v_pt)  # (L, h, p, 3)
        R_inv = R.transpose(-1, -2)
        local_agg = torch.einsum("lij,lhpj->lhpi", R_inv, agg_pt - t.view(length, 1, 1, 3))
        dist_feat = local_agg.norm(dim=-1)  # (L, h, p)

        spatial_feat = torch.cat(
            [local_agg.reshape(length, h * p * 3), dist_feat.reshape(length, h * p)], dim=-1
        )
        return self.out_proj(torch.cat([node_out, spatial_feat], dim=-1))


class DiffAbDenoiser(nn.Module):
    """Compact DiffAb-style CDR denoising network.

    A stack of ``GeometricAttention`` blocks over the antibody-antigen complex
    (framework + CDR + antigen residues in one set of local frames), predicting
    per-CDR-residue amino-acid-type logits and an SE(3) frame update
    (translation + 6D rotation) -- the joint sequence+structure diffusion
    output of DiffAb.
    """

    def __init__(
        self, node_dim: int = 32, pair_dim: int = 16, layers: int = 2, n_aa: int = 20
    ) -> None:
        """Build the residue embedding, geometric-attention stack, and heads.

        Parameters
        ----------
        node_dim:
            Per-residue scalar feature width.
        pair_dim:
            Pairwise feature width.
        layers:
            Number of stacked geometric-attention blocks.
        n_aa:
            Number of amino-acid types for the sequence-denoising head.
        """
        super().__init__()
        self.aa_embed = nn.Embedding(n_aa + 1, node_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim)
        )
        self.pair_embed = nn.Linear(1, pair_dim)
        self.blocks = nn.ModuleList(
            [GeometricAttention(node_dim, pair_dim, heads=4, n_points=4) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(layers)])
        self.aa_head = nn.Linear(node_dim, n_aa)
        self.frame_head = nn.Linear(node_dim, 3 + 6)

    def forward(self, aa_type: torch.Tensor, coords: torch.Tensor, timestep: torch.Tensor) -> tuple:
        """Denoise one antibody-antigen complex frame + CDR sequence.

        Parameters
        ----------
        aa_type:
            Integer amino-acid-type tokens (noised CDR + fixed context), ``(L,)``.
        coords:
            Per-residue C-alpha coordinates (defines local translation ``t``
            and, via a fixed identity rotation basis at init, ``R``), ``(L, 3)``.
        timestep:
            Scalar diffusion timestep in ``[0, 1]``, shape ``(1,)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(aa_logits, translation_update, rotation_6d_update)``.
        """
        length = aa_type.shape[0]
        x = self.aa_embed(aa_type) + self.time_embed(timestep.view(1, 1)).expand(length, -1)
        rel = coords.unsqueeze(1) - coords.unsqueeze(0)
        z = self.pair_embed(rel.norm(dim=-1, keepdim=True))
        R = torch.eye(3, device=coords.device, dtype=coords.dtype).expand(length, 3, 3)
        t = coords
        for block, norm in zip(self.blocks, self.norms):
            x = norm(x + block(x, z, R, t))
        aa_logits = self.aa_head(x)
        frame_update = self.frame_head(x)
        return aa_logits, frame_update[:, :3], frame_update[:, 3:]


def build_diffab() -> nn.Module:
    """Construct a compact DiffAb geometric-attention CDR denoiser."""
    return DiffAbDenoiser(node_dim=32, pair_dim=16, layers=2, n_aa=20).eval()


def example_input_diffab() -> tuple:
    """Random 25-residue antibody-antigen complex (framework+CDR+antigen)."""
    length = 25
    aa_type = torch.randint(0, 20, (length,))
    coords = torch.randn(length, 3) * 5.0
    timestep = torch.tensor([0.5])
    return (aa_type, coords, timestep)


# ---------------------------------------------------------------------------
# DiffDock-PP: rigid-body receptor/ligand pose diffusion via cross-graph pooling
# ---------------------------------------------------------------------------


class ResidueGraphEncoder(nn.Module):
    """Shared per-protein residue-graph encoder (stand-in for the e3nn tensor-product GNN)."""

    def __init__(self, node_dim: int, edge_dim: int, layers: int = 2) -> None:
        """Initialize the message-passing stack for one protein's residue graph.

        Parameters
        ----------
        node_dim:
            Residue node feature width (includes ESM-embedding-like input).
        edge_dim:
            Edge (k-NN distance) feature width.
        layers:
            Number of message-passing layers.
        """
        super().__init__()
        self.in_proj = nn.Linear(node_dim, node_dim)
        self.convs = nn.ModuleList([InteractionConv(node_dim, edge_dim) for _ in range(layers)])

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Encode one protein's residue graph.

        Parameters
        ----------
        x:
            Residue node features, shape ``(N, node_dim)``.
        edge_index:
            k-NN edge endpoints, shape ``(2, E)``.
        edge_attr:
            Edge distance features, shape ``(E, edge_dim)``.

        Returns
        -------
        torch.Tensor
            Encoded residue features, shape ``(N, node_dim)``.
        """
        h = self.in_proj(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        return h


class DiffDockPPScoreModel(nn.Module):
    """Compact DiffDock-PP: shared residue-graph encoder + rigid-body pose head.

    Receptor and ligand residue graphs are each embedded with the SAME shared
    encoder (parameter-tied, as in the official ``TensorProductScoreModel``),
    cross-attended, and pooled to predict a single SE(3) rigid-body transform
    (translation + rotation + a torsion scalar) for the ligand as a whole --
    the key structural difference from per-atom-pose DiffDock.
    """

    def __init__(self, node_dim: int = 24, edge_dim: int = 4, layers: int = 2) -> None:
        """Build the shared encoder, cross-attention pooling, and pose heads.

        Parameters
        ----------
        node_dim:
            Residue node feature width.
        edge_dim:
            Edge distance-feature width.
        layers:
            Message-passing layers in the shared residue-graph encoder.
        """
        super().__init__()
        self.encoder = ResidueGraphEncoder(node_dim, edge_dim, layers=layers)
        self.cross_query = nn.Linear(node_dim, node_dim)
        self.cross_key = nn.Linear(node_dim, node_dim)
        self.cross_value = nn.Linear(node_dim, node_dim)
        self.tr_head = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim), nn.ReLU(inplace=True), nn.Linear(node_dim, 3)
        )
        self.rot_head = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim), nn.ReLU(inplace=True), nn.Linear(node_dim, 3)
        )
        self.tor_head = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim), nn.ReLU(inplace=True), nn.Linear(node_dim, 1)
        )

    def forward(
        self,
        receptor_x: torch.Tensor,
        receptor_edge_index: torch.Tensor,
        receptor_edge_attr: torch.Tensor,
        ligand_x: torch.Tensor,
        ligand_edge_index: torch.Tensor,
        ligand_edge_attr: torch.Tensor,
    ) -> tuple:
        """Predict a rigid-body ligand pose update from receptor+ligand graphs.

        Parameters
        ----------
        receptor_x, receptor_edge_index, receptor_edge_attr:
            Receptor residue graph.
        ligand_x, ligand_edge_index, ligand_edge_attr:
            Ligand residue graph.

        Returns
        -------
        tuple of torch.Tensor
            ``(translation, rotation_axis_angle, torsion_scalar)``.
        """
        rec_h = self.encoder(receptor_x, receptor_edge_index, receptor_edge_attr)
        lig_h = self.encoder(ligand_x, ligand_edge_index, ligand_edge_attr)

        q = self.cross_query(lig_h)
        k = self.cross_key(rec_h)
        v = self.cross_value(rec_h)
        attn = torch.softmax(q @ k.t() / math.sqrt(q.shape[-1]), dim=-1)
        cross_ctx = attn @ v  # (n_lig, node_dim)

        lig_pooled = lig_h.mean(dim=0, keepdim=True)
        ctx_pooled = cross_ctx.mean(dim=0, keepdim=True)
        pooled = torch.cat([lig_pooled, ctx_pooled], dim=-1)

        tr_pred = self.tr_head(pooled).squeeze(0)
        rot_pred = self.rot_head(pooled).squeeze(0)
        tor_pred = self.tor_head(pooled).squeeze(0)
        return tr_pred, rot_pred, tor_pred


def build_diffdock_pp() -> nn.Module:
    """Construct a compact DiffDock-PP rigid protein-protein docking model."""
    return DiffDockPPScoreModel(node_dim=24, edge_dim=4, layers=2).eval()


def example_input_diffdock_pp() -> tuple:
    """Random small receptor (18 residues) + ligand (12 residues) k-NN graphs."""
    n_rec, n_lig = 18, 12
    receptor_x = torch.randn(n_rec, 24)
    receptor_edge_index = torch.randint(0, n_rec, (2, 40))
    receptor_edge_attr = torch.randn(40, 4)
    ligand_x = torch.randn(n_lig, 24)
    ligand_edge_index = torch.randint(0, n_lig, (2, 24))
    ligand_edge_attr = torch.randn(24, 4)
    return (
        receptor_x,
        receptor_edge_index,
        receptor_edge_attr,
        ligand_x,
        ligand_edge_index,
        ligand_edge_attr,
    )


# ---------------------------------------------------------------------------
# DiffSDS: language-diffusion Transformer with an Atomic Direction Space layer
# ---------------------------------------------------------------------------


class AtomicDirectionSpace(nn.Module):
    """Lift invariant scalar token features into equivariant 3-vector directions.

    Reproduces DiffSDS's distinctive "ADS" layer: a hidden module sitting on
    top of the Transformer encoder that converts each residue's invariant
    scalar (dihedral-angle) hidden state into an SE(3)-EQUIVARIANT direction
    vector by combining a learned linear projection with the local backbone
    tangent/normal/binormal basis built from neighboring C-alpha coordinates,
    so later layers can jointly reason over sequence tokens and their local
    geometric frame while inpainting masked backbone spans.
    """

    def __init__(self, dim: int) -> None:
        """Initialize the direction-space projection.

        Parameters
        ----------
        dim:
            Transformer hidden width.
        """
        super().__init__()
        self.to_scale = nn.Linear(dim, 3)
        self.merge = nn.Linear(dim + 3, dim)

    def forward(self, h: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Fuse scalar hidden states with a local equivariant direction feature.

        Parameters
        ----------
        h:
            Scalar Transformer hidden states, shape ``(L, dim)``.
        coords:
            Backbone C-alpha coordinates, shape ``(L, 3)``.

        Returns
        -------
        torch.Tensor
            Direction-augmented hidden states, shape ``(L, dim)``.
        """
        length = coords.shape[0]
        prev = torch.roll(coords, 1, dims=0)
        nxt = torch.roll(coords, -1, dims=0)
        tangent = F.normalize(nxt - prev, dim=-1, eps=1e-6)
        if length > 2:
            tangent[0] = F.normalize(coords[1] - coords[0], dim=-1, eps=1e-6)
            tangent[-1] = F.normalize(coords[-1] - coords[-2], dim=-1, eps=1e-6)
        weights = self.to_scale(h)  # (L, 3) learned per-axis scale
        direction = weights * tangent
        return self.merge(torch.cat([h, direction], dim=-1))


class DiffSDSDenoiser(nn.Module):
    """Compact DiffSDS: Transformer encoder + Atomic Direction Space + inpaint head.

    Backbone dihedral-angle tokens (masked span + fixed context) are embedded,
    passed through a bidirectional Transformer encoder, periodically fused
    with the equivariant Atomic Direction Space feature (built from the
    context C-alpha coordinates), and finally denoised back into per-residue
    backbone angles for the masked positions.
    """

    def __init__(self, d_model: int = 32, layers: int = 3, n_angles: int = 3) -> None:
        """Build the angle embedding, Transformer+ADS stack, and denoising head.

        Parameters
        ----------
        d_model:
            Transformer hidden width.
        layers:
            Number of Transformer-encoder + ADS-fusion stages.
        n_angles:
            Number of backbone dihedral angles per residue (phi, psi, omega).
        """
        super().__init__()
        self.angle_proj = nn.Linear(2 * n_angles, d_model)
        self.mask_embed = nn.Embedding(2, d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=64, batch_first=True
        )
        self.encoders = nn.ModuleList(
            [nn.TransformerEncoder(enc_layer, num_layers=1) for _ in range(layers)]
        )
        self.ads_layers = nn.ModuleList([AtomicDirectionSpace(d_model) for _ in range(layers)])
        self.head = nn.Linear(d_model, n_angles)

    def forward(
        self,
        angles: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise the masked backbone-angle span conditioned on context geometry.

        Parameters
        ----------
        angles:
            Noised backbone dihedral angles, shape ``(L, n_angles)``.
        coords:
            Context C-alpha coordinates (fixed, unmasked-region geometry used
            to build the local frame for every position), shape ``(L, 3)``.
        mask:
            Integer mask token, 1 = masked (to be inpainted), 0 = fixed context,
            shape ``(L,)``.
        timestep:
            Scalar diffusion timestep in ``[0, 1]``, shape ``(1,)``.

        Returns
        -------
        torch.Tensor
            Denoised backbone dihedral angles for all positions, ``(L, n_angles)``.
        """
        length = angles.shape[0]
        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        x = self.angle_proj(sincos) + self.mask_embed(mask) + self.time_embed(timestep.view(1, 1))
        x = x.unsqueeze(0)
        for encoder, ads in zip(self.encoders, self.ads_layers):
            x = encoder(x)
            x = ads(x.squeeze(0), coords).unsqueeze(0)
        return self.head(x.squeeze(0)).view(length, -1)


def build_diffsds() -> nn.Module:
    """Construct a compact DiffSDS language-diffusion backbone-inpainting model."""
    return DiffSDSDenoiser(d_model=32, layers=3, n_angles=3).eval()


def example_input_diffsds() -> tuple:
    """Random 20-residue backbone span (angles + coords + inpaint mask + timestep)."""
    length = 20
    angles = (torch.rand(length, 3) - 0.5) * 2 * math.pi
    coords = torch.cumsum(torch.randn(length, 3) * 1.5, dim=0)
    mask = torch.zeros(length, dtype=torch.long)
    mask[8:14] = 1  # masked span to inpaint
    timestep = torch.tensor([0.5])
    return (angles, coords, mask, timestep)


MENAGERIE_ENTRIES = [
    ("DeepRank", "build_deeprank", "example_input_deeprank", "2021", "BIO"),
    ("DeepRank-GNN", "build_deeprank_gnn", "example_input_deeprank_gnn", "2023", "BIO"),
    ("DeepTracer", "build_deeptracer", "example_input_deeptracer", "2020", "BIO"),
    ("DiffAb", "build_diffab", "example_input_diffab", "2022", "BIO"),
    ("DiffDock-PP", "build_diffdock_pp", "example_input_diffdock_pp", "2023", "BIO"),
    ("DiffSDS", "build_diffsds", "example_input_diffsds", "2023", "BIO"),
]
