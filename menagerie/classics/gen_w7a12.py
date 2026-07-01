"""Compact faithful classics for four structural/sequence biology architectures.

Sources checked (repo code inspected via GitHub API, base env only, no clone
or pip install):
  - SPOT-RNA2: https://github.com/jaswindersingh2/SPOT-RNA2 (``utils/
    SPOT-RNA2.py`` restores frozen TensorFlow ``model{0..3}`` checkpoints;
    ``README.md`` Figure 1 pipeline diagram). Singh, Wang & Zhou, "Improved
    RNA Secondary Structure and Tertiary Base-pairing Prediction using
    Evolutionary Profile, Mutational Coupling and Two-dimensional Transfer
    Learning", Nucleic Acids Research 2021. SPOT-RNA2 extends SPOT-RNA's
    single-sequence dilated-CNN base-pairing predictor by stacking rich 2-D
    evolutionary features (direct-coupling-analysis / mutational-coupling
    matrix, co-variation / mutual-information matrix, and a consensus
    secondary-structure probability map derived from the profile) on top of
    the outer-concatenated 1-D sequence+PSSM features, then runs the same
    residual-dilated-2D-CNN base-pair map predictor, four-model ensemble,
    with a bpRNA-pretrain -> PDB-transfer-learn two-stage recipe (single
    forward pass reimplemented here). Reimplemented as ``SpotRna2Model``:
    a 1-D profile encoder (sequence one-hot + PSSM) outer-concatenated into
    a 2-D map, channel-concatenated with the raw DCA/co-variation/consensus-
    SS 2-D feature stack, then a residual dilated-conv-2D tower (matching
    SPOT-RNA's ``SpotRnaResBlock`` residual-dilation pattern) producing a
    symmetric base-pairing probability matrix -- capturing the paper's
    defining "add evolutionary 2-D features to the 2-D transfer-learning
    backbone" mechanism rather than a generic CNN stub.
  - tFold: https://github.com/TencentAI4S/tfold (``tfold/model/arch/
    ag_model/model.py`` ``AgModel`` -- single-chain ``ComplexStructureModel``
    ligand trunk plus a separate ``DockingModelSM`` receptor-ligand docking
    head; ``tfold/model/module/evoformer`` for the Evoformer-family trunk
    used by the ligand structure model; ``tfold/model/arch/alphafold`` for
    the underlying AlphaFold2-style trunk this repo builds on). Wu et al.,
    "Fast and accurate modeling and design of antibody-antigen complex using
    tFold", Nature Communications 2024. tFold-Ag's defining mechanism is
    exactly the two-stage split visible in ``AgModel.forward``: (1) an
    Evoformer-style single+pair-representation trunk (row/column attention
    on a pseudo-MSA-derived single track, triangle-multiplicative pair
    updates, communicated via pair-bias attention -- reusing the same
    Evoformer-block primitive family established in
    ``menagerie/classics/openfold_af2.py``, defined locally here in compact
    form) folds each chain (ligand) independently into single/pair features
    plus coordinates; (2) a lightweight cross-attention ``DockingModule``
    then takes the frozen per-chain single/pair features and predicts a
    single rigid-body transform (rotation quaternion + translation) that
    docks the ligand onto the receptor, i.e. structure prediction is
    factored into "fold each chain" + "dock the chains" rather than one
    joint trunk over the concatenated complex -- the tFold-Ag-specific
    design this reimplementation preserves as ``TFoldAgModel``.
  - SweetNet: https://github.com/BojarLab/SweetNet (``SweetNet_code.ipynb``,
    ``glycowork.py`` -- legacy repo, functionality now lives in the
    ``glycowork`` pip package). Burkholz, Quackenbush & Bojar, "Using Graph
    Convolutional Neural Networks to Learn a Representation for Glycans",
    Cell Chemical Biology 2021. Glycans are represented as graphs
    (monosaccharide residues = nodes carrying a learned glycoletter
    embedding, glycosidic linkages = edges); the notebook's ``SweetNet``
    class defines the exact mechanism -- three stages of
    ``GraphConv -> TopKPooling``, each stage's post-pool node features
    summarized by concatenated global-max + global-mean pooling, the three
    per-stage pooled summaries *summed* together, then an MLP classifier
    head. Reimplemented essentially verbatim (same layer types, same
    pool-then-readout-then-sum pattern) as ``SweetNetModel`` using
    ``torch_geometric.nn.GraphConv``/``TopKPooling``/``global_max_pool``/
    ``global_mean_pool`` on a small random glycan graph.
  - ThermoMPNN: https://github.com/Kuhlman-Lab/ThermoMPNN. Dieckhaus,
    Brocidiacono, Randolph & Kuhlman, "Transfer learning to leverage larger
    datasets for improved prediction of protein stability changes", PNAS
    2024. ThermoMPNN's defining mechanism is transfer learning from a
    *frozen* pretrained ProteinMPNN structural-graph encoder: backbone
    coordinates are run through ProteinMPNN's message-passing encoder
    (weights frozen, no gradient), the per-residue encoder embeddings at
    the mutated position (plus a learned wild-type/mutant amino-acid
    identity embedding) are concatenated, and a lightweight MLP head
    predicts the folding stability change (ddG) -- reusing the message-
    passing residue-graph encoder pattern already established in
    ``menagerie/classics/proteinmpnn.py`` (defined locally, frozen via
    ``torch.no_grad()`` in ``forward``) feeding a compact ddG-prediction
    MLP. This also covers the ThermoMPNN-D (dimer/double-mutant) variant's
    core single-mutant scoring path, which shares the same frozen-encoder
    + MLP-head mechanism per the paper.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# SPOT-RNA2: SPOT-RNA's residual dilated-2D-CNN backbone extended with
# evolutionary-profile 2-D features (DCA / co-variation / consensus-SS maps)
# concatenated onto the outer-product sequence+PSSM map.
# ---------------------------------------------------------------------------


class SpotRna2ResBlock(nn.Module):
    """Dilated residual convolution block over the 2-D base-pair map."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize the two dilated convolutions and norms.

        Parameters
        ----------
        channels:
            Number of feature-map channels.
        dilation:
            Dilation rate for both convolutions in this block.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a dilated residual convolution block.

        Parameters
        ----------
        x:
            Feature map ``(B, C, L, L)``.

        Returns
        -------
        Tensor
            Updated feature map, same shape.
        """

        h = F.elu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.elu(x + h)


class SpotRna2Model(nn.Module):
    """Compact SPOT-RNA2 base-pairing predictor (single ensemble member).

    Outer-concatenates the 1-D sequence+PSSM profile into a 2-D map,
    channel-stacks it with the raw 2-D evolutionary feature maps (DCA
    coupling, co-variation/mutual-information, consensus secondary
    structure), then runs a residual dilated-conv-2D tower to predict a
    symmetric base-pairing probability matrix.
    """

    def __init__(
        self,
        seq_channels: int = 4,
        pssm_channels: int = 4,
        evo_channels: int = 3,
        hidden: int = 32,
        num_blocks: int = 4,
    ) -> None:
        """Initialize the profile encoder and residual dilated tower.

        Parameters
        ----------
        seq_channels:
            One-hot nucleotide channel count.
        pssm_channels:
            Position-specific scoring-matrix channel count.
        evo_channels:
            Number of raw 2-D evolutionary feature maps (DCA, co-variation,
            consensus secondary structure).
        hidden:
            Tower feature-map width.
        num_blocks:
            Number of dilated residual blocks (dilation doubles each block).
        """

        super().__init__()
        profile_channels = seq_channels + pssm_channels
        self.profile_proj = nn.Linear(profile_channels * 2, hidden)
        self.evo_proj = nn.Conv2d(evo_channels, hidden, 1)
        self.stem = nn.Conv2d(hidden * 2, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(
            [SpotRna2ResBlock(hidden, dilation=2**i) for i in range(num_blocks)]
        )
        self.out = nn.Conv2d(hidden, 1, 3, padding=1)

    def forward(self, profile: Tensor, evo_features: Tensor) -> Tensor:
        """Predict a symmetric RNA base-pairing probability map.

        Parameters
        ----------
        profile:
            Per-nucleotide sequence+PSSM profile ``(B, L, seq+pssm)``.
        evo_features:
            Raw 2-D evolutionary feature maps ``(B, evo_channels, L, L)``.

        Returns
        -------
        Tensor
            Base-pairing probabilities ``(B, L, L)``.
        """

        length = profile.shape[1]
        row = profile.unsqueeze(2).expand(-1, -1, length, -1)
        col = profile.unsqueeze(1).expand(-1, length, -1, -1)
        outer = torch.cat([row, col], dim=-1)
        seq_map = self.profile_proj(outer).permute(0, 3, 1, 2)
        evo_map = self.evo_proj(evo_features)
        x = F.elu(self.stem(torch.cat([seq_map, evo_map], dim=1)))
        for block in self.blocks:
            x = block(x)
        logits = self.out(x).squeeze(1)
        return torch.sigmoid((logits + logits.transpose(1, 2)) * 0.5)


def build_spot_rna2() -> nn.Module:
    """Build a compact SPOT-RNA2 base-pairing predictor.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return SpotRna2Model().eval()


def example_input_spot_rna2() -> Tuple[Tensor, Tensor]:
    """Return a random RNA profile plus evolutionary 2-D feature stack.

    Returns
    -------
    tuple[Tensor, Tensor]
        Sequence+PSSM profile ``(1, L, 8)`` and evolutionary feature maps
        ``(1, 3, L, L)``.
    """

    length = 20
    profile = torch.rand(1, length, 8)
    evo = torch.rand(1, 3, length, length)
    return profile, evo


# ---------------------------------------------------------------------------
# tFold-Ag: Evoformer-style per-chain folding trunk + separate cross-
# attention docking head that predicts a rigid-body transform placing the
# folded ligand onto the receptor.
# ---------------------------------------------------------------------------


class TFoldEvoformerBlock(nn.Module):
    """Compact Evoformer-family block: pair-biased row attention + triangle
    multiplicative update on the pair track."""

    def __init__(self, dim_s: int, dim_z: int, n_head: int = 4) -> None:
        """Initialize single-track attention and pair-track triangle update.

        Parameters
        ----------
        dim_s:
            Single-representation feature width.
        dim_z:
            Pair-representation feature width.
        n_head:
            Number of attention heads for pair-biased row attention.
        """

        super().__init__()
        self.n_head = n_head
        self.norm_s = nn.LayerNorm(dim_s)
        self.q = nn.Linear(dim_s, dim_s)
        self.k = nn.Linear(dim_s, dim_s)
        self.v = nn.Linear(dim_s, dim_s)
        self.pair_bias = nn.Linear(dim_z, n_head)
        self.s_out = nn.Linear(dim_s, dim_s)
        self.s_transition = nn.Sequential(
            nn.Linear(dim_s, dim_s * 2), nn.GELU(), nn.Linear(dim_s * 2, dim_s)
        )
        self.norm_z = nn.LayerNorm(dim_z)
        self.left = nn.Linear(dim_z, dim_z)
        self.right = nn.Linear(dim_z, dim_z)
        self.gate = nn.Linear(dim_z, dim_z)
        self.tri_out = nn.Linear(dim_z, dim_z)
        self.z_transition = nn.Sequential(
            nn.Linear(dim_z, dim_z * 2), nn.GELU(), nn.Linear(dim_z * 2, dim_z)
        )

    def forward(self, s: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Update single and pair representations by one Evoformer block.

        Parameters
        ----------
        s:
            Single representation ``(B, L, dim_s)``.
        z:
            Pair representation ``(B, L, L, dim_z)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated single and pair representations.
        """

        batch, length, dim_s = s.shape
        head_dim = dim_s // self.n_head
        s_norm = self.norm_s(s)
        q = self.q(s_norm).view(batch, length, self.n_head, head_dim).transpose(1, 2)
        k = self.k(s_norm).view(batch, length, self.n_head, head_dim).transpose(1, 2)
        v = self.v(s_norm).view(batch, length, self.n_head, head_dim).transpose(1, 2)
        bias = self.pair_bias(z).permute(0, 3, 1, 2)
        attn = torch.softmax((q @ k.transpose(-1, -2)) / head_dim**0.5 + bias, dim=-1)
        s_msg = (attn @ v).transpose(1, 2).reshape(batch, length, dim_s)
        s = s + self.s_out(s_msg)
        s = s + self.s_transition(s)

        z_norm = self.norm_z(z)
        left = self.left(z_norm)
        right = self.right(z_norm)
        gate = torch.sigmoid(self.gate(z_norm))
        tri = torch.einsum("bikc,bjkc->bijc", left, right) * gate
        z = z + self.tri_out(tri)
        z = z + self.z_transition(z)
        return s, z


class TFoldFoldingTrunk(nn.Module):
    """Per-chain Evoformer-style trunk producing single/pair features and a
    lightweight rigid-body backbone coordinate readout."""

    def __init__(self, dim_s: int = 32, dim_z: int = 16, n_layers: int = 2) -> None:
        """Initialize the sequence/pair embedding and Evoformer stack.

        Parameters
        ----------
        dim_s:
            Single-representation feature width.
        dim_z:
            Pair-representation feature width.
        n_layers:
            Number of Evoformer blocks.
        """

        super().__init__()
        self.seq_embed = nn.Embedding(21, dim_s)
        self.pair_embed = nn.Linear(dim_s * 2, dim_z)
        self.blocks = nn.ModuleList([TFoldEvoformerBlock(dim_s, dim_z) for _ in range(n_layers)])
        self.coord_head = nn.Linear(dim_s, 3)

    def forward(self, tokens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Fold a single chain from its amino-acid sequence.

        Parameters
        ----------
        tokens:
            Amino-acid ids ``(B, L)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Single features ``(B, L, dim_s)``, pair features
            ``(B, L, L, dim_z)``, and predicted CA coordinates ``(B, L, 3)``.
        """

        s = self.seq_embed(tokens)
        length = s.shape[1]
        row = s.unsqueeze(2).expand(-1, -1, length, -1)
        col = s.unsqueeze(1).expand(-1, length, -1, -1)
        z = self.pair_embed(torch.cat([row, col], dim=-1))
        for block in self.blocks:
            s, z = block(s, z)
        coords = self.coord_head(s)
        return s, z, coords


class TFoldDockingModule(nn.Module):
    """Cross-attention head predicting a rigid-body transform that docks
    the folded ligand chain onto the (already-known) receptor chain."""

    def __init__(self, dim_s: int = 32, n_head: int = 4) -> None:
        """Initialize the ligand-to-receptor cross-attention and pose head.

        Parameters
        ----------
        dim_s:
            Single-representation feature width shared by both chains.
        n_head:
            Number of cross-attention heads.
        """

        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim_s, n_head, batch_first=True)
        self.norm = nn.LayerNorm(dim_s)
        self.pose_head = nn.Linear(dim_s, 7)

    def forward(self, ligand_s: Tensor, receptor_s: Tensor) -> Tensor:
        """Predict a docking rigid-body transform (quaternion + translation).

        Parameters
        ----------
        ligand_s:
            Ligand single features ``(B, Ll, dim_s)``.
        receptor_s:
            Receptor single features ``(B, Lr, dim_s)``.

        Returns
        -------
        Tensor
            Docking pose ``(B, 7)`` (4 quaternion + 3 translation components).
        """

        attended, _ = self.cross_attn(ligand_s, receptor_s, receptor_s)
        pooled = self.norm(attended).mean(dim=1)
        return self.pose_head(pooled)


class TFoldAgModel(nn.Module):
    """Compact tFold-Ag: fold ligand + receptor independently, then dock.

    Mirrors ``AgModel.forward``'s two-stage design -- each chain is folded
    by the shared Evoformer-style trunk, then a separate docking module
    predicts the rigid-body transform placing the ligand onto the receptor.
    """

    def __init__(self, dim_s: int = 32, dim_z: int = 16) -> None:
        """Initialize the shared folding trunk and docking head.

        Parameters
        ----------
        dim_s:
            Single-representation feature width.
        dim_z:
            Pair-representation feature width.
        """

        super().__init__()
        self.trunk = TFoldFoldingTrunk(dim_s=dim_s, dim_z=dim_z)
        self.docking = TFoldDockingModule(dim_s=dim_s)

    def forward(
        self, ligand_tokens: Tensor, receptor_tokens: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Fold ligand and receptor chains, then predict the docking pose.

        Parameters
        ----------
        ligand_tokens:
            Ligand amino-acid ids ``(B, Ll)``.
        receptor_tokens:
            Receptor amino-acid ids ``(B, Lr)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Ligand CA coordinates, receptor CA coordinates, and docking pose.
        """

        ligand_s, _, ligand_coords = self.trunk(ligand_tokens)
        receptor_s, _, receptor_coords = self.trunk(receptor_tokens)
        pose = self.docking(ligand_s, receptor_s)
        return ligand_coords, receptor_coords, pose


def build_tfold_ag() -> nn.Module:
    """Build a compact tFold-Ag antibody-antigen docking model.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return TFoldAgModel().eval()


def example_input_tfold_ag() -> Tuple[Tensor, Tensor]:
    """Return small ligand and receptor amino-acid token sequences.

    Returns
    -------
    tuple[Tensor, Tensor]
        Ligand tokens ``(1, 10)`` and receptor tokens ``(1, 14)``.
    """

    return torch.randint(0, 21, (1, 10)), torch.randint(0, 21, (1, 14))


# ---------------------------------------------------------------------------
# SweetNet: glycan graph classifier. Three stages of
# GraphConv -> TopKPooling, each stage read out via concatenated global-max
# + global-mean pooling, the three per-stage readouts summed, then an MLP
# classifier head -- reimplemented essentially verbatim from the notebook.
# ---------------------------------------------------------------------------


class SweetNetModel(nn.Module):
    """Compact SweetNet glycan graph convolutional classifier."""

    def __init__(self, lib_size: int = 64, hidden: int = 32, num_classes: int = 5) -> None:
        """Initialize the glycoletter embedding, conv/pool stack, and head.

        Parameters
        ----------
        lib_size:
            Number of distinct glycoletter tokens.
        hidden:
            Graph-conv feature width.
        num_classes:
            Number of output taxonomy classes.
        """

        super().__init__()
        from torch_geometric.nn import GraphConv, TopKPooling

        self.item_embedding = nn.Embedding(lib_size + 1, hidden)
        self.conv1 = GraphConv(hidden, hidden)
        self.pool1 = TopKPooling(hidden, ratio=0.8)
        self.conv2 = GraphConv(hidden, hidden)
        self.pool2 = TopKPooling(hidden, ratio=0.8)
        self.conv3 = GraphConv(hidden, hidden)
        self.pool3 = TopKPooling(hidden, ratio=0.8)
        self.lin1 = nn.Linear(hidden * 2, hidden * 4)
        self.lin2 = nn.Linear(hidden * 4, hidden)
        self.lin3 = nn.Linear(hidden, num_classes)
        self.bn1 = nn.BatchNorm1d(hidden * 4)
        self.bn2 = nn.BatchNorm1d(hidden)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Classify a batch of glycan graphs.

        Parameters
        ----------
        x:
            Glycoletter node ids ``(N,)``.
        edge_index:
            Glycosidic-linkage edge index ``(2, E)``.
        batch:
            Graph-assignment vector ``(N,)``.

        Returns
        -------
        Tensor
            Per-graph class logits ``(num_graphs, num_classes)``.
        """

        h = self.item_embedding(x)

        h = F.leaky_relu(self.conv1(h, edge_index))
        h, edge_index, _, batch, _, _ = self.pool1(h, edge_index, None, batch)
        from torch_geometric.nn import global_max_pool as gmp
        from torch_geometric.nn import global_mean_pool as gap

        x1 = torch.cat([gmp(h, batch), gap(h, batch)], dim=1)

        h = F.leaky_relu(self.conv2(h, edge_index))
        h, edge_index, _, batch, _, _ = self.pool2(h, edge_index, None, batch)
        x2 = torch.cat([gmp(h, batch), gap(h, batch)], dim=1)

        h = F.leaky_relu(self.conv3(h, edge_index))
        h, edge_index, _, batch, _, _ = self.pool3(h, edge_index, None, batch)
        x3 = torch.cat([gmp(h, batch), gap(h, batch)], dim=1)

        pooled = x1 + x2 + x3
        h = self.bn1(F.leaky_relu(self.lin1(pooled)))
        h = self.bn2(F.leaky_relu(self.lin2(h)))
        return self.lin3(h)


def build_sweetnet() -> nn.Module:
    """Build a compact SweetNet glycan classifier.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return SweetNetModel().eval()


def example_input_sweetnet() -> Tuple[Tensor, Tensor, Tensor]:
    """Return a small random glycan graph batch.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Glycoletter node ids, glycosidic-linkage edge index, and the
        graph-assignment batch vector for two small glycan graphs.
    """

    x = torch.randint(0, 64, (14,))
    edges: List[Tuple[int, int]] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 7),
    ]
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    batch = torch.tensor([0] * 7 + [1] * 7, dtype=torch.long)
    return x, edge_index, batch


# ---------------------------------------------------------------------------
# ThermoMPNN: frozen ProteinMPNN structural-graph encoder (transfer-learning
# source) + a lightweight ddG-prediction MLP head over the mutated position's
# encoder embedding and wild-type/mutant identity embedding.
# ---------------------------------------------------------------------------


def _knn_edges_thermo(ca: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Build a residue k-nearest-neighbor graph from C-alpha coordinates.

    Parameters
    ----------
    ca:
        C-alpha coordinates ``(B, L, 3)``.
    k:
        Number of neighbors per residue.

    Returns
    -------
    tuple[Tensor, Tensor]
        Neighbor indices and distances.
    """

    dist = torch.cdist(ca, ca)
    vals, idx = dist.topk(k + 1, largest=False)
    return idx[:, :, 1:], vals[:, :, 1:]


def _gather_nodes_thermo(nodes: Tensor, idx: Tensor) -> Tensor:
    """Gather node features at neighbor indices.

    Parameters
    ----------
    nodes:
        Node features ``(B, L, C)``.
    idx:
        Neighbor indices ``(B, L, K)``.

    Returns
    -------
    Tensor
        Neighbor features ``(B, L, K, C)``.
    """

    batch, length, channels = nodes.shape
    flat = nodes.reshape(batch * length, channels)
    offset = torch.arange(batch, device=nodes.device).view(batch, 1, 1) * length
    return flat[(idx + offset).reshape(-1)].view(batch, length, idx.shape[-1], channels)


class ThermoMpnnEncoderLayer(nn.Module):
    """Frozen ProteinMPNN-style edge-conditioned message-passing layer."""

    def __init__(self, dim: int) -> None:
        """Initialize message and edge-update projections.

        Parameters
        ----------
        dim:
            Hidden feature size.
        """

        super().__init__()
        self.msg = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.edge = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.node_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)

    def forward(self, node: Tensor, edge: Tensor, idx: Tensor) -> Tuple[Tensor, Tensor]:
        """Update node and edge features by one message-passing round.

        Parameters
        ----------
        node:
            Residue features ``(B, L, C)``.
        edge:
            Edge features ``(B, L, K, C)``.
        idx:
            Neighbor indices.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated node and edge features.
        """

        neigh = _gather_nodes_thermo(node, idx)
        src = node.unsqueeze(2).expand_as(neigh)
        messages = self.msg(torch.cat([src, neigh, edge], dim=-1))
        node = self.node_norm(node + messages.mean(dim=2))
        edge = self.edge_norm(edge + self.edge(torch.cat([edge, messages], dim=-1)))
        return node, edge


class FrozenProteinMpnnEncoder(nn.Module):
    """Frozen ProteinMPNN structural-graph encoder used as a fixed feature
    extractor (transfer-learning source for ThermoMPNN)."""

    def __init__(self, dim: int = 32, k: int = 6, layers: int = 3) -> None:
        """Initialize the backbone featurizer and message-passing stack.

        Parameters
        ----------
        dim:
            Hidden width.
        k:
            Residue graph degree.
        layers:
            Number of message-passing blocks.
        """

        super().__init__()
        self.k = k
        self.node_in = nn.Linear(9, dim)
        self.edge_in = nn.Linear(8, dim)
        self.layers = nn.ModuleList([ThermoMpnnEncoderLayer(dim) for _ in range(layers)])

    def forward(self, backbone: Tensor) -> Tensor:
        """Encode backbone coordinates into frozen per-residue embeddings.

        Parameters
        ----------
        backbone:
            Backbone atom coordinates ``(B, L, 3, 3)`` for N, CA, C.

        Returns
        -------
        Tensor
            Per-residue structural embeddings ``(B, L, dim)``.
        """

        n_coord, ca, c_coord = backbone.unbind(dim=2)
        idx, dist = _knn_edges_thermo(ca, self.k)
        forward_vec = F.normalize(c_coord - ca, dim=-1)
        backward_vec = F.normalize(n_coord - ca, dim=-1)
        node = self.node_in(torch.cat([ca, forward_vec, backward_vec], dim=-1))
        neigh_ca = _gather_nodes_thermo(ca, idx)
        rel = neigh_ca - ca.unsqueeze(2)
        edge = self.edge_in(
            torch.cat([dist.unsqueeze(-1), rel, rel.abs(), rel.norm(dim=-1, keepdim=True)], dim=-1)
        )
        for layer in self.layers:
            node, edge = layer(node, edge, idx)
        return node


class ThermoMpnnModel(nn.Module):
    """Compact ThermoMPNN: frozen ProteinMPNN encoder + ddG-prediction MLP.

    Transfer-learns from a frozen structural-graph encoder: the encoder
    runs under ``torch.no_grad()`` (no gradient into the pretrained
    backbone), the mutated-position embedding is concatenated with a
    learned wild-type/mutant amino-acid identity embedding, and a
    lightweight MLP head regresses the folding stability change (ddG).
    """

    def __init__(self, dim: int = 32, vocab: int = 21) -> None:
        """Initialize the frozen encoder, identity embedding, and ddG head.

        Parameters
        ----------
        dim:
            Encoder hidden width.
        vocab:
            Amino-acid vocabulary size.
        """

        super().__init__()
        self.encoder = FrozenProteinMpnnEncoder(dim=dim)
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.aa_embed = nn.Embedding(vocab, dim)
        self.ddg_head = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self, backbone: Tensor, mut_position: Tensor, wt_aa: Tensor, mut_aa: Tensor
    ) -> Tensor:
        """Predict the folding-stability change (ddG) for a point mutation.

        Parameters
        ----------
        backbone:
            Backbone atom coordinates ``(B, L, 3, 3)`` for N, CA, C.
        mut_position:
            Index of the mutated residue, one per batch element ``(B,)``.
        wt_aa:
            Wild-type amino-acid id at the mutated position ``(B,)``.
        mut_aa:
            Mutant amino-acid id at the mutated position ``(B,)``.

        Returns
        -------
        Tensor
            Predicted ddG ``(B, 1)``.
        """

        with torch.no_grad():
            embeddings = self.encoder(backbone)
        batch = torch.arange(embeddings.shape[0], device=embeddings.device)
        site_embedding = embeddings[batch, mut_position]
        wt_embedding = self.aa_embed(wt_aa)
        mut_embedding = self.aa_embed(mut_aa)
        features = torch.cat([site_embedding, wt_embedding, mut_embedding], dim=-1)
        return self.ddg_head(features)


def build_thermompnn() -> nn.Module:
    """Build a compact ThermoMPNN ddG predictor.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return ThermoMpnnModel().eval()


def example_input_thermompnn() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return a small protein backbone and one point-mutation query.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Backbone coordinates, mutated-residue index, wild-type amino-acid
        id, and mutant amino-acid id.
    """

    length = 12
    ca = torch.randn(1, length, 3).cumsum(dim=1)
    n_coord = ca + torch.tensor([0.2, 0.0, 0.0])
    c_coord = ca + torch.tensor([-0.2, 0.1, 0.0])
    backbone = torch.stack([n_coord, ca, c_coord], dim=2)
    mut_position = torch.tensor([5])
    wt_aa = torch.tensor([3])
    mut_aa = torch.tensor([11])
    return backbone, mut_position, wt_aa, mut_aa


MENAGERIE_ENTRIES = [
    ("SPOT-RNA2", "build_spot_rna2", "example_input_spot_rna2", "2021", "BIO"),
    ("tFold", "build_tfold_ag", "example_input_tfold_ag", "2024", "BIO"),
    ("SweetNet", "build_sweetnet", "example_input_sweetnet", "2021", "BIO"),
    ("ThermoMPNN", "build_thermompnn", "example_input_thermompnn", "2024", "BIO"),
]
