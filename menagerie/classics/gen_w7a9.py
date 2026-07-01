"""Compact faithful classics for six structural-biology architectures.

Sources checked (repo README/paper abstract inspected via web search, base
env only, no clone or pip install):
  - ProteinSolver: https://github.com/ostrokach/proteinsolver (also
    gitlab.com/ostrokach/proteinsolver). Strokach, Becerra, Corbi-Verge,
    Perez-Riba & Kim, "Fast and Flexible Protein Design Using Deep Graph
    Neural Networks", Cell Systems 2020. Frames protein sequence design as
    a Sudoku-like constraint-satisfaction problem on a graph: nodes are
    residues (one-hot amino-acid state, progressively unmasked), edges are
    spatial contacts from the target 3D structure (encoded via pairwise
    distance/orientation features). A deep stack of message-passing graph
    layers (edge-conditioned message = MLP over [node_i, node_j, edge_ij],
    aggregated by sum over neighbours, then a node-update MLP + residual)
    repeatedly propagates constraints between connected residues; the final
    per-node linear head emits a distribution over the 20 amino acids,
    which is decoded one residue at a time (here we reproduce the per-node
    message-passing tower and read out all positions in one forward pass,
    matching the "self-supervised, then greedy/CSP decode" architecture's
    encoder half that TorchLens can trace deterministically).
  - RaptorX-Contact: https://github.com/j3xugit/RaptorX-Contact. Wang,
    Sun, Zhao, Shen, Xiong & Xu, "Accurate De Novo Prediction of Protein
    Contact Map by Ultra-Deep Learning Model", PLOS Comp Bio 2017 / Xu,
    "Distance-based Protein Folding Powered by Deep Learning", PNAS 2019.
    1D sequence profile (PSSM/MSA-derived features) is first passed through
    a stack of 1D residual conv blocks, then converted into a 2D residue x
    residue map via outer-concatenation; a deep stack of 2D residual blocks
    with DILATED convolutions (dilation cycling through {1, 2, 4} to grow
    receptive field cheaply, the paper's defining trick) refines the pair
    map, and a final 1x1 conv head predicts binned inter-residue
    contact/distance distributions. We reproduce the 1D->2D conversion +
    cyclically-dilated 2D ResNet exactly; this is the architecture-defining
    mechanism (distinguishes it from plain non-dilated contact ResNets).
  - RaptorX-Property: https://github.com/Indicator/RaptorX-SS8 (the
    Deep Convolutional Neural Fields successor is described in Wang, Peng,
    Ma & Xu, "Protein Secondary Structure Prediction Using Deep
    Convolutional Neural Fields", Scientific Reports 2016, and served at
    RaptorX-Property, Nucleic Acids Research 2016). A 1D deep residual
    convolutional network (multiple stacked dilated 1D conv blocks over the
    per-residue PSSM/profile sequence) produces per-residue features that
    feed three PARALLEL heads: 8-class secondary structure, 3-bin relative
    solvent accessibility, and binary disorder -- the defining "one shared
    deep 1D-conv trunk, three structural-property heads" multi-task design.
  - RFdiffusionAA (RoseTTAFold-All-Atom diffusion): Krishna, Wang, Ahern
    et al. (Baker Lab), "Generalized Biomolecular Modeling and Design with
    RoseTTAFold All-Atom", Science 2024;
    https://github.com/baker-laboratory/rf_diffusion_all_atom. Extends
    protein-only RFdiffusion (already a separate classic in this catalog)
    by ATOMIZING the small-molecule/ligand into individual-atom nodes (each
    atom is its own graph node/frame, vs. one frame per residue for the
    protein) that share the same pair-biased residue-attention backbone.
    The defining new mechanism vs. plain RFdiffusion is the joint
    residue-token + atom-token sequence with a residue<->atom cross-pair
    bias, letting the diffusion model denoise protein backbone frames
    *conditioned on* a rigid small-molecule scaffold of atom coordinates.
    We build a compact two-token-type (residue, ligand-atom) pair-biased
    denoiser reflecting this atomization, distinct from the pure-protein
    RFdiffusion classic already in the catalog.
  - RiboDiffusion: https://github.com/ml4bio/RiboDiffusion. Huang, Wang,
    He, Yi & collaborators, "RiboDiffusion: tertiary structure-based RNA
    inverse folding with generative diffusion models", Bioinformatics 2024
    (ISMB), arXiv:2404.11199. Two-module architecture: (1) a GVP-GNN
    (Geometric Vector Perceptron graph network, Jing et al. 2021) structure
    encoder that carries paired SCALAR and VECTOR node/edge features
    through equivariant-style linear+gating updates over the nucleotide
    backbone graph (encodes the fixed 3D RNA structure), and (2) a
    Transformer-based sequence module that iteratively denoises a
    diffusion-corrupted one-hot nucleotide sequence, cross-attending to the
    GVP-GNN structural embedding at every diffusion step. We reproduce both
    modules compactly: a scalar+vector GVP-style graph block, and a
    transformer decoder block conditioned on the pooled structural
    embedding plus a diffusion-timestep embedding.
  - RibonanzaNet: https://github.com/Shujun-He/RibonanzaNet. He, Huang
    et al., "Ribonanza: deep learning of RNA structure through dual
    crowdsourcing", Nature Methods 2024 (Kaggle "Ribonanza" 2023 origin).
    Embeds the nucleotide sequence, then builds a pairwise representation
    via an outer product of the (downsampled) sequence features plus
    relative-position encoding; each encoder layer applies (a) a 1D
    residual convolution over the sequence track, (b) pair-BIASED
    self-attention (the pairwise map supplies an additive attention bias,
    the defining mechanism carried over from RNAdegformer/AlphaFold-style
    pair conditioning), and (c) a triangular multiplicative update that
    refines the pairwise map from the sequence track (outer-product-mean)
    before the next layer. We reproduce the conv + pair-biased-attention +
    triangular-update encoder stack compactly.

All models are tiny random-init nn.Module reimplementations (architecture
catalog, not a trained-weights zoo); no code or weights were copied from any
upstream repository.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# ProteinSolver: graph message-passing constraint-satisfaction sequence design
# ---------------------------------------------------------------------------


class ProteinSolverMessageLayer(nn.Module):
    """One edge-conditioned message-passing + node-update layer."""

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        """Initialize the message and update MLPs.

        Parameters
        ----------
        node_dim:
            Per-residue node feature width.
        edge_dim:
            Per-contact edge feature width.
        """

        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.ELU(),
            nn.Linear(node_dim, node_dim),
        )
        self.update_mlp = nn.Sequential(nn.Linear(2 * node_dim, node_dim), nn.ELU())
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self, nodes: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Propagate one round of edge-conditioned messages.

        Parameters
        ----------
        nodes:
            Residue node features, shape ``(N, node_dim)``.
        edge_index:
            Contact edge list, shape ``(2, E)``.
        edge_attr:
            Contact edge features, shape ``(E, edge_dim)``.

        Returns
        -------
        torch.Tensor
            Updated node features, shape ``(N, node_dim)``.
        """

        src, dst = edge_index[0], edge_index[1]
        msg_in = torch.cat([nodes[src], nodes[dst], edge_attr], dim=-1)
        messages = self.message_mlp(msg_in)
        agg = torch.zeros_like(nodes).index_add_(0, dst, messages)
        updated = self.update_mlp(torch.cat([nodes, agg], dim=-1))
        return self.norm(nodes + updated)


class ProteinSolver(nn.Module):
    """Compact ProteinSolver-style constraint-propagation sequence-design GNN."""

    def __init__(
        self, n_residues: int = 24, node_dim: int = 32, edge_dim: int = 8, n_layers: int = 6
    ) -> None:
        """Initialize the residue embedding, contact-edge embedding, and message tower.

        Parameters
        ----------
        n_residues:
            Number of residues in the compact chain.
        node_dim:
            Residue node feature width.
        edge_dim:
            Contact edge feature width.
        n_layers:
            Number of stacked message-passing layers.
        """

        super().__init__()
        self.n_residues = n_residues
        self.node_embed = nn.Linear(21, node_dim)  # 20 AA + mask token
        self.edge_embed = nn.Linear(2, edge_dim)
        self.layers = nn.ModuleList(
            [ProteinSolverMessageLayer(node_dim, edge_dim) for _ in range(n_layers)]
        )
        self.readout = nn.Linear(node_dim, 20)

    def forward(
        self, masked_seq: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor
    ) -> torch.Tensor:
        """Propagate structural constraints and predict per-residue amino-acid logits.

        Parameters
        ----------
        masked_seq:
            One-hot (+ mask channel) residue states, shape ``(N, 21)``.
        edge_index:
            Contact graph edges, shape ``(2, E)``.
        edge_feat:
            Raw contact-edge geometry (distance, orientation), shape ``(E, 2)``.

        Returns
        -------
        torch.Tensor
            Per-residue amino-acid logits, shape ``(N, 20)``.
        """

        nodes = self.node_embed(masked_seq)
        edges = self.edge_embed(edge_feat)
        for layer in self.layers:
            nodes = layer(nodes, edge_index, edges)
        return self.readout(nodes)


def build_proteinsolver() -> nn.Module:
    """Build a compact ProteinSolver constraint-propagation GNN.

    Returns
    -------
    nn.Module
        Random-initialized ProteinSolver-style model in eval mode.
    """

    return ProteinSolver().eval()


def example_input_proteinsolver() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a masked sequence with a ring-plus-chord contact graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(masked_seq, edge_index, edge_feat)``.
    """

    n = 24
    masked_seq = torch.zeros(n, 21)
    masked_seq[:, 20] = 1.0  # all positions start masked
    chain = torch.arange(n)
    src = torch.cat([chain[:-1], chain[1:], chain[::2][:-1], chain[::2][1:]])
    dst = torch.cat([chain[1:], chain[:-1], chain[::2][1:], chain[::2][:-1]])
    edge_index = torch.stack([src, dst], dim=0)
    dist = torch.rand(edge_index.shape[1], 1) * 8.0 + 3.0
    orient = torch.rand(edge_index.shape[1], 1) * math.pi
    edge_feat = torch.cat([dist, orient], dim=-1)
    return masked_seq, edge_index, edge_feat


# ---------------------------------------------------------------------------
# RaptorX-Contact: 1D profile -> outer-concat -> cyclically-dilated 2D ResNet
# ---------------------------------------------------------------------------


class RaptorXResBlock1D(nn.Module):
    """1D residual block over the per-residue sequence profile."""

    def __init__(self, channels: int) -> None:
        """Initialize two 1D convolutions with batch norm.

        Parameters
        ----------
        channels:
            Feature channel width.
        """

        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a 1D residual conv block.

        Parameters
        ----------
        x:
            Sequence features, shape ``(B, C, L)``.

        Returns
        -------
        torch.Tensor
            Refined sequence features, shape ``(B, C, L)``.
        """

        y = F.elu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.elu(x + y)


class RaptorXDilatedResBlock2D(nn.Module):
    """2D residual block with a cyclically chosen dilation rate."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize two dilated 2D convolutions with batch norm.

        Parameters
        ----------
        channels:
            Feature channel width.
        dilation:
            Dilation rate for both convolutions in this block.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a dilated 2D residual conv block.

        Parameters
        ----------
        x:
            Pairwise map features, shape ``(B, C, L, L)``.

        Returns
        -------
        torch.Tensor
            Refined pairwise map features, shape ``(B, C, L, L)``.
        """

        y = F.elu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.elu(x + y)


class RaptorXContact(nn.Module):
    """Compact RaptorX-Contact ultra-deep dilated-ResNet contact predictor."""

    def __init__(
        self,
        seq_len: int = 32,
        in_dim: int = 20,
        channels_1d: int = 16,
        channels_2d: int = 24,
        n_1d_blocks: int = 2,
        n_2d_blocks: int = 6,
    ) -> None:
        """Initialize the 1D profile trunk, 2D dilated-ResNet trunk, and contact head.

        Parameters
        ----------
        seq_len:
            Number of residues.
        in_dim:
            Input profile feature width (PSSM-like).
        channels_1d:
            1D trunk channel width.
        channels_2d:
            2D trunk channel width.
        n_1d_blocks:
            Number of 1D residual blocks.
        n_2d_blocks:
            Number of dilated 2D residual blocks (dilation cycles 1, 2, 4).
        """

        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, channels_1d, kernel_size=1)
        self.blocks_1d = nn.ModuleList([RaptorXResBlock1D(channels_1d) for _ in range(n_1d_blocks)])
        self.pair_proj = nn.Conv2d(2 * channels_1d, channels_2d, kernel_size=1)
        dilations = [1, 2, 4]
        self.blocks_2d = nn.ModuleList(
            [
                RaptorXDilatedResBlock2D(channels_2d, dilations[i % len(dilations)])
                for i in range(n_2d_blocks)
            ]
        )
        self.contact_head = nn.Conv2d(channels_2d, 1, kernel_size=1)

    def forward(self, profile: torch.Tensor) -> torch.Tensor:
        """Predict a residue-pair contact probability map.

        Parameters
        ----------
        profile:
            Per-residue sequence profile, shape ``(B, L, in_dim)``.

        Returns
        -------
        torch.Tensor
            Contact logits, shape ``(B, L, L)``.
        """

        x = profile.transpose(1, 2)
        x = F.elu(self.in_proj(x))
        for block in self.blocks_1d:
            x = block(x)
        length = x.shape[-1]
        left = x.unsqueeze(-1).expand(-1, -1, -1, length)
        right = x.unsqueeze(-2).expand(-1, -1, length, -1)
        pair = torch.cat([left, right], dim=1)
        pair = F.elu(self.pair_proj(pair))
        for block in self.blocks_2d:
            pair = block(pair)
        contact = self.contact_head(pair).squeeze(1)
        return 0.5 * (contact + contact.transpose(-1, -2))


def build_raptorx_contact() -> nn.Module:
    """Build a compact RaptorX-Contact dilated-ResNet contact predictor.

    Returns
    -------
    nn.Module
        Random-initialized RaptorX-Contact-style model in eval mode.
    """

    return RaptorXContact().eval()


def example_input_raptorx_contact() -> torch.Tensor:
    """Create a synthetic per-residue sequence-profile tensor.

    Returns
    -------
    torch.Tensor
        Profile features of shape ``(1, 32, 20)``.
    """

    return torch.randn(1, 32, 20)


# ---------------------------------------------------------------------------
# RaptorX-Property: shared 1D dilated-conv trunk, 3 parallel property heads
# ---------------------------------------------------------------------------


class RaptorXPropertyResBlock(nn.Module):
    """1D dilated residual block for the shared property trunk."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize a dilated 1D residual conv block.

        Parameters
        ----------
        channels:
            Feature channel width.
        dilation:
            Dilation rate for this block.
        """

        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a dilated 1D residual conv block.

        Parameters
        ----------
        x:
            Sequence features, shape ``(B, C, L)``.

        Returns
        -------
        torch.Tensor
            Refined sequence features, shape ``(B, C, L)``.
        """

        y = F.elu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.elu(x + y)


class RaptorXProperty(nn.Module):
    """Compact RaptorX-Property: shared trunk, SS8 / RSA / disorder heads."""

    def __init__(self, in_dim: int = 20, channels: int = 24, n_blocks: int = 4) -> None:
        """Initialize the shared dilated-conv trunk and three property heads.

        Parameters
        ----------
        in_dim:
            Input profile feature width.
        channels:
            Trunk channel width.
        n_blocks:
            Number of dilated residual blocks (dilation cycles 1, 2, 4, 8).
        """

        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, channels, kernel_size=1)
        dilations = [1, 2, 4, 8]
        self.trunk = nn.ModuleList(
            [
                RaptorXPropertyResBlock(channels, dilations[i % len(dilations)])
                for i in range(n_blocks)
            ]
        )
        self.ss8_head = nn.Conv1d(channels, 8, kernel_size=1)
        self.rsa_head = nn.Conv1d(channels, 3, kernel_size=1)
        self.disorder_head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, profile: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict secondary structure, solvent accessibility, and disorder.

        Parameters
        ----------
        profile:
            Per-residue sequence profile, shape ``(B, L, in_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(ss8_logits, rsa_logits, disorder_logits)``, each with a
            leading ``(B, ., L)`` layout.
        """

        x = profile.transpose(1, 2)
        x = F.elu(self.in_proj(x))
        for block in self.trunk:
            x = block(x)
        return self.ss8_head(x), self.rsa_head(x), self.disorder_head(x)


def build_raptorx_property() -> nn.Module:
    """Build a compact RaptorX-Property multi-task 1D dilated-conv model.

    Returns
    -------
    nn.Module
        Random-initialized RaptorX-Property-style model in eval mode.
    """

    return RaptorXProperty().eval()


def example_input_raptorx_property() -> torch.Tensor:
    """Create a synthetic per-residue sequence-profile tensor.

    Returns
    -------
    torch.Tensor
        Profile features of shape ``(1, 40, 20)``.
    """

    return torch.randn(1, 40, 20)


# ---------------------------------------------------------------------------
# RFdiffusionAA: atomized protein + ligand-atom pair-biased diffusion denoiser
# ---------------------------------------------------------------------------


class AllAtomPairBiasedBlock(nn.Module):
    """Shared pair-biased attention block over a mixed residue/atom token set."""

    def __init__(self, dim: int, pair_dim: int, heads: int = 4) -> None:
        """Initialize the mixed-token pair-biased attention + pair update.

        Parameters
        ----------
        dim:
            Token feature width (shared by residue and ligand-atom tokens).
        pair_dim:
            Pair feature width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.pair_bias = nn.Linear(pair_dim, heads)
        self.pair_update = nn.Sequential(nn.Linear(pair_dim + dim, pair_dim), nn.SiLU())
        self.ff = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 3), nn.SiLU(), nn.Linear(dim * 3, dim)
        )

    def forward(
        self, tokens: torch.Tensor, pair: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply pair-biased attention across residue and ligand-atom tokens.

        Parameters
        ----------
        tokens:
            Mixed residue/atom token features, shape ``(B, T, D)``.
        pair:
            Token-pair features, shape ``(B, T, T, P)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated tokens and pair features.
        """

        n_tok = tokens.shape[1]
        bias = (
            self.pair_bias(pair)
            .permute(0, 3, 1, 2)
            .reshape(tokens.shape[0] * self.heads, n_tok, n_tok)
        )
        y, _ = self.attn(self.norm(tokens), self.norm(tokens), self.norm(tokens), attn_mask=bias)
        tokens = tokens + y
        pair_context = tokens[:, :, None, :] + tokens[:, None, :, :]
        pair = pair + self.pair_update(torch.cat([pair, pair_context], dim=-1))
        return tokens + self.ff(tokens), pair


class RFDiffusionAllAtom(nn.Module):
    """Compact RFdiffusionAA-style atomized protein+ligand diffusion denoiser."""

    def __init__(
        self,
        n_residues: int = 10,
        n_ligand_atoms: int = 6,
        dim: int = 40,
        pair_dim: int = 20,
    ) -> None:
        """Initialize residue-token, ligand-atom-token, and shared pair-biased tower.

        Parameters
        ----------
        n_residues:
            Number of protein backbone residue tokens.
        n_ligand_atoms:
            Number of atomized small-molecule ligand tokens.
        dim:
            Shared token feature width.
        pair_dim:
            Pair feature width.
        """

        super().__init__()
        self.n_residues = n_residues
        self.n_ligand_atoms = n_ligand_atoms
        self.n_tokens = n_residues + n_ligand_atoms
        self.residue_embed = nn.Linear(3, dim)
        self.atom_embed = nn.Linear(3 + 8, dim)  # coord + element one-hot(8)
        self.type_embed = nn.Embedding(2, dim)  # 0=residue, 1=ligand-atom
        self.time_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.pair_embed = nn.Sequential(
            nn.Linear(4, pair_dim), nn.SiLU(), nn.Linear(pair_dim, pair_dim)
        )
        self.blocks = nn.ModuleList([AllAtomPairBiasedBlock(dim, pair_dim) for _ in range(2)])
        self.coord_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 3)
        )

    def forward(
        self, noisy_residue_ca: torch.Tensor, ligand_atom_feat: torch.Tensor
    ) -> torch.Tensor:
        """Denoise protein backbone coordinates conditioned on a fixed ligand scaffold.

        Parameters
        ----------
        noisy_residue_ca:
            Noisy protein CA coordinates, shape ``(B, n_residues, 3)``.
        ligand_atom_feat:
            Ligand atom coordinates + element one-hot, shape
            ``(B, n_ligand_atoms, 11)``.

        Returns
        -------
        torch.Tensor
            Denoised protein CA coordinates, shape ``(B, n_residues, 3)``.
        """

        batch = noisy_residue_ca.shape[0]
        time = noisy_residue_ca.mean(dim=(1, 2), keepdim=True).reshape(batch, 1)
        time_bias = self.time_embed(time).unsqueeze(1)

        residue_tok = self.residue_embed(noisy_residue_ca) + time_bias
        residue_tok = residue_tok + self.type_embed(
            torch.zeros(batch, self.n_residues, dtype=torch.long, device=residue_tok.device)
        )
        ligand_tok = self.atom_embed(ligand_atom_feat) + time_bias
        ligand_tok = ligand_tok + self.type_embed(
            torch.ones(batch, self.n_ligand_atoms, dtype=torch.long, device=ligand_tok.device)
        )
        tokens = torch.cat([residue_tok, ligand_tok], dim=1)

        coords = torch.cat([noisy_residue_ca, ligand_atom_feat[..., :3]], dim=1)
        rel = coords[:, :, None, :] - coords[:, None, :, :]
        dist = rel.norm(dim=-1, keepdim=True)
        idx = torch.arange(self.n_tokens, device=coords.device, dtype=coords.dtype)
        sep = (idx[:, None] - idx[None, :]).abs().view(1, self.n_tokens, self.n_tokens, 1)
        pair = self.pair_embed(torch.cat([rel[..., :3], dist + sep / self.n_tokens], dim=-1))

        for block in self.blocks:
            tokens, pair = block(tokens, pair)
        residue_out = tokens[:, : self.n_residues]
        return noisy_residue_ca + self.coord_head(residue_out)


def build_rfdiffusionaa() -> nn.Module:
    """Build a compact RFdiffusionAA atomized protein+ligand denoiser.

    Returns
    -------
    nn.Module
        Random-initialized RFdiffusionAA-style model in eval mode.
    """

    return RFDiffusionAllAtom().eval()


def example_input_rfdiffusionaa() -> tuple[torch.Tensor, torch.Tensor]:
    """Create noisy protein CA coordinates and a fixed ligand-atom scaffold.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(noisy_residue_ca, ligand_atom_feat)``.
    """

    noisy_ca = torch.randn(1, 10, 3)
    coords = torch.randn(1, 6, 3)
    element = torch.zeros(1, 6, 8)
    element.scatter_(-1, torch.randint(0, 8, (1, 6, 1)), 1.0)
    ligand_feat = torch.cat([coords, element], dim=-1)
    return noisy_ca, ligand_feat


# ---------------------------------------------------------------------------
# RiboDiffusion: GVP-GNN structure encoder + transformer diffusion decoder
# ---------------------------------------------------------------------------


class GVPBlock(nn.Module):
    """Geometric Vector Perceptron block: paired scalar + vector features."""

    def __init__(self, scalar_dim: int, vector_dim: int) -> None:
        """Initialize scalar and vector linear maps and a gating nonlinearity.

        Parameters
        ----------
        scalar_dim:
            Scalar feature channel width.
        vector_dim:
            Vector feature channel width (each a 3-vector).
        """

        super().__init__()
        self.vector_to_scalar = nn.Linear(vector_dim, vector_dim)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim + vector_dim, scalar_dim),
            nn.SiLU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        self.vector_gate = nn.Sequential(nn.Linear(scalar_dim, vector_dim), nn.Sigmoid())
        self.vector_mix = nn.Linear(vector_dim, vector_dim, bias=False)

    def forward(
        self, scalar: torch.Tensor, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update scalar and vector node features via GVP-style mixing.

        Parameters
        ----------
        scalar:
            Scalar node features, shape ``(N, scalar_dim)``.
        vector:
            Vector node features, shape ``(N, vector_dim, 3)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(scalar, vector)`` node features.
        """

        vector_norm = vector.norm(dim=-1)
        scalar_update = self.scalar_mlp(torch.cat([scalar, vector_norm], dim=-1))
        scalar_out = scalar + scalar_update
        gate = self.vector_gate(scalar_out).unsqueeze(-1)
        vector_mixed = self.vector_mix(vector.transpose(-1, -2)).transpose(-1, -2)
        vector_out = vector + gate * vector_mixed
        return scalar_out, vector_out


class RiboDiffusionDecoderBlock(nn.Module):
    """Transformer decoder block conditioned on the pooled structure embedding."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize self-attention, structure cross-conditioning, and FFN.

        Parameters
        ----------
        dim:
            Token feature width.
        heads:
            Number of self-attention heads.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cond_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, seq_tok: torch.Tensor, struct_cond: torch.Tensor) -> torch.Tensor:
        """Denoise sequence tokens conditioned on the structure embedding.

        Parameters
        ----------
        seq_tok:
            Diffusion-corrupted nucleotide token embeddings, shape ``(B, N, D)``.
        struct_cond:
            Per-node structural conditioning embedding, shape ``(B, N, D)``.

        Returns
        -------
        torch.Tensor
            Denoised sequence token embeddings, shape ``(B, N, D)``.
        """

        h = self.norm1(seq_tok)
        y, _ = self.self_attn(h, h, h)
        seq_tok = seq_tok + y + self.cond_proj(struct_cond)
        seq_tok = seq_tok + self.ffn(self.norm2(seq_tok))
        return seq_tok


class RiboDiffusion(nn.Module):
    """Compact RiboDiffusion: GVP-GNN structure encoder + diffusion decoder."""

    def __init__(self, n_nucleotides: int = 20, scalar_dim: int = 32, vector_dim: int = 8) -> None:
        """Initialize the GVP structure encoder and transformer sequence decoder.

        Parameters
        ----------
        n_nucleotides:
            Number of nucleotides in the compact RNA chain.
        scalar_dim:
            Scalar feature width used throughout.
        vector_dim:
            Vector feature channel width for the GVP encoder.
        """

        super().__init__()
        self.n_nucleotides = n_nucleotides
        self.scalar_dim = scalar_dim
        self.scalar_in = nn.Linear(4, scalar_dim)  # backbone dihedral-style scalars
        self.vector_in = nn.Linear(1, vector_dim, bias=False)
        self.gvp_blocks = nn.ModuleList([GVPBlock(scalar_dim, vector_dim) for _ in range(2)])
        self.time_embed = nn.Sequential(
            nn.Linear(1, scalar_dim), nn.SiLU(), nn.Linear(scalar_dim, scalar_dim)
        )
        self.seq_embed = nn.Linear(4, scalar_dim)  # one-hot A/C/G/U (diffusion state)
        self.decoder_blocks = nn.ModuleList(
            [RiboDiffusionDecoderBlock(scalar_dim) for _ in range(2)]
        )
        self.readout = nn.Linear(scalar_dim, 4)

    def forward(
        self, backbone_coords: torch.Tensor, noisy_seq: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Denoise a corrupted nucleotide sequence conditioned on fixed 3D structure.

        Parameters
        ----------
        backbone_coords:
            Per-nucleotide backbone C1' coordinates, shape ``(B, N, 3)``.
        noisy_seq:
            Diffusion-corrupted one-hot nucleotide state, shape ``(B, N, 4)``.
        timestep:
            Diffusion timestep, shape ``(B, 1)``.

        Returns
        -------
        torch.Tensor
            Denoised nucleotide logits, shape ``(B, N, 4)``.
        """

        rel = backbone_coords[:, :, None, :] - backbone_coords[:, None, :, :]
        dist = rel.norm(dim=-1)
        neighbor_dist = dist.mean(dim=-1, keepdim=True)
        curvature = (backbone_coords[:, 1:] - backbone_coords[:, :-1]).norm(dim=-1)
        curvature = F.pad(curvature, (0, 1), value=0.0).unsqueeze(-1)
        scalar_feat = torch.cat(
            [
                neighbor_dist,
                curvature,
                backbone_coords.mean(dim=-1, keepdim=True),
                dist.std(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        scalar = self.scalar_in(scalar_feat)
        # Seed per-node vector channels from the local backbone displacement,
        # broadcast to `vector_dim` channels via a learned per-channel scale.
        local_disp = F.pad(
            backbone_coords[:, 1:] - backbone_coords[:, :-1], (0, 0, 0, 1), value=0.0
        )
        channel_scale = self.vector_in.weight.squeeze(-1)  # (vector_dim,)
        vector = local_disp.unsqueeze(2) * channel_scale.view(1, 1, -1, 1)
        for block in self.gvp_blocks:
            scalar, vector = block(scalar, vector)

        time_bias = self.time_embed(timestep).unsqueeze(1)
        struct_cond = scalar + time_bias

        seq_tok = self.seq_embed(noisy_seq) + time_bias
        for block in self.decoder_blocks:
            seq_tok = block(seq_tok, struct_cond)
        return self.readout(seq_tok)


def build_ribodiffusion() -> nn.Module:
    """Build a compact RiboDiffusion GVP-GNN + transformer denoiser.

    Returns
    -------
    nn.Module
        Random-initialized RiboDiffusion-style model in eval mode.
    """

    return RiboDiffusion().eval()


def example_input_ribodiffusion() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a synthetic RNA backbone, noisy one-hot sequence, and timestep.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(backbone_coords, noisy_seq, timestep)``.
    """

    backbone = torch.cumsum(torch.randn(1, 20, 3) * 0.5, dim=1)
    noisy_seq = torch.softmax(torch.randn(1, 20, 4), dim=-1)
    timestep = torch.full((1, 1), 0.5)
    return backbone, noisy_seq, timestep


# ---------------------------------------------------------------------------
# RibonanzaNet: conv + pair-biased attention + triangular-multiplicative update
# ---------------------------------------------------------------------------


class TriangleMultiplicativeUpdate(nn.Module):
    """Outer-product-mean-style triangular update of the pairwise map."""

    def __init__(self, pair_dim: int, hidden_dim: int) -> None:
        """Initialize the projection and gating layers for the triangular update.

        Parameters
        ----------
        pair_dim:
            Pairwise feature width.
        hidden_dim:
            Hidden width for the left/right projections.
        """

        super().__init__()
        self.norm_in = nn.LayerNorm(pair_dim)
        self.left_proj = nn.Linear(pair_dim, hidden_dim)
        self.right_proj = nn.Linear(pair_dim, hidden_dim)
        self.left_gate = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.Sigmoid())
        self.right_gate = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.Sigmoid())
        self.out_proj = nn.Linear(hidden_dim, pair_dim)
        self.out_norm = nn.LayerNorm(pair_dim)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """Refine the pairwise map via a triangular multiplicative update.

        Parameters
        ----------
        pair:
            Pairwise features, shape ``(B, L, L, P)``.

        Returns
        -------
        torch.Tensor
            Updated pairwise features, shape ``(B, L, L, P)``.
        """

        x = self.norm_in(pair)
        left = self.left_proj(x) * self.left_gate(x)
        right = self.right_proj(x) * self.right_gate(x)
        # "outgoing" triangular update: sum over the shared index k.
        update = torch.einsum("bikc,bjkc->bijc", left, right)
        return pair + self.out_norm(self.out_proj(update))


class RibonanzaNetLayer(nn.Module):
    """One RibonanzaNet encoder layer: conv, pair-biased attention, tri-update."""

    def __init__(self, dim: int, pair_dim: int, heads: int = 4) -> None:
        """Initialize the residual conv, pair-biased attention, and triangular update.

        Parameters
        ----------
        dim:
            Sequence-track feature width.
        pair_dim:
            Pairwise-track feature width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=1)
        self.conv_norm = nn.LayerNorm(dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.pair_bias_proj = nn.Linear(pair_dim, heads)
        self.outer_mean_proj = nn.Linear(dim, pair_dim // 2)
        self.pair_update_proj = nn.Linear(pair_dim // 2, pair_dim)
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.tri_update = TriangleMultiplicativeUpdate(pair_dim, pair_dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, seq: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one conv + pair-biased-attention + triangular-update layer.

        Parameters
        ----------
        seq:
            Sequence-track features, shape ``(B, L, D)``.
        pair:
            Pairwise-track features, shape ``(B, L, L, P)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(seq, pair)`` tracks.
        """

        conv_out = self.conv(seq.transpose(1, 2)).transpose(1, 2)
        seq = self.conv_norm(seq + conv_out)

        n_len = seq.shape[1]
        bias = (
            self.pair_bias_proj(pair)
            .permute(0, 3, 1, 2)
            .reshape(seq.shape[0] * self.heads, n_len, n_len)
        )
        h = self.attn_norm(seq)
        attn_out, _ = self.attn(h, h, h, attn_mask=bias)
        seq = seq + attn_out
        seq = seq + self.ffn(seq)

        outer = self.outer_mean_proj(seq)
        outer_pair = outer[:, :, None, :] * outer[:, None, :, :]
        pair = self.pair_norm(pair + self.pair_update_proj(outer_pair))
        pair = self.tri_update(pair)
        return seq, pair


class RibonanzaNet(nn.Module):
    """Compact RibonanzaNet: conv + pair-biased attention + triangular update stack."""

    def __init__(self, dim: int = 32, pair_dim: int = 16, n_layers: int = 3) -> None:
        """Initialize the nucleotide embedding, pairwise seed, and encoder stack.

        Parameters
        ----------
        dim:
            Sequence-track feature width.
        pair_dim:
            Pairwise-track feature width.
        n_layers:
            Number of stacked encoder layers.
        """

        super().__init__()
        self.embed = nn.Embedding(5, dim)  # A, C, G, U, pad
        self.pos_embed = nn.Embedding(64, dim)
        self.pair_seed = nn.Linear(2 * dim, pair_dim)
        self.layers = nn.ModuleList([RibonanzaNetLayer(dim, pair_dim) for _ in range(n_layers)])
        self.reactivity_head = nn.Linear(dim, 2)  # DMS + 2A3 chemical-mapping channels

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict per-nucleotide chemical-mapping reactivity profiles.

        Parameters
        ----------
        tokens:
            Nucleotide token ids in ``{0..4}``, shape ``(B, L)``.

        Returns
        -------
        torch.Tensor
            Per-nucleotide reactivity predictions, shape ``(B, L, 2)``.
        """

        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).clamp(max=63)
        seq = self.embed(tokens) + self.pos_embed(positions).unsqueeze(0)
        pair = self.pair_seed(
            torch.cat(
                [
                    seq[:, :, None, :].expand(-1, -1, length, -1),
                    seq[:, None, :, :].expand(-1, length, -1, -1),
                ],
                dim=-1,
            )
        )
        for layer in self.layers:
            seq, pair = layer(seq, pair)
        return self.reactivity_head(seq)


def build_ribonanzanet() -> nn.Module:
    """Build a compact RibonanzaNet conv+pair-attention+triangular-update model.

    Returns
    -------
    nn.Module
        Random-initialized RibonanzaNet-style model in eval mode.
    """

    return RibonanzaNet().eval()


def example_input_ribonanzanet() -> torch.Tensor:
    """Create a synthetic nucleotide token sequence.

    Returns
    -------
    torch.Tensor
        Token ids of shape ``(1, 24)``.
    """

    return torch.randint(0, 4, (1, 24))


MENAGERIE_ENTRIES = [
    ("ProteinSolver", "build_proteinsolver", "example_input_proteinsolver", "2020", "BIO"),
    ("RaptorX-Contact", "build_raptorx_contact", "example_input_raptorx_contact", "2017", "BIO"),
    ("RaptorX-Property", "build_raptorx_property", "example_input_raptorx_property", "2016", "BIO"),
    ("RFdiffusionAA", "build_rfdiffusionaa", "example_input_rfdiffusionaa", "2024", "BIO"),
    ("RiboDiffusion", "build_ribodiffusion", "example_input_ribodiffusion", "2024", "BIO"),
    ("RibonanzaNet", "build_ribonanzanet", "example_input_ribonanzanet", "2023", "BIO"),
]
