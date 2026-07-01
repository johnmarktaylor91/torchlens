"""Wave 7 batch 19 menagerie classics: molecular-simulation / mass-spectrometry /
retrosynthesis / NMR / dynamical-systems / particle-physics-tracking family.

Sources checked (repo_url / desc_source columns of the build queue, web research
2026-07-01; no cloning, no pip installs beyond the base env):

  - GPUMD (NEP -- neuroevolution potential): https://github.com/brucefan1983/GPUMD ;
    Fan et al. 2021 PRB "Neuroevolution machine learning potentials: Combining
    high accuracy and low cost in atomistic simulations and application to
    heat transport" + Fan, Wang, Ying et al. 2022 (arXiv:2205.10046) "GPUMD: A
    package for constructing accurate machine-learned potentials...". GPUMD
    itself is a CUDA/C++ MD engine (no Python nn.Module to port), but the
    learned-potential architecture it trains (NEP) is precisely documented in
    the papers and https://gpumd.org/potentials/nep.html: for each atom, a
    per-neighbor two-body radial descriptor built from a linear combination of
    Chebyshev polynomials (order-``n_max`` basis) of a smoothly-cutoff radial
    function, summed over neighbors into fixed-length radial + angular
    (three/four-body ACE-style) descriptor blocks, concatenated into one
    per-atom feature vector, and fed through a small *shared* MLP (one hidden
    layer, "neuroevolution"-trained in the original via a separable natural
    evolution strategy, but architecturally an ordinary feedforward net) that
    outputs a scalar per-atom energy; total potential energy is the sum over
    atoms. Reproduced here as a compact per-atom Chebyshev-radial + 3-body
    angular descriptor (fixed neighbor list, cutoff-smoothed) feeding a
    shared 2-layer MLP energy head with the atom-sum readout -- the
    Behler-Parrinello-style "local descriptor -> shared per-atom MLP -> sum"
    design that is NEP's central architectural idea, without the CUDA kernels
    or the evolutionary optimizer (irrelevant to forward-pass architecture).

  - GrAFF-MS: https://github.com/murphy17/graff-ms ; Murphy et al., ICML 2023,
    "Efficiently predicting high resolution mass spectra with graph neural
    networks". Confirmed from ``src/graff.py`` (``GrAFF`` class) and
    ``src/gnn.py`` (``GINE``/``GINELayer``/``GINEEdgeLayer``/``SignNet``): a
    molecular graph is embedded (atom/bond one-hot -> MLP), combined with a
    *sign-invariant Laplacian-eigenvector positional encoding* (SignNet:
    ``phi(v) + phi(-v)`` fed through a shared MLP ``phi`` then a second MLP
    ``rho``, which forces invariance to the sign ambiguity of eigenvectors),
    and passed through a stack of GINE-style message-passing layers that
    update *both* node and edge features each round (``GINELayer`` updates
    atoms via edge-gated sum-aggregation, ``GINEEdgeLayer`` updates each bond
    from its two endpoint atoms). An attention-weighted sum-pool
    (``softmax`` gate per atom, then weighted sum) collapses the graph to one
    molecule vector, which is combined with instrument/adduct covariates and
    decoded (residual MLP stack) into a *distribution over a fixed vocabulary
    of candidate molecular-formula peaks* (the paper's key idea: predicting a
    probability distribution over discrete formula/peak bins rather than a
    dense binned spectrum), plus a small isotope-shift head. Reproduced here
    as a compact GINE node+edge message-passing GNN with sign-invariant
    eigenvector positional encoding, attention pooling, and a vocabulary
    classification head -- the two hallmark ideas (edge-updating GINE +
    sign-invariant SignNet PE) preserved; ``torch_geometric``/``torch_scatter``
    dependencies replaced by dense fixed-graph index-based scatter ops on a
    small fixed molecule so the forward pass is torchlens-traceable.

  - Graph2SMILES: https://github.com/coleygroup/Graph2SMILES ; Tu & Coley,
    JCIM 2022, "Permutation Invariant Graph-to-Sequence Model for Template-
    Free Retrosynthesis and Reaction Prediction". Confirmed from
    ``models/graph2seq_series_rel.py`` (``Graph2SeqSeriesRel``),
    ``models/graphfeat.py`` (``GraphFeatEncoder``), and
    ``models/attention_xl.py`` (``MultiHeadedRelAttention``): a directed
    message-passing graph encoder (D-MPNN-style, matching the architecture
    already captured in this catalog's ``chemprop_dmpnn.py``) produces
    per-atom embeddings, which are then refined by a stack of
    *graph-distance-conditioned relative self-attention* layers (Transformer-
    XL-style ``a+c``/``b+d`` decomposition, where the relative-position term
    is looked up from a *bucketed graph shortest-path-distance matrix*
    instead of a sequence offset -- the paper's distinguishing idea of
    injecting 2D molecular-graph topology into a Transformer's attention
    bias), and finally a standard Transformer decoder autoregressively emits
    SMILES tokens for the product/precursor. Reproduced here as a compact
    D-MPNN atom encoder + graph-distance-bucketed relative-attention encoder
    + Transformer decoder, with a small fixed reaction graph and target
    token sequence as the example input (OpenNMT dependency dropped in favor
    of plain ``nn.TransformerDecoderLayer``, decoder semantics unchanged).

  - GT-NMR: https://github.com/AnanWu-XMU/GT-NMR ; Zhang et al., J.
    Cheminformatics 2024 (doi 10.1186/s13321-024-00927-9), "GT-NMR: a novel
    graph transformer-based approach for accurate prediction of NMR chemical
    shifts". Repo ships only data/README (no public source at audit time);
    architecture confirmed from the paper (PMC11590296): atom/bond features
    are embedded and combined with a *relative random-walk-probability
    (RRWP) positional encoding* (``P = [I, M, M^2, ..., M^{K-1}]`` built from
    the random-walk transition matrix, diagonal used as node PE, off-diagonal
    used as an edge/attention-bias PE), fed through graph-transformer blocks
    whose attention injects edge features and the RRWP bias directly into
    the pre-softmax score (``sigma(rho(W_Q x_i + W_K x_j * W_E^w w_ij + W_E^b
    e_ij))`` with the signed-square-root activation ``rho(x) = relu(sqrt(x))
    - relu(sqrt(-x))``) and rescales node outputs by a *log-degree adaptive
    scaler* (``x_i' = x_i * theta1 + log(1+d_i) * x_i * theta2``) to retain
    graph-topology sensitivity that plain attention would otherwise wash
    out, followed by a 3-layer MLP node-regression head predicting one
    chemical-shift scalar per atom. Reproduced here as a compact RRWP-PE +
    edge-conditioned-attention + degree-scaler graph transformer with the
    exact signed-sqrt activation and degree-rescale formula, on a small
    fixed molecular graph.

  - Hamiltonian Neural Network (HNN): https://github.com/greydanus/hamiltonian-nn ;
    Greydanus, Dzamba & Yosinski, NeurIPS 2019 (arXiv:1806.01242), "Hamiltonian
    Neural Networks". Confirmed from ``hnn.py`` (``HNN.time_derivative``) and
    ``nn_models.py`` (``MLP``): a plain MLP maps phase-space coordinates
    ``(q, p)`` to a *scalar* Hamiltonian energy ``H(q, p)``; the vector field
    used to advance the system is not the MLP's output directly but its
    *symplectic gradient*, computed via ``torch.autograd.grad`` with
    ``create_graph=True`` and then multiplied by the canonical symplectic
    permutation matrix (``dq/dt = dH/dp``, ``dp/dt = -dH/dq``), so a single
    scalar-valued network yields a divergence-free, energy-conserving vector
    field purely through this double-backward construction -- the paper's
    entire distinctive mechanism. Reproduced here exactly: an MLP outputs a
    scalar Hamiltonian, ``forward`` returns the on-the-fly symplectic
    gradient via ``autograd.grad(..., create_graph=True)`` and the canonical
    permutation matrix, matching ``assume_canonical_coords=True`` in the
    original.

  - HEPTrkX: https://github.com/HEPTrkX/heptrkx-gnn-tracking ; HEP.TrkX
    collaboration (Farrell et al.), 2018-2020 (arXiv:2012.01249 summarizes
    the program), "Novel deep learning methods for track reconstruction" /
    the associated ``GNNSegmentClassifier``. Confirmed verbatim from
    ``models/gnn.py``: hits are graph nodes; candidate track segments
    (edges) are represented via two sparse bipartite incidence matrices
    ``Ri``/``Ro`` (edge-to-incoming-node, edge-to-outgoing-node). An
    ``EdgeNetwork`` gathers each edge's two endpoint features (via
    ``Ri^T X`` / ``Ro^T X`` batched matmuls) and scores it with a sigmoid
    MLP; a ``NodeNetwork`` re-aggregates edge-weighted neighbor messages
    back onto each node (``Ri @ (Ro^T X)`` etc., weighted by the current
    edge scores) and updates node features with a residual MLP; edge and
    node networks are iterated ``n_iters`` times with the raw input features
    re-concatenated (shortcut) at every iteration, and a final
    ``EdgeNetwork`` pass outputs the track-segment classification logits.
    Reproduced here verbatim (this is a small, already-compact reference
    architecture) with a small fixed hit graph and Ri/Ro incidence matrices
    as the example input.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# GPUMD / NEP: neuroevolution potential (Chebyshev radial + 3-body angular
# per-atom descriptor -> shared per-atom MLP energy head -> atom-sum readout)
# ---------------------------------------------------------------------------


class NEPDescriptor(nn.Module):
    """Per-atom Chebyshev radial + 3-body angular descriptor (NEP-style)."""

    def __init__(
        self, n_radial_basis: int = 8, n_angular_basis: int = 6, cutoff: float = 4.0
    ) -> None:
        super().__init__()
        self.n_radial_basis = n_radial_basis
        self.n_angular_basis = n_angular_basis
        self.cutoff = cutoff

    def _fc(self, r: Tensor) -> Tensor:
        """Smooth cosine cutoff function f_c(r) on [0, cutoff]."""

        x = (r / self.cutoff).clamp(max=1.0)
        return 0.5 * (torch.cos(math.pi * x) + 1.0)

    def _chebyshev(self, x: Tensor, order: int) -> Tensor:
        """Stack of Chebyshev polynomials T_0..T_{order-1} evaluated at x."""

        polys = [torch.ones_like(x), x]
        for _ in range(2, order):
            polys.append(2.0 * x * polys[-1] - polys[-2])
        return torch.stack(polys[:order], dim=-1)

    def forward(self, positions: Tensor, neighbor_index: Tensor) -> Tensor:
        """Build per-atom radial + angular descriptors.

        Parameters
        ----------
        positions:
            Atom coordinates, shape ``(n_atoms, 3)``.
        neighbor_index:
            Fixed neighbor-atom indices per atom, shape ``(n_atoms, n_neigh)``.

        Returns
        -------
        Tensor
            Per-atom descriptor, shape ``(n_atoms, n_radial_basis + n_angular_basis)``.
        """

        n_atoms, n_neigh = neighbor_index.shape
        center = positions.unsqueeze(1)  # (n_atoms, 1, 3)
        neigh_pos = positions[neighbor_index]  # (n_atoms, n_neigh, 3)
        rij = neigh_pos - center
        r = rij.norm(dim=-1).clamp(min=1e-6)  # (n_atoms, n_neigh)
        x_scaled = 2.0 * (r / self.cutoff).clamp(max=1.0) - 1.0
        cheb = self._chebyshev(x_scaled, self.n_radial_basis)  # (n_atoms, n_neigh, n_radial)
        fc = self._fc(r).unsqueeze(-1)
        radial = (cheb * fc).sum(dim=1)  # sum over neighbors -> (n_atoms, n_radial)

        # 3-body angular part: pairs of neighbors (j, k) around center i.
        unit = rij / r.unsqueeze(-1)
        cos_ijk = torch.einsum("nid,njd->nij", unit, unit)  # (n_atoms, n_neigh, n_neigh)
        leg = self._chebyshev(cos_ijk.clamp(-1.0, 1.0), self.n_angular_basis)
        fc_pair = (fc.squeeze(-1).unsqueeze(2) * fc.squeeze(-1).unsqueeze(1)).unsqueeze(-1)
        angular = (leg * fc_pair).sum(dim=(1, 2))  # (n_atoms, n_angular)

        return torch.cat([radial, angular], dim=-1)


class NEP(nn.Module):
    """Neuroevolution potential: local descriptor -> shared per-atom MLP -> sum."""

    def __init__(self, n_radial_basis: int = 8, n_angular_basis: int = 6, hidden: int = 32) -> None:
        super().__init__()
        self.descriptor = NEPDescriptor(n_radial_basis, n_angular_basis)
        in_dim = n_radial_basis + n_angular_basis
        self.energy_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, positions: Tensor, neighbor_index: Tensor) -> Tensor:
        """Predict total potential energy for a small atomic configuration.

        Parameters
        ----------
        positions:
            Atom coordinates, shape ``(n_atoms, 3)``.
        neighbor_index:
            Fixed neighbor-atom indices per atom, shape ``(n_atoms, n_neigh)``.

        Returns
        -------
        Tensor
            Scalar total potential energy, shape ``(1,)``.
        """

        desc = self.descriptor(positions, neighbor_index)
        per_atom_energy = self.energy_mlp(desc)  # (n_atoms, 1)
        return per_atom_energy.sum().unsqueeze(0)


def build_gpumd_nep() -> nn.Module:
    """Build a compact NEP potential.

    Returns
    -------
    nn.Module
        Random-initialized NEP in eval mode.
    """

    return NEP().eval()


def example_input_gpumd_nep() -> tuple[Tensor, Tensor]:
    """Create a small atomic configuration (8 atoms, 4 neighbors each).

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(positions, neighbor_index)``.
    """

    torch.manual_seed(0)
    n_atoms, n_neigh = 8, 4
    positions = torch.randn(n_atoms, 3) * 2.0
    neighbor_index = torch.randint(0, n_atoms, (n_atoms, n_neigh))
    return positions, neighbor_index


# ---------------------------------------------------------------------------
# GrAFF-MS: GINE node+edge message passing + sign-invariant SignNet PE +
# attention pooling + vocabulary-distribution mass-spectrum head
# ---------------------------------------------------------------------------


class SignNet(nn.Module):
    """Sign-invariant Laplacian-eigenvector positional encoding."""

    def __init__(self, num_eigs: int, phi_dim: int, rho_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(2, phi_dim),
            nn.SiLU(),
            nn.Linear(phi_dim, phi_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(phi_dim, rho_dim),
            nn.SiLU(),
            nn.Linear(rho_dim, embed_dim),
        )
        self.num_eigs = num_eigs

    def forward(self, eigvecs: Tensor, eigvals: Tensor) -> Tensor:
        """Encode eigenvectors sign-invariantly.

        Parameters
        ----------
        eigvecs:
            Per-node Laplacian eigenvector loadings, shape ``(n_nodes, num_eigs)``.
        eigvals:
            Broadcast eigenvalues, shape ``(n_nodes, num_eigs)``.

        Returns
        -------
        Tensor
            Per-node positional embedding, shape ``(n_nodes, embed_dim)``.
        """

        pos = self.phi(torch.stack([eigvecs, eigvals], dim=-1))
        neg = self.phi(torch.stack([-eigvecs, eigvals], dim=-1))
        return self.rho((pos + neg).sum(dim=1))


class GINELayer(nn.Module):
    """GINE node-update layer: edge-gated neighbor sum-aggregation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, e: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        """Update node features by aggregating (neighbor + edge) messages."""

        msg = self.act(x[src] + e)  # (n_edges, dim)
        agg = torch.zeros_like(x).index_add(0, dst, msg)
        return self.norm(x + self.act(self.lin(agg)))


class GINEEdgeLayer(nn.Module):
    """GINE edge-update layer: refresh each bond from its two endpoint atoms."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(3 * dim, dim)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, e: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        """Update edge features from concatenated (edge, src atom, dst atom)."""

        de = torch.cat([e, x[src], x[dst]], dim=-1)
        return self.norm(e + self.act(self.lin(de)))


class GrAFF(nn.Module):
    """GrAFF-MS: GINE encoder + SignNet PE + attention pool + vocab spectrum head."""

    def __init__(
        self,
        atom_dim: int = 12,
        bond_dim: int = 6,
        model_dim: int = 32,
        depth: int = 3,
        num_eigs: int = 4,
        vocab_size: int = 24,
    ) -> None:
        super().__init__()
        self.node_emb = nn.Linear(atom_dim, model_dim)
        self.edge_emb = nn.Linear(bond_dim, model_dim)
        self.signnet = SignNet(num_eigs, phi_dim=16, rho_dim=model_dim, embed_dim=model_dim)
        self.node_layers = nn.ModuleList([GINELayer(model_dim) for _ in range(depth)])
        self.edge_layers = nn.ModuleList([GINEEdgeLayer(model_dim) for _ in range(depth)])
        self.attn = nn.Linear(model_dim, 1)
        self.decoder = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.clf = nn.Linear(model_dim, vocab_size)

    def forward(
        self,
        atom_feat: Tensor,
        bond_feat: Tensor,
        src: Tensor,
        dst: Tensor,
        eigvecs: Tensor,
        eigvals: Tensor,
    ) -> Tensor:
        """Predict a probability distribution over the mass-spectrum peak vocabulary.

        Parameters
        ----------
        atom_feat:
            Atom one-hot features, shape ``(n_atoms, atom_dim)``.
        bond_feat:
            Directed bond one-hot features, shape ``(n_edges, bond_dim)``.
        src, dst:
            Directed edge endpoint indices, each shape ``(n_edges,)``.
        eigvecs, eigvals:
            Laplacian eigenvector loadings / eigenvalues, shape ``(n_atoms, num_eigs)``.

        Returns
        -------
        Tensor
            Log-probabilities over the peak vocabulary, shape ``(vocab_size,)``.
        """

        x = self.node_emb(atom_feat) + self.signnet(eigvecs, eigvals)
        e = self.edge_emb(bond_feat)
        for node_layer, edge_layer in zip(self.node_layers, self.edge_layers):
            x = node_layer(x, e, src, dst)
            e = edge_layer(x, e, src, dst)
        weights = torch.softmax(self.attn(x), dim=0)
        pooled = (x * weights).sum(dim=0)
        z = self.decoder(pooled)
        return F.log_softmax(self.clf(z), dim=-1)


def build_graff_ms() -> nn.Module:
    """Build a compact GrAFF-MS.

    Returns
    -------
    nn.Module
        Random-initialized GrAFF-MS in eval mode.
    """

    return GrAFF().eval()


def example_input_graff_ms() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create a small fixed molecular graph (6-atom ring, directed bonds).

    Returns
    -------
    tuple[Tensor, ...]
        ``(atom_feat, bond_feat, src, dst, eigvecs, eigvals)``.
    """

    torch.manual_seed(0)
    n_atoms, num_eigs = 6, 4
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    edges = ring + [(j, i) for i, j in ring]
    src = torch.tensor([i for i, _ in edges], dtype=torch.long)
    dst = torch.tensor([j for _, j in edges], dtype=torch.long)
    atom_feat = torch.randn(n_atoms, 12)
    bond_feat = torch.randn(len(edges), 6)
    eigvecs = torch.randn(n_atoms, num_eigs)
    eigvals = torch.rand(1, num_eigs).expand(n_atoms, num_eigs)
    return atom_feat, bond_feat, src, dst, eigvecs, eigvals


# ---------------------------------------------------------------------------
# Graph2SMILES: D-MPNN atom encoder + graph-distance-bucketed relative
# self-attention + Transformer decoder (autoregressive SMILES generation)
# ---------------------------------------------------------------------------


class DMPNNAtomEncoder(nn.Module):
    """Directed message-passing atom encoder (GraphFeatEncoder-style)."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden: int = 48, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        self.w_i = nn.Linear(atom_dim + bond_dim, hidden, bias=False)
        self.w_h = nn.Linear(hidden, hidden, bias=False)
        self.w_o = nn.Linear(atom_dim + hidden, hidden)

    def forward(self, atom_feat: Tensor, bond_msg_feat: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        """Encode atoms via directed-bond message passing.

        Parameters
        ----------
        atom_feat:
            Atom features, shape ``(n_atoms, atom_dim)``.
        bond_msg_feat:
            ``[atom_feat[src]; bond_feat]`` per directed bond, shape ``(n_bonds, atom_dim+bond_dim)``.
        src, dst:
            Directed-bond endpoint indices, each shape ``(n_bonds,)``.

        Returns
        -------
        Tensor
            Per-atom encoding, shape ``(n_atoms, hidden)``.
        """

        n_bonds = bond_msg_feat.shape[0]
        m0 = F.relu(self.w_i(bond_msg_feat))  # (n_bonds, hidden) directed msg i->j
        m = m0
        for _ in range(self.depth):
            incoming = torch.zeros_like(m).index_add(0, dst, m)  # sum over msgs into each atom
            agg = incoming[src]  # message reaching each bond's tail atom
            m = F.relu(m0 + self.w_h(agg))
        incoming_final = torch.zeros(atom_feat.shape[0], m.shape[1], device=m.device).index_add(
            0, dst, m
        )
        return F.relu(self.w_o(torch.cat([atom_feat, incoming_final], dim=-1)))


class DistanceRelativeAttention(nn.Module):
    """Graph-shortest-path-distance-conditioned relative self-attention (Transformer-XL style)."""

    def __init__(self, dim: int, heads: int, n_buckets: int) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.rel_emb = nn.Embedding(n_buckets, dim)
        self.u = nn.Parameter(torch.zeros(heads, self.dim_head))
        self.v_bias = nn.Parameter(torch.zeros(heads, self.dim_head))

    def forward(self, x: Tensor, distance_bucket: Tensor) -> Tensor:
        """Apply one relative-attention block.

        Parameters
        ----------
        x:
            Atom encodings, shape ``(n_atoms, dim)``.
        distance_bucket:
            Bucketed pairwise graph shortest-path distances, shape ``(n_atoms, n_atoms)``.

        Returns
        -------
        Tensor
            Updated atom encodings, shape ``(n_atoms, dim)``.
        """

        n = x.shape[0]
        q = self.q(x).view(n, self.heads, self.dim_head)
        k = self.k(x).view(n, self.heads, self.dim_head)
        v = self.v(x).view(n, self.heads, self.dim_head)
        rel = self.rel_emb(distance_bucket).view(n, n, self.heads, self.dim_head)

        a_c = torch.einsum("qhd,khd->hqk", q + self.u, k)
        b_d = torch.einsum("qhd,qkhd->hqk", q + self.v_bias, rel)
        scores = (a_c + b_d) / math.sqrt(self.dim_head)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn, v).reshape(n, -1)
        return x + self.out(out)


class Graph2SMILES(nn.Module):
    """Graph2SMILES: D-MPNN encoder + distance-relative-attention + Transformer decoder."""

    def __init__(
        self,
        atom_dim: int = 16,
        bond_dim: int = 8,
        hidden: int = 48,
        vocab_size: int = 40,
        n_buckets: int = 8,
    ) -> None:
        super().__init__()
        self.mpn = DMPNNAtomEncoder(atom_dim, bond_dim, hidden)
        self.attn_encoder = DistanceRelativeAttention(hidden, heads=4, n_buckets=n_buckets)
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=4 * hidden, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.output_layer = nn.Linear(hidden, vocab_size)

    def forward(
        self,
        atom_feat: Tensor,
        bond_msg_feat: Tensor,
        src: Tensor,
        dst: Tensor,
        distance_bucket: Tensor,
        tgt_tokens: Tensor,
    ) -> Tensor:
        """Encode a reaction graph and decode a SMILES token sequence.

        Parameters
        ----------
        atom_feat:
            Atom features, shape ``(n_atoms, atom_dim)``.
        bond_msg_feat:
            Directed-bond message features, shape ``(n_bonds, atom_dim+bond_dim)``.
        src, dst:
            Directed-bond endpoint indices, each shape ``(n_bonds,)``.
        distance_bucket:
            Bucketed atom-pair graph distances, shape ``(n_atoms, n_atoms)``.
        tgt_tokens:
            Target SMILES token ids (decoder input), shape ``(1, tgt_len)``.

        Returns
        -------
        Tensor
            Next-token logits, shape ``(1, tgt_len, vocab_size)``.
        """

        hatom = self.mpn(atom_feat, bond_msg_feat, src, dst)
        memory = self.attn_encoder(hatom, distance_bucket).unsqueeze(0)  # (1, n_atoms, hidden)
        tgt = self.tok_emb(tgt_tokens)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        dec_out = self.decoder(tgt, memory, tgt_mask=causal_mask)
        return self.output_layer(dec_out)


def build_graph2smiles() -> nn.Module:
    """Build a compact Graph2SMILES.

    Returns
    -------
    nn.Module
        Random-initialized Graph2SMILES in eval mode.
    """

    return Graph2SMILES().eval()


def example_input_graph2smiles() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create a small fixed reaction graph and target token sequence.

    Returns
    -------
    tuple[Tensor, ...]
        ``(atom_feat, bond_msg_feat, src, dst, distance_bucket, tgt_tokens)``.
    """

    torch.manual_seed(0)
    n_atoms, n_buckets = 6, 8
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    edges = ring + [(j, i) for i, j in ring]
    src = torch.tensor([i for i, _ in edges], dtype=torch.long)
    dst = torch.tensor([j for _, j in edges], dtype=torch.long)
    atom_feat = torch.randn(n_atoms, 16)
    bond_msg_feat = torch.randn(len(edges), 16 + 8)
    distance_bucket = torch.randint(0, n_buckets, (n_atoms, n_atoms))
    distance_bucket = torch.maximum(distance_bucket, distance_bucket.t())  # symmetric distances
    tgt_tokens = torch.randint(0, 40, (1, 5))
    return atom_feat, bond_msg_feat, src, dst, distance_bucket, tgt_tokens


# ---------------------------------------------------------------------------
# GT-NMR: RRWP positional encoding + edge-conditioned graph-transformer
# (signed-sqrt attention + log-degree adaptive scaler) + node regression head
# ---------------------------------------------------------------------------


def _signed_sqrt(x: Tensor) -> Tensor:
    """Signed-square-root activation: relu(sqrt(x)) - relu(sqrt(-x))."""

    return F.relu(x.clamp(min=0.0)).sqrt() - F.relu((-x).clamp(min=0.0)).sqrt()


class GTNMRLayer(nn.Module):
    """One graph-transformer block: edge-conditioned attention + degree scaler."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_ew = nn.Linear(dim, dim)
        self.w_eb = nn.Linear(dim, dim)
        self.w_a = nn.Linear(dim, heads)
        self.w_o = nn.Linear(dim, dim)
        self.theta1 = nn.Parameter(torch.ones(dim))
        self.theta2 = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, edge_feat: Tensor, log_degree: Tensor) -> Tensor:
        """Apply edge-conditioned attention with a log-degree adaptive scaler.

        Parameters
        ----------
        x:
            Node features, shape ``(n, dim)``.
        edge_feat:
            Dense pairwise edge/PE features, shape ``(n, n, dim)``.
        log_degree:
            ``log(1 + degree)`` per node, shape ``(n, 1)``.

        Returns
        -------
        Tensor
            Updated node features, shape ``(n, dim)``.
        """

        n = x.shape[0]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        e_hat = _signed_sqrt(
            q.unsqueeze(1) + k.unsqueeze(0) * self.w_ew(edge_feat) + self.w_eb(edge_feat)
        )  # (n, n, dim)
        scores = self.w_a(e_hat).permute(2, 0, 1)  # (heads, n, n)
        attn = torch.softmax(scores, dim=-1)
        v_heads = v.view(n, self.heads, self.dim_head)
        e_heads = e_hat.view(n, n, self.heads, self.dim_head)
        msg = torch.einsum("hqk,khd->qhd", attn, v_heads) + torch.einsum(
            "hqk,qkhd->qhd", attn, e_heads
        )
        out = self.w_o(msg.reshape(n, -1))
        out = out * self.theta1 + log_degree * out * self.theta2
        x = self.norm(x + out)
        x = self.ffn_norm(x + self.ffn(x))
        return x


class GTNMR(nn.Module):
    """GT-NMR: RRWP-PE graph transformer with node-level chemical-shift regression."""

    def __init__(
        self, atom_dim: int = 12, dim: int = 32, depth: int = 2, heads: int = 4, rrwp_k: int = 6
    ) -> None:
        super().__init__()
        self.node_emb = nn.Linear(atom_dim + rrwp_k, dim)
        self.edge_emb = nn.Linear(rrwp_k, dim)
        self.layers = nn.ModuleList([GTNMRLayer(dim, heads) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, atom_feat: Tensor, rrwp: Tensor, log_degree: Tensor) -> Tensor:
        """Predict a chemical-shift scalar per atom.

        Parameters
        ----------
        atom_feat:
            Atom features, shape ``(n_atoms, atom_dim)``.
        rrwp:
            Relative random-walk-probability tensor ``P``, shape ``(n_atoms, n_atoms, rrwp_k)``;
            the diagonal ``rrwp[i, i]`` is used as the node positional encoding.
        log_degree:
            ``log(1 + degree)`` per atom, shape ``(n_atoms, 1)``.

        Returns
        -------
        Tensor
            Per-atom chemical-shift predictions, shape ``(n_atoms, 1)``.
        """

        n = atom_feat.shape[0]
        node_pe = rrwp[torch.arange(n), torch.arange(n)]  # diagonal -> (n_atoms, rrwp_k)
        x = self.node_emb(torch.cat([atom_feat, node_pe], dim=-1))
        edge_feat = self.edge_emb(rrwp)  # (n_atoms, n_atoms, dim)
        for layer in self.layers:
            x = layer(x, edge_feat, log_degree)
        return self.head(x)


def build_gt_nmr() -> nn.Module:
    """Build a compact GT-NMR.

    Returns
    -------
    nn.Module
        Random-initialized GT-NMR in eval mode.
    """

    return GTNMR().eval()


def example_input_gt_nmr() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small fixed molecular graph with an RRWP positional encoding.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(atom_feat, rrwp, log_degree)``.
    """

    torch.manual_seed(0)
    n_atoms, rrwp_k = 6, 6
    ring_adj = torch.zeros(n_atoms, n_atoms)
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    for i, j in ring:
        ring_adj[i, j] = 1.0
        ring_adj[j, i] = 1.0
    degree = ring_adj.sum(dim=-1, keepdim=True)
    transition = ring_adj / degree.clamp(min=1.0)
    powers = [torch.eye(n_atoms)]
    for _ in range(rrwp_k - 1):
        powers.append(powers[-1] @ transition)
    rrwp = torch.stack(powers, dim=-1)  # (n_atoms, n_atoms, rrwp_k)
    atom_feat = torch.randn(n_atoms, 12)
    log_degree = torch.log1p(degree)
    return atom_feat, rrwp, log_degree


# ---------------------------------------------------------------------------
# Hamiltonian Neural Network: scalar-Hamiltonian MLP + symplectic gradient
# via double-backward autograd (dq/dt = dH/dp, dp/dt = -dH/dq)
# ---------------------------------------------------------------------------


class HNN(nn.Module):
    """Hamiltonian Neural Network: MLP predicts H(q, p); forward returns its symplectic gradient."""

    def __init__(self, input_dim: int = 2, hidden: int = 64) -> None:
        super().__init__()
        assert input_dim % 2 == 0, "HNN expects canonical (q, p) coordinates in pairs."
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, 1, bias=False)
        half = input_dim // 2
        permutation = torch.cat([torch.eye(input_dim)[half:], -torch.eye(input_dim)[:half]])
        self.register_buffer("permutation", permutation)

    def hamiltonian(self, x: Tensor) -> Tensor:
        """Scalar Hamiltonian energy H(q, p) for each phase-space point."""

        h = torch.tanh(self.linear1(x))
        h = torch.tanh(self.linear2(h))
        return self.linear3(h)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the symplectic gradient (time derivative) of phase-space state ``x``.

        Parameters
        ----------
        x:
            Canonical coordinates ``(q, p)``, shape ``(batch, input_dim)``. Must allow
            gradient tracking.

        Returns
        -------
        Tensor
            Time derivative ``dx/dt = (dH/dp, -dH/dq)``, shape ``(batch, input_dim)``.
        """

        x = x.requires_grad_(True)
        h = self.hamiltonian(x)
        (grad_h,) = torch.autograd.grad(h.sum(), x, create_graph=True)
        return grad_h @ self.permutation


def build_hamiltonian_nn() -> nn.Module:
    """Build a compact Hamiltonian Neural Network.

    Returns
    -------
    nn.Module
        Random-initialized HNN in eval mode.
    """

    return HNN().eval()


def example_input_hamiltonian_nn() -> Tensor:
    """Create a small batch of 2D canonical phase-space coordinates (pendulum q, p).

    Returns
    -------
    Tensor
        Shape ``(8, 2)``.
    """

    torch.manual_seed(0)
    return torch.randn(8, 2)


# ---------------------------------------------------------------------------
# HEPTrkX: bipartite (Ri/Ro) incidence-matrix edge/node message-passing GNN
# for particle-track segment classification
# ---------------------------------------------------------------------------


class EdgeNetwork(nn.Module):
    """Scores each candidate track segment (edge) from its two endpoint hits."""

    def __init__(self, input_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, r_in: Tensor, r_out: Tensor) -> Tensor:
        """Score edges given incidence matrices.

        Parameters
        ----------
        x:
            Hit (node) features, shape ``(n_hits, input_dim)``.
        r_in, r_out:
            Bipartite incidence matrices, each shape ``(n_hits, n_edges)``.

        Returns
        -------
        Tensor
            Per-edge score, shape ``(n_edges,)``.
        """

        b_out = r_out.t() @ x
        b_in = r_in.t() @ x
        b = torch.cat([b_out, b_in], dim=-1)
        return self.network(b).squeeze(-1)


class NodeNetwork(nn.Module):
    """Re-aggregates edge-weighted neighbor messages back onto each hit."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, e: Tensor, r_in: Tensor, r_out: Tensor) -> Tensor:
        """Update hit features from edge-weighted incoming/outgoing neighbor sums."""

        b_out = r_out.t() @ x
        b_in = r_in.t() @ x
        r_w_out = r_out * e.unsqueeze(0)
        r_w_in = r_in * e.unsqueeze(0)
        m_in = r_w_in @ b_out
        m_out = r_w_out @ b_in
        m = torch.cat([m_in, m_out, x], dim=-1)
        return self.network(m)


class HEPTrkX(nn.Module):
    """HEPTrkX segment classifier: iterated edge network + node network on a hit graph."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 8, n_iters: int = 3) -> None:
        super().__init__()
        self.n_iters = n_iters
        self.input_network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.edge_network = EdgeNetwork(input_dim + hidden_dim, hidden_dim)
        self.node_network = NodeNetwork(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: Tensor, r_in: Tensor, r_out: Tensor) -> Tensor:
        """Classify each candidate track segment as real/fake.

        Parameters
        ----------
        x:
            Hit coordinates, shape ``(n_hits, input_dim)``.
        r_in, r_out:
            Bipartite incidence matrices, each shape ``(n_hits, n_edges)``.

        Returns
        -------
        Tensor
            Per-edge (track-segment) score, shape ``(n_edges,)``.
        """

        h = self.input_network(x)
        h = torch.cat([h, x], dim=-1)
        for _ in range(self.n_iters):
            e = self.edge_network(h, r_in, r_out)
            h = self.node_network(h, e, r_in, r_out)
            h = torch.cat([h, x], dim=-1)
        return self.edge_network(h, r_in, r_out)


def build_heptrkx() -> nn.Module:
    """Build a compact HEPTrkX segment classifier.

    Returns
    -------
    nn.Module
        Random-initialized HEPTrkX in eval mode.
    """

    return HEPTrkX().eval()


def example_input_heptrkx() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small fixed hit graph with bipartite incidence matrices.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(hit_coords, r_in, r_out)`` for 10 hits and 12 candidate segments.
    """

    torch.manual_seed(0)
    n_hits, n_edges = 10, 12
    hit_coords = torch.randn(n_hits, 3)
    src = torch.randint(0, n_hits, (n_edges,))
    dst = torch.randint(0, n_hits, (n_edges,))
    r_out = F.one_hot(src, n_hits).float().t()  # (n_hits, n_edges)
    r_in = F.one_hot(dst, n_hits).float().t()  # (n_hits, n_edges)
    return hit_coords, r_in, r_out


MENAGERIE_ENTRIES = [
    ("Neuroevolution Potential (NEP)", "build_gpumd_nep", "example_input_gpumd_nep", "2022", "BIO"),
    ("GrAFF-MS", "build_graff_ms", "example_input_graff_ms", "2023", "BIO"),
    ("Graph2SMILES", "build_graph2smiles", "example_input_graph2smiles", "2022", "BIO"),
    ("GT-NMR", "build_gt_nmr", "example_input_gt_nmr", "2024", "BIO"),
    (
        "Hamiltonian Neural Network",
        "build_hamiltonian_nn",
        "example_input_hamiltonian_nn",
        "2019",
        "SEQ",
    ),
    ("HEPTrkX", "build_heptrkx", "example_input_heptrkx", "2020", "GRAPH"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")
