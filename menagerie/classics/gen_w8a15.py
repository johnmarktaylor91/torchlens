"""Molecular/materials generative-modeling architecture family: gen_w8a15.

Sources checked (repo_url / desc_source from the build queue, plus web search and
``gh api`` reads of the actual upstream source when the queue's repo_url pointed to
an org landing page rather than the model code):
  - ChemformerMapper is SKIPPED: the AstraZeneca MolecularAI Chemformer
    (arXiv:2106.06082) is already present as ``build_chemformer`` /
    ``example_input_chemformer`` in ``menagerie/classics/gen_w7a13.py`` -- a
    duplicate of this candidate (same repo, same paper).
  - ChemTS: https://github.com/tsudalab/ChemTS; Yang et al., Sci. Technol. Adv.
    Mater. 2017 (arXiv:1710.00616). Molecule design via Monte Carlo Tree Search
    with a neural rollout policy: the trainable component (and the only part
    that is a persistent ``nn.Module``, since the MCTS itself is a stateless
    search loop over that policy) is a character-level SMILES language model,
    confirmed from ``train_RNN/train_RNN.py`` in the official repo: an
    ``Embedding`` layer feeding two stacked ``GRU(256)`` layers (with dropout
    between them) and a ``TimeDistributed(Dense(vocab, softmax))`` output head
    that scores the next SMILES token at every position. Reimplemented here as
    a compact 2-layer GRU character-level SMILES generator/rollout-policy.
  - CMPNN (Communicative Message Passing Neural Network):
    https://github.com/SY575/CMPNN; Song et al., IJCAI 2020. Confirmed from
    ``chemprop/models/mpn.py`` in the official repo: a directed D-MPNN over
    atoms and bonds where, unlike vanilla MPNN, atom and bond messages are
    *communicated* every step -- an atom's incoming bond messages are combined
    with a sum-AND-max "message booster" (``agg.sum(dim=1) * agg.max(dim=1)``,
    not just a sum) before updating atom hidden states, and outgoing bond
    messages are recomputed as directed atom-to-bond differences each round.
    After message passing, node features are passed through a **BatchGRU**
    readout booster (a bidirectional GRU over each molecule's atom-hidden
    sequence, atom-count-padded, seeded with the max-pooled atom hidden as the
    initial state) before mean-pooling to a molecule vector. Reimplemented
    here with dense adjacency/booster message passing (functionally identical
    to the officially released directed sparse formulation, at small molecule
    sizes) plus the BatchGRU readout booster.
  - ConfGF: https://github.com/DeepGraphLearning/ConfGF; Shi et al., ICML 2021
    (arXiv:2105.03902). Confirmed from ``confgf/models/scorenet.py``: a Graph
    Isomorphism Network (GIN) message-passing backbone runs over a
    *multi-order-extended* molecular graph (edges added for up to ``order``-hop
    neighbors, each tagged with a distinct high-order edge type beyond the
    original bond types) and scores every (extended) edge's interatomic
    distance with a noise-conditional score network: a random Gaussian noise
    level ``sigma`` perturbs the true distances, the GIN + edge MLP predicts
    ``f_theta(d, sigma)``, and the final score is explicitly rescaled as
    ``f_theta_sigma(d) = f_theta(d, sigma) / sigma`` (the NCSN parameterization
    from Song & Ermon 2019) -- this is the "gradient field of the log-density
    of interatomic distances" the candidate notes describe, later Langevin-
    walked at sample time (sampling loop, not a persistent module). Reimplemented
    here as a compact GIN scorenet with a small explicit sigma schedule and the
    ``/sigma`` rescaling, operating on a fixed 2-hop-extended fully-connected
    small-molecule graph.
  - ConfVAE: https://github.com/MinkaiXu/ConfVAE-ICML21; Xu et al., ICML 2021
    (arXiv:2105.07246). Confirmed from ``models/vae.py``: a GNN encoder
    (``GNNPrior``/``GNNEncoder``, message-passing convs with a softplus
    edge-additive update) maps the 2D molecular graph to a per-node Gaussian
    latent (mu, log-sigma), reparameterized to sample a latent code; the
    "bilevel programming" of the paper composes (i) this VAE-style latent
    inference/generation of pairwise-distance targets with (ii) an inner-loop
    geometry-embedding solver (``diff_embed_3D``: several steps of gradient
    descent on 3D coordinates so their pairwise distances match the predicted
    targets, i.e. classic distance-geometry stress-majorization done via
    autograd instead of eigendecomposition). Reimplemented here as a GNN
    encoder producing a reparameterized latent, decoded by an MLP into target
    inter-atomic distances, followed by a small explicit unrolled
    gradient-descent distance-embedding loop that realizes 3D coordinates from
    those target distances -- both stages captured as ordinary autograd-visible
    tensor ops so the whole pipeline traces as one forward pass.
  - CrystalFlow: the build queue's ``repo_url``
    (``WanyuGroup/AI-for-Crystal-Materials``) is a *survey/index* page (a
    curated links table of "AI for crystal materials" papers), not the actual
    CrystalFlow implementation. The real official repo, confirmed via web
    search of arXiv:2412.11693 ("CrystalFlow: A Flow-Based Generative Model for
    Crystalline Materials", Luo et al. 2024/Nature Comms 2025), is
    https://github.com/ixsluo/CrystalFlow. Confirmed from
    ``diffcsp/pl_modules/cspnet.py`` (the DiffCSP-family periodic-crystal GNN
    CrystalFlow builds on): a fully-connected crystal graph is embedded per
    edge with (a) a sinusoidal fractional-coordinate-difference embedding
    (``SinusoidsEmbedding``), and (b) lattice-matrix Gram inner-product
    features (``lattices_rep @ lattices_rep.T``, giving an SE(3)-invariant,
    periodicity-aware edge feature), which together with atom-type embeddings
    and a sinusoidal diffusion/flow-time embedding drive stacked message-
    passing ``CSPLayer`` blocks (edge MLP -> mean-aggregate -> node MLP with a
    residual). The trained network is the conditional-flow-matching velocity
    field over (lattice matrix, fractional coordinates, atom types).
    Reimplemented here as a compact time-conditioned periodic-crystal message-
    passing network over a small fixed unit cell, predicting the flow velocity
    for lattice, fractional coordinates, and atom-type logits.

All models below are compact, randomly initialized, faithful reimplementations of
each architecture's distinctive mechanism (not generic MLP/transformer stubs), sized
small so tracing and rendering stay fast.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# ChemTS -- 2-layer stacked-GRU character-level SMILES language model used as
# the neural rollout policy inside Monte Carlo Tree Search.
# ---------------------------------------------------------------------------


class ChemTSRolloutRNN(nn.Module):
    """Character-level SMILES GRU language model (ChemTS's MCTS rollout policy).

    Mirrors the official ``train_RNN.py`` stack: token embedding, two stacked
    GRU layers with inter-layer dropout, and a per-timestep linear+softmax
    head over the SMILES vocabulary (next-token distribution used both for
    supervised training and for MCTS rollout sampling).
    """

    def __init__(self, vocab_size: int = 40, embed_dim: int = 32, hidden_dim: int = 64) -> None:
        """Build the rollout-policy RNN.

        Parameters
        ----------
        vocab_size : int
            Number of distinct SMILES character tokens (including padding/stop).
        embed_dim : int
            Token embedding width.
        hidden_dim : int
            GRU hidden width (shared by both stacked layers).
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru1 = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Score the next-token distribution at every position.

        Parameters
        ----------
        tokens : Tensor
            Integer SMILES token ids, shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Per-position next-token logits, shape ``(batch, seq_len, vocab_size)``.
        """
        x = self.embedding(tokens)
        x, _ = self.gru1(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = self.dropout2(x)
        return self.output_head(x)


def build_chemts() -> nn.Module:
    """Build a compact ChemTS rollout-policy RNN (2-layer GRU SMILES LM).

    Returns
    -------
    nn.Module
        ``ChemTSRolloutRNN`` in eval mode.
    """
    return ChemTSRolloutRNN(vocab_size=40, embed_dim=32, hidden_dim=64).eval()


def example_input_chemts() -> Tensor:
    """Build a batch of random SMILES token sequences for ``ChemTSRolloutRNN``.

    Returns
    -------
    Tensor
        Integer token ids, shape ``(2, 24)``.
    """
    return torch.randint(0, 40, (2, 24))


# ---------------------------------------------------------------------------
# CMPNN -- Communicative Message Passing Neural Network: atom<->bond dual
# message streams with a sum*max communicative booster, plus a BatchGRU
# readout booster before mean-pooling to a molecule vector.
# ---------------------------------------------------------------------------


class BatchGRUReadout(nn.Module):
    """Bidirectional-GRU readout booster over each molecule's atom features.

    Mirrors CMPNN's ``BatchGRU``: atom hidden states for one molecule are fed
    through a bidirectional GRU (seeded with the max-pooled atom hidden as the
    initial state for both directions) before the final linear projection.
    """

    def __init__(self, hidden_dim: int) -> None:
        """Build the readout booster.

        Parameters
        ----------
        hidden_dim : int
            Atom hidden-state width (the GRU is bidirectional, doubling this
            width at its output).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, atom_hidden: Tensor, atom_mask: Tensor) -> Tensor:
        """Boost per-atom hidden states with a bidirectional-GRU pass.

        Parameters
        ----------
        atom_hidden : Tensor
            Atom hidden states, shape ``(batch, n_atoms, hidden_dim)``.
        atom_mask : Tensor
            Boolean atom-presence mask, shape ``(batch, n_atoms)``.

        Returns
        -------
        Tensor
            Boosted atom hidden states, shape ``(batch, n_atoms, 2 * hidden_dim)``.
        """
        message = F.relu(atom_hidden + self.bias)
        masked = atom_hidden.masked_fill(~atom_mask.unsqueeze(-1), float("-inf"))
        init_hidden = masked.max(dim=1).values  # (batch, hidden_dim)
        init_hidden = init_hidden.unsqueeze(0).repeat(2, 1, 1)  # (2, batch, hidden_dim)
        boosted, _ = self.gru(message, init_hidden)
        return boosted


class CMPNNEncoder(nn.Module):
    """Communicative MPNN molecular encoder (dense small-graph formulation.

    Atom and bond messages are updated jointly at every step: bond-to-atom
    aggregation uses CMPNN's communicative "sum times max" booster instead of
    a plain sum, and atom-to-bond messages are recomputed as directed
    atom-hidden differences, before a BatchGRU readout booster and mean pool.
    """

    def __init__(
        self, atom_fdim: int = 12, bond_fdim: int = 6, hidden_dim: int = 32, depth: int = 3
    ) -> None:
        """Build the CMPNN encoder.

        Parameters
        ----------
        atom_fdim : int
            Raw atom feature width.
        bond_fdim : int
            Raw bond feature width.
        hidden_dim : int
            Shared atom/bond message hidden width.
        depth : int
            Number of communicative message-passing rounds.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.w_i_atom = nn.Linear(atom_fdim, hidden_dim)
        self.w_i_bond = nn.Linear(bond_fdim, hidden_dim)
        self.w_h_bond = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.lr = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gru_booster = BatchGRUReadout(hidden_dim)
        self.w_o = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(
        self,
        atom_feat: Tensor,
        bond_feat: Tensor,
        adjacency: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        """Encode a batch of small dense molecular graphs into molecule vectors.

        Parameters
        ----------
        atom_feat : Tensor
            Raw atom features, shape ``(batch, n_atoms, atom_fdim)``.
        bond_feat : Tensor
            Raw directed bond (i -> j) features, shape
            ``(batch, n_atoms, n_atoms, bond_fdim)``.
        adjacency : Tensor
            Directed bond-presence mask, shape ``(batch, n_atoms, n_atoms)``.
        atom_mask : Tensor
            Boolean atom-presence mask, shape ``(batch, n_atoms)``.

        Returns
        -------
        Tensor
            Molecule-level embeddings, shape ``(batch, hidden_dim)``.
        """
        adj = adjacency.unsqueeze(-1)  # (B, N, N, 1)
        input_atom = F.relu(self.w_i_atom(atom_feat))  # (B, N, H)
        message_atom = input_atom.clone()
        input_bond = F.relu(self.w_i_bond(bond_feat)) * adj  # (B, N, N, H)
        message_bond = input_bond.clone()

        for depth_idx in range(self.depth - 1):
            # communicative booster: sum-AND-max aggregate of incoming bond messages
            agg_sum = message_bond.sum(dim=2)  # (B, N, H)
            agg_max = message_bond.masked_fill(adj == 0, float("-inf")).amax(dim=2)
            agg_max = torch.nan_to_num(agg_max, neginf=0.0)
            agg_message = agg_sum * agg_max
            message_atom = message_atom + agg_message

            # directed bond update: atom-hidden difference along each directed edge
            rev_message = message_atom.unsqueeze(1) - message_atom.unsqueeze(
                2
            )  # h_j - h_i broadcast
            message_bond = self.w_h_bond[depth_idx](rev_message) * adj
            message_bond = F.relu(input_bond + message_bond) * adj

        agg_sum = message_bond.sum(dim=2)
        agg_max = message_bond.masked_fill(adj == 0, float("-inf")).amax(dim=2)
        agg_max = torch.nan_to_num(agg_max, neginf=0.0)
        agg_message = agg_sum * agg_max
        combined = self.lr(torch.cat([agg_message, message_atom, input_atom], dim=-1))

        boosted = self.gru_booster(combined, atom_mask)
        atom_hidden = F.relu(self.w_o(torch.cat([boosted, combined], dim=-1)))

        mask = atom_mask.unsqueeze(-1).float()
        mol_vec = (atom_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return mol_vec


def build_cmpnn() -> nn.Module:
    """Build a compact CMPNN molecular encoder.

    Returns
    -------
    nn.Module
        ``CMPNNEncoder`` in eval mode.
    """
    return CMPNNEncoder(atom_fdim=12, bond_fdim=6, hidden_dim=32, depth=3).eval()


def example_input_cmpnn() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build a small padded batch of dense molecular graphs for ``CMPNNEncoder``.

    Returns
    -------
    tuple of Tensor
        ``(atom_feat, bond_feat, adjacency, atom_mask)``.
    """
    batch, n_atoms = 2, 8
    atom_feat = torch.randn(batch, n_atoms, 12)
    adjacency = (torch.rand(batch, n_atoms, n_atoms) > 0.6).float()
    adjacency = adjacency * (1 - torch.eye(n_atoms)).unsqueeze(0)
    adjacency = torch.clamp(adjacency + adjacency.transpose(1, 2), max=1.0)
    bond_feat = torch.randn(batch, n_atoms, n_atoms, 6)
    atom_mask = torch.ones(batch, n_atoms, dtype=torch.bool)
    return atom_feat, bond_feat, adjacency, atom_mask


# ---------------------------------------------------------------------------
# ConfGF -- GIN over a multi-order-extended molecular graph, scoring
# interatomic distances with a noise-conditional score network
# (f_theta_sigma(d) = f_theta(d, sigma) / sigma).
# ---------------------------------------------------------------------------


class ConfGFGINLayer(nn.Module):
    """One GIN message-passing layer with edge-conditioned messages."""

    def __init__(self, hidden_dim: int) -> None:
        """Build a GIN layer.

        Parameters
        ----------
        hidden_dim : int
            Node/edge feature width.
        """
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_feat: Tensor, edge_feat: Tensor, adjacency: Tensor) -> Tensor:
        """Run one GIN message-passing round.

        Parameters
        ----------
        node_feat : Tensor
            Node features, shape ``(batch, n_nodes, hidden_dim)``.
        edge_feat : Tensor
            Directed edge features, shape ``(batch, n_nodes, n_nodes, hidden_dim)``.
        adjacency : Tensor
            Edge-presence mask, shape ``(batch, n_nodes, n_nodes)``.

        Returns
        -------
        Tensor
            Updated node features, shape ``(batch, n_nodes, hidden_dim)``.
        """
        adj = adjacency.unsqueeze(-1)
        hi = node_feat.unsqueeze(2).expand(-1, -1, node_feat.shape[1], -1)
        hj = node_feat.unsqueeze(1).expand(-1, node_feat.shape[1], -1, -1)
        msg_in = torch.cat([hj, edge_feat], dim=-1)
        msg = self.msg_mlp(msg_in) * adj
        agg = msg.sum(dim=2)
        return self.update_mlp(torch.cat([node_feat, agg], dim=-1))


class ConfGFScoreNet(nn.Module):
    """Noise-conditional score network over interatomic distances (ConfGF).

    A GIN backbone runs over a multi-order-extended graph (original bonds plus
    synthetic higher-order edges up to 2 hops, each with a distinct edge-type
    embedding); a sampled noise level perturbs the true distances, and the
    predicted score is rescaled by ``1 / sigma`` per the NCSN parameterization.
    """

    def __init__(
        self,
        num_atom_types: int = 16,
        num_edge_types: int = 5,
        hidden_dim: int = 32,
        num_layers: int = 3,
        sigma_begin: float = 10.0,
        sigma_end: float = 0.01,
        num_noise_levels: int = 8,
    ) -> None:
        """Build the score network.

        Parameters
        ----------
        num_atom_types : int
            Atom-type vocabulary size.
        num_edge_types : int
            Edge-type vocabulary size (original bond types + high-order types).
        hidden_dim : int
            Node/edge hidden width.
        num_layers : int
            Number of GIN message-passing layers.
        sigma_begin : float
            Largest noise level in the geometric schedule.
        sigma_end : float
            Smallest noise level in the geometric schedule.
        num_noise_levels : int
            Number of noise levels in the geometric schedule.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_emb = nn.Embedding(num_atom_types, hidden_dim)
        self.edge_emb = nn.Embedding(num_edge_types, hidden_dim)
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gin_layers = nn.ModuleList([ConfGFGINLayer(hidden_dim) for _ in range(num_layers)])
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        sigmas = torch.exp(
            torch.linspace(math.log(sigma_begin), math.log(sigma_end), num_noise_levels)
        )
        self.register_buffer("sigmas", sigmas)

    def forward(
        self,
        atom_type: Tensor,
        edge_type: Tensor,
        adjacency: Tensor,
        distances: Tensor,
        noise_level: Tensor,
    ) -> Tensor:
        """Score every extended-graph edge's (noised) interatomic distance.

        Parameters
        ----------
        atom_type : Tensor
            Integer atom types, shape ``(batch, n_atoms)``.
        edge_type : Tensor
            Integer edge types (bond or high-order), shape
            ``(batch, n_atoms, n_atoms)``.
        adjacency : Tensor
            Extended-graph edge-presence mask, shape ``(batch, n_atoms, n_atoms)``.
        distances : Tensor
            Perturbed pairwise distances, shape ``(batch, n_atoms, n_atoms)``.
        noise_level : Tensor
            Integer noise-level index per graph, shape ``(batch,)``.

        Returns
        -------
        Tensor
            Per-edge score ``d(log p)/d(distance)``, shape
            ``(batch, n_atoms, n_atoms, 1)``.
        """
        node_feat = self.node_emb(atom_type)
        edge_type_feat = self.edge_emb(edge_type)
        dist_feat = self.dist_mlp(distances.unsqueeze(-1))
        edge_feat = dist_feat * edge_type_feat

        h = node_feat
        for layer in self.gin_layers:
            h = layer(h, edge_feat, adjacency)

        hi = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
        hj = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        pair_feat = torch.cat([hi * hj, edge_feat], dim=-1)
        scores = self.output_mlp(pair_feat)

        sigma = self.sigmas[noise_level].view(-1, 1, 1, 1)
        return scores / sigma


def build_confgf() -> nn.Module:
    """Build a compact ConfGF noise-conditional distance score network.

    Returns
    -------
    nn.Module
        ``ConfGFScoreNet`` in eval mode.
    """
    return ConfGFScoreNet(
        num_atom_types=16, num_edge_types=5, hidden_dim=32, num_layers=3, num_noise_levels=8
    ).eval()


def example_input_confgf() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build a small extended-graph batch for ``ConfGFScoreNet``.

    Returns
    -------
    tuple of Tensor
        ``(atom_type, edge_type, adjacency, distances, noise_level)``.
    """
    batch, n_atoms = 2, 7
    atom_type = torch.randint(0, 16, (batch, n_atoms))
    adjacency = 1.0 - torch.eye(n_atoms).unsqueeze(0).expand(batch, -1, -1)
    edge_type = torch.randint(0, 5, (batch, n_atoms, n_atoms))
    edge_type = edge_type * adjacency.long()
    distances = torch.rand(batch, n_atoms, n_atoms) * 3.0 + 0.5
    distances = distances * adjacency
    noise_level = torch.randint(0, 8, (batch,))
    return atom_type, edge_type, adjacency, distances, noise_level


# ---------------------------------------------------------------------------
# ConfVAE -- GNN encoder produces a reparameterized latent, decoded into
# target inter-atomic distances, then realized into 3D coordinates by a
# short unrolled distance-geometry gradient-descent embedding loop.
# ---------------------------------------------------------------------------


class ConfVAEGraphConv(nn.Module):
    """Softplus edge-additive message-passing conv (ConfVAE's ``GConv``)."""

    def __init__(self, hidden_dim: int) -> None:
        """Build the conv.

        Parameters
        ----------
        hidden_dim : int
            Node/edge feature width.
        """
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, node_feat: Tensor, edge_feat: Tensor, adjacency: Tensor) -> Tensor:
        """Aggregate softplus(neighbor + edge) messages with a self-loop residual.

        Parameters
        ----------
        node_feat : Tensor
            Node features, shape ``(batch, n_nodes, hidden_dim)``.
        edge_feat : Tensor
            Directed edge features, shape ``(batch, n_nodes, n_nodes, hidden_dim)``.
        adjacency : Tensor
            Edge-presence mask, shape ``(batch, n_nodes, n_nodes)``.

        Returns
        -------
        Tensor
            Updated node features, shape ``(batch, n_nodes, hidden_dim)``.
        """
        adj = adjacency.unsqueeze(-1)
        hj = node_feat.unsqueeze(1).expand(-1, node_feat.shape[1], -1, -1)
        messages = F.softplus(hj + edge_feat) * adj
        agg = messages.sum(dim=2)
        return agg + (1.0 + self.eps) * node_feat


class ConfVAEEncoder(nn.Module):
    """GNN encoder mapping a molecular graph to a reparameterized latent code."""

    def __init__(self, hidden_dim: int = 32, latent_dim: int = 16) -> None:
        """Build the encoder.

        Parameters
        ----------
        hidden_dim : int
            GNN hidden width.
        latent_dim : int
            Per-node latent width.
        """
        super().__init__()
        self.node_emb = nn.Embedding(16, hidden_dim)
        self.edge_emb = nn.Embedding(5, hidden_dim)
        self.conv1 = ConfVAEGraphConv(hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = ConfVAEGraphConv(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.out_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_fc2 = nn.Linear(hidden_dim, latent_dim * 2)
        self.latent_dim = latent_dim

    def forward(
        self, atom_type: Tensor, edge_type: Tensor, adjacency: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode a batch of molecular graphs to per-node Gaussian latent params.

        Parameters
        ----------
        atom_type : Tensor
            Integer atom types, shape ``(batch, n_atoms)``.
        edge_type : Tensor
            Integer edge (bond) types, shape ``(batch, n_atoms, n_atoms)``.
        adjacency : Tensor
            Bond-presence mask, shape ``(batch, n_atoms, n_atoms)``.

        Returns
        -------
        tuple of Tensor
            ``(mu, log_sigma)``, each shape ``(batch, n_atoms, latent_dim)``.
        """
        b, n = atom_type.shape
        h = self.node_emb(atom_type)
        edge_feat = self.edge_emb(edge_type)

        h = self.conv1(h, edge_feat, adjacency)
        h = F.softplus(self.bn1(h.reshape(b * n, -1)).reshape(b, n, -1))
        h = self.conv2(h, edge_feat, adjacency)
        h = self.bn2(h.reshape(b * n, -1)).reshape(b, n, -1)

        h_global = h.mean(dim=1, keepdim=True).expand(-1, n, -1)
        node_feat = torch.cat([h, h_global], dim=-1)
        node_feat = F.softplus(self.out_fc1(node_feat))
        out = self.out_fc2(node_feat)
        mu, log_sigma = out[..., : self.latent_dim], out[..., self.latent_dim :]
        return mu, log_sigma


class ConfVAE(nn.Module):
    """Bilevel VAE + distance-geometry conformer generator (ConfVAE).

    Stage 1 (VAE): a GNN encoder produces a per-node reparameterized latent;
    an MLP decodes pairwise latent features into target inter-atomic
    distances. Stage 2 (distance geometry): a short, explicit, autograd-
    visible gradient-descent loop moves randomly initialized 3D coordinates
    so their pairwise distances match those targets -- mirroring the official
    ``diff_embed_3D`` inner-loop solver, unrolled for a fixed small number of
    steps so the whole pipeline traces as a single forward pass.
    """

    def __init__(self, hidden_dim: int = 32, latent_dim: int = 16, embed_steps: int = 4) -> None:
        """Build the ConfVAE conformer generator.

        Parameters
        ----------
        hidden_dim : int
            GNN hidden width.
        latent_dim : int
            Per-node latent width.
        embed_steps : int
            Number of unrolled distance-geometry gradient-descent steps.
        """
        super().__init__()
        self.encoder = ConfVAEEncoder(hidden_dim, latent_dim)
        self.dist_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self.embed_steps = embed_steps
        self.step_size = 0.1

    def forward(
        self, atom_type: Tensor, edge_type: Tensor, adjacency: Tensor, init_pos: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Sample a latent conformer and embed it into 3D coordinates.

        Parameters
        ----------
        atom_type : Tensor
            Integer atom types, shape ``(batch, n_atoms)``.
        edge_type : Tensor
            Integer edge (bond) types, shape ``(batch, n_atoms, n_atoms)``.
        adjacency : Tensor
            Bond-presence mask, shape ``(batch, n_atoms, n_atoms)``.
        init_pos : Tensor
            Initial 3D coordinate guess, shape ``(batch, n_atoms, 3)``.

        Returns
        -------
        tuple of Tensor
            ``(pos, target_distances)``: embedded coordinates
            ``(batch, n_atoms, 3)`` and the decoded target pairwise distances
            ``(batch, n_atoms, n_atoms)``.
        """
        mu, log_sigma = self.encoder(atom_type, edge_type, adjacency)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * log_sigma)

        zi = z.unsqueeze(2).expand(-1, -1, z.shape[1], -1)
        zj = z.unsqueeze(1).expand(-1, z.shape[1], -1, -1)
        target_dist = self.dist_decoder(torch.cat([zi, zj], dim=-1)).squeeze(-1)
        target_dist = target_dist * adjacency

        pos = init_pos
        for _ in range(self.embed_steps):
            diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 3)
            cur_dist = diff.norm(dim=-1).clamp_min(1e-6)
            resid = (cur_dist - target_dist) * adjacency
            grad = (resid / cur_dist).unsqueeze(-1) * diff
            pos = pos - self.step_size * grad.sum(dim=2)

        return pos, target_dist


def build_confvae() -> nn.Module:
    """Build a compact ConfVAE bilevel conformer generator.

    Returns
    -------
    nn.Module
        ``ConfVAE`` in eval mode.
    """
    return ConfVAE(hidden_dim=32, latent_dim=16, embed_steps=4).eval()


def example_input_confvae() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build a small molecular graph + coordinate guess for ``ConfVAE``.

    Returns
    -------
    tuple of Tensor
        ``(atom_type, edge_type, adjacency, init_pos)``.
    """
    batch, n_atoms = 2, 7
    atom_type = torch.randint(0, 16, (batch, n_atoms))
    adjacency = (torch.rand(batch, n_atoms, n_atoms) > 0.5).float()
    adjacency = adjacency * (1 - torch.eye(n_atoms)).unsqueeze(0)
    adjacency = torch.clamp(adjacency + adjacency.transpose(1, 2), max=1.0)
    edge_type = torch.randint(0, 5, (batch, n_atoms, n_atoms)) * adjacency.long()
    init_pos = torch.randn(batch, n_atoms, 3)
    return atom_type, edge_type, adjacency, init_pos


# ---------------------------------------------------------------------------
# CrystalFlow -- time-conditioned periodic-crystal message-passing network
# (CSPNet-style) predicting the conditional-flow-matching velocity field for
# lattice matrix, fractional coordinates, and atom types.
# ---------------------------------------------------------------------------


class CrystalFlowSinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding of fractional-coordinate differences (periodic-aware)."""

    def __init__(self, n_frequencies: int = 8) -> None:
        """Build the embedding.

        Parameters
        ----------
        n_frequencies : int
            Number of sinusoid frequencies per spatial dimension.
        """
        super().__init__()
        self.n_frequencies = n_frequencies
        freqs = 2 * math.pi * torch.arange(n_frequencies)
        self.register_buffer("frequencies", freqs)
        self.dim = n_frequencies * 2 * 3

    def forward(self, frac_diff: Tensor) -> Tensor:
        """Embed fractional-coordinate differences with sin/cos features.

        Parameters
        ----------
        frac_diff : Tensor
            Fractional coordinate differences, shape ``(..., 3)``.

        Returns
        -------
        Tensor
            Sinusoidal embedding, shape ``(..., n_frequencies * 6)``.
        """
        emb = frac_diff.unsqueeze(-1) * self.frequencies.view(*([1] * frac_diff.dim()), -1)
        emb = emb.reshape(*frac_diff.shape[:-1], -1)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def sinusoidal_time_embedding(t: Tensor, dim: int) -> Tensor:
    """Standard transformer-style sinusoidal embedding of a scalar time/step.

    Parameters
    ----------
    t : Tensor
        Time/flow-step values, shape ``(batch,)``.
    dim : int
        Embedding width (must be even).

    Returns
    -------
    Tensor
        Time embedding, shape ``(batch, dim)``.
    """
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0) / (half - 1)))
    args = t[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class CSPLayer(nn.Module):
    """One periodic-crystal message-passing block (CrystalFlow's ``CSPLayer``)."""

    def __init__(self, hidden_dim: int, dist_dim: int) -> None:
        """Build a CSP message-passing layer.

        Parameters
        ----------
        hidden_dim : int
            Node feature width.
        dist_dim : int
            Width of the concatenated edge (distance + lattice) feature.
        """
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + dist_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, node_feat: Tensor, dist_emb: Tensor, lattice_ip_flat: Tensor) -> Tensor:
        """Run one fully-connected periodic message-passing round.

        Parameters
        ----------
        node_feat : Tensor
            Node features, shape ``(batch, n_atoms, hidden_dim)``.
        dist_emb : Tensor
            Per-edge sinusoidal fractional-distance embedding, shape
            ``(batch, n_atoms, n_atoms, dist_dim)``.
        lattice_ip_flat : Tensor
            Flattened lattice Gram matrix broadcast to every edge, shape
            ``(batch, n_atoms, n_atoms, 9)``.

        Returns
        -------
        Tensor
            Updated (residual) node features, shape ``(batch, n_atoms, hidden_dim)``.
        """
        n = node_feat.shape[1]
        hi = node_feat.unsqueeze(2).expand(-1, -1, n, -1)
        hj = node_feat.unsqueeze(1).expand(-1, n, -1, -1)
        edge_in = torch.cat([hi, hj, lattice_ip_flat, dist_emb], dim=-1)
        edge_feat = self.edge_mlp(edge_in)
        agg = edge_feat.mean(dim=2)
        node_out = self.node_mlp(torch.cat([node_feat, agg], dim=-1))
        return node_feat + node_out


class CrystalFlowVelocityNet(nn.Module):
    """Time-conditioned periodic-crystal flow-matching velocity field.

    A fully-connected crystal graph is embedded per edge with a sinusoidal
    fractional-coordinate-difference feature and the flattened lattice Gram
    matrix (``L L^T``, an SE(3)-invariant periodicity-aware feature), and per
    node with atom-type + flow-time embeddings; stacked ``CSPLayer`` message
    passing produces node features that are read out into a predicted
    velocity for the lattice matrix, the fractional coordinates, and the
    atom-type logits (the conditional-flow-matching targets).
    """

    def __init__(
        self,
        num_atom_types: int = 20,
        hidden_dim: int = 32,
        num_layers: int = 3,
        n_frequencies: int = 8,
    ) -> None:
        """Build the velocity network.

        Parameters
        ----------
        num_atom_types : int
            Atom-type vocabulary size (periodic-table cap).
        hidden_dim : int
            Node hidden width.
        num_layers : int
            Number of CSP message-passing layers.
        n_frequencies : int
            Sinusoidal frequency count for the distance embedding.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_emb = nn.Embedding(num_atom_types, hidden_dim)
        self.time_dim = hidden_dim
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dist_emb = CrystalFlowSinusoidalEmbedding(n_frequencies)
        self.layers = nn.ModuleList(
            [CSPLayer(hidden_dim, self.dist_emb.dim) for _ in range(num_layers)]
        )
        self.coord_head = nn.Linear(hidden_dim, 3)
        self.type_head = nn.Linear(hidden_dim, num_atom_types)
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 9)
        )

    def forward(
        self, atom_type: Tensor, frac_coords: Tensor, lattice: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Predict the flow velocity for lattice, fractional coords, and types.

        Parameters
        ----------
        atom_type : Tensor
            Integer atom types, shape ``(batch, n_atoms)``.
        frac_coords : Tensor
            Fractional coordinates in ``[0, 1)``, shape ``(batch, n_atoms, 3)``.
        lattice : Tensor
            Lattice matrix, shape ``(batch, 3, 3)``.
        t : Tensor
            Flow-matching interpolation time in ``[0, 1]``, shape ``(batch,)``.

        Returns
        -------
        tuple of Tensor
            ``(lattice_velocity, frac_coord_velocity, type_logits)``.
        """
        b, n = atom_type.shape
        node_feat = self.atom_emb(atom_type)
        time_feat = self.time_proj(sinusoidal_time_embedding(t, self.time_dim))
        node_feat = node_feat + time_feat.unsqueeze(1)

        frac_diff = (frac_coords.unsqueeze(2) - frac_coords.unsqueeze(1)) % 1.0  # (B, N, N, 3)
        dist_emb = self.dist_emb(frac_diff)

        lattice_gram = lattice @ lattice.transpose(-1, -2)  # (B, 3, 3), SE(3)-invariant
        lattice_ip_flat = lattice_gram.reshape(b, 1, 1, 9).expand(-1, n, n, -1)

        h = node_feat
        for layer in self.layers:
            h = layer(h, dist_emb, lattice_ip_flat)

        frac_velocity = self.coord_head(h)
        type_logits = self.type_head(h)
        h_global = h.mean(dim=1)
        lattice_velocity = self.lattice_head(h_global).reshape(b, 3, 3)
        return lattice_velocity, frac_velocity, type_logits


def build_crystalflow() -> nn.Module:
    """Build a compact CrystalFlow periodic-crystal flow-matching velocity net.

    Returns
    -------
    nn.Module
        ``CrystalFlowVelocityNet`` in eval mode.
    """
    return CrystalFlowVelocityNet(
        num_atom_types=20, hidden_dim=32, num_layers=3, n_frequencies=8
    ).eval()


def example_input_crystalflow() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build a small unit cell + flow-time batch for ``CrystalFlowVelocityNet``.

    Returns
    -------
    tuple of Tensor
        ``(atom_type, frac_coords, lattice, t)``.
    """
    batch, n_atoms = 2, 6
    atom_type = torch.randint(0, 20, (batch, n_atoms))
    frac_coords = torch.rand(batch, n_atoms, 3)
    lattice = torch.eye(3).unsqueeze(0).expand(batch, -1, -1).clone()
    lattice = lattice + 0.1 * torch.randn(batch, 3, 3)
    t = torch.rand(batch)
    return atom_type, frac_coords, lattice, t


MENAGERIE_ENTRIES = [
    ("ChemTS", "build_chemts", "example_input_chemts", "2017", "BIO"),
    ("CMPNN (Communicative MPNN)", "build_cmpnn", "example_input_cmpnn", "2020", "BIO"),
    ("ConfGF", "build_confgf", "example_input_confgf", "2021", "BIO"),
    ("ConfVAE", "build_confvae", "example_input_confvae", "2021", "BIO"),
    ("CrystalFlow", "build_crystalflow", "example_input_crystalflow", "2024", "BIO"),
]
