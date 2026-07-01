"""Compact faithful reimplementations for build_queue rows 127-132 (W8A21).

Sources checked (repo browsed via ``gh api`` / web, no clone/pip-install):
  - GrammarVAE: Kusner, Paige, Hernandez-Lobato, "Grammar Variational
    Autoencoder", ICML 2017, arXiv:1703.01925. Official repo
    github.com/mkusner/grammarVAE (Theano/Keras), ``models/model_zinc.py``
    (``MoleculeVAE``). Distinctive mechanism: rather than encoding/decoding
    raw SMILES strings, GrammarVAE encodes/decodes *parse trees* of a
    context-free grammar (each molecule/expression is first parsed into a
    sequence of CFG production rules, one-hot encoded per position). The
    encoder is a stack of 1D convolutions over the one-hot rule sequence
    feeding a dense bottleneck to a Gaussian latent (mu, logvar,
    reparameterized sample). The decoder is a stack of GRUs that unroll the
    latent into a per-position distribution over grammar productions; at
    *decode/sampling* time (not training loss) a stack-based CFG parser
    tracks the current left-hand-side nonterminal and *masks* the softmax
    over productions to only those rules whose LHS matches the nonterminal
    on top of the derivation stack, guaranteeing every sampled sequence is a
    syntactically valid parse tree (hence always-valid decoded output).
    Reproduced here with a small production-rule vocabulary, a Conv1d
    encoder -> Gaussian bottleneck, a GRU decoder producing per-step rule
    logits, and an explicit stack-based grammar mask applied to those logits
    (a toy 6-rule arithmetic-expression grammar standing in for the ZINC
    SMILES grammar, since torchlens traces module structure/tensor flow,
    not grammar semantics).
  - Graph2Edits: Zhong, Song, Li, Kang, Han, Sun, "Root-aligned SMILES for
    de novo molecular retrosynthesis", extended to explicit graph edits in
    "Graph2Edits: Reaction-Center-Aware Molecular Graph Editing", Nature
    Communications 2023. Official repo
    github.com/Jamson-Zhong/Graph2Edits, ``models/graph2edits.py``
    (``Graph2Edits.compute_edit_scores``). Distinctive mechanism: rather
    than a single forward-pass sequence model, Graph2Edits predicts a
    retrosynthetic reaction as an *autoregressive sequence of graph edits*.
    A message-passing atom/bond encoder produces per-atom features for the
    current (partially-edited) product graph; those features are combined
    with the *previous edit step's* atom hidden state through a gated
    residual update (``W_vv`` initialized to identity, plus ``W_vc`` on the
    fresh encoder output, summed and ReLU'd) so the model remembers the
    graph's edit history across steps. Per-atom, per-bond, and whole-graph
    (stop) linear heads score every possible next edit (atom-label change,
    bond-order change, or "stop"); at inference the highest-scoring edit is
    applied to the graph and the loop repeats. Reproduced here with a
    compact 2-layer message-passing encoder, an identity-initialized
    recurrent atom-hidden update across a fixed number of edit steps, and
    atom/bond/graph edit-score heads mirroring the three-way scoring.
  - GraphAF: Shi, Xu, Zhu, Zhang, Zhang, Tang, "GraphAF: a Flow-based
    Autoregressive Model for Molecular Graph Generation", ICLR 2020,
    arXiv:2001.09382. The linked github.com/DeepGraphLearning/GraphAF repo
    is a pointer-only README; the maintained reference implementation lives
    in the same lab's TorchDrug library,
    github.com/DeepGraphLearning/torchdrug,
    ``torchdrug/models/flow.py`` (``GraphAutoregressiveFlow``) +
    ``torchdrug/layers/flow.py`` (``ConditionalFlow``). Distinctive
    mechanism: GraphAF builds a molecular graph one atom/bond at a time
    (autoregressive), but instead of directly outputting a discrete
    node/edge-type distribution at each step, it runs a *normalizing flow*:
    the current graph is embedded with a relational GNN into a per-node and
    graph-level condition vector, the one-hot node/edge type is dequantized
    (uniform noise added), and a stack of conditional affine-coupling
    layers (each an MLP mapping the condition vector to a
    ``(scale, bias)`` pair, ``scale`` squashed through ``tanh`` times a
    learnable rescale) transforms Gaussian noise into (or the dequantized
    one-hot into, for the log-likelihood direction) the type logits, giving
    an exact-likelihood generative model with an invertible per-step
    transform. Reproduced here with a compact relational-GNN condition
    encoder, dequantized one-hot atom/bond types, and a small stack of
    conditional affine-coupling flow layers run in the generation
    (``reverse``) direction to autoregressively grow a graph.
  - GraphBP: Liu, Luo, Wang, Ji (divelab), "Generating 3D Molecules for
    Target Protein Binding", ICML 2022, arXiv:2204.09410. Official repo
    github.com/divelab/GraphBP, ``GraphBP/model/graphbp.py``
    (``GraphBP``). Distinctive mechanism: GraphBP generates a 3D ligand
    atom-by-atom *conditioned on a fixed protein-pocket point cloud*. A
    SchNet-style continuous-filter graph network embeds the joint
    pocket+partial-ligand atoms; per new atom, an MLP "focus" classifier
    picks which existing atom to attach to, then the atom's *element type*
    and its position -- expressed as local spherical coordinates (bond
    distance, bond angle, dihedral/torsion angle relative to the two most
    recently placed reference atoms, a "local coordinate system") -- are
    each produced by a small dedicated conditional normalizing flow
    (``ST_Net_Exp`` affine-coupling MLPs) conditioned on the focus atom's
    embedding, so 3D structure is generated incrementally in an
    internal/relative coordinate frame rather than absolute xyz. Reproduced
    here with a compact continuous-filter (SchNet-like) pocket+ligand
    encoder, a focus-atom classifier, and four small conditional
    affine-coupling flows (type, distance, angle, torsion) mirroring the
    spherical-coordinate autoregressive placement.
  - GraphDF: Luo, Yan, Ji (divelab), "GraphDF: A Discrete Flow Model for
    Molecular Graph Generation", ICML 2021, arXiv:2102.01189. Official code
    ships inside the DIG (Dive into Graphs) library,
    github.com/divelab/DIG, ``dig/ggraph/method/GraphDF/model/`` --
    ``st_net.py`` (``ST_Net_Exp``) and ``df_utils.py``
    (``one_hot_argmax``). Distinctive mechanism: GraphAF's flow dequantizes
    discrete one-hot node/edge types with continuous noise before running a
    *continuous* normalizing flow, which the GraphDF paper argues creates a
    train/generation mismatch (continuous latents must later be discretized
    for sampling). GraphDF instead keeps the flow entirely in the discrete
    one-hot / integer domain: coupling-layer affine transforms are applied
    directly to one-hot vectors and a hard ``argmax`` projects each
    transformed vector back to a valid one-hot category, with a
    straight-through estimator (``one_hot_argmax``: forward uses the hard
    one-hot of the argmax, backward gradient flows through a
    softmax-with-temperature) so the whole discrete pipeline stays
    end-to-end trainable without any dequantization noise. Reproduced here
    with the same relational-GNN condition encoder family as GraphAF but a
    *discrete* coupling step: affine-transformed one-hot logits projected
    through a straight-through ``one_hot_argmax`` at every flow layer,
    autoregressively emitting discrete atom/bond types with no
    dequantization.
  - GraphGDP: Huang, Sun, Yu, Ye, Sun, Zhu, "GraphGDP: Generative Diffusion
    Processes for Permutation Invariant Graph Generation", ICDM 2022,
    arXiv:2212.01842. Official repo github.com/GRAPH-0/GraphGDP,
    ``models/pgsn.py`` (``PGSN``, "Position-enhanced Graph Score
    Network"). Distinctive mechanism: GraphGDP is a score-based diffusion
    model whose state is the *dense adjacency matrix* of a graph (rather
    than a fixed-size image or point cloud) -- Gaussian noise is added
    directly to the (continuous-relaxed) adjacency matrix over a diffusion
    schedule, and a permutation-equivariant GNN "score network" is trained
    to predict the denoising direction at each timestep. To break the
    permutation-invariance/lack-of-node-identity issue plain adjacency
    diffusion has, the score network augments each node with structural
    *positional features* derived purely from the (noisy) graph itself --
    a degree one-hot embedding and a random-walk structural encoding (the
    diagonal of powers of the (noisy, thresholded) adjacency matrix, a
    stand-in for the paper's RWSE) -- alongside a sinusoidal diffusion
    timestep embedding that conditions every GNN message-passing layer.
    Reproduced here with a compact degree + random-walk positional
    encoder, a sinusoidal timestep embedding, and a small
    timestep-conditioned GNN score head that outputs a same-shape
    denoising update to a dense adjacency matrix.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 1. GrammarVAE -- CFG-parse-tree VAE with a grammar-mask-constrained decoder
# ---------------------------------------------------------------------------

# A tiny 6-production toy grammar standing in for the ZINC/SMILES CFG:
#   0: S -> S '+' T     (LHS nonterminal 0 = 'S', pushes T=1, S=0 back)
#   1: S -> T           (LHS 0, pushes T=1)
#   2: T -> T '*' F     (LHS 1, pushes F=2, T=1 back)
#   3: T -> F           (LHS 1, pushes F=2)
#   4: F -> '(' S ')'   (LHS 2, pushes S=0)
#   5: F -> 'x'         (LHS 2, terminal, pushes nothing)
_GRAMMAR_LHS = torch.tensor([0, 0, 1, 1, 2, 2])
_GRAMMAR_RHS_PUSH = [[1, 0], [1], [2, 1], [2], [0], []]
_N_RULES = 6


def _grammar_masks() -> Tensor:
    """Build the ``(nonterminal, rule)`` legality mask used at decode time.

    Returns
    -------
    Tensor
        Boolean tensor of shape ``(3, 6)``: mask[nt, r] is True iff rule r's
        left-hand-side nonterminal is ``nt``.
    """

    n_nonterminals = 3
    mask = torch.zeros(n_nonterminals, _N_RULES, dtype=torch.bool)
    for rule_idx, lhs in enumerate(_GRAMMAR_LHS.tolist()):
        mask[lhs, rule_idx] = True
    return mask


class GrammarVAE(nn.Module):
    """Grammar VAE: Conv1d encoder -> Gaussian latent -> masked GRU decoder.

    Encodes a one-hot sequence of CFG production-rule indices, and decodes
    by unrolling a GRU and masking the per-step rule logits with the
    grammar's LHS-nonterminal legality mask (via an explicit stack
    simulation), guaranteeing syntactically valid parse trees.
    """

    def __init__(self, seq_len: int = 12, latent_dim: int = 8, hidden: int = 32) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden = hidden

        self.conv1 = nn.Conv1d(_N_RULES, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.enc_fc = nn.Linear(hidden * seq_len, hidden)
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)

        self.dec_init = nn.Linear(latent_dim, hidden)
        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden, num_layers=2, batch_first=True)
        self.rule_out = nn.Linear(hidden, _N_RULES)

        self.register_buffer("grammar_mask", _grammar_masks())

    def encode(self, rule_onehot: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a one-hot rule sequence to Gaussian latent parameters."""

        h = F.relu(self.conv1(rule_onehot.transpose(1, 2)))
        h = F.relu(self.conv2(h))
        h = self.enc_fc(h.flatten(1))
        return self.mu(h), self.logvar(h)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent ``z`` into masked per-step rule logits.

        A stack-based CFG derivation is simulated in lockstep with the GRU
        unroll: at each step the current top-of-stack nonterminal selects
        which row of ``grammar_mask`` disallows illegal productions before
        the softmax, and the *sampled* (argmax, for a deterministic trace)
        rule's right-hand-side nonterminals are pushed back onto the stack.
        """

        batch = z.shape[0]
        h0 = torch.tanh(self.dec_init(z)).unsqueeze(0).repeat(self.gru.num_layers, 1, 1)
        dec_in = h0[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
        gru_out, _ = self.gru(dec_in, h0.contiguous())
        raw_logits = self.rule_out(gru_out)  # (batch, seq_len, n_rules)

        stacks = [[0] for _ in range(batch)]  # start symbol nonterminal 0
        masked_logits = torch.zeros_like(raw_logits)
        for t in range(self.seq_len):
            for b in range(batch):
                nt = stacks[b].pop() if stacks[b] else 0
                mask = self.grammar_mask[nt]
                step_logits = raw_logits[b, t].masked_fill(~mask, -1e9)
                masked_logits[b, t] = step_logits
                rule = int(torch.argmax(step_logits).item())
                for sym in reversed(_GRAMMAR_RHS_PUSH[rule]):
                    stacks[b].append(sym)
        return masked_logits

    def forward(self, rule_onehot: Tensor) -> Tensor:
        """Encode, reparameterize, and grammar-mask-decode.

        Parameters
        ----------
        rule_onehot : Tensor
            Shape ``(batch, seq_len, n_rules)`` one-hot production sequence.

        Returns
        -------
        Tensor
            Masked rule logits, shape ``(batch, seq_len, n_rules)``.
        """

        mu, logvar = self.encode(rule_onehot)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decode(z)


def build_grammar_vae() -> nn.Module:
    """Build a compact GrammarVAE.

    Returns
    -------
    nn.Module
        ``GrammarVAE`` in eval mode.
    """

    return GrammarVAE().eval()


def example_input_grammar_vae() -> Tensor:
    """Create example input for :func:`build_grammar_vae`.

    Returns
    -------
    Tensor
        One-hot rule sequence, shape ``(2, 12, 6)``.
    """

    torch.manual_seed(0)
    idx = torch.randint(0, _N_RULES, (2, 12))
    return F.one_hot(idx, _N_RULES).float()


# ---------------------------------------------------------------------------
# 2. Graph2Edits -- autoregressive graph-edit-sequence retrosynthesis model
# ---------------------------------------------------------------------------


class _MPNEncoder(nn.Module):
    """Small message-passing atom encoder (2 rounds of edge-conditioned sum)."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden: int, depth: int = 2) -> None:
        super().__init__()
        self.atom_in = nn.Linear(atom_dim, hidden)
        self.edge_mlp = nn.ModuleList([nn.Linear(hidden + bond_dim, hidden) for _ in range(depth)])
        self.depth = depth

    def forward(self, atom_feat: Tensor, bond_index: Tensor, bond_feat: Tensor) -> Tensor:
        h = F.relu(self.atom_in(atom_feat))
        src, dst = bond_index[:, 0], bond_index[:, 1]
        for layer in self.edge_mlp:
            msg_in = torch.cat([h[src], bond_feat], dim=-1)
            msg = F.relu(layer(msg_in))
            agg = torch.zeros_like(h).index_add(0, dst, msg)
            h = F.relu(h + agg)
        return h


class Graph2Edits(nn.Module):
    """Autoregressive graph-edit predictor with an identity-gated memory.

    Each of ``n_steps`` iterations re-encodes the (conceptually
    edit-updated, here fixed for tracing) product graph and fuses the fresh
    atom features with the previous step's hidden atom state via a
    ``W_vv`` (identity-initialized) + ``W_vc`` gated residual, then scores
    every atom / bond / the whole graph for the next edit.
    """

    def __init__(
        self, atom_dim: int = 12, bond_dim: int = 4, hidden: int = 32, n_steps: int = 3
    ) -> None:
        super().__init__()
        self.encoder = _MPNEncoder(atom_dim, bond_dim, hidden)
        self.w_vv = nn.Linear(hidden, hidden, bias=False)
        nn.init.eye_(self.w_vv.weight)
        self.w_vc = nn.Linear(hidden, hidden, bias=False)
        self.atom_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, atom_dim)
        )
        self.bond_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, bond_dim)
        )
        self.stop_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.n_steps = n_steps
        self.hidden = hidden

    def forward(
        self, atom_feat: Tensor, bond_index: Tensor, bond_feat: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run ``n_steps`` autoregressive edit-scoring iterations.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(atom_edit_scores, bond_edit_scores, stop_scores)`` from the
            final iteration, shapes ``(n_atoms, atom_dim)``,
            ``(n_bonds, bond_dim)``, ``(1,)``.
        """

        n_atoms = atom_feat.shape[0]
        prev_hidden = torch.zeros(n_atoms, self.hidden)
        for _ in range(self.n_steps):
            fresh = self.encoder(atom_feat, bond_index, bond_feat)
            prev_hidden = F.relu(self.w_vv(prev_hidden) + self.w_vc(fresh))

        atom_scores = self.atom_head(prev_hidden)
        src, dst = bond_index[:, 0], bond_index[:, 1]
        bond_scores = self.bond_head(torch.cat([prev_hidden[src], prev_hidden[dst]], dim=-1))
        graph_vec = prev_hidden.sum(dim=0, keepdim=True)
        stop_score = self.stop_head(graph_vec).squeeze(0)
        return atom_scores, bond_scores, stop_score


def build_graph2edits() -> nn.Module:
    """Build a compact Graph2Edits model.

    Returns
    -------
    nn.Module
        ``Graph2Edits`` in eval mode.
    """

    return Graph2Edits().eval()


def example_input_graph2edits() -> tuple[Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_graph2edits`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(atom_feat, bond_index, bond_feat)`` for an 8-atom, 10-bond graph.
    """

    torch.manual_seed(1)
    atom_feat = torch.randn(8, 12)
    bond_index = torch.randint(0, 8, (10, 2))
    bond_feat = torch.randn(10, 4)
    return atom_feat, bond_index, bond_feat


# ---------------------------------------------------------------------------
# 3. GraphAF -- autoregressive continuous-flow molecular graph generator
# ---------------------------------------------------------------------------


class _ConditionalAffineCoupling(nn.Module):
    """One conditional affine-coupling flow layer (GraphAF's ``ConditionalFlow``)."""

    def __init__(self, dim: int, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.ReLU(), nn.Linear(hidden, dim * 2))
        self.rescale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        scale, bias = self.mlp(condition).chunk(2, dim=-1)
        scale = torch.tanh(scale) * self.rescale
        out = (x + bias) * torch.exp(scale)
        return out, scale

    def reverse(self, latent: Tensor, condition: Tensor) -> Tensor:
        scale, bias = self.mlp(condition).chunk(2, dim=-1)
        scale = torch.tanh(scale) * self.rescale
        return latent * torch.exp(-scale) - bias


class _RelationalGNN(nn.Module):
    """Small relational (edge-typed) GNN producing node + graph condition vectors."""

    def __init__(self, node_types: int, hidden: int, n_relations: int = 3, layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(node_types, hidden)
        self.rel_lin = nn.ModuleList(
            [nn.Linear(hidden, hidden * n_relations) for _ in range(layers)]
        )
        self.n_relations = n_relations
        self.hidden = hidden

    def forward(self, node_type: Tensor, rel_adj: Tensor) -> tuple[Tensor, Tensor]:
        """``rel_adj``: ``(n_relations, n_nodes, n_nodes)`` binary adjacency per bond type."""

        h = self.embed(node_type)
        for layer in self.rel_lin:
            proj = layer(h).view(h.shape[0], self.n_relations, self.hidden)
            msg = torch.einsum("rij,jrh->ih", rel_adj, proj)
            h = torch.tanh(h + msg)
        graph_vec = h.sum(dim=0, keepdim=True)
        return h, graph_vec


class GraphAF(nn.Module):
    """Flow-based autoregressive molecular graph generator.

    A relational GNN embeds the (partial) graph into node/graph condition
    vectors; a stack of conditional affine-coupling flow layers maps
    Gaussian latents to dequantized one-hot atom-type logits, run in the
    generative (``reverse``) direction for one new atom.
    """

    def __init__(self, node_types: int = 9, hidden: int = 24, n_flow_layers: int = 4) -> None:
        super().__init__()
        self.node_types = node_types
        self.gnn = _RelationalGNN(node_types, hidden)
        self.flow_layers = nn.ModuleList(
            [_ConditionalAffineCoupling(node_types, hidden, hidden) for _ in range(n_flow_layers)]
        )

    def forward(self, node_type: Tensor, rel_adj: Tensor, latent: Tensor) -> Tensor:
        """Condition on the current graph and invert the flow to new atom-type logits.

        Parameters
        ----------
        node_type : Tensor
            Shape ``(n_nodes,)`` integer atom types of the existing graph.
        rel_adj : Tensor
            Shape ``(n_relations, n_nodes, n_nodes)`` relation-typed adjacency.
        latent : Tensor
            Shape ``(1, node_types)`` standard-normal latent sample.

        Returns
        -------
        Tensor
            Shape ``(1, node_types)`` logits for the new atom's type.
        """

        _, graph_vec = self.gnn(node_type, rel_adj)
        x = latent
        for layer in reversed(self.flow_layers):
            x = layer.reverse(x, graph_vec)
        return x


def build_graphaf() -> nn.Module:
    """Build a compact GraphAF model.

    Returns
    -------
    nn.Module
        ``GraphAF`` in eval mode.
    """

    return GraphAF().eval()


def example_input_graphaf() -> tuple[Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_graphaf`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(node_type, rel_adj, latent)`` for a 6-atom partial graph.
    """

    torch.manual_seed(2)
    node_type = torch.randint(0, 9, (6,))
    rel_adj = (torch.rand(3, 6, 6) > 0.7).float()
    latent = torch.randn(1, 9)
    return node_type, rel_adj, latent


# ---------------------------------------------------------------------------
# 4. GraphBP -- 3D pocket-conditioned atom-by-atom molecule generator
# ---------------------------------------------------------------------------


class _ContinuousFilterConv(nn.Module):
    """SchNet-style continuous-filter interaction block."""

    def __init__(self, hidden: int, n_gaussians: int = 16, cutoff: float = 8.0) -> None:
        super().__init__()
        self.n_gaussians = n_gaussians
        self.cutoff = cutoff
        self.register_buffer("offsets", torch.linspace(0.0, cutoff, n_gaussians))
        self.filter_net = nn.Sequential(
            nn.Linear(n_gaussians, hidden), nn.Softplus(), nn.Linear(hidden, hidden)
        )
        self.atom_lin = nn.Linear(hidden, hidden)
        self.out_lin = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Softplus(), nn.Linear(hidden, hidden)
        )

    def forward(self, h: Tensor, pos: Tensor) -> Tensor:
        dist = torch.cdist(pos, pos)  # (n, n)
        rbf = torch.exp(-((dist.unsqueeze(-1) - self.offsets) ** 2))  # (n, n, n_gaussians)
        filt = self.filter_net(rbf)  # (n, n, hidden)
        msg = self.atom_lin(h).unsqueeze(0) * filt  # (n, n, hidden)
        agg = msg.sum(dim=1)
        return h + self.out_lin(agg)


class GraphBP(nn.Module):
    """3D atom-by-atom ligand generator conditioned on a fixed pocket.

    A continuous-filter (SchNet-like) network embeds joint pocket+partial
    ligand atoms; a focus classifier scores each existing atom as the next
    attachment point, and four small conditional affine-coupling flows
    (element type, bond distance, bond angle, dihedral torsion) generate
    the new atom's identity and local spherical placement.
    """

    def __init__(self, n_types: int = 6, hidden: int = 24) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_types, hidden)
        self.conv1 = _ContinuousFilterConv(hidden)
        self.conv2 = _ContinuousFilterConv(hidden)
        self.focus_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        self.type_flow = _ConditionalAffineCoupling(n_types, hidden, hidden)
        self.dist_flow = _ConditionalAffineCoupling(1, hidden, hidden)
        self.angle_flow = _ConditionalAffineCoupling(1, hidden * 2, hidden)
        self.torsion_flow = _ConditionalAffineCoupling(1, hidden * 3, hidden)
        self.n_types = n_types

    def forward(
        self,
        atom_type: Tensor,
        pos: Tensor,
        latent_type: Tensor,
        latent_dist: Tensor,
        latent_angle: Tensor,
        latent_torsion: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Embed the pocket+ligand and invert flows to place one new atom.

        Parameters
        ----------
        atom_type : Tensor
            Shape ``(n_atoms,)`` integer element types (pocket + ligand so far).
        pos : Tensor
            Shape ``(n_atoms, 3)`` 3D coordinates.
        latent_type, latent_dist, latent_angle, latent_torsion : Tensor
            Standard-normal latents for the new atom's type / distance /
            angle / torsion, shapes ``(1, n_types)``, ``(1, 1)``, ``(1, 1)``,
            ``(1, 1)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            ``(focus_scores, type_logits, dist, angle, torsion)``.
        """

        h = self.embed(atom_type)
        h = self.conv1(h, pos)
        h = self.conv2(h, pos)
        focus_scores = self.focus_mlp(h)

        focus_id = int(torch.argmax(focus_scores.squeeze(-1)).item())
        focus_feat = h[focus_id : focus_id + 1]

        type_logits = self.type_flow.reverse(latent_type, focus_feat)
        dist = self.dist_flow.reverse(latent_dist, focus_feat)

        second_id = (focus_id + 1) % h.shape[0]
        angle_cond = torch.cat([focus_feat, h[second_id : second_id + 1]], dim=-1)
        angle = self.angle_flow.reverse(latent_angle, angle_cond)

        third_id = (focus_id + 2) % h.shape[0]
        torsion_cond = torch.cat(
            [focus_feat, h[second_id : second_id + 1], h[third_id : third_id + 1]], dim=-1
        )
        torsion = self.torsion_flow.reverse(latent_torsion, torsion_cond)

        return focus_scores, type_logits, dist, angle, torsion


def build_graphbp() -> nn.Module:
    """Build a compact GraphBP model.

    Returns
    -------
    nn.Module
        ``GraphBP`` in eval mode.
    """

    return GraphBP().eval()


def example_input_graphbp() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_graphbp`.

    Returns
    -------
    tuple[Tensor, ...]
        ``(atom_type, pos, latent_type, latent_dist, latent_angle,
        latent_torsion)`` for a 5-atom pocket+ligand scene.
    """

    torch.manual_seed(3)
    atom_type = torch.randint(0, 6, (5,))
    pos = torch.randn(5, 3) * 3.0
    latent_type = torch.randn(1, 6)
    latent_dist = torch.randn(1, 1)
    latent_angle = torch.randn(1, 1)
    latent_torsion = torch.randn(1, 1)
    return atom_type, pos, latent_type, latent_dist, latent_angle, latent_torsion


# ---------------------------------------------------------------------------
# 5. GraphDF -- discrete normalizing flow over one-hot graph node/edge types
# ---------------------------------------------------------------------------


def _one_hot_argmax(logits: Tensor, temperature: float = 0.1) -> Tensor:
    """Straight-through hard one-hot of the argmax with a softmax-temperature backward.

    Parameters
    ----------
    logits : Tensor
        Shape ``(..., vocab_size)``.
    temperature : float
        Softmax temperature used only for the (implicit, autograd) backward pass.

    Returns
    -------
    Tensor
        Hard one-hot forward value with a soft-softmax gradient path.
    """

    vocab_size = logits.shape[-1]
    hard = F.one_hot(torch.argmax(logits, dim=-1), vocab_size).to(logits.dtype)
    soft = F.softmax(logits / temperature, dim=-1)
    return soft + (hard - soft).detach()


class _DiscreteAffineCoupling(nn.Module):
    """Discrete-flow affine-coupling step (GraphDF's ``ST_Net_Exp`` + hard projection)."""

    def __init__(self, dim: int, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cond_dim, hidden), nn.Tanh(), nn.Linear(hidden, dim * 2))

    def forward(self, x_onehot: Tensor, condition: Tensor) -> Tensor:
        scale, bias = self.mlp(condition).chunk(2, dim=-1)
        transformed = x_onehot * torch.exp(scale) + bias
        return _one_hot_argmax(transformed)


class GraphDF(nn.Module):
    """Discrete-flow autoregressive graph generator (no dequantization noise).

    Mirrors :class:`GraphAF`'s relational-GNN condition encoder but keeps
    every flow step's state a *hard* one-hot vector via a straight-through
    ``one_hot_argmax`` projection, avoiding continuous dequantization.
    """

    def __init__(self, node_types: int = 9, hidden: int = 24, n_flow_layers: int = 3) -> None:
        super().__init__()
        self.gnn = _RelationalGNN(node_types, hidden)
        self.flow_layers = nn.ModuleList(
            [_DiscreteAffineCoupling(node_types, hidden, hidden) for _ in range(n_flow_layers)]
        )
        self.node_types = node_types

    def forward(self, node_type: Tensor, rel_adj: Tensor, init_onehot: Tensor) -> Tensor:
        """Condition on the current graph and run the discrete flow forward.

        Parameters
        ----------
        node_type : Tensor
            Shape ``(n_nodes,)`` integer atom types of the existing graph.
        rel_adj : Tensor
            Shape ``(n_relations, n_nodes, n_nodes)`` relation-typed adjacency.
        init_onehot : Tensor
            Shape ``(1, node_types)`` initial one-hot (e.g. uniform prior draw).

        Returns
        -------
        Tensor
            Shape ``(1, node_types)`` final discrete one-hot atom type.
        """

        _, graph_vec = self.gnn(node_type, rel_adj)
        x = init_onehot
        for layer in self.flow_layers:
            x = layer(x, graph_vec)
        return x


def build_graphdf() -> nn.Module:
    """Build a compact GraphDF model.

    Returns
    -------
    nn.Module
        ``GraphDF`` in eval mode.
    """

    return GraphDF().eval()


def example_input_graphdf() -> tuple[Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_graphdf`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(node_type, rel_adj, init_onehot)`` for a 6-atom partial graph.
    """

    torch.manual_seed(4)
    node_type = torch.randint(0, 9, (6,))
    rel_adj = (torch.rand(3, 6, 6) > 0.7).float()
    init_onehot = F.one_hot(torch.randint(0, 9, (1,)), 9).float()
    return node_type, rel_adj, init_onehot


# ---------------------------------------------------------------------------
# 6. GraphGDP -- score-based diffusion over dense graph adjacency matrices
# ---------------------------------------------------------------------------


def _sinusoidal_timestep_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class GraphGDP(nn.Module):
    """Position-enhanced graph score network for adjacency-matrix diffusion.

    Diffusion-timestep-conditioned, permutation-equivariant GNN that
    predicts a denoising update to a dense (continuous-relaxed) adjacency
    matrix, using degree one-hot and random-walk positional node features
    computed purely from the noisy graph itself.
    """

    def __init__(self, max_nodes: int = 12, hidden: int = 24, rw_depth: int = 4) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.rw_depth = rw_depth
        self.hidden = hidden
        self.degree_max = max_nodes // 2

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.degree_proj = nn.Linear(self.degree_max + 1, hidden)
        self.rw_proj = nn.Linear(rw_depth, hidden)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, adj: Tensor, timestep: Tensor) -> Tensor:
        """Predict a denoising score for the (noisy) dense adjacency matrix.

        Parameters
        ----------
        adj : Tensor
            Shape ``(n, n)`` symmetric noisy continuous-relaxed adjacency.
        timestep : Tensor
            Shape ``(1,)`` diffusion timestep index.

        Returns
        -------
        Tensor
            Shape ``(n, n)`` predicted score (denoising direction), symmetrized.
        """

        n = adj.shape[0]
        temb = self.time_mlp(_sinusoidal_timestep_embedding(timestep, self.hidden))  # (1, hidden)

        cont_adj = adj.clamp(min=0.0, max=1.0)
        degree = cont_adj.sum(dim=-1).clamp(max=float(self.degree_max))
        degree_onehot = F.one_hot(degree.round().long(), self.degree_max + 1).float()
        node_feat = self.degree_proj(degree_onehot)

        rw_mat = cont_adj / (cont_adj.sum(dim=-1, keepdim=True) + 1e-6)
        rw_feats = []
        power = torch.eye(n)
        for _ in range(self.rw_depth):
            power = power @ rw_mat
            rw_feats.append(torch.diagonal(power).unsqueeze(-1))
        rw_feat = self.rw_proj(torch.cat(rw_feats, dim=-1))

        h = self.node_mlp(torch.cat([node_feat, rw_feat], dim=-1)) + temb
        pair = torch.cat(
            [
                h.unsqueeze(1).expand(n, n, -1),
                h.unsqueeze(0).expand(n, n, -1),
                temb.expand(n, n, -1),
            ],
            dim=-1,
        )
        score = self.edge_mlp(pair).squeeze(-1)
        return 0.5 * (score + score.transpose(0, 1))


def build_graphgdp() -> nn.Module:
    """Build a compact GraphGDP score network.

    Returns
    -------
    nn.Module
        ``GraphGDP`` in eval mode.
    """

    return GraphGDP().eval()


def example_input_graphgdp() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_graphgdp`.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(adj, timestep)`` for an 8-node noisy adjacency matrix.
    """

    torch.manual_seed(5)
    raw = torch.rand(8, 8)
    adj = 0.5 * (raw + raw.transpose(0, 1))
    adj.fill_diagonal_(0.0)
    timestep = torch.tensor([250.0])
    return adj, timestep


MENAGERIE_ENTRIES = [
    ("GrammarVAE", "build_grammar_vae", "example_input_grammar_vae", "2017", "GEN"),
    ("Graph2Edits", "build_graph2edits", "example_input_graph2edits", "2023", "GRAPH"),
    ("GraphAF", "build_graphaf", "example_input_graphaf", "2020", "GRAPH"),
    ("GraphBP", "build_graphbp", "example_input_graphbp", "2022", "GRAPH"),
    ("GraphDF", "build_graphdf", "example_input_graphdf", "2021", "GRAPH"),
    ("GraphGDP", "build_graphgdp", "example_input_graphgdp", "2022", "GRAPH"),
]
