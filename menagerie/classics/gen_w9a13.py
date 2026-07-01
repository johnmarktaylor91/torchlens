"""Compact faithful reimplementations of six combinatorial-optimization / systems-RL families.

Sources checked (paper + official source, no clone/pip-install; reimplemented from scratch
in base-env torch):

  - Learning to Branch GCNN: Gasse, Chetelat, Ferroni, Charlin, Lodi, "Exact Combinatorial
    Optimization with Graph Convolutional Neural Networks", NeurIPS 2019, arXiv:1906.01629.
    Official repos ``ds4dm/learn2branch`` (TensorFlow) and ``ds4dm/learn2branch-ecole``
    (PyTorch + PyTorch Geometric reimplementation, ``model/model.py``, class ``GNNPolicy``).
    Distinctive mechanism: MILP branching state is represented as a **bipartite graph**
    (constraint nodes on one side, variable nodes on the other, edges = nonzero
    constraint-matrix coefficients carrying a scalar edge feature) and scored by two
    alternating half-convolutions -- a ``BipartiteGraphConvolution`` message-passing layer
    that first aggregates variable-side messages into constraints (``conv_v_to_c``), then a
    *second*, separately-parameterized instance of the same layer that aggregates the
    updated constraint-side messages back into variables (``conv_c_to_v``) using the
    *reversed* edge index -- each message is a small MLP over
    ``feature_left(x_i) + feature_edge(e_ij) + feature_right(x_j)``, and a final per-variable
    MLP head produces a branching score. Reimplemented here as a compact bipartite
    message-passing block (implemented directly with ``index_add_`` scatter-sum, no
    ``torch_geometric.nn.MessagePassing`` dependency) mirroring the two-half-convolution
    variable<->constraint alternation and edge-conditioned messages that define this model
    (vs. a generic homogeneous GNN).

  - Multi-Decoder Attention Model (MDAM): Xin, Song, Cao, Zhang, "Multi-Decoder Attention
    Model with Embedding Glimpse for Solving Vehicle Routing Problems", AAAI 2021,
    arXiv:2012.10638. Official repo ``liangxinedu/MDAM`` (``nets/attention_model.py``,
    class ``AttentionModel``). Distinctive mechanism: a single shared Transformer-style
    graph encoder over the node coordinates feeds **K parallel decoder "paths"** with
    *identical structure but unshared parameters* -- each path owns its own glimpse-key/
    glimpse-value/logit-key projection (``project_node_embeddings[i]``), context projection
    (``project_fixed_context[i]``), and output projection (``project_out[i]``) -- so the K
    paths independently attend over the *same* encoder embeddings and each produce a
    distinct policy (probability distribution) over which node to visit next, giving K
    diverse rollouts from one encoder pass instead of one policy. An additional
    "Embedding Glimpse" layer perturbs the shared node embeddings per-path via a small
    learned attention pass before each path's decoding starts, encouraging path diversity.
    Reimplemented here as one shared multi-head-attention encoder + K unshared
    single-step attention decoder heads (each with its own glimpse/logit projections) plus
    a lightweight per-path embedding-glimpse perturbation -- the defining "one encoder, K
    unshared decoders" mechanism (vs. an ordinary single-decoder attention model).

  - Nazari VRP Attention/RL Model: Nazari, Oroojlooy, Snyder, Takac, "Reinforcement
    Learning for Solving the Vehicle Routing Problem", arXiv:1802.04240 (NeurIPS 2018).
    Official-adjacent PyTorch reimplementation ``mveres01/pytorch-drl4vrp``. Distinctive
    mechanism: unlike a plain seq2seq Pointer Network, static node features (2D
    coordinates) are embedded ONCE via a 1D convolution ("encoder") while *dynamic*
    per-node features (remaining demand, vehicle load) are re-embedded at *every* decoding
    step because they change as the tour is built; a GRU decoder tracks only the
    previously-chosen node, and at each step an attention layer over
    ``[static_embedding; dynamic_embedding]`` (both re-combined every step) produces a
    pointer distribution over unvisited nodes -- so the model explicitly separates
    step-invariant (static) and step-varying (dynamic) node embeddings rather than
    treating all node features as static, which is what distinguishes it from a plain
    Bello-style pointer network. Reimplemented here as a static-feature 1D-conv encoder +
    a per-step dynamic-feature encoder + GRU decoder + additive pointer attention over the
    concatenated static/dynamic embeddings.

  - Neural Combinatorial Optimization RL (Bello): Bello, Pham, Le, Norouzi, Bengio,
    "Neural Combinatorial Optimization with Reinforcement Learning", arXiv:1611.09940
    (ICLR 2017 workshop). Canonical PyTorch reimplementation
    ``pemami4911/neural-combinatorial-rl-pytorch``. Distinctive mechanism: a classic
    Pointer Network (Vinyals et al.) trained with REINFORCE -- an LSTM encoder reads the
    input sequence (e.g. 2D city coordinates) into a fixed-size sequence of hidden states,
    an LSTM decoder then runs step-by-step, and at each step a **glimpse attention** layer
    (a small additive-attention pass over the encoder states, itself feeding back into the
    query before the final pointing distribution) produces logits over the encoder
    positions; encoder positions already selected are masked out so each input is pointed
    to exactly once, closing the pointer-network permutation-decoding loop. Reimplemented
    here as an LSTM encoder/decoder pointer network with an explicit glimpse pass before
    the final pointing softmax, plus visitation masking -- the defining glimpse-then-point
    mechanism (vs. a single-pass attention pointer).

  - NF-GNN (Network Flow GNN for Malware): Busch, Kocheturov, Tresp, Seidl, "NF-GNN:
    Network Flow Graph Neural Networks for Malware Detection and Classification",
    SSDBM 2021, arXiv:2103.03939. Official repo ``birsbear/nfgnn``. Distinctive mechanism:
    network traffic is modeled as a graph of communicating hosts where **edges carry
    multi-dimensional flow-statistics features** (e.g. byte/packet counts, duration,
    protocol) rather than a single scalar weight; a message-passing layer updates BOTH
    node and edge representations jointly at every layer -- each edge embedding is
    refreatured from its two endpoint node embeddings plus its own previous embedding, and
    each node embedding aggregates messages built from its incident (updated) edge
    embeddings -- so information flows node->edge->node every layer instead of only
    node->node, letting rich per-flow statistics (not just topology) drive the malware
    classification produced by a final graph-level (mean-pool) readout + MLP head.
    Reimplemented here as a compact joint node/edge co-embedding message-passing block
    (edge-update-then-node-aggregate each layer) + graph readout classifier -- the defining
    edge-feature-centric co-embedding mechanism (vs. a topology-only GNN).

  - Pensieve: Mao, Netravali, Alizadeh, "Neural Adaptive Video Streaming with Pensieve",
    SIGCOMM 2017. Official repo ``hongzimao/pensieve`` (TensorFlow 1 + TFLearn,
    ``sim/a3c.py``, ``ActorNetwork.create_actor_network`` / ``CriticNetwork``). Distinctive
    mechanism: the actor-critic state is a heterogeneous ``(S_INFO=6, S_LEN)`` tensor
    whose six rows are semantically different (last measured throughput, last download
    time, a length-``S_LEN`` throughput history, a length-``S_LEN`` download-time history,
    a length-``A_DIM`` next-chunk-size-per-bitrate vector, and scalar buffer occupancy /
    chunks-remaining), so the network does **not** run one homogeneous encoder over the
    whole state tensor -- it splits the state into six named slices and feeds the two
    scalar rows through small per-row fully-connected embeddings, the two sequential rows
    (throughput history, download-time history) and the next-chunk-size vector through
    **1D convolutions** (kernel 4) since they are time series / per-bitrate vectors, then
    concatenates all six branch outputs and passes them through a shared dense trunk to a
    softmax bitrate-selection policy (actor) with a parallel critic head sharing the same
    six-branch state encoder but outputting a scalar value. Reimplemented here as the same
    six-branch heterogeneous state encoder (2 FC branches + 3 conv1d branches + 1 FC
    branch) feeding a shared trunk, with separate actor (softmax over bitrates) and critic
    (scalar value) heads -- the defining "state is not one flat tensor, it's six
    differently-shaped signals" mechanism.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 1. Learning to Branch GCNN (bipartite GCN for MILP branching)
# ---------------------------------------------------------------------------


class _BipartiteConv(nn.Module):
    """One half-convolution: aggregate ``src`` node messages into ``dst`` nodes.

    Mirrors ``BipartiteGraphConvolution`` in ds4dm/learn2branch-ecole: an
    edge-conditioned message ``feature_left(dst_i) + feature_edge(e_ij) +
    feature_right(src_j)`` is scatter-summed into each destination node, then
    concatenated with the destination's previous embedding and passed through
    an output MLP.
    """

    def __init__(self, emb_size: int = 32) -> None:
        super().__init__()
        self.feature_left = nn.Linear(emb_size, emb_size)
        self.feature_edge = nn.Linear(1, emb_size, bias=False)
        self.feature_right = nn.Linear(emb_size, emb_size, bias=False)
        self.feature_final = nn.Sequential(nn.ReLU(), nn.Linear(emb_size, emb_size))
        self.output_module = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )

    def forward(
        self,
        src: Tensor,
        dst: Tensor,
        src_idx: Tensor,
        dst_idx: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        """Aggregate messages from ``src`` nodes into ``dst`` nodes.

        Parameters
        ----------
        src : Tensor
            Shape ``(n_src, emb_size)`` source-side node embeddings.
        dst : Tensor
            Shape ``(n_dst, emb_size)`` destination-side node embeddings.
        src_idx : Tensor
            Shape ``(n_edges,)`` long tensor of source-node indices per edge.
        dst_idx : Tensor
            Shape ``(n_edges,)`` long tensor of destination-node indices per edge.
        edge_features : Tensor
            Shape ``(n_edges, 1)`` scalar edge feature per edge.

        Returns
        -------
        Tensor
            Shape ``(n_dst, emb_size)`` updated destination embeddings.
        """

        msg = (
            self.feature_left(dst[dst_idx])
            + self.feature_edge(edge_features)
            + self.feature_right(src[src_idx])
        )
        msg = self.feature_final(msg)
        agg = dst.new_zeros(dst.shape[0], msg.shape[-1])
        agg.index_add_(0, dst_idx, msg)
        return self.output_module(torch.cat([agg, dst], dim=-1))


class LearnToBranchGCNN(nn.Module):
    """Bipartite GCN for MILP branching variable scoring (Gasse et al. 2019)."""

    def __init__(
        self,
        cons_nfeats: int = 5,
        edge_nfeats: int = 1,
        var_nfeats: int = 19,
        emb_size: int = 32,
    ) -> None:
        super().__init__()
        self.cons_embedding = nn.Sequential(
            nn.Linear(cons_nfeats, emb_size), nn.ReLU(), nn.Linear(emb_size, emb_size), nn.ReLU()
        )
        self.var_embedding = nn.Sequential(
            nn.Linear(var_nfeats, emb_size), nn.ReLU(), nn.Linear(emb_size, emb_size), nn.ReLU()
        )
        self.conv_v_to_c = _BipartiteConv(emb_size)
        self.conv_c_to_v = _BipartiteConv(emb_size)
        self.output_module = nn.Sequential(
            nn.Linear(emb_size, emb_size), nn.ReLU(), nn.Linear(emb_size, 1, bias=False)
        )

    def forward(
        self,
        constraint_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        variable_features: Tensor,
    ) -> Tensor:
        """Score every variable node for branching priority.

        Parameters
        ----------
        constraint_features : Tensor
            Shape ``(n_cons, cons_nfeats)``.
        edge_index : Tensor
            Shape ``(2, n_edges)`` long tensor of ``(cons_idx, var_idx)`` pairs.
        edge_features : Tensor
            Shape ``(n_edges, edge_nfeats)``.
        variable_features : Tensor
            Shape ``(n_vars, var_nfeats)``.

        Returns
        -------
        Tensor
            Shape ``(n_vars,)`` per-variable branching scores.
        """

        cons = self.cons_embedding(constraint_features)
        var = self.var_embedding(variable_features)
        cons_idx, var_idx = edge_index[0], edge_index[1]
        # Half-conv 1: variables -> constraints (messages flow var_idx -> cons_idx).
        cons = self.conv_v_to_c(var, cons, var_idx, cons_idx, edge_features)
        # Half-conv 2: constraints -> variables (messages flow cons_idx -> var_idx).
        var = self.conv_c_to_v(cons, var, cons_idx, var_idx, edge_features)
        return self.output_module(var).squeeze(-1)


def build_learning_to_branch_gcnn() -> nn.Module:
    """Build a compact Learning to Branch bipartite GCN.

    Returns
    -------
    nn.Module
        Random-init compact ``LearnToBranchGCNN``.
    """

    return LearnToBranchGCNN(cons_nfeats=5, edge_nfeats=1, var_nfeats=19, emb_size=16).eval()


def example_input_learning_to_branch_gcnn() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a small random bipartite MILP graph.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(constraint_features, edge_index, edge_features, variable_features)``.
    """

    n_cons, n_vars, n_edges = 6, 10, 24
    cons_feat = torch.randn(n_cons, 5)
    var_feat = torch.randn(n_vars, 19)
    edge_index = torch.stack(
        [torch.randint(0, n_cons, (n_edges,)), torch.randint(0, n_vars, (n_edges,))], dim=0
    )
    edge_feat = torch.randn(n_edges, 1)
    return cons_feat, edge_index, edge_feat, var_feat


# ---------------------------------------------------------------------------
# 2. Multi-Decoder Attention Model (MDAM)
# ---------------------------------------------------------------------------


class _MHASelfAttn(nn.Module):
    """Standard multi-head self-attention encoder block (pre-norm, feed-forward)."""

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply one self-attention + feed-forward block.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, n_nodes, embed_dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch, n_nodes, embed_dim)``.
        """

        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class _MDAMDecoderPath(nn.Module):
    """One unshared decoder "path": glimpse attention + pointing logits over nodes."""

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.project_node = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.project_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.glimpse_perturb = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, node_embed: Tensor, graph_context: Tensor) -> Tensor:
        """Produce a per-node pointing distribution for this path.

        Parameters
        ----------
        node_embed : Tensor
            Shape ``(batch, n_nodes, embed_dim)`` shared encoder embeddings.
        graph_context : Tensor
            Shape ``(batch, embed_dim)`` mean-pooled graph context (decode query seed).

        Returns
        -------
        Tensor
            Shape ``(batch, n_nodes)`` pointing log-probabilities for this path.
        """

        batch, n_nodes, embed_dim = node_embed.shape
        # Embedding-glimpse: perturb the shared node embeddings for this path.
        perturbed, _ = self.glimpse_perturb(node_embed, node_embed, node_embed)
        node_embed = node_embed + perturbed

        glimpse_key, glimpse_val, logit_key = self.project_node(node_embed).chunk(3, dim=-1)
        query = self.project_query(self.project_context(graph_context)).unsqueeze(1)

        def _split_heads(t: Tensor) -> Tensor:
            return t.view(batch, -1, self.n_heads, self.head_dim).transpose(1, 2)

        q = _split_heads(query)
        k = _split_heads(glimpse_key)
        v = _split_heads(glimpse_val)
        compat = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        heads = torch.matmul(F.softmax(compat, dim=-1), v)
        glimpse = self.project_out(heads.transpose(1, 2).reshape(batch, 1, embed_dim))

        logits = torch.matmul(glimpse, logit_key.transpose(-2, -1)).squeeze(1) / math.sqrt(
            embed_dim
        )
        return F.log_softmax(logits, dim=-1)


class MDAM(nn.Module):
    """Multi-Decoder Attention Model: shared encoder + K unshared decoder paths."""

    def __init__(
        self,
        node_dim: int = 2,
        embed_dim: int = 32,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_paths: int = 3,
    ) -> None:
        super().__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.encoder = nn.ModuleList(
            [_MHASelfAttn(embed_dim, n_heads) for _ in range(n_encoder_layers)]
        )
        self.paths = nn.ModuleList([_MDAMDecoderPath(embed_dim, n_heads) for _ in range(n_paths)])

    def forward(self, nodes: Tensor) -> Tensor:
        """Encode nodes once and decode K unshared pointing distributions.

        Parameters
        ----------
        nodes : Tensor
            Shape ``(batch, n_nodes, node_dim)`` node coordinates.

        Returns
        -------
        Tensor
            Shape ``(batch, n_paths, n_nodes)`` stacked per-path log-probabilities.
        """

        h = self.init_embed(nodes)
        for layer in self.encoder:
            h = layer(h)
        graph_context = h.mean(dim=1)
        return torch.stack([path(h, graph_context) for path in self.paths], dim=1)


def build_mdam() -> nn.Module:
    """Build a compact MDAM.

    Returns
    -------
    nn.Module
        Random-init compact ``MDAM``.
    """

    return MDAM(node_dim=2, embed_dim=32, n_heads=4, n_encoder_layers=2, n_paths=3).eval()


def example_input_mdam() -> Tensor:
    """Create a batch of random VRP-style node coordinates.

    Returns
    -------
    Tensor
        Shape ``(batch, n_nodes, 2)``.
    """

    return torch.rand(4, 12, 2)


# ---------------------------------------------------------------------------
# 3. Nazari VRP Attention/RL Model
# ---------------------------------------------------------------------------


class NazariVRPAttention(nn.Module):
    """Static/dynamic-split pointer attention model for VRP (Nazari et al. 2018)."""

    def __init__(self, static_dim: int = 2, dynamic_dim: int = 2, hidden_dim: int = 32) -> None:
        super().__init__()
        self.static_encoder = nn.Conv1d(static_dim, hidden_dim, kernel_size=1)
        self.dynamic_encoder = nn.Conv1d(dynamic_dim, hidden_dim, kernel_size=1)
        self.decoder_input_encoder = nn.Conv1d(static_dim, hidden_dim, kernel_size=1)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.attn_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_ref = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, static: Tensor, dynamic: Tensor, n_steps: int = 5) -> Tensor:
        """Roll out ``n_steps`` pointer-attention decoding steps.

        Parameters
        ----------
        static : Tensor
            Shape ``(batch, static_dim, n_nodes)`` static node coordinates.
        dynamic : Tensor
            Shape ``(batch, dynamic_dim, n_nodes)`` dynamic node features
            (e.g. demand, load), assumed fixed across the toy rollout.
        n_steps : int
            Number of decode steps to unroll.

        Returns
        -------
        Tensor
            Shape ``(batch, n_steps, n_nodes)`` pointer log-probabilities per step.
        """

        batch, _, n_nodes = static.shape
        static_h = self.static_encoder(static).transpose(1, 2)  # (batch, n_nodes, hidden)
        dynamic_h = self.dynamic_encoder(dynamic).transpose(1, 2)

        decoder_hidden = static_h.new_zeros(batch, static_h.shape[-1])
        last_node = static[:, :, 0:1]  # start at depot

        outputs = []
        for _ in range(n_steps):
            decoder_input = self.decoder_input_encoder(last_node).squeeze(-1)
            decoder_hidden = self.gru(decoder_input, decoder_hidden)

            combined = torch.cat([static_h, dynamic_h], dim=-1)
            ref = self.attn_ref(combined)
            query = self.attn_query(decoder_hidden).unsqueeze(1)
            logits = self.attn_v(torch.tanh(ref + query)).squeeze(-1)
            log_probs = F.log_softmax(logits, dim=-1)
            outputs.append(log_probs)

            chosen = log_probs.argmax(dim=-1)
            last_node = static.gather(2, chosen.view(batch, 1, 1).expand(batch, static.shape[1], 1))

        return torch.stack(outputs, dim=1)


def build_nazari_vrp_attention() -> nn.Module:
    """Build a compact Nazari VRP attention/RL model.

    Returns
    -------
    nn.Module
        Random-init compact ``NazariVRPAttention``.
    """

    return NazariVRPAttention(static_dim=2, dynamic_dim=2, hidden_dim=32).eval()


def example_input_nazari_vrp_attention() -> tuple[Tensor, Tensor]:
    """Create random static/dynamic VRP node feature tensors.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(static, dynamic)`` each shape ``(batch, feat_dim, n_nodes)``.
    """

    batch, n_nodes = 4, 15
    static = torch.rand(batch, 2, n_nodes)
    dynamic = torch.rand(batch, 2, n_nodes)
    return static, dynamic


# ---------------------------------------------------------------------------
# 4. Neural Combinatorial Optimization RL (Bello) -- LSTM pointer network
# ---------------------------------------------------------------------------


class NeuralCombinatorialPointerNet(nn.Module):
    """LSTM encoder/decoder Pointer Network with a glimpse pass (Bello et al. 2016)."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.glimpse_ref = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.glimpse_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.glimpse_v = nn.Linear(hidden_dim, 1, bias=False)

        self.point_ref = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.point_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.point_v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a sequence then decode a full pointer permutation via glimpse+point.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, input_dim)`` input sequence (e.g. city coords).

        Returns
        -------
        Tensor
            Shape ``(batch, seq_len, seq_len)`` per-step pointing log-probabilities
            (masked so already-chosen positions get ``-inf``).
        """

        batch, seq_len, _ = x.shape
        embedded = self.embed(x)
        enc_out, (h, c) = self.encoder(embedded)

        mask = x.new_zeros(batch, seq_len, dtype=torch.bool)
        decoder_input = embedded[:, 0:1, :]
        outputs = []
        for _step in range(seq_len):
            dec_out, (h, c) = self.decoder(decoder_input, (h, c))
            query = dec_out[:, -1, :]

            # Glimpse: attention over encoder states, fed back into the query.
            g_logits = self.glimpse_v(
                torch.tanh(self.glimpse_ref(enc_out) + self.glimpse_query(query).unsqueeze(1))
            ).squeeze(-1)
            g_logits = g_logits.masked_fill(mask, float("-inf"))
            g_weights = F.softmax(g_logits, dim=-1).unsqueeze(-1)
            glimpsed_query = (g_weights * enc_out).sum(dim=1)

            # Point: final pointing distribution using the glimpsed query.
            p_logits = self.point_v(
                torch.tanh(self.point_ref(enc_out) + self.point_query(glimpsed_query).unsqueeze(1))
            ).squeeze(-1)
            p_logits = p_logits.masked_fill(mask, float("-inf"))
            log_probs = F.log_softmax(p_logits, dim=-1)
            outputs.append(log_probs)

            chosen = log_probs.argmax(dim=-1)
            mask = mask.scatter(1, chosen.unsqueeze(-1), True)
            decoder_input = embedded.gather(
                1, chosen.view(batch, 1, 1).expand(batch, 1, embedded.shape[-1])
            )

        return torch.stack(outputs, dim=1)


def build_neural_combinatorial_optimization_rl() -> nn.Module:
    """Build a compact Bello-style LSTM pointer network.

    Returns
    -------
    nn.Module
        Random-init compact ``NeuralCombinatorialPointerNet``.
    """

    return NeuralCombinatorialPointerNet(input_dim=2, hidden_dim=32).eval()


def example_input_neural_combinatorial_optimization_rl() -> Tensor:
    """Create a batch of random TSP-style coordinate sequences.

    Returns
    -------
    Tensor
        Shape ``(batch, seq_len, 2)``.
    """

    return torch.rand(4, 10, 2)


# ---------------------------------------------------------------------------
# 5. NF-GNN (Network Flow GNN for malware detection)
# ---------------------------------------------------------------------------


class NFGNNLayer(nn.Module):
    """One joint node/edge co-embedding message-passing layer."""

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.edge_update = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, edge_dim)
        )
        self.node_update = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim), nn.ReLU(), nn.Linear(node_dim, node_dim)
        )

    def forward(
        self, node_feat: Tensor, edge_index: Tensor, edge_feat: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Update edge embeddings from endpoints, then aggregate into node embeddings.

        Parameters
        ----------
        node_feat : Tensor
            Shape ``(n_nodes, node_dim)``.
        edge_index : Tensor
            Shape ``(2, n_edges)`` long tensor of ``(src, dst)`` pairs.
        edge_feat : Tensor
            Shape ``(n_edges, edge_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(node_feat, edge_feat)``.
        """

        src, dst = edge_index[0], edge_index[1]
        new_edge_feat = self.edge_update(
            torch.cat([node_feat[src], node_feat[dst], edge_feat], dim=-1)
        )

        msg = self.node_update(torch.cat([node_feat[src], new_edge_feat], dim=-1))
        agg = node_feat.new_zeros(node_feat.shape[0], msg.shape[-1])
        agg.index_add_(0, dst, msg)
        new_node_feat = node_feat + agg
        return new_node_feat, new_edge_feat


class NFGNN(nn.Module):
    """Network Flow GNN for malware detection/classification (Busch et al. 2021)."""

    def __init__(
        self,
        node_in_dim: int = 8,
        edge_in_dim: int = 6,
        hidden_dim: int = 32,
        n_layers: int = 2,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        self.layers = nn.ModuleList([NFGNNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, node_feat: Tensor, edge_index: Tensor, edge_feat: Tensor) -> Tensor:
        """Classify a network-flow graph.

        Parameters
        ----------
        node_feat : Tensor
            Shape ``(n_nodes, node_in_dim)`` per-host node features.
        edge_index : Tensor
            Shape ``(2, n_edges)`` long tensor of ``(src, dst)`` pairs.
        edge_feat : Tensor
            Shape ``(n_edges, edge_in_dim)`` per-flow statistics.

        Returns
        -------
        Tensor
            Shape ``(n_classes,)`` graph-level classification logits.
        """

        h = self.node_embed(node_feat)
        e = self.edge_embed(edge_feat)
        for layer in self.layers:
            h, e = layer(h, edge_index, e)
        pooled = h.mean(dim=0)
        return self.readout(pooled)


def build_nf_gnn() -> nn.Module:
    """Build a compact NF-GNN.

    Returns
    -------
    nn.Module
        Random-init compact ``NFGNN``.
    """

    return NFGNN(node_in_dim=8, edge_in_dim=6, hidden_dim=24, n_layers=2, n_classes=5).eval()


def example_input_nf_gnn() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small random network-flow graph.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(node_feat, edge_index, edge_feat)``.
    """

    n_nodes, n_edges = 10, 20
    node_feat = torch.randn(n_nodes, 8)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    edge_feat = torch.randn(n_edges, 6)
    return node_feat, edge_index, edge_feat


# ---------------------------------------------------------------------------
# 6. Pensieve (ABR actor-critic with a 6-branch heterogeneous state encoder)
# ---------------------------------------------------------------------------


class _PensieveStateEncoder(nn.Module):
    """Six-branch heterogeneous state encoder shared by actor and critic.

    Mirrors ``ActorNetwork.create_actor_network`` in hongzimao/pensieve: two
    scalar rows go through per-row FC branches, three sequential rows
    (throughput history, download-time history, next-chunk sizes per
    bitrate) go through 1D convolutions, and one more scalar row (buffer /
    chunks-remaining) goes through another FC branch; all six branch
    outputs are concatenated and passed through a shared dense trunk.
    """

    def __init__(
        self, a_dim: int = 6, s_len: int = 8, branch_dim: int = 128, trunk_dim: int = 128
    ) -> None:
        super().__init__()
        self.a_dim = a_dim
        self.s_len = s_len

        self.fc_last_quality = nn.Linear(1, branch_dim)
        self.fc_buffer = nn.Linear(1, branch_dim)
        self.conv_throughput = nn.Conv1d(1, branch_dim, kernel_size=4)
        self.conv_download_time = nn.Conv1d(1, branch_dim, kernel_size=4)
        self.conv_next_chunk_sizes = nn.Conv1d(1, branch_dim, kernel_size=4)
        self.fc_chunks_remaining = nn.Linear(1, branch_dim)

        conv_out_len = s_len - 4 + 1
        next_chunk_out_len = a_dim - 4 + 1
        merged_dim = (
            branch_dim  # last quality (scalar)
            + branch_dim  # buffer (scalar)
            + branch_dim * conv_out_len  # throughput history
            + branch_dim * conv_out_len  # download-time history
            + branch_dim * next_chunk_out_len  # next-chunk sizes per bitrate
            + branch_dim  # chunks remaining (scalar)
        )
        self.trunk = nn.Linear(merged_dim, trunk_dim)

    def forward(self, state: Tensor) -> Tensor:
        """Encode the heterogeneous 6-row ABR state into one flat trunk vector.

        Parameters
        ----------
        state : Tensor
            Shape ``(batch, 6, s_len)``. Row layout: ``[last_quality,
            buffer, throughput_history, download_time_history,
            next_chunk_sizes(:a_dim), chunks_remaining]``, each row
            right-padded/zero-filled to ``s_len`` where narrower.

        Returns
        -------
        Tensor
            Shape ``(batch, trunk_dim)`` shared representation.
        """

        last_quality = F.relu(self.fc_last_quality(state[:, 0:1, -1]))
        buffer = F.relu(self.fc_buffer(state[:, 1:2, -1]))
        throughput = F.relu(self.conv_throughput(state[:, 2:3, :])).flatten(1)
        download_time = F.relu(self.conv_download_time(state[:, 3:4, :])).flatten(1)
        next_chunk_sizes = F.relu(self.conv_next_chunk_sizes(state[:, 4:5, : self.a_dim])).flatten(
            1
        )
        chunks_remaining = F.relu(self.fc_chunks_remaining(state[:, 5:6, -1]))

        merged = torch.cat(
            [last_quality, buffer, throughput, download_time, next_chunk_sizes, chunks_remaining],
            dim=-1,
        )
        return F.relu(self.trunk(merged))


class Pensieve(nn.Module):
    """Pensieve actor-critic ABR policy (Mao, Netravali, Alizadeh 2017)."""

    def __init__(
        self, a_dim: int = 6, s_len: int = 8, branch_dim: int = 32, trunk_dim: int = 32
    ) -> None:
        super().__init__()
        self.actor_encoder = _PensieveStateEncoder(a_dim, s_len, branch_dim, trunk_dim)
        self.critic_encoder = _PensieveStateEncoder(a_dim, s_len, branch_dim, trunk_dim)
        self.actor_head = nn.Linear(trunk_dim, a_dim)
        self.critic_head = nn.Linear(trunk_dim, 1)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Produce a bitrate policy distribution and a state value.

        Parameters
        ----------
        state : Tensor
            Shape ``(batch, 6, s_len)`` heterogeneous ABR state.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(policy, value)`` of shapes ``(batch, a_dim)`` (softmax
            probabilities) and ``(batch, 1)``.
        """

        policy = F.softmax(self.actor_head(self.actor_encoder(state)), dim=-1)
        value = self.critic_head(self.critic_encoder(state))
        return policy, value


def build_pensieve() -> nn.Module:
    """Build a compact Pensieve actor-critic ABR model.

    Returns
    -------
    nn.Module
        Random-init compact ``Pensieve``.
    """

    return Pensieve(a_dim=6, s_len=8, branch_dim=32, trunk_dim=32).eval()


def example_input_pensieve() -> Tensor:
    """Create a random heterogeneous 6-row ABR state batch.

    Returns
    -------
    Tensor
        Shape ``(batch, 6, s_len)``.
    """

    return torch.rand(4, 6, 8)


MENAGERIE_ENTRIES = [
    (
        "Learning to Branch GCNN",
        "build_learning_to_branch_gcnn",
        "example_input_learning_to_branch_gcnn",
        "2019",
        "GRAPH",
    ),
    ("Multi-Decoder Attention Model (MDAM)", "build_mdam", "example_input_mdam", "2021", "SEQ"),
    (
        "Nazari VRP Attention/RL Model",
        "build_nazari_vrp_attention",
        "example_input_nazari_vrp_attention",
        "2018",
        "SEQ",
    ),
    (
        "Neural Combinatorial Optimization RL",
        "build_neural_combinatorial_optimization_rl",
        "example_input_neural_combinatorial_optimization_rl",
        "2016",
        "SEQ",
    ),
    (
        "NF-GNN (Network Flow GNN for Malware)",
        "build_nf_gnn",
        "example_input_nf_gnn",
        "2021",
        "GRAPH",
    ),
    ("Pensieve", "build_pensieve", "example_input_pensieve", "2017", "RL"),
]
