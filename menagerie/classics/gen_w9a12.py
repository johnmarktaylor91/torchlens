"""Compact faithful reimplementations for build_queue rows 73-78 (W9A12).

Sources checked (repo browsed via ``gh api``, no clone/pip-install):
  - DeepGate: Li, Shi, Pan, He, Xu, "DeepGate: Learning Neural Representations
    of Logic Gates", DAC 2022, arXiv:2111.14616. Official repo
    ``zshi0616/DeepGate``, ``src/models/recgnn.py`` /
    ``src/models/dag_convgnn.py``. Distinctive mechanism: an And-Inverter-Graph
    (AIG) is embedded by a *level-synchronous, bidirectional recurrent* GNN --
    nodes are processed strictly in topological order (one DAG "level" at a
    time), a GAT-style attention aggregator pools each node's fan-in (forward
    pass) / fan-out (reverse pass) neighborhood into a message, and a GRU cell
    consumes that message to update the node's hidden state; running forward
    over all levels then backward over all levels for several rounds lets each
    gate's embedding encode both its structural function and its logic-cone
    context. Reproduced here as a compact per-level forward+reverse
    GATConv-then-GRUCell sweep over a small fixed AIG topology (levels
    computed once from the DAG edge list), which is the reference's exact
    level-by-level recurrent-GNN scheme at toy scale.
  - DeepPlace: Cheng, Yan, "DeepPlace: Learning to Place Applications using
    Deep Reinforcement Learning", NeurIPS 2021 workshop /
    ``PKUterran/DeepPlace``, ``a2c_ppo_acktr/model.py`` (``CNNBase``).
    Distinctive mechanism: joint macro+standard-cell chip placement is framed
    as RL; the actor-critic backbone fuses two encoders -- a CNN over the
    rasterized *canvas* image (current partial placement mask) and a 3-layer
    GCN over the *netlist graph* (per-cell/macro node features) -- concatenated
    into one feature vector consumed by policy and value heads. Reproduced
    here as a small CNN canvas encoder + 3-layer ``GCNConv`` netlist encoder
    whose pooled outputs are concatenated and fed to actor (placement-logit)
    and critic (value) linear heads, mirroring the reference's exact
    dual-encoder fusion.
  - Devign: Zhou, Liu, Siow, Du, Liu, "Devign: Effective Vulnerability
    Identification by Learning Comprehensive Program Semantics via Graph
    Neural Networks", NeurIPS 2019, arXiv:1909.03496. Official-derived repo
    ``epicosy/devign``, ``src/process/model.py`` (``Net``/``Conv``).
    Distinctive mechanism: a code property graph is embedded with a *Gated
    Graph Conv* (``GatedGraphConv``, GRU-updated multi-step message passing),
    then a two-branch 1-D conv classification head processes (a) the
    concatenation of the GGNN output with the original node features and (b)
    the GGNN output alone, each through parallel Conv1d+MaxPool1d stacks
    reduced to a scalar via a linear layer, and the *two scalar branch outputs
    are multiplied together* before a final sigmoid -- a distinctive gated
    dual-conv fusion head. Reproduced here at matching structure and toy
    scale.
  - FinGAT: Hsu, Tsai, Cheng, Ku, Peng, "FinGAT: Financial Graph Attention
    Networks for Recommending Top-K Profitable Stocks", TKDE 2021/arXiv,
    official repo ``Roytsai27/Financial-GraphAttention``,
    ``model/graph_pool.py`` (``CategoricalGraphAtt``). Distinctive mechanism:
    a *hierarchical* stock encoder -- each stock's weekly price/feature
    sequence is passed through a per-week GRU + temporal-attention pooling
    block (``SequenceEncoder``/``AttentionBlock``) to get a stock embedding;
    stock embeddings are then related two ways: an *inner* ``GATConv`` over
    intra-sector (stock-stock) edges, and after attention-pooling stocks into
    sector embeddings, an *outer* ``GATConv`` over inter-sector edges; the
    inner-graph embedding, pooled sector embedding, and raw stock embedding
    are concatenated and fused by a linear+ReLU before parallel regression
    (expected return) and classification (profitable/not) heads. Reproduced
    here with the same two-level (intra-sector GAT + inter-sector GAT) graph
    attention hierarchy and dual regression/classification output.
  - HATS (Hierarchical Graph Attention Network for Stock Movement Prediction):
    Kim, Kim, Lee, Yoo, "HATS: A Hierarchical Graph Attention Network for
    Stock Movement Prediction", arXiv:1908.07999. Official repo
    ``dmis-lab/hats``, ``node_classification/src/models/HATS.py`` (TensorFlow
    1.x; reimplemented faithfully in torch since the source framework is not
    torch). Distinctive mechanism: each company's price/feature time series is
    encoded by an LSTM into a state vector; for *each relation type*
    (industry, supply-chain, ownership, ...) a relation-specific feature-level
    attention gathers each node's neighbors under that relation, scores each
    neighbor by an MLP over [neighbor state, self state, relation embedding],
    softmax-normalizes over neighbors, and produces a relation-specific
    weighted-sum representation; a second, relation-level attention then
    scores each relation's representation against its relation embedding,
    softmax-normalizes over relations, and averages the relation reps into one
    graph-summary vector that is added back onto the self LSTM state before a
    final classifier. Reproduced here as a compact per-relation
    neighbor-attention + relation-level-attention stack over a small toy
    multi-relation adjacency.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv, GatedGraphConv


# ---------------------------------------------------------------------------
# DeepGate: level-synchronous bidirectional recurrent GNN over an AIG
# ---------------------------------------------------------------------------


class _LevelGATGRU(nn.Module):
    """One directional (forward or reverse) level-synchronous GAT+GRU sweep."""

    def __init__(self, dim_hidden: int) -> None:
        super().__init__()
        self.gat = GATConv(dim_hidden, dim_hidden, add_self_loops=False)
        self.gru = nn.GRUCell(dim_hidden, dim_hidden)

    def forward(self, h: Tensor, edge_index: Tensor, levels: list[Tensor]) -> Tensor:
        """Sweep node states level-by-level, aggregating with GAT then GRU.

        Parameters
        ----------
        h : Tensor
            Node hidden states, shape ``(num_nodes, dim_hidden)``.
        edge_index : Tensor
            Directed edges (fan-in direction for the forward sweep, or the
            flipped edge list for the reverse sweep), shape ``(2, num_edges)``.
        levels : list[Tensor]
            Node-index tensors, one per topological level (level 0 = sources).

        Returns
        -------
        Tensor
            Updated node hidden states, same shape as ``h``.
        """

        for level_idx in range(1, len(levels)):
            msg = self.gat(h, edge_index)
            l_node = levels[level_idx]
            updated = self.gru(msg.index_select(0, l_node), h.index_select(0, l_node))
            h = h.index_copy(0, l_node, updated)
        return h


class DeepGate(nn.Module):
    """DeepGate: bidirectional level-synchronous recurrent GNN for AIGs."""

    def __init__(self, dim_hidden: int = 16, num_rounds: int = 2) -> None:
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_rounds = num_rounds
        self.node_type_embed = nn.Embedding(3, dim_hidden)  # PI / AND / NOT
        self.forward_sweep = _LevelGATGRU(dim_hidden)
        self.backward_sweep = _LevelGATGRU(dim_hidden)
        self.predictor = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, 1)
        )

    def forward(
        self,
        node_type: Tensor,
        edge_index: Tensor,
        levels: list[Tensor],
    ) -> Tensor:
        """Embed AIG gates and predict a per-gate probability (e.g. of "1").

        Parameters
        ----------
        node_type : Tensor
            Per-node gate-type id (0=PI, 1=AND, 2=NOT), shape ``(num_nodes,)``.
        edge_index : Tensor
            Forward (fan-in -> fan-out) directed edges, shape ``(2, num_edges)``.
        levels : list[Tensor]
            Node-index tensors giving the topological levels of the DAG.

        Returns
        -------
        Tensor
            Per-node predicted logit, shape ``(num_nodes, 1)``.
        """

        h = self.node_type_embed(node_type)
        rev_edge_index = edge_index.flip(0)
        for _ in range(self.num_rounds):
            h = self.forward_sweep(h, edge_index, levels)
            h = self.backward_sweep(h, rev_edge_index, list(reversed(levels)))
        return self.predictor(h)


def build_deepgate() -> nn.Module:
    """Build a compact :class:`DeepGate` instance.

    Returns
    -------
    nn.Module
        A ``DeepGate`` model in ``eval()`` mode.
    """

    return DeepGate(dim_hidden=16, num_rounds=2).eval()


def example_input_deepgate() -> tuple[Tensor, Tensor, list[Tensor]]:
    """Create example input for :func:`build_deepgate`.

    Returns
    -------
    tuple[Tensor, Tensor, list[Tensor]]
        Node types ``(9,)``, forward edge index ``(2, 10)``, and a 4-level
        topological-order node-index list for a small 2-input-AND-chain AIG.
    """

    torch.manual_seed(0)
    # levels: 0={0,1,2,3} PIs, 1={4,5} ANDs, 2={6} AND, 3={7,8} NOT+AND
    node_type = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 6, 7, 4],
            [4, 4, 5, 5, 6, 6, 7, 8, 8, 8],
        ],
        dtype=torch.long,
    )
    levels = [
        torch.tensor([0, 1, 2, 3], dtype=torch.long),
        torch.tensor([4, 5], dtype=torch.long),
        torch.tensor([6], dtype=torch.long),
        torch.tensor([7, 8], dtype=torch.long),
    ]
    return node_type, edge_index, levels


# ---------------------------------------------------------------------------
# DeepPlace: CNN(canvas) + GCN(netlist) fused actor-critic for chip placement
# ---------------------------------------------------------------------------


class DeepPlace(nn.Module):
    """DeepPlace: CNN canvas encoder + GCN netlist encoder, fused actor-critic."""

    def __init__(
        self,
        canvas_channels: int = 3,
        node_feat_dim: int = 4,
        hidden_dim: int = 32,
        num_actions: int = 8,
    ) -> None:
        super().__init__()
        self.canvas_cnn = nn.Sequential(
            nn.Conv2d(canvas_channels, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.gcn1 = GCNConv(node_feat_dim, 16)
        self.gcn2 = GCNConv(16, 32)
        self.gcn3 = GCNConv(32, 16)
        self.actor_head = nn.Linear(16 + 16, num_actions)
        self.critic_head = nn.Linear(16 + 16, 1)

    def forward(
        self, canvas: Tensor, node_feats: Tensor, edge_index: Tensor, target_node: int
    ) -> tuple[Tensor, Tensor]:
        """Fuse the canvas and netlist encoders and produce policy + value.

        Parameters
        ----------
        canvas : Tensor
            Rasterized placement canvas, shape ``(1, canvas_channels, H, W)``.
        node_feats : Tensor
            Netlist node features, shape ``(num_nodes, node_feat_dim)``.
        edge_index : Tensor
            Netlist connectivity, shape ``(2, num_edges)``.
        target_node : int
            Index of the node currently being placed.

        Returns
        -------
        tuple[Tensor, Tensor]
            Action logits ``(1, num_actions)`` and scalar state value
            ``(1, 1)``.
        """

        canvas_feat = self.canvas_cnn(canvas)
        x = F.relu(self.gcn1(node_feats, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))
        node_feat = x[target_node].unsqueeze(0)
        fused = torch.cat([canvas_feat, node_feat], dim=1)
        return self.actor_head(fused), self.critic_head(fused)


def build_deepplace() -> nn.Module:
    """Build a compact :class:`DeepPlace` instance.

    Returns
    -------
    nn.Module
        A ``DeepPlace`` model in ``eval()`` mode.
    """

    return DeepPlace(canvas_channels=3, node_feat_dim=4, hidden_dim=32, num_actions=8).eval()


def example_input_deepplace() -> tuple[Tensor, Tensor, Tensor, int]:
    """Create example input for :func:`build_deepplace`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, int]
        Canvas ``(1, 3, 32, 32)``, node features ``(10, 4)``, edge index
        ``(2, 14)``, and target node index.
    """

    torch.manual_seed(0)
    canvas = torch.randn(1, 3, 32, 32)
    node_feats = torch.randn(10, 4)
    src = torch.randint(0, 10, (14,))
    dst = torch.randint(0, 10, (14,))
    edge_index = torch.stack([src, dst], dim=0)
    return canvas, node_feats, edge_index, 3


# ---------------------------------------------------------------------------
# Devign: GatedGraphConv over a code property graph + gated dual-conv head
# ---------------------------------------------------------------------------


class _DevignConvHead(nn.Module):
    """Two-branch Conv1d+MaxPool classification head, branches multiplied."""

    def __init__(self, in_channels: int, ggnn_dim: int, node_feat_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.mp1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.mp2 = nn.MaxPool1d(kernel_size=2, stride=2)
        concat_len = ggnn_dim + node_feat_dim
        self.fc_concat = nn.Linear(concat_len // 4, 1)
        self.fc_hidden = nn.Linear(ggnn_dim // 4, 1)
        self.drop = nn.Dropout(p=0.0)  # deterministic for tracing

    def forward(self, hidden: Tensor, x: Tensor) -> Tensor:
        """Fuse the GGNN hidden state with raw features through dual conv branches.

        Parameters
        ----------
        hidden : Tensor
            GatedGraphConv output, shape ``(num_nodes, ggnn_dim)``.
        x : Tensor
            Original per-node features, shape ``(num_nodes, node_feat_dim)``.

        Returns
        -------
        Tensor
            Per-node vulnerability probability, shape ``(num_nodes,)``.
        """

        concat = torch.cat([hidden, x], dim=1).unsqueeze(1)
        z = self.mp1(F.relu(self.conv1(concat)))
        z = self.mp2(self.conv2(z))
        z = z.flatten(1)

        y = hidden.unsqueeze(1)
        y = self.mp1(F.relu(self.conv1(y)))
        y = self.mp2(self.conv2(y))
        y = y.flatten(1)

        res = self.fc_concat(z) * self.fc_hidden(y)
        res = self.drop(res)
        return torch.sigmoid(res.flatten())


class Devign(nn.Module):
    """Devign: GatedGraphConv code-property-graph encoder + gated dual-conv head."""

    def __init__(self, node_feat_dim: int = 12, ggnn_dim: int = 12, num_steps: int = 4) -> None:
        super().__init__()
        self.ggc = GatedGraphConv(out_channels=ggnn_dim, num_layers=num_steps)
        self.head = _DevignConvHead(in_channels=1, ggnn_dim=ggnn_dim, node_feat_dim=node_feat_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Classify each node's vulnerability probability.

        Parameters
        ----------
        x : Tensor
            Per-node code-property-graph features, shape ``(num_nodes, node_feat_dim)``.
        edge_index : Tensor
            Graph connectivity, shape ``(2, num_edges)``.

        Returns
        -------
        Tensor
            Per-node vulnerability probability, shape ``(num_nodes,)``.
        """

        hidden = self.ggc(x, edge_index)
        return self.head(hidden, x)


def build_devign() -> nn.Module:
    """Build a compact :class:`Devign` instance.

    Returns
    -------
    nn.Module
        A ``Devign`` model in ``eval()`` mode.
    """

    return Devign(node_feat_dim=12, ggnn_dim=12, num_steps=4).eval()


def example_input_devign() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_devign`.

    Returns
    -------
    tuple[Tensor, Tensor]
        Node features ``(16, 12)`` and edge index ``(2, 24)``.
    """

    torch.manual_seed(0)
    x = torch.randn(16, 12)
    src = torch.randint(0, 16, (24,))
    dst = torch.randint(0, 16, (24,))
    edge_index = torch.stack([src, dst], dim=0)
    return x, edge_index


# ---------------------------------------------------------------------------
# FinGAT: hierarchical temporal encoder + intra-/inter-sector GAT
# ---------------------------------------------------------------------------


class _SequenceEncoder(nn.Module):
    """Per-stock GRU + temporal attention pooling over a price/feature window."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, seq: Tensor) -> Tensor:
        """Encode a weekly sequence into one attention-pooled vector.

        Parameters
        ----------
        seq : Tensor
            Per-stock feature sequence, shape ``(num_stocks, time_step, input_dim)``.

        Returns
        -------
        Tensor
            Pooled stock embedding, shape ``(num_stocks, hidden_dim)``.
        """

        out, _ = self.gru(seq)
        scores = F.softmax(self.attn(out), dim=1)
        return (scores * out).sum(dim=1)


class FinGAT(nn.Module):
    """FinGAT: hierarchical GRU+attention sequence encoder + 2-level GAT fusion."""

    def __init__(
        self,
        input_dim: int = 6,
        time_step: int = 5,
        hidden_dim: int = 16,
        num_sectors: int = 3,
    ) -> None:
        super().__init__()
        self.num_sectors = num_sectors
        self.seq_encoder = _SequenceEncoder(input_dim, hidden_dim)
        self.inner_gat = GATConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.sector_pool_attn = nn.Linear(hidden_dim, 1)
        self.outer_gat = GATConv(hidden_dim, hidden_dim, add_self_loops=False)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.reg_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        seq: Tensor,
        inner_edge_index: Tensor,
        outer_edge_index: Tensor,
        sector_of_stock: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Predict expected return and profitability probability per stock.

        Parameters
        ----------
        seq : Tensor
            Per-stock weekly feature sequences, shape
            ``(num_stocks, time_step, input_dim)``.
        inner_edge_index : Tensor
            Intra-sector (stock-stock) edges, shape ``(2, num_inner_edges)``.
        outer_edge_index : Tensor
            Inter-sector (sector-sector) edges, shape ``(2, num_outer_edges)``.
        sector_of_stock : Tensor
            Sector id for each stock, shape ``(num_stocks,)``, values in
            ``[0, num_sectors)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Regression output ``(num_stocks,)`` and classification probability
            ``(num_stocks,)``.
        """

        stock_emb = self.seq_encoder(seq)
        inner_graph_emb = self.inner_gat(stock_emb, inner_edge_index)

        sector_emb = torch.zeros(self.num_sectors, stock_emb.shape[1], dtype=stock_emb.dtype)
        for sector_id in range(self.num_sectors):
            mask = sector_of_stock == sector_id
            members = stock_emb[mask]
            scores = F.softmax(self.sector_pool_attn(members), dim=0)
            sector_emb[sector_id] = (scores * members).sum(dim=0)

        sector_emb = self.outer_gat(sector_emb, outer_edge_index)
        sector_emb_per_stock = sector_emb[sector_of_stock]

        fusion_vec = torch.cat([stock_emb, sector_emb_per_stock, inner_graph_emb], dim=-1)
        fusion_vec = F.relu(self.fusion(fusion_vec))

        reg_out = self.reg_head(fusion_vec).flatten()
        cls_out = torch.sigmoid(self.cls_head(fusion_vec)).flatten()
        return reg_out, cls_out


def build_fingat() -> nn.Module:
    """Build a compact :class:`FinGAT` instance.

    Returns
    -------
    nn.Module
        A ``FinGAT`` model in ``eval()`` mode.
    """

    return FinGAT(input_dim=6, time_step=5, hidden_dim=16, num_sectors=3).eval()


def example_input_fingat() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_fingat`.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Weekly sequence ``(9, 5, 6)``, inner edges ``(2, 12)``, outer edges
        ``(2, 4)``, and per-stock sector ids ``(9,)`` across 3 sectors of 3
        stocks each.
    """

    torch.manual_seed(0)
    seq = torch.randn(9, 5, 6)
    src = torch.randint(0, 9, (12,))
    dst = torch.randint(0, 9, (12,))
    inner_edge_index = torch.stack([src, dst], dim=0)
    outer_edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    sector_of_stock = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    return seq, inner_edge_index, outer_edge_index, sector_of_stock


# ---------------------------------------------------------------------------
# HATS: LSTM state + per-relation feature attention + relation-level attention
# ---------------------------------------------------------------------------


class HATS(nn.Module):
    """HATS: LSTM company encoder + per-relation and relation-level attention."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 16,
        num_relations: int = 3,
        num_labels: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.relation_embed = nn.Embedding(num_relations, hidden_dim)
        self.node_attn = nn.Linear(hidden_dim * 3, 1)
        self.rel_attn = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, seq: Tensor, rel_adj: Tensor) -> Tensor:
        """Predict a per-company movement class using multi-relation attention.

        Parameters
        ----------
        seq : Tensor
            Per-company lookback feature sequence, shape
            ``(num_companies, lookback, input_dim)``.
        rel_adj : Tensor
            Dense relation adjacency, shape
            ``(num_relations, num_companies, num_companies)``, 1.0 where an
            edge of that relation type exists (including possible self-loops
            being masked out by the caller).

        Returns
        -------
        Tensor
            Per-company class logits, shape ``(num_companies, num_labels)``.
        """

        _, (h_n, _) = self.lstm(seq)
        state = h_n.squeeze(0)  # (num_companies, hidden_dim)
        num_companies = state.shape[0]

        rel_reps = []
        for rel_idx in range(self.num_relations):
            rel_emb = self.relation_embed.weight[rel_idx].unsqueeze(0).unsqueeze(0)
            rel_emb = rel_emb.expand(num_companies, num_companies, -1)
            neighbors = state.unsqueeze(0).expand(num_companies, num_companies, -1)
            self_exp = state.unsqueeze(1).expand(num_companies, num_companies, -1)
            att_x = torch.cat([neighbors, self_exp, rel_emb], dim=-1)
            scores = self.node_attn(att_x).squeeze(-1)
            mask = rel_adj[rel_idx]
            scores = scores.masked_fill(mask == 0, float("-inf"))
            weights = F.softmax(scores, dim=1)
            weights = torch.nan_to_num(weights, nan=0.0)
            rel_rep = torch.bmm(weights.unsqueeze(1), neighbors).squeeze(1)
            rel_reps.append(rel_rep)

        all_rel_rep = torch.stack(rel_reps, dim=0)  # (num_relations, num_companies, hidden)
        rel_emb_flat = self.relation_embed.weight.unsqueeze(1).expand(-1, num_companies, -1)
        rel_att_x = torch.cat([all_rel_rep, rel_emb_flat], dim=-1)
        rel_scores = F.softmax(self.rel_attn(rel_att_x), dim=0)
        rel_summary = (all_rel_rep * rel_scores).sum(dim=0)

        updated_state = rel_summary + state
        return self.classifier(updated_state)


def build_hats() -> nn.Module:
    """Build a compact :class:`HATS` instance.

    Returns
    -------
    nn.Module
        A ``HATS`` model in ``eval()`` mode.
    """

    return HATS(input_dim=5, hidden_dim=16, num_relations=3, num_labels=3).eval()


def example_input_hats() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_hats`.

    Returns
    -------
    tuple[Tensor, Tensor]
        Lookback sequence ``(8, 10, 5)`` for 8 companies over 10 timesteps,
        and a dense multi-relation adjacency ``(3, 8, 8)``.
    """

    torch.manual_seed(0)
    seq = torch.randn(8, 10, 5)
    rel_adj = (torch.rand(3, 8, 8) > 0.6).float()
    eye = torch.eye(8).unsqueeze(0)
    rel_adj = rel_adj * (1 - eye)  # no self-loops
    # guarantee at least one neighbor per (relation, company) to avoid all-masked rows
    rel_adj[:, :, 0] = 1.0
    rel_adj = rel_adj * (1 - eye)
    return seq, rel_adj


MENAGERIE_ENTRIES = [
    ("DeepGate", "build_deepgate", "example_input_deepgate", "2022", "GRAPH"),
    ("DeepPlace", "build_deepplace", "example_input_deepplace", "2021", "RL"),
    ("Devign", "build_devign", "example_input_devign", "2019", "GRAPH"),
    ("FinGAT", "build_fingat", "example_input_fingat", "2021", "GRAPH"),
    ("HATS", "build_hats", "example_input_hats", "2019", "GRAPH"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")
