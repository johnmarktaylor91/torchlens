# FAITHFUL PORT of bowenliu16/rl_graph_generation @ 2f278c46a179cc43583298d24983e42fa0d536a
#   (repo: https://github.com/bowenliu16/rl_graph_generation, original framework: TensorFlow 1.x)
#
# GCPN ("Graph Convolutional Policy Network for Goal-Directed Molecular Graph
# Generation", NeurIPS 2018). The trainable architecture is the GCNPolicy
# actor-critic network (rl-baselines/baselines/ppo1/gcn_policy.py) used by
# the PPO rollout in pposgd_simple_gcn.py: a stack of relational (per-edge-
# type) GCN layers embeds the molecular graph state (adjacency tensor +
# atom-type node features), then four MLP heads read off the GCN node
# embeddings to build the factorized action distribution (stop / first-atom
# / second-atom / bond-type) plus a value head, exactly the network
# GCNPolicy.act() -> self._act() exercises every PPO rollout step.
#
# The real repo is TensorFlow 1.x (tf.get_variable/tf.variable_scope/
# tf.layers.dense, no tf.keras) and therefore cannot run in this base torch
# env. Every layer/mechanism below is transcribed FAITHFULLY from the actual
# gcn_policy.py source (GCN_batch, GCNPolicy._init, kind='small' path) into
# self-contained torch: the per-edge-type GCN aggregation (GCN_batch), the
# node-embedding stack (gcn1 -> gcn1_i... -> gcn2), and the five MLP heads
# (linear_stop1/2, linear_select1/2, logits_second1/2, logits_edge1/2,
# value1/2) with the same layer shapes, activations (ReLU on GCN + hidden
# MLP layers, tanh-free per source), aggregation ('sum' default per
# args.gcn_aggregate default), and masking-by-node-count logic. Only the
# stochastic tf.distributions.Categorical .sample() calls are replaced with
# argmax (deterministic action selection) so the network exposes a plain
# differentiable forward pass suitable for a single-input trace; the
# underlying logits computation graph (the actual trained network) is
# unchanged. The rl-baselines PPO/env scaffolding (pposgd_simple_gcn.py,
# gym_molecule env, reward shaping, MPI) is NOT part of the architecture and
# is not ported.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# GCN_batch (per-edge-type relational graph convolution), faithfully ported
# from gcn_policy.py::GCN_batch. adj: B x E x N x N (E edge/bond types),
# node_feature: B x 1 x N x Cin -> out: B x 1 x N x Cout (aggregate='sum'
# reproduces the TF default args.gcn_aggregate used by GCNPolicy kind='small').
# ---------------------------------------------------------------------------
class GCNBatch(nn.Module):
    def __init__(
        self, edge_dim, in_channels, out_channels, is_act=True, is_normalize=False, aggregate="sum"
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_act = is_act
        self.is_normalize = is_normalize
        self.aggregate = aggregate

        # W: [1, edge_dim, in_channels, out_channels] (glorot_uniform in source)
        self.weight = nn.Parameter(torch.empty(1, edge_dim, in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, adj, node_feature):
        # adj: B x E x N x N ; node_feature: B x 1 x N x Cin
        batch_size = adj.shape[0]
        node_feature_tiled = node_feature.expand(-1, self.edge_dim, -1, -1)  # B x E x N x Cin
        weight_tiled = self.weight.expand(batch_size, -1, -1, -1)  # B x E x Cin x Cout
        node_embedding = adj @ node_feature_tiled @ weight_tiled  # B x E x N x Cout

        if self.is_act:
            node_embedding = F.relu(node_embedding)

        if self.aggregate == "sum":
            node_embedding = node_embedding.sum(dim=1, keepdim=True)
        elif self.aggregate == "mean":
            node_embedding = node_embedding.mean(dim=1, keepdim=True)
        elif self.aggregate == "concat":
            node_embedding = torch.cat(node_embedding.split(1, dim=1), dim=3)
        else:
            raise ValueError("GCN aggregate error!")

        if self.is_normalize:
            node_embedding = F.normalize(node_embedding, dim=-1)

        return node_embedding


# ---------------------------------------------------------------------------
# GCNPolicy, faithfully ported from gcn_policy.py::GCNPolicy._init
# (kind='small' path: a plain GCN_emb stack, no scaffold branch, no
# has_residual/has_concat, matching the repo's default args). Builds the
# same five heads GCNPolicy.act() reads: logits_stop, logits_first,
# logits_second (own-prediction branch, the one .act() actually samples
# from), logits_edge (own-prediction branch), and vpred.
# ---------------------------------------------------------------------------
class GCNPolicy(nn.Module):
    def __init__(
        self,
        num_node_types,
        max_nodes,
        num_edge_types,
        emb_size=8,
        layer_num_g=3,
        stop_shift=-3.0,
        gcn_aggregate="sum",
    ):
        super().__init__()
        self.num_node_types = num_node_types
        self.max_nodes = max_nodes
        self.num_edge_types = num_edge_types
        self.emb_size = emb_size
        self.layer_num_g = layer_num_g
        self.stop_shift = stop_shift

        # ob_node = tf.layers.dense(ob['node'], 8, use_bias=False, name='emb')
        self.emb = nn.Linear(num_node_types, 8, bias=False)

        # gcn1 .. gcn1_{layer_num_g-2} .. gcn2 (GCN_batch stack, kind='small', no residual/concat)
        gcn_in = 8
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(
            GCNBatch(num_edge_types, gcn_in, emb_size, is_act=True, aggregate=gcn_aggregate)
        )
        for _ in range(layer_num_g - 2):
            self.gcn_layers.append(
                GCNBatch(num_edge_types, emb_size, emb_size, is_act=True, aggregate=gcn_aggregate)
            )
        # gcn2: is_act=False, is_normalize=True (bn==0 branch)
        self.gcn_layers.append(
            GCNBatch(
                num_edge_types,
                emb_size,
                emb_size,
                is_act=False,
                is_normalize=True,
                aggregate=gcn_aggregate,
            )
        )

        # ### 2 predict stop
        self.linear_stop1 = nn.Linear(emb_size, emb_size, bias=False)
        self.linear_stop2 = nn.Linear(emb_size, 2)

        # ### 3.1 select first node
        self.linear_select1 = nn.Linear(emb_size, emb_size)
        self.linear_select2 = nn.Linear(emb_size, 1)

        # ### 3.2 select second node (own-prediction MLP branch)
        self.logits_second1 = nn.Linear(2 * emb_size, emb_size)
        self.logits_second2 = nn.Linear(emb_size, 1)

        # ### 3.3 predict edge type (own-prediction MLP branch)
        self.logits_edge1 = nn.Linear(2 * emb_size, emb_size)
        self.logits_edge2 = nn.Linear(emb_size, num_edge_types)

        # value head
        self.value1 = nn.Linear(emb_size, emb_size, bias=False)
        self.value2 = nn.Linear(emb_size, 1)

    def forward(self, adj, node, node_len):
        """
        adj:  B x E x N x N  float adjacency tensor (per bond/edge type)
        node: B x 1 x N x C  float one-hot-ish atom-type node features
        node_len: B  number of valid (non-padded) nodes per graph, used to
            build the same sequence-mask GCNPolicy._init derives from
            ob['node'] (tf.sequence_mask(ob_len, ...)).
        """
        B, _, N, _ = node.shape
        device = node.device

        ob_node = self.emb(node)  # B x 1 x N x 8

        emb_node = ob_node
        for i, gcn in enumerate(self.gcn_layers):
            emb_node = gcn(adj, emb_node)
        emb_node = emb_node.squeeze(1)  # B x N x F

        # logits_mask: valid-node sequence mask (all effective nodes)
        idx = torch.arange(N, device=device).unsqueeze(0)  # 1 x N
        logits_mask = idx < node_len.unsqueeze(1)  # B x N

        # ### 2 predict stop
        emb_stop = F.relu(self.linear_stop1(emb_node))  # B x N x F
        logits_stop = emb_stop.sum(dim=1)  # B x F
        logits_stop = self.linear_stop2(logits_stop)  # B x 2
        stop_shift = torch.tensor([0.0, self.stop_shift], device=device)
        logits_stop = logits_stop + stop_shift

        # ### 3.1 select first node
        logits_first = F.relu(self.linear_select1(emb_node))  # B x N x F
        logits_first = self.linear_select2(logits_first).squeeze(-1)  # B x N
        logits_first = torch.where(
            logits_mask, logits_first, torch.full_like(logits_first, -1000.0)
        )
        ac_first = torch.argmax(logits_first, dim=-1)  # B

        first_onehot = F.one_hot(ac_first, num_classes=N).to(dtype=torch.bool)  # B x N
        emb_first = emb_node[first_onehot].unsqueeze(1)  # B x 1 x F

        # ### 3.2 select second node (own-prediction MLP branch)
        emb_cat = torch.cat([emb_first.expand(-1, N, -1), emb_node], dim=2)  # B x N x 2F
        logits_second = F.relu(self.logits_second1(emb_cat))
        logits_second = self.logits_second2(logits_second).squeeze(-1)  # B x N
        first_mask = ~first_onehot
        logits_second_mask = logits_mask & first_mask
        logits_second = torch.where(
            logits_second_mask, logits_second, torch.full_like(logits_second, -1000.0)
        )
        ac_second = torch.argmax(logits_second, dim=-1)  # B

        second_onehot = F.one_hot(ac_second, num_classes=N).to(dtype=torch.bool)
        emb_second = emb_node[second_onehot].unsqueeze(1)  # B x 1 x F

        # ### 3.3 predict edge type (own-prediction MLP branch)
        emb_cat_edge = torch.cat([emb_first, emb_second], dim=-1)  # B x 1 x 2F
        logits_edge = F.relu(self.logits_edge1(emb_cat_edge))
        logits_edge = self.logits_edge2(logits_edge).squeeze(1)  # B x num_edge_types

        # value head: vpred = dense(relu, emb_size) -> reduce_max(axis=1) -> dense(1)
        vpred = F.relu(self.value1(emb_node))  # B x N x F
        vpred = vpred.max(dim=1).values  # B x F
        vpred = self.value2(vpred)  # B x 1

        return logits_stop, logits_first, logits_second, logits_edge, vpred


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_gcpn():
    """Tiny-size real GCPN GCNPolicy (molecular-graph atom/bond-type sizes
    matching the ZINC-drug-like default gym_molecule env: ~10 atom types,
    3 bond types + implicit self-loop channel)."""
    return GCNPolicy(
        num_node_types=10,
        max_nodes=12,
        num_edge_types=3,
        emb_size=8,
        layer_num_g=3,
    )


def example_input_gcpn():
    """Tiny (adj, node, node_len) batch matching GCNPolicy.forward's expected
    per-edge-type adjacency tensor, one-hot-ish node features, and valid-node
    counts."""
    torch.manual_seed(0)
    batch, edge_types, n, atom_types = 2, 3, 12, 10
    adj = torch.rand(batch, edge_types, n, n)
    adj = (adj + adj.transpose(-1, -2)) / 2
    node = (
        F.one_hot(torch.randint(0, atom_types, (batch, n)), num_classes=atom_types)
        .unsqueeze(1)
        .float()
    )
    node_len = torch.full((batch,), n, dtype=torch.long)
    return (adj, node, node_len)


MENAGERIE_ENTRIES = [
    ("GCPN", build_gcpn, example_input_gcpn, 2018, "PORT"),
]
