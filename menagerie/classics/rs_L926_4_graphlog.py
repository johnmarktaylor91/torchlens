# SOURCE: vendored from chao1224/GraphMVP @ main
#   (repo: https://github.com/chao1224/GraphMVP; GraphLoG baseline included
#   in the GraphMVP codebase -- GraphLoG itself is Xu, Wang, Cao, Barati
#   Farimani, Zhang, Zhang, ICML 2021, "Self-Supervised Graph-Level
#   Representation Learning with Local and Global Structure"; official
#   GraphLoG code is not in a standalone repo, but GraphMVP's
#   src_classification/pretrain_GraphLoG.py reimplements the exact same
#   pretraining pipeline it cites, reusing GraphMVP's own shared
#   src_classification/models/molecule_gnn_model.py GNN encoder)
#   src_classification/pretrain_GraphLoG.py (ProjectNet + pool_func,
#   verbatim) + src_classification/models/molecule_gnn_model.py (GINConv /
#   GCNConv / GATConv / GraphSAGEConv / GNN, verbatim; only GINConv is
#   instantiated for the repo's own default `--gnn_type gin`, but all four
#   message-passing conv classes are kept since `GNN.__init__` selects
#   among them by string), copied verbatim (imports only adjusted to be
#   self-contained in this single file).
#
# GraphLoG (Xu et al., ICML 2021) is a self-supervised molecular graph
# pretraining method built on a standard 5-layer GIN message-passing
# encoder (`GNN`, from the widely-used Hu et al. 2020 pretrain-gnns
# codebase that both GraphLoG and GraphMVP share). GraphLoG's own
# contribution is a hierarchical local-global contrastive objective:
# node-level masked-node InfoNCE ("local" structure), graph-level InfoNCE
# across augmented views ("global" structure), and a hierarchical
# prototype tree (`proto_NCE_loss` / `init_proto` / `init_proto_lowest`,
# an online k-means-like EMA-updated prototype hierarchy with no learned
# parameters of its own) that graph embeddings are contrastively pulled
# toward. Only the architecture that a single forward pass exercises is
# vendored here: `GNN` (atom/chirality embedding -> stacked GIN message-
# passing layers with batchnorm -> Jumping-Knowledge node representation)
# feeding `pool_func` (mean/sum/max graph pooling) and `ProjectNet` (2-layer
# MLP projection head) -- exactly the `node_reps -> graph_reps ->
# graph_reps_proj` pipeline the repo's own `init_proto_lowest` runs. The
# InfoNCE loss functions and prototype-hierarchy bookkeeping
# (`intra_NCE_loss`, `inter_NCE_loss`, `proto_NCE_loss`, `update_proto_
# lowest`, `init_proto`, `init_proto_lowest`, `mask_nodes`) are training-
# time loss/EMA orchestration over plain tensors, not a distinct nn.Module
# forward pass, so they are not vendored; no architecture code was
# rewritten.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add

MENAGERIE_ZOO = "vendored-pytorch"

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


# ---------------------------------------------------------------------------
# Verbatim from models/molecule_gnn_model.py
# ---------------------------------------------------------------------------


class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        self.aggr = aggr
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.aggr = aggr
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        # assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        norm = self.norm(edge_index[0], x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__(node_dim=0)
        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, heads * emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        x = self.weight_linear(x)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out += self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()
        self.aggr = aggr

        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0.0, gnn_type="gin"):
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation


# ---------------------------------------------------------------------------
# Verbatim from pretrain_GraphLoG.py (ProjectNet + pool_func only)
# ---------------------------------------------------------------------------


class ProjectNet(nn.Module):
    def __init__(self, rep_dim):
        super(ProjectNet, self).__init__()
        self.rep_dim = rep_dim
        self.proj = nn.Sequential(
            nn.Linear(self.rep_dim, self.rep_dim), nn.ReLU(), nn.Linear(self.rep_dim, self.rep_dim)
        )

    def forward(self, x):
        return self.proj(x)


def pool_func(x, batch, mode="mean"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)
    else:
        raise ValueError("mode must be sum or mean or max")


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


class GraphLoGPretrain(nn.Module):
    """GraphLoG's real forward-pass pipeline: the shared GIN encoder's
    node representations -> mean graph pooling -> the projection head --
    exactly the `node_reps -> graph_reps -> graph_reps_proj` computation
    the repo's own `init_proto_lowest` runs on every batch. This wrapper
    module (not present verbatim in the repo, since the repo calls the
    three pieces inline in training-loop functions rather than composing
    them into one nn.Module) introduces no new architecture -- it only
    chains the vendored GNN and ProjectNet exactly as the repo's own
    training loop does."""

    def __init__(
        self, num_layer, emb_dim, JK="last", drop_ratio=0.0, gnn_type="gin", graph_pooling="mean"
    ):
        super().__init__()
        self.gnn = GNN(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio, gnn_type=gnn_type)
        rep_dim = (num_layer + 1) * emb_dim if JK == "concat" else emb_dim
        self.proj = ProjectNet(rep_dim)
        self.graph_pooling = graph_pooling

    def forward(self, x, edge_index, edge_attr, batch):
        node_reps = self.gnn(x, edge_index, edge_attr)
        graph_reps = pool_func(node_reps, batch, mode=self.graph_pooling)
        graph_reps_proj = self.proj(graph_reps)
        return graph_reps_proj


def build_graphlog():
    """Tiny-size real GraphLoG pretraining pipeline (5-layer GIN encoder
    default gnn_type, matching the repo's own `--num_layer 5 --emb_dim 300
    --gnn_type gin --JK last --graph_pooling mean` defaults, shrunk to
    emb_dim=16 / num_layer=2 for a fast trace)."""
    return GraphLoGPretrain(
        num_layer=2, emb_dim=16, JK="last", drop_ratio=0.0, gnn_type="gin", graph_pooling="mean"
    )


def example_input_graphlog():
    """Two tiny molecular graphs batched together, matching GNN.forward's
    (x, edge_index, edge_attr) contract: x is [N, 2] (atom type, chirality
    tag) atom features, edge_index is [2, E] bond connectivity, edge_attr
    is [E, 2] (bond type, bond direction) bond features, batch is the
    [N]-length graph-membership vector `pool_func` needs."""
    torch.manual_seed(0)
    # molecule 1: 4 atoms, a 4-cycle (8 directed edges)
    x1 = torch.stack([torch.randint(0, 118, (4,)), torch.randint(0, 3, (4,))], dim=1)
    edge_index1 = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    edge_attr1 = torch.stack(
        [torch.randint(0, 4, (edge_index1.size(1),)), torch.randint(0, 2, (edge_index1.size(1),))],
        dim=1,
    )

    # molecule 2: 3 atoms, a path (4 directed edges)
    x2 = torch.stack([torch.randint(0, 118, (3,)), torch.randint(0, 3, (3,))], dim=1)
    edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr2 = torch.stack(
        [torch.randint(0, 4, (edge_index2.size(1),)), torch.randint(0, 2, (edge_index2.size(1),))],
        dim=1,
    )

    x = torch.cat([x1, x2], dim=0)
    edge_index = torch.cat([edge_index1, edge_index2 + x1.size(0)], dim=1)
    edge_attr = torch.cat([edge_attr1, edge_attr2], dim=0)
    batch = torch.cat(
        [torch.zeros(x1.size(0), dtype=torch.long), torch.ones(x2.size(0), dtype=torch.long)]
    )

    return (x, edge_index, edge_attr, batch)


MENAGERIE_ENTRIES = [
    (
        "GraphLoG",
        build_graphlog,
        example_input_graphlog,
        2021,
        "CODE",
    ),
]
