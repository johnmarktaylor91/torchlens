# SOURCE: vendored from OSU-BMBL/deepmaps @ cd53d8bc0df7fceafebeab36ebd281b3d801186c
# (pyHGT/conv.py + pyHGT/model.py)
#
# DeepMAPS (Ma, Sun, Chen et al. 2023, "DeepMAPS: Single-cell biological network
# inference using heterogeneous graph transformer") builds a heterogeneous
# gene-cell graph and embeds it with a Heterogeneous Graph Transformer (HGT,
# Hu et al. 2020) stack: type-specific K/Q/V projections + relation-specific
# attention/message matrices + target-type-specific aggregation, wrapped in a
# multi-layer torch_geometric MessagePassing conv (`HGTConv`/`GeneralConv`)
# and driven by the `GNN` module that DeepMAPS' `hgt.py` actually instantiates
# for its "reduction != raw" (default AE-reduced-features) training path:
#   gnn = GNN(conv_name=args.layer_type, in_dim=encoded.shape[1], n_hid=...,
#             n_heads=..., n_layers=..., dropout=..., num_types=2,
#             num_relations=2, use_RTE=False)
#   node_rep = gnn.forward(node_feature, node_type, edge_time, edge_index, edge_type)
# Copied verbatim (aside from stripping unused CLI/argparse/training glue);
# every layer/mechanism (type-specific adapters, per-(source,target,relation)
# attention loop, softmax-by-target-node, skip-connection gate, LayerNorm) is
# the real DeepMAPS/pyHGT code, not a reimplementation.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax

MENAGERIE_ZOO = "vendored-pytorch"


class HGTConv(MessagePassing):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_types,
        num_relations,
        n_heads,
        dropout=0.2,
        use_norm=True,
        use_RTE=True,
        **kwargs,
    ):
        super(HGTConv, self).__init__(node_dim=0, aggr="add", **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.total_rel = num_types * num_relations * num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.use_RTE = use_RTE
        self.att = None
        self.res_att = None
        self.res = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for t in range(num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
        """
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        """
        self.relation_pri = nn.Parameter(torch.ones(num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        if self.use_RTE:
            self.emb = RelTemporalEncoding(in_dim)

        glorot(self.relation_att)
        glorot(self.relation_msg)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, node_inp, node_type, edge_index, edge_type, edge_time):
        return self.propagate(
            edge_index,
            node_inp=node_inp,
            node_type=node_type,
            edge_type=edge_type,
            edge_time=edge_time,
        )

    def message(
        self, edge_index_i, node_inp_i, node_inp_j, node_type_i, node_type_j, edge_type, edge_time
    ):
        """
        j: source, i: target; <j, i>
        """
        data_size = edge_index_i.size(0)
        """
            Create Attention and Message tensor beforehand.
        """
        self.res_att = torch.zeros(data_size, self.n_heads).to(node_inp_i.device)
        res_msg = torch.zeros(data_size, self.n_heads, self.d_k).to(node_inp_i.device)

        for source_type in range(self.num_types):
            sb = node_type_j == int(source_type)
            k_linear = self.k_linears[source_type]
            v_linear = self.v_linears[source_type]
            for target_type in range(self.num_types):
                tb = (node_type_i == int(target_type)) & sb
                q_linear = self.q_linears[target_type]
                for relation_type in range(self.num_relations):
                    """
                    idx is all the edges with meta relation <source_type, relation_type, target_type>
                    """
                    idx = (edge_type == int(relation_type)) & tb
                    if idx.sum() == 0:
                        continue
                    """
                        Get the corresponding input node representations by idx.
                        Add tempotal encoding to source representation (j)
                    """
                    target_node_vec = node_inp_i[idx]
                    source_node_vec = node_inp_j[idx]
                    if self.use_RTE:
                        source_node_vec = self.emb(source_node_vec, edge_time[idx])
                    """
                        Step 1: Heterogeneous Mutual Attention
                    """
                    q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    k_mat = torch.bmm(
                        k_mat.transpose(1, 0), self.relation_att[relation_type]
                    ).transpose(1, 0)
                    self.res_att[idx] = (
                        (q_mat * k_mat).sum(dim=-1)
                        * self.relation_pri[relation_type]
                        / self.sqrt_dk
                    )
                    """
                        Step 2: Heterogeneous Message Passing
                    """
                    v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
                    res_msg[idx] = torch.bmm(
                        v_mat.transpose(1, 0), self.relation_msg[relation_type]
                    ).transpose(1, 0)
        """
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        """
        res = res_msg * softmax(self.res_att.view(-1, self.n_heads, 1), edge_index_i)

        return res.view(-1, self.out_dim)

    def update(self, aggr_out, node_inp, node_type):
        """
        Step 3: Target-specific Aggregation
        x = W[node_type] * gelu(Agg(x)) + x
        """
        aggr_out = F.gelu(aggr_out)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(node_inp.device)
        for target_type in range(self.num_types):
            idx = node_type == int(target_type)
            if idx.sum() == 0:
                continue
            trans_out = self.drop(self.a_linears[target_type](aggr_out[idx]))
            """
                Add skip connection with learnable weight self.skip[t_id]
            """
            alpha = torch.sigmoid(self.skip[target_type])
            if self.use_norm:
                res[idx] = self.norms[target_type](trans_out * alpha + node_inp[idx] * (1 - alpha))
            else:
                res[idx] = trans_out * alpha + node_inp[idx] * (1 - alpha)
        self.res = res
        return res

    def __repr__(self):
        return "{}(in_dim={}, out_dim={}, num_types={}, num_types={})".format(
            self.__class__.__name__, self.in_dim, self.out_dim, self.num_types, self.num_relations
        )


class RelTemporalEncoding(nn.Module):
    """
    Implement the Temporal Encoding (Sinusoid) function.
    """

    def __init__(self, n_hid, max_len=240, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_hid, 2) * -(math.log(10000.0) / n_hid))
        emb = nn.Embedding(max_len, n_hid)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term) / math.sqrt(n_hid)
        emb.weight.data[:, 1::2] = torch.cos(position * div_term) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.emb(t))


class GeneralConv(nn.Module):
    def __init__(
        self,
        conv_name,
        in_hid,
        out_hid,
        num_types,
        num_relations,
        n_heads,
        dropout,
        use_norm=True,
        use_RTE=True,
    ):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        self.res_att = None
        self.res = None
        if self.conv_name == "hgt":
            self.base_conv = HGTConv(
                in_hid, out_hid, num_types, num_relations, n_heads, dropout, use_norm, use_RTE
            )
        elif self.conv_name == "gcn":
            self.base_conv = GCNConv(in_hid, out_hid)
        elif self.conv_name == "gat":
            self.base_conv = GATConv(in_hid, out_hid // n_heads, heads=n_heads)

    def forward(self, meta_xs, node_type, edge_index, edge_type, edge_time):
        if self.conv_name == "hgt":
            a = self.base_conv(meta_xs, node_type, edge_index, edge_type, edge_time)
            self.res_att = self.base_conv.res_att
            self.res = self.base_conv.res
            return a
        elif self.conv_name == "gcn":
            return self.base_conv(meta_xs, edge_index)
        elif self.conv_name == "gat":
            return self.base_conv(meta_xs, edge_index)


class GNN(nn.Module):
    def __init__(
        self,
        in_dim,
        n_hid,
        num_types,
        num_relations,
        n_heads,
        n_layers,
        dropout=0.2,
        conv_name="hgt",
        prev_norm=True,
        last_norm=True,
        use_RTE=True,
    ):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        self.att = None
        self.emb = None
        self.conv_name = conv_name
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for _l in range(n_layers - 1):
            self.gcs.append(
                GeneralConv(
                    conv_name,
                    n_hid,
                    n_hid,
                    num_types,
                    num_relations,
                    n_heads,
                    dropout,
                    use_norm=prev_norm,
                    use_RTE=use_RTE,
                )
            )
        self.gcs.append(
            GeneralConv(
                conv_name,
                n_hid,
                n_hid,
                num_types,
                num_relations,
                n_heads,
                dropout,
                use_norm=last_norm,
                use_RTE=use_RTE,
            )
        )

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type):
        res = torch.zeros(node_feature.size(0), self.n_hid).to(node_feature.device)
        for t_id in range(self.num_types):
            idx = node_type == int(t_id)
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        self.att = {}
        i = 0
        self.emb = {}
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
            if self.conv_name == "hgt":
                self.att[i] = gc.res_att
                i = i + 1
        self.att = self.att[0]
        return meta_xs


# ---------------------------------------------------------------------------
# menagerie staging entry point
# ---------------------------------------------------------------------------
# Real usage in DeepMAPS' hgt.py (the "reduction != raw" / default AE-feature
# path):
#   gnn = GNN(conv_name=args.layer_type, in_dim=encoded.shape[1], n_hid=args.n_hid,
#             n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
#             num_types=2, num_relations=2, use_RTE=False)
#   node_rep = gnn.forward(node_feature, node_type, edge_time, edge_index, edge_type)
# A tiny heterogeneous gene-cell graph (2 node types: gene=0, cell=1; 2 relation
# types) is built below to exercise the real forward path at trace speed.

IN_DIM = 16
N_HID = 8
N_HEADS = 2
N_LAYERS = 2
N_GENE = 3
N_CELL = 3


def build_deepmaps_hgt():
    return GNN(
        conv_name="hgt",
        in_dim=IN_DIM,
        n_hid=N_HID,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=0.0,
        num_types=2,
        num_relations=2,
        use_RTE=False,
    )


def example_input_deepmaps_hgt():
    n_nodes = N_GENE + N_CELL
    node_feature = torch.randn(n_nodes, IN_DIM)
    # node types: first N_GENE nodes are type 0 (gene), rest are type 1 (cell)
    node_type = torch.cat(
        [torch.zeros(N_GENE, dtype=torch.long), torch.ones(N_CELL, dtype=torch.long)]
    )
    # bipartite gene<->cell edges (both directions), 2 relation types
    gene_idx = torch.arange(N_GENE)
    cell_idx = torch.arange(N_GENE, N_GENE + N_CELL)
    src = torch.cat([gene_idx.repeat_interleave(N_CELL), cell_idx.repeat(N_GENE)])
    dst = torch.cat([cell_idx.repeat(N_GENE), gene_idx.repeat_interleave(N_CELL)])
    edge_index = torch.stack([src, dst], dim=0)
    n_edges = edge_index.size(1)
    edge_type = torch.randint(0, 2, (n_edges,), dtype=torch.long)
    edge_time = torch.zeros(n_edges, dtype=torch.long)
    return (node_feature, node_type, edge_time, edge_index, edge_type)


MENAGERIE_ENTRIES = [
    ("DeepMAPS-HGT", build_deepmaps_hgt, example_input_deepmaps_hgt, 2023, "SOURCE_AVAILABLE"),
]
