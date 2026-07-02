# FAITHFUL PORT of junhongmit/FraudGT @ main (original framework: PyTorch + custom graphgym-style config registry)
"""FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial
Fraud Detection.

Faithfully ported from junhongmit/FraudGT:
  - fraudGT/layer/gt_layer.py            (GTLayer, 'SparseNodeTransformer' branch)
  - fraudGT/network/gt_model.py           (GTModel: encoder -> GTLayer stack -> head)
  - fraudGT/network/utils.py              (GTPreNN: linear pre-projection)
  - fraudGT/graphgym/models/layer.py      (MLP head)

The reference model cannot be constructed directly in the base env: every
class is wired through FraudGT's private ``graphgym``-style global config
object (``cfg.gt.*``, populated only after ``fraudGT.graphgym.config`` runs
its full CLI/yaml init) and a decorator-based registry
(``register.head_dict``, ``register.node_encoder_dict``, ...) that requires
importing the whole ``fraudGT`` package (not on PyPI, not a base lib) and
constructing a real ``torch_geometric`` ``dataset`` object up front just to
read ``dataset.metadata()``. None of that infrastructure is architecture --
it is generic experiment plumbing. This port keeps every real architectural
op from the paper's namesake mechanism (the sparse node-level multi-head
attention: per-node-type Q/K/V/output projections, per-edge-type key/value
edge re-weighting via ``edge_weights``/``msg_weights``, scaled dot-product
edge scores with segment-softmax via ``scatter_max``/``scatter_add_``) and
transcribes it for a single homogeneous node type / single edge type
configuration (dropping FraudGT's heterogeneous-metadata dictionary-of-dicts
bookkeeping, which is data-shape plumbing, not a distinct computation), wired
into the same encoder -> pre-GT linear -> GTLayer -> MLP head pipeline as the
real ``GTModel.forward``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

MENAGERIE_ZOO = "ported-pytorch"


class GTPreNN(nn.Module):
    """Port of fraudGT/network/utils.py::GTPreNN (a GeneralMultiLayer('linear', ...)
    wrapper): linear projection into the transformer's hidden width, followed
    by an activation (final_act=True in the reference call).
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.linear(x))


class SparseNodeTransformerLayer(nn.Module):
    """Port of fraudGT/layer/gt_layer.py::GTLayer, global_model_type ==
    'SparseNodeTransformer' branch, specialized to ONE node type and ONE edge
    type (self-loops on that node type), matching the reference math:

      q,k,v = per-node-type linear projections of h
      edge_attr, edge_gate = per-edge-type linear projections of edge features
      edge_k = edge_weights @ edge_k   (per-edge-type learned key re-weighting)
      edge_scores = sum(edge_q * edge_k [+ edge_attr], dim=-1) / sqrt(D), clamped to [-5, 5]
      edge_v = edge_v * sigmoid(edge_gate)                     (if edge features present)
      softmax via scatter_max/scatter_add_ over destination nodes (segment softmax)
      out = scatter_add_(dst_nodes, attn * edge_v)             (weighted aggregation)
      out = o_lin(out)                                          (per-node-type output proj)

    followed by the reference's residual-add, pre/post normalization, and the
    feed-forward block (``_ff_block``), matching GTLayer.forward exactly for
    the 'Fixed' residual + 'Single' ffn configuration.
    """

    def __init__(self, dim_h, num_heads=4, layer_norm=False, batch_norm=True, edge_weight=True):
        super().__init__()
        assert dim_h % num_heads == 0
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.head_dim = dim_h // num_heads
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.q_lin = nn.Linear(dim_h, dim_h)
        self.k_lin = nn.Linear(dim_h, dim_h)
        self.v_lin = nn.Linear(dim_h, dim_h)
        self.o_lin = nn.Linear(dim_h, dim_h)
        self.e_lin = nn.Linear(dim_h, dim_h)
        self.g_lin = nn.Linear(dim_h, dim_h)

        H, D = num_heads, self.head_dim
        self.edge_weight = edge_weight
        if edge_weight:
            self.edge_weights = nn.Parameter(torch.empty(H, D, D))
            self.msg_weights = nn.Parameter(torch.empty(H, D, D))
            nn.init.xavier_uniform_(self.edge_weights)
            nn.init.xavier_uniform_(self.msg_weights)

        if layer_norm:
            self.norm1_global = nn.LayerNorm(dim_h)
            self.norm2_ffn = nn.LayerNorm(dim_h)
        elif batch_norm:
            self.norm1_global = nn.BatchNorm1d(dim_h)
            self.norm2_ffn = nn.BatchNorm1d(dim_h)

        self.dropout_global = nn.Dropout(0.0)
        self.dropout_attn = nn.Dropout(0.0)

        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.ff_dropout1 = nn.Dropout(0.0)
        self.ff_dropout2 = nn.Dropout(0.0)

    def _ff_block(self, x):
        x = self.ff_dropout1(F.relu(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, h, edge_index, edge_attr):
        """
        h: (L, dim_h) node features for the single (homogeneous) node type
        edge_index: (2, E) src/dst node indices
        edge_attr: (E, dim_h) edge features
        """
        h_in = h
        edge_attr_in = edge_attr

        if self.layer_norm or self.batch_norm:
            h_n = self.norm1_global(h)
        else:
            h_n = h

        H, D = self.num_heads, self.head_dim
        L = h_n.shape[0]
        q = self.q_lin(h_n).view(L, H, D)
        k = self.k_lin(h_n).view(L, H, D)
        v = self.v_lin(h_n).view(L, H, D)
        edge_attr_proj = self.e_lin(edge_attr).view(-1, H, D)
        edge_gate = self.g_lin(edge_attr).view(-1, H, D)

        # transpose to (H, L, D) / (H, E, D), matching the reference's transpose(0, 1)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        edge_attr_proj = edge_attr_proj.transpose(0, 1)
        edge_gate = edge_gate.transpose(0, 1)

        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        num_edges = edge_index.shape[1]

        edge_q = q[:, dst_nodes, :]
        edge_k = k[:, src_nodes, :]
        edge_v = v[:, src_nodes, :]

        if self.edge_weight:
            # (H, E, D, D) via broadcasting the per-head weight matrix over edges
            edge_weight = self.edge_weights.unsqueeze(1).expand(H, num_edges, D, D)
            edge_k = edge_k.unsqueeze(-1)  # (H, E, D, 1)
            edge_k = torch.matmul(edge_weight, edge_k).squeeze(-1)  # (H, E, D)

        edge_scores = edge_q * edge_k
        edge_scores = edge_scores + edge_attr_proj
        edge_v = edge_v * torch.sigmoid(edge_gate)

        edge_scores = torch.sum(edge_scores, dim=-1) / math.sqrt(D)  # (H, E)
        edge_scores = torch.clamp(edge_scores, min=-5, max=5)

        expanded_dst_nodes = dst_nodes.unsqueeze(0).expand(H, num_edges)
        max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L)
        max_scores = max_scores.gather(1, expanded_dst_nodes)
        exp_scores = torch.exp(edge_scores - max_scores)
        sum_exp_scores = torch.zeros((H, L), device=edge_scores.device)
        sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)
        attn = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
        attn = self.dropout_attn(attn).unsqueeze(-1)  # (H, E, 1)

        out = torch.zeros((H, L, D), device=q.device)
        out.scatter_add_(1, dst_nodes.view(1, num_edges, 1).expand(H, num_edges, D), attn * edge_v)
        out = out.transpose(0, 1).contiguous().view(L, H * D)
        h_attn = self.o_lin(out)

        edge_attr_out = edge_attr_proj.transpose(0, 1).contiguous().view(num_edges, H * D)
        edge_attr_out = self.g_lin(edge_attr_out)  # reference's oe_lin analogue

        h_attn = self.dropout_global(h_attn)

        # residual='Fixed'
        h_attn = h_attn + h_in
        edge_attr_out = edge_attr_out + edge_attr_in

        # feed-forward block, ffn='Single'
        if self.layer_norm or self.batch_norm:
            h_ffn_in = self.norm2_ffn(h_attn)
        else:
            h_ffn_in = h_attn
        h_out = h_attn + self._ff_block(h_ffn_in)

        return h_out, edge_attr_out


class MLPHead(nn.Module):
    """Port of fraudGT/graphgym/models/layer.py::MLP (num_layers=2 path)."""

    def __init__(self, dim_in, dim_out, dim_inner=None):
        super().__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_inner, bias=True),
            nn.ReLU(),
            nn.Linear(dim_inner, dim_out, bias=True),
        )

    def forward(self, x):
        return self.model(x)


class FraudGTModel(nn.Module):
    """Port of fraudGT/network/gt_model.py::GTModel, specialized to a single
    homogeneous node/edge type (the reference's non-hetero fallback path:
    ``self.metadata = [("node_type",), (("node_type", "edge_type", "node_type"),)]``),
    with ``layers_pre_gt=1``, ``layers=2`` GT layers, ``jumping_knowledge=False``,
    ``residual='Fixed'``, matching FraudGTModel.forward's control flow.
    """

    def __init__(self, dim_in, dim_edge_in, dim_hidden, dim_out, num_layers=2, num_heads=4):
        super().__init__()
        self.node_encoder = nn.Linear(dim_in, dim_hidden)
        self.edge_encoder = nn.Linear(dim_edge_in, dim_hidden)
        self.pre_gt = GTPreNN(dim_hidden, dim_hidden)
        self.convs = nn.ModuleList(
            [SparseNodeTransformerLayer(dim_hidden, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.post_gt = MLPHead(dim_hidden, dim_out)

    def forward(self, x, edge_index, edge_attr):
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        h = self.pre_gt(h)
        for conv in self.convs:
            h, e = conv(h, edge_index, e)
        return self.post_gt(h)


def build_fraudgt():
    return FraudGTModel(
        dim_in=16, dim_edge_in=8, dim_hidden=32, dim_out=2, num_layers=2, num_heads=4
    )


def example_input_fraudgt():
    torch.manual_seed(0)
    num_nodes = 20
    num_edges = 60
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 8)
    return (x, edge_index, edge_attr)


MENAGERIE_ENTRIES = [
    ("FraudGT", "build_fraudgt", "example_input_fraudgt", 2024, "SOURCE_AVAILABLE"),
]
