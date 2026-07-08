# SOURCE: vendored from harryjo97/GDSS @ 24cc490e0c5b39cbd265fda33ab2bc8b528d4454
#   (repo: https://github.com/harryjo97/GDSS)
#
# GDSS ("Score-Based Generative Modeling of Graphs via the System of
# Stochastic Differential Equations", ICML 2022). The trainable architecture
# is the pair of score networks that jointly denoise node features (X) and
# the adjacency matrix (A) under the coupled graph SDE: ScoreNetworkX (a
# stack of DenseGCNConv layers) and ScoreNetworkA (a stack of graph
# multi-head-Attention layers, AttentionLayer/Attention from Baek et al.
# 2021, each internally wrapping DenseGCNConv as its Q/K/V projections).
# Both networks are pure PyTorch (no custom ops) and share the small
# DenseGCNConv/MLP layer library plus mask_x/mask_adjs/pow_tensor tensor
# utilities.
#
# Code below is copied verbatim from models/layers.py, models/attention.py,
# models/ScoreNetwork_A.py, models/ScoreNetwork_X.py, and the three tensor
# helpers (mask_x, mask_adjs, pow_tensor) from utils/graph_utils.py that
# those model files import. Only import paths were collapsed into this
# single file (module-level `from models.layers import ...` /
# `from utils.graph_utils import ...` replaced by same-file references); no
# architecture code was rewritten, approximated, or simplified. The
# data-prep-only graph_utils helpers that need `networkx.from_numpy_matrix`
# / `to_numpy_matrix` (removed in networkx>=3.0) -- adjs_to_graphs,
# graphs_to_tensor, graphs_to_adj, init_flags -- are not part of the
# forward-pass architecture and are omitted; node_flags/mask_x/mask_adjs/
# pow_tensor (all pure-tensor, no networkx dependency) are kept.

import math

import torch
import torch.nn.functional as F
from torch.nn import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Verbatim (tensor-only subset) from utils/graph_utils.py
# ---------------------------------------------------------------------------


def mask_x(x, flags):
    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    return x * flags[:, :, None]


def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    """
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs


def node_flags(adj, eps=1e-5):
    flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)
    if len(flags.shape) == 3:
        flags = flags[:, 0, :]
    return flags


def pow_tensor(x, cnum):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum - 1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)
    return xc


def node_feature_to_matrix(x):
    """
    :param x:  BS x N x F
    :return: x_pair: BS x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(x.size(0), x.size(1), x.size(1), -1)  # BS x N x N x F
    x_pair = torch.cat([x_b, x_b.transpose(1, 2)], dim=-1)  # BS x N x N x 2F
    return x_pair


# ---------------------------------------------------------------------------
# Verbatim from models/layers.py
# ---------------------------------------------------------------------------


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


# -------- GCN layer --------
class DenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`."""

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super(DenseGCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj, mask=None, add_loop=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)


# -------- MLP layer --------
class MLP(torch.nn.Module):
    def __init__(
        self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False, activate_func=F.relu
    ):
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.activate_func = activate_func

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = torch.nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

            if self.use_bn:
                self.batch_norms = torch.nn.ModuleList()
                for layer in range(num_layers - 1):
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                if self.use_bn:
                    h = self.batch_norms[layer](h)
                h = self.activate_func(h)
            return self.linears[self.num_layers - 1](h)


# ---------------------------------------------------------------------------
# Verbatim from models/attention.py
# ---------------------------------------------------------------------------


# -------- Graph Multi-Head Attention (GMH) --------
# -------- From Baek et al. (2021) --------
class Attention(torch.nn.Module):
    def __init__(self, in_dim, attn_dim, out_dim, num_heads=4, conv="GCN"):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv

        self.gnn_q, self.gnn_k, self.gnn_v = self.get_gnn(in_dim, attn_dim, out_dim, conv)
        self.activation = torch.tanh
        self.softmax_dim = 2

    def forward(self, x, adj, flags, attention_mask=None):
        if self.conv == "GCN":
            Q = self.gnn_q(x, adj)
            K = self.gnn_k(x, adj)
        else:
            Q = self.gnn_q(x)
            K = self.gnn_k(x)

        V = self.gnn_v(x, adj)
        dim_split = self.attn_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.out_dim)
            A = self.activation(attention_mask + attention_score)
        else:
            A = self.activation(
                Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.out_dim)
            )  # (B x num_heads) x N x N

        # -------- (B x num_heads) x N x N --------
        A = A.view(-1, *adj.shape)
        A = A.mean(dim=0)
        A = (A + A.transpose(-1, -2)) / 2

        return V, A

    def get_gnn(self, in_dim, attn_dim, out_dim, conv="GCN"):
        if conv == "GCN":
            gnn_q = DenseGCNConv(in_dim, attn_dim)
            gnn_k = DenseGCNConv(in_dim, attn_dim)
            gnn_v = DenseGCNConv(in_dim, out_dim)

            return gnn_q, gnn_k, gnn_v

        elif conv == "MLP":
            num_layers = 2
            gnn_q = MLP(num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh)
            gnn_k = MLP(num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh)
            gnn_v = DenseGCNConv(in_dim, out_dim)

            return gnn_q, gnn_k, gnn_v

        else:
            raise NotImplementedError(f"{conv} not implemented.")


# -------- Layer of ScoreNetworkA --------
class AttentionLayer(torch.nn.Module):
    def __init__(
        self,
        num_linears,
        conv_input_dim,
        attn_dim,
        conv_output_dim,
        input_dim,
        output_dim,
        num_heads=4,
        conv="GCN",
    ):
        super(AttentionLayer, self).__init__()

        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn_dim = attn_dim
            self.attn.append(
                Attention(
                    conv_input_dim, self.attn_dim, conv_output_dim, num_heads=num_heads, conv=conv
                )
            )

        self.hidden_dim = 2 * max(input_dim, output_dim)
        self.mlp = MLP(
            num_linears,
            2 * input_dim,
            self.hidden_dim,
            output_dim,
            use_bn=False,
            activate_func=F.elu,
        )
        self.multi_channel = MLP(
            2,
            input_dim * conv_output_dim,
            self.hidden_dim,
            conv_output_dim,
            use_bn=False,
            activate_func=F.elu,
        )

    def forward(self, x, adj, flags):
        """
        :param x:  B x N x F_i
        :param adj: B x C_i x N x N
        :return: x_out: B x N x F_o, adj_out: B x C_o x N x N
        """
        mask_list = []
        x_list = []
        for _ in range(len(self.attn)):
            _x, mask = self.attn[_](x, adj[:, _, :, :], flags)
            mask_list.append(mask.unsqueeze(-1))
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)

        mlp_in = torch.cat([torch.cat(mask_list, dim=-1), adj.permute(0, 2, 3, 1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0, 3, 1, 2)
        _adj = _adj + _adj.transpose(-1, -2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out


# ---------------------------------------------------------------------------
# Verbatim from models/ScoreNetwork_A.py (BaselineNetworkLayer/BaselineNetwork
# kept because ScoreNetworkA subclasses BaselineNetwork)
# ---------------------------------------------------------------------------


class BaselineNetworkLayer(torch.nn.Module):
    def __init__(
        self, num_linears, conv_input_dim, conv_output_dim, input_dim, output_dim, batch_norm=False
    ):
        super(BaselineNetworkLayer, self).__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.convs.append(DenseGCNConv(conv_input_dim, conv_output_dim))
        self.hidden_dim = max(input_dim, output_dim)
        self.mlp_in_dim = input_dim + 2 * conv_output_dim
        self.mlp = MLP(
            num_linears,
            self.mlp_in_dim,
            self.hidden_dim,
            output_dim,
            use_bn=False,
            activate_func=F.elu,
        )
        self.multi_channel = MLP(
            2,
            input_dim * conv_output_dim,
            self.hidden_dim,
            conv_output_dim,
            use_bn=False,
            activate_func=F.elu,
        )

    def forward(self, x, adj, flags):
        x_list = []
        for _ in range(len(self.convs)):
            _x = self.convs[_](x, adj[:, _, :, :])
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)

        x_matrix = node_feature_to_matrix(x_out)
        mlp_in = torch.cat([x_matrix, adj.permute(0, 2, 3, 1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0, 3, 1, 2)
        _adj = _adj + _adj.transpose(-1, -2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out


class BaselineNetwork(torch.nn.Module):
    def __init__(
        self,
        max_feat_num,
        max_node_num,
        nhid,
        num_layers,
        num_linears,
        c_init,
        c_hid,
        c_final,
        adim,
        num_heads=4,
        conv="GCN",
    ):
        super(BaselineNetwork, self).__init__()

        self.nfeat = max_feat_num
        self.max_node_num = max_node_num
        self.nhid = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(
                    BaselineNetworkLayer(
                        self.num_linears, self.nfeat, self.nhid, self.c_init, self.c_hid
                    )
                )
            elif _ == self.num_layers - 1:
                self.layers.append(
                    BaselineNetworkLayer(
                        self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_final
                    )
                )
            else:
                self.layers.append(
                    BaselineNetworkLayer(
                        self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_hid
                    )
                )

        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=1,
            use_bn=False,
            activate_func=F.elu,
        )
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(
            self.max_node_num
        )
        self.mask.unsqueeze_(0)

    def forward(self, x, adj, flags=None):
        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        score = self.final(adjs).view(*out_shape)

        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score


class ScoreNetworkA(BaselineNetwork):
    def __init__(
        self,
        max_feat_num,
        max_node_num,
        nhid,
        num_layers,
        num_linears,
        c_init,
        c_hid,
        c_final,
        adim,
        num_heads=4,
        conv="GCN",
    ):
        super(ScoreNetworkA, self).__init__(
            max_feat_num,
            max_node_num,
            nhid,
            num_layers,
            num_linears,
            c_init,
            c_hid,
            c_final,
            adim,
            num_heads=4,
            conv="GCN",
        )

        self.adim = adim
        self.num_heads = num_heads
        self.conv = conv

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(
                    AttentionLayer(
                        self.num_linears,
                        self.nfeat,
                        self.nhid,
                        self.nhid,
                        self.c_init,
                        self.c_hid,
                        self.num_heads,
                        self.conv,
                    )
                )
            elif _ == self.num_layers - 1:
                self.layers.append(
                    AttentionLayer(
                        self.num_linears,
                        self.nhid,
                        self.adim,
                        self.nhid,
                        self.c_hid,
                        self.c_final,
                        self.num_heads,
                        self.conv,
                    )
                )
            else:
                self.layers.append(
                    AttentionLayer(
                        self.num_linears,
                        self.nhid,
                        self.adim,
                        self.nhid,
                        self.c_hid,
                        self.c_hid,
                        self.num_heads,
                        self.conv,
                    )
                )

        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=1,
            use_bn=False,
            activate_func=F.elu,
        )
        self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(
            self.max_node_num
        )
        self.mask.unsqueeze_(0)

    def forward(self, x, adj, flags):
        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        score = self.final(adjs).view(*out_shape)

        self.mask = self.mask.to(score.device)
        score = score * self.mask

        score = mask_adjs(score, flags)

        return score


# ---------------------------------------------------------------------------
# Verbatim from models/ScoreNetwork_X.py
# ---------------------------------------------------------------------------


class ScoreNetworkX(torch.nn.Module):
    def __init__(self, max_feat_num, depth, nhid):
        super(ScoreNetworkX, self).__init__()

        self.nfeat = max_feat_num
        self.depth = depth
        self.nhid = nhid

        self.layers = torch.nn.ModuleList()
        for _ in range(self.depth):
            if _ == 0:
                self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
            else:
                self.layers.append(DenseGCNConv(self.nhid, self.nhid))

        self.fdim = self.nfeat + self.depth * self.nhid
        self.final = MLP(
            num_layers=3,
            input_dim=self.fdim,
            hidden_dim=2 * self.fdim,
            output_dim=self.nfeat,
            use_bn=False,
            activate_func=F.elu,
        )

        self.activation = torch.tanh

    def forward(self, x, adj, flags):
        x_list = [x]
        for _ in range(self.depth):
            x = self.layers[_](x, adj)
            x = self.activation(x)
            x_list.append(x)

        xs = torch.cat(x_list, dim=-1)  # B x N x (F + num_layers x H)
        out_shape = (adj.shape[0], adj.shape[1], -1)
        x = self.final(xs).view(*out_shape)

        x = mask_x(x, flags)

        return x


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------


def build_gdss_scorenet_a():
    """Tiny-size real GDSS ScoreNetworkA: the graph-multi-head-attention
    score network that denoises the adjacency matrix under the GDSS SDE."""
    return ScoreNetworkA(
        max_feat_num=6,
        max_node_num=9,
        nhid=8,
        num_layers=3,
        num_linears=2,
        c_init=2,
        c_hid=4,
        c_final=4,
        adim=8,
        num_heads=2,
        conv="GCN",
    )


def example_input_gdss_scorenet_a():
    """Tiny (x, adj, flags) batch matching ScoreNetworkA.forward's expected
    node-feature / adjacency / node-mask tensors, returned as a positional
    tuple (x, adj, flags) so tl.trace unpacks it into model(x, adj, flags)."""
    torch.manual_seed(0)
    batch, n, feat = 2, 9, 6
    x = torch.randn(batch, n, feat)
    adj_sym = torch.rand(batch, n, n)
    adj_sym = (adj_sym + adj_sym.transpose(-1, -2)) / 2
    flags = torch.ones(batch, n)
    return (x, adj_sym, flags)


def build_gdss_scorenet_x():
    """Tiny-size real GDSS ScoreNetworkX: the DenseGCNConv-stack score
    network that denoises node features under the GDSS SDE."""
    return ScoreNetworkX(max_feat_num=6, depth=3, nhid=8)


def example_input_gdss_scorenet_x():
    """Tiny (x, adj, flags) batch matching ScoreNetworkX.forward's expected
    node-feature / adjacency / node-mask tensors, returned as a positional
    tuple (x, adj, flags) so tl.trace unpacks it into model(x, adj, flags)."""
    torch.manual_seed(0)
    batch, n, feat = 2, 9, 6
    x = torch.randn(batch, n, feat)
    adj_sym = torch.rand(batch, n, n)
    adj_sym = (adj_sym + adj_sym.transpose(-1, -2)) / 2
    flags = torch.ones(batch, n)
    return (x, adj_sym, flags)


MENAGERIE_ENTRIES = [
    ("GDSS-ScoreNetworkA", build_gdss_scorenet_a, example_input_gdss_scorenet_a, 2022, "CODE"),
    ("GDSS-ScoreNetworkX", build_gdss_scorenet_x, example_input_gdss_scorenet_x, 2022, "CODE"),
]
