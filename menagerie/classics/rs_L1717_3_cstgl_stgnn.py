# SOURCE: vendored from huankoh/CST-GL @ main
#
# Vendors the real nn.Module classes from layer.py (nconv/linear/mixprop graph-convolution
# building blocks + dilated_inception temporal-conv + LayerNorm), mtcl.py (graph_constructor,
# the "Multivariate Time Series Correlation Layer" that learns a sparse adjacency over sensor
# nodes), and stgnn.py (the top-level `stgnn` spatio-temporal graph neural network: dilated
# inception temporal convolutions + mix-hop graph propagation + skip connections), used in the
# CST-GL paper for graph-based ICS/SCADA process-anomaly detection (e.g. on the WADI dataset,
# per the catalog queue's "Graph-WADI-AD" framing). Construction args below mirror run.py's
# CLI defaults, shrunk to a tiny node/channel count for a fast trace.
#
# Repo: https://github.com/huankoh/CST-GL @ main
# Files: layer.py, mtcl.py, stgnn.py

from __future__ import division

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

MENAGERIE_ZOO = "vendored-pytorch"


# ---- layer.py ----
class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum("ncwl,vw->ncvl", (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        # adj normalization
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        # graph propagation
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)  # alpha=0.05
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3) :]
        x = torch.cat(x, dim=1)
        return x


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(
                input,
                tuple(input.shape[1:]),
                self.weight[:, idx, :],
                self.bias[:, idx, :],
                self.eps,
            )
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )


# ---- mtcl.py ----
class graph_constructor(nn.Module):
    """
    Graph Constructor is the Multivariate Time Series Correlation Layer (MTCL) in the paper.
    """

    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float("0"))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)  # rand for numerical stability
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


# ---- stgnn.py ----
class stgnn(nn.Module):
    def __init__(
        self,
        gcn_true,
        buildA_true,
        gcn_depth,
        num_nodes,
        device,
        predefined_A=None,
        static_feat=None,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        seq_length=12,
        in_dim=2,
        out_dim=12,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
    ):
        super(stgnn, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)
        )
        self.gc = graph_constructor(
            num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat
        )

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential**layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1
                    + i
                    * (kernel_size - 1)
                    * (dilation_exponential**layers - 1)
                    / (dilation_exponential - 1)
                )
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i
                        + (kernel_size - 1)
                        * (dilation_exponential**j - 1)
                        / (dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels, dilation_factor=new_dilation
                    )
                )
                self.gate_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels, dilation_factor=new_dilation
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=residual_channels,
                        kernel_size=(1, 1),
                    )
                )
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.seq_length - rf_size_j + 1),
                        )
                    )
                else:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.receptive_field - rf_size_j + 1),
                        )
                    )

                if self.gcn_true:
                    self.gconv1.append(
                        mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
                    )
                    self.gconv2.append(
                        mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
                    )

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm(
                            (residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                            elementwise_affine=layer_norm_affline,
                        )
                    )
                else:
                    self.norm.append(
                        LayerNorm(
                            (residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                            elementwise_affine=layer_norm_affline,
                        )
                    )

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True
        )
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self.seq_length - self.receptive_field + 1),
                bias=True,
            )
        else:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self.receptive_field),
                bias=True,
            )
            self.skipE = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
            )

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, (
            "input sequence length not equal to preset sequence length"
        )

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3) :]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        if self.training:
            return x
        else:
            return x, adp


def build_cstgl_stgnn():
    device = torch.device("cpu")
    model = stgnn(
        gcn_true=True,
        buildA_true=True,
        gcn_depth=2,
        num_nodes=8,
        device=device,
        predefined_A=None,
        dropout=0.1,
        subgraph_size=4,
        node_dim=16,
        dilation_exponential=1,
        conv_channels=8,
        residual_channels=8,
        skip_channels=16,
        end_channels=32,
        seq_length=5,
        in_dim=1,
        out_dim=1,
        layers=2,
        propalpha=0.1,
        tanhalpha=20,
        layer_norm_affline=True,
    )
    model.eval()
    return model


def example_input_cstgl_stgnn():
    # input shape: (batch, in_dim, num_nodes, seq_length)
    return torch.randn(2, 1, 8, 5)


MENAGERIE_ENTRIES = [
    (
        "Graph-WADI-AD (Process-graph ICS anomaly)",
        build_cstgl_stgnn,
        example_input_cstgl_stgnn,
        2021,
        "SOURCE_AVAILABLE",
    ),
]
