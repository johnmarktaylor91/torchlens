# SOURCE: vendored from Nikitavolzhin/KAGNN-for-CHILI @ main
# Files: ekan.py (KANLinear/KAN -- itself vendored upstream from Blealtan/efficient-kan,
#        as declared in the repo's own file header), classification_models.py
#        (make_kan/KAGIN_cls).
#
# KA-GNN (Kolmogorov-Arnold Graph Neural Network) replaces the standard MLP update/readout
# heads inside classic message-passing GNN layers (GIN/GCN/EdgeConv) with learnable
# B-spline "KAN" layers (Kolmogorov-Arnold Networks, Liu et al. 2024) instead of
# linear-plus-fixed-activation blocks. KAGIN_cls wraps torch_geometric's GINConv with a
# KAN as its internal update MLP, stacks several such layers with BatchNorm/dropout, and
# reads out through a final KAN head -- graph classification via global_add_pool or, when
# node_level=True, direct per-node KAN classification. Vendored verbatim (only the
# graph-classification model class + its KAN dependency; the dgl/CHILI-specific dataset
# and training-script files are not part of the architecture).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_add_pool

MENAGERIE_ZOO = "vendored-pytorch"


# --- ekan.py (verbatim; upstream-attributed to Blealtan/efficient-kan in the source repo) ---
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        splines_out = self.b_splines(x).view(x.size(0), -1)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            splines_out,
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


# --- classification_models.py (verbatim architecture-relevant subset) ---
# The following function: make_kan, KAGIN_cls are from https://github.com/RomanBresson/KAGNN
# KAGIN_cls was modified (upstream, in the vendored source repo) for compatibility with
# node-level tasks.
def make_kan(num_features, hidden_dim, out_dim, hidden_layers, grid_size, spline_order):
    sizes = [num_features] + [hidden_dim] * (hidden_layers - 1) + [out_dim]
    return KAN(layers_hidden=sizes, grid_size=grid_size, spline_order=spline_order)


class KAGIN_cls(nn.Module):
    def __init__(
        self,
        gnn_layers,
        num_features,
        hidden_dim,
        num_classes,
        hidden_layers,
        grid_size,
        spline_order,
        dropout,
    ):
        super(KAGIN_cls, self).__init__()
        self.n_layers = gnn_layers
        lst = list()
        lst.append(
            GINConv(
                make_kan(
                    num_features, hidden_dim, hidden_dim, hidden_layers, grid_size, spline_order
                )
            )
        )
        for i in range(gnn_layers - 1):
            lst.append(
                GINConv(
                    make_kan(
                        hidden_dim, hidden_dim, hidden_dim, hidden_layers, grid_size, spline_order
                    )
                )
            )
        self.conv = nn.ModuleList(lst)
        lst = list()
        for i in range(gnn_layers):
            lst.append(nn.BatchNorm1d(hidden_dim))
        self.bn = nn.ModuleList(lst)
        self.kan = make_kan(
            hidden_dim, hidden_dim, num_classes, hidden_layers, grid_size, spline_order
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x, edge_index, batch, edge_attr=None, edge_weight=None, node_level: bool = True
    ):
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            x = self.bn[i](x)
            x = self.dropout(x)
        if node_level:
            x = self.kan(x)
            return F.log_softmax(x, dim=1)
        x = global_add_pool(x, batch)
        x = self.kan(x)
        return F.log_softmax(x, dim=1)


def build_kagin_cls():
    torch.manual_seed(0)
    return KAGIN_cls(
        gnn_layers=2,
        num_features=8,
        hidden_dim=16,
        num_classes=4,
        hidden_layers=2,
        grid_size=4,
        spline_order=3,
        dropout=0.0,
    )


def example_input_kagin_cls():
    # A single small batched graph: 10 nodes / 20 directed edges (undirected pairs),
    # matching the real forward() signature (x, edge_index, batch, node_level=True).
    torch.manual_seed(0)
    n_nodes = 10
    x = torch.randn(n_nodes, 8)
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 5),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
    ]
    src = [a for a, b in edges] + [b for a, b in edges]
    dst = [b for a, b in edges] + [a for a, b in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    return (x, edge_index, batch)


MENAGERIE_ENTRIES = [
    ("KA-GNN (KAGIN)", "build_kagin_cls", "example_input_kagin_cls", 2024, "vendored-pytorch"),
]
