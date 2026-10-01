# SOURCE: vendored from aicb-ZhangLabs/Impeller @ main
# (model.py, classes PathGNN and PathGNNLayer)
#
# Impeller's flagship contribution: a path-based heterogeneous graph learning method for
# spatial transcriptomic data imputation (Bioinformatics 2024). `PathGNN` aggregates node
# features along pre-sampled random-walk paths (per edge type) with learnable per-step path
# weights, then linearly projects back to the gene-expression feature space; `PathGNNLayer`
# implements the per-layer path aggregation + type-wise fusion (hstack for 2 edge types) +
# linear + ReLU. Both classes are copied verbatim from model.py aside from dropping the
# unused `torch_sparse`/`RGCNConv`/other-baseline imports at module scope (model.py imports many
# baseline GNNs -- GCNConv, GATConv, SAGEConv, TransformerConv, RGCNConv, a custom GATConv,
# STAGATE -- for its ablation baselines; none of those are used by PathGNN/PathGNNLayer, so they
# are omitted here to keep this file self-contained for PathGNN alone).
#
# The real repo constructs `paths`/`path_types` via `utils.get_paths`, which depends on `dgl`
# (node2vec random walks over the input graph) -- a non-base-lib dependency. `dgl` is used only
# for *data prep* (sampling which node-index paths to aggregate over), not for anything inside
# the PathGNN/PathGNNLayer architecture itself, so for tracing we synthesize the same-shaped
# `paths`/`path_types` tensors directly (random node-index paths) instead of installing dgl.
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class PathGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        dropout,
        num_layers,
        num_paths,
        path_length,
        num_edge_types,
        alpha,
        operator_type="independent",
    ):
        super(PathGNN, self).__init__()
        self._dropout = dropout
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc_in.weight, gain=1.414)
        self.in_act = nn.ReLU()

        self.fc_out = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_normal_(self.fc_out.weight, gain=1.414)
        self.out_act = nn.ReLU()

        self.layers = nn.ModuleList(
            [
                PathGNNLayer(hidden_dim, num_paths, path_length, num_edge_types)
                for _ in range(num_layers)
            ]
        )

        if operator_type == "global":
            self.path_weights = nn.ModuleList(
                [
                    nn.ParameterList(
                        [
                            nn.Parameter(torch.Tensor(1, path_length, 1))
                            for _ in range(num_edge_types)
                        ]
                    )
                ]
            )
        elif operator_type == "shared_layer":
            self.path_weights = nn.ModuleList(
                [
                    nn.ParameterList(
                        [
                            nn.Parameter(torch.Tensor(1, path_length, hidden_dim))
                            for _ in range(num_edge_types)
                        ]
                    )
                ]
            )
        elif operator_type == "shared_channel":
            self.path_weights = nn.ModuleList(
                [
                    nn.ParameterList(
                        [
                            nn.Parameter(torch.Tensor(1, path_length, 1))
                            for _ in range(num_edge_types)
                        ]
                    )
                    for _ in range(num_layers)
                ]
            )
        elif operator_type == "independent":
            self.path_weights = nn.ModuleList(
                [
                    nn.ParameterList(
                        [
                            nn.Parameter(torch.Tensor(1, path_length, hidden_dim))
                            for _ in range(num_edge_types)
                        ]
                    )
                    for _ in range(num_layers)
                ]
            )

        for path_weight_layer in self.path_weights:
            for path_weight in path_weight_layer:
                nn.init.xavier_normal_(path_weight, gain=1.414)

        self.num_layers = num_layers
        self.num_paths = num_paths
        self.path_length = path_length
        self.num_edge_types = num_edge_types
        self.alpha = alpha
        self.operator_type = operator_type

    def forward(self, input_x, paths, path_types):
        in_feats = F.dropout(input_x, p=self._dropout, training=self.training)
        in_feats = self.fc_in(in_feats)
        in_feats = self.in_act(in_feats)

        feats = in_feats
        for i in range(self.num_layers):
            if self.operator_type == "global" or self.operator_type == "shared_layer":
                feats = self.layers[i](feats, paths, path_types, self.path_weights[0])
            elif self.operator_type == "shared_channel" or self.operator_type == "independent":
                feats = self.layers[i](feats, paths, path_types, self.path_weights[i])
            else:
                raise NotImplementedError
            feats = self.alpha * in_feats + (1 - self.alpha) * feats

        feats = F.dropout(feats, p=self._dropout, training=self.training)
        out = self.fc_out(feats)
        out = self.out_act(out)
        return out

    def setup_optimizer(self, lr, wd, lr_oc, wd_oc):
        param_list = [
            {"params": self.layers.parameters(), "lr": lr, "weight_decay": wd},
            {
                "params": itertools.chain(*[self.fc_in.parameters(), self.fc_out.parameters()]),
                "lr": lr_oc,
                "weight_decay": wd_oc,
            },
        ]
        return torch.optim.Adam(param_list)


class PathGNNLayer(nn.Module):
    def __init__(self, hidden_dim, num_path, path_length, num_edge_types):
        super(PathGNNLayer, self).__init__()

        self.fc = nn.Linear(num_edge_types * hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.num_path = num_path
        self.path_length = path_length
        self.num_edge_types = num_edge_types

    def forward(self, feats, paths, path_types, path_weights):
        """
        feats: (num_nodes, d),
        paths: (num_path, num_nodes, path_length)
        path_types: (num_path,) contains the edge type of each path
        """
        results = []
        for edge_type, path_weight in enumerate(path_weights):
            mask = path_types == edge_type  # select the paths of this type
            paths_of_type = paths[mask]  # (num_paths_of_type, num_nodes, path_length)
            path_feats = feats[paths_of_type]  # (num_paths_of_type, num_nodes, path_length, d)
            path_feats = (path_feats * path_weight).sum(dim=2)  # (num_paths_of_type, num_nodes, d)
            path_feats = path_feats.mean(dim=0)  # (num_nodes, d)
            results.append(path_feats)
        if self.num_edge_types == 2:
            fout = torch.hstack((results[0], results[1]))
        else:
            fout = results[0]

        fout = self.fc(fout)
        fout = F.relu(fout)
        return fout


class ImpellerPathGNNTraceWrapper(nn.Module):
    """Thin wrapper binding fixed-shape `paths`/`path_types` buffers so PathGNN can be traced
    from a single example-input tensor (TorchLens' recipe/module contract is one concrete-tensor
    call). The random-walk path *indices* are ordinary trace-time data (any valid node-index path
    of the right shape exercises identical PathGNN/PathGNNLayer architecture), not part of the
    model's learned architecture."""

    def __init__(
        self, num_nodes, in_dim, hidden_dim, num_layers, num_paths, path_length, num_edge_types
    ):
        super().__init__()
        self.pathgnn = PathGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=in_dim,
            dropout=0.1,
            num_layers=num_layers,
            num_paths=num_paths,
            path_length=path_length,
            num_edge_types=num_edge_types,
            alpha=0.5,
            operator_type="independent",
        )
        # Fixed random-walk path indices over `num_nodes` nodes (mirrors utils.get_paths' output
        # shape: (num_paths, num_nodes, path_length), long dtype, values in [0, num_nodes)).
        paths = torch.randint(0, num_nodes, (num_paths, num_nodes, path_length))
        path_types = torch.cat(
            [
                torch.zeros(num_paths // 2, dtype=torch.long),
                torch.ones(num_paths - num_paths // 2, dtype=torch.long),
            ]
        )
        self.register_buffer("paths", paths)
        self.register_buffer("path_types", path_types)

    def forward(self, x):
        return self.pathgnn(x, self.paths, self.path_types)


def build_impeller_pathgnn():
    return ImpellerPathGNNTraceWrapper(
        num_nodes=12,
        in_dim=16,
        hidden_dim=8,
        num_layers=2,
        num_paths=4,
        path_length=3,
        num_edge_types=2,
    )


def example_input_impeller_pathgnn():
    return torch.rand(12, 16)


MENAGERIE_ENTRIES = [
    (
        "Impeller-PathGNN",
        build_impeller_pathgnn,
        example_input_impeller_pathgnn,
        2024,
        "SOURCE_AVAILABLE",
    ),
]
