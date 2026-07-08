# SOURCE: vendored from cquzys/stImpute @ 5a096592af0746b073b48d08a171355ab1a32797
# https://github.com/cquzys/stImpute/blob/main/model.py
#
# stImpute imputes unmeasured genes in spatial transcriptomics (ST) data by
# transferring information from a paired scRNA-seq reference. The `Trans`
# module is the core graph-based transfer network: it embeds each gene's
# expression-across-cells vector through `gnnlayers` GraphSAGE-style mean-
# aggregation blocks (`GS_block`, using a gene-gene similarity graph built
# from ESM-2 sequence embeddings or expression correlation), flattens, and
# maps through an MLP (`trans`) to predict the target gene's expression
# across cells; `reliable` is a second head trained to score imputation
# confidence. Vendored verbatim (only whitespace/lint-clean; no
# architectural changes) -- imports/relative-paths trimmed to the
# `Trans`/`GS_block` forward path (the pip-installable base-lib subset);
# the file's stImpute()/AutoEncoder training-orchestration code and its
# `from utils import *` (scipy/sklearn helper functions) are intentionally
# not vendored since they are training-loop glue, not part of the traced
# nn.Module architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

MENAGERIE_ZOO = "vendored-pytorch"


class GS_block(nn.Module):
    def __init__(self, input_dim: int = 50, output_dim: int = 50):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim * 2, output_dim))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: Tensor, adj: Tensor):
        neigh_feats = self.aggregate(x, adj)
        combined = torch.cat(
            [x.reshape(-1, self.input_dim), neigh_feats.reshape(-1, self.input_dim)], dim=1
        )
        combined = F.relu(combined @ self.weight)
        combined = F.normalize(combined, 2, 1).reshape(x.shape[0], -1)
        return combined

    def aggregate(self, x: Tensor, adj: Tensor):
        n = len(adj)
        adj = adj - torch.eye(n, device=adj.device)
        adj /= adj.sum(1, keepdim=True) + 1e-12
        return adj.mm(x)


class Trans(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cell_num: int,
        hidden_dim: int = 256,
        gnnlayers: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.seed = seed

        self.n_neighbors = input_dim
        self.mse = nn.MSELoss(reduction="mean")
        self.cos_by_col = nn.CosineSimilarity(dim=1)
        self.cos_by_row = nn.CosineSimilarity(dim=0)

        self.graphlayers = nn.ModuleList([GS_block(input_dim, input_dim) for _ in range(gnnlayers)])
        self.trans = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reliable = nn.Sequential(
            nn.Linear(cell_num, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X: Tensor, graph: Tensor):
        X_hat = X.t()
        for layer in self.graphlayers:
            X_hat = layer(X_hat, graph)
        X_hat = X_hat.reshape(-1, self.n_neighbors)
        Y_hat = self.trans(X_hat).reshape(X.shape[1], -1).t()
        return Y_hat


def build_stimpute():
    # Tiny sizing for menagerie tracing: input_dim (n_neighbors/cells)=8,
    # cell_num (n_genes for the reliable head)=20, output_dim=1, matching
    # the real repo's stImpute() call: Trans(input_dim=n_neighbors,
    # output_dim=1, cell_num=spatial_df.shape[0], gnnlayers=2).
    return Trans(input_dim=8, output_dim=1, cell_num=20, hidden_dim=16, gnnlayers=2)


def example_input_stimpute():
    # X: (n_cells, n_genes) expression matrix; graph: (n_genes, n_genes)
    # gene-gene similarity adjacency, matching build_graph_by_gene() output
    # in the real repo's utils.py.
    n_cells, n_genes = 8, 20
    X = torch.randn(n_cells, n_genes)
    graph = torch.rand(n_genes, n_genes)
    return (X, graph)


MENAGERIE_ENTRIES = [
    ("stImpute", build_stimpute, example_input_stimpute, 2023, "vendored-pytorch"),
]
