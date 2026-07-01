# SOURCE: vendored from OmicsML/dance @ main
# https://github.com/OmicsML/dance/blob/main/dance/modules/multi_modality/joint_embedding/scmogcn.py
#
# scMoGNN (Wen, Tang, Ding, Jin, Xu, Zhao, Tang & He 2022, arXiv:2203.01884,
# "Graph Neural Networks for Multimodal Single-Cell Data Integration"; NeurIPS
# 2022 Multimodal Single-Cell Data Integration challenge winner) integrates
# paired multi-omics single-cell data (e.g. RNA+ATAC / RNA+ADT) into a joint
# embedding. The full published pipeline builds a bipartite cell<->feature
# graph and runs DGL heterogeneous graph convolution (cell_feature_propagation
# in the original scmogcn.py) BEFORE the traceable model; that graph-prep step
# is feature engineering, not the nn.Module itself, and requires dgl (not a
# base lib here). The `ScMoGCN` class -- the actual trainable encoder/decoder
# network the paper's ScMoGCNWrapper.fit()/predict() calls as
# `self.model(batch_x)` on the already-propagated cell features -- is a plain
# torch.nn.Module using only nn.Linear/nn.BatchNorm1d/nn.GELU/nn.Sequential/
# F.dropout, no DGL dependency. Vendored verbatim (import path only; DGL-only
# `cell_feature_propagation`/`propagation_layer_combination` graph-prep
# helpers and the ScMoGCNWrapper training/eval harness are intentionally
# omitted since they are not part of the architecture and pull in dgl).

import torch.nn as nn
import torch.nn.functional as F
import torch

MENAGERIE_ZOO = "vendored-pytorch"


class ScMoGCN(nn.Module):
    def __init__(self, nb_cell_types, nb_batches, nb_phases, input_dimension):
        super().__init__()
        self.nb_cell_types = nb_cell_types
        self.nb_batches = nb_batches
        self.nb_phases = nb_phases

        self.linear1 = nn.Linear(input_dimension, 150)
        self.linear2 = nn.Linear(150, 120)
        self.linear3 = nn.Linear(120, 100)
        self.linear4 = nn.Linear(100, 61)

        self.bn1 = nn.BatchNorm1d(150)
        self.bn2 = nn.BatchNorm1d(120)
        self.bn3 = nn.BatchNorm1d(100)

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.act3 = nn.GELU()

        self.decoder = nn.Sequential(
            nn.Linear(61, 150),
            nn.ReLU(),
            nn.Linear(150, input_dimension),
            nn.ReLU(),
        )

    def encoder(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.linear2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.linear3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.linear4(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x0 = x
        x = self.decoder(x)

        return (
            x,
            x0[:, : self.nb_cell_types],
            x0[:, self.nb_cell_types : self.nb_cell_types + self.nb_batches],
            x0[
                :,
                self.nb_cell_types + self.nb_batches : self.nb_cell_types
                + self.nb_batches
                + self.nb_phases,
            ],
        )


def build_scmogcn():
    # Real defaults come from the NeurIPS 2022 challenge datasets (e.g.
    # nb_cell_types up to ~45, input_dimension = concatenated propagated
    # RNA+ATAC/ADT feature dim, often several hundred to a few thousand).
    # Shrunk here for menagerie tracing; architecture (4-layer GELU/BatchNorm
    # encoder -> 61-dim latent split into cell-type/batch/phase heads,
    # 2-layer ReLU decoder) is unchanged.
    return ScMoGCN(nb_cell_types=10, nb_batches=6, nb_phases=2, input_dimension=128)


def example_input_scmogcn():
    # (batch, input_dimension) already cell-feature-propagated joint
    # RNA+second-modality embedding (post-DGL-graph-conv concat), matching
    # what ScMoGCNWrapper.fit() passes as `batch_x` into `self.model(batch_x)`.
    return (torch.randn(4, 128),)


MENAGERIE_ENTRIES = [
    ("scMoGNN", "build_scmogcn", "example_input_scmogcn", 2022, "vendored-pytorch"),
]
