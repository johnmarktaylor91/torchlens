# SOURCE: vendored from kuanglab/GNTD @ main (789e5cb, GNTD/NTD.py)
#
# GNTD (Graph-regularized Neural Tensor Decomposition): a 3-mode (gene x spot-x x spot-y)
# neural tensor decomposition model for spatial-transcriptome imputation, regularized during
# training (outside this module) by Cartesian-product graph Laplacians over a gene PPI graph
# and a spatial neighbor graph (Yang, Zhao & Kuang, "GNTD: reconstructing spatial transcriptomes
# with graph-guided neural tensor decomposition", Nature Communications 2023). `NTD` (in
# GNTD/NTD.py) is the actual trainable nn.Module: three per-mode Embedding + Linear + PReLU
# stacks that nonlinearly project mode indices into a shared rank-d latent space, then combine
# them via a trilinear einsum + ReLU into the reconstructed 3-way tensor. `GNTD/GNTD.py`'s
# `GNTD` class (not vendored here) is a plain-Python training-loop orchestrator wrapping this
# module with data preprocessing and graph-Laplacian regularization terms, not an nn.Module
# itself and not part of the architecture. Copied verbatim from the real repo's NTD class.
import torch
from torch.nn import Embedding, LeakyReLU, Linear, Parameter, PReLU

MENAGERIE_ZOO = "vendored-pytorch"


class NTD(torch.nn.Module):
    def __init__(self, n_x, n_y, n_g, rank, random_state=1234567):
        super().__init__()
        torch.manual_seed(random_state)

        # Define embedding layer along x, y, g modes
        self.embedding_x = Embedding(n_x, rank)
        self.embedding_y = Embedding(n_y, rank)
        self.embedding_g = Embedding(n_g, rank)
        # Define nonlinear mapping layer along x, y, g modes
        self.lin_x_1 = Linear(rank, rank)
        self.lin_y_1 = Linear(rank, rank)
        self.lin_g_1 = Linear(rank, rank)
        self.prelu = PReLU(init=0.9)

    def forward(self, x_index, y_index, g_index):
        # Linear factors
        x = self.embedding_x(x_index)
        y = self.embedding_y(y_index)
        g = self.embedding_g(g_index)

        # Nonlinear factors
        x = self.lin_x_1(x)
        x = self.prelu(x)
        y = self.lin_y_1(y)
        y = self.prelu(y)
        g = self.lin_g_1(g)
        g = self.prelu(g)

        # Nonlinear aggregation
        o = torch.einsum("im,jm,km->ijk", g, x, y)
        o = o.relu_()

        return x, y, g, o


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------
_N_X, _N_Y, _N_G, _RANK = 8, 8, 6, 4


def build_gntd_ntd():
    model = NTD(n_x=_N_X, n_y=_N_Y, n_g=_N_G, rank=_RANK)
    model.eval()
    return model


def example_input_gntd_ntd():
    x_index = torch.arange(_N_X, dtype=torch.long)
    y_index = torch.arange(_N_Y, dtype=torch.long)
    g_index = torch.arange(_N_G, dtype=torch.long)
    return (x_index, y_index, g_index)


MENAGERIE_ENTRIES = [
    (
        "GNTD",
        build_gntd_ntd,
        example_input_gntd_ntd,
        2023,
        MENAGERIE_ZOO,
    ),
]
