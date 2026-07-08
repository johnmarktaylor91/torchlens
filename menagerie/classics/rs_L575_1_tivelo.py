# SOURCE: vendored from cuhklinlab/TIVelo @ afd9117577e7e184d584313e895cca871fbe4bcc
#   tivelo/velocity/model_rate.py: DNNLayer
#
# TIVelo (Xu, Zheng, et al., Nature Communications 2025) is a trajectory-informed RNA
# velocity estimation tool for scRNA-seq. Its "rate" velocity-inference mode learns a
# kinetic-rate DNN: an MLP that takes the concatenated unspliced/spliced expression
# vectors (x_u, x_s) for each cell and predicts three per-gene kinetic parameters
# (alpha=transcription rate, beta=splicing rate, gamma=degradation rate), which are then
# combined analytically into the RNA velocity vectors v_u, v_s (real code:
# tivelo/velocity/model_rate.py `DNNLayer.forward`). This is the genuine trainable
# nn.Module in the repo (the alternative "simple"/optimization-based velocity mode in
# model.py directly optimizes nn.Parameter tensors with no forward-pass network, so it
# is not a menagerie candidate). DNNLayer is reproduced verbatim below; only the relative
# import (`from ..utils.velocity_genes import compute_velocity_genes`, used solely by the
# standalone `DNN` training wrapper class, not by DNNLayer itself) is dropped since it is
# unused by the traced module.

import torch
import torch.nn as nn


class DNNLayer(nn.Module):
    def __init__(self, n_genes, n_dims=[256, 64]):
        super(DNNLayer, self).__init__()
        self.n_genes = n_genes

        self.alpha, self.beta, self.gamma = None, None, None

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Sequential(nn.Linear(self.n_genes * 2, n_dims[0]), nn.ReLU(True)))
        # hidden layers
        for i in range(len(n_dims) - 1):
            self.layers.append(nn.Sequential(nn.Linear(n_dims[i], n_dims[i + 1]), nn.ReLU(True)))

        out_layer_dim = n_genes * 3
        self.layers.append(nn.Sequential(nn.Linear(n_dims[-1], out_layer_dim), nn.ReLU(True)))

    def forward(self, x_u, x_s):
        h = torch.cat([x_u, x_s], dim=1)
        for i, layer in enumerate(self.layers):
            h = layer(h)
        x = h

        self.alpha = x[:, 0 : self.n_genes]
        self.beta = x[:, self.n_genes : 2 * self.n_genes]
        self.gamma = x[:, 2 * self.n_genes : 3 * self.n_genes]

        v_u = self.alpha - self.beta * x_u
        v_s = self.beta * x_u - self.gamma * x_s

        return v_u, v_s

    def get_current_batch_kinetic_rates(self):
        return self.current_kinetic_rates


def build_tivelo_dnn():
    # Real usage (tivelo/velocity/model_rate.py `DNN.__init__`): n_dims=[256, 64] by
    # default; n_genes is dataset-dependent (number of velocity genes). Shrunk here for
    # tiny tracing while preserving the real 2-hidden-layer architecture shape.
    return DNNLayer(n_genes=8, n_dims=[16, 8])


def example_input_tivelo_dnn():
    # Real usage: x_u, x_s are (n_cells, n_genes) unspliced/spliced expression matrices
    # (adata.layers["Mu"], adata.layers["Ms"]), batched via BatchSampler/DataLoader.
    batch, n_genes = 4, 8
    x_u = torch.rand(batch, n_genes)
    x_s = torch.rand(batch, n_genes)
    return (x_u, x_s)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("TIVelo-DNN", "build_tivelo_dnn", "example_input_tivelo_dnn", 2025, MENAGERIE_ZOO),
]
