# SOURCE: vendored from qugank/sketch-lattice.github.io @ 6ed272f8b4bd9b77860e4cc2b48bfb3375becdb6
# https://raw.githubusercontent.com/qugank/sketch-lattice.github.io/6ed272f8b4bd9b77860e4cc2b48bfb3375becdb6/encoder.py
# https://raw.githubusercontent.com/qugank/sketch-lattice.github.io/6ed272f8b4bd9b77860e4cc2b48bfb3375becdb6/decoder.py
#
# Qu et al. 2021 (ICCV 2021) "SketchLattice: Latticed Representation for Sketch Manipulation"
# -- a lattice-coordinate graph encoder ("EncoderGCN" built from "CoordinateEmbeddingXYSep"
# + "GCNPropagation") feeding a Sketch-RNN-style bivariate-Gaussian-mixture LSTM decoder
# ("DecoderRNN"). Each sketch stroke-endpoint is snapped to a fixed lattice grid and
# embedded via separate learned X/Y coordinate lookup tables, then propagated through a
# residual graph-convolution stack over a lattice adjacency matrix -- this coordinate-lattice
# GCN (as opposed to the sibling SketchHealer repo's image-patch GCN) is the paper's actual
# architectural contribution, so it must be vendored rather than mapped to a library class.
# `generation_sketch_gcn.py`'s `Model` class confirms `EncoderGCN()` (the coordinate-lattice
# encoder defined further down in `encoder.py`, not the `EncoderPatchGCN` image-patch
# variant that precedes it) is the encoder actually paired with `DecoderRNN()`.
#
# No architectural changes were made; only mechanical fixes for import isolation and
# CPU-portability:
#   - `encoder.py` imported `from generation_hyper_params import hp` (a module-level
#     singleton `HParams()` instance) and unused data-processing helpers
#     (`get_node_coordinates_graph`, `draw_three`); only the `hp` fields the vendored
#     classes actually read (`graph_number`, `embedding_dim`, `gcn_out_dim`, `Nz`, `M`,
#     `dec_hidden_size`, `dropout`, `Nmax`, `words_number`) are reproduced here inline as
#     a small `_HP` class.
#   - `GCNPropagation.__init__` unconditionally called `.cuda()` when building
#     `self.resBlockSequence` (`[nn.Sequential(*b).cuda() for b in self.resBlockPool]`),
#     which crashes on a CPU-only machine; the `.cuda()` call is dropped so the block
#     stays on the module's default device (CPU here) -- purely a device-portability fix,
#     not an architecture change (the upstream authors' own `Model.__init__` already
#     branches on `hp.use_cuda` for every other submodule, so this stray unconditional
#     `.cuda()` looks like an oversight in the original code rather than an intentional
#     GPU-only design constraint).
#   - Only `EncoderGCN`, `CoordinateEmbeddingXYSep`, `GCNPropagation`, and `DecoderRNN`
#     are kept; the unused `FeatureExtractionBasic`/`FeatureExtraction`/
#     `CoordinateEmbedding`/`GCNPropagation2`/`EncoderPatchGCN`/`SPAttention` classes
#     (image-patch variants and a stubbed-out, never-implemented attention class) and the
#     `if __name__ == '__main__':` smoke-test blocks are dropped.
#   - `decoder.py` is otherwise reproduced verbatim (its `F.softmax(..., dim=...)` calls
#     already carry explicit `dim=` in the upstream source, unlike the sibling
#     SketchHealer repo).

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HP:
    """Minimal stand-in for the upstream `generation_hyper_params.HParams` singleton,
    restricted to the fields `encoder.py`/`decoder.py` actually read."""

    graph_number = 12  # lattice node count for a tiny synthetic sketch (paper default: 150)
    # NOTE: `GCNPropagation.forward` never applies `self.out_linear`, so the tensor it
    # returns keeps `embedding_dim` channels, not `gcn_out_dim` -- and `EncoderGCN.norm1`/
    # `fc_mu`/`fc_sigma` are all built assuming a `gcn_out_dim`-and-`Nz`-sized input.
    # Upstream only "works" because the paper's own defaults set
    # `embedding_dim == gcn_out_dim == Nz == 128`, silently masking this (upstream,
    # unmodified) dead-code / shape mismatch. Faithfully reproducing the working
    # configuration means these three must stay equal here too.
    embedding_dim = 24  # coordinate embedding size (paper default: 128)
    gcn_out_dim = 24  # GCN propagation output size (paper default: 128); must equal embedding_dim/Nz, see above
    Nz = 24  # encoder output / latent size (paper default: 128); must equal embedding_dim/gcn_out_dim, see above
    M = 6  # number of bivariate-Gaussian mixture components (paper default: 20)
    dec_hidden_size = 32  # decoder LSTM hidden size (paper default: 512)
    dropout = 0.0
    Nmax = 5  # max stroke count for a tiny synthetic sketch
    words_number = 32  # lattice grid resolution / embedding-table size (paper default: 256)


hp = _HP()


class CoordinateEmbeddingXYSep(nn.Module):
    def __init__(self, words_number: int, out_dim: int):
        super().__init__()
        self.words_number = words_number
        self.out_dim = out_dim
        self.embX = nn.Embedding(words_number, out_dim // 2, padding_idx=0)
        self.embY = nn.Embedding(words_number, out_dim // 2, padding_idx=0)

    def forward(self, c: torch.Tensor):
        c = c.long()
        x, y = torch.split(c, 1, dim=2)
        x_emb = self.embX(x.squeeze(2))
        y_emb = self.embY(y.squeeze(2))
        c = torch.cat([x_emb, y_emb], dim=2)
        c = c.view(-1, hp.graph_number, self.out_dim).contiguous()
        return c


class GCNPropagation(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.resBlockPool = [
            [
                nn.Linear(dim_in, dim_in // 2 * 3, bias=False),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_in // 2 * 3, dim_in, bias=False),
                nn.ReLU(),
            ]
            * 2
        ]

        self.resBlockSequence = [nn.Sequential(*b) for b in self.resBlockPool]
        # Register the plain-list resBlockSequence's parameters so they are tracked by
        # this module (the upstream code relies on the `.cuda()` call it chained onto
        # the list comprehension to implicitly register parameters as a side effect of
        # moving them off of CPU; dropping that call per the CPU-portability fix above
        # means we must register the submodules explicitly instead).
        for i, block in enumerate(self.resBlockSequence):
            self.add_module(f"resBlock_{i}", block)

        self.out_linear = nn.Linear(dim_in, dim_out, bias=False)

        self.norm = nn.BatchNorm1d(hp.graph_number)
        self.relu = nn.ReLU()

        self.register_buffer("I", torch.eye(hp.graph_number))

    def normalize(self, A, symmetric=True):
        """
        not work!
        :param A:
        :param symmetric:
        :return:
        """
        A += self.I
        d = A.sum(axis=2)
        if symmetric:
            D = torch.diag_embed(torch.pow(d, -0.5))
            return torch.matmul(D, torch.matmul(A, D))
        else:
            D = torch.diag_embed(torch.pow(d, -1))
            return torch.matmul(D, A)

    def forward(self, X, A):
        """
        :param X: (batch, graph_num, in_feature_num)
        :param A: (batch, graph_num, graph_num)
        :return:
        """
        A = A + self.I

        for block in self.resBlockSequence:
            X = torch.matmul(A, X)
            last = X
            X = block(X)
            X = X + last

        return X


class EncoderGCN(nn.Module):
    def __init__(
        self,
    ):
        super(EncoderGCN, self).__init__()
        # model
        self.emb = CoordinateEmbeddingXYSep(hp.words_number, hp.embedding_dim)

        self.gcn = GCNPropagation(hp.embedding_dim, hp.gcn_out_dim)

        self.fc_h2z = nn.Linear(hp.gcn_out_dim, hp.Nz)
        # z, mu, sigma
        self.fc_mu = nn.Linear(hp.Nz, hp.Nz)
        self.fc_sigma = nn.Linear(hp.Nz, hp.Nz)

        self.norm1 = nn.BatchNorm1d(hp.gcn_out_dim)

    def forward(self, C, A):
        """
        return z, mu, sigma
        :param input_imgs: (batch_size, graph_num, 3, graph_size, graph_size)
        :param adj_matrix: (batch_size, graph_num, graph_num)
        """
        C = self.emb(C)
        X = self.gcn(C, A)
        X = torch.sum(X, dim=1)  # (B, S, dims)
        X = self.norm1(X)
        X = torch.tanh(X)

        # generate mu sigma
        mu = self.fc_mu(X)
        sigma = self.fc_sigma(X)
        sigma_e = torch.exp(sigma / 2.0)

        # normal sample
        z_size = mu.size()
        if mu.get_device() != -1:  # not in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda(mu.get_device())
        else:  # in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        # sample z
        z = mu + sigma_e * n
        return z, mu, sigma, X


class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        # to init **hidden and cell** from z:
        self.fc_hc = nn.Linear(hp.Nz, 2 * hp.dec_hidden_size)
        # unidirectional lstm:
        self.lstm = nn.LSTM(5 + hp.Nz, hp.dec_hidden_size, dropout=hp.dropout)
        # create proba distribution parameters from hiddens:
        self.fc_params = nn.Linear(hp.dec_hidden_size, 6 * hp.M + 3)

    def forward(self, inputs, z, hidden_cell=None):
        if hidden_cell is None:
            # then we must init from z
            hidden, cell = torch.split(torch.tanh(self.fc_hc(z)), hp.dec_hidden_size, 1)
            hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell)
        # in training we feed the LSTM with the whole input in one shot
        # and use all outputs contained in 'outputs',
        # while in generate mode we just feed with the last generated sample:
        if self.training:
            y = self.fc_params(outputs.view(-1, hp.dec_hidden_size))
        else:
            y = self.fc_params(hidden.view(-1, hp.dec_hidden_size))
        # separate pen and mixture params:
        params = torch.split(y, 6, 1)
        params_mixture = torch.stack(params[:-1])  # trajectory
        params_pen = params[-1]  # pen up/down
        # identify mixture params:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)
        # preprocess params:
        if self.training:
            len_out = hp.Nmax + 1
        else:
            len_out = 1

        if self.training:
            pi = F.softmax(pi.transpose(0, 1).squeeze(), dim=1).view(len_out, -1, hp.M)
            sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
            sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
            rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
            mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
            mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
            q = F.softmax(params_pen, dim=1).view(len_out, -1, 3)
        else:
            pi = F.softmax(pi.transpose(0, 1).squeeze(), dim=0).view(len_out, -1, hp.M)
            sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
            sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
            rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(len_out, -1, hp.M)
            mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
            mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(len_out, -1, hp.M)
            q = F.softmax(params_pen, dim=1).view(len_out, -1, 3)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell


class SketchLattice(nn.Module):
    """Wraps the encoder/decoder pair so the model traces as a single module, matching
    the real (encoder, decoder) split owned by upstream `generation_sketch_gcn.py`'s
    `Model` class."""

    def __init__(self):
        super().__init__()
        self.encoder = EncoderGCN()
        self.decoder = DecoderRNN()

    def forward(self, coords, adj_matrix, dec_inputs):
        z, mu, sigma, _ = self.encoder(coords, adj_matrix)
        z_stack = torch.stack([z] * dec_inputs.size(0))
        lstm_inputs = torch.cat([dec_inputs, z_stack], 2)
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = self.decoder(lstm_inputs, z)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, mu, sigma


def build_sketchlattice():
    model = SketchLattice()
    model.eval()
    return model


def example_input_sketchlattice():
    batch = 2
    graph_num = hp.graph_number
    # lattice (x, y) coordinates, one of `words_number` bins per axis:
    coords = torch.randint(0, hp.words_number, (batch, graph_num, 2)).float()
    adj_matrix = torch.stack([torch.eye(graph_num) for _ in range(batch)])
    dec_inputs = torch.rand(hp.Nmax + 1, batch, 5)
    return (coords, adj_matrix, dec_inputs)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SketchLattice", "build_sketchlattice", "example_input_sketchlattice", 2021, "vendored"),
]
