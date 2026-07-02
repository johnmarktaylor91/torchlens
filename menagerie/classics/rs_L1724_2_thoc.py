# SOURCE: vendored from https://github.com/carrtesy/THOC-Pytorch @ master
# (model.py)
#
# THOC (Temporal Hierarchical One-Class network for time-series anomaly
# detection, Shin et al. NeurIPS 2020) -- real repo code, vendored verbatim.
# A dilated recurrent neural network (DRNN, GRU cells with exponentially
# growing dilation rates, modified from the zalandoresearch dilated-RNN
# reference the repo itself credits) produces multiscale temporal features;
# a hierarchy of soft cluster assignments (learned cluster-center parameter
# matrices, cosine-similarity soft assignment, small per-level MLP updates
# fusing each level's hidden state) computes a THOC deep one-class distance
# loss, plus an orthogonality regularizer on the cluster centers and a
# temporal self-supervision (TSS) auxiliary loss predicting future window
# values from each DRNN level. No layer, cell type, or dataflow inside the
# architecture was changed -- only the `if __name__ == "__main__"` smoke
# block at the bottom of the real file was dropped (not architecture).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class THOC(nn.Module):
    def __init__(self, C, W, n_hidden, tau=1, device="cpu"):
        super(THOC, self).__init__()
        self.device = device
        self.C, self.W = C, W
        self.L = math.floor(math.log(W, 2)) + 1  # number of DRNN layers
        self.tau = 1
        self.DRNN = DRNN(n_input=C, n_hidden=n_hidden, n_layers=self.L, device=device)

        self.clusters = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_hidden, K_l)) for K_l in range(self.L * 2, 0, -2)]
        )
        for c in self.clusters:
            nn.init.xavier_uniform_(c)

        self.cnets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_hidden, n_hidden),
                    nn.ReLU(),
                )
                for _ in range(self.L)  # nn that maps f_bar to f_hat
            ]
        )

        self.MLP = nn.Sequential(
            nn.Linear(n_hidden * 2, n_hidden * 2),
            nn.ReLU(),
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )

        self.TSSnets = nn.ModuleList([nn.Linear(n_hidden, C) for _ in range(self.L)])

    def forward(self, X):
        """
        :param X: (B, W, C)
        :return: Losses,
        """
        B, W, C = X.shape
        MTF_output = self.DRNN(X)  # Multiscale Temporal Features from dilated RNN.

        # THOC
        L_THOC = 0
        anomaly_scores = torch.zeros(B, device=self.device)
        f_t_bar = MTF_output[0][:, -1, :].unsqueeze(1)
        Ps, Rs = [], []

        for i, cluster in enumerate(self.clusters):
            # Assignment
            eps = 1e-08
            f_norm, c_norm = (
                torch.norm(f_t_bar, dim=-1, keepdim=True),
                torch.norm(cluster, dim=0, keepdim=True),
            )
            f_norm, c_norm = (
                torch.max(f_norm, eps * torch.ones_like(f_norm, device=self.device)),
                torch.max(c_norm, eps * torch.ones_like(c_norm, device=self.device)),
            )
            cosine_similarity = torch.einsum("Bph,hn->Bpn", f_t_bar / f_norm, cluster / c_norm)

            P_ij = F.softmax(
                cosine_similarity / self.tau, dim=-1
            )  # (B, num_cluster_{l-1}, num_cluster_{l})
            R_ij = P_ij.squeeze(1) if len(Ps) == 0 else torch.einsum("Bp,Bpn->Bn", Rs[i - 1], P_ij)
            Ps.append(P_ij)
            Rs.append(R_ij)

            # Update
            c_vectors = self.cnets[i](f_t_bar)  # (B, num_cluster_{l-1}, hidden_dim)
            f_t_bar = torch.einsum(
                "Bnp,Bph->Bnh", P_ij.transpose(-1, -2), c_vectors
            )  # (B, num_cluster_{l}, hidden_dim)

            # fuse last hidden state
            B, num_prev_clusters, num_next_clusters = P_ij.shape
            if i != self.L - 1:
                last_hidden_state = (
                    MTF_output[i + 1][:, -1, :].unsqueeze(1).repeat(1, num_next_clusters, 1)
                )
                f_t_bar = self.MLP(torch.cat((f_t_bar, last_hidden_state), dim=-1))

            d = 1 - cosine_similarity
            w = R_ij.unsqueeze(1).repeat(1, num_prev_clusters, 1)
            distances = w * d
            anomaly_scores += torch.mean(distances, dim=(1, 2))
            L_THOC += torch.mean(distances)
        anomaly_scores /= len(self.clusters)

        # ORTH
        L_orth = 0
        for cluster in self.clusters:
            c_sq = cluster.T @ cluster  # (K_l, K_l)
            K_l, _ = c_sq.shape
            L_orth += torch.linalg.matrix_norm(c_sq - torch.eye(K_l, device=self.device))
        L_orth /= len(self.clusters)

        # TSS
        L_TSS = 0
        for i, net in enumerate(self.TSSnets):
            src, tgt = MTF_output[i][:, : -(2**i), :], X[:, 2**i :, :]
            L_TSS += F.mse_loss(net(src), tgt)
        L_TSS /= len(self.TSSnets)

        loss_dict = {
            "L_THOC": L_THOC,
            "L_orth": L_orth,
            "L_TSS": L_TSS,
        }
        return anomaly_scores, loss_dict


# Dilated RNN code modified from:
# https://github.com/zalandoresearch/pytorch-dilated-rnn/blob/master/drnn.py
class DRNN(nn.Module):
    def __init__(
        self,
        n_input,
        n_hidden,
        n_layers,
        dropout=0,
        cell_type="GRU",
        batch_first=True,
        device="cpu",
    ):
        super(DRNN, self).__init__()

        self.dilations = [2**i for i in range(n_layers)]
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.device = device

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        inputs = inputs.transpose(0, 1) if self.batch_first else inputs
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])
            _output = inputs.transpose(0, 1) if self.batch_first else inputs
            outputs.append(_output)
        return outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(
                dilated_inputs, cell, batch_size, rate, hidden_size
            )
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(
                dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden
            )

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == "LSTM":
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize : (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(
            dilated_outputs.size(0) * rate, batchsize, dilated_outputs.size(2)
        )
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = (n_steps % rate) == 0

        if not is_even:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(
                dilated_steps * rate - inputs.size(0), inputs.size(1), inputs.size(2)
            ).to(self.device)

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim).to(self.device)
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim).to(self.device)
            return (hidden, memory)
        else:
            return hidden


def build_thoc():
    # Tiny config matching the real THOCTrainer construction
    # THOC(C=num_channels, W=window_size, n_hidden=hidden_dim, device=device):
    # C=4 channels, W=16 window (real repo default is 100; a smaller
    # power-of-two window keeps the DRNN dilation stack tiny: L =
    # floor(log2(16))+1 = 5 layers), n_hidden=8.
    return THOC(C=4, W=16, n_hidden=8, device="cpu")


def example_input_thoc():
    # Real forward(X) takes X: (batch, window_size, num_channels).
    return torch.randn(3, 16, 4)


MENAGERIE_ENTRIES = [
    (
        "THOC",
        build_thoc,
        example_input_thoc,
        2020,
        MENAGERIE_ZOO,
    ),
]
