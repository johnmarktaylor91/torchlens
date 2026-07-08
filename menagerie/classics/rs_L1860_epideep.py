# SOURCE: vendored from https://github.com/epideep/source @ master
# (deepClusteringwCLTime.py + rnnAttention.py)
# EpiDeep: Exploiting Features of Future for Prediction of Epidemic Diseases
# (Adhikari, Xu, Ramakrishnan & Prakash, KDD 2019). The model classes below
# (`RNNTime`, `DeepClusteringTime`, and their shared `buildNetwork` helper) are
# copied verbatim from the official repo's deepClusteringwCLTime.py and
# rnnAttention.py. Only import paths were adjusted (repo used bare `from
# rnnAttention import RNNTime`; inlined here into one file for staging) and
# training-only methods that depend on sklearn (`fit`, `pre_train`, `predict`,
# `embed`) were dropped since they are not part of the forward architecture --
# `forward_clustering_first`/`forward_clustering_second` (the encoder/decoder +
# soft-assignment path) and `RNNTime.forward` (the LSTM+attention regressor)
# are kept exactly as in the source.
"""Vendored EpiDeep model definition (DeepClusteringTime + RNNTime)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "leakyReLU":
            net.append(nn.LeakyReLU())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class RNNTime(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_internal, emd_size, out_size):
        super(RNNTime, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # encoder
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # attention model
        self.attn = torch.nn.Parameter(torch.randn(hidden_size, 1))  # (20)*1

        # decoder
        self.dropout = nn.Dropout(p=0)
        self.fc = nn.Linear(hidden_size + emd_size, num_internal)
        self.fc2 = nn.Linear(num_internal, num_internal)
        self.fc3 = nn.Linear(num_internal, num_internal)
        self.fc4 = nn.Linear(num_internal, out_size)
        self.stmax = nn.Softmax()
        self.activation = nn.LeakyReLU()

    def forward(self, x, emd):
        # Set initial states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate RNN, lstm output shape: output,(h_n,c_n)
        out, (hidden, cell) = self.lstm(x, (h0, c0))
        hidden_state = hidden[1].unsqueeze(0)
        hidden_state = hidden_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(out, hidden_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2), dim=-1).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        out = torch.bmm(torch.transpose(out, 1, 2), weights).squeeze(2)

        out = out.unsqueeze(2)

        # Dropout
        out = self.dropout(out)

        out_new = torch.squeeze(out, 2)

        out1 = torch.cat((out_new, emd), 1)  # merge out_new and emd through the dimension 1
        out2 = self.activation(self.fc(out1))
        out3 = self.activation(self.fc2(out2))
        out4 = self.activation(self.fc3(out3))

        out = self.stmax(self.fc4(out4))

        return out


class DeepClusteringTime(nn.Module):
    def __init__(
        self,
        input1_dim,
        embed1_dim,
        input2_dim,
        embed2_dim,
        n_centroids,
        encode_layers=[500, 200],
        decode_layers=[200, 500],
        mapping_layers=[100, 200, 100],
        output_size=53,
    ):
        super(self.__class__, self).__init__()
        self.input1_dim = input1_dim
        self.embed1_dim = embed1_dim
        self.n_centroids = n_centroids
        self.input2_dim = input2_dim
        self.embed2_dim = embed2_dim

        self.first_encoder = buildNetwork([input1_dim] + encode_layers + [embed1_dim])
        self.first_decoder = buildNetwork([embed1_dim] + encode_layers + [input1_dim])

        self.first_cluster_layer = Parameter(torch.Tensor(n_centroids, embed1_dim))
        torch.nn.init.xavier_normal_(self.first_cluster_layer.data)

        self.second_encoder = buildNetwork([input2_dim] + encode_layers + [embed2_dim])
        self.second_decoder = buildNetwork([embed2_dim] + encode_layers + [input2_dim])

        self.second_cluster_layer = Parameter(torch.Tensor(n_centroids, embed2_dim))
        torch.nn.init.xavier_normal_(self.second_cluster_layer.data)

        self.mapper = buildNetwork(
            [embed1_dim] + mapping_layers + [embed2_dim], activation="leakyReLU"
        )

        self.regressor = RNNTime(1, 20, 2, 20, embed1_dim, output_size)
        self.alpha = 1

    def forward_clustering_first(self, x1):
        z1 = self.first_encoder(x1)
        x1_bar = self.first_decoder(z1)

        q = 1.0 / (
            1.0
            + torch.sum(torch.pow(z1.unsqueeze(1) - self.first_cluster_layer, 2), 2) / self.alpha
        )
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x1_bar, q, z1

    def forward_clustering_second(self, x2):
        z2 = self.second_encoder(x2)
        x2_bar = self.second_decoder(z2)

        q = 1.0 / (
            1.0
            + torch.sum(torch.pow(z2.unsqueeze(1) - self.second_cluster_layer, 2), 2) / self.alpha
        )
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x2_bar, q, z2

    def forward(self, x1, rnn_data):
        """Staging-only forward: clusters x1 through the first autoencoder,
        maps the embedding through `mapper`, and feeds it plus `rnn_data`
        through the RNNTime regressor -- mirrors the `predict()` path in the
        original repo (first_encoder -> mapper -> regressor.forward)."""
        x1_bar, q1, z1 = self.forward_clustering_first(x1)
        translated_emb = self.mapper(z1)
        pred = self.regressor(rnn_data, translated_emb)
        return pred, x1_bar, q1


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny config, scaled down from the original
# repo defaults -- encode_layers=[500,200] etc -- for fast tracing).
# ---------------------------------------------------------------------------

_INPUT1_DIM = 16
_EMBED1_DIM = 8
_INPUT2_DIM = 16
_EMBED2_DIM = 8
_N_CENTROIDS = 4
_OUTPUT_SIZE = 5
_SEQ_LEN = 6
_BATCH = 3


def build_epideep():
    return DeepClusteringTime(
        input1_dim=_INPUT1_DIM,
        embed1_dim=_EMBED1_DIM,
        input2_dim=_INPUT2_DIM,
        embed2_dim=_EMBED2_DIM,
        n_centroids=_N_CENTROIDS,
        encode_layers=[32, 16],
        decode_layers=[16, 32],
        mapping_layers=[16, 16],
        output_size=_OUTPUT_SIZE,
    )


def example_input_epideep():
    x1 = torch.rand(_BATCH, _INPUT1_DIM)
    rnn_data = torch.rand(_BATCH, _SEQ_LEN, 1)
    return (x1, rnn_data)


MENAGERIE_ENTRIES = [
    ("EpiDeep", "build_epideep", "example_input_epideep", 2019, "vendored-pytorch"),
]
