# SOURCE: vendored from fahadahaf/satori @ master (satori/models.py)
# https://raw.githubusercontent.com/fahadahaf/satori/master/satori/models.py
#
# Ahmad, F. & Shrikumar, A. (2021) "SATORI: A System for Ontology-guided Visual
# Exploration of Biomedical Data Repositories" -- SATORI's `AttentionNet` is a
# CNN + multi-head self-attention model for regulatory-genomics sequence
# classification (interpretable attention over TF-binding motifs). A single
# Conv1d+BatchNorm+activation(+MaxPool) stem feeds `numMultiHeads` independent
# per-head Linear Q/K/V projections (scaled dot-product attention computed
# manually, exactly as in the original), whose outputs are concatenated and
# passed through a `MultiHeadLinear` + ReLU readout, an optional
# sum-then-normalize pooling over the sequence axis, then a final linear
# classifier head `fc3`. `PositionalEncoding` and `Exponential` (a learnable
# exponential activation used in place of ReLU when `CNN1_useexponential`) are
# copied verbatim too, though the traced config below uses the CNN-only,
# ReLU, no-RNN, no-positional-encoding branch (SATORI's shipped
# `CNN-MH-noEmbds` config, see `modelsparam/CNN-MH-noEmbds_hyperParams.txt`).
# All classes are copied verbatim from the real `satori/models.py`; no
# architectural changes were made. Only the `argSpace`/`params` construction
# below (a tiny `SimpleNamespace` + dict) is new scaffolding to instantiate
# the real class at a small size for a fast trace.

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter  # noqa: F401  (kept for parity with source)


class PositionalEncoding(nn.Module):
    # Taken from: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


class Exponential(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class AttentionNet(nn.Module):  # for the model that uses CNN, RNN (optionally), and MH attention
    def __init__(self, argSpace, params, device=None, genPAttn=True, reuseWeightsQK=False):
        super(AttentionNet, self).__init__()
        self.numMultiHeads = params["num_multiheads"]
        self.SingleHeadSize = params["singlehead_size"]  # SingleHeadSize
        self.MultiHeadSize = params["multihead_size"]  # MultiHeadSize
        self.usepooling = params["use_pooling"]
        self.pooling_val = params["pooling_val"]
        self.readout_strategy = params["readout_strategy"]
        self.kmerSize = params["embd_kmersize"]
        self.useRNN = params["use_RNN"]
        self.useCNN = params["use_CNN"]
        self.CNN1useExponential = params["CNN1_useexponential"]
        self.usePE = params["use_posEnc"]
        self.useCNNpool = params["use_CNNpool"]
        self.RNN_hiddenSize = params["RNN_hiddensize"]
        self.numCNNfilters = params["CNN_filters"]
        self.filterSize = params["CNN_filtersize"]
        self.CNNpoolSize = params["CNN_poolsize"]
        self.CNNpadding = params["CNN_padding"]
        self.numClasses = argSpace.numLabels
        self.device = device
        self.reuseWeightsQK = reuseWeightsQK
        self.numInputChannels = params["input_channels"]  # number of channels, one hot encoding
        self.genPAttn = genPAttn

        if self.usePE:
            self.pe = PositionalEncoding(d_model=self.numInputChannels, dropout=0.1)

        if self.useCNN and self.useCNNpool:
            self.layer1 = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.numInputChannels,
                    out_channels=self.numCNNfilters,
                    kernel_size=self.filterSize,
                    padding=self.CNNpadding,
                    bias=False,
                ),
                nn.BatchNorm1d(num_features=self.numCNNfilters),
                nn.ReLU() if not self.CNN1useExponential else Exponential(),
                nn.MaxPool1d(kernel_size=self.CNNpoolSize),
            )
            self.dropout1 = nn.Dropout(p=0.2)

        if self.useCNN and not self.useCNNpool:
            self.layer1 = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.numInputChannels,
                    out_channels=self.numCNNfilters,
                    kernel_size=self.filterSize,
                    padding=self.CNNpadding,
                    bias=False,
                ),
                nn.BatchNorm1d(num_features=self.numCNNfilters),
                nn.ReLU() if not self.CNN1useExponential else Exponential(),
            )
            self.dropout1 = nn.Dropout(p=0.2)

        if self.useRNN:
            self.RNN = nn.LSTM(
                self.numInputChannels if not self.useCNN else self.numCNNfilters,
                self.RNN_hiddenSize,
                num_layers=2,
                bidirectional=True,
            )
            self.dropoutRNN = nn.Dropout(p=0.4)
            self.Q = nn.ModuleList(
                [
                    nn.Linear(in_features=2 * self.RNN_hiddenSize, out_features=self.SingleHeadSize)
                    for i in range(0, self.numMultiHeads)
                ]
            )
            self.K = nn.ModuleList(
                [
                    nn.Linear(in_features=2 * self.RNN_hiddenSize, out_features=self.SingleHeadSize)
                    for i in range(0, self.numMultiHeads)
                ]
            )
            self.V = nn.ModuleList(
                [
                    nn.Linear(in_features=2 * self.RNN_hiddenSize, out_features=self.SingleHeadSize)
                    for i in range(0, self.numMultiHeads)
                ]
            )

        if not self.useRNN and not self.useCNN:
            self.Q = nn.ModuleList(
                [
                    nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize)
                    for i in range(0, self.numMultiHeads)
                ]
            )
            self.K = nn.ModuleList(
                [
                    nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize)
                    for i in range(0, self.numMultiHeads)
                ]
            )
            self.V = nn.ModuleList(
                [
                    nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize)
                    for i in range(0, self.numMultiHeads)
                ]
            )

        if not self.useRNN and self.useCNN:
            self.Q = nn.ModuleList(
                [
                    nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize)
                    for i in range(0, self.numMultiHeads)
                ]
            )
            self.K = nn.ModuleList(
                [
                    nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize)
                    for i in range(0, self.numMultiHeads)
                ]
            )
            self.V = nn.ModuleList(
                [
                    nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize)
                    for i in range(0, self.numMultiHeads)
                ]
            )

        # reuse weights between Query (Q) and Key (K)
        if self.reuseWeightsQK:
            for i in range(0, self.numMultiHeads):
                self.K[i].weight = Parameter(self.Q[i].weight.t())

        self.RELU = nn.ModuleList([nn.ReLU() for i in range(0, self.numMultiHeads)])
        self.MultiHeadLinear = nn.Linear(
            in_features=self.SingleHeadSize * self.numMultiHeads, out_features=self.MultiHeadSize
        )  # 50
        self.MHReLU = nn.ReLU()

        self.fc3 = nn.Linear(in_features=self.MultiHeadSize, out_features=self.numClasses)

    def attention(self, query, key, value, mask=None, dropout=0.0):
        # based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, inputs):
        output = inputs

        if self.usePE:
            output = self.pe(output)

        if self.useCNN:
            output = self.layer1(output)
            output = self.dropout1(output)
            output = output.permute(0, 2, 1)

        if self.useRNN:
            output, _ = self.RNN(output)
            F_RNN = output[:, :, : self.RNN_hiddenSize]
            R_RNN = output[:, :, self.RNN_hiddenSize :]
            output = torch.cat((F_RNN, R_RNN), 2)
            output = self.dropoutRNN(output)

        pAttn_concat = torch.Tensor([]).to(self.device)
        attn_concat = torch.Tensor([]).to(self.device)
        for i in range(0, self.numMultiHeads):
            query, key, value = self.Q[i](output), self.K[i](output), self.V[i](output)
            attnOut, p_attn = self.attention(query, key, value, dropout=0.2)
            attnOut = self.RELU[i](attnOut)
            if self.usepooling:
                attnOut = self.MAXPOOL[i](attnOut.permute(0, 2, 1)).permute(0, 2, 1)
            attn_concat = torch.cat((attn_concat, attnOut), dim=2)
            if self.genPAttn:
                pAttn_concat = torch.cat((pAttn_concat, p_attn), dim=2)

        output = self.MultiHeadLinear(attn_concat)
        output = self.MHReLU(output)

        if self.readout_strategy == "normalize":
            output = output.sum(axis=1)
            output = (output - output.mean()) / output.std()

        output = self.fc3(output)
        assert not torch.isnan(output).any()
        if self.genPAttn:
            return output, pAttn_concat
        else:
            return output


def build_satori_attentionnet():
    # Real shipped config: SATORI's `CNN-MH-noEmbds` hyperparameter set
    # (modelsparam/CNN-MH-noEmbds_hyperParams.txt) -- CNN stem + multi-head
    # attention, no RNN, no positional encoding, ReLU (not Exponential) CNN1
    # activation, sum-then-normalize readout. CNN_filters/multihead_size shrunk
    # (200->16 / 100->8) for a fast trace; num_multiheads/singlehead_size kept
    # at the real defaults (8 / 32). numLabels is a small placeholder (2, binary
    # motif-classification task as in the real Human_Promoters/TAL-GATA demos).
    params = {
        "input_channels": 4,  # DNA one-hot channels (real default)
        "num_multiheads": 8,  # real default
        "singlehead_size": 32,  # real default
        "multihead_size": 16,  # shrunk from real default 100
        "use_pooling": False,  # real default (CNN-MH-noEmbds config)
        "pooling_val": 6,
        "readout_strategy": "normalize",  # real default
        "embd_kmersize": 3,
        "use_RNN": False,  # real default (CNN-MH-noEmbds config)
        "use_CNN": True,  # real default
        "CNN1_useexponential": False,  # real default
        "use_posEnc": False,  # real default
        "use_CNNpool": True,  # real default
        "RNN_hiddensize": 100,
        "CNN_filters": 16,  # shrunk from real default 200
        "CNN_filtersize": 13,  # real default
        "CNN_poolsize": 6,  # real default
        "CNN_padding": 6,  # real default (matched to filtersize=13)
    }
    argSpace = SimpleNamespace(numLabels=2)
    model = AttentionNet(
        argSpace, params, device=torch.device("cpu"), genPAttn=True, reuseWeightsQK=False
    )
    model.eval()
    return model


def example_input_satori_attentionnet():
    # (batch, channels=4, seq_len) one-hot DNA-sequence input, as SATORI's
    # DataLoader yields (see satori/datasets.py DatasetLoadAll -> forward pass
    # in satori/experiment.py calls model(seq) with seq shaped (B, 4, L)).
    return torch.rand(2, 4, 200)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "SATORI-AttentionNet",
        "build_satori_attentionnet",
        "example_input_satori_attentionnet",
        2021,
        "vendored",
    ),
]
