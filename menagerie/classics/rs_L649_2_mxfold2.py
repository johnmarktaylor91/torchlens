# SOURCE: vendored from mxfold/mxfold2 @ master
# Files: mxfold2/fold/embedding.py, mxfold2/fold/transformer.py, mxfold2/fold/layers.py
#        (NeuralNet and its submodules only)
# MXfold2: RNA secondary structure prediction. The package's ZukerFold/RNAFold/MixedFold
# wrapper classes (mxfold2/fold/zuker.py, rnafold.py, mix.py) subclass AbstractFold, whose
# forward() dispatches into a compiled pybind11 extension (mxfold2.interface, the C++ Zuker
# dynamic-programming decoder) wrapped in torch.no_grad() -- that DP decoder is not a traceable
# nn.Module and mxfold2's compiled `interface` extension is not installed in this env (rung-2
# vendoring stops at code that runs in the base env). The real, differentiable, pure-torch
# neural net inside ZukerFold is `NeuralNet` (mxfold2/fold/layers.py): a CNN/BiLSTM/Transformer
# sequence encoder feeding PairedLayer/UnpairedLayer 2D conv heads that score every (i, j) base
# pair and every unpaired base -- this is the trainable architecture MXfold2 actually learns;
# the C++ code only turns those learned scores into a structure via classical Zuker DP (not a
# neural computation). We vendor and trace NeuralNet directly. Vendored verbatim aside from
# flattening the package-relative imports (`from .embedding import ...` etc.) into this single
# file.
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


"""============================================================================================="""
""" fold/embedding.py """
"""============================================================================================="""


class OneHotEmbedding(nn.Module):
    def __init__(self, ksize=0):
        super(OneHotEmbedding, self).__init__()
        self.n_out = 4
        self.ksize = ksize
        eye = np.identity(4, dtype=np.float32)
        zero = np.zeros(4, dtype=np.float32)
        self.onehot = defaultdict(
            lambda: np.ones(4, dtype=np.float32) / 4,
            {"a": eye[0], "c": eye[1], "g": eye[2], "t": eye[3], "u": eye[3], "0": zero},
        )

    def encode(self, seq):
        seq = [self.onehot[s] for s in seq.lower()]
        seq = np.vstack(seq)
        return seq.transpose()

    def pad_all(self, seq, pad_size):
        pad = "n" * pad_size
        seq = [pad + s + pad for s in seq]
        l = max([len(s) for s in seq])
        seq = [s + "0" * (l - len(s)) for s in seq]
        return seq

    def forward(self, seq):
        seq = self.pad_all(seq, self.ksize // 2)
        seq = [self.encode(s) for s in seq]
        return torch.from_numpy(np.stack(seq))  # pylint: disable=no-member


class SparseEmbedding(nn.Module):
    def __init__(self, dim):
        super(SparseEmbedding, self).__init__()
        self.n_out = dim
        self.embedding = nn.Embedding(6, dim, padding_idx=0)
        self.vocb = defaultdict(lambda: 5, {"0": 0, "a": 1, "c": 2, "g": 3, "t": 4, "u": 4})

    def __call__(self, seq):
        seq = torch.LongTensor([[self.vocb[c] for c in s.lower()] for s in seq])
        seq = seq.to(self.embedding.weight.device)
        return self.embedding(seq).transpose(1, 2)


"""============================================================================================="""
""" fold/transformer.py """
"""============================================================================================="""


class TransformerLayer(nn.Module):
    def __init__(self, n_in, n_head, n_hidden, n_layers, dropout=0.5):
        super(TransformerLayer, self).__init__()
        self.pos_encoder = PositionalEncoding(n_in, dropout, max_len=1000)
        encoder_layers = TransformerEncoderLayer(n_in, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers, nn.LayerNorm(n_in))
        self.n_in = self.n_out = n_in

    def forward(self, x):  # (B, C, N)
        x = x.permute(2, 0, 1)  # (N, B, C)
        x = x * math.sqrt(self.n_in)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x.permute(1, 0, 2)  # (B, N, C)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):  # (N, B, C)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


"""============================================================================================="""
""" fold/layers.py """
"""============================================================================================="""


class CNNLayer(nn.Module):
    def __init__(
        self,
        n_in,
        num_filters=(128,),
        filter_size=(7,),
        pool_size=(1,),
        dilation=1,
        dropout_rate=0.0,
        resnet=False,
    ):
        super(CNNLayer, self).__init__()
        self.resnet = resnet
        self.net = nn.ModuleList()
        for n_out, ksize, p in zip(num_filters, filter_size, pool_size):
            self.net.append(
                nn.Sequential(
                    nn.Conv1d(
                        n_in,
                        n_out,
                        kernel_size=ksize,
                        dilation=2**dilation,
                        padding=2**dilation * (ksize // 2),
                    ),
                    nn.MaxPool1d(p, stride=1, padding=p // 2) if p > 1 else nn.Identity(),
                    nn.GroupNorm(1, n_out),  # same as LayerNorm?
                    nn.CELU(),
                    nn.Dropout(p=dropout_rate),
                )
            )
            n_in = n_out

    def forward(self, x):  # (B=1, 4, N)
        for net in self.net:
            x_a = net(x)
            x = x + x_a if self.resnet and x.shape[1] == x_a.shape[1] else x_a
        return x


class CNNLSTMEncoder(nn.Module):
    def __init__(
        self,
        n_in,
        num_filters=(256,),
        filter_size=(7,),
        pool_size=(1,),
        dilation=0,
        num_lstm_layers=0,
        num_lstm_units=0,
        num_att=0,
        dropout_rate=0.0,
        resnet=True,
    ):
        super(CNNLSTMEncoder, self).__init__()
        self.resnet = resnet
        self.n_in = self.n_out = n_in
        while len(num_filters) > len(filter_size):
            filter_size = tuple(filter_size) + (filter_size[-1],)
        while len(num_filters) > len(pool_size):
            pool_size = tuple(pool_size) + (pool_size[-1],)
        if num_lstm_layers == 0 and num_lstm_units > 0:
            num_lstm_layers = 1

        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv = self.lstm = self.att = None

        if len(num_filters) > 0 and num_filters[0] > 0:
            self.conv = CNNLayer(
                n_in,
                num_filters,
                filter_size,
                pool_size,
                dilation,
                dropout_rate=dropout_rate,
                resnet=self.resnet,
            )
            self.n_out = n_in = num_filters[-1]

        if num_lstm_layers > 0:
            self.lstm = nn.LSTM(
                n_in,
                num_lstm_units,
                num_layers=num_lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout_rate if num_lstm_layers > 1 else 0,
            )
            self.n_out = n_in = num_lstm_units * 2
            self.lstm_ln = nn.LayerNorm(self.n_out)

        if num_att > 0:
            self.att = nn.MultiheadAttention(self.n_out, num_att, dropout=dropout_rate)

    def forward(self, x):  # (B, n_in, N)
        if self.conv is not None:
            x = self.conv(x)  # (B, C, N)
        x = torch.transpose(x, 1, 2)  # (B, N, C)

        if self.lstm is not None:
            x_a, _ = self.lstm(x)
            x_a = self.lstm_ln(x_a)
            x_a = self.dropout(F.celu(x_a))  # (B, N, H*2)
            x = x + x_a if self.resnet and x.shape[2] == x_a.shape[2] else x_a

        if self.att is not None:
            x = torch.transpose(x, 0, 1)
            x_a, _ = self.att(x, x, x)
            x = x + x_a
            x = torch.transpose(x, 0, 1)

        return x


class Transform2D(nn.Module):
    def __init__(self, join="cat", context_length=0):
        super(Transform2D, self).__init__()
        self.join = join

    def forward(self, x_l, x_r):
        assert x_l.shape == x_r.shape
        B, N, C = x_l.shape
        x_l = x_l.view(B, N, 1, C).expand(B, N, N, C)
        x_r = x_r.view(B, 1, N, C).expand(B, N, N, C)
        if self.join == "cat":
            x = torch.cat((x_l, x_r), dim=3)  # (B, N, N, C*2)
        elif self.join == "add":
            x = x_l + x_r  # (B, N, N, C)
        elif self.join == "mul":
            x = x_l * x_r  # (B, N, N, C)

        return x


class PairedLayer(nn.Module):
    def __init__(
        self,
        n_in,
        n_out=1,
        filters=(),
        ksize=(),
        fc_layers=(),
        dropout_rate=0.0,
        exclude_diag=True,
        resnet=True,
    ):
        super(PairedLayer, self).__init__()

        self.resnet = resnet
        self.exclude_diag = exclude_diag
        while len(filters) > len(ksize):
            ksize = tuple(ksize) + (ksize[-1],)

        self.conv = nn.ModuleList()
        for m, k in zip(filters, ksize):
            self.conv.append(
                nn.Sequential(
                    nn.Conv2d(n_in, m, k, padding=k // 2),
                    nn.GroupNorm(1, m),
                    nn.CELU(),
                    nn.Dropout(p=dropout_rate),
                )
            )
            n_in = m

        fc = []
        for m in fc_layers:
            fc += [nn.Linear(n_in, m), nn.LayerNorm(m), nn.CELU(), nn.Dropout(p=dropout_rate)]
            n_in = m
        fc += [nn.Linear(n_in, n_out)]
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        diag = 1 if self.exclude_diag else 0
        B, N, _, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x_u = torch.triu(x.view(B * C, N, N), diagonal=diag).view(B, C, N, N)
        x_l = torch.tril(x.view(B * C, N, N), diagonal=-1).view(B, C, N, N)
        x = torch.cat((x_u, x_l), dim=0).view(B * 2, C, N, N)
        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.resnet and x.shape[1] == x_a.shape[1] else x_a  # (B*2, n_out, N, N)
        x_u, x_l = torch.split(x, B, dim=0)  # (B, n_out, N, N) * 2
        x_u = torch.triu(x_u.view(B, -1, N, N), diagonal=diag)
        x_l = torch.tril(x_u.view(B, -1, N, N), diagonal=-1)
        x = x_u + x_l  # (B, n_out, N, N)
        x = x.permute(0, 2, 3, 1).view(B * N * N, -1)
        x = self.fc(x)
        return x.view(B, N, N, -1)  # (B, N, N, n_out)


class UnpairedLayer(nn.Module):
    def __init__(
        self, n_in, n_out=1, filters=(), ksize=(), fc_layers=(), dropout_rate=0.0, resnet=True
    ):
        super(UnpairedLayer, self).__init__()

        self.resnet = resnet
        while len(filters) > len(ksize):
            ksize = tuple(ksize) + (ksize[-1],)

        self.conv = nn.ModuleList()
        for m, k in zip(filters, ksize):
            self.conv.append(
                nn.Sequential(
                    nn.Conv1d(n_in, m, k, padding=k // 2),
                    nn.GroupNorm(1, m),
                    nn.CELU(),
                    nn.Dropout(p=dropout_rate),
                )
            )
            n_in = m

        fc = []
        for m in fc_layers:
            fc += [nn.Linear(n_in, m), nn.LayerNorm(m), nn.CELU(), nn.Dropout(p=dropout_rate)]
            n_in = m
        fc += [nn.Linear(n_in, n_out)]  # , nn.LayerNorm(n_out) ]
        self.fc = nn.Sequential(*fc)

    def forward(self, x, x_base=None):
        B, N, C = x.shape
        x = x.transpose(1, 2)  # (B, n_in, N)
        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.resnet and x.shape[1] == x_a.shape[1] else x_a
        x = x.transpose(1, 2).view(B * N, -1)  # (B, N, n_out)
        x = self.fc(x)
        return x.view(B, N, -1)


class LengthLayer(nn.Module):
    def __init__(self, n_in, layers=(), dropout_rate=0.5):
        super(LengthLayer, self).__init__()
        self.n_in = n_in
        n = n_in if isinstance(n_in, int) else np.prod(n_in)

        l = []
        for m in layers:
            l += [nn.Linear(n, m), nn.CELU(), nn.Dropout(p=dropout_rate)]
            n = m
        l += [nn.Linear(n, 1)]
        self.net = nn.Sequential(*l)

        if isinstance(self.n_in, int):
            self.x = torch.tril(torch.ones((self.n_in, self.n_in)))
        else:
            n = np.prod(self.n_in)
            x = np.fromfunction(
                lambda i, j, k, l: np.logical_and(k <= i, l <= j), (*self.n_in, *self.n_in)
            )
            self.x = torch.from_numpy(x.astype(np.float32)).reshape(n, n)

    def forward(self, x):
        return self.net(x)

    def make_param(self):
        device = next(self.net.parameters()).device
        x = self.forward(self.x.to(device))
        return x.reshape((self.n_in,) if isinstance(self.n_in, int) else self.n_in)


class NeuralNet(nn.Module):
    def __init__(
        self,
        embed_size=0,
        num_filters=(96,),
        filter_size=(5,),
        dilation=0,
        pool_size=(1,),
        num_lstm_layers=0,
        num_lstm_units=0,
        num_att=0,
        num_transformer_layers=0,
        num_transformer_hidden_units=2048,
        num_transformer_att=8,
        no_split_lr=False,
        pair_join="cat",
        num_paired_filters=(),
        paired_filter_size=(),
        num_hidden_units=(32,),
        dropout_rate=0.0,
        fc_dropout_rate=0.0,
        exclude_diag=True,
        n_out_paired_layers=0,
        n_out_unpaired_layers=0,
        **kwargs,
    ):
        super(NeuralNet, self).__init__()

        self.no_split_lr = no_split_lr
        self.pair_join = pair_join
        self.embedding = OneHotEmbedding() if embed_size == 0 else SparseEmbedding(embed_size)
        n_in = self.embedding.n_out

        if num_transformer_layers == 0:
            self.encoder = CNNLSTMEncoder(
                n_in,
                num_filters=num_filters,
                filter_size=filter_size,
                pool_size=pool_size,
                dilation=dilation,
                num_att=num_att,
                num_lstm_layers=num_lstm_layers,
                num_lstm_units=num_lstm_units,
                dropout_rate=dropout_rate,
            )
        else:
            self.encoder = TransformerLayer(
                n_in,
                n_head=num_transformer_att,
                n_hidden=num_transformer_hidden_units,
                n_layers=num_transformer_layers,
                dropout=dropout_rate,
            )
        n_in = self.encoder.n_out

        if self.pair_join != "bilinear":
            self.transform2d = Transform2D(join=pair_join)

            n_in_paired = n_in // 2 if pair_join != "cat" else n_in
            if self.no_split_lr:
                n_in_paired *= 2

            self.fc_paired = PairedLayer(
                n_in_paired,
                n_out_paired_layers,
                filters=num_paired_filters,
                ksize=paired_filter_size,
                exclude_diag=exclude_diag,
                fc_layers=num_hidden_units,
                dropout_rate=fc_dropout_rate,
            )
            if n_out_unpaired_layers > 0:
                self.fc_unpaired = UnpairedLayer(
                    n_in,
                    n_out_unpaired_layers,
                    filters=num_paired_filters,
                    ksize=paired_filter_size,
                    fc_layers=num_hidden_units,
                    dropout_rate=fc_dropout_rate,
                )
            else:
                self.fc_unpaired = None

        else:
            n_in_paired = n_in // 2 if not self.no_split_lr else n_in
            self.bilinear = nn.Bilinear(n_in_paired, n_in_paired, n_out_paired_layers)
            self.linear = nn.Linear(n_in, n_out_unpaired_layers)

    def forward(self, seq):
        device = next(self.parameters()).device
        x = self.embedding(["0" + s for s in seq]).to(device)  # (B, 4, N)
        x = self.encoder(x)

        if self.no_split_lr:
            x_l, x_r = x, x
        else:
            x_l = x[:, :, 0::2]
            x_r = x[:, :, 1::2]
        x_r = x_r[:, :, torch.arange(x_r.shape[-1] - 1, -1, -1)]  # reverse the last axis

        if self.pair_join != "bilinear":
            x_lr = self.transform2d(x_l, x_r)

            score_paired = self.fc_paired(x_lr)
            if self.fc_unpaired is not None:
                score_unpaired = self.fc_unpaired(x)
            else:
                score_unpaired = None

            return score_paired, score_unpaired

        else:
            B, N, C = x_l.shape
            x_l = x_l.view(B, N, 1, C).expand(B, N, N, C).reshape(B * N * N, -1)
            x_r = x_r.view(B, 1, N, C).expand(B, N, N, C).reshape(B * N * N, -1)
            score_paired = self.bilinear(x_l, x_r).view(B, N, N, -1)
            score_unpaired = self.linear(x)

            return score_paired, score_unpaired


MENAGERIE_ZOO = "vendored-pytorch"


def build_mxfold2_neuralnet():
    torch.manual_seed(0)
    return NeuralNet(
        embed_size=8,
        num_filters=(16,),
        filter_size=(5,),
        num_lstm_layers=1,
        num_lstm_units=8,
        num_hidden_units=(16,),
        n_out_paired_layers=2,
        n_out_unpaired_layers=1,
    )


def example_input_mxfold2_neuralnet():
    # `seq` must be a sequence-of-strings, not a bare tensor -- NeuralNet.forward builds its own
    # tensors internally via SparseEmbedding. A plain `list[str]` here would collide with
    # TorchLens's ergonomic text-tokenization input coercion (a `list[str]` positional arg is
    # heuristically treated as batched text needing a tokenizer); `seq` is RNA sequence data, not
    # natural-language text, so it is passed as a `tuple[str]` -- the real model code only ever
    # iterates it (`['0' + s for s in seq]`), which a tuple supports identically to a list.
    return (("acgugcua",),)


MENAGERIE_ENTRIES = [
    (
        "MXfold2-NeuralNet",
        "build_mxfold2_neuralnet",
        "example_input_mxfold2_neuralnet",
        2021,
        "SOURCE_AVAILABLE",
    ),
]
