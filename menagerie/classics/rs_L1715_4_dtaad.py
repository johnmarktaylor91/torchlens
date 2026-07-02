# SOURCE: vendored from https://github.com/Yu-Lingrui/DTAAD @ master
#
# Source: src/models.py, class DTAAD (the repo's proposed model: "Tcn_Local + Tcn_Global +
# Callback + Transformer + MAML" dual-branch time-series anomaly detector). DTAAD combines
# a local temporal-convolutional branch (`Tcn_Local`, src/gltcn.py) and a global dilated
# TCN branch (`Tcn_Global`, src/gltcn.py) that both feed causal-masked-free
# TransformerEncoder stacks (`TransformerEncoderLayer`/`PositionalEncoding`,
# src/dlutils.py) through a residual feed-forward + sigmoid decoder head, with the global
# branch's TCN re-run on a "callback" signal formed from the local branch's output
# residual. `Tcn_Local`/`Tcn_Global`/`Chomp1d`/`TemporalCnn` (src/gltcn.py) and
# `PositionalEncoding`/`TransformerEncoderLayer` (src/dlutils.py) are copied verbatim;
# `__init__`/`forward`/`callback` of `DTAAD` itself are copied verbatim. The real repo's
# `models.py` module-level imports `dgl`/`dgl.nn` (needed only by the sibling `MTAD_GAT`
# and `GDN` classes, not by `DTAAD`) and pulls `lr`/`math` from
# `src/constants.py`/`src/dlutils.py`; those two non-architectural globals are inlined
# here (`lr` is an unused-in-forward training hyperparameter attribute, `math` is stdlib)
# so this file has zero DTAAD-repo-specific import dependencies. One portability fix:
# `TransformerEncoderLayer.forward()` now accepts/ignores `**kwargs` since torch's real
# `nn.TransformerEncoder.forward()` (torch >= 2.x) forwards an `is_causal` kwarg this
# custom layer (written before that kwarg existed) didn't declare; no dataflow changed.

import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

MENAGERIE_ZOO = "vendored-pytorch"


# --- src/gltcn.py (verbatim) -------------------------------------------------


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """Cropping module: crops the extra rightmost padding (padding applied on both
        sides) so the convolution stays causal."""
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalCnn(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalCnn, self).__init__()
        self.conv = weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp = Chomp1d(padding)
        self.leakyrelu = nn.LeakyReLU(True)
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv, self.chomp, self.leakyrelu, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, out_channel, seq_len)
        :return: size of (Batch, out_channel, seq_len)
        """
        out = self.net(x)
        return out


class Tcn_Local(nn.Module):
    def __init__(self, num_outputs, kernel_size=3, dropout=0.2):  # k>=3
        super(Tcn_Local, self).__init__()
        layers = []
        num_levels = 3
        out_channels = num_outputs
        for i in range(num_levels):
            layers += [
                TemporalCnn(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=(kernel_size - 1),
                    dropout=dropout,
                )
            ]  # causal conv via padding + Chomp1d slicing

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: size of (Batch, out_channel, seq_len)
        :return: size of (Batch, out_channel, seq_len)
        """
        return self.network(x)


class Tcn_Global(nn.Module):
    def __init__(self, num_inputs, num_outputs, kernel_size=3, dropout=0.2):  # k>=d
        super(Tcn_Global, self).__init__()
        layers = []
        num_levels = math.ceil(math.log2((num_inputs - 1) * (2 - 1) / (kernel_size - 1) + 1))
        out_channels = num_outputs
        for i in range(num_levels):
            dilation_size = 2**i
            layers += [
                TemporalCnn(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: size of (Batch, out_channel, seq_len)
        :return: size of (Batch, out_channel, seq_len)
        """
        return self.network(x)


# --- src/dlutils.py (verbatim, DTAAD-relevant subset) ------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.att = None
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # NOTE (portability fix): accept and ignore **kwargs (e.g. `is_causal`) --
        # newer torch's real `nn.TransformerEncoder.forward()` forwards an `is_causal`
        # kwarg to each layer that this custom layer (written pre-`is_causal`) doesn't
        # declare. No architecture/dataflow change; the original repo's four-line body
        # is untouched.
        src2 = self.self_attn(src, src, src)[0]
        self.att = self.self_attn(src, src, src)[1]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


# --- src/models.py (verbatim, DTAAD class) ------------------------------------


# Proposed Model + Tcn_Local + Tcn_Global + Callback + Transformer + MAML
class DTAAD(nn.Module):
    def __init__(self, feats):
        super(DTAAD, self).__init__()
        self.name = "DTAAD"
        self.lr = 0.0001  # inlined from src/constants.py's per-dataset lr_d table
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.l_tcn = Tcn_Local(
            num_outputs=feats, kernel_size=4, dropout=0.2
        )  # K=3&4 (Batch, output_channel, seq_len)
        self.g_tcn = Tcn_Global(
            num_inputs=self.n_window, num_outputs=feats, kernel_size=3, dropout=0.2
        )
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers1 = TransformerEncoderLayer(
            d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )  # (seq_len, Batch, output_channel)
        encoder_layers2 = TransformerEncoderLayer(
            d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder1 = nn.TransformerEncoder(
            encoder_layers1, num_layers=1
        )  # only one layer
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layers2, num_layers=1)
        self.fcn = nn.Linear(feats, feats)
        self.decoder1 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())
        self.decoder2 = nn.Sequential(nn.Linear(self.n_window, 1), nn.Sigmoid())

    def callback(self, src, c):
        src2 = src + c
        g_atts = self.g_tcn(src2)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src2 = self.pos_encoder(src2)
        memory = self.transformer_encoder2(src2)
        return memory

    def forward(self, src):
        l_atts = self.l_tcn(src)
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src1 = self.pos_encoder(src1)
        z1 = self.transformer_encoder1(src1)
        c1 = z1 + self.fcn(z1)
        x1 = self.decoder1(c1.permute(1, 2, 0))
        z2 = self.fcn(self.callback(src, x1))
        c2 = z2 + self.fcn(z2)
        x2 = self.decoder2(c2.permute(1, 2, 0))
        return x1.permute(0, 2, 1), x2.permute(0, 2, 1)  # (Batch, 1, output_channel)


def build_dtaad():
    # feats=8: number of multivariate time-series channels (real repo's smallest
    # documented dataset dimensionality range, e.g. NAB/MBA-scale feature counts).
    return DTAAD(feats=8)


def example_input_dtaad():
    # src: (Batch, feats, n_window) -- the real repo's `backprop()` (main.py) builds
    # windows of shape (bs, n_window, feats) via convert_to_windows(), then does
    # `window = d.permute(0, 2, 1)` before calling `model(window)`, i.e. Tcn_Local/
    # Tcn_Global consume (Batch, out_channel, seq_len) = (Batch, feats, n_window).
    return torch.randn(4, 8, 10)


MENAGERIE_ENTRIES = [
    ("DTAAD", build_dtaad, example_input_dtaad, 2023, MENAGERIE_ZOO),
]
