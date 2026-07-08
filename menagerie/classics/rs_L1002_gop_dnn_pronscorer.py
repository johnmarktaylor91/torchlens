# SOURCE: vendored from MarceloSancinetti/epa-gop-pykaldi @ 7fc307141eb5e04a647f8fe099b1c8bd39b98df4
# (src/pytorch_models/FTDNN.py, src/pytorch_models/FTDNNPronscorer.py)
#
# GOP-DNN pronunciation scoring: a Factorized Time-Delay Neural Network (TDNN-F,
# Povey et al. 2018 "Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural
# Networks") acoustic backbone -- an `InputLayer` (LDA + kernel projection over
# stacked MFCC context + i-vectors) feeding 17 factorized-TDNN layers (each a
# semi-orthogonal low-rank bottleneck + affine + ReLU + BatchNorm, with a learned
# 0.75-scaled skip-connection carried across layers via
# `sum_outputs_and_feed_to_layer`) -- topped with a sigmoid `OutputLayer` producing
# a per-phone pronunciation (goodness-of-pronunciation) score. This is the actual
# PyTorch model used for phone-level GOP-DNN pronunciation scoring in this line of
# CAPT (computer-assisted pronunciation training) research (the queue candidate's
# upstream `jimbozhang/kaldi-gop` is pure Kaldi/C++ GMM-GOP with no PyTorch model at
# all; `epa-gop-pykaldi` is the pykaldi-based GOP-DNN successor line -- see e.g.
# Sancinetti et al., "A pronunciation scoring system based on Wav2Vec2.0 and DNN
# acoustic features" -- and provides the real convertible-from-Kaldi-chain-model
# PyTorch acoustic model this candidate's own note ("kaldi-gop"/GOP-DNN) points at).
# `FTDNNLayer`/`InputLayer`/`sum_outputs_and_feed_to_layer`/`FTDNN`/`OutputLayer`/
# `FTDNNPronscorer` below are the REAL model code from `FTDNN.py` +
# `FTDNNPronscorer.py`, copied verbatim (only base-lib deps: torch, torch.nn,
# torch.nn.functional -- all installed).
#
# Dropped: `from IPython import embed` (an unused interactive-debug import present
# in both source files, never called) -- not part of the model architecture.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class FTDNNLayer(nn.Module):
    def __init__(
        self,
        semi_orth_in_dim,
        semi_orth_out_dim,
        affine_in_dim,
        out_dim,
        time_offset,
        dropout_p=0,
        device="cpu",
    ):
        """
        3 stage factorised TDNN http://danielpovey.com/files/2018_interspeech_tdnnf.pdf
        """
        super(FTDNNLayer, self).__init__()
        self.semi_orth_in_dim = semi_orth_in_dim
        self.semi_orth_out_dim = semi_orth_out_dim
        self.affine_in_dim = affine_in_dim
        self.out_dim = out_dim
        self.time_offset = time_offset
        self.dropout_p = dropout_p
        self.device = device

        self.sorth = nn.Linear(self.semi_orth_in_dim, self.semi_orth_out_dim, bias=False)
        self.affine = nn.Linear(self.affine_in_dim, self.out_dim, bias=True)
        self.nl = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=0.001)
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        time_offset = self.time_offset
        if time_offset != 0:
            padding = x[:, 0, :][:, None, :]
            xd = torch.cat([padding] * time_offset + [x], axis=1)
            xd = xd[:, :-time_offset, :]
            x = torch.cat([xd, x], axis=2)
        x = self.sorth(x)
        if time_offset != 0:
            padding = x[:, -1, :][:, None, :]
            padding = torch.zeros(padding.shape)
            if self.device == "cuda":
                padding = padding.cuda()
            xd = torch.cat([x] + [padding] * time_offset, axis=1)
            xd = xd[:, time_offset:, :]
            x = torch.cat([x, xd], axis=2)
        x = self.affine(x)
        x = self.nl(x)
        x = x.transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        x = self.dropout(x)
        return x


class InputLayer(nn.Module):
    def __init__(self, input_dim=220, output_dim=1536, dropout_p=0):
        super(InputLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.lda = nn.Linear(self.input_dim, self.input_dim)
        self.kernel = nn.Linear(self.input_dim, self.output_dim)

        self.nonlinearity = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim, affine=False, eps=0.001)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        mfccs = x[:, :, :40]
        ivectors = x[:, :, -100:]
        padding_first = mfccs[:, 0, :][:, None, :]
        padding_last = mfccs[:, -1, :][:, None, :]
        context_first = torch.cat([padding_first, mfccs[:, :-1, :]], axis=1)
        context_last = torch.cat([mfccs[:, 1:, :], padding_last], axis=1)
        x = torch.cat([context_first, mfccs, context_last, ivectors], axis=2)
        x = self.lda(x)
        x = self.kernel(x)
        x = self.nonlinearity(x)

        x = x.transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        x = self.drop(x)
        return x


def sum_outputs_and_feed_to_layer(x, x_2, layer):
    x_3 = x * 0.75 + x_2
    x = x_3
    x_2 = layer(x_3)
    return x, x_2


class FTDNN(nn.Module):
    def __init__(self, in_dim=220, batchnorm=None, dropout_p=0, device_name="cpu"):
        super(FTDNN, self).__init__()

        self.layer01 = InputLayer(input_dim=in_dim, output_dim=1536)
        self.layer02 = FTDNNLayer(3072, 160, 320, 1536, 1, dropout_p=dropout_p, device=device_name)
        self.layer03 = FTDNNLayer(3072, 160, 320, 1536, 1, dropout_p=dropout_p, device=device_name)
        self.layer04 = FTDNNLayer(3072, 160, 320, 1536, 1, dropout_p=dropout_p, device=device_name)
        self.layer05 = FTDNNLayer(1536, 160, 160, 1536, 0, dropout_p=dropout_p, device=device_name)
        self.layer06 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer07 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer08 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer09 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer10 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer11 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer12 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer13 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer14 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer15 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer16 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer17 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer18 = nn.Linear(1536, 256, bias=False)  # This is the prefinal-l layer

    def forward(self, x):
        """
        Input must be (batch_size, seq_len, in_dim)
        """
        x = self.layer01(x)
        x_2 = self.layer02(x)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer03)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer04)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer05)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer06)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer07)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer08)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer09)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer10)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer11)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer12)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer13)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer14)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer15)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer16)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer17)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer18)
        return x_2


class PronscorerOutputLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False):
        super(PronscorerOutputLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bn = use_bn

        if use_bn:
            self.bn = nn.BatchNorm1d(self.in_dim, affine=False)
        self.linear = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.nl = nn.Sigmoid()

    def forward(self, x):
        if self.use_bn:
            x = x.transpose(1, 2)
            x = self.bn(x).transpose(1, 2)
        x = self.linear(x)
        return x


class FTDNNPronscorer(nn.Module):
    def __init__(self, out_dim=40, batchnorm=None, dropout_p=0, device_name="cpu"):
        super(FTDNNPronscorer, self).__init__()

        use_final_bn = False
        if batchnorm in ["final", "last", "firstlast"]:
            use_final_bn = True

        self.ftdnn = FTDNN(batchnorm=batchnorm, dropout_p=dropout_p, device_name=device_name)
        self.output_layer = PronscorerOutputLayer(256, out_dim, use_bn=use_final_bn)

    def forward(self, x):
        """
        Input must be (batch_size, seq_len, in_dim)
        """
        x = self.ftdnn(x)
        x = self.output_layer(x)

        return x


# ---------------------------------------------------------------------------
# staging build/example (tiny sizes for tracing)
#
# NOTE: `FTDNN`'s per-layer semi_orth/affine dims (3072, 160, 320, 1536, ...) are
# architectural constants from the real source (they encode the specific
# low-rank-factorization + splicing-context sizes of the trained Kaldi chain model
# this class was converted from) -- not free hyperparameters, so they are NOT
# shrunk here. `in_dim=220` (40 MFCC + 40*2 spliced context + 100 i-vector, per
# InputLayer.forward's slicing) and `seq_len` are the only free dims for tracing.
# ---------------------------------------------------------------------------


def build_gop_dnn_pronscorer():
    model = FTDNNPronscorer(out_dim=40, batchnorm=None, dropout_p=0.0, device_name="cpu")
    model.eval()
    return model


def example_input_gop_dnn_pronscorer():
    batch, seq_len, in_dim = 2, 12, 220
    return torch.randn(batch, seq_len, in_dim)


MENAGERIE_ENTRIES = [
    (
        "GOP-DNN Pronunciation Scoring (FTDNN)",
        build_gop_dnn_pronscorer,
        example_input_gop_dnn_pronscorer,
        2020,
        "vendored-pytorch",
    ),
]
