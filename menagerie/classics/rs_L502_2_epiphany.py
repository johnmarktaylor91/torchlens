# SOURCE: vendored from arnavmdas/epiphany @ main (epiphany/model_5kb.py)
#
# Epiphany (Yang, Das, et al., Cell Systems 2023) predicts Hi-C contact maps
# (chromatin 3D structure) from 1D epigenomic tracks (ChIP-seq / ATAC-seq
# coverage) using a CNN + bidirectional-LSTM encoder trained adversarially
# against a CNN discriminator. This vendors the real `ConvBlock` and `Net`
# (the 5kb-resolution encoder; repo README: "model_5kb.py: Epiphany model
# (for 5kb Hi-C map prediction), including both the encoder and the
# discriminator") classes verbatim from epiphany/model_5kb.py: a 4-stage
# 1D-conv stack (kernel widths 17/7/5/5 with maxpools 4/4/4/adaptive) that
# extracts per-window features via `torch.as_strided` sliding windows over
# the input epigenomic tracks, followed by 2 stacked bidirectional LSTMs
# (with a residual add between them) and a 2-layer dense head. Only the
# `loss()` training-time method (needs label tensors) is dropped; the
# forward-pass architecture (conv stack -> as_strided windowing -> BiLSTM
# stack -> dense head) is unchanged.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_width, stride=1, pool_size=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_width, stride=1)
        self.act = nn.ReLU()
        self.pool_size = pool_size

        if pool_size > 0:
            self.pool = nn.MaxPool1d(self.pool_size, self.pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        if self.pool_size > 0:
            x = self.pool(x)

        return x


class Net(nn.Module):
    def __init__(self, num_layers=1, input_channels=30, window_size=12000):
        super(Net, self).__init__()
        self.input_channels = input_channels
        self.window_size = window_size

        self.conv1 = ConvBlock(
            in_channels=self.input_channels, out_channels=70, kernel_width=17, stride=1, pool_size=4
        )
        self.do1 = nn.Dropout(p=0.1)
        self.conv2 = ConvBlock(
            in_channels=70, out_channels=90, kernel_width=7, stride=1, pool_size=4
        )
        self.do2 = nn.Dropout(p=0.1)
        self.conv3 = ConvBlock(
            in_channels=90, out_channels=70, kernel_width=5, stride=1, pool_size=4
        )
        self.do3 = nn.Dropout(p=0.1)
        self.conv4 = ConvBlock(in_channels=70, out_channels=20, kernel_width=5, stride=1)
        self.pool = nn.AdaptiveMaxPool1d(900 // 20)
        self.do4 = nn.Dropout(p=0.1)

        self.rnn1 = nn.LSTM(
            input_size=900,
            hidden_size=2400,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn2 = nn.LSTM(
            input_size=4800,
            hidden_size=2400,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(4800, 1200)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(1200, 200)  # CHANGED 100 -> 200

    def forward(self, x, hidden_state=None, seq_length=200):
        assert x.shape[0] == self.input_channels
        x = torch.as_strided(
            x, (seq_length, self.input_channels, self.window_size), (50, x.shape[1], 1)
        )
        x = self.conv1(x)
        x = self.do1(x)
        x = self.conv2(x)
        x = self.do2(x)
        x = self.conv3(x)
        x = self.do3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.do4(x)
        x = x.view(1, seq_length, x.shape[1] * x.shape[2])
        res1, hidden_state = self.rnn1(x, None)
        res2, hidden_state = self.rnn2(res1, None)
        res2 = res2 + res1
        x = self.fc(res2)
        x = self.act(x)
        x = self.fc2(x)
        return x, hidden_state


# ---------------------------------------------------------------------------
# menagerie staging entry point
# ---------------------------------------------------------------------------
# Real usage: input_channels=30 epigenomic tracks, window_size=12000bp,
# seq_length=200 sliding windows (a full chromosome-scale Hi-C map row).
# Use a scaled-down window_size/input_channels/seq_length that empirically
# preserves the conv+pool stack's flatten dim (900) the LSTM stack expects
# (measured: window_size=600 -> conv1..conv4+adaptive-pool flatten dim==900,
# identical to the real window_size=12000 case since AdaptiveMaxPool1d always
# pins the final pooled length to 900//20==45 regardless of input length).
_TINY_INPUT_CHANNELS = 6
_TINY_WINDOW_SIZE = 600
_TINY_SEQ_LENGTH = 4
_STRIDE = 50
_TOTAL_LEN = (_TINY_SEQ_LENGTH - 1) * _STRIDE + _TINY_WINDOW_SIZE


def build_epiphany():
    return Net(num_layers=1, input_channels=_TINY_INPUT_CHANNELS, window_size=_TINY_WINDOW_SIZE)


def example_input_epiphany():
    x = torch.randn(_TINY_INPUT_CHANNELS, _TOTAL_LEN)
    return (x, None, _TINY_SEQ_LENGTH)


MENAGERIE_ENTRIES = [
    ("Epiphany", build_epiphany, example_input_epiphany, 2023, "SOURCE_AVAILABLE"),
]
