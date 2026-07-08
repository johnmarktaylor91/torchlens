# FAITHFUL PORT of haotianteng/Chiron @ 56a5362b1cdb (original framework: TensorFlow 1.x)
# https://github.com/haotianteng/Chiron
# https://raw.githubusercontent.com/haotianteng/Chiron/master/chiron/cnn.py
# https://raw.githubusercontent.com/haotianteng/Chiron/master/chiron/rnn.py
# https://raw.githubusercontent.com/haotianteng/Chiron/master/chiron/chiron_model.py
#
# Teng, Zhang, Peters, Zhang, Zhang, Xu, Lu, Zhang, Wang, Vella, Coin 2018
# (GigaScience) "Chiron: translating nanopore raw signal directly into
# nucleotide sequence using deep learning" -- raw nanopore-current basecaller.
# Architecture (default `dna_model1` + 3-layer stacked bidirectional-LSTM
# `rnn_layers`, exactly as wired together in `chiron_model.inference` for
# `cnn_config={'model': 'dna_model1'}` / `rnn_config={'layer_num': 3,
# 'hidden_num': 100, 'cell_type': 'LSTM'}`, the package's shipped defaults in
# `chiron_model.read_config`):
#   1. Raw 1-D signal reshaped to a "2D" tensor of shape
#      [batch, 1, time, 1] (TF's NHWC 4D-conv convention for what is really a
#      1D signal -- width-only convolutions, height axis pinned to 1).
#   2. `DNA_model1`: 3 stacked bottleneck residual blocks (`residual_layer`),
#      each a 1x1 down-conv -> 1xk conv -> 1x1 up-conv branch (BN+ReLU after
#      every conv except the last) added to a 1x1 shortcut branch (BN only on
#      block 1, no BN on blocks 2-3, matching the source's `i_bn` flags), then
#      a final ReLU -- fixed at out_channel=256 and kernel width k=3 for every
#      block (the source's declared defaults). `simple_global_bn` (per-channel
#      batch statistics over N,H,W with learnable scale/offset) is ported as
#      `nn.BatchNorm2d`, which is functionally equivalent under training-mode
#      batch statistics.
#   3. CNN feature map [batch, time, 256] fed to a 3-layer stacked
#      bidirectional LSTM (`rnn_layers`, hidden_num=100 per direction), ported
#      via `nn.LSTM(bidirectional=True, num_layers=3)` (PyTorch's stacked
#      bidirectional LSTM is the direct equivalent of TF's
#      `stack_bidirectional_dynamic_rnn` with per-layer independent fw/bw
#      cells).
#   4. `rnn_fnn_layer`: element-wise-weighted sum-then-bias over the two
#      direction outputs (`weight_out`/`biases_out`, shape [2, hidden_num]),
#      NOT a concat+matmul -- ported verbatim as
#      `(stacked_dirs * weight_out).sum(dim=2) + biases_out`.
#   5. A final linear class head (`weight_class`/`bias_class`) projecting
#      hidden_num -> class_n (5: A,G,C,T,<ctc-blank>).
# `getcnnlogit` (the `rnn_layer_num == 0` fallback head) is not exercised
# here since the default config always routes through the RNN branch. The CTC
# loss / decoding (`chiron_model.loss` / `prediction`) is training/inference
# post-processing outside the traced network and is not part of this module.

import torch
import torch.nn as nn


class ResidualLayer(nn.Module):
    """Bottleneck residual block, ported from `cnn.residual_layer`."""

    def __init__(self, in_channel, out_channel, k=3, i_bn=False):
        super().__init__()
        self.branch1_conv = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), bias=False)
        self.branch1_bn = nn.BatchNorm2d(out_channel) if i_bn else None

        self.branch2_conv2a = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), bias=False)
        self.branch2_bn2a = nn.BatchNorm2d(out_channel)
        self.branch2_conv2b = nn.Conv2d(
            out_channel, out_channel, kernel_size=(1, k), padding=(0, k // 2), bias=False
        )
        self.branch2_bn2b = nn.BatchNorm2d(out_channel)
        self.branch2_conv2c = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 1), bias=False)
        self.branch2_bn2c = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        indata_cp = self.branch1_conv(x)
        if self.branch1_bn is not None:
            indata_cp = self.branch1_bn(indata_cp)

        out = torch.relu(self.branch2_bn2a(self.branch2_conv2a(x)))
        out = torch.relu(self.branch2_bn2b(self.branch2_conv2b(out)))
        out = self.branch2_bn2c(self.branch2_conv2c(out))

        return torch.relu(indata_cp + out)


class DNAModel1(nn.Module):
    """Ported from `cnn.DNA_model1`: 3 stacked residual layers, 256 channels."""

    def __init__(self, in_channel=1, out_channel=256):
        super().__init__()
        self.res_layer1 = ResidualLayer(in_channel, out_channel, k=3, i_bn=True)
        self.res_layer2 = ResidualLayer(out_channel, out_channel, k=3, i_bn=False)
        self.res_layer3 = ResidualLayer(out_channel, out_channel, k=3, i_bn=False)

    def forward(self, x):
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        return x


class RNNFNNLayer(nn.Module):
    """Ported from `rnn.rnn_layers`'s `rnn_fnn_layer` block: per-direction
    weighted sum (not concat) followed by a linear class head."""

    def __init__(self, hidden_num, class_n):
        super().__init__()
        self.weight_out = nn.Parameter(torch.empty(2, hidden_num))
        self.bias_out = nn.Parameter(torch.zeros(hidden_num))
        nn.init.trunc_normal_(self.weight_out, std=(2.0 / (2 * hidden_num)) ** 0.5)
        self.class_fc = nn.Linear(hidden_num, class_n)
        nn.init.trunc_normal_(self.class_fc.weight, std=(2.0 / hidden_num) ** 0.5)
        nn.init.zeros_(self.class_fc.bias)

    def forward(self, lasth):
        # lasth: [batch, time, 2 * hidden_num] -> [batch, time, 2, hidden_num]
        batch, time_, two_hidden = lasth.shape
        hidden_num = two_hidden // 2
        lasth_rs = lasth.view(batch, time_, 2, hidden_num)
        lasth_out = (lasth_rs * self.weight_out).sum(dim=2) + self.bias_out
        return self.class_fc(lasth_out)


class ChironDNAModel(nn.Module):
    """Ported from `chiron_model.inference` wired for
    `cnn_config={'model': 'dna_model1'}`,
    `rnn_config={'layer_num': 3, 'hidden_num': 100, 'cell_type': 'LSTM'}`."""

    def __init__(self, cnn_channel=256, hidden_num=100, rnn_layer_num=3, class_n=5):
        super().__init__()
        self.cnn = DNAModel1(in_channel=1, out_channel=cnn_channel)
        self.rnn = nn.LSTM(
            input_size=cnn_channel,
            hidden_size=hidden_num,
            num_layers=rnn_layer_num,
            batch_first=True,
            bidirectional=True,
        )
        self.rnn_fnn = RNNFNNLayer(hidden_num, class_n)

    def forward(self, signal):
        # signal: [batch, max_time] raw nanopore current samples
        batch, max_time = signal.shape
        net = signal.view(batch, 1, 1, max_time)  # NCHW: height axis pinned to 1
        cnn_feature = self.cnn(net)  # [batch, 256, 1, max_time]
        cnn_feature = cnn_feature.squeeze(2).transpose(1, 2)  # [batch, max_time, 256]
        lasth, _ = self.rnn(cnn_feature)  # [batch, max_time, 2 * hidden_num]
        logits = self.rnn_fnn(lasth)  # [batch, max_time, class_n]
        return logits


def build_chiron():
    return ChironDNAModel(cnn_channel=32, hidden_num=16, rnn_layer_num=3, class_n=5)


def example_input_chiron():
    return (torch.randn(2, 64),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("Chiron", "build_chiron", "example_input_chiron", 2018, "ported"),
]
