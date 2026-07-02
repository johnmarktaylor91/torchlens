# SOURCE: vendored from jzi040941/PercepNet @ 8ffae4337d23f920176ac2a7426e84610fa338ab
# https://raw.githubusercontent.com/jzi040941/PercepNet/8ffae4337d23f920176ac2a7426e84610fa338ab/rnn_train.py
#
# PercepNet (Valin & Isik, Interspeech 2020, "A Perceptually-Motivated Approach for
# Low-Complexity, Real-Time Enhancement of Fullband Speech") -- unofficial PyTorch
# training-side reimplementation by Seonghun Noh (jzi040941/PercepNet), used to train
# weights that are then exported (`dump_percepnet.py`) into the repo's real-time C++
# inference engine (`src/rnn.cpp`, `src/nnet.cpp`). The PyTorch side is the actual
# trainable network: 70-dim per-frame features (ERB gain-band + pitch-correlation
# features) -> a dense front-end, a strided/padded Conv1d stack aligned frame-for-frame
# with the C++ inference DNN, three stacked single-layer GRUs, then two parallel heads
# -- an 18-band "gain band" (gb) sigmoid head fed by a concat of every intermediate
# layer's output, and a "correlation band"/ratio (rb) sigmoid head fed by a second GRU
# over [gru3_out, convout] -- concatenated into the final 68-d (34+34) per-frame output
# (perceptual gains and pitch-filter strengths per ERB band as PercepNet's loss target).
#
# `PercepNet` is transcribed verbatim from `rnn_train.py` (the actual nn.Module the
# repo trains and later dumps into C++ header format via `dump_percepnet.py`). No
# architectural changes were made -- every Linear/Conv1d/GRU layer, its arguments,
# every permute/slice/concat in forward() is unchanged. Only the module-level test/CLI
# code (`test()`, `CustomLoss`, `Trainer`, dataset classes, h5py/tensorboardX/argparse
# plumbing) is dropped since it is training-only, not part of the traced architecture;
# the module's own `test()` at the bottom of `rnn_train.py` establishes the (batch,
# time, input_dim=70) input convention used here for `example_input_percepnet`.

import torch
import torch.nn as nn


class PercepNet(nn.Module):
    def __init__(self, input_dim=70):
        super(PercepNet, self).__init__()

        self.fc = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 512, 5, stride=1, padding=4), nn.ReLU()
        )  # padding for align with c++ dnn
        self.conv2 = nn.Sequential(nn.Conv1d(512, 512, 3, stride=1, padding=2), nn.Tanh())
        self.gru1 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru2 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru3 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru_gb = nn.GRU(512, 512, 1, batch_first=True)
        self.gru_rb = nn.GRU(1024, 128, 1, batch_first=True)
        self.fc_gb = nn.Sequential(nn.Linear(512 * 5, 34), nn.Sigmoid())
        self.fc_rb = nn.Sequential(nn.Linear(128, 34), nn.Sigmoid())

    def forward(self, x):
        x = self.fc(x)
        x = x.permute([0, 2, 1])  # B, D, T
        x = self.conv1(x)
        x = x[:, :, :-4]
        convout = self.conv2(x)
        convout = convout[:, :, :-2]  # align with c++ dnn
        convout = convout.permute([0, 2, 1])  # B, T, D

        gru1_out, gru1_state = self.gru1(convout)
        gru2_out, gru2_state = self.gru2(gru1_out)
        gru3_out, gru3_state = self.gru3(gru2_out)
        gru_gb_out, gru_gb_state = self.gru_gb(gru3_out)
        concat_gb_layer = torch.cat((convout, gru1_out, gru2_out, gru3_out, gru_gb_out), -1)
        gb = self.fc_gb(concat_gb_layer)

        # concat rb need fix
        concat_rb_layer = torch.cat((gru3_out, convout), -1)
        rnn_rb_out, gru_rb_state = self.gru_rb(concat_rb_layer)
        rb = self.fc_rb(rnn_rb_out)

        output = torch.cat((gb, rb), -1)
        return output


def build_percepnet():
    torch.manual_seed(0)
    model = PercepNet(input_dim=70)
    model.eval()
    return model


def example_input_percepnet():
    torch.manual_seed(0)
    # Matches the repo's own rnn_train.py::test() convention: (batch, time, input_dim).
    return torch.randn(2, 8, 70)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("PercepNet", "build_percepnet", "example_input_percepnet", 2020, MENAGERIE_ZOO),
]
