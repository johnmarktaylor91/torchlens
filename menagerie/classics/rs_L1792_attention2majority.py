# FAITHFUL PORT of JoeSu666/Attention2majority @ main (original framework: TensorFlow/Keras)
#
# Source: https://github.com/JoeSu666/Attention2majority/blob/main/milmodels.py
# "Attention2majority: weak multiple instance learning for regenerative kidney grading
# on whole slide images" (MICCAI 2020 workshop). Official implementation targets
# Camelyon16 WSI classification and kidney-grading MIL; the released `milmodels.py`
# is TensorFlow/Keras (`tf.keras.layers`/`tf.keras.models`), not installable against
# this repo's base-lib env, so the gated-attention MIL classifier is transcribed
# faithfully into torch. Every mechanism from the real `gatedattention` +
# `AttMILbinary` Keras classes is preserved 1:1:
#   - gated attention: V(x)=tanh(Linear(x)), U(x)=sigmoid(Linear(x)), energy=V*U,
#     attention logits = Linear(energy) -> softmax over the instance/bag axis,
#     bag embedding = attention-weighted sum of instance features (this is the
#     Ilse et al. 2018 gated-attention-MIL formulation used verbatim in the repo).
#   - classifier head: Dropout(0.2) -> Linear(channels, 1) -> Sigmoid (binary MIL
#     head, matches `AttMILbinary.WC2`).
#
# channels = inputshape[-1] // 2 in the original `AttMILbinary.build()`, i.e. the
# gated-attention hidden width is half the instance feature dimension.

import torch
import torch.nn as nn


class GatedAttention(nn.Module):
    """Port of `gatedattention` (Keras Layer) from milmodels.py."""

    def __init__(self, in_features, channels):
        super().__init__()
        self.channels = channels
        self.V0 = nn.Linear(in_features, channels, bias=False)
        self.U0 = nn.Linear(in_features, channels, bias=False)
        self.Wa0 = nn.Linear(channels, 1, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (1, N, in_features) -- a single bag of N instances (matches the
        # original code's implicit batch_size=1 bag processing: `x = x[0]`).
        x = x[0]
        v0 = self.tanh(self.V0(x))
        u0 = self.sigmoid(self.U0(x))
        energy0 = v0 * u0

        att0 = self.Wa0(energy0).unsqueeze(0)  # (1, N, 1)
        att0 = self.softmax(att0)
        x = x.unsqueeze(0)  # (1, N, in_features)
        # weighted sum over the instance axis: (1,1,in_features)
        hs0 = torch.bmm(att0.transpose(1, 2), x)
        hs = hs0.squeeze(1)  # (1, in_features)
        return att0, hs


class AttMILbinary(nn.Module):
    """Port of `AttMILbinary` (Keras Model) from milmodels.py: gated-attention MIL
    aggregation with a binary (sigmoid) slide-level classification head.

    Note: the gated-attention weighted sum (`hs0 = self.dot([att0, x])` in the
    original) pools over the *original* instance features `x`, not the
    tanh/sigmoid-gated `channels`-width energy -- so `hs`, and therefore the
    `WC2` classifier's input width, is `in_features`, matching Keras's
    shape-inferring `layers.Dense(1)`.
    """

    def __init__(self, in_features):
        super().__init__()
        channels = in_features // 2
        self.gatedattention = GatedAttention(in_features, channels)
        self.dropout = nn.Dropout(p=0.2)
        self.wc2 = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att0, hs = self.gatedattention(x)
        hs = self.dropout(hs)
        s = self.sigmoid(self.wc2(hs))
        return s


MENAGERIE_ZOO = "ported-pytorch"

_IN_FEATURES = 64
_BAG_SIZE = 12


def build_attention2majority():
    return AttMILbinary(in_features=_IN_FEATURES)


def example_input_attention2majority():
    return torch.randn(1, _BAG_SIZE, _IN_FEATURES)


MENAGERIE_ENTRIES = [
    (
        "Attention2Majority",
        build_attention2majority,
        example_input_attention2majority,
        2020,
        MENAGERIE_ZOO,
    ),
]
