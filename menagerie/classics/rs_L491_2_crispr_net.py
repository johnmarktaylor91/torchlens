# FAITHFUL PORT of JasonLinjc/CRISPR-Net @ master (original framework: TensorFlow/Keras)
#
# CRISPR-Net (Advanced Science 2020) scores CRISPR sgRNA off-target activity from a
# one-hot on/off-target sequence-pair encoding, handling both mismatches and indels. The
# official repo ships a trained TF1/Keras model (`code/CRISPR_Net.py` just loads
# `code/scoring_models/CRISPR_Net_CIRCLE_elevation_SITE_structure.json` +
# `..._weights.h5` via `keras.models.model_from_json` -- there is no runnable base-torch
# code to vendor), so the Keras functional-model JSON (the model's real, as-shipped
# architecture graph) was parsed layer-by-layer and transcribed here faithfully:
#
#   main_input (1, 24, 7)                                        [NHWC-style: 1 "row", 24
#     |-- conv2d_1: Conv2D(10, kernel=(1,1), pad=same, linear)     off-target positions, 7
#     |-- conv2d_2: Conv2D(10, kernel=(1,2), pad=same, linear)     encoding channels]
#     |-- conv2d_3: Conv2D(10, kernel=(1,3), pad=same, linear)
#     |-- conv2d_4: Conv2D(10, kernel=(1,5), pad=same, linear)
#   each conv branch -> ReLU
#   concat([main_input, relu(conv2d_1..4)], axis=channel) -> 7 + 4*10 = 47 channels
#   reshape to (24, 47)                       [drop the singleton "row" spatial dim]
#   Bidirectional(LSTM(15), return_sequences=True, merge_mode='concat') -> (24, 30)
#   flatten -> Dense(80, relu) -> Dense(20, relu) -> Dropout(0.35) -> Dense(1, sigmoid)
#
# Every layer, kernel size, filter count, and ordering below matches the parsed JSON graph
# exactly; only the framework primitives are swapped for their torch equivalents (Keras
# `same` padding on kernel width 1/2/3/5 stride-1 convs is `padding="same"` in
# torch.nn.Conv2d, supported since torch 1.9).
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class ReluConvBranch(nn.Module):
    """One `conv2d_k` + `activation_k` (ReLU) branch from the parsed Keras graph."""

    def __init__(self, in_channels, out_channels, kernel_w):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_w),
            stride=(1, 1),
            padding="same",
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class CRISPRNet(nn.Module):
    """Faithful port of the CRISPR-Net Keras functional model graph."""

    def __init__(self, in_channels=7, conv_filters=10, lstm_units=15):
        super().__init__()
        self.branch_1x1 = ReluConvBranch(in_channels, conv_filters, kernel_w=1)
        self.branch_1x2 = ReluConvBranch(in_channels, conv_filters, kernel_w=2)
        self.branch_1x3 = ReluConvBranch(in_channels, conv_filters, kernel_w=3)
        self.branch_1x5 = ReluConvBranch(in_channels, conv_filters, kernel_w=5)

        concat_channels = in_channels + 4 * conv_filters  # main_input + 4 conv branches
        self.lstm = nn.LSTM(
            input_size=concat_channels,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        self.dense_1 = nn.Linear(24 * lstm_units * 2, 80)
        self.dense_2 = nn.Linear(80, 20)
        self.dropout = nn.Dropout(p=0.35)
        self.main_output = nn.Linear(20, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels=7, height=1, width=24) -- NCHW torch layout for the
        # (1, 24, 7) NHWC Keras `main_input`.
        c1 = self.branch_1x1(x)
        c2 = self.branch_1x2(x)
        c3 = self.branch_1x3(x)
        c4 = self.branch_1x5(x)

        concatenated = torch.cat([x, c1, c2, c3, c4], dim=1)  # channel-axis concat
        # Drop the singleton height dim and present as (batch, seq_len=24, channels).
        seq = concatenated.squeeze(2).permute(0, 2, 1)

        lstm_out, _ = self.lstm(seq)
        flattened = torch.flatten(lstm_out, start_dim=1)

        hidden = self.relu(self.dense_1(flattened))
        hidden = self.relu(self.dense_2(hidden))
        hidden = self.dropout(hidden)
        return self.sigmoid(self.main_output(hidden))


def build_crispr_net():
    return CRISPRNet()


def example_input_crispr_net():
    # (batch, channels=7, height=1, width=24) matching the Keras (None, 1, 24, 7) NHWC
    # `main_input` batch_input_shape.
    return torch.randn(2, 7, 1, 24)


MENAGERIE_ENTRIES = [
    ("CRISPR-Net", build_crispr_net, example_input_crispr_net, 2020, "ported-pytorch"),
]
