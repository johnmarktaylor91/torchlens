# FAITHFUL PORT of ohlerlab/DeepRiPe @ master (original framework: Keras 1.x/TF1.x, via
# Scripts/models.py's `create_model4`)
#
# DeepRiPe (dual-branch CNN + Bidirectional GRU): a two-branch 1D-convolutional encoder over
# an RNA sequence one-hot track ("left": Conv1D(90,k=7) -> MaxPool(4,stride=2) -> Dropout) and
# a region-type one-hot track ("right": Conv1D(90,k=7) -> MaxPool(10,stride=5) -> Dropout),
# concatenated along the sequence axis, fed through a Bidirectional GRU(60, return_sequences),
# flattened, and read out through Dense(250, relu) -> Dense(num_task, sigmoid) for multi-task
# RBP-binding-site classification (Ghanbari & Ohler, Genome Research 2020). The original repo
# ships only in ancient Keras submodule paths (keras.layers.merge/normalization/wrappers,
# keras.layers.recurrent, tf.set_random_seed) that predate the modern tf.keras namespace and
# cannot be installed cleanly alongside torch/torchvision/etc. in this environment, so this is
# a faithful architectural transcription of `create_model4` (Scripts/models.py) into base-env
# torch: every layer (Conv1D->Conv1d, MaxPooling1D->MaxPool1d, the sequence-axis concat, the
# Bidirectional(GRU(60, return_sequences=True))->nn.GRU(bidirectional=True), Flatten, the two
# Dense heads with relu/sigmoid) is preserved, including Keras's channels-last convention
# (samples, steps, channels) handled by permuting to channels-first for the torch Conv1d/GRU
# calls and back. Dropout is included as an inert (eval-mode) nn.Dropout to match Keras's
# `Dropout(0.25)` layers.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class DeepRiPeModel4(nn.Module):
    """Faithful port of DeepRiPe's `create_model4` (dual-branch CNN + BiGRU)."""

    def __init__(self, num_task, input_len_l, input_len_r):
        super().__init__()
        left_dim = 4
        right_dim = 4
        num_units = 60

        nb_f_l, f_len_l, p_len_l, s_l = 90, 7, 4, 2
        nb_f_r, f_len_r, p_len_r, s_r = 90, 7, 10, 5

        # left branch (sequence track): Conv1D -> MaxPool1D -> Dropout
        self.left_conv1 = nn.Conv1d(in_channels=left_dim, out_channels=nb_f_l, kernel_size=f_len_l)
        self.left_pool1 = nn.MaxPool1d(kernel_size=p_len_l, stride=s_l)
        self.left_drop1 = nn.Dropout(0.25)

        # right branch (region-type track): Conv1D -> MaxPool1D -> Dropout
        self.right_conv1 = nn.Conv1d(
            in_channels=right_dim, out_channels=nb_f_r, kernel_size=f_len_r
        )
        self.right_pool1 = nn.MaxPool1d(kernel_size=p_len_r, stride=s_r)
        self.right_drop1 = nn.Dropout(0.25)

        # concat(left, right, axis=sequence) -> Bidirectional GRU(60, return_sequences=True)
        self.gru = nn.GRU(
            input_size=nb_f_l, hidden_size=num_units, batch_first=True, bidirectional=True
        )

        left_out_len = (input_len_l - f_len_l + 1 - p_len_l) // s_l + 1
        right_out_len = (input_len_r - f_len_r + 1 - p_len_r) // s_r + 1
        merged_len = left_out_len + right_out_len
        flat_dim = merged_len * (num_units * 2)

        self.hidden1 = nn.Linear(flat_dim, 250)
        self.output_layer = nn.Linear(250, num_task)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, left_input, right_input):
        # Keras Input shape is (batch, steps, channels); Conv1d wants (batch, channels, steps).
        left = left_input.transpose(1, 2)
        right = right_input.transpose(1, 2)

        left = self.relu(self.left_conv1(left))
        left = self.left_pool1(left)
        left = self.left_drop1(left)

        right = self.relu(self.right_conv1(right))
        right = self.right_pool1(right)
        right = self.right_drop1(right)

        # concat along the "sequence"/steps axis (Keras axis=-2, channels-last) ==
        # concat along the length dim after transposing back to (batch, steps, channels).
        left = left.transpose(1, 2)
        right = right.transpose(1, 2)
        merged = torch.cat([left, right], dim=1)

        gru_out, _ = self.gru(merged)
        flat = gru_out.reshape(gru_out.size(0), -1)
        hidden = self.relu(self.hidden1(flat))
        out = self.sigmoid(self.output_layer(hidden))
        return out


def build_deepripe():
    # Shrunk sequence/region lengths for a fast trace-sized build; num_task=5 RBPs.
    return DeepRiPeModel4(num_task=5, input_len_l=150, input_len_r=100)


def example_input_deepripe():
    left_input = torch.rand(2, 150, 4)  # (batch, seq_len, one-hot RNA bases)
    right_input = torch.rand(2, 100, 4)  # (batch, region_len, one-hot region types)
    return (left_input, right_input)


MENAGERIE_ENTRIES = [
    ("DeepRiPe", build_deepripe, example_input_deepripe, 2020, "SOURCE_AVAILABLE"),
]
