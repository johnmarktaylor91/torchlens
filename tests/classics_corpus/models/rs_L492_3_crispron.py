# FAITHFUL PORT of RTH-tools/crispron @ master (original framework: TensorFlow 2 / Keras,
# bin/DeepCRISPRon_train.py)
#
# CRISPRon predicts Cas9 on-target editing efficiency from a 30-mer (4-base one-hot,
# length 30) plus a scalar deltaG (binding free energy) side input. The real repo ships
# only trained `tf.keras` SavedModel binaries (`data/deep_models/best/*.model.best/`); the
# architecture itself lives in `bin/DeepCRISPRon_train.py` as inline `tf.keras` layer
# calls (no importable model class, and the trained artifacts are TF SavedModel protobufs,
# not loadable from a torch-only env). This module transcribes that architecture
# faithfully into self-contained torch: three parallel Conv1D branches (kernel widths
# 3/5/7, channel counts 100/70/40) over the one-hot 30-mer, each ReLU -> Dropout ->
# AveragePooling1D(2) -> flatten, concatenated, fed through a first dense+dropout layer,
# concatenated with the raw deltaG scalar, then two more dense+dropout layers, ending in a
# single-unit regression head (`Dense(1, name="output")`).
import torch
import torch.nn.functional as f
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"

SEQ_LEN = 30
DEPTH = 4


class CRISPRon(nn.Module):
    """Faithful port of the Keras model built in bin/DeepCRISPRon_train.py."""

    def __init__(self, seq_len: int = SEQ_LEN, depth: int = DEPTH):
        super().__init__()
        self.seq_len = seq_len

        # Conv1D(100, 3, activation='relu', name="conv_3")
        self.conv3 = nn.Conv1d(depth, 100, kernel_size=3, padding="same")
        self.drop3 = nn.Dropout(0.3)
        self.pool3 = nn.AvgPool1d(kernel_size=2, ceil_mode=True)

        # Conv1D(70, 5, activation='relu', name="conv_5")
        self.conv5 = nn.Conv1d(depth, 70, kernel_size=5, padding="same")
        self.drop5 = nn.Dropout(0.3)
        self.pool5 = nn.AvgPool1d(kernel_size=2, ceil_mode=True)

        # Conv1D(40, 7, activation='relu', name="conv_7")
        self.conv7 = nn.Conv1d(depth, 40, kernel_size=7, padding="same")
        self.drop7 = nn.Dropout(0.3)
        self.pool7 = nn.AvgPool1d(kernel_size=2, ceil_mode=True)

        pooled_len = -(-seq_len // 2)  # ceil_mode AvgPool1d(2) output length
        concat_dim = pooled_len * (100 + 70 + 40)

        # Dense(80, activation='relu', name="dense_0")
        self.dense0 = nn.Linear(concat_dim, 80)
        self.drop_d0 = nn.Dropout(0.3)

        # Dense(80, activation='relu', name="dense_1") over concat([dense0_out, deltaG])
        self.dense1 = nn.Linear(80 + 1, 80)
        self.drop_d1 = nn.Dropout(0.3)

        # Dense(60, activation='relu', name="dense_2")
        self.dense2 = nn.Linear(80, 60)
        self.drop_d2 = nn.Dropout(0.3)

        # Dense(1, name="output")
        self.output_layer = nn.Linear(60, 1)

    def forward(self, seq_onehot, delta_g):
        # seq_onehot: (batch, seq_len, depth) one-hot 30-mer; delta_g: (batch, 1)
        x = seq_onehot.transpose(1, 2)  # (batch, depth, seq_len) for Conv1d

        c3 = self.pool3(self.drop3(f.relu(self.conv3(x)))).flatten(1)
        c5 = self.pool5(self.drop5(f.relu(self.conv5(x)))).flatten(1)
        c7 = self.pool7(self.drop7(f.relu(self.conv7(x)))).flatten(1)

        concat = torch.cat([c3, c5, c7], dim=-1)
        dense0_out = self.drop_d0(f.relu(self.dense0(concat)))

        concat1 = torch.cat([dense0_out, delta_g], dim=-1)
        dense1_out = self.drop_d1(f.relu(self.dense1(concat1)))
        dense2_out = self.drop_d2(f.relu(self.dense2(dense1_out)))
        return self.output_layer(dense2_out)


def build_crispron():
    return CRISPRon()


def example_input_crispron():
    return (torch.randn(2, SEQ_LEN, DEPTH), torch.randn(2, 1))


MENAGERIE_ENTRIES = [
    ("CRISPRon", build_crispron, example_input_crispron, 2021, "REIMPLEMENT"),
]
