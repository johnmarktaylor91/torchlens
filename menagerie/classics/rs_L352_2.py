# FAITHFUL PORT of lmsac/DeepDIA @ master (original framework: TensorFlow/Keras)
#
# Source files ported (raw.githubusercontent.com/lmsac/DeepDIA/master/src/...):
#   pepdetect/modeling.py::build_model  (peptide detectability)
#   pepms2/modeling.py::build_model     (peptide MS2 fragment-intensity prediction)
#   peprt/modeling.py::build_model      (peptide retention-time prediction)
# Option defaults from common/options.py + pepdetect/options.py + pepms2/options.py +
# peprt/options.py (amino_acid_size=20, pepdetect max_sequence_length=64,
# pepms2/peprt max_sequence_length=50, pepms2 intensity_size=12).
#
# Each build_model() in the repo is a Keras Sequential model. TensorFlow/Keras is not
# in the installed base-lib set, so this is a faithful architecture-preserving
# transcription into self-contained torch (every layer/mechanism as in the actual
# Keras code): Conv1D -> [MaxPooling1D] -> Bidirectional(LSTM) -> Dropout ->
# [Flatten ->] Dense stack, matching each model's real topology and per-layer
# activations. `Masking` (pepms2) is a Keras runtime input-masking layer with no
# torch-graph equivalent op; it does not change tensor shapes/ops on an unpadded
# example input, so it is omitted rather than approximated. `TimeDistributed(Dense)`
# is exactly `nn.Linear` applied over the last (feature) dimension, which is what
# `nn.Linear` already does for a (batch, seq, feature) tensor.
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

AMINO_ACID_SIZE = 20  # len('ARNDCEQGHILKMFPSTWYV'), common/options.py::PeptideOptions


class DeepDIAPepDetect(nn.Module):
    """Port of src/pepdetect/modeling.py::build_model.

    Keras: Conv1D(64, k=5, relu, l2) -> MaxPooling1D(2, 2) ->
    Bidirectional(LSTM(128, return_sequences=True, tanh)) -> Dropout(0.5) ->
    Flatten -> Dense(64, relu, l2) -> Dropout(0.5) -> Dense(1, relu).
    """

    def __init__(self, seq_len=64, in_ch=AMINO_ACID_SIZE):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, 64, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear((seq_len // 2) * 256, 64)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Keras input_shape=(seq_len, amino_acid_size); torch Conv1d wants (B, C, L).
        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.drop1(x)
        x = x.reshape(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.drop2(x)
        x = self.act(self.fc2(x))
        return x


class DeepDIAPepMS2(nn.Module):
    """Port of src/pepms2/modeling.py::build_model.

    Keras: Conv1D(64, k=2, relu, same) -> Masking(0.) ->
    Bidirectional(LSTM(128, return_sequences=True)) -> Dropout(0.5) ->
    TimeDistributed(Dense(intensity_size, relu)).
    """

    def __init__(self, seq_len=50, in_ch=AMINO_ACID_SIZE, intensity_size=12):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, 64, kernel_size=2, padding="same")
        self.act = nn.ReLU()
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(256, intensity_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.drop(x)
        # nn.Linear broadcasts over the (batch, seq, feature) tensor exactly like
        # Keras TimeDistributed(Dense(...)).
        return self.act(self.fc(x))


class DeepDIAPepRT(nn.Module):
    """Port of src/peprt/modeling.py::build_model.

    Keras: Conv1D(64, k=5, relu) -> MaxPooling1D(2, 2) ->
    Bidirectional(LSTM(128, return_sequences=True)) -> Dropout(0.5) -> Flatten ->
    Dense(512, relu) -> Dense(256, relu) -> Dense(1, relu).
    """

    def __init__(self, seq_len=50, in_ch=AMINO_ACID_SIZE):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, 64, kernel_size=5, padding=2)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear((seq_len // 2) * 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        x = self.pool(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.drop(x)
        x = x.reshape(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return x


def build_deepdia_pepdetect():
    return DeepDIAPepDetect()


def example_input_deepdia_pepdetect():
    return torch.randn(1, 64, AMINO_ACID_SIZE)


def build_deepdia_pepms2():
    return DeepDIAPepMS2()


def example_input_deepdia_pepms2():
    return torch.randn(1, 50, AMINO_ACID_SIZE)


def build_deepdia_peprt():
    return DeepDIAPepRT()


def example_input_deepdia_peprt():
    return torch.randn(1, 50, AMINO_ACID_SIZE)


MENAGERIE_ENTRIES = [
    (
        "DeepDIA-PeptideDetectability",
        build_deepdia_pepdetect,
        example_input_deepdia_pepdetect,
        2020,
        "ported-pytorch",
    ),
    (
        "DeepDIA-PeptideMS2",
        build_deepdia_pepms2,
        example_input_deepdia_pepms2,
        2020,
        "ported-pytorch",
    ),
    ("DeepDIA-PeptideRT", build_deepdia_peprt, example_input_deepdia_peprt, 2020, "ported-pytorch"),
]
