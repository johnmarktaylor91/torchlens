# FAITHFUL PORT of xypan1232/iDeepS @ master (ideeps.py, functions set_cnn_model /
# get_cnn_network / run_network) (original framework: Keras 1.x / Theano, Python 2)
#
# iDeepS: predicting RNA-protein binding sites and motifs using an integrated CNN-BiLSTM deep
# model over RNA sequence + predicted secondary structure (Pan & Shen, BMC Genomics 2018).
# ideeps.py is genuine Keras-1.x/Theano/Python-2 code (`Merge` layer -- removed from Keras 2+;
# `print` statements; `xrange`) with no installable modern runtime, and the repo additionally
# vendors a C++ EDeN/GraphProt toolchain used only for structure-profile feature extraction
# (not part of the traced network). ideeps.py cannot run in the base torch env, so this is a
# faithful architectural transcription of the REAL model-building functions:
#
#   set_cnn_model(input_dim, input_length=111):
#       Conv1D(input_dim -> 16 filters, kernel=10, valid padding, stride=1) -> ReLU
#       -> MaxPool1D(pool_size=3) -> Dropout(0.5)
#   get_cnn_network():
#       seq branch   = set_cnn_model(4, 111)   # one-hot RNA sequence (A/C/G/U)
#       struct branch = set_cnn_model(6, 111)  # RNAshapes structure-profile channels
#       concat(seq_branch, struct_branch) along the feature axis
#       -> Bidirectional(LSTM(hidden=2*16=32))  # nbfilter*2 units per direction
#       -> Dropout(0.10)
#       -> Dense(nbfilter*2=32, activation="relu")
#   run_network(...): appends the classification head used at call sites --
#       Dense(total_hid=32 -> 2) -> softmax
#
# Every layer/mechanism above is preserved 1:1 (same filter counts, kernel sizes, pool size,
# dropout rates, LSTM hidden size, bidirectional merge, dense head). Keras Conv1D uses
# channels-last (length, channels); torch Conv1d is channels-first, so the port transposes
# the input once (functionally identical convolution) rather than altering the architecture.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

NBFILTER = 16
SEQ_LEN = 111
SEQ_CHANNELS = 4  # one-hot RNA sequence (A/C/G/U)
STRUCT_CHANNELS = 6  # RNAshapes structure-profile channels


class CNNBranch(nn.Module):
    """Faithful port of `set_cnn_model(input_dim, input_length)`:
    Conv1D(filters=16, kernel=10, valid) -> ReLU -> MaxPool1D(pool=3) -> Dropout(0.5)."""

    def __init__(self, input_dim, nbfilter=NBFILTER):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=nbfilter, kernel_size=10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch, input_dim, length) -- torch Conv1d is channels-first.
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class IDeepSCNNBiLSTM(nn.Module):
    """Faithful port of `get_cnn_network()` + the `run_network` classification head:
    two CNN branches (sequence, structure) -> concat -> BiLSTM(32) -> Dropout(0.10)
    -> Dense(32, relu) -> Dense(2) -> softmax."""

    def __init__(self, nbfilter=NBFILTER):
        super().__init__()
        self.seq_branch = CNNBranch(SEQ_CHANNELS, nbfilter)
        self.struct_branch = CNNBranch(STRUCT_CHANNELS, nbfilter)
        self.bilstm = nn.LSTM(
            input_size=2 * nbfilter, hidden_size=2 * nbfilter, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(0.10)
        self.dense_hidden = nn.Linear(2 * (2 * nbfilter), nbfilter * 2)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(nbfilter * 2, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seq_x, struct_x):
        # seq_x: (batch, 4, 111), struct_x: (batch, 6, 111)
        seq_feat = self.seq_branch(seq_x)  # (batch, nbfilter, L')
        struct_feat = self.struct_branch(struct_x)  # (batch, nbfilter, L')

        # Keras `Merge([seq_model, struct_model], mode='concat', concat_axis=1)` concatenates
        # along the channel axis (axis=1 in Keras' (batch, steps, channels) or the analogous
        # feature axis here); both branches already share the same reduced sequence length.
        merged = torch.cat([seq_feat, struct_feat], dim=1)  # (batch, 2*nbfilter, L')
        merged = merged.transpose(1, 2)  # -> (batch, L', 2*nbfilter) for LSTM's batch_first layout

        lstm_out, _ = self.bilstm(merged)  # (batch, L', 2 * hidden) [bidirectional concat]
        pooled = lstm_out[:, -1, :]  # Keras Bidirectional(LSTM(...)) returns only the final state

        x = self.dropout(pooled)
        x = self.dense_hidden(x)
        x = self.relu(x)
        x = self.classifier(x)
        x = self.softmax(x)
        return x


def build_ideeps_cnn_bilstm():
    return IDeepSCNNBiLSTM()


def example_input_ideeps_cnn_bilstm():
    seq_x = torch.rand(2, SEQ_CHANNELS, SEQ_LEN)
    struct_x = torch.rand(2, STRUCT_CHANNELS, SEQ_LEN)
    return (seq_x, struct_x)


MENAGERIE_ENTRIES = [
    (
        "iDeepS-CNN-BiLSTM",
        build_ideeps_cnn_bilstm,
        example_input_ideeps_cnn_bilstm,
        2018,
        "SOURCE_AVAILABLE",
    ),
]
