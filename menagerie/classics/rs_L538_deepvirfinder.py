# FAITHFUL PORT of jessieren/DeepVirFinder @ master (training.py) (original framework: Keras/Theano)
#
# DeepVirFinder: a siamese 1D-CNN that scores metagenomic contigs as viral vs. host by
# reading a one-hot-encoded (seq_len, 4) DNA window from both the forward strand and its
# reverse complement through weight-SHARED Conv1D -> GlobalMaxPooling1D -> Dense ->
# Dense(sigmoid) branches, then averaging the two branch scores (Ren et al. 2020,
# "Identifying viruses from metagenomic data using deep learning", Quantitative Biology).
#
# The repo only ships pretrained Keras `.h5` weight files (loaded via
# `keras.models.load_model`); the architecture ITSELF is fully specified in
# `training.py`'s `##### build model #####` block (forward_input/reverse_input ->
# shared `hidden_layers` list -> Average()) with no separate model-definition module.
# Keras/Theano is not installed in this base env, so the architecture is transcribed
# faithfully into base-env torch rather than vendored verbatim: Conv1D(filters, kernel,
# relu) -> GlobalMaxPooling1D -> Dropout -> Dense(relu) -> Dropout -> Dense(1, sigmoid),
# applied with IDENTICAL, weight-tied parameters to the forward and reverse-complement
# encoded strand, then Average()'d -- every layer/mechanism from the real `hidden_layers`
# list, in order, is present here.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class DeepVirFinderCNN(nn.Module):
    """Faithful port of the siamese Conv1D + GlobalMaxPooling1D + Dense(relu) +
    Dense(sigmoid) branch (weight-shared across forward/reverse-complement strands,
    then averaged) defined inline in training.py's `##### build model #####` block."""

    def __init__(
        self,
        channel_num: int = 4,
        nb_filter1: int = 100,
        filter_len1: int = 10,
        nb_dense: int = 100,
        dropout_pool: float = 0.1,
        dropout_dense: float = 0.1,
    ):
        super().__init__()
        # Conv1D(filters=nb_filter1, kernel_size=filter_len1, activation='relu')
        self.conv1 = nn.Conv1d(channel_num, nb_filter1, kernel_size=filter_len1)
        self.relu_conv = nn.ReLU()
        # GlobalMaxPooling1D()
        # Dropout(dropout_pool)
        self.dropout_pool = nn.Dropout(dropout_pool)
        # Dense(nb_dense, activation='relu')
        self.dense1 = nn.Linear(nb_filter1, nb_dense)
        self.relu_dense = nn.ReLU()
        # Dropout(dropout_dense)
        self.dropout_dense = nn.Dropout(dropout_dense)
        # Dense(1, activation='sigmoid')
        self.dense2 = nn.Linear(nb_dense, 1)
        self.sigmoid = nn.Sigmoid()

    def _branch(self, x):
        # x: (batch, seq_len, channel_num); Keras Conv1D is channels-last, torch is
        # channels-first, so transpose to (batch, channel_num, seq_len).
        x = x.transpose(1, 2)
        x = self.relu_conv(self.conv1(x))
        x, _ = torch.max(x, dim=2)  # GlobalMaxPooling1D over the sequence axis
        x = self.dropout_pool(x)
        x = self.relu_dense(self.dense1(x))
        x = self.dropout_dense(x)
        x = self.sigmoid(self.dense2(x))
        return x

    def forward(self, forward_input, reverse_input):
        # Weight-shared siamese branches (same `hidden_layers` applied to both inputs
        # in the real code), then Average()([forward_output, reverse_output]).
        forward_output = self._branch(forward_input)
        reverse_output = self._branch(reverse_input)
        return (forward_output + reverse_output) / 2.0


def build_deepvirfinder():
    model = DeepVirFinderCNN(channel_num=4, nb_filter1=16, filter_len1=6, nb_dense=16)
    model.eval()
    return model


def example_input_deepvirfinder():
    # Real input: one-hot-encoded forward-strand and reverse-complement DNA windows,
    # each (batch, seq_len, 4); use a short window for tracing speed.
    forward_input = torch.rand(2, 40, 4)
    reverse_input = torch.rand(2, 40, 4)
    return (forward_input, reverse_input)


MENAGERIE_ENTRIES = [
    ("DeepVirFinder", build_deepvirfinder, example_input_deepvirfinder, 2020, "SOURCE_AVAILABLE"),
]
