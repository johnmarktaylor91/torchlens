# FAITHFUL PORT of lje00006/DeepCas9 @ master (original framework: R + MXNet)
# (DeepCas9-CNN_models_weights/DeepCas9_293T-symbol.json: the real trained MXNet
# computation-graph definition; encodeOntargetSeq.R / DeepCas9_scores.R for I/O shape)
#
# DeepCas9 (Xue et al., "Prediction of CRISPR sgRNA activity using a deep convolutional
# neural network"): a small CNN over a one-hot-encoded 30-nucleotide (4-row x 30-col)
# spCas9 target sequence, predicting a scalar on-target activity score. The repo ships no
# Python/PyTorch code at all -- inference is driven by R calling a saved MXNet symbol/params
# pair. The exact architecture is not described in prose anywhere in the repo; it is
# reconstructed directly from the serialized MXNet symbol graph
# (DeepCas9_293T-symbol.json), which lists the real op sequence and hyperparameters:
#   Convolution(kernel=(4,4), num_filter=50) -> Activation(relu) ->
#   Pooling(kernel=(2,1), stride=(2,1), max) -> Flatten ->
#   FullyConnected(num_hidden=500) -> Activation(relu) ->
#   FullyConnected(num_hidden=1) -> LinearRegressionOutput
# Input layout per DeepCas9_scores.R (`dim(test_onehot)<-c(4,30,1,ncol(test_onehot))`, an
# R/MXNet column-major dim vector which unpacks -- fastest-varying axis first -- to Python
# NCHW (N=ncol, C=1, H=30 sequence position, W=4 one-hot nucleotide). This orientation is
# confirmed by the graph's own hyperparameters: Conv2d kernel=(4,4) spans the full W=4
# nucleotide axis (collapsing it to 1) while sliding over the H=30 sequence axis, and the
# subsequent Pooling(kernel=(2,1), stride=(2,1)) only pools along H -- consistent only with
# H=sequence-position, W=nucleotide-channel (the reverse orientation makes the conv/pool
# shapes emit degenerate zero-size tensors).
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class DeepCas9(nn.Module):
    def __init__(self, seq_len=30, n_bases=4, num_filter=50, hidden=500):
        super().__init__()
        self.conv = nn.Conv2d(1, num_filter, kernel_size=(4, 4))
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self._flat_dim = self._infer_flat_dim(seq_len, n_bases)
        self.fc1 = nn.Linear(self._flat_dim, hidden)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)

    def _infer_flat_dim(self, seq_len, n_bases):
        with torch.no_grad():
            x = torch.zeros(1, 1, seq_len, n_bases)
            x = self.pool(self.relu1(self.conv(x)))
            return x.numel()

    def forward(self, x):
        # x: (batch, 1, 30, 4) one-hot DNA sequence, NCHW (H=sequence, W=nucleotide)
        x = self.relu1(self.conv(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x


def build_deepcas9():
    return DeepCas9()


def example_input_deepcas9():
    # (batch, channel=1, rows=30 sequence positions, cols=4 one-hot nucleotides)
    return torch.randn(2, 1, 30, 4)


MENAGERIE_ENTRIES = [
    ("DeepCas9", build_deepcas9, example_input_deepcas9, 2018, "REIMPLEMENT"),
]
