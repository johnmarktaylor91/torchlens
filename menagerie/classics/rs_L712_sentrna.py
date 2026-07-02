# FAITHFUL PORT of jadeshi/SentRNA @ master (original framework: TensorFlow 1.x /
# tf.compat.v1 graph-mode)
#
# SentRNA: "SentRNA: Fully automated RNA secondary structure design"
# (Shi, Koodli, Yu, Silver, Das/Eterna project, PLOS ONE 2021 preprint /
# https://github.com/jadeshi/SentRNA)
#
# Repo checked: https://github.com/jadeshi/SentRNA (master). The actual trainable
# network is a small feedforward classifier defined in
# `SentRNA/util/feedforward.py::TensorflowClassifierModel` (TF1 placeholder/graph
# style: `tf.placeholder`, `tf.Variable`, `tf.Session`). Fetched verbatim from
# https://raw.githubusercontent.com/jadeshi/SentRNA/master/SentRNA/util/feedforward.py
# -- the ENTIRE architecture is the `full_forward` method:
#
#   def full_forward(self, n_layers, x, weights, biases):
#       layer = tf.add(tf.matmul(x, weights[0]), biases[0])
#       layer = tf.nn.relu(layer)
#       for i in range(1, n_layers - 1):
#           layer = tf.add(tf.matmul(layer, weights[i]), biases[i])
#           layer = tf.nn.relu(layer)
#       return tf.matmul(layer, weights[n_layers - 1]) + biases[n_layers - 1]
#
# i.e. a plain fully-connected MLP: (n_layers - 1) Linear+ReLU hidden layers
# followed by one final Linear layer with NO activation (softmax cross-entropy is
# applied externally as the training loss in `TensorflowClassifierModel.cost`, not
# as part of the forward graph). `layer_sizes` is built in `run.py`
# (`SentRNA/run.py:14-17`, fetched from
# https://raw.githubusercontent.com/jadeshi/SentRNA/master/SentRNA/run.py) as:
#
#   layer_sizes = [input_size] + [hidden_size] * n_layers + [4]
#
# with CLI defaults `n_layers=3` hidden layers and `hidden_size=100`
# (`SentRNA/run.py:169-170`); the 4-way output head predicts one of the four RNA
# nucleotides (A/C/G/U) for the base position under consideration. `input_size` is
# data-dependent (concatenation of local pairing/nearest-neighbor/mutual-information
# features built by `util/featurize_util.py::featurize`); this port uses a
# representative fixed feature width (matching the concrete example input below) --
# the architecture (depth, width, activation placement) is transcribed exactly from
# the real `full_forward` code, only the (inherently data-dependent) input
# cardinality is fixed to a concrete value for tracing.
#
# Random-initialized only -- the menagerie captures architecture, not trained
# parameters.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

SENTRNA_INPUT_SIZE = 64  # representative fixed width of the concatenated
# pair/nearest-neighbor/mutual-information feature vector (data-dependent upstream)
SENTRNA_HIDDEN_SIZE = 100  # util/featurize_util.py -> feedforward.py; run.py default
SENTRNA_N_HIDDEN_LAYERS = 3  # run.py --n_layers default
SENTRNA_N_CLASSES = 4  # A / C / G / U


class SentRNAClassifier(nn.Module):
    """Faithful port of TensorflowClassifierModel.full_forward: (n_layers - 1)
    Linear+ReLU hidden layers followed by one final unactivated Linear layer."""

    def __init__(
        self,
        input_size: int = SENTRNA_INPUT_SIZE,
        hidden_size: int = SENTRNA_HIDDEN_SIZE,
        n_hidden_layers: int = SENTRNA_N_HIDDEN_LAYERS,
        n_classes: int = SENTRNA_N_CLASSES,
    ):
        super().__init__()
        layer_sizes = [input_size] + [hidden_size] * n_hidden_layers + [n_classes]
        self.hidden = nn.ModuleList(
            [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 2)]
        )
        self.output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for linear in self.hidden:
            h = torch.relu(linear(h))
        return self.output_layer(h)


def build_sentrna():
    return SentRNAClassifier()


def example_input_sentrna():
    return (torch.rand(1, SENTRNA_INPUT_SIZE),)


MENAGERIE_ENTRIES = [
    ("SentRNA", "build_sentrna", "example_input_sentrna", 2021, "ported-pytorch"),
]
