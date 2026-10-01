# SOURCE: vendored from https://github.com/Thijsvanede/DeepLog @ master
#
# Source: deeplog/deeplog.py, class DeepLog -- the CCS 2017 DeepLog log-key anomaly
# detection model: one-hot encodes an integer log-key sequence, feeds it through a
# stacked LSTM, and predicts a distribution over the next log key via a linear head +
# LogSoftmax. The real repo's `DeepLog` subclasses `torchtrain.Module`, a thin
# training/predict-loop mixin from the `torchtrain` package (not a base lib here); that
# mixin contributes no architecture (no extra layers, no forward-path changes) -- it only
# adds `.fit()`/`.predict()` convenience wrappers on top of `nn.Module`. This vendored
# copy therefore subclasses `nn.Module` directly and keeps `__init__`/`forward` verbatim
# (including the exact one-hot -> LSTM -> last-timestep -> Linear -> LogSoftmax path);
# only the `torchtrain`-derived `.predict()`/`.save()`/`.load()` convenience methods
# (unrelated to the traced graph) were dropped.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class DeepLog(nn.Module):
    """DeepLog model used for training and predicting logs (real repo architecture)."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        """DeepLog model used for training and predicting logs.

        Parameters
        ----------
        input_size : int
            Dimension of input layer.

        hidden_size : int
            Dimension of hidden layer.

        output_size : int
            Dimension of output layer.

        num_layers : int, default=2
            Number of hidden layers, i.e. stacked LSTM modules.
        """
        # Initialise nn.Module
        super(DeepLog, self).__init__()

        # Store input parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Initialise model layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    ########################################################################
    #                       Forward through network                        #
    ########################################################################

    def forward(self, X):
        """Forward sample through DeepLog.

        Parameters
        ----------
        X : tensor
            Input to forward through DeepLog network.

        Returns
        -------
        result : tensor
        """
        # One hot encode X
        X = F.one_hot(X.to(torch.int64), self.input_size).to(torch.float)

        # Set initial hidden states
        hidden = self._get_initial_state(X)
        state = self._get_initial_state(X)

        # Perform LSTM layer
        out, hidden = self.lstm(X, (hidden, state))
        # Perform output layer
        out = self.out(out[:, -1, :])
        # Create probability
        out = self.softmax(out)

        # Return result
        return out

    ########################################################################
    #                         Auxiliary functions                          #
    ########################################################################

    def _get_initial_state(self, X):
        """Return a given hidden state for X."""
        # Return tensor of correct shape as device
        return torch.zeros(
            self.num_layers,
            X.size(0),
            self.hidden_size,
        ).to(X.device)


def build_deeplog():
    # Real repo default sizing for HDFS log-key vocab (examples/example_hdfs.py uses
    # input_size=300); kept small here (input_size=30) since only the architecture,
    # not learned weights, needs to be faithfully represented.
    return DeepLog(input_size=30, hidden_size=64, output_size=30, num_layers=2)


def example_input_deeplog():
    # (n_samples, seq_len) integer log-key indices in [0, input_size).
    return torch.randint(0, 30, (4, 10))


MENAGERIE_ENTRIES = [
    ("DeepLog", build_deeplog, example_input_deeplog, 2017, MENAGERIE_ZOO),
]
