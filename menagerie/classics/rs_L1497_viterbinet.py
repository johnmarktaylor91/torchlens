# SOURCE: vendored from ShlezingerLab/viterbinet-official @ main
#   python_code/detectors/viterbinet/viterbinet_detector.py (verbatim architecture)
#
# The original nirshlezinger1/ViterbiNet repo (the queue's linked repo, cited in
# "ViterbiNet: A Deep Learning Based Viterbi Algorithm for Symbol Detection", Shlezinger
# et al., IEEE TWC 2020) ships ONLY MATLAB code (.m files) -- no PyTorch anywhere in that
# repo despite the queue notes claiming "PyTorch". ShlezingerLab/viterbinet-official is
# the same research group's (Nir Shlezinger's lab) own official PyTorch reimplementation
# of the identical ViterbiNet algorithm (README: "A minimalistic python implementation of
# the ViterbiNet algorithm, published in: Shlezinger, N. ... (2020)"), so this is real,
# author-group-maintained source code, not a third-party guess.
#
# Only change from the real `viterbinet_detector.py`: the original imports `DEVICE` from
# `python_code/__init__.py`, which as a side effect instantiates a `Config()` singleton
# that reads a YAML config file from disk on import -- unrelated to the model architecture.
# `DEVICE` and the `Phase` enum values are inlined directly below (identical values) so
# the module is self-contained. No architecture code was rewritten.
from __future__ import annotations

from enum import Enum

import numpy as np
import torch
import torch.nn as nn

# python_code/__init__.py (DEVICE constant, verbatim)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# python_code/utils/constants.py::Phase (verbatim)
class Phase(Enum):
    TRAIN = "train"
    TEST = "test"


HIDDEN1_SIZE = 75
HIDDEN2_SIZE = 16


def create_transition_table(n_states: int) -> np.ndarray:
    """
    creates transition table of size [n_states,2]
    previous state of state i and input bit b is the state in cell [i,b]
    """
    transition_table = np.concatenate([np.arange(n_states), np.arange(n_states)]).reshape(
        n_states, 2
    )
    return transition_table


def acs_block(
    in_prob: torch.Tensor, llrs: torch.Tensor, transition_table: torch.Tensor, n_states: int
):
    """
    Viterbi ACS block
    :param in_prob: last stage probabilities, [batch_size,n_states]
    :param llrs: edge probabilities, [batch_size,1]
    :param transition_table: transitions
    :param n_states: number of states
    :return: current stage probabilities, [batch_size,n_states]
    """
    transition_ind = transition_table.reshape(-1).repeat(in_prob.size(0)).long()
    batches_ind = torch.arange(in_prob.size(0)).repeat_interleave(2 * n_states)
    trellis = (in_prob + llrs)[batches_ind, transition_ind]
    reshaped_trellis = trellis.reshape(-1, n_states, 2)
    return torch.min(reshaped_trellis, dim=2)[0]


class ViterbiNetDetector(nn.Module):
    """
    This implements the VA decoder by a parameterization of the cost calculation by an NN for each stage
    """

    def __init__(self, n_states: int):
        super(ViterbiNetDetector, self).__init__()
        self.n_states = n_states
        self.transition_table_array = create_transition_table(n_states)
        self.transition_table = torch.Tensor(self.transition_table_array).to(DEVICE)
        self._initialize_dnn()

    def _initialize_dnn(self):
        layers = [nn.Linear(1, HIDDEN1_SIZE), nn.ReLU(), nn.Linear(HIDDEN1_SIZE, self.n_states)]
        self.net = nn.Sequential(*layers).to(DEVICE)

    def forward(self, rx: torch.Tensor, phase: str) -> torch.Tensor:
        """
        The forward pass of the ViterbiNet algorithm
        :param rx: input values, size [batch_size,transmission_length]
        :param phase: 'train' or 'val'
        :returns if in 'train' - return the estimated priors [batch_size,transmission_length,n_states]
        if in 'val' - return the detected words [n_batch,transmission_length]
        """
        in_prob = torch.zeros([1, self.n_states]).to(DEVICE)
        priors = self.net(rx)

        if phase == Phase.TEST:
            detected_word = torch.zeros(rx.shape).to(DEVICE)
            for i in range(rx.shape[0]):
                detected_word[i] = torch.argmin(in_prob, dim=1) % 2
                out_prob = acs_block(in_prob, -priors[i], self.transition_table, self.n_states)
                in_prob = out_prob
            return detected_word
        else:
            return priors


# --------------------------------------------------------------------------------
# menagerie staging entry points
# --------------------------------------------------------------------------------

MENAGERIE_ZOO = "vendored-pytorch"


class ViterbiNetTrainWrapped(nn.Module):
    """Thin wrapper pinning phase='train' (the plain per-symbol-priors DNN forward,
    real ViterbiNetDetector code path) so torchlens can trace with a tensor-only
    example input. The 'test' phase runs the same real DNN plus a real but
    input-length-dependent Viterbi ACS decoding loop over rx.shape[0]."""

    def __init__(self, detector: ViterbiNetDetector):
        super().__init__()
        self.detector = detector

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        return self.detector(rx, Phase.TRAIN)


def build_viterbinet():
    # Real ViterbiNetDetector, tiny n_states (paper commonly uses 2^(memory_length),
    # e.g. n_states=16 for a memory-length-4 channel; a small n_states=4 here is still
    # the real, unmodified architecture).
    return ViterbiNetTrainWrapped(ViterbiNetDetector(n_states=4))


def example_input_viterbinet():
    # rx: [batch_size, 1] received (noisy) channel observations; matches
    # nn.Linear(1, HIDDEN1_SIZE) input contract in the real detector.
    return (torch.randn(8, 1),)


MENAGERIE_ENTRIES = [
    ("ViterbiNet", build_viterbinet, example_input_viterbinet, 2020, MENAGERIE_ZOO),
]
