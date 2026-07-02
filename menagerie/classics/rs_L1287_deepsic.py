# SOURCE: vendored from ShlezingerLab/deepsic-official @ 4fc818c1f2b8bca4d67f072a6840536b3ab15522
# python_code/detectors/deepsic/deepsic_detector.py + python_code/utils/constants.py (N_ANTS, N_USERS)
"""DeepSIC: Deep Soft Interference Cancellation detector for multiuser MIMO.

DeepSICDetector is the per-user, per-iteration base network of the DeepSIC
architecture (Shlezinger et al.), trained iteratively but architecturally a
plain 2-layer MLP classifier over a received-signal + prior-probability
feature vector.
"""

import torch
from torch import nn

# python_code/utils/constants.py
N_USERS = 4  # number of users
N_ANTS = 4  # number of antennas

HIDDEN_BASE_SIZE = 64


class DeepSICDetector(nn.Module):
    """
    The DeepSIC Network Architecture

    ===========Architecture=========
    DeepSICNet(
      (fullyConnectedLayer): Linear(...)
      (reluLayer): ReLU()
      (fullyConnectedLayer2): Linear(...)
    ================================
    """

    def __init__(self):
        super(DeepSICDetector, self).__init__()
        classes_num = 2
        hidden_size = HIDDEN_BASE_SIZE * classes_num
        base_rx_size = N_ANTS
        linear_input = base_rx_size + (classes_num - 1) * (N_USERS - 1)  # from DeepSIC paper
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, classes_num)

    def forward(self, rx: torch.Tensor) -> torch.Tensor:
        out0 = self.activation(self.fc1(rx))
        out1 = self.fc2(out0)
        return out1


def build_deepsic():
    return DeepSICDetector()


def example_input_deepsic():
    classes_num = 2
    linear_input = N_ANTS + (classes_num - 1) * (N_USERS - 1)
    return torch.randn(1, linear_input)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepSIC", "build_deepsic", "example_input_deepsic", 2020, "vendored-pytorch"),
]
