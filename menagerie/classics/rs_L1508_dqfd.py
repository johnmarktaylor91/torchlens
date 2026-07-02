# SOURCE: vendored from https://github.com/hw9603/DQfD-PyTorch @ master
# (DQNfromDemo/Test/CartPole.py :: Net2 -- the real Q-network the reference repo's own
# DQfD training script constructs and passes into DQfD.DeepQL(Net2, ...); the DeepQL
# class in DQNfromDemo/DQfD.py is the training-loop/loss/replay-buffer glue, not the
# network architecture, so it is not vendored here. No architectural changes to Net2.)
"""DQfD (Deep Q-learning from Demonstrations; Hester et al., 2018) augments DQN training
with a small set of expert demonstrations, combining a 1-step double-DQN TD loss, an
n-step return loss, and a large-margin supervised classification loss over the demo
actions. The Q-network itself is architecture-unmodified DQN: this vendors `Net2`, the
real Q-network class the reference DQfD-PyTorch repo's own CartPole test script builds
and trains via `DQfD.DeepQL(Net2, ...)` -- a small MLP mapping observation -> per-action
Q-values, exactly as used for the low-dimensional-state (CartPole / Atari-RAM) DQfD
variant in that repo."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net2(nn.Module):
    """Q-network: observation -> per-action Q-values (DQfD's predictNet/targetNet)."""

    def __init__(self, in_features=4, hidden=40, n_actions=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, n_actions)

    def forward(self, s):
        x = self.fc1(s)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def build_dqfd():
    # CartPole-v1 config from the reference repo's own CartPole.py: 4-dim state, 40
    # hidden units, 2 actions.
    return Net2(in_features=4, hidden=40, n_actions=2)


def example_input_dqfd():
    return torch.randn(8, 4)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DQfD (Deep Q-learning from Demonstrations) Q-network",
        build_dqfd,
        example_input_dqfd,
        2018,
        MENAGERIE_ZOO,
    ),
]
