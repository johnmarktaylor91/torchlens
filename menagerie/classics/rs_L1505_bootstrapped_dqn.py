# SOURCE: vendored from https://github.com/JoungheeKim/bootsrapped-dqn @ main
# (model.py + utills.py, minimal changes: merged into one file, imports fixed)
"""Bootstrapped DQN (Osband et al., 2016) -- ensemble of Q-heads sharing a
conv trunk, for posterior-sampling-style exploration in Atari-style DQN."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class HeadNet(nn.Module):
    def __init__(self, reshape_size, n_actions=4):
        super(HeadNet, self).__init__()
        self.fc1 = nn.Linear(reshape_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CoreNet(nn.Module):
    def __init__(self, h, w, num_channels=4):
        super(CoreNet, self).__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        self.reshape_size = convw * convh * 64

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # size after conv3
        x = x.view(-1, self.reshape_size)
        return x


class EnsembleNet(nn.Module):
    def __init__(self, n_ensemble, n_actions, h, w, num_channels):
        super(EnsembleNet, self).__init__()
        self.core_net = CoreNet(h=h, w=w, num_channels=num_channels)
        reshape_size = self.core_net.reshape_size
        self.net_list = nn.ModuleList(
            [HeadNet(reshape_size=reshape_size, n_actions=n_actions) for k in range(n_ensemble)]
        )

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, x, k=None):
        if k is not None:
            return self.net_list[k](self.core_net(x))
        else:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads


def build_bootstrapped_dqn():
    # Tiny Atari-style stack: 4-frame history, small spatial size, small ensemble.
    return EnsembleNet(n_ensemble=3, n_actions=4, h=42, w=42, num_channels=4)


def example_input_bootstrapped_dqn():
    return torch.randn(2, 4, 42, 42)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Bootstrapped DQN",
        build_bootstrapped_dqn,
        example_input_bootstrapped_dqn,
        2016,
        MENAGERIE_ZOO,
    ),
]
