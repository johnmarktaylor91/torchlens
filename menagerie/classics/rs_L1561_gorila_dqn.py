# SOURCE: vendored from kareido/pytorl @ master
# https://github.com/kareido/pytorl (pytorl/networks/atari_conv.py)
#
# Gorila (General Reinforcement Learning Architecture, Nair et al. 2015, arXiv:1507.04296)
# is a massively-parallel distributed DQN system; its "contribution" is the distributed
# actor/learner/parameter-server infrastructure (see pytorl/agents/dist_DQN.py and
# examples/gorila_dqn/{client,server,launcher}.py), not the Q-network architecture itself,
# which is the standard Nature-DQN convolutional Q-network. pytorl is not pip-installable
# (no PyPI release), so the real class is vendored verbatim here (imports/paths only
# trimmed) rather than referenced via a recipe import.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class Q_Network(nn.Module):
    """
    build a 2-D observation Q-network for DQN and its variants

    Args:
        input_size: a 3-D tuple (or equivalent Iterable) indicating the size of input
        num_actions: corresponding to the size of output
        backbone: a pytorch network, if not specified, using original DQN instead
        replace_fc: the name of last linear(i.e. 'fc' or 'last_linear'), if not
            specified, using backbone's last output layer instead.
    """

    def __init__(
        self,
        input_size=None,
        num_actions=None,
        backbone=None,
        replace_fc=None,
    ):
        super(Q_Network, self).__init__()
        if input_size is not None:
            assert len(input_size) == 3, "input_size must be a 3-D Iterable"
        self.input_size, self.num_actions = input_size, num_actions
        if backbone:
            self.network = backbone
            if replace_fc is not None:
                if num_actions is None:
                    raise ValueError("num_actions must be specified if enabling replace_fc")
                last_fc = getattr(self.network, replace_fc)
                last_fc_in = last_fc.__dict__["in_features"]
                last_fc_out = last_fc.__dict__["out_features"]
                if last_fc_out != num_actions:
                    last_fc = nn.Linear(last_fc_in, num_actions)
            if input_size is not None:
                self.check_forward()
            else:
                print("warning: skip precheck due to input_size not specified", flush=True)

        else:
            if not (input_size and num_actions):
                raise ValueError(
                    "must specify input_size and num_actions if backbone not specified"
                )
            self.network = _Original_DQN(input_size, num_actions)
            self.check_forward()

    def forward(self, x):
        return self.network(x)

    def check_forward(self):
        mock = torch.zeros(1, *self.input_size)
        try:
            self.network(mock)
        except Exception:
            raise ValueError("network forward failure, presumably due to invalid input_size")


class _Original_DQN(nn.Module):
    """
    this is the Q-network used in the original DeepMind DQN paper: Human-level control through
        deep reinforcement learning (https://www.nature.com/articles/nature14236)

    Args:
        input_size: a 3-D tuple (or equivalent Iterable) indicating the size of input
        num_actions: corresponding to the size of output
    """

    def __init__(self, input_size, num_actions):
        super(_Original_DQN, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(self.input_size[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, self.num_actions)

    @property
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_size)).shape[1]

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_gorila_dqn() -> nn.Module:
    return Q_Network(input_size=(4, 84, 84), num_actions=6)


def example_input_gorila_dqn() -> torch.Tensor:
    return torch.randn(2, 4, 84, 84)


MENAGERIE_ENTRIES = [
    (
        "Gorila (Massively Parallel DQN)",
        build_gorila_dqn,
        example_input_gorila_dqn,
        2015,
        MENAGERIE_ZOO,
    ),
]
