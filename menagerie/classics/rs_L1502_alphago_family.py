"""Staging module for AlphaGo-family PyTorch policy/value networks (queue rows 1523, 1524).

Two independent vendored models:
  1. OthelloNNet -- from suragnair/alpha-zero-general @ f1a78e0505c6
     (community PyTorch reimplementation of AlphaGo's dual policy+value CNN
     head, per queue notes for "AlphaGo policy/value networks").
  2. AlphaZeroNet -- from michaelnny/alpha_zero @ 41ec8d65b3db
     ("AlphaGo Zero" candidate: residual-CNN dual-head policy+value net used
     for self-play Go/Gomoku, no supervised phase).
"""

import math
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# SOURCE: vendored from suragnair/alpha-zero-general @ f1a78e0505c6
# File: othello/pytorch/OthelloNNet.py
# =============================================================================


class _OthelloArgs:
    """Minimal stand-in for the `args` Namespace the original NNet expects."""

    def __init__(self, num_channels: int = 32, dropout: float = 0.3):
        self.num_channels = num_channels
        self.dropout = dropout


class OthelloNNet(nn.Module):
    """Dual policy/value CNN head (community PyTorch port of AlphaGo-style net).

    Original constructor took `(game, args)` where `game` exposed
    `getBoardSize()` / `getActionSize()`. We inline those two integers
    directly to avoid vendoring the full Othello game harness, without
    altering the network architecture itself.
    """

    def __init__(self, board_x: int, board_y: int, action_size: int, args: _OthelloArgs):
        self.board_x, self.board_y = board_x, board_y
        self.action_size = action_size
        self.args = args

        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training
        )  # batch_size x 1024
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training
        )  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


def build_othello_nnet() -> OthelloNNet:
    """Tiny 6x6 Othello-sized board so the net stays small but real."""
    board_x = board_y = 6
    action_size = board_x * board_y + 1
    args = _OthelloArgs(num_channels=16, dropout=0.3)
    net = OthelloNNet(board_x, board_y, action_size, args)
    net.eval()
    return net


def example_input_othello_nnet() -> torch.Tensor:
    board_x = board_y = 6
    return torch.randn(2, board_x, board_y)


# =============================================================================
# SOURCE: vendored from michaelnny/alpha_zero @ 41ec8d65b3db
# File: alpha_zero/core/network.py
# =============================================================================


class NetworkOutputs(NamedTuple):
    pi_prob: torch.Tensor
    value: torch.Tensor


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def initialize_weights(net: nn.Module) -> None:
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(net, nn.Module)

    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")

            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ResNetBlock(nn.Module):
    """Basic redisual block."""

    def __init__(
        self,
        num_filters: int,
    ) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class AlphaZeroNet(nn.Module):
    """Policy network for AlphaZero agent."""

    def __init__(
        self,
        input_shape: Tuple,
        num_actions: int,
        num_res_block: int = 19,
        num_filters: int = 256,
        num_fc_units: int = 256,
        gomoku: bool = False,
    ) -> None:
        super().__init__()
        c, h, w = input_shape

        # We need to use additional padding for Gomoku to fix agent shortsighted on edge cases
        num_padding = 3 if gomoku else 1

        conv_out_hw = calc_conv2d_output((h, w), 3, 1, num_padding)
        # FIX BUG, Python 3.7 has no math.prod()
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        # First convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=num_padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResNetBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * conv_out, num_fc_units),
            nn.ReLU(),
            nn.Linear(num_fc_units, 1),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> NetworkOutputs:
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""

        conv_block_out = self.conv_block(x)
        features = self.res_blocks(conv_block_out)

        # Predict raw logits distributions wrt policy
        pi_logits = self.policy_head(features)

        # Predict evaluated value from current player's perspective.
        value = self.value_head(features)

        return NetworkOutputs(pi_prob=pi_logits, value=value)


def build_alphazero_net() -> AlphaZeroNet:
    """Tiny 9x9 Go-sized board, small res-block count for a lightweight capture."""
    input_shape = (17, 9, 9)
    num_actions = 9 * 9 + 1
    net = AlphaZeroNet(
        input_shape=input_shape,
        num_actions=num_actions,
        num_res_block=2,
        num_filters=16,
        num_fc_units=16,
        gomoku=False,
    )
    net.eval()
    return net


def example_input_alphazero_net() -> torch.Tensor:
    return torch.randn(2, 17, 9, 9)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "AlphaGo policy/value networks",
        "build_othello_nnet",
        "example_input_othello_nnet",
        2016,
        MENAGERIE_ZOO,
    ),
    (
        "AlphaGo Zero",
        "build_alphazero_net",
        "example_input_alphazero_net",
        2017,
        MENAGERIE_ZOO,
    ),
]
