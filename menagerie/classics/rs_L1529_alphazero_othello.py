# SOURCE: vendored from suragnair/alpha-zero-general @ master
# https://github.com/suragnair/alpha-zero-general
# Files: othello/pytorch/OthelloNNet.py
#
# AlphaZero-style dual-head policy/value network (Silver et al. 2017/2018),
# instantiated on the Othello board game -- the repo's well-known, widely
# reused general-purpose reimplementation of the AlphaZero self-play
# architecture (MCTS + a single conv-tower network producing a joint
# policy (pi) and value (v) head). No architectural change from the
# original file: 4 conv layers + batchnorm feature tower, followed by two
# FC layers with batchnorm/dropout, then a policy head (log-softmax over
# board actions) and a scalar value head (tanh).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class OthelloNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
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


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------


class _DotDict(dict):
    """Minimal stand-in for the repo's `utils.dotdict` (attribute access on a dict)."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class _TinyOthelloGame:
    """Minimal stand-in for othello.OthelloGame -- only the two accessors
    OthelloNNet's constructor calls (getBoardSize/getActionSize) are needed."""

    def __init__(self, board_size: int = 8):
        self._n = board_size

    def getBoardSize(self):
        return (self._n, self._n)

    def getActionSize(self):
        return self._n * self._n + 1  # board squares + 1 pass move, as in OthelloGame


def build_alphazero_othello():
    torch.manual_seed(0)
    game = _TinyOthelloGame(board_size=8)
    args = _DotDict(
        {
            "num_channels": 8,
            "dropout": 0.3,
        }
    )
    model = OthelloNNet(game, args)
    model.eval()
    return model


def example_input_alphazero_othello():
    torch.manual_seed(0)
    # A single 8x8 Othello board state, as passed to OthelloNNet.forward.
    return torch.randn(1, 8, 8)


MENAGERIE_ENTRIES = [
    (
        "AlphaZero (Othello, dual-head ResNet-style policy/value net)",
        "build_alphazero_othello",
        "example_input_alphazero_othello",
        2018,
        MENAGERIE_ZOO,
    ),
]
