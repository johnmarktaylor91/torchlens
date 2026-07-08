# SOURCE: vendored from RLAgent/gated-path-planning-networks @ master
# https://github.com/RLAgent/gated-path-planning-networks
# Files: models/GPPN.py (Planner), no changes to the architecture; only the
# `args` object (originally an `argparse.Namespace` built by
# `utils/experiment.py:parse_args`) was replaced with a plain SimpleNamespace
# carrying the same three hyperparameters (`l_h`, `k`, `f`) the real module
# reads off of it.
#
# Gated Path Planning Network (GPPN; Lee, Chiang & Metz-style follow-up to the
# Value Iteration Network, "Gated Path Planning Networks", ICML 2018): a
# recurrent convolutional planner that runs `k` steps of an LSTM-gated value
# iteration directly over a maze's spatial grid, then reads off a per-cell,
# per-orientation action policy. Input is a concatenation of the maze free-space
# map and a one-hot goal-location map along the channel axis.
from types import SimpleNamespace

import torch
import torch.nn as nn


# Gated Path Planning Network module
class Planner(nn.Module):
    """
    Implementation of the Gated Path Planning Network.
    """

    def __init__(self, num_orient, num_actions, args):
        super(Planner, self).__init__()

        self.num_orient = num_orient
        self.num_actions = num_actions

        self.l_h = args.l_h
        self.k = args.k
        self.f = args.f

        self.hid = nn.Conv2d(
            in_channels=(num_orient + 1),  # maze map + goal location
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True,
        )

        self.h0 = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True,
        )

        self.c0 = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=self.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True,
        )

        self.conv = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=1,
            kernel_size=(self.f, self.f),
            stride=1,
            padding=int((self.f - 1.0) / 2),
            bias=True,
        )

        self.lstm = nn.LSTMCell(1, self.l_h)

        self.policy = nn.Conv2d(
            in_channels=self.l_h,
            out_channels=num_actions * num_orient,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False,
        )

        self.sm = nn.Softmax2d()

    def forward(self, map_design, goal_map):
        maze_size = map_design.size()[-1]
        X = torch.cat([map_design, goal_map], 1)

        hid = self.hid(X)
        h0 = self.h0(hid).transpose(1, 3).contiguous().view(-1, self.l_h)
        c0 = self.c0(hid).transpose(1, 3).contiguous().view(-1, self.l_h)

        last_h, last_c = h0, c0
        for _ in range(0, self.k - 1):
            h_map = last_h.view(-1, maze_size, maze_size, self.l_h)
            h_map = h_map.transpose(3, 1)
            inp = self.conv(h_map).transpose(1, 3).contiguous().view(-1, 1)

            last_h, last_c = self.lstm(inp, (last_h, last_c))

        hk = last_h.view(-1, maze_size, maze_size, self.l_h).transpose(3, 1)
        logits = self.policy(hk)

        # Normalize over actions
        logits = logits.view(-1, self.num_actions, maze_size, maze_size)
        probs = self.sm(logits)

        # Reshape to output dimensions
        logits = logits.view(-1, self.num_orient, self.num_actions, maze_size, maze_size)
        probs = probs.view(-1, self.num_orient, self.num_actions, maze_size, maze_size)
        logits = torch.transpose(logits, 1, 2).contiguous()
        probs = torch.transpose(probs, 1, 2).contiguous()

        return logits, probs, h0, hk


def build_gppn() -> nn.Module:
    """Tiny real GPPN Planner: NEWS mechanism (num_orient=1, num_actions=4,
    the repo's default `--mechanism news`), with a small hidden width / VI
    step count / kernel size for a fast trace (repo defaults: l_h=150, k=10,
    f=3 -- shrunk here to l_h=16, k=3, f=3)."""
    args = SimpleNamespace(l_h=16, k=3, f=3)
    model = Planner(num_orient=1, num_actions=4, args=args)
    model.eval()
    return model


def example_input_gppn():
    # (map_design, goal_map): each (batch, num_orient, maze_size, maze_size).
    # map_design is the free-space mask (1.0 = walkable), goal_map is a
    # one-hot goal-location indicator -- matching utils/maze.py's dataset
    # encoding fed into Planner.forward in train.py/eval.py.
    maze_size = 8
    map_design = torch.ones(1, 1, maze_size, maze_size)
    goal_map = torch.zeros(1, 1, maze_size, maze_size)
    goal_map[0, 0, maze_size - 1, maze_size - 1] = 1.0
    return (map_design, goal_map)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("GPPN (Gated Path Planning Network)", build_gppn, example_input_gppn, 2018, MENAGERIE_ZOO),
]
