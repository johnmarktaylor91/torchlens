# SOURCE: vendored from https://github.com/PWhiddy/Growing-Neural-Cellular-Automata-Pytorch @ master
# (CA_Basic/basic_growth.py, CAModel class)
#
# "Growing Neural Cellular Automata" (Mordvintsev et al., distill.pub 2020) --
# PyTorch reimplementation used for the extended "PCG" (procedural content
# generation, e.g. Pokemon-sprite growth) experiments in this repo. The
# `CAModel` is the shared per-cell update rule: a fixed 3x3 Sobel/identity
# perceptual filter feeds a 1x1-conv MLP (env_d*3 -> 208 -> env_d) applied
# identically to every cell of the grid (the "cellular automaton" step).
# Vendored verbatim; only the outer CASimulator training/rendering harness
# (image loading, autoregressive rollout, loss/backprop loop) is dropped in
# favor of a tiny build_/example_input_ staging harness that traces one
# CAModel forward call on the perception-filtered state tensor it is
# actually applied to (batch, env_d*3, H, W) -> (batch, env_d, H, W).
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAModel(nn.Module):
    def __init__(self, env_d):
        super(CAModel, self).__init__()
        self.conv1 = nn.Conv2d(env_d * 3, 208, 1)
        self.conv2 = nn.Conv2d(208, env_d, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)


# --- staging harness: build + example input ---------------------------------


def build_neural_cellular_automata_pcg():
    # tiny env depth for a fast trace; matches the shape convention used by
    # CASimulator.raw_senses() (env_d*3 perception channels in, env_d state
    # channels out).
    return CAModel(env_d=16)


def example_input_neural_cellular_automata_pcg():
    batch_size = 2
    env_d = 16
    height, width = 24, 24
    return torch.randn(batch_size, env_d * 3, height, width)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Neural Cellular Automata PCG",
        build_neural_cellular_automata_pcg,
        example_input_neural_cellular_automata_pcg,
        2020,
        MENAGERIE_ZOO,
    ),
]
