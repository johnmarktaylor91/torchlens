# FAITHFUL PORT of google-deepmind/open_spiel @ master
# (open_spiel/python/algorithms/alpha_zero/model_nnx.py: ConvBlock,
# ResidualBlock, PolicyHead, ValueHead, AlphaZeroModel, lines 114-351)
# (original framework: JAX/Flax nnx)
#
# OpenSpiel's AlphaZero policy-value network (DeepMind, "AlphaGo Zero"/
# "AlphaZero" architecture as implemented for the general-game-playing
# OpenSpiel framework: https://github.com/google-deepmind/open_spiel). The
# repo's Python model lives only in JAX/Flax (`model_nnx.py`, using
# `flax.nnx`/`chex`/`optax`/`orbax`, none of which are installed in the base
# torch env and are not reasonably installable alongside this project's torch
# stack); the repo's only PyTorch AlphaZero implementation
# (`open_spiel/algorithms/alpha_zero_torch/`) is C++ LibTorch source
# (`model.cc`/`vpnet.cc`), not Python. Per the source docstring, "the resnet
# model copies the one in [the AlphaGo Zero/AlphaZero] paper when set with
# width 256 and depth 20" -- this port faithfully transcribes that `resnet`
# variant: a stem `ConvBlock` (conv+batchnorm+activation) feeding `nn_depth`
# stacked `ResidualBlock`s (two conv+batchnorm blocks with a skip-add,
# activation applied after the addition -- matching the JAX
# `ResidualBlock.__call__`'s `y = self.activation(y + residual)`), followed by
# separate `PolicyHead` (1x1 conv reducing channels to 2, flatten, MLP to
# `output_size` logits) and `ValueHead` (1x1 conv reducing channels to 1,
# flatten, MLP to a single tanh-bounded scalar) -- mirroring the JAX
# `PolicyHead`/`ValueHead` `model_type != "mlp"` branch exactly. The JAX code
# is NHWC (channels-last, `input_shape=(H, W, C)`); this port uses the
# standard torch NCHW layout with equivalent channel/spatial semantics
# (`nn.Conv2d(in, out, kernel_size, padding="same")`, `nn.BatchNorm2d`,
# `nn.Flatten()`) -- a data-layout translation only, not an architecture
# change.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class ConvBlock(nn.Module):
    """Convolutional block: conv (SAME padding) + BatchNorm + activation.
    Ports JAX `ConvBlock.__call__`: `y = act(bn(conv(x)))`."""

    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation() if activation is not None else nn.Identity()

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.activation(y)
        return y


class ResidualBlock(nn.Module):
    """Two ConvBlocks (second with no activation) + skip-add, activation
    applied AFTER the addition. Ports JAX `ResidualBlock.__call__`:
    `y = act(conv2(conv1(x)) + x)`."""

    def __init__(self, channels, kernel_size, activation=nn.ReLU):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, activation)
        self.conv2 = ConvBlock(channels, channels, kernel_size, activation=None)
        self.activation = activation()

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.activation(y + residual)
        return y


class PolicyHead(nn.Module):
    """1x1 conv to 2 channels -> flatten -> MLP(nn_width) -> MLP(out_features,
    no activation). Ports JAX `PolicyHead` `model_type != "mlp"` branch."""

    def __init__(self, in_channels, spatial_size, nn_width, out_features, activation=nn.ReLU):
        super().__init__()
        self.torso = nn.Sequential(
            ConvBlock(in_channels, 2, kernel_size=1, activation=activation),
            nn.Flatten(),
            nn.Linear(spatial_size * 2, nn_width),
            activation(),
        )
        self.policy_head = nn.Linear(nn_width, out_features)

    def forward(self, x):
        y = self.torso(x)
        return self.policy_head(y)


class ValueHead(nn.Module):
    """1x1 conv to 1 channel -> flatten -> MLP(nn_width) -> MLP(1) -> tanh.
    Ports JAX `ValueHead` `model_type != "mlp"` branch."""

    def __init__(self, in_channels, spatial_size, nn_width, activation=nn.ReLU):
        super().__init__()
        self.torso = nn.Sequential(
            ConvBlock(in_channels, 1, kernel_size=1, activation=activation),
            nn.Flatten(),
            nn.Linear(spatial_size * 1, nn_width),
            activation(),
        )
        self.value_head = nn.Sequential(nn.Linear(nn_width, 1), nn.Tanh())

    def forward(self, x):
        y = self.torso(x)
        return self.value_head(y).squeeze(-1)


class AlphaZeroModel(nn.Module):
    """AlphaZero-style resnet policy-value model: stem ConvBlock -> nn_depth
    stacked ResidualBlocks -> {PolicyHead, ValueHead}. Ports JAX
    `AlphaZeroModel.__init__`/`__call__` `model_type == "resnet"` branch
    (the "copies the one in [the AlphaZero] paper" variant)."""

    def __init__(self, input_shape, output_size, nn_width, nn_depth, activation=nn.ReLU):
        super().__init__()
        in_channels, h, w = input_shape
        self.stem = ConvBlock(in_channels, nn_width, kernel_size=3, activation=activation)
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(nn_width, kernel_size=3, activation=activation)
                for _ in range(nn_depth)
            ]
        )
        spatial_size = h * w
        self.policy_head = PolicyHead(nn_width, spatial_size, nn_width, output_size, activation)
        self.value_head = ValueHead(nn_width, spatial_size, nn_width, activation)

    def forward(self, observations):
        x = self.stem(observations)
        x = self.res_blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


def build_alphazero_resnet():
    # Small board-game-like observation, e.g. connect-four style: (C=3, H=6, W=7)
    return AlphaZeroModel(input_shape=(3, 6, 7), output_size=7, nn_width=16, nn_depth=3)


def example_input_alphazero_resnet():
    return (torch.randn(2, 3, 6, 7),)


MENAGERIE_ENTRIES = [
    (
        "AlphaZero (OpenSpiel resnet)",
        build_alphazero_resnet,
        example_input_alphazero_resnet,
        2018,
        "ported-pytorch",
    ),
]
