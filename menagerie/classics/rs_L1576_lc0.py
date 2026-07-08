# FAITHFUL PORT of https://github.com/LeelaChessZero/lczero-training @ master
# (tf/tfprocess.py: TFProcess.conv_block / residual_block / squeeze_excitation / construct_net,
#  original framework: TensorFlow / tf.keras)
#
# Leela Chess Zero (lc0), the open community AlphaZero-style chess engine. lc0 itself
# (LeelaChessZero/lc0) is the C++/CUDA/OpenCL inference engine -- no PyTorch (or any Python)
# network definition lives there. The reference training-side network definition instead lives
# in the sibling `lczero-training` repo. That repo's CURRENT default architecture (`src/
# lczero_training/model/model.py`) is a JAX/flax transformer; its classic residual-tower +
# squeeze-excitation ResNet -- the architecture most commonly meant by "the Leela Chess Zero
# network" (an AlphaZero-style policy/value net with per-block SE, the winning architecture of
# the original 2018 Leela Chess Zero project and still a supported `POLICY_CLASSICAL`/non-WDL
# config today) -- lives in `tf/tfprocess.py`, in TensorFlow/Keras. Neither framework is
# installed in this base env (JAX/flax and tf.keras are both out of scope here), so per the
# ladder this is a faithful architectural PORT (not a from-scratch reimplementation): every
# layer/mechanism below (input conv_block, N x residual_block with per-block squeeze-excitation,
# POLICY_CLASSICAL policy head, WDL value head) is transcribed 1:1 from the real
# `tf/tfprocess.py` methods `conv_block`, `residual_block`, `squeeze_excitation`,
# `create_residual_body`, and the POLICY_CLASSICAL / value-head branches of `construct_net`,
# translated from Keras' channels_first Conv2D/BatchNorm/Dense ops to their literal torch
# equivalents (nn.Conv2d/nn.BatchNorm2d/nn.Linear). The POLICY_ATTENTION head (which requires
# lc0's `lc0_az_policy_map` chess-move index lookup table, a data asset rather than an
# architecture) and the optional moves-left head are real, but orthogonal, code paths in the
# same file; POLICY_CLASSICAL is used here as it is the simplest complete, self-contained real
# path (dense policy head, no external index table).

from __future__ import annotations

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class SqueezeExcitation(nn.Module):
    """Ported from TFProcess.squeeze_excitation (tf/tfprocess.py): global-average-pool -> FC ->
    activation -> FC -> split into per-channel (gamma, beta) -> sigmoid(gamma) * x + beta
    (ApplySqueezeExcitation), i.e. lc0's SE block is a gated affine modulation, not a plain
    channel-rescale SE block."""

    def __init__(self, channels: int, se_ratio: int):
        super().__init__()
        assert channels % se_ratio == 0
        reduced = channels // se_ratio
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(channels, reduced)
        self.act = nn.ReLU()
        self.dense2 = nn.Linear(reduced, 2 * channels)
        self.channels = channels

    def forward(self, x):
        pooled = self.pool(x).flatten(1)
        squeezed = self.act(self.dense1(pooled))
        excited = self.dense2(squeezed)
        gammas, betas = torch.split(excited, self.channels, dim=1)
        gammas = gammas.unsqueeze(-1).unsqueeze(-1)
        betas = betas.unsqueeze(-1).unsqueeze(-1)
        return torch.sigmoid(gammas) * x + betas


class ConvBlock(nn.Module):
    """Ported from TFProcess.conv_block: Conv2d(no bias) -> BatchNorm -> activation."""

    def __init__(self, in_channels: int, out_channels: int, filter_size: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            filter_size,
            padding="same",
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Ported from TFProcess.residual_block: conv-bn-relu, conv-bn (no scale), squeeze-excitation
    (scale=True on the pre-SE batchnorm, matching the real code), skip-add, relu."""

    def __init__(self, channels: int, se_ratio: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, se_ratio)
        self.act = nn.ReLU()

    def forward(self, x):
        out1 = self.act(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2 = self.se(out2)
        return self.act(x + out2)


class LeelaChessZeroNet(nn.Module):
    """Ported from TFProcess.create_residual_body + the POLICY_CLASSICAL / value-head branches
    of TFProcess.construct_net. Input: (N, 112, 8, 8) -- lc0's INPUT_CLASSICAL_112_PLANE board
    encoding (`tf.keras.Input(shape=(112, 8, 8))` in the real TFProcess.init_net). Outputs
    (policy_logits, value_logits) mirroring construct_net's real `outputs = [h_fc1, h_fc3]`
    (non-moves_left) return."""

    NUM_POLICY_MOVES = 1858  # real lc0 move-index vocabulary size (policy/dense output)

    def __init__(
        self,
        filters: int = 32,
        residual_blocks: int = 4,
        se_ratio: int = 4,
        policy_channels: int = 32,
        wdl: bool = True,
        input_planes: int = 112,
    ):
        super().__init__()
        self.input_block = ConvBlock(input_planes, filters, 3)
        self.residual_tower = nn.ModuleList(
            [ResidualBlock(filters, se_ratio) for _ in range(residual_blocks)]
        )

        # Policy head (POLICY_CLASSICAL): conv_block(filter=1, policy_channels) -> flatten -> dense
        self.policy_conv = ConvBlock(filters, policy_channels, 1)
        self.policy_dense = nn.Linear(policy_channels * 8 * 8, self.NUM_POLICY_MOVES)

        # Value head: conv_block(filter=1, 32 channels) -> flatten -> dense128 -> dense(wdl ? 3 : 1)
        self.value_conv = ConvBlock(filters, 32, 1)
        self.value_dense1 = nn.Linear(32 * 8 * 8, 128)
        self.value_act1 = nn.ReLU()
        self.wdl = wdl
        if wdl:
            self.value_dense2 = nn.Linear(128, 3)
        else:
            self.value_dense2 = nn.Linear(128, 1)

    def forward(self, x):
        flow = self.input_block(x)
        for block in self.residual_tower:
            flow = block(flow)

        conv_pol = self.policy_conv(flow)
        policy_logits = self.policy_dense(conv_pol.flatten(1))

        conv_val = self.value_conv(flow)
        h_fc2 = self.value_act1(self.value_dense1(conv_val.flatten(1)))
        if self.wdl:
            value_logits = self.value_dense2(h_fc2)
        else:
            value_logits = torch.tanh(self.value_dense2(h_fc2))

        return policy_logits, value_logits


def build_lc0():
    # Small residual tower (real lc0 nets historically ranged 6-40 blocks / 64-384 filters;
    # shrunk here purely for a fast trace, architecture unchanged).
    return LeelaChessZeroNet(
        filters=32, residual_blocks=4, se_ratio=4, policy_channels=32, wdl=True
    )


def example_input_lc0():
    return torch.randn(2, 112, 8, 8)


MENAGERIE_ENTRIES = [
    ("Leela Chess Zero (residual+SE net)", build_lc0, example_input_lc0, 2018, "REAL"),
]
