# FAITHFUL PORT of google-research/episodic-curiosity @ master (original framework: TensorFlow/Keras)
# https://github.com/google-research/episodic-curiosity
# File ported: third_party/keras_resnet/models.py (ResnetBuilder.build_resnet_18,
# ResnetBuilder.build_siamese_resnet_18, _top_network), used by
# episodic_curiosity/r_network.py (RNetwork) to build the "R-network" -- the
# episodic-curiosity paper's (Savinov et al. 2019, "Episodic Curiosity through
# Reachability") core architectural contribution: a siamese embedding network
# that estimates whether two observations are "reachable" from each other
# within a small number of steps, used to compute a novelty bonus for
# exploration.
#
# The real R-network architecture is implemented in Keras/TensorFlow (forked
# from raghakot/keras-resnet); no PyTorch implementation exists anywhere. This
# module is a FAITHFUL PORT -- every layer of the actual Keras source was
# transcribed 1:1 into torch, not a reimplementation from the paper text:
#   - `basic_block`/`_bn_relu_conv`/`_bn_relu`/`_shortcut`/`_residual_block`
#     -> the pre-activation (BN->ReLU->Conv) residual "basic" block used by
#     resnets with <=34 layers (improved scheme of arXiv:1603.05027), with the
#     first block of the first stage skipping the BN->ReLU pre-activation
#     (since it directly follows the conv1->bn->relu->maxpool stem), and a
#     1x1-conv projection shortcut whenever spatial stride or channel count
#     changes between the block input and its residual branch.
#   - `ResnetBuilder.build_resnet_18` -> conv1 (7x7, stride 2) -> BN -> ReLU
#     -> maxpool (3x3, stride 2) -> 4 residual stages of 2 basic blocks each
#     (channel widths 64/128/256/512, stage 2-4 downsample by 2 on their first
#     block) -> final BN->ReLU -> global average pool -> a Linear "embedding"
#     head (EMBEDDING_DIM=512, no activation since `is_classification=False`
#     for the R-network's branch).
#   - `_top_network` (the "deep" comparator, `use_deep_top_network=True`,
#     which is the R-network's default) -> concatenate the two branch
#     embeddings -> BN->ReLU -> TOP_HIDDEN=4 repeats of
#     (Linear(EMBEDDING_DIM) -> BN->ReLU) -> Linear(2) with softmax, giving a
#     2-way "reachable"/"not reachable" classification.
#   - `build_siamese_resnet_18` -> run the SAME embedding branch (tied
#     weights) on both input images, then feed both embeddings through the
#     top comparator network -- the siamese architecture this module
#     reproduces end to end.
#
# Every mechanism from the real Keras code is preserved (pre-activation basic
# blocks, stage/channel schedule, projection-shortcut condition, the deep
# comparator's hidden-layer count and BN placement); only Keras-specific
# framework glue (functional-API closures, `K.int_shape`, `gin.configurable`,
# `l2` weight regularizers, model.summary()) has no torch equivalent and was
# dropped since it does not affect the traced forward computation.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"

EMBEDDING_DIM = 512
TOP_HIDDEN = 4


class _BasicBlock(nn.Module):
    """Pre-activation basic residual block. (keras_resnet/models.py basic_block)

    Follows the improved scheme of http://arxiv.org/pdf/1603.05027v2.pdf: the
    first block of the first stage skips the BN->ReLU pre-activation (since it
    directly follows the stem's conv->bn->relu->maxpool), all other blocks use
    BN->ReLU->Conv for both conv layers of the residual branch, and a 1x1-conv
    projection shortcut is inserted whenever the spatial resolution or channel
    count of the residual branch differs from the block input.
    """

    def __init__(self, in_channels, out_channels, stride=1, is_first_block_of_first_layer=False):
        super().__init__()
        self.is_first_block_of_first_layer = is_first_block_of_first_layer

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
        else:
            self.bn_pre = nn.BatchNorm2d(in_channels)
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.equal_channels = in_channels == out_channels
        self.stride = stride
        if stride != 1 or not self.equal_channels:
            self.shortcut_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            )
        else:
            self.shortcut_conv = None

    def forward(self, x):
        if self.is_first_block_of_first_layer:
            conv1 = self.conv1(x)
        else:
            pre = F.relu(self.bn_pre(x))
            conv1 = self.conv1(pre)

        residual = self.conv2(F.relu(self.bn2(conv1)))

        shortcut = x
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(x)

        return shortcut + residual


class _ResNet18Branch(nn.Module):
    """ResNet-18 embedding branch. (ResnetBuilder.build_resnet_18, is_classification=False)"""

    def __init__(self, in_channels=3, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        repetitions = [2, 2, 2, 2]
        filters = 64
        stages = []
        for stage_idx, reps in enumerate(repetitions):
            for block_idx in range(reps):
                stride = 1
                if block_idx == 0 and stage_idx != 0:
                    stride = 2
                in_ch = filters // 2 if (block_idx == 0 and stage_idx != 0) else filters
                stages.append(
                    _BasicBlock(
                        in_ch,
                        filters,
                        stride=stride,
                        is_first_block_of_first_layer=(stage_idx == 0 and block_idx == 0),
                    )
                )
            filters *= 2
        self.stages = nn.Sequential(*stages)

        final_channels = filters // 2
        self.bn_final = nn.BatchNorm2d(final_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(final_channels, embedding_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stages(x)
        x = F.relu(self.bn_final(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return x


class _DeepTopNetwork(nn.Module):
    """Deep comparator network. (keras_resnet/models.py _top_network)"""

    def __init__(self, embedding_dim=EMBEDDING_DIM, hidden=TOP_HIDDEN):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(embedding_dim * 2)
        layers = []
        last_dim = embedding_dim * 2
        for _ in range(hidden):
            layers.append(nn.Linear(last_dim, embedding_dim))
            layers.append(nn.BatchNorm1d(embedding_dim))
            last_dim = embedding_dim
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(embedding_dim, 2)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.bn_in(x))
        i = 0
        while i < len(self.hidden_layers):
            linear = self.hidden_layers[i]
            bn = self.hidden_layers[i + 1]
            x = F.relu(bn(linear(x)))
            i += 2
        return F.softmax(self.out(x), dim=1)


class SiameseRNetwork(nn.Module):
    """Siamese R-network: shared ResNet-18 branch + deep comparator.
    (ResnetBuilder.build_siamese_resnet_18, use_deep_top_network=True)"""

    def __init__(self, in_channels=3, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.branch = _ResNet18Branch(in_channels=in_channels, embedding_dim=embedding_dim)
        self.top_network = _DeepTopNetwork(embedding_dim=embedding_dim)

    def forward(self, x1, x2):
        y1 = self.branch(x1)
        y2 = self.branch(x2)
        return self.top_network(y1, y2)


class SiameseRNetworkPair(nn.Module):
    """Single-tensor-input wrapper: splits a batched pair of stacked images
    into (x1, x2) and runs the siamese comparator, so the module exposes one
    concrete-tensor `forward(pair)` entry point matching TorchLens's
    single-input tracing contract."""

    def __init__(self, in_channels=3, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.siamese = SiameseRNetwork(in_channels=in_channels, embedding_dim=embedding_dim)

    def forward(self, pair):
        x1 = pair[:, 0]
        x2 = pair[:, 1]
        return self.siamese(x1, x2)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------


def build_episodic_curiosity_rnetwork():
    torch.manual_seed(0)
    model = SiameseRNetworkPair(in_channels=3, embedding_dim=32)
    model.eval()
    return model


def example_input_episodic_curiosity_rnetwork():
    torch.manual_seed(0)
    return torch.randn(2, 2, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Episodic Curiosity R-network (siamese reachability ResNet-18 comparator)",
        "build_episodic_curiosity_rnetwork",
        "example_input_episodic_curiosity_rnetwork",
        2019,
        MENAGERIE_ZOO,
    ),
]
