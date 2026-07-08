# FAITHFUL PORT of nsavinov/SPTM @ master (original framework: Keras (TF1-era, Theano-dim-ordering
# helpers) / raghakot/keras-resnet fork)
# https://github.com/nsavinov/SPTM/blob/master/src/common/resnet.py
# https://github.com/nsavinov/SPTM/blob/master/src/common/constants.py (NUM_EMBEDDING/TOP_HIDDEN/
# NORMALIZATION_ON/NET_* config values only)
#
# Savinov, Dosovitskiy & Koltun, "Semi-Parametric Topological Memory for Navigation"
# (ICLR 2018). The reachability estimator ("edge network") is a SIAMESE ResNet-18
# pair (`ResnetBuilder.build_siamese_resnet_18`): two weight-shared ResNet-18 branches
# (`build_resnet_18`, itself `raghakot/keras-resnet`, credited in the original file
# header) each embed one of the two input RGB frames into a `NUM_EMBEDDING`-dim vector,
# the two embeddings are concatenated, and a `TOP_HIDDEN`-deep
# BN->Dense->BN-ReLU "top network" classifies the pair as reachable/unreachable
# (2-way softmax). This IS the paper's core architectural contribution -- the
# siamese-embed + concat + MLP-classify reachability head -- so it is ported (not
# constructed from an unmodified base-library class).
#
# `resnet.py` is TF1-era Keras (`keras.layers.merge`, `keras.layers.normalization`,
# `K.image_dim_ordering()`, Python-2 `xrange`) which cannot run against any installed
# base lib. Every layer/mechanism in `ResnetBuilder.build`, `_residual_block`,
# `basic_block`, `_bn_relu`/`_conv_bn_relu`/`_bn_relu_conv` (the "improved" BN->ReLU->
# Conv pre-activation ordering the file cites from He et al. 1603.05027),
# `_shortcut` (1x1-conv projection shortcut only when the block changes
# spatial/channel shape), and the `_top_network` classifier head is transcribed
# FAITHFULLY into `torch.nn` below: BatchNorm2d/ReLU/Conv2d in the same pre-activation
# order, the same `[2, 2, 2, 2]` ResNet-18 repetition schedule with the same
# `is_first_block_of_first_layer` special-case (conv straight off the input, no
# leading BN->ReLU, matching upstream's "don't repeat bn->relu since we just did
# bn->relu->maxpool" comment), the same channel-doubling-per-stage / stride-2-except-
# first-stage schedule, the same 7x7 stride-2 stem + 3x3 stride-2 maxpool, the same
# global-average-pool + FC embedding head (`is_classification=False`, so
# `build_resnet_18` returns the raw `NUM_EMBEDDING`-dim embedding vector, matching how
# `build_siamese_resnet_18` calls it), and the same `_top_network` stack (per-layer
# BN->ReLU-for-dense before each of the `TOP_HIDDEN` Dense(NUM_EMBEDDING) blocks, then a
# final 2-way softmax Dense). `NORMALIZATION_ON` (L2-normalize each branch embedding) is
# `False` in upstream `constants.py`, matching the default here.
#
# Framework differences accounted for (no architectural change): Keras
# `Conv2D(padding="same")` pads asymmetrically for even kernel/stride combos in
# general, but every conv here uses odd kernel sizes (1x1/3x3/7x7) with Keras "same"
# padding, which is exactly torch's symmetric `padding=k//2` for those kernels -- so
# `nn.Conv2d(..., padding=k//2)` reproduces the same spatial semantics. Keras's default
# NHWC layout vs torch's NCHW is a pure memory-layout difference (both dimension-order
# branches in `_handle_dim_ordering` build the identical channels-first arithmetic
# graph once conceptually indexed by (batch, channel, row, col), which is what this
# port implements directly). `he_normal` Keras init corresponds to
# `nn.init.kaiming_normal_`; PyTorch's default `Conv2d`/`Linear` init is used for
# simplicity since the trace-validation build here is at tiny random scale (init
# scheme does not change graph topology/operator set).

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_EMBEDDING = 512
TOP_HIDDEN = 4
NORMALIZATION_ON = False


class BnReluConv(nn.Module):
    """BN -> ReLU -> Conv block (`_bn_relu_conv`)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        return self.conv(F.relu(self.bn(x)))


class ConvBnRelu(nn.Module):
    """Conv -> BN -> ReLU block (`_conv_bn_relu`, used only for the stem)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """`basic_block`: two 3x3 BN-ReLU-Conv units + projection shortcut."""

    def __init__(self, in_channels, out_channels, stride, is_first_block_of_first_layer):
        super().__init__()
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
        else:
            self.conv1 = BnReluConv(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = BnReluConv(out_channels, out_channels, kernel_size=3, stride=1)

        self.needs_shortcut_conv = (stride != 1) or (in_channels != out_channels)
        if self.needs_shortcut_conv:
            self.shortcut_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )

    def forward(self, x):
        residual = self.conv2(self.conv1(x))
        shortcut = self.shortcut_conv(x) if self.needs_shortcut_conv else x
        return shortcut + residual


def _make_residual_block(in_channels, out_channels, repetitions, is_first_layer):
    """`_residual_block`: stack `repetitions` BasicBlocks, downsampling on the first
    block of every stage except the network's very first stage."""
    layers = []
    channels = in_channels
    for i in range(repetitions):
        stride = 1 if (i != 0 or is_first_layer) else 2
        layers.append(
            BasicBlock(
                channels,
                out_channels,
                stride,
                is_first_block_of_first_layer=(is_first_layer and i == 0),
            )
        )
        channels = out_channels
    return nn.Sequential(*layers), channels


class ResNet18Embedding(nn.Module):
    """`ResnetBuilder.build_resnet_18(..., is_classification=False)`: stem + 4 stages
    of BasicBlocks ([2, 2, 2, 2] repetitions) + global-avg-pool + FC embedding head."""

    def __init__(self, in_channels, num_outputs):
        super().__init__()
        self.stem_conv = ConvBnRelu(in_channels, 64, kernel_size=7, stride=2)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        repetitions = [2, 2, 2, 2]
        stages = []
        channels = 64
        filters = 64
        for i, r in enumerate(repetitions):
            stage, channels = _make_residual_block(channels, filters, r, is_first_layer=(i == 0))
            stages.append(stage)
            filters *= 2
        self.stages = nn.ModuleList(stages)

        self.final_bn = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels, num_outputs)

    def forward(self, x):
        x = self.stem_pool(self.stem_conv(x))
        for stage in self.stages:
            x = stage(x)
        x = F.relu(self.final_bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


class BnReluDense(nn.Module):
    """`_bn_relu_for_dense`: BatchNorm1d -> ReLU."""

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return F.relu(self.bn(x))


class TopNetwork(nn.Module):
    """`_top_network`: BN-ReLU-for-dense before each of TOP_HIDDEN Dense(NUM_EMBEDDING)
    blocks, then a final 2-way softmax classifier."""

    def __init__(self, in_features, num_embedding=NUM_EMBEDDING, top_hidden=TOP_HIDDEN):
        super().__init__()
        self.pre_bn = BnReluDense(in_features)
        blocks = []
        feat = in_features
        for _ in range(top_hidden):
            blocks.append(nn.Linear(feat, num_embedding))
            blocks.append(BnReluDense(num_embedding))
            feat = num_embedding
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(feat, 2)

    def forward(self, x):
        x = self.pre_bn(x)
        for block in self.blocks:
            x = block(x)
        return F.softmax(self.classifier(x), dim=-1)


class SiameseResnet18Reachability(nn.Module):
    """`ResnetBuilder.build_siamese_resnet_18`: two weight-shared ResNet-18 branches
    embed the two stacked RGB frames, embeddings are concatenated, and the top
    network predicts 2-way reachable/unreachable."""

    def __init__(
        self,
        height=120,
        width=160,
        branch_channels=3,
        num_embedding=NUM_EMBEDDING,
        normalization_on=NORMALIZATION_ON,
    ):
        super().__init__()
        self.branch_channels = branch_channels
        self.normalization_on = normalization_on
        self.branch = ResNet18Embedding(branch_channels, num_embedding)
        self.top_network = TopNetwork(2 * num_embedding, num_embedding)

    def forward(self, x):
        """
        :param x: (batch, 2 * branch_channels, height, width) -- two RGB frames
            stacked on the channel axis (matches the Keras `Lambda` channel-slice
            used upstream to split the concatenated-frame input).
        """
        first = x[:, : self.branch_channels]
        second = x[:, self.branch_channels :]
        first_embed = self.branch(first)
        second_embed = self.branch(second)
        if self.normalization_on:
            first_embed = F.normalize(first_embed, p=2, dim=1)
            second_embed = F.normalize(second_embed, p=2, dim=1)
        raw_result = torch.cat([first_embed, second_embed], dim=1)
        return self.top_network(raw_result)


def build_sptm_reachability():
    """Tiny-config siamese reachability network (CPU, small spatial size)."""
    return SiameseResnet18Reachability(height=32, width=32, branch_channels=3, num_embedding=16)


def example_input_sptm_reachability():
    batch_size = 2
    x = torch.randn(batch_size, 6, 32, 32)
    return (x,)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "SPTM Siamese Reachability Network",
        "build_sptm_reachability",
        "example_input_sptm_reachability",
        2018,
        "navigation",
    ),
]
