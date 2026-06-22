"""Efficient CNN Architectures: PackNet, CondenseNet, MSDNet.

PackNet: Mallya & Lazebnik 2018, arXiv:1711.05769
CondenseNet: Huang et al. 2018, arXiv:1711.09224
MSDNet: Huang et al. 2018, arXiv:1703.09844

Paper (PackNet): https://arxiv.org/abs/1711.05769
Paper (CondenseNet): https://arxiv.org/abs/1711.09224
Paper (MSDNet): https://arxiv.org/abs/1703.09844
Source (PackNet): https://github.com/arunmallya/packnet
Source (CondenseNet): https://github.com/ShichenLiu/CondenseNet
Source (MSDNet): https://github.com/gaohuang/MSDNet

PackNet-VGG16:
  VGG16 backbone with per-task iterative pruning using binary weight masks.
  The forward graph = VGG16 feature layers + classifier. Masks are applied as
  elementwise multiplications (masking out pruned weights). Compact version:
  channels reduced 8x (64->8, 128->16, 256->32, 512->64) for fast tracing.
  The per-task mask registry is the PackNet primitive; shown as constant 1s here.

CondenseNet:
  Dense connectivity with learned group convolution (LGC): a 1x1 conv where
  each group's input connections are learned/pruned during training, then
  followed by group conv. The LGC + permute/index layer is the primitive.
  Compact DenseNet-like structure: 3 dense blocks with LGC bottleneck.
  Growth rate=8, groups=4, compact depth.

MSDNet (Multi-Scale DenseNet):
  Multi-scale feature maps maintained in parallel across layers (coarse/fine
  scales), with dense cross-scale connections + early-exit classifiers.
  The multi-scale grid (S scales x L layers) with cross-scale dense connections
  and multiple output classifiers is the distinctive primitive.
  Compact: 3 scales, 3 layers per block, 2 early-exit classifiers, C=16.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# PackNet-VGG16
# ===========================================================================


class MaskedConv2d(nn.Conv2d):
    """Conv2d with a binary pruning mask (PackNet masking primitive)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Start with all weights unmasked (all 1s)
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mask: pruned weights become 0
        masked_weight = self.weight * self.mask
        return F.conv2d(
            x, masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def _masked_conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, pad: int = 1) -> nn.Sequential:
    return nn.Sequential(
        MaskedConv2d(in_ch, out_ch, k, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PackNetVGG16(nn.Module):
    """VGG16 with PackNet binary weight masks for iterative pruning + task packing.

    PackNet (Mallya & Lazebnik 2018): sequentially prune and retrain for each
    new task; binary masks store which weights belong to each task.
    The forward graph is VGG16 with elementwise mask multiplications on all conv weights.
    Compact: channels scaled 8x down (64->8 etc.) for fast tracing.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # VGG16 topology, compact channels (8x reduction)
        self.features = nn.Sequential(
            # Block 1: 2 conv, 64 ch (compact: 8)
            _masked_conv_bn_relu(3, 8),
            _masked_conv_bn_relu(8, 8),
            nn.MaxPool2d(2, 2),
            # Block 2: 2 conv, 128 ch (compact: 16)
            _masked_conv_bn_relu(8, 16),
            _masked_conv_bn_relu(16, 16),
            nn.MaxPool2d(2, 2),
            # Block 3: 3 conv, 256 ch (compact: 32)
            _masked_conv_bn_relu(16, 32),
            _masked_conv_bn_relu(32, 32),
            _masked_conv_bn_relu(32, 32),
            nn.MaxPool2d(2, 2),
            # Block 4: 3 conv, 512 ch (compact: 64)
            _masked_conv_bn_relu(32, 64),
            _masked_conv_bn_relu(64, 64),
            _masked_conv_bn_relu(64, 64),
            nn.MaxPool2d(2, 2),
            # Block 5: 3 conv, 512 ch (compact: 64)
            _masked_conv_bn_relu(64, 64),
            _masked_conv_bn_relu(64, 64),
            _masked_conv_bn_relu(64, 64),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)


# ===========================================================================
# CondenseNet
# ===========================================================================


class LearnedGroupConv(nn.Module):
    """Learned Group Convolution (LGC) -- the CondenseNet primitive.

    During training, input connections for each group are learned/pruned.
    Here we simulate post-condensation structure with fixed groups + index layer.
    Each group receives condensed_feature_size inputs (uniformly selected).
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 4) -> None:
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        # After condensation: each group picks in_channels // groups inputs
        self.condense_factor = max(1, in_channels // groups)
        # Index: which input channels each group reads (static post-condensation)
        # Stored as a plain Python list so TorchLens doesn't try to trace it as a buffer.
        idx = []
        for g in range(groups):
            for i in range(self.condense_factor):
                idx.append((g * self.condense_factor + i) % in_channels)
        self._idx: list[int] = idx
        # Group conv: groups groups, each taking condense_factor channels -> out_channels//groups
        self.group_conv = nn.Conv2d(
            groups * self.condense_factor,
            out_channels,
            kernel_size=1,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Index-select (permute) input channels per group
        x = x[:, self._idx, :, :]
        x = self.group_conv(x)
        return self.bn(x)


class CondenseDenseLayer(nn.Module):
    """One dense layer in CondenseNet: BN-ReLU-LGC_1x1 -> BN-ReLU-GroupConv_3x3."""

    def __init__(self, in_ch: int, growth_rate: int, groups: int = 4) -> None:
        super().__init__()
        bottleneck_ch = growth_rate * 4
        self.lgc = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            LearnedGroupConv(in_ch, bottleneck_ch, groups=groups),
        )
        self.conv3x3 = nn.Sequential(
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(),
            nn.Conv2d(bottleneck_ch, growth_rate, 3, padding=1, groups=groups, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv3x3(self.lgc(x))
        return torch.cat([x, new_features], dim=1)


class CondenseTransition(nn.Module):
    """Transition block between dense stages (avg pool + 1x1 conv)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(self.relu(self.bn(x))))


class CondenseNet(nn.Module):
    """CondenseNet (Huang et al. 2018): dense connectivity + learned group convolutions.

    Three dense blocks with LGC bottleneck layers and transition pooling.
    Compact: growth_rate=8, groups=4, 3 layers per block.
    """

    def __init__(self, num_classes: int = 10, growth_rate: int = 8, groups: int = 4) -> None:
        super().__init__()
        init_ch = growth_rate * 2
        self.stem = nn.Conv2d(3, init_ch, 3, padding=1, bias=False)

        # Dense block 1
        ch = init_ch
        self.dense1 = self._make_dense_block(ch, growth_rate, groups, n_layers=3)
        ch = ch + 3 * growth_rate
        self.trans1 = CondenseTransition(ch, ch // 2)
        ch = ch // 2

        # Dense block 2
        self.dense2 = self._make_dense_block(ch, growth_rate, groups, n_layers=3)
        ch = ch + 3 * growth_rate
        self.trans2 = CondenseTransition(ch, ch // 2)
        ch = ch // 2

        # Dense block 3
        self.dense3 = self._make_dense_block(ch, growth_rate, groups, n_layers=3)
        ch = ch + 3 * growth_rate

        self.bn_final = nn.BatchNorm2d(ch)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, num_classes)

    def _make_dense_block(
        self, in_ch: int, growth: int, groups: int, n_layers: int
    ) -> nn.Sequential:
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(CondenseDenseLayer(ch, growth, groups))
            ch += growth
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = F.relu(self.bn_final(x))
        x = self.gap(x).flatten(1)
        return self.fc(x)


# ===========================================================================
# MSDNet (Multi-Scale Dense Network with Early Exit)
# ===========================================================================


class MSDLayer(nn.Module):
    """One MSDNet layer: processes S scales, takes inputs from all prior layers at all scales.

    For each scale s: output_s = conv(concat(all_prior_scale_s_outputs)) +
                                 downsample_conv(scale_{s-1} if s>0)
    This implements the cross-scale dense connection signature.
    """

    def __init__(self, in_channels_per_scale: List[int], out_ch: int, n_scales: int) -> None:
        super().__init__()
        self.n_scales = n_scales
        # Same-scale dense connection conv
        self.same_scale_convs = nn.ModuleList()
        # Cross-scale (finer -> coarser) conv
        self.cross_scale_convs = nn.ModuleList()
        for s in range(n_scales):
            in_ch = in_channels_per_scale[s]
            if s > 0:
                # also takes output from scale s-1 (finer) via stride-2 conv
                in_ch_from_finer = in_channels_per_scale[s - 1]
                self.cross_scale_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch_from_finer, out_ch, 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
                    )
                )
            else:
                self.cross_scale_convs.append(None)  # type: ignore[arg-type]
            self.same_scale_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                )
            )

    def forward(self, scale_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        new_feats = []
        for s in range(self.n_scales):
            h = self.same_scale_convs[s](scale_feats[s])
            if s > 0 and self.cross_scale_convs[s] is not None:
                h = h + self.cross_scale_convs[s](scale_feats[s - 1])
            new_feats.append(h)
        return new_feats


class MSDClassifier(nn.Module):
    """Early-exit classifier for one scale in MSDNet."""

    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.gap(self.conv(x)).flatten(1))


class MSDNet(nn.Module):
    """Multi-Scale Dense Network (Huang et al. 2018) with early-exit classifiers.

    Architecture: multi-scale feature grid (S scales x L layers) with dense
    cross-scale connections. Early-exit classifiers after intermediate blocks.
    The multi-scale grid + early exits is the distinctive primitive.

    Compact: 3 scales, 3 MSD layers per block, 2 exit classifiers, C=16.
    """

    def __init__(
        self,
        in_ch: int = 3,
        n_scales: int = 3,
        base_ch: int = 16,
        layers_per_block: int = 3,
        n_exits: int = 2,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales
        self.n_exits = n_exits

        # Stem: produce initial multi-scale features
        # Scale 0: full res; scale s: 1/(2^s) res
        self.stem = nn.ModuleList()
        for s in range(n_scales):
            stride = 2**s
            self.stem.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, base_ch, 3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(base_ch),
                    nn.ReLU(),
                )
            )

        # MSD blocks with early exits
        self.blocks = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        ch = base_ch  # channels per scale after each block
        in_chs = [base_ch] * n_scales
        for exit_idx in range(n_exits):
            block_layers = nn.ModuleList()
            for _ in range(layers_per_block):
                layer = MSDLayer(in_chs, ch, n_scales)
                block_layers.append(layer)
                in_chs = [ch] * n_scales
            self.blocks.append(block_layers)
            # Classifier reads coarsest scale
            self.classifiers.append(MSDClassifier(ch, num_classes))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Build initial multi-scale feature list
        feats = [stem(x) for stem in self.stem]

        outputs = []
        for exit_idx, block_layers in enumerate(self.blocks):
            for layer in block_layers:
                feats = layer(feats)
            # Early exit on coarsest scale
            out = self.classifiers[exit_idx](feats[-1])
            outputs.append(out)
        return outputs


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def build_packnet_vgg16() -> nn.Module:
    return PackNetVGG16()


def build_condensenet() -> nn.Module:
    return CondenseNet()


def build_msdnet() -> nn.Module:
    return MSDNet()


# ---------------------------------------------------------------------------
# Example inputs
# ---------------------------------------------------------------------------


def example_input_vgg() -> torch.Tensor:
    return torch.zeros(1, 3, 32, 32)


def example_input_condensenet() -> torch.Tensor:
    return torch.zeros(1, 3, 32, 32)


def example_input_msdnet() -> torch.Tensor:
    return torch.zeros(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MENAGERIE_ENTRIES = [
    ("PackNet-VGG16", "build_packnet_vgg16", "example_input_vgg", "2018", "DC"),
    ("CondenseNet", "build_condensenet", "example_input_condensenet", "2018", "DC"),
    ("MSDNet", "build_msdnet", "example_input_msdnet", "2018", "DC"),
]
