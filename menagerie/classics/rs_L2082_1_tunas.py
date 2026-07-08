# FAITHFUL PORT of google-research/google-research @ 3d89d3becff1253b24625571314c785dc8cf8de4
# (original framework: TensorFlow 1.x compat.v1, Cloud-TPU-only training/search stack)
#
# TuNAS: "Can weight sharing outperform random architecture search? An investigation
# with TuNAS" (Bender, Liu, et al., CVPR 2020). tunas/ is a one-shot weight-sharing
# neural architecture search (NAS) framework built on top of a MobileNetV3-like search
# space (tunas/mobile_search_space_v3.py::mobilenet_v3_like_search). The search space
# and its OneOf/schema tree-walk pruning logic (tunas/schema.py, tunas/search_space_
# utils.py::prune_model_spec) are TF1.x/TPU-coupled and not reasonably installable in
# this base env (custom rematlib supernet layers, TPU stateless-batchnorm masking).
#
# This module faithfully transcribes ONE concrete TuNAS-discovered architecture: the
# top-accuracy `mobilenet_v3_like_search` genotype released by the authors in
# searched_architectures.csv (http://storage.googleapis.com/gresearch/tunas/
# searched_architectures.csv), indices string:
#   0:0:1:1:1:3:1:0:0:1:1:0:0:0:1:0:0:0:1:2:2:4:1:0:2:1:0:0:1:0:0:0:2:0:0:3:2:5:1:0:2:
#   0:0:0:1:0:0:1:1:1:0:4:1:5:1:0:0:0:1:0:1:1:0:0:0:1:0:3:1:5:1:2:1:1:0:2:1:1:0:2:1:1:
#   0:2:6:6
# (simulated_pixel1_time_ms=56.58, 90epoch_validation_accuracy=0.7645,
#  360epoch_test_accuracy=0.7541 -- the best of the 5 released mobilenet_v3_like_search
#  rows). The genotype was decoded by re-implementing the exact tree-walk used by the
#  real code (tunas/schema.py::OneOf / map_oneofs post-order traversal, tunas/
#  search_space_utils.py::prune_model_spec index-popping, tunas/mobile_search_space_v3.
#  py::_mobilenet_v3_large_search_base / mobilenet_v3_like_search block topology and
#  choice lists, tunas/search_space_utils.py::scale_filters rounding) against the real
#  released indices -- not guessed from a paper description. The resulting per-block
#  kernel size / expansion filters / squeeze-excite / activation / optional-skip choices
#  below are the exact decode output; every MBConv block mirrors
#  tunas/rematlib/mobile_model_v3.py's DepthwiseBottleneck (expand 1x1 -> depthwise KxK
#  -> optional SE -> project 1x1, swish6/relu6 activations, residual only for stride=1
#  same-channel blocks, blocks that decoded to the ZeroSpec choice are dropped as in the
#  original one-shot pruning).

from __future__ import annotations

import torch
from torch import Tensor, nn


def _make_divisible(value: int, divisor: int = 8) -> int:
    """Round `value` up to the nearest multiple of `divisor` (min `divisor`)."""

    return max(divisor, int(round(value / divisor)) * divisor)


class SqueezeExcite(nn.Module):
    """Squeeze-and-excite gate, matching TuNAS's default se ratio of 0.25."""

    def __init__(self, channels: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        reduced = max(1, _make_divisible(int(channels * se_ratio)))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, 1)
        self.act = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(reduced, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.avg_pool(x)
        scale = self.act(self.fc1(scale))
        scale = self.gate(self.fc2(scale))
        return x * scale


def _activation(name: str) -> nn.Module:
    if name == "swish6":
        # TuNAS's swish6 = x * relu6(x + 3) / 6 (hard-swish); rematlib uses this as
        # a TPU-friendly swish approximation.
        return nn.Hardswish(inplace=False)
    if name == "relu":
        return nn.ReLU6(inplace=False)
    raise ValueError(f"unsupported activation {name!r}")


class DepthwiseBottleneck(nn.Module):
    """MBConv block: expand 1x1 -> depthwise KxK -> optional SE -> project 1x1.

    Mirrors tunas/rematlib/mobile_model_v3.py's translation of DepthwiseBottleneckSpec.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expansion_filters: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        activation: str,
        residual: bool,
    ) -> None:
        super().__init__()
        self.residual = residual
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        if expansion_filters != in_ch:
            layers += [
                nn.Conv2d(in_ch, expansion_filters, 1, bias=False),
                nn.BatchNorm2d(expansion_filters),
                _activation(activation),
            ]
        layers += [
            nn.Conv2d(
                expansion_filters,
                expansion_filters,
                kernel_size,
                stride=stride,
                padding=pad,
                groups=expansion_filters,
                bias=False,
            ),
            nn.BatchNorm2d(expansion_filters),
        ]
        self.pre_se = nn.Sequential(*layers)
        self.act_after_dw = _activation(activation)
        self.se = SqueezeExcite(expansion_filters) if use_se else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(expansion_filters, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.pre_se(x)
        y = self.act_after_dw(y)
        y = self.se(y)
        y = self.project(y)
        return x + y if self.residual else y


class ConvBNAct(nn.Module):
    """Plain conv (+ optional BN) (+ optional activation), used for stem/head convs."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        use_bn: bool,
        activation: str | None,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, bias=not use_bn)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation is not None:
            layers.append(_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TuNASMobileNetV3Like(nn.Module):
    """One concrete architecture sampled from TuNAS's mobilenet_v3_like_search space.

    Topology and per-block choices (kernel size, expansion filters, squeeze-excite,
    activation, and which optional residual blocks survive pruning) were decoded from
    the real released best-accuracy `mobilenet_v3_like_search` genotype -- see module
    docstring for the exact indices string and decode method.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.stem = ConvBNAct(3, 16, kernel_size=3, stride=2, use_bn=True, activation="swish6")
        # residual(optional(sepconv)) at the stem position decoded to ZeroSpec -> dropped.

        self.stage2 = nn.Sequential(
            DepthwiseBottleneck(
                16, 24, 64, kernel_size=5, stride=2, use_se=True, activation="relu", residual=False
            ),
            # 3 optional residual bnecks in this block all decoded to ZeroSpec -> dropped.
        )

        self.stage3 = nn.Sequential(
            DepthwiseBottleneck(
                24, 40, 120, kernel_size=7, stride=2, use_se=True, activation="relu", residual=False
            ),
            DepthwiseBottleneck(
                40, 40, 120, kernel_size=3, stride=1, use_se=True, activation="relu", residual=True
            ),
            DepthwiseBottleneck(
                40, 40, 80, kernel_size=3, stride=1, use_se=False, activation="relu", residual=True
            ),
            DepthwiseBottleneck(
                40, 40, 120, kernel_size=3, stride=1, use_se=False, activation="relu", residual=True
            ),
        )

        self.stage4 = nn.Sequential(
            DepthwiseBottleneck(
                40,
                80,
                240,
                kernel_size=7,
                stride=2,
                use_se=True,
                activation="swish6",
                residual=False,
            ),
            DepthwiseBottleneck(
                80,
                80,
                240,
                kernel_size=3,
                stride=1,
                use_se=False,
                activation="swish6",
                residual=True,
            ),
            DepthwiseBottleneck(
                80,
                80,
                160,
                kernel_size=3,
                stride=1,
                use_se=False,
                activation="swish6",
                residual=True,
            ),
            DepthwiseBottleneck(
                80,
                80,
                160,
                kernel_size=5,
                stride=1,
                use_se=True,
                activation="swish6",
                residual=True,
            ),
        )

        self.stage5 = nn.Sequential(
            DepthwiseBottleneck(
                80,
                128,
                480,
                kernel_size=5,
                stride=1,
                use_se=True,
                activation="swish6",
                residual=False,
            ),
            # first optional residual bneck decoded to ZeroSpec -> dropped.
            DepthwiseBottleneck(
                128,
                128,
                256,
                kernel_size=3,
                stride=1,
                use_se=True,
                activation="swish6",
                residual=True,
            ),
            DepthwiseBottleneck(
                128,
                128,
                128,
                kernel_size=3,
                stride=1,
                use_se=True,
                activation="swish6",
                residual=True,
            ),
        )

        self.stage6 = nn.Sequential(
            DepthwiseBottleneck(
                128,
                192,
                768,
                kernel_size=5,
                stride=2,
                use_se=True,
                activation="swish6",
                residual=False,
            ),
            DepthwiseBottleneck(
                192,
                192,
                384,
                kernel_size=7,
                stride=1,
                use_se=True,
                activation="swish6",
                residual=True,
            ),
            DepthwiseBottleneck(
                192,
                192,
                384,
                kernel_size=7,
                stride=1,
                use_se=True,
                activation="swish6",
                residual=True,
            ),
            DepthwiseBottleneck(
                192,
                192,
                384,
                kernel_size=7,
                stride=1,
                use_se=True,
                activation="swish6",
                residual=True,
            ),
        )

        self.head_conv1 = ConvBNAct(
            192, 1024, kernel_size=1, stride=1, use_bn=True, activation="swish6"
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head_conv2 = ConvBNAct(
            1024, 2048, kernel_size=1, stride=1, use_bn=False, activation="swish6"
        )
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, image: Tensor) -> Tensor:
        x = self.stem(image)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.head_conv1(x)
        x = self.avgpool(x)
        x = self.head_conv2(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_tunas_mbv3like() -> nn.Module:
    """Build the decoded TuNAS mobilenet_v3_like_search architecture (random init)."""

    return TuNASMobileNetV3Like(num_classes=10).eval()


def example_input_tunas_mbv3like() -> Tensor:
    """Return a small RGB classification input."""

    return torch.randn(1, 3, 96, 96)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    (
        "TuNAS-MobileNetV3Like",
        "build_tunas_mbv3like",
        "example_input_tunas_mbv3like",
        "2020",
        "CV",
    )
]
