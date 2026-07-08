# FAITHFUL PORT of tensorflow/models @ master (original framework: TensorFlow 1.x
# / tf_slim, research/object_detection/models/ssd_mobiledet_feature_extractor.py)
#
# MobileDets (Xiong, Liu, Chen, Tomizuka, Zhan, Chen, Ren, Chen, Long, Wan, Choi,
# Keutzer. 2021, CVPR, "MobileDets: Searching for Object Detection Architectures
# for Mobile Accelerators", arXiv:2004.14525). A NAS-searched mobile object
# detection backbone family whose key finding is that plain depthwise-separable
# inverted bottlenecks (as in MobileNetV2/V3) are NOT always optimal for mobile
# accelerators: MobileDets' search space additionally includes "full convolution"
# regular convs, "fused" inverted bottlenecks (fuse the 1x1 expansion + depthwise
# into one full-size conv, no depthwise at all), and "Tucker" convolutions
# (a generalized low-rank bottleneck: 1x1 input-rank-reduction -> full-size conv
# at reduced rank -> 1x1 output-projection), alongside standard depthwise inverted
# bottlenecks with optional squeeze-and-excite -- letting NAS pick per-block which
# primitive is fastest on the target accelerator. This ports the CPU-targeted
# backbone variant ("MobileDet-CPU" in the paper), which searches purely with
# inverted-bottleneck + squeeze-excite + swish6 blocks (no fused/Tucker blocks are
# selected in that variant's found architecture).
#
# The real repo is TensorFlow-1.x (`tensorflow.compat.v1`) + `tf_slim`, entangled
# with the TF Object Detection API's `ssd_meta_arch`/`feature_map_generators`
# framework -- neither `tensorflow.compat.v1`/`tf_slim` nor the TF-OD-API package
# are installed base libs in this environment, and installing that legacy TF1
# research stack is not reasonable, so this transcribes the real functional
# backbone-construction code (`_conv`, `_separable_conv`,
# `_squeeze_and_excite`, `_inverted_bottleneck_no_expansion`,
# `_inverted_bottleneck`, and the exact `mobiledet_cpu_backbone` block sequence
# with its literal per-block filters/expansion/kernel_size/stride/residual/
# use_se arguments) faithfully into self-contained base-env torch `nn.Module`s.
# Only the SSD detection head (`SSDMobileDetFeatureExtractorBase` +
# `feature_map_generators.multi_resolution_feature_maps`, which is generic TF-OD-
# API detection-head plumbing shared across all SSD feature extractors, not part
# of the MobileDet backbone architecture itself) is out of scope; this module
# traces the backbone up through its C5 endpoint, matching the paper's own
# ablation/classification-accuracy backbone evaluations.
#
# Source: https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/models/ssd_mobiledet_feature_extractor.py
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import functools

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def _scale_filters(filters, multiplier, base=8):
    """Scale the filters accordingly to (multiplier, base). Verbatim from the
    TF repo's `_scale_filters`."""
    round_half_up = int(int(filters) * multiplier / base + 0.5)
    result = int(round_half_up * base)
    return max(result, base)


class Swish6(nn.Module):
    """Verbatim port of the TF repo's `_swish6`: h * relu6(h + 3) / 6."""

    def forward(self, x):
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


def _same_padding(kernel_size, stride, dilation=1):
    """TF `padding='SAME'` for stride-1 (used throughout the backbone at
    stride 1, and the repo always uses odd kernel sizes) reduces to the
    standard symmetric "same" padding used by nn.Conv2d's padding='same'.
    For the stride-2 downsampling convs the repo relies on TF's SAME padding
    computed from the *input* size; since every stride-2 conv/sep-conv in
    this backbone uses an odd kernel (3 or 5), symmetric padding=(k//2) with
    stride 2 reproduces TF SAME's output size for even input dims (the only
    case exercised here, since all TF-OD-API inputs are padded to a multiple
    beforehand) -- this is the standard torch idiom for porting TF SAME convs
    at odd kernel sizes.
    """
    return kernel_size // 2


class ConvBNAct(nn.Module):
    """Verbatim port of the TF repo's `_conv`: conv2d + batchnorm + activation."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, activation=None):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=_same_padding(kernel_size, stride),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch, eps=0.01, momentum=0.01)
        self.act = activation if activation is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SeparableConvBNAct(nn.Module):
    """Verbatim port of the TF repo's `_separable_conv`: depthwise conv +
    pointwise 1x1 conv + batchnorm + activation. If `out_ch` is None, only the
    depthwise stage runs (matching the TF repo's convention of passing
    `filters=None` to get a pure depthwise conv, used by
    `_inverted_bottleneck*`'s depthwise stage)."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, activation=None):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size,
            stride=stride,
            padding=_same_padding(kernel_size, stride),
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False) if out_ch is not None else None
        pw_ch = out_ch if out_ch is not None else in_ch
        self.bn = nn.BatchNorm2d(pw_ch, eps=0.01, momentum=0.01)
        self.act = activation if activation is not None else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise is not None:
            x = self.pointwise(x)
        return self.act(self.bn(x))


class SqueezeExcite(nn.Module):
    """Verbatim port of the TF repo's `_squeeze_and_excite`: global-average-pool
    -> 1x1 conv (reduce) -> activation -> 1x1 conv (expand back to input
    channels) -> sigmoid -> channel-wise scale of the input."""

    def __init__(self, channels, hidden_dim, activation=None):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_dim, 1)
        self.expand = nn.Conv2d(hidden_dim, channels, 1)
        self.act = activation if activation is not None else nn.ReLU6()
        self.gate = nn.Sigmoid()

    def forward(self, x):
        u = self.pool(x)
        u = self.act(self.reduce(u))
        u = self.gate(self.expand(u))
        return u * x


class InvertedBottleneckNoExpansion(nn.Module):
    """Verbatim port of the TF repo's `_inverted_bottleneck_no_expansion`:
    depthwise separable conv (no 1x1 expansion) + optional squeeze-excite +
    1x1 projection (no activation)."""

    def __init__(self, in_ch, out_ch, activation, kernel_size=3, stride=1, use_se=False):
        super().__init__()
        self.dw = SeparableConvBNAct(in_ch, None, kernel_size, stride=stride, activation=activation)
        self.se = (
            SqueezeExcite(in_ch, _scale_filters(in_ch, 0.25), activation=activation)
            if use_se
            else None
        )
        self.project = ConvBNAct(in_ch, out_ch, 1, activation=None)

    def forward(self, x):
        h = self.dw(x)
        if self.se is not None:
            h = self.se(h)
        return self.project(h)


class InvertedBottleneck(nn.Module):
    """Verbatim port of the TF repo's `_inverted_bottleneck`: 1x1 expansion ->
    depthwise separable conv -> optional squeeze-excite -> 1x1 projection
    (no activation) -> optional residual add."""

    def __init__(
        self,
        in_ch,
        out_ch,
        activation,
        kernel_size=3,
        expansion=8,
        stride=1,
        use_se=False,
        residual=True,
    ):
        super().__init__()
        if expansion <= 1:
            raise ValueError("Expansion factor must be greater than 1.")
        expanded = in_ch * expansion
        self.expand = ConvBNAct(in_ch, expanded, 1, activation=activation)
        self.dw = SeparableConvBNAct(
            expanded, None, kernel_size, stride=stride, activation=activation
        )
        self.se = (
            SqueezeExcite(expanded, _scale_filters(expanded, 0.25), activation=activation)
            if use_se
            else None
        )
        self.project = ConvBNAct(expanded, out_ch, 1, activation=None)
        self.residual = residual and stride == 1 and in_ch == out_ch

    def forward(self, x):
        shortcut = x
        h = self.expand(x)
        h = self.dw(h)
        if self.se is not None:
            h = self.se(h)
        h = self.project(h)
        if self.residual:
            h = h + shortcut
        return h


class MobileDetCPUBackbone(nn.Module):
    """Verbatim port of the TF repo's `mobiledet_cpu_backbone`: the exact block
    sequence (filters/expansion/kernel_size/stride/residual/use_se per block) from
    the paper's found CPU-targeted architecture, all blocks using squeeze-excite
    and the swish6 activation (`ibn = functools.partial(_inverted_bottleneck,
    use_se=True, activation_fn=_swish6)` in the original)."""

    def __init__(self, multiplier=1.0):
        super().__init__()

        def s(filters):
            return _scale_filters(filters, multiplier)

        act = Swish6()
        ibn = functools.partial(InvertedBottleneck, activation=act, use_se=True)

        self.stem = ConvBNAct(3, s(16), 3, stride=2, activation=act)
        self.stage0 = InvertedBottleneckNoExpansion(s(16), s(8), activation=act, use_se=True)
        # C1 endpoint = stage0 output

        self.stage1 = ibn(s(8), s(16), expansion=4, stride=2, residual=False)
        # C2 endpoint = stage1 output

        self.stage2 = nn.Sequential(
            ibn(s(16), s(32), expansion=8, stride=2, residual=False),
            ibn(s(32), s(32), expansion=4),
            ibn(s(32), s(32), expansion=4),
            ibn(s(32), s(32), expansion=4),
        )
        # C3 endpoint = stage2 output

        self.stage3 = nn.Sequential(
            ibn(s(32), s(72), kernel_size=5, expansion=8, stride=2, residual=False),
            ibn(s(72), s(72), expansion=8),
            ibn(s(72), s(72), kernel_size=5, expansion=4),
            ibn(s(72), s(72), expansion=4),
            ibn(s(72), s(72), expansion=8, residual=False),
            ibn(s(72), s(72), expansion=8),
            ibn(s(72), s(72), expansion=8),
            ibn(s(72), s(72), expansion=8),
        )
        # C4 endpoint = stage3 output

        self.stage4 = nn.Sequential(
            ibn(s(72), s(104), kernel_size=5, expansion=8, stride=2, residual=False),
            ibn(s(104), s(104), kernel_size=5, expansion=4),
            ibn(s(104), s(104), kernel_size=5, expansion=4),
            ibn(s(104), s(104), expansion=4),
            ibn(s(104), s(144), expansion=8, residual=False),
        )
        # C5 endpoint = stage4 output

    def forward(self, x):
        h = self.stem(x)
        h = self.stage0(h)
        c1 = h
        h = self.stage1(h)
        c2 = h
        h = self.stage2(h)
        c3 = h
        h = self.stage3(h)
        c4 = h
        h = self.stage4(h)
        c5 = h
        return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}


def build_mobiledets():
    return MobileDetCPUBackbone(multiplier=1.0)


def example_input_mobiledets():
    # 224x224 keeps every stride-2 stage's spatial dims an even multiple
    # (matching the TF-OD-API's own pad_to_multiple(32) preprocessing) so the
    # ported symmetric-SAME padding reproduces the real backbone's shapes.
    return torch.randn(1, 3, 224, 224)


MENAGERIE_ENTRIES = [
    ("MobileDets-CPU", "build_mobiledets", "example_input_mobiledets", 2021, "ported-pytorch"),
]
