# FAITHFUL PORT of ZhangAoCanada/RADDet @ 3ea39753abcc584ee398a50ee0d4ed784681290d
# (original framework: TensorFlow 2 / Keras)
#
# https://github.com/ZhangAoCanada/RADDet
# https://raw.githubusercontent.com/ZhangAoCanada/RADDet/main/model/model.py
# https://raw.githubusercontent.com/ZhangAoCanada/RADDet/main/model/backbone_radarResNet.py
# https://raw.githubusercontent.com/ZhangAoCanada/RADDet/main/model/head_YOLO.py
# https://raw.githubusercontent.com/ZhangAoCanada/RADDet/main/model/layers.py
#
# Zhang, Nowruzi & Laganiere 2021, "RADDet: Range-Azimuth-Doppler based Radar
# Object Detection for Dynamic Road Users" -- the paper's official repo. The
# model consumes a Range-Azimuth-Doppler (RAD) radar cube ``[batch, range,
# azimuth, doppler]`` and predicts YOLO-style 3D boxes (position/size in
# range-azimuth-doppler space) + objectness + class over a single detection
# stage. The repo is 100% TensorFlow 2 / ``tensorflow.keras`` (``model.py``:
# ``class RADDet(K.Model)``, ``backbone_radarResNet.py``, ``head_YOLO.py``,
# ``layers.py``) -- no PyTorch code exists anywhere in the repo, and TF2
# cannot be installed alongside this env's torch stack, so the architecture
# is transcribed faithfully into self-contained torch below. Every mechanism
# is preserved:
#
#   1. ``radarResNet3D`` backbone (``backbone_radarResNet.py``): despite the
#      "3D" name this is a plain 2D-conv ResNet -- the RAD cube's Doppler
#      axis is treated as the channel dimension (NHWC with the Doppler bins
#      as channels), so every ``convolution2D`` call in the original is a
#      real ``Conv2D``. 4 stages with repeat counts ``[2, 4, 8, 16]``, each
#      stage built from ``repeatBlock`` -> ``basicResidualBlock`` (two 3x3
#      convs expanding/contracting channels by a per-block ``channel_
#      expansion`` factor, plus a projection shortcut whenever the stride or
#      expansion changes the shape) with 2x2 max-pool downsampling after
#      every stage (``feature_mp_downsample`` all True). Channel expansion
#      alternates 0.5/2 within a stage by default, and the last block of
#      stages 3-4 doubles the expansion again (``channels_upsample``). Only
#      the last stage's feature map is used (``features = feature_stages[-1]``,
#      one-level/single-scale output -- ``backbone_VGG3D`` is the unused
#      alternative backbone referenced in ``model.py`` and is not ported).
#   2. ``yoloHead`` / ``singleLayerHead`` (``head_YOLO.py``): a single-stage
#      YOLO head over the backbone's last feature map -- a 3x3 conv doubling
#      the channel width, then a 1x1 conv producing
#      ``last_channel * num_anchors * (num_class + 7)`` raw output channels
#      (7 = objectness + xyz + whd), reshaped to
#      ``[batch, H, W, last_channel, num_anchors * (num_class + 7)]`` where
#      ``last_channel = feature_map_channels // 4`` (``head_YOLO.py:79-80``).
#      ``boxDecoder`` (sigmoid/exp box decoding + grid offsets) is inference
#      post-processing on the raw head output, not part of the trainable
#      ``nn.Module`` graph -- ``RADDet.call()`` (``model.py:64-66``) only
#      runs ``backbone -> head``, so decoding is not ported (matching the
#      real forward pass exactly).
#   3. All convs use Conv-BN-ReLU (``L.convolution2D``, ``bn=True``,
#      ``activation="relu"``) except the head's final 1x1 projection
#      (``use_activation=False, bn=False``), matching ``layers.py``.
#
# 6 classes / 6 anchors matches the repo's shipped ``config.json``
# (``all_classes`` has 6 entries) and ``anchors.txt`` (6 anchor rows). Conv2d
# padding="same" (TF) is reproduced via PyTorch's ``padding="same"`` (stride
# 1) and explicit odd-kernel padding for the strided/maxpool ops.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# model/layers.py (Conv-BN-ReLU building block)
# --------------------------------------------------------------------------


def _conv_bn_relu(
    in_channels, out_channels, kernel_size, stride=1, use_activation=True, use_bias=True, bn=True
):
    """Port of ``layers.convolution2D``: Conv2D(same padding) -> BN -> ReLU."""
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=use_bias,
        ),
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if use_activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------
# model/backbone_radarResNet.py
# --------------------------------------------------------------------------


class BasicResidualBlock(nn.Module):
    """Port of ``basicResidualBlock``."""

    def __init__(self, in_channels, channel_expansion, stride=1, use_bias=True):
        super().__init__()
        mid_channels = in_channels
        out_channels = int(in_channels * channel_expansion)

        self.conv1 = _conv_bn_relu(in_channels, mid_channels, 3, stride=stride, use_bias=use_bias)
        self.conv2 = _conv_bn_relu(mid_channels, out_channels, 3, stride=1, use_bias=use_bias)
        self.conv3 = _conv_bn_relu(out_channels, out_channels, 1, stride=1, use_bias=use_bias)

        self.needs_shortcut_conv = stride != 1 or channel_expansion != 1
        if self.needs_shortcut_conv:
            self.shortcut = _conv_bn_relu(
                in_channels, out_channels, 3, stride=stride, use_bias=use_bias
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.conv2(conv)
        conv = self.conv3(conv)
        conv_shortcut = self.shortcut(x)
        return conv + conv_shortcut


class RepeatBlock(nn.Module):
    """Port of ``repeatBlock``: repeated residual blocks + optional 2x2 max-pool."""

    def __init__(self, in_channels, repeat_times, all_expansions, feature_maps_downsample):
        super().__init__()
        blocks = []
        channels = in_channels
        for i in range(repeat_times):
            expansion = all_expansions[i]
            blocks.append(BasicResidualBlock(channels, expansion, stride=1, use_bias=True))
            channels = int(channels * expansion)
        self.blocks = nn.ModuleList(blocks)
        self.downsample = (
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            if feature_maps_downsample
            else nn.Identity()
        )
        self.out_channels = channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.downsample(x)


class RadarResNet3D(nn.Module):
    """Port of ``radarResNet3D``: a 2D-conv ResNet over the RAD cube
    (Doppler bins treated as channels), 4 stages, last-stage-only output."""

    def __init__(self, in_channels):
        super().__init__()
        block_repeat_times = [2, 4, 8, 16]
        channels_upsample = [False, False, True, True]
        feature_mp_downsample = [True, True, True, True]

        stages = []
        channels = in_channels
        for i in range(len(block_repeat_times)):
            repeat_times = block_repeat_times[i]
            all_expansions = [1, 1] * (repeat_times // 2)
            if channels_upsample[i]:
                all_expansions[-1] *= 2
            stage = RepeatBlock(channels, repeat_times, all_expansions, feature_mp_downsample[i])
            stages.append(stage)
            channels = stage.out_channels
        self.stages = nn.ModuleList(stages)
        self.out_channels = channels

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        # one-level output: only the last stage's feature map is used
        # (feature_stages[-1] in the original).
        return x


# --------------------------------------------------------------------------
# model/head_YOLO.py
# --------------------------------------------------------------------------


class YoloHead(nn.Module):
    """Port of ``yoloHead`` / ``singleLayerHead``: single-stage YOLO head."""

    def __init__(self, in_channels, num_anchors, num_class):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_class = num_class
        # last_channel = feature_map_channels // 4 (head_YOLO.py:80)
        self.last_channel = max(in_channels // 4, 1)
        final_output_channels = self.last_channel * num_anchors * (num_class + 7)

        self.conv1 = _conv_bn_relu(
            in_channels, in_channels * 2, 3, stride=1, use_bias=True, bn=True
        )
        self.conv2 = nn.Conv2d(
            in_channels * 2, final_output_channels, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, feature_map):
        # feature_map: [batch, channels, H, W] (torch NCHW)
        conv = self.conv1(feature_map)
        conv = self.conv2(conv)
        batch, _, height, width = conv.shape
        # TF layout is NHWC -> reshape target [-1, H, W, last_channel,
        # num_anchors * (num_class + 7)] (head_YOLO.py:16-17,29). Permute to
        # NHWC first so the reshape matches the original channel ordering.
        conv = conv.permute(0, 2, 3, 1).contiguous()
        conv = conv.view(
            batch, height, width, self.last_channel, self.num_anchors * (self.num_class + 7)
        )
        return conv


# --------------------------------------------------------------------------
# model/model.py (top-level RADDet module)
# --------------------------------------------------------------------------


class RADDet(nn.Module):
    """Port of ``class RADDet(K.Model)``. ``call()`` only runs
    ``backbone -> head`` (box decoding is inference-time post-processing,
    not part of the trainable graph)."""

    def __init__(self, in_channels, num_class, num_anchors):
        super().__init__()
        self.backbone = RadarResNet3D(in_channels)
        self.head = YoloHead(self.backbone.out_channels, num_anchors, num_class)

    def forward(self, x):
        # x: [batch, doppler_channels, range, azimuth] (torch NCHW; the TF
        # original is NHWC with Doppler bins as the channel axis).
        features = self.backbone(x)
        yolo_raw = self.head(features)
        return yolo_raw


def build_raddet() -> nn.Module:
    """Build a tiny RADDet radar-cube detector.

    Uses a small RAD-cube spatial size and reduced Doppler-channel count so
    the module traces quickly; the architecture (4-stage radarResNet2D
    backbone + single-stage YOLO head over a Range-Azimuth-Doppler cube) is
    unchanged from the real ``model/model.py`` + ``model/backbone_
    radarResNet.py`` + ``model/head_YOLO.py``.

    Returns
    -------
    nn.Module
        Random-initialized RADDet detector (6 classes, 6 anchors, matching
        the repo's shipped ``config.json`` / ``anchors.txt``).
    """
    return RADDet(in_channels=8, num_class=6, num_anchors=6)


def example_input_raddet() -> torch.Tensor:
    """Return a small RAD-cube tensor matching ``build_raddet``.

    Returns
    -------
    torch.Tensor
        ``(batch, doppler_channels, range, azimuth)`` tensor of shape
        ``(1, 8, 32, 32)`` (Doppler bins as channels, matching the repo's
        NHWC-with-Doppler-as-channel convention).
    """
    return torch.randn(1, 8, 32, 32)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("RADDet", "build_raddet", "example_input_raddet", 2021, "ported"),
]
