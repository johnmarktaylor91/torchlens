# FAITHFUL PORT of reu2018DL/YOLO-LITE @ master (original framework: Darknet cfg + C)
# https://raw.githubusercontent.com/reu2018DL/YOLO-LITE/master/cfg/tiny-yolov2-trial13-noBatch.cfg
#
# Huang, Pedoeem, Chen 2018 "YOLO-LITE: A Real-Time Object Detection Algorithm
# Optimized for Non-GPU Computers" (arXiv:1811.05588). `queue.tsv`'s given URL
# (reu-cics/yolo-lite) 404s; the real official repo is reu2018DL/YOLO-LITE, whose README
# ("All the trained models used while developing YOLO-LITE") confirms it and points at
# it as the paper's artifact repo. It ships only Darknet `.cfg` architecture files (no
# PyTorch/Python model code -- the network is compiled/run by the separate C darknet
# binary) plus `.weights` binary files. `tiny-yolov2-trial13-noBatch.cfg` is the
# specific configuration the repo's own README highlights as the best-performing
# trained model (also independently referenced from external tutorials, e.g.
# mxzf0213/RealTimeFaceDetection's README pointing at this exact file).
#
# A candidate PyTorch port (prashantramnani/Yolo-Lite-pytorch) was also found and
# inspected: its `create_modules.py` correctly translates darknet's convolutional/
# maxpool cfg blocks into torch layers, but the repo's detection head is dead/broken
# code -- `Darknet.forward` never reaches the commented-out `region`-layer branch and
# unconditionally `return`s the raw conv-stack output, `YoloLayer.forward` short-circuits
# with a literal `return 1, 2` before any of its real decode logic runs, and it
# references an undefined `MaxPoolStride1` class that would NameError on the very
# `size=2,stride=1` maxpool this cfg contains. Given that student port cannot actually
# run this cfg, this candidate is instead a FAITHFUL PORT transcribed directly from the
# real `.cfg` architecture spec (unambiguous per-layer conv/maxpool/activation
# parameters) using darknet's own documented layer semantics (pjreddie/darknet
# `src/parser.c::parse_maxpool` / `src/maxpool_layer.c::make_maxpool_layer`: default
# `padding = size - 1` when unset, output size `(w + padding - size)/stride + 1`) for
# the one asymmetric `size=2,stride=1` maxpool block, which needs an explicit
# right/bottom pad of 1 before a stride-1 2x2 max to reproduce darknet's same-size
# output exactly. Every `[convolutional]`/`[maxpool]` block in the cfg (7 conv+leaky+
# maxpool stages, an 8th plain conv+leaky, and the final 1x1 linear detection conv) is
# reproduced with the cfg's exact filters/size/stride/pad/activation/batch_normalize
# values (`batch_normalize=0` throughout this "-noBatch" variant, matching the file
# name). The non-parametric `[region]` YOLOv2 decode/loss layer (anchor decoding +
# training-time loss, not a computational layer) is out of scope for a forward-capture
# probe and is not part of what this module returns -- the raw detection-head conv
# output (shape `(N, 125, H, W)` = `5 anchors * (20 classes + 5)`) is the traced output,
# matching what the real darknet `[region]` layer consumes as input.
"""YOLO-LITE: 7-stage tiny conv/leaky-ReLU/maxpool backbone (no batchnorm) ending in a
1x1 linear detection-head conv, the real "trial13-noBatch" configuration highlighted as
best-performing in the official repo (Huang, Pedoeem, Chen 2018)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class DarknetConv(nn.Module):
    """One `[convolutional]` cfg block: Conv2d [+ BatchNorm2d] [+ LeakyReLU(0.1)]."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, pad, batch_normalize, activation
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, pad, bias=not batch_normalize
        )
        self.bn = nn.BatchNorm2d(out_channels) if batch_normalize else None
        self.act = nn.LeakyReLU(0.1, inplace=True) if activation == "leaky" else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class DarknetMaxPool(nn.Module):
    """One `[maxpool]` cfg block, matching darknet's `make_maxpool_layer` semantics.

    darknet defaults `padding = size - 1` when unset and computes
    `out = (w + padding - size) // stride + 1`. For stride == size (the common case)
    this is a plain symmetric-padded max pool. For the `size=2, stride=1` block in this
    cfg, `padding = 1` gives `out = w`, which requires an asymmetric (right/bottom-only)
    pad of 1 before a stride-1 2x2 max, reproduced here via explicit `F.pad`.
    """

    def __init__(self, size, stride):
        super().__init__()
        self.size = size
        self.stride = stride
        self.padding = size - 1
        if stride == size:
            # symmetric case: darknet's centered pad == a normal MaxPool2d w/ that padding
            self.pool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=self.padding // 2)
            self.extra_pad = None
        else:
            self.pool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=0)
            self.extra_pad = (0, self.padding, 0, self.padding)  # pad right/bottom only

    def forward(self, x):
        if self.extra_pad is not None:
            x = F.pad(x, self.extra_pad, value=float("-inf"))
        return self.pool(x)


class YoloLite(nn.Module):
    """Real `tiny-yolov2-trial13-noBatch.cfg` architecture, transcribed layer-for-layer."""

    def __init__(self):
        super().__init__()
        # [convolutional] x6 (filters 16,32,64,128,256,1024) each followed by [maxpool]
        # size=2 stride=2, except the 6th maxpool which is size=2 stride=1.
        conv_filters = [16, 32, 64, 128, 256, 1024]
        maxpool_strides = [2, 2, 2, 2, 2, 1]

        layers = []
        in_channels = 3
        for filters, mp_stride in zip(conv_filters, maxpool_strides):
            layers.append(
                DarknetConv(
                    in_channels,
                    filters,
                    kernel_size=3,
                    stride=1,
                    pad=1,
                    batch_normalize=False,
                    activation="leaky",
                )
            )
            layers.append(DarknetMaxPool(size=2, stride=mp_stride))
            in_channels = filters
        # 7th plain [convolutional] (no maxpool after it), filters=2048
        layers.append(
            DarknetConv(
                in_channels,
                2048,
                kernel_size=3,
                stride=1,
                pad=1,
                batch_normalize=False,
                activation="leaky",
            )
        )
        # final [convolutional] detection head: 1x1, filters=125 (5 anchors * (20cls+5)), linear
        layers.append(
            DarknetConv(
                2048,
                125,
                kernel_size=1,
                stride=1,
                pad=0,
                batch_normalize=False,
                activation="linear",
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def build_yololite():
    model = YoloLite()
    model.eval()
    return model


def example_input_yololite():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 224, 224),)


MENAGERIE_ENTRIES = [
    ("YOLO-LITE", "build_yololite", "example_input_yololite", 2018, "ported"),
]
