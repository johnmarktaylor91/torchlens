# FAITHFUL PORT of mlcommons/tiny @ 1afd2c98 (original framework: TensorFlow/Keras)
# (benchmark/training/keyword_spotting/keras_model.py: get_model(), model_name == 'ds_cnn'
#  branch)
#
# DS-CNN (depthwise-separable-conv keyword-spotting network) is the MLCommons Tiny v1.0
# "Keyword Spotting" reference benchmark model. Its architecture traces to Zhang et al.
# 2018, "Hello Edge: Keyword Spotting on Microcontrollers" (arXiv:1711.07128), which is
# itself based on MobileNet's depthwise-separable convolution (Howard et al. 2017,
# arXiv:1704.04861). mlcommons/tiny is the maintained, canonical reference repo for this
# exact benchmark row (superseding the older, TF1.x-`contrib`-based ARM reference impl at
# PeterMS123/KWS-DS-CNN-for-embedded, whose `tensorflow.contrib.slim` import is no longer
# importable under any installed TF/torch base lib in this environment -- confirmed
# `ModuleNotFoundError` on import). The mlcommons/tiny reference itself is Keras/TF (not
# torch), so it cannot be vendored directly; this is a faithful layer-for-layer PORT of
# `get_model(args)`'s `model_name == "ds_cnn"` branch (the file's other branches, `fc4` and
# `td_cnn`, are separate architectures and are not ported here).
#
# Architecture (faithfully reproduced, every layer in source order):
#   Conv2D(64, kernel=(10,4), stride=(2,2), padding='same') -> BN -> ReLU -> Dropout(0.2)
#   4x [ DepthwiseConv2D(kernel=(3,3), padding='same') -> BN -> ReLU
#        -> Conv2D(64, kernel=(1,1), padding='same') -> BN -> ReLU ]
#   Dropout(0.4) -> AveragePooling2D(pool_size=(H/2, W/2)) -> Flatten -> Dense(12, softmax)
#
# Keras `padding='same'` computes asymmetric (TF-style) zero-padding for ANY stride (not
# just stride=1, which is all `torch.nn.Conv2d(padding="same")` supports) -- `_same_pad2d`
# below reproduces TF's SAME-padding formula exactly (pad_total = max((ceil(in/stride)-1)*
# stride + (kernel-1)*dilation + 1 - in, 0), split floor/ceil low/high) so strided layers
# (the stem conv, stride=(2,2)) get the same effective padding as the real Keras graph.
# Model defaults (`filters=64`, `weight_decay=1e-4` unused here since L2 only affects
# training loss) and the fixed 4-depthwise-separable-layer topology are the file's
# hardcoded literal values, not user-overridable in the original -- ported verbatim.
# `label_count=12` is the file's hardcoded `get_model()` default (12-keyword classification
# task). Input spatial size (spectrogram_length=49, dct_coefficient_count=10) comes from
# `prepare_model_settings()` under the repo's own `kws_util.py` argparse defaults
# (sample_rate=16000, clip_duration_ms=1000, window_size_ms=30, window_stride_ms=20,
# dct_coefficient_count=10) -> spectrogram_length = 1 + (16000-480)//320 = 49.

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


def _same_pad2d(x: torch.Tensor, kernel_size, stride) -> torch.Tensor:
    """Reproduce TF/Keras `padding='same'` (asymmetric SAME padding) for Conv2d/DepthwiseConv2d
    ahead of a torch conv called with padding=0, matching Keras semantics for any stride
    (torch's own `padding="same"` string only supports stride=1)."""
    kh, kw = kernel_size
    sh, sw = stride
    ih, iw = x.shape[-2], x.shape[-1]
    pad_h = max((math.ceil(ih / sh) - 1) * sh + kh - ih, 0)
    pad_w = max((math.ceil(iw / sw) - 1) * sw + kw - iw, 0)
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return nn.functional.pad(x, (left, right, top, bottom))


class SameConv2d(nn.Module):
    """Conv2D(..., padding='same') ported from keras_model.py's `Conv2D(filters, (10,4),
    strides=(2,2), padding='same', ...)` stem layer and the (1,1) pointwise layers."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        x = _same_pad2d(x, self.kernel_size, self.stride)
        return self.conv(x)


class SameDepthwiseConv2d(nn.Module):
    """DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same') ported from
    keras_model.py's 4 depthwise-separable-conv layers."""

    def __init__(self, channels, kernel_size=(3, 3), stride=(1, 1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(
            channels, channels, kernel_size, stride=stride, padding=0, groups=channels
        )

    def forward(self, x):
        x = _same_pad2d(x, self.kernel_size, self.stride)
        return self.conv(x)


class DSCNN(nn.Module):
    """Ported from keras_model.py `get_model(args)`, `model_name == "ds_cnn"` branch."""

    def __init__(self, spectrogram_length=49, dct_coefficient_count=10, label_count=12, filters=64):
        super().__init__()
        self.spectrogram_length = spectrogram_length
        self.dct_coefficient_count = dct_coefficient_count

        # Input pure conv2d
        self.conv0 = SameConv2d(1, filters, kernel_size=(10, 4), stride=(2, 2))
        self.bn0 = nn.BatchNorm2d(filters)
        self.act0 = nn.ReLU()
        self.drop0 = nn.Dropout(p=0.2)

        # 4 layers of separable depthwise conv2d (kernel_size=(3,3), stride=(1,1) each)
        self.dw = nn.ModuleList()
        self.dw_bn = nn.ModuleList()
        self.pw = nn.ModuleList()
        self.pw_bn = nn.ModuleList()
        for _ in range(4):
            self.dw.append(SameDepthwiseConv2d(filters, kernel_size=(3, 3), stride=(1, 1)))
            self.dw_bn.append(nn.BatchNorm2d(filters))
            self.pw.append(SameConv2d(filters, filters, kernel_size=(1, 1), stride=(1, 1)))
            self.pw_bn.append(nn.BatchNorm2d(filters))
        self.act = nn.ReLU()

        self.drop_final = nn.Dropout(p=0.4)
        # stride-(2,2) stem halves both spatial dims (SAME padding); final_pool_size =
        # (input_shape[0]//2, input_shape[1]//2) exactly as in keras_model.py.
        pool_h = spectrogram_length // 2
        pool_w = dct_coefficient_count // 2
        self.avg_pool = nn.AvgPool2d(kernel_size=(pool_h, pool_w))
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(filters, label_count)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (batch, 1, spectrogram_length, dct_coefficient_count) -- Keras NHWC
        # (batch, time, freq, 1) transposed to torch NCHW (batch, 1, time, freq).
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.act0(x)
        x = self.drop0(x)

        for i in range(4):
            x = self.dw[i](x)
            x = self.dw_bn[i](x)
            x = self.act(x)
            x = self.pw[i](x)
            x = self.pw_bn[i](x)
            x = self.act(x)

        x = self.drop_final(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.softmax(x)


# ---- end ported source ----


def build_ds_cnn():
    torch.manual_seed(0)
    return DSCNN(spectrogram_length=49, dct_coefficient_count=10, label_count=12, filters=64)


def example_input_ds_cnn():
    torch.manual_seed(0)
    # NCHW: (batch, 1 channel, 49 time frames, 10 MFCC coefficients) -- kws_util.py
    # argparse defaults (sample_rate=16000, clip_duration_ms=1000, window_size_ms=30,
    # window_stride_ms=20, dct_coefficient_count=10).
    return torch.randn(2, 1, 49, 10)


MENAGERIE_ENTRIES = [
    ("DS-CNN (MLCommons Tiny KWS)", "build_ds_cnn", "example_input_ds_cnn", 2018, MENAGERIE_ZOO),
]
