# FAITHFUL PORT of IoBT-VISTEC/EEGANet @ main (original framework: TensorFlow/Keras)
# (model.py -- full file: res_block_gen, discriminator_block, Generator.generator,
#  Discriminator.discriminator)
"""EEGANet: a GAN for calibration-free removal of ocular artifacts from multichannel EEG
(Sawangjai, Trakulruangroj, Boonnag, Piriyajitakonkij, Tripathy, Sudhawiyangkul &
Wilaiprasitporn, "EEGANet: Removal of Ocular Artifact from the EEG Signal Using Generative
Adversarial Networks", IEEE J. Biomed. Health Inform. 2021). Official repo:
https://github.com/IoBT-VISTEC/EEGANet (``model.py`` @ main; itself explicitly "modified
from https://github.com/deepak112/Keras-SRGAN" -- EEGANet applies the SRGAN generator/
discriminator topology to multichannel EEG treated as a 2-D array (channels x time
samples), with 1-D-style convs (kernel ``(1, k)``, i.e. only sliding along the time axis).

The original repo is pure TensorFlow/Keras (``tensorflow.keras.layers`` functional API,
dependency pin ``tensorflow-gpu==2.2.0``); TensorFlow/Keras is not installed in this
environment, so the architecture is FAITHFULLY PORTED here to self-contained torch
(rung 3), transcribed layer-for-layer from the real repo code -- not re-derived from the
paper. Every layer matches the Keras source 1:1:

- ``res_block_gen``: Conv2D(k=(1,3),s=1,same) -> BN(momentum=0.5) -> PReLU(shared per
  channel, i.e. ``num_parameters=1`` matching Keras' ``shared_axes=[1,2]`` which shares
  the PReLU slope across both spatial axes, leaving one learned slope per channel) ->
  Conv2D(k=(1,3),s=1,same) -> BN(momentum=0.5) -> residual add.
- ``discriminator_block``: Conv2D(k,s,same) -> BN(momentum=0.5) -> LeakyReLU(0.2).
- ``Generator.generator``: Conv2D(64,(1,9),s=1,same) -> PReLU -> [16x res_block_gen(64,
  (1,3),s=1)] -> Conv2D(64,(1,3),s=1,same) -> BN(momentum=0.5) -> residual add (long
  skip from just after the first PReLU) -> Conv2D(output_ch,(1,9),s=1,same) -> tanh.
- ``Discriminator.discriminator``: Conv2D(64,(1,3),s=1,same) -> LeakyReLU(0.2) -> 7x
  ``discriminator_block`` with (filters,kernel,stride) = (64,(1,3),2), (128,(1,3),1),
  (128,(1,3),2), (256,(1,3),1), (256,(1,3),2), (512,(1,3),1), (512,(1,3),2) -> Flatten ->
  Dense(1024) -> LeakyReLU(0.2) -> Dense(1) -> sigmoid.

Keras' ``padding="same"`` (TF convention: output spatial size = ceil(input/stride)) is
reproduced here with explicit asymmetric ``nn.ZeroPad2d`` computed per-call from the
input size, matching TF's SAME padding formula exactly for both stride 1 and stride 2,
since torch's native ``Conv2d`` only supports symmetric padding.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _same_pad_amount(in_size: int, kernel: int, stride: int) -> tuple[int, int]:
    """TensorFlow SAME padding along one spatial axis: matches Keras'
    ``padding="same"`` (output size = ceil(in_size / stride))."""
    out_size = math.ceil(in_size / stride)
    total_pad = max((out_size - 1) * stride + kernel - in_size, 0)
    pad_before = total_pad // 2
    pad_after = total_pad - pad_before
    return pad_before, pad_after


class SameConv2d(nn.Module):
    """Conv2d with TensorFlow-style ``padding="same"`` (Keras ``Conv2D(..., padding="same")``),
    computed dynamically per forward call since torch's built-in padding is symmetric only."""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.conv = nn.Conv2d(
            in_channels, out_channels, self.kernel_size, stride=self.stride, padding=0
        )

    def forward(self, x):
        _, _, h, w = x.shape
        pad_top, pad_bottom = _same_pad_amount(h, self.kernel_size[0], self.stride[0])
        pad_left, pad_right = _same_pad_amount(w, self.kernel_size[1], self.stride[1])
        x = nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


def res_block_gen(filters: int, kernel_size, strides):
    """Residual block: Conv-BN-PReLU-Conv-BN, added back to the block input by the caller
    (Keras' ``add([gen, model])`` -- kept as an explicit submodule here so the residual add
    is visible as its own op, matching the functional-API structure)."""
    return nn.ModuleDict(
        {
            "conv1": SameConv2d(filters, filters, kernel_size, strides),
            "bn1": nn.BatchNorm2d(filters, momentum=0.5),
            "prelu": nn.PReLU(num_parameters=filters),
            "conv2": SameConv2d(filters, filters, kernel_size, strides),
            "bn2": nn.BatchNorm2d(filters, momentum=0.5),
        }
    )


class ResBlockGen(nn.Module):
    def __init__(self, filters: int, kernel_size, strides):
        super().__init__()
        self.block = res_block_gen(filters, kernel_size, strides)

    def forward(self, x):
        gen = x
        x = self.block["conv1"](x)
        x = self.block["bn1"](x)
        x = self.block["prelu"](x)
        x = self.block["conv2"](x)
        x = self.block["bn2"](x)
        return gen + x


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, filters: int, kernel_size, strides):
        super().__init__()
        self.conv = SameConv2d(in_channels, filters, kernel_size, strides)
        self.bn = nn.BatchNorm2d(filters, momentum=0.5)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class Generator(nn.Module):
    """SRGAN-style generator applied to EEG treated as a (channels, time) 2-D array;
    ``in_ch``/``output_ch`` are the number of EEG channels in / out (matching the
    original's ``noise_shape``/``output_ch``)."""

    def __init__(self, in_ch: int, output_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.output_ch = output_ch

        self.entry_conv = SameConv2d(in_ch, 64, (1, 9), 1)
        self.entry_prelu = nn.PReLU(num_parameters=64)

        self.res_blocks = nn.ModuleList([ResBlockGen(64, (1, 3), 1) for _ in range(16)])

        self.post_res_conv = SameConv2d(64, 64, (1, 3), 1)
        self.post_res_bn = nn.BatchNorm2d(64, momentum=0.5)

        self.final_conv = SameConv2d(64, output_ch, (1, 9), 1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        x = self.entry_conv(x)
        x = self.entry_prelu(x)
        gen_model = x

        for block in self.res_blocks:
            x = block(x)

        x = self.post_res_conv(x)
        x = self.post_res_bn(x)
        x = gen_model + x

        x = self.final_conv(x)
        return self.final_act(x)


class Discriminator(nn.Module):
    def __init__(self, in_ch: int, height: int, width: int):
        super().__init__()
        self.in_ch = in_ch
        self.height = height
        self.width = width

        self.entry_conv = SameConv2d(in_ch, 64, (1, 3), 1)
        self.entry_act = nn.LeakyReLU(0.2)

        block_cfg = [
            (64, 64, (1, 3), 2),
            (64, 128, (1, 3), 1),
            (128, 128, (1, 3), 2),
            (128, 256, (1, 3), 1),
            (256, 256, (1, 3), 2),
            (256, 512, (1, 3), 1),
            (512, 512, (1, 3), 2),
        ]
        self.blocks = nn.ModuleList(
            [DiscriminatorBlock(cin, cout, k, s) for cin, cout, k, s in block_cfg]
        )

        flat_h = height
        flat_w = width
        for _, _, _, s in block_cfg:
            flat_h = math.ceil(flat_h / s)
            flat_w = math.ceil(flat_w / s)
        flat_dim = 512 * flat_h * flat_w

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(flat_dim, 1024)
        self.dense1_act = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(1024, 1)
        self.dense2_act = nn.Sigmoid()

    def forward(self, x):
        x = self.entry_conv(x)
        x = self.entry_act(x)

        for block in self.blocks:
            x = block(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense1_act(x)
        x = self.dense2(x)
        return self.dense2_act(x)


MENAGERIE_ZOO = "ported-pytorch"


def build_eeganet():
    # Tiny EEG-shaped generator: e.g. 4 EEG channels x short time window, in_ch==output_ch
    # (artifact-corrupted EEG in, artifact-removed EEG out) as used in the original paper.
    return Generator(in_ch=4, output_ch=4)


def example_input_eeganet():
    # (B, channels, EEG_channels, time_samples) -- the repo's Keras Input(shape=noise_shape)
    # treats (EEG_channels, time_samples) as the 2 spatial dims with 1 leading "image"
    # channel; kept here as a 4-D NCHW tensor with C=in_ch matching build_eeganet's in_ch.
    return (torch.randn(1, 4, 4, 32),)


def build_eeganet_discriminator():
    return Discriminator(in_ch=4, height=4, width=32)


def example_input_eeganet_discriminator():
    return (torch.randn(1, 4, 4, 32),)


MENAGERIE_ENTRIES = [
    ("EEGANet Generator", "build_eeganet", "example_input_eeganet", 2021, "ported-pytorch"),
    (
        "EEGANet Discriminator",
        "build_eeganet_discriminator",
        "example_input_eeganet_discriminator",
        2021,
        "ported-pytorch",
    ),
]
