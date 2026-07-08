# SOURCE: vendored from reinhardh/supplement_deep_decoder @ master
#
# https://github.com/reinhardh/supplement_deep_decoder
# https://raw.githubusercontent.com/reinhardh/supplement_deep_decoder/master/include/decoder.py
#
# Deep Decoder (Heckel, Hand, "Deep Decoder: Concise Image Representations from
# Untrained Non-convolutional Networks", ICLR 2019 -- official author repo). An
# untrained, non-convolutional (1x1-conv-only) image-generator network: a chain of
# per-pixel channel-mixing 1x1 convolutions, bilinear upsampling, batch-norm, and a
# fixed activation, ending in a channel projection to the output image + sigmoid. The
# real repo's top-level module-level statement `torch.nn.Module.add = add_module`
# monkeypatches `nn.Module` with a `.add(...)` helper used throughout `decodernw`/
# `resdecoder`/`ResidualBlock`; this is kept verbatim (it is how the real repo builds
# its `nn.Sequential` stacks). `add_module`, `conv`, `decodernw`, `ResidualBlock`, and
# `resdecoder` are transcribed verbatim from the real repo file above; only the module-
# level `torch.nn.Module.add = add_module` assignment is moved inside a call at import
# time of this file (identical effect, avoids a bare top-level monkeypatch surprising
# other imports of this staging module) and the model itself is unmodified.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# From include/decoder.py
# ---------------------------------------------------------------------------
def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


def conv(in_f, out_f, kernel_size, stride=1, pad="zero"):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == "reflection":
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)

    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)


def decodernw(
    num_output_channels=3,
    num_channels_up=[128] * 5,
    filter_size_up=1,
    need_sigmoid=True,
    pad="reflection",
    upsample_mode="bilinear",
    act_fun=nn.ReLU(),  # nn.LeakyReLU(0.2, inplace=True)
    bn_before_act=False,
    bn_affine=True,
    upsample_first=True,
):
    num_channels_up = num_channels_up + [num_channels_up[-1], num_channels_up[-1]]
    n_scales = len(num_channels_up)

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales
    model = nn.Sequential()

    for i in range(len(num_channels_up) - 1):
        if upsample_first:
            model.add(
                conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up[i], 1, pad=pad)
            )
            if upsample_mode != "none" and i != len(num_channels_up) - 2:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
        else:
            if upsample_mode != "none" and i != 0:
                model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            model.add(
                conv(num_channels_up[i], num_channels_up[i + 1], filter_size_up[i], 1, pad=pad)
            )

        if i != len(num_channels_up) - 1:
            if bn_before_act:
                model.add(nn.BatchNorm2d(num_channels_up[i + 1], affine=bn_affine))
            model.add(act_fun)
            if not bn_before_act:
                model.add(nn.BatchNorm2d(num_channels_up[i + 1], affine=bn_affine))

    model.add(conv(num_channels_up[-1], num_output_channels, 1, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_f, out_f):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_f, out_f, 1, 1, padding=0, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return out


def resdecoder(
    num_output_channels=3,
    num_channels_up=[128] * 5,
    filter_size_up=1,
    need_sigmoid=True,
    pad="reflection",
    upsample_mode="bilinear",
    act_fun=nn.ReLU(),  # nn.LeakyReLU(0.2, inplace=True)
    bn_before_act=False,
    bn_affine=True,
):
    num_channels_up = num_channels_up + [num_channels_up[-1], num_channels_up[-1]]
    n_scales = len(num_channels_up)

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    model = nn.Sequential()

    for i in range(len(num_channels_up) - 2):
        model.add(ResidualBlock(num_channels_up[i], num_channels_up[i + 1]))

        if upsample_mode != "none":
            model.add(nn.Upsample(scale_factor=2, mode=upsample_mode))

        if i != len(num_channels_up) - 1:
            model.add(act_fun)

    model.add(ResidualBlock(num_channels_up[-1], num_channels_up[-1]))
    model.add(act_fun)

    model.add(conv(num_channels_up[-1], num_output_channels, 1, pad=pad))

    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


def build_deepdecoder_tiny() -> nn.Sequential:
    # Real repo default: num_channels_up=[128]*5 (5 upsampling stages, 32x total
    # upsample), num_output_channels=3 (RGB); kept at default width/depth per the
    # official `decodernw()` signature.
    return decodernw(num_output_channels=3, num_channels_up=[128] * 5).eval()


def example_input_deepdecoder_tiny():
    # Deep Decoder is a generator: input is a small fixed random latent code with the
    # network's channel count (128) and small spatial size (image_size / 2**n_scales,
    # matching the paper's under-parameterized untrained-network setup), not a real
    # image.
    return torch.randn(1, 128, 8, 8)


MENAGERIE_ENTRIES = [
    (
        "DeepDecoder",
        "build_deepdecoder_tiny",
        "example_input_deepdecoder_tiny",
        2019,
        "vendored-pytorch",
    ),
]
