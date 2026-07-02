# SOURCE: vendored from https://github.com/DmitryUlyanov/deep-image-prior @ master
#   Vendored files: models/common.py (Concat, act/bn/conv helpers used by skip()) and
#   models/skip.py (the `skip` encoder-decoder-with-skip-connections builder function).
#   `models/downsampler.py` (Downsampler, lanczos-kernel downsampling) is NOT vendored
#   because the default `downsample_mode='stride'` path used here never instantiates it
#   (see common.py `conv()`: Downsampler is only constructed when
#   downsample_mode in {'lanczos2','lanczos3'}). No other changes to the architecture.
#
# "Deep Image Prior" (Ulyanov, Vedaldi, Lempitsky; CVPR 2018). DIIP uses an untrained
# convolutional encoder-decoder (the `skip` network below) as an implicit image prior:
# optimizing the network's weights (not a latent code) to reconstruct a single
# corrupted image acts as a strong regularizer for inverse problems (denoising,
# inpainting, super-resolution). We trace the `skip` net at its default settings
# (5-scale conv/BN/LeakyReLU downsample-then-upsample UNet-like architecture with
# 1x1 skip connections and a final Sigmoid), constructed via `get_net`-equivalent
# call with a tiny 2-channel noise input at 32x32 for a fast random-init trace.

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- from models/common.py, unmodified ----
def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)
        ):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[:, :, diff2 : diff2 + target_shape2, diff3 : diff3 + target_shape3]
                )

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
    https://arxiv.org/abs/1710.05941
    The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun="LeakyReLU"):
    """
    Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == "LeakyReLU":
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == "Swish":
            return Swish()
        elif act_fun == "ELU":
            return nn.ELU()
        elif act_fun == "none":
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad="zero", downsample_mode="stride"):
    downsampler = None
    if stride != 1 and downsample_mode != "stride":
        if downsample_mode == "avg":
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == "max":
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ["lanczos2", "lanczos3"]:
            # Downsampler intentionally not vendored -- see module header. This branch
            # is unreachable with the default 'stride' downsample_mode used below.
            raise NotImplementedError("lanczos downsample_mode not vendored in this staging module")
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == "reflection":
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


# ---- from models/skip.py, unmodified ----
def skip(
    num_input_channels=2,
    num_output_channels=3,
    num_channels_down=[16, 32, 64, 128, 128],
    num_channels_up=[16, 32, 64, 128, 128],
    num_channels_skip=[4, 4, 4, 4, 4],
    filter_size_down=3,
    filter_size_up=3,
    filter_skip_size=1,
    need_sigmoid=True,
    need_bias=True,
    pad="zero",
    upsample_mode="nearest",
    downsample_mode="stride",
    act_fun="LeakyReLU",
    need1x1_up=True,
):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip_ = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(
            bn(
                num_channels_skip[i]
                + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])
            )
        )

        if num_channels_skip[i] != 0:
            skip_.add(
                conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad)
            )
            skip_.add(bn(num_channels_skip[i]))
            skip_.add(act(act_fun))

        deeper.add(
            conv(
                input_depth,
                num_channels_down[i],
                filter_size_down[i],
                2,
                bias=need_bias,
                pad=pad,
                downsample_mode=downsample_mode[i],
            )
        )
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(
            conv(
                num_channels_down[i],
                num_channels_down[i],
                filter_size_down[i],
                bias=need_bias,
                pad=pad,
            )
        )
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(
            conv(
                num_channels_skip[i] + k,
                num_channels_up[i],
                filter_size_up[i],
                1,
                bias=need_bias,
                pad=pad,
            )
        )
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


# ---------------------------------------------------------------------------
# Menagerie staging harness
# ---------------------------------------------------------------------------
def build_diip():
    """Deep Image Prior 'skip' net at reduced (2-scale) depth for a fast tiny trace;
    channel/filter settings otherwise match the repo's default `skip()` signature."""
    torch.manual_seed(0)
    return skip(
        num_input_channels=2,
        num_output_channels=3,
        num_channels_down=[16, 32],
        num_channels_up=[16, 32],
        num_channels_skip=[4, 4],
        upsample_mode="nearest",
        downsample_mode="stride",
        need_sigmoid=True,
        need_bias=True,
        pad="zero",
        act_fun="LeakyReLU",
    )


def example_input_diip():
    torch.manual_seed(0)
    return torch.rand(1, 2, 32, 32)


MENAGERIE_ENTRIES = [
    ("Deep Image Prior (DIIP)", "build_diip", "example_input_diip", 2018, "vendored-pytorch"),
]
