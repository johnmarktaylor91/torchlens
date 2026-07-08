# SOURCE: vendored from MrGiovanni/ModelsGenesis @ master
# (https://raw.githubusercontent.com/MrGiovanni/ModelsGenesis/master/pytorch/unet3d.py)
#
# Models Genesis (Zhou et al., "Models Genesis: Generic Autodidactic Models for 3D
# Medical Image Analysis", MICCAI 2019) is a self-supervised pretraining scheme for 3D
# medical-image models; its backbone (and the architecture this repo's PyTorch
# implementation ships and trains) is a 3D U-Net (`UNet3D` below): four
# `DownTransition` encoder stages (each a pair of Conv3d+ContBatchNorm3d+activation
# blocks, downsampled via MaxPool3d except at the bottleneck) mirrored by three
# `UpTransition` decoder stages (ConvTranspose3d upsampling + skip-concatenation +
# another `_make_nConv` pair), ending in a 1x1x1 `OutputTransition` conv + sigmoid.
#
# The class bodies below (ContBatchNorm3d, LUConv, _make_nConv, DownTransition,
# UpTransition, OutputTransition, UNet3D) are copied verbatim from the real
# `pytorch/unet3d.py` file; only the unused, already-commented-out `InputTransition`
# class from the original file was dropped (dead code in the source itself), and
# imports were left as-is (torch, torch.nn, torch.nn.functional only -- both base libs).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == "relu":
            self.activation = nn.ReLU(out_chan)
        elif act == "prelu":
            self.activation = nn.PReLU(out_chan)
        elif act == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth + 1)), act)
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), act)
    else:
        layer1 = LUConv(in_channel, 32 * (2**depth), act)
        layer2 = LUConv(32 * (2**depth), 32 * (2**depth) * 2, act)

    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans + outChans // 2, depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act="relu"):
        super(UNet3D, self).__init__()

        self.down_tr64 = DownTransition(1, 0, act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)

        self.up_tr256 = UpTransition(512, 512, 2, act)
        self.up_tr128 = UpTransition(256, 256, 1, act)
        self.up_tr64 = UpTransition(128, 128, 0, act)
        self.out_tr = OutputTransition(64, n_class)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        self.out512, self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)
        return self.out


def build_modelsgenesis():
    torch.manual_seed(0)
    return UNet3D(n_class=1, act="relu")


def example_input_modelsgenesis():
    torch.manual_seed(0)
    # 3 MaxPool3d(2) stages in the encoder require each spatial dim divisible by 8;
    # 32 is the smallest comfortable size that keeps every intermediate feature map
    # at least 1 voxel wide after the deepest pool.
    return torch.randn(1, 1, 32, 32, 32)


MENAGERIE_ENTRIES = [
    ("Models Genesis", build_modelsgenesis, example_input_modelsgenesis, 2019, MENAGERIE_ZOO),
]
