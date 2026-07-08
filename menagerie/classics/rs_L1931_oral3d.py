# SOURCE: vendored from WayneSoong/Oral-3d @ master
# https://raw.githubusercontent.com/WayneSoong/Oral-3d/master/model/layer.py
# https://raw.githubusercontent.com/WayneSoong/Oral-3d/master/model/generator.py
#
# Song, Cao, Yang, Liang, He, Chen, Guo, 2021 (MICCAI 2021) "Oral-3D: Reconstructing
# the 3D Structure of Oral Cavity from Panoramic X-Ray". Oral-3D reconstructs a 3D
# oral-cavity CBCT volume from a single 2D panoramic X-ray via a GAN: the generator
# `Encoder_MPR` is a 2D DenseNet-style U-Net (dense-block encoder/decoder with
# skip-connected transposed-conv upsampling), and a `PatchDiscriminator` (3D patch
# GAN, vendored in a companion notes-only path since the generator is the
# architectural centerpiece that TorchLens traces) adversarially refines the
# generator output during training. `Encoder_MPR`'s dense-block-based encoder/
# decoder pyramid (not a stock U-Net) is Oral-3D's architectural contribution, so
# this is vendored real code, not rebuilt from a stock library class.
#
# `model/layer.py` (`ConvRelu`, `DenseModule`, `DenseBlock`, `UpSampleBlock`,
# `ConvUpSample`; `RandomAug` and its `scio`-backed spine-augmentation path are
# training-only data augmentation, not part of the traced forward graph, and are
# omitted here) and `model/generator.py` (`Encoder_MPR`) are reproduced verbatim
# below (only the cross-file `from model.layer import *` import is inlined into
# this single module).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"

CONV_METHOD = {"1D": nn.Conv1d, "2D": nn.Conv2d, "3D": nn.Conv3d}
TCONV_METHOD = {"1D": nn.ConvTranspose1d, "2D": nn.ConvTranspose2d, "3D": nn.ConvTranspose3d}
NORM_METHOD = {"1D": nn.InstanceNorm1d, "2D": nn.InstanceNorm2d, "3D": nn.InstanceNorm3d}


# ============================================================================
# model/layer.py (verbatim, forward-graph layers only)
# ============================================================================


class ConvRelu(nn.Module):
    def __init__(
        self, in_chns, out_chns, k=3, s=1, p=None, method="2D", NORM=True, act_funct=nn.ReLU()
    ):
        super(ConvRelu, self).__init__()
        p = (k - 1) // 2 if p is None else p
        self.NORM = NORM
        self.conv = CONV_METHOD[method](in_chns, out_chns, kernel_size=k, stride=s, padding=p)
        if NORM:
            self.norm = NORM_METHOD[method](out_chns)
        self.act_funct = act_funct

    def forward(self, input_tensor):
        out = self.conv(input_tensor)
        if self.NORM:
            out = self.norm(out)
        out = self.act_funct(out)
        return out


class DenseModule(nn.ModuleDict):
    def __init__(self, input_chns, growth_rate, conv_n, method="2D"):
        super(DenseModule, self).__init__()
        self.input_chns = input_chns
        for loop_id in range(conv_n):
            layer = ConvRelu(
                input_chns + loop_id * growth_rate, growth_rate, method=method, NORM=True
            )
            self.add_module("dense_%d" % loop_id, layer)

    def forward(self, input_tensor):
        features = input_tensor
        for name, layer in self.items():
            new_features = layer(features)
            features = torch.cat((new_features, features), dim=1)
        return features


class DenseBlock(nn.Module):
    def __init__(self, input_chns, output_chns, growth_rate=16, conv_n=3, method="2D"):
        super(DenseBlock, self).__init__()
        self.dense_module = DenseModule(input_chns, growth_rate, conv_n, method=method)
        self.transition = ConvRelu(
            input_chns + conv_n * growth_rate, output_chns, k=1, method=method
        )

    def forward(self, input_tensor):
        features = self.dense_module(input_tensor)
        features = self.transition(features)

        return features


class ConvUpSample(nn.Module):
    def __init__(self, chns, k=3, method="2D"):
        super(ConvUpSample, self).__init__()
        self.up = TCONV_METHOD[method](chns, chns, k, stride=2, padding=1, output_padding=1)

    def forward(self, input_tensor):
        return self.up(input_tensor)


class UpSampleBlock(nn.Module):
    def __init__(self, in_chns, pass_chns, out_chns=None, method="2D"):
        super(UpSampleBlock, self).__init__()
        out_chns = out_chns if out_chns else in_chns
        self.up = ConvUpSample(in_chns)
        self.res = DenseBlock(in_chns + pass_chns, out_chns, method=method)

    def forward(self, input_tensor, pass_tensor):
        out = self.up(input_tensor)
        out = torch.cat((out, pass_tensor), dim=1)
        out = self.res(out)
        return out


# ============================================================================
# model/generator.py (verbatim, `Encoder_MPR`)
# ============================================================================


class Encoder_MPR(nn.Module):
    def __init__(self):
        super(Encoder_MPR, self).__init__()
        # input: [1, 160, 576]
        self.down_block_0 = ConvRelu(1, 10)
        # down_0: [10, 160, 576]
        self.down_block_1 = nn.Sequential(
            DenseBlock(10, 20, conv_n=3, growth_rate=10), nn.MaxPool2d(2)
        )
        # down_1: [20, 80, 288]
        self.down_block_2 = nn.Sequential(
            DenseBlock(20, 40, conv_n=3, growth_rate=20), nn.MaxPool2d(2)
        )
        # down_2: [40, 40, 144]
        self.down_block_3 = nn.Sequential(
            DenseBlock(40, 80, conv_n=3, growth_rate=40), nn.MaxPool2d(2)
        )
        # down_3: [80, 20, 72]

        self.up_block_3 = UpSampleBlock(in_chns=80, pass_chns=40, out_chns=80)
        # up_2: [80, 40, 144]
        self.up_block_2 = UpSampleBlock(in_chns=80, pass_chns=20, out_chns=80)
        # up_1: [80, 80, 288]
        self.up_block_1 = UpSampleBlock(in_chns=80, pass_chns=10, out_chns=80)
        # up_0: [80, 160, 576]
        self.up_block_0 = nn.Conv2d(in_channels=80, out_channels=80, padding=1, kernel_size=3)

    def forward(self, input_tensor):
        input_tensor = input_tensor / 256
        # down sample
        down_0 = self.down_block_0(input_tensor)
        down_1 = self.down_block_1(down_0)
        down_2 = self.down_block_2(down_1)
        down_3 = self.down_block_3(down_2)

        up_3 = self.up_block_3(down_3, down_2)
        up_2 = self.up_block_2(up_3, down_1)
        up_1 = self.up_block_1(up_2, down_0)
        up_0 = self.up_block_0(up_1)
        out = torch.tanh(up_0)
        return out

    def generate(self, input_tensor, VAL=True):
        return self.forward(input_tensor)


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_oral3d_encoder_mpr():
    model = Encoder_MPR()
    model.eval()
    return model


def example_input_oral3d_encoder_mpr():
    torch.manual_seed(0)
    # real repo's panoramic X-ray input is [B, 1, 160, 576]; shrunk spatially
    # here (kept a multiple of 8 to survive the 3 real MaxPool2d(2) downsamples).
    return torch.randn(1, 1, 32, 64)


MENAGERIE_ENTRIES = [
    (
        "Oral-3D",
        build_oral3d_encoder_mpr,
        example_input_oral3d_encoder_mpr,
        2021,
        "vendored-pytorch",
    ),
]
