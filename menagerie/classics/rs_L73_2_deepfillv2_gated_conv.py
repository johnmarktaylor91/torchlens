# SOURCE: vendored from zhaoyuzhi/deepfillv2 @ master
# Files: deepfillv2/network.py, deepfillv2/network_module.py
# https://github.com/zhaoyuzhi/deepfillv2
#
# The queue entry's official repo (JiahuiYu/generative_inpainting, the
# DeepFillv2 / "Free-Form Image Inpainting with Gated Convolution" ICCV
# 2019 oral) is TensorFlow 1.x (`inpaint_model.py` builds the graph with
# `tf.contrib`/`tf.layers`, which do not run on any base-env framework
# here). zhaoyuzhi/deepfillv2 is a widely-used, faithful PyTorch
# reimplementation of the same architecture (coarse-to-fine two-stage
# generator with gated convolutions replacing every plain conv, spectral-
# normalized PatchGAN discriminator) and is the same code family the
# queue's own notes call out ("multiple PyTorch reimplementations
# (nipponjo/deepfillv2-pytorch, zhaoyuzhi/deepfillv2)"). Used here per
# rung 2 (vendor real repo code).
#
# Minimal changes from the original source:
#   - Merged network.py and network_module.py into one file (relative
#     `from network_module import *` resolved by inlining network_module's
#     classes above network.py's classes).
#   - Dropped the `weights_init` helper (training-only) and the
#     `PerceptualNet` VGG-16 feature extractor (used only for a perceptual
#     training loss, not part of the inpainting architecture itself).
#   - Removed a stray `print(img.shape, mask.shape)` debug statement from
#     `GatedGenerator.forward` (present in the original source; cosmetic
#     only, does not affect the computation graph).
#   - No changes to any nn.Module architecture: GatedGenerator,
#     PatchDiscriminator, GatedConv2d, TransposeGatedConv2d, Conv2dLayer,
#     LayerNorm, and SpectralNorm are reproduced verbatim from the source.
#
# Architecture (unmodified from source): DeepFillv2's GatedGenerator is a
# two-stage coarse-to-fine network. Stage 1 ("coarse") is an encoder-
# decoder over the masked image concatenated with the mask (4 input
# channels): strided GatedConv2d downsampling, a dilated-conv bottleneck
# (dilations 1/1/2/4/8/16/1/1) for large receptive field without losing
# resolution, then TransposeGatedConv2d upsampling back to RGB with a
# final Tanh. Stage 2 ("refinement") repeats the same coarse-to-fine
# encoder/dilated-bottleneck/decoder shape, but is fed the *stage-1
# output* composited back into the unmasked regions. Every convolution
# in both stages is a GatedConv2d: two parallel convs (one produces
# features, the other -- through a sigmoid -- produces a per-pixel soft
# gate) whose product is the layer's output, letting the network learn
# which spatial locations (valid pixels vs. hole vs. hole boundary) to
# trust at each layer, in contrast to plain or partial convolutions.
# PatchDiscriminator is a spectral-normalized fully-convolutional patch
# discriminator (SN-PatchGAN) over the image+mask channels.

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter


# -----------------------------------------------
#                Normal ConvBlock
# -----------------------------------------------
class Conv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="zero",
        activation="lrelu",
        norm="none",
        sn=False,
    ):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == "ln":
            self.norm = LayerNorm(out_channels)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
                )
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
            )

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeConv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="zero",
        activation="lrelu",
        norm="none",
        sn=False,
        scale_factor=2,
    ):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            pad_type,
            activation,
            norm,
            sn,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv2d(x)
        return x


# -----------------------------------------------
#                Gated ConvBlock
# -----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="reflect",
        activation="lrelu",
        norm="none",
        sn=False,
    ):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == "ln":
            self.norm = LayerNorm(out_channels)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
                )
            )
            self.mask_conv2d = SpectralNorm(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
                )
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
            )
            self.mask_conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
            )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeGatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        pad_type="zero",
        activation="lrelu",
        norm="none",
        sn=True,
        scale_factor=2,
    ):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            pad_type,
            activation,
            norm,
            sn,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.gated_conv2d(x)
        return x


# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)  # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# -----------------------------------------------
#                  SpectralNorm
# -----------------------------------------------
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# -----------------------------------------------
#                   Generator
# -----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GatedGenerator(nn.Module):
    def __init__(self, opt):
        super(GatedGenerator, self).__init__()
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(
                opt.in_channels,
                opt.latent_channels,
                7,
                1,
                3,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm="none",
            ),
            GatedConv2d(
                opt.latent_channels,
                opt.latent_channels * 2,
                4,
                2,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 2,
                opt.latent_channels * 4,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                4,
                2,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            # Bottleneck
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                2,
                dilation=2,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                4,
                dilation=4,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                8,
                dilation=8,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                16,
                dilation=16,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            # decoder
            TransposeGatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 2,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 2,
                opt.latent_channels * 2,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            TransposeGatedConv2d(
                opt.latent_channels * 2,
                opt.latent_channels,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels,
                opt.out_channels,
                7,
                1,
                3,
                pad_type=opt.pad_type,
                activation="tanh",
                norm="none",
            ),
        )
        self.refinement = nn.Sequential(
            # encoder
            GatedConv2d(
                opt.in_channels,
                opt.latent_channels,
                7,
                1,
                3,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm="none",
            ),
            GatedConv2d(
                opt.latent_channels,
                opt.latent_channels * 2,
                4,
                2,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 2,
                opt.latent_channels * 4,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                4,
                2,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            # Bottleneck
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                2,
                dilation=2,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                4,
                dilation=4,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                8,
                dilation=8,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                16,
                dilation=16,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 4,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            # decoder
            TransposeGatedConv2d(
                opt.latent_channels * 4,
                opt.latent_channels * 2,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels * 2,
                opt.latent_channels * 2,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            TransposeGatedConv2d(
                opt.latent_channels * 2,
                opt.latent_channels,
                3,
                1,
                1,
                pad_type=opt.pad_type,
                activation=opt.activation,
                norm=opt.norm,
            ),
            GatedConv2d(
                opt.latent_channels,
                opt.out_channels,
                7,
                1,
                3,
                pad_type=opt.pad_type,
                activation="tanh",
                norm="none",
            ),
        )

    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # Coarse
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), 1)  # in: [B, 4, H, W]
        first_out = self.coarse(first_in)  # out: [B, 3, H, W]
        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat((second_masked_img, mask), 1)  # in: [B, 4, H, W]
        second_out = self.refinement(second_in)  # out: [B, 3, H, W]
        return first_out, second_out


# -----------------------------------------------
#                  Discriminator
# -----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(
            opt.in_channels,
            opt.latent_channels,
            7,
            1,
            3,
            pad_type=opt.pad_type,
            activation=opt.activation,
            norm="none",
            sn=True,
        )
        self.block2 = Conv2dLayer(
            opt.latent_channels,
            opt.latent_channels * 2,
            4,
            2,
            1,
            pad_type=opt.pad_type,
            activation=opt.activation,
            norm=opt.norm,
            sn=True,
        )
        self.block3 = Conv2dLayer(
            opt.latent_channels * 2,
            opt.latent_channels * 4,
            4,
            2,
            1,
            pad_type=opt.pad_type,
            activation=opt.activation,
            norm=opt.norm,
            sn=True,
        )
        self.block4 = Conv2dLayer(
            opt.latent_channels * 4,
            opt.latent_channels * 4,
            4,
            2,
            1,
            pad_type=opt.pad_type,
            activation=opt.activation,
            norm=opt.norm,
            sn=True,
        )
        self.block5 = Conv2dLayer(
            opt.latent_channels * 4,
            opt.latent_channels * 4,
            4,
            2,
            1,
            pad_type=opt.pad_type,
            activation=opt.activation,
            norm=opt.norm,
            sn=True,
        )
        self.block6 = Conv2dLayer(
            opt.latent_channels * 4,
            1,
            4,
            2,
            1,
            pad_type=opt.pad_type,
            activation="none",
            norm="none",
            sn=True,
        )

    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        x = self.block1(x)  # out: [B, 64, 256, 256]
        x = self.block2(x)  # out: [B, 128, 128, 128]
        x = self.block3(x)  # out: [B, 256, 64, 64]
        x = self.block4(x)  # out: [B, 256, 32, 32]
        x = self.block5(x)  # out: [B, 256, 16, 16]
        x = self.block6(x)  # out: [B, 256, 8, 8]
        return x


class _DeepFillV2Opt:
    """Tiny stand-in for the argparse.Namespace built in the original train.py CLI."""

    def __init__(self):
        self.in_channels = 4
        self.out_channels = 3
        self.latent_channels = 8
        self.pad_type = "zero"
        self.activation = "lrelu"
        self.norm = "in"


def build_deepfillv2_gated_generator():
    return GatedGenerator(_DeepFillV2Opt())


def example_input_deepfillv2_gated_generator():
    torch.manual_seed(0)
    image = torch.rand(1, 3, 32, 32)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 10:20, 10:20] = 1.0
    return (image, mask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepFillv2 Gated Convolution Inpainting",
        "build_deepfillv2_gated_generator",
        "example_input_deepfillv2_gated_generator",
        2019,
        "vendored",
    ),
]
