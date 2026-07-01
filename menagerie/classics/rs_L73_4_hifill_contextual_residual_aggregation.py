# SOURCE: vendored from wangyx240/High-Resolution-Image-Inpainting-GAN @ master
# Files: inpainting_network.py, network_module.py
# https://github.com/wangyx240/High-Resolution-Image-Inpainting-GAN
#
# The queue entry's official repo (Atlas200dk/sample-imageinpainting-HiFill)
# ships only a frozen TensorFlow `.pb` graph and a Huawei-Ascend `.om`
# binary (GPU_CPU/pb/hifill.pb, Huawei_Ascend/inpaint.om) -- no trainable
# nn.Module/tf.keras source, just an inference-only frozen graph plus a
# thin `test.py` session-runner. wangyx240/High-Resolution-Image-Inpainting-
# GAN is the paper authors'-adjacent, widely-cited "Unofficial Pytorch
# Re-implementation of 'Contextual Residual Aggregation for Ultra High-
# Resolution Image Inpainting' (CVPR 2020 Oral)" (the HiFill paper), with
# real trainable PyTorch source (its own README states this explicitly).
# Used here per rung 2 (vendor real repo code) since HiFill's own repo has
# no runnable/trainable source to vendor.
#
# Minimal changes from the original source:
#   - Merged inpainting_network.py and network_module.py into one file
#     (relative `from network_module import *` resolved by inlining
#     network_module's classes above inpainting_network.py's classes).
#   - Dropped the `from torchsummary import summary` import and the
#     `if __name__ == "__main__":` CLI/`summary(...)` debug block (a
#     third-party model-summary printer used only for a standalone debug
#     script, not part of the architecture).
#   - Dropped `PerceptualNet` (VGG-16 feature extractor used only for a
#     perceptual training loss).
#   - No changes to any nn.Module architecture: Coarse, GatedGenerator,
#     PatchDiscriminator, GatedConv2d, TransposeGatedConv2d, Conv2dLayer,
#     depth_separable_conv, sc_conv, and SpectralNorm are reproduced
#     verbatim from the source, including the Contextual Residual
#     Aggregation (CRA) attention math (cal_patch/compute_attention/
#     attention_transfer/extract_image_patches/cosine_Matrix) and its
#     hardcoded 256/512 intermediate-resolution assumptions (this
#     architecture is specifically designed around a fixed 512x512
#     working resolution with a 32x32 patch grid for CRA, so the example
#     input below uses that native size rather than a smaller synthetic
#     one).
#
# Architecture (unmodified from source): a coarse-to-fine gated-
# convolution generator (same GatedConv2d/TransposeGatedConv2d family as
# DeepFillv2) extended with the paper's two contributions. (1) "Light-
# Weight Gated Convolution": the coarse stage's GatedConv2d uses a
# single-channel gating conv (`sc_conv`, out_channels=1, broadcast over
# all output channels) instead of a full per-channel gate, and the
# refinement stage's GatedConv2d uses a depthwise-separable gating conv
# (`depth_separable_conv`), both far cheaper than DeepFillv2's full dense
# gating conv -- the paper's mechanism for scaling gated convolution to
# ultra-high-resolution (up to 8K) images. (2) Contextual Residual
# Aggregation (CRA): the coarse output is downsampled to 256x256 and
# refined at that low resolution (cheap), then the refinement network
# computes patch-wise cosine-similarity attention between hole and
# non-hole 32x32 patches of its own bottleneck features (cal_patch /
# compute_attention / cosine_Matrix), and uses that attention map to
# aggregate *high-resolution residual detail* from non-hole regions into
# the hole regions at three decoder stages (attention_transfer applied to
# pl1/pl2/pl3 skip features) -- reconstructing sharp high-frequency
# texture in the hole without ever running the expensive network at full
# resolution. A spectral-normalized PatchDiscriminator (Conv2dLayer stack)
# is included for completeness (used only during adversarial training).

import torch
import torch.nn as nn
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
        pad_type="replicate",
        activation="none",
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
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
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


class depth_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(depth_separable_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class sc_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super(sc_conv, self).__init__()
        self.single_channel_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )

    def forward(self, input):
        out = self.single_channel_conv(input)
        return out


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
        pad_type="replicate",
        activation="elu",
        norm="none",
        sc=False,
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
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
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
        if sc:
            self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
            )
            self.mask_conv2d = sc_conv(
                in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
            )
            self.mask_conv2d = depth_separable_conv(
                in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation
            )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_in):
        x = self.pad(x_in)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        if self.norm:
            conv = self.norm(conv)
        if self.activation:
            conv = self.activation(conv)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
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
        sc=False,
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
            sc,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.gated_conv2d(x)
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
class Coarse(nn.Module):
    def __init__(self, opt):
        super(Coarse, self).__init__()
        # Initialize the padding scheme
        self.coarse1 = nn.Sequential(
            # encoder
            GatedConv2d(4, 32, 5, 2, 2, activation=opt.activation, norm=opt.norm, sc=True),
            GatedConv2d(32, 32, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True),
            GatedConv2d(32, 64, 3, 2, 1, activation=opt.activation, norm=opt.norm, sc=True),
        )
        self.coarse2 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True),
        )
        self.coarse3 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True),
        )
        self.coarse4 = nn.Sequential(
            GatedConv2d(
                64, 64, 3, 1, 2, dilation=2, activation=opt.activation, norm=opt.norm, sc=True
            ),
            GatedConv2d(
                64, 64, 3, 1, 2, dilation=2, activation=opt.activation, norm=opt.norm, sc=True
            ),
            GatedConv2d(
                64, 64, 3, 1, 2, dilation=2, activation=opt.activation, norm=opt.norm, sc=True
            ),
        )
        self.coarse5 = nn.Sequential(
            GatedConv2d(
                64, 64, 3, 1, 4, dilation=4, activation=opt.activation, norm=opt.norm, sc=True
            ),
            GatedConv2d(
                64, 64, 3, 1, 4, dilation=4, activation=opt.activation, norm=opt.norm, sc=True
            ),
            GatedConv2d(
                64, 64, 3, 1, 4, dilation=4, activation=opt.activation, norm=opt.norm, sc=True
            ),
        )
        self.coarse6 = nn.Sequential(
            GatedConv2d(
                64, 64, 3, 1, 8, dilation=8, activation=opt.activation, norm=opt.norm, sc=True
            ),
            GatedConv2d(
                64, 64, 3, 1, 8, dilation=8, activation=opt.activation, norm=opt.norm, sc=True
            ),
            GatedConv2d(
                64, 64, 3, 1, 8, dilation=8, activation=opt.activation, norm=opt.norm, sc=True
            ),
        )
        self.coarse7 = nn.Sequential(
            GatedConv2d(
                64, 64, 3, 1, 16, dilation=16, activation=opt.activation, norm=opt.norm, sc=True
            ),
            GatedConv2d(
                64, 64, 3, 1, 16, dilation=16, activation=opt.activation, norm=opt.norm, sc=True
            ),
        )
        self.coarse8 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True),
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True),
        )
        # decoder
        self.coarse9 = nn.Sequential(
            TransposeGatedConv2d(
                64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True
            ),
            TransposeGatedConv2d(
                64, 32, 3, 1, 1, activation=opt.activation, norm=opt.norm, sc=True
            ),
            GatedConv2d(32, 3, 3, 1, 1, activation="none", norm=opt.norm, sc=True),
            nn.Tanh(),
        )

    def forward(self, first_in):
        first_out = self.coarse1(first_in)
        first_out = self.coarse2(first_out) + first_out
        first_out = self.coarse3(first_out) + first_out
        first_out = self.coarse4(first_out) + first_out
        first_out = self.coarse5(first_out) + first_out
        first_out = self.coarse6(first_out) + first_out
        first_out = self.coarse7(first_out) + first_out
        first_out = self.coarse8(first_out) + first_out
        first_out = self.coarse9(first_out)
        first_out = torch.clamp(first_out, 0, 1)
        return first_out


class GatedGenerator(nn.Module):
    def __init__(self, opt):
        super(GatedGenerator, self).__init__()

        # Coarse Network
        self.coarse = Coarse(opt)

        # Refinement Network
        self.refinement1 = nn.Sequential(
            GatedConv2d(3, 32, 5, 2, 2, activation=opt.activation, norm=opt.norm),  # [B,32,256,256]
            GatedConv2d(32, 32, 3, 1, 1, activation=opt.activation, norm=opt.norm),
        )
        self.refinement2 = nn.Sequential(
            # encoder
            GatedConv2d(32, 64, 3, 2, 1, activation=opt.activation, norm=opt.norm),
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm),
        )
        self.refinement3 = nn.Sequential(
            GatedConv2d(64, 128, 3, 2, 1, activation=opt.activation, norm=opt.norm)
        )
        self.refinement4 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 1, activation=opt.activation, norm=opt.norm),
            GatedConv2d(128, 128, 3, 1, 1, activation=opt.activation, norm=opt.norm),
        )
        self.refinement5 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 2, dilation=2, activation=opt.activation, norm=opt.norm),
            GatedConv2d(128, 128, 3, 1, 4, dilation=4, activation=opt.activation, norm=opt.norm),
        )
        self.refinement6 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 8, dilation=8, activation=opt.activation, norm=opt.norm),
            GatedConv2d(128, 128, 3, 1, 16, dilation=16, activation=opt.activation, norm=opt.norm),
        )
        self.refinement7 = nn.Sequential(
            GatedConv2d(256, 128, 3, 1, 1, activation=opt.activation, norm=opt.norm),
            TransposeGatedConv2d(128, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm),
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm),
        )
        self.refinement8 = nn.Sequential(
            TransposeGatedConv2d(128, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm),
            GatedConv2d(64, 32, 3, 1, 1, activation=opt.activation, norm=opt.norm),
        )
        self.refinement9 = nn.Sequential(
            TransposeGatedConv2d(64, 32, 3, 1, 1, activation=opt.activation, norm=opt.norm),
            GatedConv2d(32, 3, 3, 1, 1, activation="none", norm=opt.norm),
            nn.Tanh(),
        )
        self.conv_pl3 = nn.Sequential(
            GatedConv2d(128, 128, 3, 1, 1, activation=opt.activation, norm=opt.norm)
        )
        self.conv_pl2 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, norm=opt.norm),
            GatedConv2d(64, 64, 3, 1, 2, dilation=2, activation=opt.activation, norm=opt.norm),
        )
        self.conv_pl1 = nn.Sequential(
            GatedConv2d(32, 32, 3, 1, 1, activation=opt.activation, norm=opt.norm),
            GatedConv2d(32, 32, 3, 1, 2, dilation=2, activation=opt.activation, norm=opt.norm),
        )

    def forward(self, img, mask):
        img_256 = F.interpolate(img, size=[256, 256], mode="bilinear")
        mask_256 = F.interpolate(mask, size=[256, 256], mode="nearest")
        first_masked_img = img_256 * (1 - mask_256) + mask_256
        first_in = torch.cat((first_masked_img, mask_256), 1)  # in: [B, 4, H, W]
        first_out = self.coarse(first_in)  # out: [B, 3, H, W]
        first_out = F.interpolate(first_out, size=[512, 512], mode="bilinear")
        # Refinement
        second_in = img * (1 - mask) + first_out * mask
        pl1 = self.refinement1(second_in)  # out: [B, 32, 256, 256]
        pl2 = self.refinement2(pl1)  # out: [B, 64, 128, 128]
        second_out = self.refinement3(pl2)  # out: [B, 128, 64, 64]
        second_out = self.refinement4(second_out) + second_out  # out: [B, 128, 64, 64]
        second_out = self.refinement5(second_out) + second_out
        pl3 = self.refinement6(second_out) + second_out  # out: [B, 128, 64, 64]
        # Calculate Attention
        patch_fb = self.cal_patch(32, mask, 512)
        att = self.compute_attention(pl3, patch_fb)

        second_out = torch.cat(
            (pl3, self.conv_pl3(self.attention_transfer(pl3, att))), 1
        )  # out: [B, 256, 64, 64]
        second_out = self.refinement7(second_out)  # out: [B, 64, 128, 128]
        second_out = torch.cat(
            (second_out, self.conv_pl2(self.attention_transfer(pl2, att))), 1
        )  # out: [B, 128, 128, 128]
        second_out = self.refinement8(second_out)  # out: [B, 32, 256, 256]
        second_out = torch.cat(
            (second_out, self.conv_pl1(self.attention_transfer(pl1, att))), 1
        )  # out: [B, 64, 256, 256]
        second_out = self.refinement9(second_out)  # out: [B, 3, H, W]
        second_out = torch.clamp(second_out, 0, 1)
        return first_out, second_out

    def cal_patch(self, patch_num, mask, raw_size):
        pool = nn.MaxPool2d(raw_size // patch_num)  # patch_num=32
        patch_fb = pool(mask)  # out: [B, 1, 32, 32]
        return patch_fb

    def compute_attention(self, feature, patch_fb):  # in: [B, C:128, 64, 64]
        b = feature.shape[0]
        feature = F.interpolate(
            feature, scale_factor=0.5, mode="bilinear"
        )  # in: [B, C:128, 32, 32]
        p_fb = torch.reshape(patch_fb, [b, 32 * 32, 1])
        p_matrix = torch.bmm(p_fb, (1 - p_fb).permute([0, 2, 1]))
        f = feature.permute([0, 2, 3, 1]).reshape([b, 32 * 32, 128])
        c = self.cosine_Matrix(f, f) * p_matrix
        s = F.softmax(c, dim=2) * p_matrix
        return s

    def attention_transfer(self, feature, attention):  # feature: [B, C, H, W]
        b_num, c, h, w = feature.shape
        f = self.extract_image_patches(feature, 32)
        f = torch.reshape(f, [b_num, f.shape[1] * f.shape[2], -1])
        f = torch.bmm(attention, f)
        f = torch.reshape(f, [b_num, 32, 32, h // 32, w // 32, c])
        f = f.permute([0, 5, 1, 3, 2, 4])
        f = torch.reshape(f, [b_num, c, h, w])
        return f

    def extract_image_patches(self, img, patch_num):
        b, c, h, w = img.shape
        img = torch.reshape(img, [b, c, patch_num, h // patch_num, patch_num, w // patch_num])
        img = img.permute([0, 2, 4, 3, 5, 1])
        return img

    def cosine_Matrix(self, _matrixA, _matrixB):
        _matrixA_matrixB = torch.bmm(_matrixA, _matrixB.permute([0, 2, 1]))
        _matrixA_norm = torch.sqrt((_matrixA * _matrixA).sum(axis=2)).unsqueeze(dim=2)
        _matrixB_norm = torch.sqrt((_matrixB * _matrixB).sum(axis=2)).unsqueeze(dim=2)
        return _matrixA_matrixB / torch.bmm(_matrixA_norm, _matrixB_norm.permute([0, 2, 1]))


# -----------------------------------------------
#                  Discriminator
# -----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.sn = True
        self.norm = "in"
        self.batchsize = opt.batch_size
        self.block1 = Conv2dLayer(4, 64, 3, 2, 1, activation="lrelu", norm=self.norm, sn=self.sn)
        self.block2 = Conv2dLayer(64, 128, 3, 2, 1, activation="lrelu", norm=self.norm, sn=self.sn)
        self.block3 = Conv2dLayer(128, 256, 3, 2, 1, activation="lrelu", norm=self.norm, sn=self.sn)
        self.block4 = Conv2dLayer(256, 256, 3, 2, 1, activation="lrelu", norm=self.norm, sn=self.sn)
        self.block5 = Conv2dLayer(256, 256, 3, 2, 1, activation="lrelu", norm=self.norm, sn=self.sn)
        self.block6 = Conv2dLayer(256, 16, 3, 2, 1, activation="lrelu", norm=self.norm, sn=self.sn)
        self.block7 = torch.nn.Linear(1024, 1)

    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        x = self.block1(x)  # out: [B, 64, 256, 256]
        x = self.block2(x)  # out: [B, 128, 128, 128]
        x = self.block3(x)  # out: [B, 256, 64, 64]
        x = self.block4(x)  # out: [B, 256, 32, 32]
        x = self.block5(x)  # out: [B, 256, 16, 16]
        x = self.block6(x)  # out: [B, 256, 8, 8]
        x = x.reshape([x.shape[0], -1])
        x = self.block7(x)
        return x


class _HiFillOpt:
    """Tiny stand-in for the argparse.Namespace built in the original CLI."""

    def __init__(self):
        self.activation = "elu"
        self.norm = "none"
        self.batch_size = 1


def build_hifill_contextual_residual_aggregation():
    return GatedGenerator(_HiFillOpt())


def example_input_hifill_contextual_residual_aggregation():
    # CRA's attention math hardcodes a 512x512 working resolution with a
    # 32x32 patch grid (see cal_patch/compute_attention above), so the
    # example input must match that native size.
    torch.manual_seed(0)
    image = torch.rand(1, 3, 512, 512)
    mask = torch.zeros(1, 1, 512, 512)
    mask[:, :, 200:300, 200:300] = 1.0
    return (image, mask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "HiFill Contextual Residual Aggregation",
        "build_hifill_contextual_residual_aggregation",
        "example_input_hifill_contextual_residual_aggregation",
        2020,
        "vendored",
    ),
]
