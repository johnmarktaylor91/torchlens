# SOURCE: vendored from Sooyyoungg/AesFA @ main
# Files: networks.py (Encoder, Decoder, define_network), blocks.py
#        (OctConv, AdaOctConv, KernelPredictor, AdaConv2d, Oct_Conv_aftup,
#        Oct_conv_reLU/lreLU/up)
# https://github.com/Sooyyoungg/AesFA
#
# Minimal changes from the original source:
#   - Dropped everything in blocks.py that is training/checkpoint plumbing
#     unrelated to the architecture itself: `model_save`, `model_load`,
#     `test_model_load`, `get_scheduler`, `update_learning_rate` (these
#     import `path`/`torch.optim.lr_scheduler`, not part of the network).
#   - Dropped `EFDM_loss`, `calc_content_loss`, `calc_style_loss` from
#     networks.py (loss functions, not part of the Encoder/Decoder
#     architecture) and the `AesFA`/`AesFA_test` training wrapper classes
#     from model.py (those wire in VGG perceptual loss + optimizers/
#     schedulers, which are training-time concerns, not new architecture --
#     the actual style-transfer network is `Encoder` + `Decoder` called
#     directly, matching `AesFA_test.forward`'s real inference path:
#     `content_A = netE.forward_test(real_A, 'content')`,
#     `style_B = netS.forward_test(real_B, 'style')`,
#     `out = netG(content_A, style_B)`).
#   - `define_network` kept verbatim as the real constructor entry point
#     used by the original `AesFA.__init__`.
#
# Architecture (unmodified from source): AesFA (AAAI 2024, "Attentive Style
# Frequency-Aware GAN..." / "Frequency-Aware GAN" style-transfer paper). Two
# structurally-identical Octave-Convolution encoders (`Encoder`, shared class
# used for both `netE` content-encoder and `netS` style-encoder) decompose
# each image into high/low spatial-frequency feature streams via `OctConv`
# (Octave Convolution: cross-frequency H2H/H2L/L2H/L2L conv paths with
# avg-pool/upsample frequency mixing). The style path's frequency-split
# features are globally pooled (style_kernel adaptive-avg-pool) into a
# compact style code. The `Decoder` fuses content octave-features with the
# style code via `AdaOctConv` -- an Adaptive-Octave-Convolution block that
# predicts per-sample spatial+pointwise depthwise-separable kernels
# (`KernelPredictor` + `AdaConv2d`, same adaptive-convolution mechanism as
# AdaConv CVPR'21) *separately* for the high- and low-frequency streams, then
# runs them through another `OctConv` -- upsampling and repeating across
# three decoder stages before a final frequency-recombining 1x1 conv.

import math

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# blocks.py (architecture-relevant classes only)
# ---------------------------------------------------------------------------


class Oct_Conv_aftup(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, pad_type, alpha_in, alpha_out
    ):
        super(Oct_Conv_aftup, self).__init__()
        lf_in = int(in_channels * alpha_in)
        lf_out = int(out_channels * alpha_out)
        hf_in = in_channels - lf_in
        hf_out = out_channels - lf_out

        self.conv_h = nn.Conv2d(
            in_channels=hf_in,
            out_channels=hf_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
        )
        self.conv_l = nn.Conv2d(
            in_channels=lf_in,
            out_channels=lf_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
        )

    def forward(self, x):
        hf, lf = x
        hf = self.conv_h(hf)
        lf = self.conv_l(lf)
        return hf, lf


class Oct_conv_reLU(nn.ReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_reLU, self).forward(hf)
        lf = super(Oct_conv_reLU, self).forward(lf)
        return hf, lf


class Oct_conv_lreLU(nn.LeakyReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_lreLU, self).forward(hf)
        lf = super(Oct_conv_lreLU, self).forward(lf)
        return hf, lf


class Oct_conv_up(nn.Upsample):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_up, self).forward(hf)
        lf = super(Oct_conv_up, self).forward(lf)
        return hf, lf


############## Encoder ##############
class OctConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        pad_type="reflect",
        alpha_in=0.5,
        alpha_out=0.5,
        type="normal",
        freq_ratio=[1, 1],
    ):
        super(OctConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.freq_ratio = freq_ratio

        hf_ch_in = int(in_channels * (1 - self.alpha_in))
        hf_ch_out = int(out_channels * (1 - self.alpha_out))
        lf_ch_in = in_channels - hf_ch_in
        lf_ch_out = out_channels - hf_ch_out

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.is_dw = groups == in_channels

        if type == "first":
            self.convh = nn.Conv2d(
                in_channels,
                hf_ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=pad_type,
                bias=False,
            )
            self.convl = nn.Conv2d(
                in_channels,
                lf_ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=pad_type,
                bias=False,
            )
        elif type == "last":
            self.convh = nn.Conv2d(
                hf_ch_in,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=pad_type,
                bias=False,
            )
            self.convl = nn.Conv2d(
                lf_ch_in,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=pad_type,
                bias=False,
            )
        else:
            self.L2L = nn.Conv2d(
                lf_ch_in,
                lf_ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=math.ceil(alpha_in * groups),
                padding_mode=pad_type,
                bias=False,
            )
            if self.is_dw:
                self.L2H = None
                self.H2L = None
            else:
                self.L2H = nn.Conv2d(
                    lf_ch_in,
                    hf_ch_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    padding_mode=pad_type,
                    bias=False,
                )
                self.H2L = nn.Conv2d(
                    hf_ch_in,
                    lf_ch_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    padding_mode=pad_type,
                    bias=False,
                )
            self.H2H = nn.Conv2d(
                hf_ch_in,
                hf_ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=math.ceil(groups - alpha_in * groups),
                padding_mode=pad_type,
                bias=False,
            )

    def forward(self, x):
        if self.type == "first":
            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)
            return hf, lf
        elif self.type == "last":
            hf, lf = x
            out_h = self.convh(hf)
            out_l = self.convl(self.upsample(lf))
            output = out_h * self.freq_ratio[0] + out_l * self.freq_ratio[1]
            return output, out_h, out_l
        else:
            hf, lf = x
            if self.is_dw:
                hf, lf = self.H2H(hf), self.L2L(lf)
            else:
                hf, lf = (
                    self.H2H(hf) + self.L2H(self.upsample(lf)),
                    self.L2L(lf) + self.H2L(self.avg_pool(hf)),
                )
            return hf, lf


############## Decoder ##############
class AdaOctConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        group_div,
        style_channels,
        kernel_size,
        stride,
        padding,
        oct_groups,
        alpha_in,
        alpha_out,
        type="normal",
    ):
        super(AdaOctConv, self).__init__()
        self.in_channels = in_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.type = type

        h_in = int(in_channels * (1 - self.alpha_in))
        l_in = in_channels - h_in

        n_groups_h = h_in // group_div
        n_groups_l = l_in // group_div

        style_channels_h = int(style_channels * (1 - self.alpha_in))
        style_channels_l = int(style_channels - style_channels_h)

        kernel_size_h = kernel_size[0]
        kernel_size_l = kernel_size[1]
        kernel_size_A = kernel_size[2]

        self.kernelPredictor_h = KernelPredictor(
            in_channels=h_in,
            out_channels=h_in,
            n_groups=n_groups_h,
            style_channels=style_channels_h,
            kernel_size=kernel_size_h,
        )
        self.kernelPredictor_l = KernelPredictor(
            in_channels=l_in,
            out_channels=l_in,
            n_groups=n_groups_l,
            style_channels=style_channels_l,
            kernel_size=kernel_size_l,
        )

        self.AdaConv_h = AdaConv2d(in_channels=h_in, out_channels=h_in, n_groups=n_groups_h)
        self.AdaConv_l = AdaConv2d(in_channels=l_in, out_channels=l_in, n_groups=n_groups_l)

        self.OctConv = OctConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_A,
            stride=stride,
            padding=padding,
            groups=oct_groups,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type=type,
        )

        self.relu = Oct_conv_lreLU()

    def forward(self, content, style, cond="train"):
        c_hf, c_lf = content
        s_hf, s_lf = style
        h_w_spatial, h_w_pointwise, h_bias = self.kernelPredictor_h(s_hf)
        l_w_spatial, l_w_pointwise, l_bias = self.kernelPredictor_l(s_lf)

        if cond == "train":
            output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
            output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
            output = output_h, output_l

            output = self.relu(output)

            output = self.OctConv(output)
            if self.type != "last":
                output = self.relu(output)
            return output

        if cond == "test":
            output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
            output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
            output = output_h, output_l
            output = self.relu(output)
            output = self.OctConv(output)
            if self.type != "last":
                output = self.relu(output)
            return output


class KernelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, style_channels, kernel_size):
        super(KernelPredictor, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.w_channels = style_channels
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(
            style_channels,
            in_channels * out_channels // n_groups,
            kernel_size=kernel_size,
            padding=(math.ceil(padding), math.ceil(padding)),
            padding_mode="reflect",
        )
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels, out_channels * out_channels // n_groups, kernel_size=1),
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(style_channels, out_channels, kernel_size=1)
        )

    def forward(self, w):
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(
            len(w),
            self.out_channels,
            self.in_channels // self.n_groups,
            self.kernel_size,
            self.kernel_size,
        )

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(
            len(w), self.out_channels, self.out_channels // self.n_groups, 1, 1
        )
        bias = self.bias(w)
        bias = bias.reshape(len(w), self.out_channels)
        return w_spatial, w_pointwise, bias


class AdaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super(AdaConv2d, self).__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(math.ceil(padding), math.floor(padding)),
            padding_mode="reflect",
        )

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = F.instance_norm(x)

        ys = []
        for i in range(len(x)):
            y = self.forward_single(x[i : i + 1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)

        ys = self.conv(ys)
        return ys

    def forward_single(self, x, w_spatial, w_pointwise, bias):
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (math.ceil(padding), math.floor(padding), math.ceil(padding), math.floor(padding))

        x = F.pad(x, pad=pad, mode="reflect")
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x


# ---------------------------------------------------------------------------
# networks.py (Encoder, Decoder, define_network)
# ---------------------------------------------------------------------------


def define_network(net_type, config=None):
    net = None
    alpha_in = config.alpha_in
    alpha_out = config.alpha_out
    sk = config.style_kernel

    if net_type == "Encoder":
        net = Encoder(
            in_dim=config.input_nc,
            nf=config.nf,
            style_kernel=[sk, sk],
            alpha_in=alpha_in,
            alpha_out=alpha_out,
        )
    elif net_type == "Generator":
        net = Decoder(
            nf=config.nf,
            out_dim=config.output_nc,
            style_channel=256,
            style_kernel=[sk, sk, 3],
            alpha_in=alpha_in,
            freq_ratio=config.freq_ratio,
            alpha_out=alpha_out,
        )
    return net


class Encoder(nn.Module):
    def __init__(self, in_dim, nf=64, style_kernel=[3, 3], alpha_in=0.5, alpha_out=0.5):
        super(Encoder, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_dim, out_channels=nf, kernel_size=7, stride=1, padding=3
        )

        self.OctConv1_1 = OctConv(
            in_channels=nf,
            out_channels=nf,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=64,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="first",
        )
        self.OctConv1_2 = OctConv(
            in_channels=nf,
            out_channels=2 * nf,
            kernel_size=1,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.OctConv1_3 = OctConv(
            in_channels=2 * nf,
            out_channels=2 * nf,
            kernel_size=3,
            stride=1,
            padding=1,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )

        self.OctConv2_1 = OctConv(
            in_channels=2 * nf,
            out_channels=2 * nf,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=128,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.OctConv2_2 = OctConv(
            in_channels=2 * nf,
            out_channels=4 * nf,
            kernel_size=1,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.OctConv2_3 = OctConv(
            in_channels=4 * nf,
            out_channels=4 * nf,
            kernel_size=3,
            stride=1,
            padding=1,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )

        self.pool_h = nn.AdaptiveAvgPool2d((style_kernel[0], style_kernel[0]))
        self.pool_l = nn.AdaptiveAvgPool2d((style_kernel[1], style_kernel[1]))

        self.relu = Oct_conv_lreLU()

    def forward(self, x):
        enc_feat = []
        out = self.conv(x)

        out = self.OctConv1_1(out)
        out = self.relu(out)
        out = self.OctConv1_2(out)
        out = self.relu(out)
        out = self.OctConv1_3(out)
        out = self.relu(out)
        enc_feat.append(out)

        out = self.OctConv2_1(out)
        out = self.relu(out)
        out = self.OctConv2_2(out)
        out = self.relu(out)
        out = self.OctConv2_3(out)
        out = self.relu(out)
        enc_feat.append(out)

        out_high, out_low = out
        out_sty_h = self.pool_h(out_high)
        out_sty_l = self.pool_l(out_low)
        out_sty = out_sty_h, out_sty_l

        return out, out_sty, enc_feat

    def forward_test(self, x, cond):
        out = self.conv(x)

        out = self.OctConv1_1(out)
        out = self.relu(out)
        out = self.OctConv1_2(out)
        out = self.relu(out)
        out = self.OctConv1_3(out)
        out = self.relu(out)

        out = self.OctConv2_1(out)
        out = self.relu(out)
        out = self.OctConv2_2(out)
        out = self.relu(out)
        out = self.OctConv2_3(out)
        out = self.relu(out)

        if cond == "style":
            out_high, out_low = out
            out_sty_h = self.pool_h(out_high)
            out_sty_l = self.pool_l(out_low)
            return out_sty_h, out_sty_l
        else:
            return out


class Decoder(nn.Module):
    def __init__(
        self,
        nf=64,
        out_dim=3,
        style_channel=512,
        style_kernel=[3, 3, 3],
        alpha_in=0.5,
        alpha_out=0.5,
        freq_ratio=[1, 1],
        pad_type="reflect",
    ):
        super(Decoder, self).__init__()

        group_div = [1, 2, 4, 8]
        self.up_oct = Oct_conv_up(scale_factor=2)

        self.AdaOctConv1_1 = AdaOctConv(
            in_channels=4 * nf,
            out_channels=4 * nf,
            group_div=group_div[0],
            style_channels=style_channel,
            kernel_size=style_kernel,
            stride=1,
            padding=1,
            oct_groups=4 * nf,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.OctConv1_2 = OctConv(
            in_channels=4 * nf,
            out_channels=2 * nf,
            kernel_size=1,
            stride=1,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.oct_conv_aftup_1 = Oct_Conv_aftup(
            in_channels=2 * nf,
            out_channels=2 * nf,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_type=pad_type,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
        )

        self.AdaOctConv2_1 = AdaOctConv(
            in_channels=2 * nf,
            out_channels=2 * nf,
            group_div=group_div[1],
            style_channels=style_channel,
            kernel_size=style_kernel,
            stride=1,
            padding=1,
            oct_groups=2 * nf,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.OctConv2_2 = OctConv(
            in_channels=2 * nf,
            out_channels=nf,
            kernel_size=1,
            stride=1,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.oct_conv_aftup_2 = Oct_Conv_aftup(nf, nf, 3, 1, 1, pad_type, alpha_in, alpha_out)

        self.AdaOctConv3_1 = AdaOctConv(
            in_channels=nf,
            out_channels=nf,
            group_div=group_div[2],
            style_channels=style_channel,
            kernel_size=style_kernel,
            stride=1,
            padding=1,
            oct_groups=nf,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="normal",
        )
        self.OctConv3_2 = OctConv(
            in_channels=nf,
            out_channels=nf // 2,
            kernel_size=1,
            stride=1,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            type="last",
            freq_ratio=freq_ratio,
        )

        self.conv4 = nn.Conv2d(in_channels=nf // 2, out_channels=out_dim, kernel_size=1)

    def forward(self, content, style):
        out = self.AdaOctConv1_1(content, style)
        out = self.OctConv1_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_1(out)

        out = self.AdaOctConv2_1(out, style)
        out = self.OctConv2_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_2(out)

        out = self.AdaOctConv3_1(out, style)
        out = self.OctConv3_2(out)
        out, out_high, out_low = out

        out = self.conv4(out)
        out_high = self.conv4(out_high)
        out_low = self.conv4(out_low)

        return out, out_high, out_low

    def forward_test(self, content, style):
        out = self.AdaOctConv1_1(content, style, "test")
        out = self.OctConv1_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_1(out)

        out = self.AdaOctConv2_1(out, style, "test")
        out = self.OctConv2_2(out)
        out = self.up_oct(out)
        out = self.oct_conv_aftup_2(out)

        out = self.AdaOctConv3_1(out, style, "test")
        out = self.OctConv3_2(out)

        out = self.conv4(out[0])
        return out


# ---------------------------------------------------------------------------
# Menagerie harness
# ---------------------------------------------------------------------------


class _Config:
    """Minimal stand-in for the original repo's Config class (config.py),
    carrying only the fields `define_network` reads. `nf` is kept at the
    real default (Config.nf=64) because Encoder hardcodes `groups=64`/
    `groups=128` in its first two OctConv stages (tied to nf=64) -- shrinking
    nf would silently change the group structure, not just size."""

    input_nc = 3
    output_nc = 3
    nf = 64
    style_kernel = 3
    alpha_in = 0.5
    alpha_out = 0.5
    freq_ratio = [1, 1]


class AesFAStyleTransfer(nn.Module):
    """Inference-path wiring matching the original `AesFA_test.forward`:
    content-encode -> style-encode -> AdaOctConv-fused decode."""

    def __init__(self, config):
        super().__init__()
        self.netE = define_network(net_type="Encoder", config=config)
        self.netS = define_network(net_type="Encoder", config=config)
        self.netG = define_network(net_type="Generator", config=config)

    def forward(self, content_img, style_img):
        content_A = self.netE.forward_test(content_img, "content")
        style_B = self.netS.forward_test(style_img, "style")
        trs_AtoB, trs_AtoB_high, trs_AtoB_low = self.netG(content_A, style_B)
        return trs_AtoB, trs_AtoB_high, trs_AtoB_low


def build_aesfa():
    return AesFAStyleTransfer(_Config()).eval()


def example_input_aesfa():
    content = torch.randn(1, 3, 64, 64)
    style = torch.randn(1, 3, 64, 64)
    return (content, style)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "AesFA (Aesthetic Frequency-Aware style transfer, Octave-Conv AdaConv)",
        "build_aesfa",
        "example_input_aesfa",
        2024,
        "vendored",
    ),
]
