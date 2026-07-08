# SOURCE: vendored from zhaoyuzhi/Semantic-Colorization-GAN @ main
# (train/network.py, train/network_module.py)
"""SCGAN: Saliency Map-guided Colorization with Generative Adversarial Network.

Zhao, Yuzhi, et al. "Saliency-guided Image Colorization with Deep Nets and Conditional
Adversarial Networks." IEEE TCSVT 2020. Official PyTorch repo confirmed.

The generator is a U-Net colorization backbone augmented with (1) a VGG-style
GlobalFeatureExtractor that pools a global scene-context vector into the bottleneck
via concatenation, and (2) an AttentionPredictionNet decoder head that jointly predicts
a spatial saliency map from three intermediate decoder feature maps. This file vendors
the real generator (SCGAN class) + its Conv2dLayer/TransposeConv2dLayer/SpectralNorm/
LayerNorm building blocks from train/network_module.py verbatim; only cosmetic changes
(inlining the `from network_module import *` into a single file, and a lightweight
argparse-free `Opt` config object replacing the CLI opt namespace) were made so the
module is self-contained. No architecture was altered.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# train/network_module.py (verbatim, only the classes SCGAN's generator uses)
# ---------------------------------------------------------------------------
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
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


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
        sn=True,
    ):
        super(Conv2dLayer, self).__init__()
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

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

        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding=0,
                    dilation=dilation,
                    bias=False,
                )
            )
        else:
            self.conv2d = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                dilation=dilation,
                bias=False,
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
        sn=True,
        scale_factor=2,
    ):
        super(TransposeConv2dLayer, self).__init__()
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


# ---------------------------------------------------------------------------
# train/network.py (verbatim: weights_init, GlobalFeatureExtractor,
# AttentionPredictionNet, SCGAN generator)
# ---------------------------------------------------------------------------
def weights_init(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, opt):
        super(GlobalFeatureExtractor, self).__init__()
        self.conv1_1 = Conv2dLayer(
            opt.in_channels, 64, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm="none"
        )
        self.conv1_2 = Conv2dLayer(
            64, 64, 3, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv2_1 = Conv2dLayer(
            64, 128, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv2_2 = Conv2dLayer(
            128, 128, 3, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv3_1 = Conv2dLayer(
            128, 256, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv3_2 = Conv2dLayer(
            256, 256, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv3_3 = Conv2dLayer(
            256, 256, 3, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv4_1 = Conv2dLayer(
            256, 512, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv4_2 = Conv2dLayer(
            512, 512, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv4_3 = Conv2dLayer(
            512, 512, 3, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv5_1 = Conv2dLayer(
            512, 512, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv5_2 = Conv2dLayer(
            512, 512, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.conv5_3 = Conv2dLayer(
            512, 512, 3, 1, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )
        self.pool5 = Conv2dLayer(
            512, 512, 3, 2, 1, pad_type=opt.pad, activation=opt.activ_g, norm=opt.norm_g
        )

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        return x


class AttentionPredictionNet(nn.Module):
    def __init__(self, opt):
        super(AttentionPredictionNet, self).__init__()
        self.conv11 = TransposeConv2dLayer(
            opt.start_channels * 8,
            opt.latent_channels,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.conv12 = TransposeConv2dLayer(
            opt.latent_channels,
            opt.latent_channels,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.conv2 = TransposeConv2dLayer(
            opt.start_channels * 4,
            opt.latent_channels,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.conv3 = Conv2dLayer(
            opt.start_channels * 2,
            opt.latent_channels,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.conv4 = TransposeConv2dLayer(
            opt.latent_channels * 3,
            opt.latent_channels,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.final = Conv2dLayer(
            opt.latent_channels, 1, 3, 1, 1, pad_type=opt.pad, activation="sigmoid", norm="none"
        )

    def forward(self, x1, x2, x3):
        x1 = self.conv11(x1)
        x1 = self.conv12(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x = torch.cat((x1, x2, x3), 1)
        x = self.conv4(x)
        x = self.final(x)
        return x


class SCGAN(nn.Module):
    """SCGAN's generator."""

    def __init__(self, opt):
        super(SCGAN, self).__init__()
        self.global_feature_network = GlobalFeatureExtractor(opt)
        self.attention_prediction_network = AttentionPredictionNet(opt)
        self.down1 = Conv2dLayer(
            opt.in_channels,
            opt.start_channels,
            7,
            1,
            3,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm="none",
        )
        self.down2 = Conv2dLayer(
            opt.start_channels,
            opt.start_channels * 2,
            3,
            2,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.down3 = Conv2dLayer(
            opt.start_channels * 2,
            opt.start_channels * 4,
            3,
            2,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.down4 = Conv2dLayer(
            opt.start_channels * 4,
            opt.start_channels * 8,
            3,
            2,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.down5 = Conv2dLayer(
            opt.start_channels * 8,
            opt.start_channels * 8,
            3,
            2,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.down6 = Conv2dLayer(
            opt.start_channels * 8,
            opt.start_channels * 8,
            3,
            2,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.down7 = Conv2dLayer(
            opt.start_channels * 8 + 512,
            opt.start_channels * 8,
            3,
            2,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.down8 = Conv2dLayer(
            opt.start_channels * 8,
            opt.start_channels * 8,
            3,
            2,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.down9 = Conv2dLayer(
            opt.start_channels * 8,
            opt.start_channels * 8,
            3,
            2,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm="none",
        )
        self.up1 = TransposeConv2dLayer(
            opt.start_channels * 8,
            opt.start_channels * 8,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.up2 = TransposeConv2dLayer(
            opt.start_channels * 16,
            opt.start_channels * 8,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.up3 = TransposeConv2dLayer(
            opt.start_channels * 16,
            opt.start_channels * 8,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.up4 = TransposeConv2dLayer(
            opt.start_channels * 16,
            opt.start_channels * 8,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.up5 = TransposeConv2dLayer(
            opt.start_channels * 16,
            opt.start_channels * 8,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.up6 = TransposeConv2dLayer(
            opt.start_channels * 16,
            opt.start_channels * 4,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.up7 = TransposeConv2dLayer(
            opt.start_channels * 8,
            opt.start_channels * 2,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.up8 = TransposeConv2dLayer(
            opt.start_channels * 4,
            opt.start_channels,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation=opt.activ_g,
            norm=opt.norm_g,
        )
        self.up9 = Conv2dLayer(
            opt.start_channels * 2,
            opt.out_channels,
            3,
            1,
            1,
            pad_type=opt.pad,
            activation="tanh",
            norm="none",
        )

    def forward(self, x):
        global_feature = self.global_feature_network(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)
        down6_with_gf = torch.cat((down6, global_feature), 1)
        down7 = self.down7(down6_with_gf)
        down8 = self.down8(down7)
        down9 = self.down9(down8)
        up1 = self.up1(down9)
        up1 = torch.cat((down8, up1), 1)
        up2 = self.up2(up1)
        up2 = torch.cat((down7, up2), 1)
        up3 = self.up3(up2)
        up3 = torch.cat((down6, up3), 1)
        up4 = self.up4(up3)
        up4 = torch.cat((down5, up4), 1)
        up5 = self.up5(up4)
        up5_ = torch.cat((down4, up5), 1)
        up6 = self.up6(up5_)
        up6_ = torch.cat((down3, up6), 1)
        up7 = self.up7(up6_)
        up7_ = torch.cat((down2, up7), 1)
        up8 = self.up8(up7_)
        up8 = torch.cat((down1, up8), 1)
        up9 = self.up9(up8)
        sal = self.attention_prediction_network(up5, up6, up7)
        return up9, sal


# ---------------------------------------------------------------------------
# Menagerie build/example helpers
# ---------------------------------------------------------------------------
class _SCGANOpt:
    """Minimal stand-in for the repo's argparse Namespace (train/network.py __main__)."""

    def __init__(self):
        self.in_channels = 1
        self.out_channels = 3
        self.start_channels = 8
        self.latent_channels = 8
        self.pad = "reflect"
        self.activ_g = "lrelu"
        self.norm_g = "bn"


def build_scgan():
    model = SCGAN(_SCGANOpt())
    model.eval()
    return model


def example_input_scgan():
    # 256x256 (repo default) needed: down1..down9 is 8 stride-2 downsamples, so
    # smaller inputs collapse the bottleneck to a 1x1 spatial map that BatchNorm2d
    # rejects in training mode.
    return torch.randn(1, 1, 256, 256)


MENAGERIE_ENTRIES = [
    ("SCGAN-Colorization", build_scgan, example_input_scgan, 2020, MENAGERIE_ZOO),
]
