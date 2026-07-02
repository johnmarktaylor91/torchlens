# SOURCE: vendored from bahjat-kawar/snips_torch @ main
# https://raw.githubusercontent.com/bahjat-kawar/snips_torch/main/models/ncsnv2.py
# https://raw.githubusercontent.com/bahjat-kawar/snips_torch/main/models/layers.py
# https://raw.githubusercontent.com/bahjat-kawar/snips_torch/main/models/normalization.py
#
# SNIPS (Kawar, Vaksman, Elad, NeurIPS 2021, "SNIPS: Solving Noisy Inverse Problems
# Stochastically") -- official PyTorch implementation. SNIPS reuses the NCSNv2 score
# network (Song & Ermon, NeurIPS 2020, "Improved Techniques for Training Score-Based
# Generative Models") verbatim as its score estimator inside an annealed Langevin
# posterior-sampling scheme for noisy linear inverse problems; the repo ships NCSNv2's
# real architecture unmodified in `models/ncsnv2.py` + `models/layers.py` +
# `models/normalization.py`. `NCSNv2` is transcribed verbatim: begin_conv -> 4
# unconditional dilated-residual stages (res1..res4, InstanceNorm++ by default) ->
# RefineNet-style multi-scale-fusion decoder (refine1..refine4, CRP/RCU/MSF blocks) ->
# normalizer/act/end_conv -> divide by per-noise-level sigma. Only the `config` object
# construction is adapted here (a tiny SimpleNamespace standing in for the repo's YAML
# `configs/celeba.yml`, using the same field names/defaults) to keep the module
# self-contained; no layer, block, or forward-pass logic was changed.

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# ---------------------------------------------------------------------------
# models/normalization.py (verbatim)
# ---------------------------------------------------------------------------


def get_normalization(config, conditional=True):
    norm = config.model.normalization
    if conditional:
        if norm == "NoneNorm":
            return ConditionalNoneNorm2d
        elif norm == "InstanceNorm++":
            return ConditionalInstanceNorm2dPlus
        elif norm == "InstanceNorm":
            return ConditionalInstanceNorm2d
        elif norm == "BatchNorm":
            return ConditionalBatchNorm2d
        elif norm == "VarianceNorm":
            return ConditionalVarianceNorm2d
        else:
            raise NotImplementedError("{} does not exist!".format(norm))
    else:
        if norm == "BatchNorm":
            return nn.BatchNorm2d
        elif norm == "InstanceNorm":
            return nn.InstanceNorm2d
        elif norm == "InstanceNorm++":
            return InstanceNorm2dPlus
        elif norm == "VarianceNorm":
            return VarianceNorm2d
        elif norm == "NoneNorm":
            return NoneNorm2d
        elif norm is None:
            return None
        else:
            raise NotImplementedError("{} does not exist!".format(norm))


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out


class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        h = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalVarianceNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.embed = nn.Embedding(num_classes, num_features)
        self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-5)

        gamma = self.embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class VarianceNorm2d(nn.Module):
    def __init__(self, num_features, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)

    def forward(self, x):
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-5)

        out = self.alpha.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalNoneNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()
            self.embed.weight.data[:, num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * x
        return out


class NoneNorm2d(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()

    def forward(self, x):
        return x


class InstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, : 2 * num_features].normal_(1, 0.02)
            self.embed.weight.data[:, 2 * num_features :].zero_()
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(
                -1, self.num_features, 1, 1
            )
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


# ---------------------------------------------------------------------------
# models/layers.py (verbatim)
# ---------------------------------------------------------------------------


def get_act(config):
    if config.model.nonlinearity.lower() == "elu":
        return nn.ELU()
    elif config.model.nonlinearity.lower() == "relu":
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == "lrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == "swish":

        def swish(x):
            return x * torch.sigmoid(x)

        return swish
    else:
        raise NotImplementedError("activation function does not exist!")


def spectral_norm(layer, n_iters=1):
    return torch.nn.utils.spectral_norm(layer, n_power_iterations=n_iters)


def conv1x1(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "1x1 convolution"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def conv3x3(in_planes, out_planes, stride=1, bias=True, spec_norm=False):
    "3x3 convolution with padding"
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
    if spec_norm:
        conv = spectral_norm(conv)

    return conv


def stride_conv3x3(in_planes, out_planes, kernel_size, bias=True, spec_norm=False):
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=2,
        padding=kernel_size // 2,
        bias=bias,
    )
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True, spec_norm=False):
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias
    )
    if spec_norm:
        conv = spectral_norm(conv)

    return conv


class CRPBlock(nn.Module):
    def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True, spec_norm=False):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(
                conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm)
            )
        self.n_stages = n_stages
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        else:
            self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)

        self.act = act

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.maxpool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class CondCRPBlock(nn.Module):
    def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU(), spec_norm=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.normalizer = normalizer
        for i in range(n_stages):
            self.norms.append(normalizer(features, num_classes, bias=True))
            self.convs.append(
                conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm)
            )

        self.n_stages = n_stages
        self.maxpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.maxpool(path)
            path = self.convs[i](path)

            x = path + x
        return x


class RCUBlock(nn.Module):
    def __init__(self, features, n_blocks, n_stages, act=nn.ReLU(), spec_norm=False):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(
                    self,
                    "{}_{}_conv".format(i + 1, j + 1),
                    conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm),
                )

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, "{}_{}_conv".format(i + 1, j + 1))(x)

            x += residual
        return x


class CondRCUBlock(nn.Module):
    def __init__(
        self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU(), spec_norm=False
    ):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(
                    self,
                    "{}_{}_norm".format(i + 1, j + 1),
                    normalizer(features, num_classes, bias=True),
                )
                setattr(
                    self,
                    "{}_{}_conv".format(i + 1, j + 1),
                    conv3x3(features, features, stride=1, bias=False, spec_norm=spec_norm),
                )

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act
        self.normalizer = normalizer

    def forward(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, "{}_{}_norm".format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, "{}_{}_conv".format(i + 1, j + 1))(x)

            x += residual
        return x


class MSFBlock(nn.Module):
    def __init__(self, in_planes, features, spec_norm=False):
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        self.convs = nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(
                conv3x3(in_planes[i], features, stride=1, bias=True, spec_norm=spec_norm)
            )

    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = F.interpolate(h, size=shape, mode="bilinear", align_corners=True)
            sums += h
        return sums


class CondMSFBlock(nn.Module):
    def __init__(self, in_planes, features, num_classes, normalizer, spec_norm=False):
        """
        :param in_planes: tuples of input planes
        """
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.features = features
        self.normalizer = normalizer

        for i in range(len(in_planes)):
            self.convs.append(
                conv3x3(in_planes[i], features, stride=1, bias=True, spec_norm=spec_norm)
            )
            self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = F.interpolate(h, size=shape, mode="bilinear", align_corners=True)
            sums += h
        return sums


class RefineBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        features,
        act=nn.ReLU(),
        start=False,
        end=False,
        maxpool=True,
        spec_norm=False,
    ):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(RCUBlock(in_planes[i], 2, 2, act, spec_norm=spec_norm))

        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act, spec_norm=spec_norm)

        if not start:
            self.msf = MSFBlock(in_planes, features, spec_norm=spec_norm)

        self.crp = CRPBlock(features, 2, act, maxpool=maxpool, spec_norm=spec_norm)

    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h


class CondRefineBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        features,
        num_classes,
        normalizer,
        act=nn.ReLU(),
        start=False,
        end=False,
        spec_norm=False,
    ):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act, spec_norm=spec_norm)
            )

        self.output_convs = CondRCUBlock(
            features, 3 if end else 1, 2, num_classes, normalizer, act, spec_norm=spec_norm
        )

        if not start:
            self.msf = CondMSFBlock(
                in_planes, features, num_classes, normalizer, spec_norm=spec_norm
            )

        self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act, spec_norm=spec_norm)

    def forward(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h


class ConvMeanPool(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size=3,
        biases=True,
        adjust_padding=False,
        spec_norm=False,
    ):
        super().__init__()
        if not adjust_padding:
            conv = nn.Conv2d(
                input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases
            )
            if spec_norm:
                conv = spectral_norm(conv)
            self.conv = conv
        else:
            conv = nn.Conv2d(
                input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases
            )
            if spec_norm:
                conv = spectral_norm(conv)

            self.conv = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)), conv)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = (
            sum(
                [
                    output[:, :, ::2, ::2],
                    output[:, :, 1::2, ::2],
                    output[:, :, ::2, 1::2],
                    output[:, :, 1::2, 1::2],
                ]
            )
            / 4.0
        )
        return output


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, spec_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases
        )
        if spec_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, inputs):
        output = inputs
        output = (
            sum(
                [
                    output[:, :, ::2, ::2],
                    output[:, :, 1::2, ::2],
                    output[:, :, ::2, 1::2],
                    output[:, :, 1::2, 1::2],
                ]
            )
            / 4.0
        )
        return self.conv(output)


class UpsampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, spec_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases
        )
        if spec_norm:
            self.conv = spectral_norm(self.conv)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)


class ConditionalResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_classes,
        resample=None,
        act=nn.ELU(),
        normalization=ConditionalBatchNorm2d,
        adjust_padding=False,
        dilation=None,
        spec_norm=False,
    ):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == "down":
            if dilation is not None:
                self.conv1 = dilated_conv3x3(
                    input_dim, input_dim, dilation=dilation, spec_norm=spec_norm
                )
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = dilated_conv3x3(
                    input_dim, output_dim, dilation=dilation, spec_norm=spec_norm
                )
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(
                    input_dim, output_dim, 3, adjust_padding=adjust_padding, spec_norm=spec_norm
                )
                conv_shortcut = partial(
                    ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding, spec_norm=spec_norm
                )

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
                self.conv1 = dilated_conv3x3(
                    input_dim, output_dim, dilation=dilation, spec_norm=spec_norm
                )
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = dilated_conv3x3(
                    output_dim, output_dim, dilation=dilation, spec_norm=spec_norm
                )
            else:
                conv_shortcut = nn.Conv2d
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise Exception("invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, num_classes)

    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        resample=None,
        act=nn.ELU(),
        normalization=nn.BatchNorm2d,
        adjust_padding=False,
        dilation=None,
        spec_norm=False,
    ):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.normalization = normalization
        if resample == "down":
            if dilation is not None:
                self.conv1 = dilated_conv3x3(
                    input_dim, input_dim, dilation=dilation, spec_norm=spec_norm
                )
                self.normalize2 = normalization(input_dim)
                self.conv2 = dilated_conv3x3(
                    input_dim, output_dim, dilation=dilation, spec_norm=spec_norm
                )
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
            else:
                self.conv1 = conv3x3(input_dim, input_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(input_dim)
                self.conv2 = ConvMeanPool(
                    input_dim, output_dim, 3, adjust_padding=adjust_padding, spec_norm=spec_norm
                )
                conv_shortcut = partial(
                    ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding, spec_norm=spec_norm
                )

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation, spec_norm=spec_norm)
                self.conv1 = dilated_conv3x3(
                    input_dim, output_dim, dilation=dilation, spec_norm=spec_norm
                )
                self.normalize2 = normalization(output_dim)
                self.conv2 = dilated_conv3x3(
                    output_dim, output_dim, dilation=dilation, spec_norm=spec_norm
                )
            else:
                conv_shortcut = partial(conv1x1, spec_norm=spec_norm)
                self.conv1 = conv3x3(input_dim, output_dim, spec_norm=spec_norm)
                self.normalize2 = normalization(output_dim)
                self.conv2 = conv3x3(output_dim, output_dim, spec_norm=spec_norm)
        else:
            raise Exception("invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


# ---------------------------------------------------------------------------
# models/__init__.py::get_sigmas (verbatim, minus the config.device .to() call
# which is inlined into NCSNv2.__init__ via register_buffer as in the original)
# ---------------------------------------------------------------------------


def get_sigmas(config):
    import numpy as np

    if config.model.sigma_dist == "geometric":
        sigmas = torch.tensor(
            np.exp(
                np.linspace(
                    np.log(config.model.sigma_begin),
                    np.log(config.model.sigma_end),
                    config.model.num_classes,
                )
            )
        ).float()
    elif config.model.sigma_dist == "uniform":
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float()
    else:
        raise NotImplementedError("sigma distribution not supported")

    return sigmas


# ---------------------------------------------------------------------------
# models/ncsnv2.py (verbatim)
# ---------------------------------------------------------------------------


class NCSNv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = num_classes = config.model.num_classes  # noqa: F841

        self.act = act = get_act(config)
        self.register_buffer("sigmas", get_sigmas(config))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)

        self.normalizer = self.norm(ngf, self.num_classes)
        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList(
            [
                ResidualBlock(self.ngf, self.ngf, resample=None, act=act, normalization=self.norm),
                ResidualBlock(self.ngf, self.ngf, resample=None, act=act, normalization=self.norm),
            ]
        )

        self.res2 = nn.ModuleList(
            [
                ResidualBlock(
                    self.ngf, 2 * self.ngf, resample="down", act=act, normalization=self.norm
                ),
                ResidualBlock(
                    2 * self.ngf, 2 * self.ngf, resample=None, act=act, normalization=self.norm
                ),
            ]
        )

        self.res3 = nn.ModuleList(
            [
                ResidualBlock(
                    2 * self.ngf,
                    2 * self.ngf,
                    resample="down",
                    act=act,
                    normalization=self.norm,
                    dilation=2,
                ),
                ResidualBlock(
                    2 * self.ngf,
                    2 * self.ngf,
                    resample=None,
                    act=act,
                    normalization=self.norm,
                    dilation=2,
                ),
            ]
        )

        if config.data.image_size == 28:
            self.res4 = nn.ModuleList(
                [
                    ResidualBlock(
                        2 * self.ngf,
                        2 * self.ngf,
                        resample="down",
                        act=act,
                        normalization=self.norm,
                        adjust_padding=True,
                        dilation=4,
                    ),
                    ResidualBlock(
                        2 * self.ngf,
                        2 * self.ngf,
                        resample=None,
                        act=act,
                        normalization=self.norm,
                        dilation=4,
                    ),
                ]
            )
        else:
            self.res4 = nn.ModuleList(
                [
                    ResidualBlock(
                        2 * self.ngf,
                        2 * self.ngf,
                        resample="down",
                        act=act,
                        normalization=self.norm,
                        adjust_padding=False,
                        dilation=4,
                    ),
                    ResidualBlock(
                        2 * self.ngf,
                        2 * self.ngf,
                        resample=None,
                        act=act,
                        normalization=self.norm,
                        dilation=4,
                    ),
                ]
            )

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.0
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output


# ---------------------------------------------------------------------------
# Staging entry points
# ---------------------------------------------------------------------------


def _tiny_config():
    """Standalone stand-in for the repo's `configs/celeba.yml`, using the same
    field names/section layout, but with a small image size + narrow ngf/num_classes
    so a random-init trace is cheap. No architectural fields changed."""
    config = types.SimpleNamespace()
    config.data = types.SimpleNamespace(
        image_size=32,
        channels=3,
        logit_transform=False,
        rescaled=False,
    )
    config.model = types.SimpleNamespace(
        sigma_begin=90,
        num_classes=8,
        sigma_dist="geometric",
        sigma_end=0.01,
        normalization="InstanceNorm++",
        nonlinearity="elu",
        ngf=8,
    )
    return config


def build_snips():
    torch.manual_seed(0)
    model = NCSNv2(_tiny_config())
    model.eval()
    return model


def example_input_snips():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32)
    y = torch.randint(0, 8, (1,))
    return (x, y)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SNIPS", "build_snips", "example_input_snips", 2021, MENAGERIE_ZOO),
]
