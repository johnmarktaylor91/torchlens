# SOURCE: vendored from https://github.com/xiaomi-automl/SCARLET-NAS @ master
# (models/basic_ops.py, models/Scarlet_A.py)
#
# SCARLET-NAS ("Bridging the gap Between Stability and Scalability in Neural
# Architecture Search", Chu et al. 2019): a linearly-equivalent-transformation
# evolutionary-supernet search producing a family of EfficientNet-style mobile
# backbones (Scarlet-A/B/C). The classes below are the REAL Scarlet-A model
# code from the official Xiaomi AutoML repository -- MBConv-style
# InvertedResidual blocks with HSwish/HSigmoid activations and a
# squeeze-excite (SqueezeExcite) module, exactly as published. No architecture
# was altered; only the `from models.basic_ops import *` relative import was
# inlined into a single file for menagerie staging.

import math

import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


def stem(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def separable_conv(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_before_pooling(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channel,
        reduction=4,
        squeeze_act=nn.ReLU(inplace=True),
        excite_act=HSigmoid(inplace=True),
    ):
        super(SqueezeExcite, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel // reduction,
            kernel_size=1,
            bias=True,
        )
        self.squeeze_act = squeeze_act
        self.excite_conv = nn.Conv2d(
            in_channels=in_channel // reduction,
            out_channels=in_channel,
            kernel_size=1,
            bias=True,
        )
        self.excite_act = excite_act

    def forward(self, inputs):
        feature_pooling = self.global_pooling(inputs)
        feature_squeeze_conv = self.squeeze_conv(feature_pooling)
        feature_squeeze_act = self.squeeze_act(feature_squeeze_conv)
        feature_excite_conv = self.excite_conv(feature_squeeze_act)
        feature_excite_act = self.excite_act(feature_excite_conv)
        return inputs * feature_excite_act


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, expand_ratio, is_use_se):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.is_use_se = is_use_se
        padding = kernel_size // 2
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = HSwish(inplace=True)
        self.conv2 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.act2 = HSwish(inplace=True)
        if self.is_use_se is True:
            self.mid_se = SqueezeExcite(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        inputs = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.is_use_se is True:
            x = self.mid_se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.use_res_connect:
            return inputs + x
        else:
            return x


class ScarletA(nn.Module):
    def __init__(self, n_class=1000, input_size=224):
        super(ScarletA, self).__init__()
        assert input_size % 32 == 0
        mb_config = [
            # expansion, out_channel, kernel_size, stride, se
            [3, 32, 7, 2, False],
            [6, 32, 5, 1, False],
            [3, 40, 5, 2, False],
            [3, 40, 7, 1, True],
            [6, 40, 3, 1, True],
            [3, 40, 5, 1, True],
            [6, 80, 3, 2, True],
            [3, 80, 3, 1, False],
            [3, 80, 7, 1, True],
            [3, 80, 7, 1, False],
            [3, 96, 5, 1, False],
            [3, 96, 7, 1, True],
            [3, 96, 3, 1, False],
            [3, 96, 7, 1, True],
            [6, 192, 3, 2, True],
            [6, 192, 5, 1, True],
            [3, 192, 3, 1, True],
            [6, 192, 3, 1, True],
            [6, 320, 7, 1, True],
        ]
        input_channel = 16
        last_channel = 1280

        self.last_channel = last_channel
        self.stem = stem(3, 32, 2)
        self.separable_conv = separable_conv(32, 16)
        self.mb_module = list()
        for each_config in mb_config:
            if each_config == "identity":
                self.mb_module.append(Identity())
                continue
            t, c, k, s, e = each_config
            output_channel = c
            self.mb_module.append(
                InvertedResidual(input_channel, output_channel, k, s, expand_ratio=t, is_use_se=e)
            )
            input_channel = output_channel
        self.mb_module = nn.Sequential(*self.mb_module)
        self.conv_before_pooling = conv_before_pooling(input_channel, self.last_channel)
        self.classifier = nn.Linear(self.last_channel, n_class)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.separable_conv(x)
        x = self.mb_module(x)
        x = self.conv_before_pooling(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_N_CLASS = 10
_INPUT_SIZE = 32  # smallest value satisfying `input_size % 32 == 0`


def build_scarletnas():
    import torch

    torch.manual_seed(0)
    model = ScarletA(n_class=_N_CLASS, input_size=_INPUT_SIZE)
    model.eval()
    return model


def example_input_scarletnas():
    import torch

    torch.manual_seed(0)
    return torch.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE)


MENAGERIE_ENTRIES = [
    ("ScarletNAS", "build_scarletnas", "example_input_scarletnas", 2019, MENAGERIE_ZOO),
]
