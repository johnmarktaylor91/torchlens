# SOURCE: vendored from https://github.com/zzzxxxttt/pytorch_DoReFaNet @ a12b6171
# (utils/quant_dorefa.py: uniform_quantize, weight_quantize_fn, activation_quantize_fn,
#  conv2d_Q_fn, linear_Q_fn; nets/cifar_resnet.py: PreActBlock_conv_Q, PreActResNet,
#  resnet20/resnet56)
#
# DoReFa-Net (Zhou et al. 2016, "DoReFa-Net: Training Low Bitwidth Convolutional Neural
# Networks with Low Bitwidth Gradients", arXiv:1606.06160). The official implementation is
# TensorFlow (tensorpack/examples/DoReFa-Net); zzzxxxttt/pytorch_DoReFaNet is the
# well-known community PyTorch port (cross-referenced widely as the canonical PyTorch
# DoReFa-Net reimplementation). Architecture: a straight-through-estimator uniform
# quantization autograd Function (`uniform_quantize`) backs both a weight quantizer
# (tanh-normalize to [-1,1], STE-round to k bits, rescale) and an activation quantizer
# (clamp to [0,1], STE-round to k bits) -- these are DoReFa-Net's defining architectural
# contribution (quantized weights AND activations, not just weights as in earlier binary
# nets). `conv2d_Q_fn`/`linear_Q_fn` wrap `nn.Conv2d`/`nn.Linear` to quantize their weight
# before the real conv/linear op each forward call. The backbone is a CIFAR-style
# pre-activation ResNet (`PreActResNet`, He et al. pre-act BasicBlock) built entirely from
# these quantized conv layers plus a quantized activation after each BN+ReLU. Vendored
# verbatim; only the `from utils.quant_dorefa import *` relative import is inlined into
# this single file (both source files concatenated, nothing else changed).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- utils/quant_dorefa.py (vendored verbatim) ----
def uniform_quantize(k):
    class qfn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2**k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input

    return qfn().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 8 or w_bit == 32
        self.w_bit = w_bit
        self.uniform_q = uniform_quantize(k=w_bit)

    def forward(self, x):
        if self.w_bit == 32:
            weight_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            weight_q = self.uniform_q(x / E) * E
        else:
            weight = torch.tanh(x)
            max_w = torch.max(torch.abs(weight)).detach()
            weight = weight / 2 / max_w + 0.5
            weight_q = max_w * (2 * self.uniform_q(weight) - 1)
        return weight_q


class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))
        return activation_q


def conv2d_Q_fn(w_bit):
    class Conv2d_Q(nn.Conv2d):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
        ):
            super(Conv2d_Q, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
            )
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            return F.conv2d(
                input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups
            )

    return Conv2d_Q


def linear_Q_fn(w_bit):
    class Linear_Q(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear_Q, self).__init__(in_features, out_features, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input):
            weight_q = self.quantize_fn(self.weight)
            return F.linear(input, weight_q, self.bias)

    return Linear_Q


# ---- nets/cifar_resnet.py (vendored verbatim) ----
class PreActBlock_conv_Q(nn.Module):
    """Pre-activation version of the BasicBlock."""

    def __init__(self, wbit, abit, in_planes, out_planes, stride=1):
        super(PreActBlock_conv_Q, self).__init__()
        Conv2d = conv2d_Q_fn(w_bit=wbit)
        self.act_q = activation_quantize_fn(a_bit=abit)

        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv0 = Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.skip_conv = None
        if stride != 1:
            self.skip_conv = Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
            )
            self.skip_bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.act_q(F.relu(self.bn0(x)))

        if self.skip_conv is not None:
            shortcut = self.skip_conv(out)
            shortcut = self.skip_bn(shortcut)
        else:
            shortcut = x

        out = self.conv0(out)
        out = self.act_q(F.relu(self.bn1(out)))
        out = self.conv1(out)
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_units, wbit, abit, num_classes):
        super(PreActResNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.layers = nn.ModuleList()
        in_planes = 16
        strides = (
            [1] * (num_units[0]) + [2] + [1] * (num_units[1] - 1) + [2] + [1] * (num_units[2] - 1)
        )
        channels = [16] * num_units[0] + [32] * num_units[1] + [64] * num_units[2]
        for stride, channel in zip(strides, channels):
            self.layers.append(block(wbit, abit, in_planes, channel, stride))
            in_planes = channel

        self.bn = nn.BatchNorm2d(64)
        self.logit = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.conv0(x)
        for layer in self.layers:
            out = layer(out)
        out = self.bn(out)
        out = out.mean(dim=2).mean(dim=2)
        out = self.logit(out)
        return out


def resnet20(wbits, abits, num_classes=10):
    return PreActResNet(PreActBlock_conv_Q, [3, 3, 3], wbits, abits, num_classes=num_classes)


def resnet56(wbits, abits, num_classes=10):
    return PreActResNet(PreActBlock_conv_Q, [9, 9, 9], wbits, abits, num_classes=num_classes)


# ---- end vendored source ----


def build_dorefanet_resnet20():
    # train configs in the repo default to 1-bit weights / 2-bit activations
    # (cifar_train_eval.py argparse defaults: --wbits 1 --abits 2).
    torch.manual_seed(0)
    return resnet20(wbits=1, abits=2, num_classes=10)


def example_input_dorefanet_resnet20():
    torch.manual_seed(0)
    return torch.randn(2, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "DoReFa-Net (ResNet-20, W1A2)",
        "build_dorefanet_resnet20",
        "example_input_dorefanet_resnet20",
        2016,
        MENAGERIE_ZOO,
    ),
]
