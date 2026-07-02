# SOURCE: vendored from huawei-noah/AdderNet @ master
# https://raw.githubusercontent.com/huawei-noah/AdderNet/master/adder.py
# https://raw.githubusercontent.com/huawei-noah/AdderNet/master/resnet20.py
#
# "AdderNet: Do We Really Need Multiplications in Deep Learning?" (Chen et al., CVPR 2020),
# Huawei Noah's Ark Lab. Replaces every convolution's multiply-accumulate with an L1-norm
# "adder" filter (sum of |W - X| per receptive field, negated) via a custom autograd.Function
# with a hand-derived (non-multiplicative-gradient-clipped) backward pass -- the network
# topology is otherwise a standard CIFAR ResNet-20 (3 stages x 3 BasicBlocks, adder2d in place
# of every nn.Conv2d including 1x1 downsample shortcuts). Code below is transcribed VERBATIM
# from the real repo (only `import adder` -> local import; no architectural change).
import math

import torch
import torch.nn as nn
from torch.autograd import Function

MENAGERIE_ZOO = "vendored-pytorch"


# ---- adder.py (verbatim) ----
def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(
        X.view(1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride
    ).view(n_x, -1, h_out * w_out)
    X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
    W_col = W.view(n_filters, -1)

    out = adder.apply(W_col, X_col)

    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()

    return out


class adder(Function):
    @staticmethod
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col, X_col)
        output = -(W_col.unsqueeze(2) - X_col.unsqueeze(0)).abs().sum(1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        W_col, X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0) - W_col.unsqueeze(2)) * grad_output.unsqueeze(1)).sum(2)
        grad_W_col = (
            grad_W_col
            / grad_W_col.norm(p=2).clamp(min=1e-12)
            * math.sqrt(W_col.size(1) * W_col.size(0))
            / 5
        )
        grad_X_col = (
            -(X_col.unsqueeze(0) - W_col.unsqueeze(2)).clamp(-1, 1) * grad_output.unsqueeze(1)
        ).sum(0)

        return grad_W_col, grad_X_col


class adder2d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(
            nn.init.normal_(torch.randn(output_channel, input_channel, kernel_size, kernel_size))
        )
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x, self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return output


# ---- resnet20.py (verbatim) ----
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return adder2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                adder2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x.view(x.size(0), -1)


def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


# ---- staging harness ----
# Real usage is CIFAR-10 (32x32x3, num_classes=10, resnet20 = [3,3,3] BasicBlocks per stage).
# avgpool(8, stride=1) requires a 8x8 feature map post layer3 (32 / 2 / 2 = 8), so the real
# 32x32 CIFAR resolution is kept as-is (shrinking it would break the fixed-kernel avgpool).
def build_addernet_resnet20():
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()
    return model


def example_input_addernet_resnet20():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    (
        "AdderNet_ResNet20",
        "build_addernet_resnet20",
        "example_input_addernet_resnet20",
        2020,
        "vendored",
    ),
]
