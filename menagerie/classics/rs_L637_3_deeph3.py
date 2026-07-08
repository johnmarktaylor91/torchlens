# SOURCE: vendored from Graylab/deepH3-distances-orientations @ master
# Files combined: deeph3/resnets/ResNet1D.py, deeph3/resnets/ResNet2D.py,
# deeph3/layers/OuterConcatenation2D.py, deeph3/H3ResNet.py
# Only minimal changes: merged the four source files into one module (relative package
# imports flattened), and reduced default block/plane counts in build_deeph3() for a tiny
# random-init trace (the real defaults are num_blocks1D=3, num_blocks2D=21-25,
# init_planes=32/64). Architecture (ResBlock1D/2D pre-activation residual blocks, the
# OuterConcatenation2D outer-product pairwise expansion, symmetrized dist/omega output
# heads) is untouched.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import stack

MENAGERIE_ZOO = "vendored-pytorch"


# --- deeph3/resnets/ResNet1D.py ---
class ResBlock1D(nn.Module):
    """A basic residual block with 1D CNNs."""

    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, shortcut=None):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            padding=kernel_size // 2,
        )
        self.bn1 = nn.BatchNorm1d(planes)
        self.activation = F.relu
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.stride = stride

        if shortcut is None and stride == 1:
            self.shortcut = lambda x: F.pad(x, pad=(0, 0, 0, planes - x.shape[1], 0, 0))
        elif shortcut is None and stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(self.expansion * planes),
            )
        else:
            self.shortcut = shortcut

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, in_channels, block, num_blocks, init_planes=64, kernel_size=3):
        super(ResNet1D, self).__init__()
        if not (init_planes != 0 and ((init_planes & (init_planes - 1)) == 0)):
            raise ValueError("The initial number of planes must be a power of 2")

        self.activation = F.relu
        self.kernel_size = kernel_size
        self.init_planes = init_planes
        self.in_planes = self.init_planes
        self.num_layers = len(num_blocks)

        self.conv1 = nn.Conv1d(
            in_channels,
            self.in_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.in_planes)

        self.layers = []
        for i in range(0, self.num_layers):
            new_layer = self._make_layer(
                block,
                int(self.init_planes * math.pow(2, i)),
                num_blocks[i],
                stride=1,
                kernel_size=kernel_size,
            )
            self.layers.append(new_layer)
            setattr(self, "layer{}".format(i), new_layer)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, kernel_size=kernel_size))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        return out


# --- deeph3/resnets/ResNet2D.py ---
class ResBlock2D(nn.Module):
    """A basic residual block with 2D CNNs."""

    expansion = 1

    def __init__(
        self, in_planes, planes, kernel_size=(3, 3), dilation=(1, 1), stride=1, shortcut=None
    ):
        super(ResBlock2D, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=(stride, 1),
            padding=(
                ((kernel_size[0] - 1) * dilation[0]) // 2,
                ((kernel_size[0] - 1) * dilation[0]) // 2,
            ),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.activation = F.relu
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=(stride, 1),
            padding=(
                ((kernel_size[0] - 1) * dilation[0]) // 2,
                ((kernel_size[0] - 1) * dilation[0]) // 2,
            ),
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

        if shortcut is None and stride == 1:
            self.shortcut = lambda x: F.pad(x, pad=(0, 0, 0, 0, 0, planes - x.shape[1], 0, 0))
        elif shortcut is None and stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.shortcut = shortcut

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet2D(nn.Module):
    def __init__(
        self, in_channels, block, num_blocks, init_planes=64, kernel_size=3, dilation_cycle=5
    ):
        super(ResNet2D, self).__init__()
        if not (init_planes != 0 and ((init_planes & (init_planes - 1)) == 0)):
            raise ValueError("The initial number of planes must be a power of 2")

        self.activation = F.relu
        self.kernel_size = kernel_size
        self.init_planes = init_planes
        self.in_planes = self.init_planes
        self.num_layers = len(num_blocks)

        self.conv1 = nn.Conv2d(
            in_channels,
            self.in_planes,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(kernel_size // 2, kernel_size // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layers = []
        for i in range(0, self.num_layers):
            new_layer = self._make_layer(
                block,
                int(self.init_planes * math.pow(2, i)),
                num_blocks[i],
                stride=1,
                kernel_size=kernel_size,
                dilation_cycle=dilation_cycle,
            )
            self.layers.append(new_layer)
            setattr(self, "layer{}".format(i), new_layer)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size, dilation_cycle):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            dilation = int(math.pow(2, i % dilation_cycle)) if dilation_cycle > 0 else 1
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride=stride,
                    kernel_size=kernel_size,
                    dilation=(dilation, dilation),
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        return out


# --- deeph3/layers/OuterConcatenation2D.py ---
class OuterConcatenation2D(nn.Module):
    """Transforms sequential data to pairwise data using an outer concatenation."""

    def __init__(self):
        super(OuterConcatenation2D, self).__init__()

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError("Expected three dimensional shape, got shape {}".format(x.shape))

        x = torch.transpose(x, 1, 2)

        vert_expansion = x.clone()
        vert_expansion.unsqueeze_(2)
        vert_expansion = vert_expansion.expand(
            vert_expansion.shape[0],
            vert_expansion.shape[1],
            vert_expansion.shape[1],
            vert_expansion.shape[3],
        )

        x_shape = x.shape
        pair = x
        pair.unsqueeze_(1)
        pair = pair.expand(pair.shape[0], x_shape[1], pair.shape[2], pair.shape[3])
        out_tensor = torch.cat([vert_expansion, pair], dim=3)

        out_tensor = torch.einsum("bijc -> bcij", out_tensor)

        return out_tensor


# --- deeph3/H3ResNet.py ---
class H3ResNet(nn.Module):
    def __init__(
        self,
        in_planes,
        num_out_bins=26,
        num_blocks1D=3,
        num_blocks2D=21,
        dilation_cycle=5,
        dropout_proportion=0.2,
    ):
        super(H3ResNet, self).__init__()
        if isinstance(num_blocks1D, list):
            if len(num_blocks1D) > 1:
                raise NotImplementedError("Multi-layer resnets not supported")
            num_blocks1D = num_blocks1D[0]
        if isinstance(num_blocks2D, int):
            num_blocks2D = [num_blocks2D]

        self._num_out_bins = num_out_bins
        self.resnet1D = ResNet1D(
            in_planes, ResBlock1D, [num_blocks1D], init_planes=32, kernel_size=17
        )
        self.seq2pairwise = OuterConcatenation2D()

        expansion1D = int(math.pow(2, self.resnet1D.num_layers - 1))
        out_planes1D = self.resnet1D.init_planes * expansion1D
        in_planes2D = 2 * out_planes1D

        self.resnet2D = ResNet2D(
            in_planes2D,
            ResBlock2D,
            num_blocks2D,
            init_planes=64,
            kernel_size=5,
            dilation_cycle=dilation_cycle,
        )

        expansion2D = int(math.pow(2, self.resnet2D.num_layers - 1))
        out_planes2D = self.resnet2D.init_planes * expansion2D

        self.out_dropout = nn.Dropout2d(p=dropout_proportion)

        self.out_conv_dist = nn.Conv2d(
            out_planes2D,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2,
        )
        self.out_conv_omega = nn.Conv2d(
            out_planes2D,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2,
        )
        self.out_conv_theta = nn.Conv2d(
            out_planes2D,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2,
        )
        self.out_conv_phi = nn.Conv2d(
            out_planes2D,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2,
        )

    def forward(self, x):
        out = self.resnet1D(x)
        out = self.seq2pairwise(out)
        out = self.resnet2D(out)
        out = self.out_dropout(out)

        out_dist = self.out_conv_dist(out)
        out_omega = self.out_conv_omega(out)
        out_theta = self.out_conv_theta(out)
        out_phi = self.out_conv_phi(out)

        out_dist = out_dist + out_dist.transpose(2, 3)
        out_omega = out_omega + out_omega.transpose(2, 3)

        return stack([out_dist, out_omega, out_theta, out_phi]).transpose(0, 1)


def build_deeph3():
    # Real defaults (num_blocks1D=3, num_blocks2D~21-25, init_planes=32/64) shrunk to
    # num_blocks1D=1, num_blocks2D=[2] via dilation_cycle for a tiny random-init trace.
    return H3ResNet(in_planes=21, num_out_bins=8, num_blocks1D=1, num_blocks2D=2, dilation_cycle=2)


def example_input_deeph3():
    torch.manual_seed(0)
    # [batch, in_planes=21 (one-hot amino acid channels), length] matching
    # deeph3/util.py::get_logits_from_model's seq.unsqueeze(0).transpose(1, 2) input prep.
    batch, in_planes, length = 1, 21, 16
    x = torch.zeros(batch, in_planes, length)
    idx = torch.randint(0, in_planes, (length,))
    x[0, idx, torch.arange(length)] = 1.0
    return (x,)


MENAGERIE_ENTRIES = [
    ("DeepH3", build_deeph3, example_input_deeph3, 2020, MENAGERIE_ZOO),
]
