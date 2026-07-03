# SOURCE: vendored from DequanWang/weightnet.pytorch @ master
# https://raw.githubusercontent.com/DequanWang/weightnet.pytorch/master/mmcls/models/backbones/weightnet.py
#
# Wang, Zhang, Shao, Zhu, Sun 2020 "WeightNet: Revisiting the Design Space of Weight
# Networks" (ECCV 2020, arXiv:2007.11823). The official reference implementation lives
# at megvii-model/WeightNet but is written against MegEngine (not an installed base
# lib here); DequanWang/weightnet.pytorch is the real PyTorch port (an mmclassification
# fork) whose `mmcls/models/backbones/weightnet.py` contains the actual
# `conv2d_sample_by_sample`, `WeightNetConv`, `WeightNetConvDW`, `InvertedResidual`, and
# `WeightNet` classes copied verbatim below -- the ShuffleNetV2-style backbone whose
# grouped 1x1 convs are replaced by WeightNet's per-sample weight-generating FC layers,
# exactly matching the megvii-model/WeightNet original `shufflenet_v2.py` (see comment
# in the vendored `InvertedResidual` docstring pointing back to that file).
#
# The only non-architectural change: the source imports `mmcv.cnn` helper *factories*
# (`ConvModule`, `build_conv_layer`, `build_norm_layer`, `build_activation_layer`) and
# `mmcv.runner.load_checkpoint`, none of which are installed base libs here. Every call
# site in this file uses the plain-default configs (`conv_cfg=None` ->
# `nn.Conv2d`, `norm_cfg=dict(type='BN')` -> `nn.BatchNorm2d`,
# `act_cfg=dict(type='ReLU')` -> `nn.ReLU`), so those factories are inlined as their
# exact dispatch target for these configs (mmcv's `ConvModule` with a `None` conv_cfg is
# a plain `Conv2d` + `BatchNorm2d` + `ReLU` in sequence with `conv->norm->act` order;
# `build_norm_layer(dict(type='BN'), c)` returns `('bn', nn.BatchNorm2d(c))`;
# `build_activation_layer(dict(type='ReLU'))` returns `nn.ReLU(inplace=True)`). No
# architectural computation is altered. `init_weights`/checkpoint-loading and the
# `train()`/`_freeze_stages()` eval-mode bookkeeping are dropped as non-architecture.
"""WeightNet: ShuffleNetV2-style backbone whose grouped convolutions are replaced by
WeightNet's grouped-FC weight-generating layers (Wang et al., ECCV 2020)."""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

MENAGERIE_ZOO = "vendored-pytorch"


# --- inlined equivalent of mmcv.cnn.ConvModule(conv_cfg=None, norm_cfg=dict(type='BN'),
# act_cfg=dict(type='ReLU')): plain Conv2d -> BatchNorm2d -> ReLU(inplace=True) ---
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


def conv2d_sample_by_sample(
    x: torch.Tensor,
    weight: torch.Tensor,
    oup: int,
    inp: int,
    ksize: int,
    stride: int,
    padding: int,
    groups: int,
) -> torch.Tensor:
    batch_size = x.shape[0]
    if batch_size == 1:
        out = torch.nn.functional.conv2d(
            x,
            weight=weight.view(oup, inp, ksize, ksize),
            stride=stride,
            padding=padding,
            groups=groups,
        )
    else:
        out = torch.nn.functional.conv2d(
            x.view(1, -1, x.shape[2], x.shape[3]),
            weight.view(batch_size * oup, inp, ksize, ksize),
            stride=stride,
            padding=padding,
            groups=groups * batch_size,
        )
        out = out.view(batch_size, oup, out.shape[2], out.shape[3])
    return out


class WeightNetConv(nn.Module):
    r"""Applies WeightNet to a standard convolution.
    The grouped fc layer directly generates the convolutional kernel,
    this layer has M*inp inputs, G*oup groups and oup*inp*ksize*ksize outputs.
    M/G control the amount of parameters.
    """

    def __init__(self, inp, oup, ksize, stride, M=2, G=2):
        super().__init__()
        inp_gap = max(16, inp // 16)
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride
        self.padding = ksize // 2

        self.wn_fc1 = nn.Conv2d(inp_gap, M * oup, 1, 1, 0, groups=1, bias=True)
        self.wn_fc2 = nn.Conv2d(
            M * oup, oup * inp * ksize * ksize, 1, 1, 0, groups=G * oup, bias=False
        )

    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = torch.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
        return conv2d_sample_by_sample(
            x, x_w, self.oup, self.inp, self.ksize, self.stride, self.padding, 1
        )


class WeightNetConvDW(nn.Module):
    r"""Here we show a grouping manner when we apply WeightNet to a depthwise convolution.
    The grouped fc layer directly generates the convolutional kernel, has fewer parameters
    while achieving comparable results.
    This layer has M/G*inp inputs, inp groups and inp*ksize*ksize outputs.
    """

    def __init__(self, inp, ksize, stride, M=2, G=2):
        super().__init__()
        inp_gap = max(16, inp // 16)
        self.inp = inp
        self.ksize = ksize
        self.stride = stride
        self.padding = ksize // 2

        self.wn_fc1 = nn.Conv2d(inp_gap, M // G * inp, 1, 1, 0, groups=1, bias=True)
        self.wn_fc2 = nn.Conv2d(M // G * inp, inp * ksize * ksize, 1, 1, 0, groups=inp, bias=False)

    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = torch.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
        return conv2d_sample_by_sample(
            x, x_w, self.inp, 1, self.ksize, self.stride, self.padding, self.inp
        )


# https://github.com/megvii-model/WeightNet/blob/master/shufflenet_v2.py
class InvertedResidual(nn.Module):
    """InvertedResidual block for WeightNet (adapted from ShuffleNetV2) backbone."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        with_cp=False,
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.with_cp = with_cp
        self.reduce = nn.Conv2d(
            in_channels,
            max(16, in_channels // 16),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        branch_features = out_channels // 2

        if self.stride > 1:
            self.wnet_proj_1 = WeightNetConvDW(in_channels, 3, self.stride)
            self.norm_proj_1 = nn.BatchNorm2d(in_channels)
            self.wnet_proj_2 = WeightNetConv(in_channels, in_channels, 1, 1)
            self.norm_proj_2 = nn.BatchNorm2d(in_channels)
            self.relu_proj_2 = nn.ReLU(inplace=True)

        self.wnet1 = WeightNetConv(in_channels, branch_features, 1, 1)
        self.norm1 = nn.BatchNorm2d(branch_features)
        self.relu1 = nn.ReLU(inplace=True)

        self.wnet2 = WeightNetConvDW(branch_features, 3, self.stride)
        self.norm2 = nn.BatchNorm2d(branch_features)

        self.wnet3 = WeightNetConv(branch_features, out_channels - in_channels, 1, 1)
        self.norm3 = nn.BatchNorm2d(out_channels - in_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]

    def forward(self, x):
        def _inner_forward(old_x):
            if self.stride == 1:
                x_proj, x = self.channel_shuffle(old_x)
            elif self.stride == 2:
                x_proj, x = old_x, old_x
            x_gap = self.reduce(x.mean(dim=[2, 3], keepdim=True))

            x = self.wnet1(x, x_gap)
            x = self.norm1(x)
            x = self.relu1(x)
            x = self.wnet2(x, x_gap)
            x = self.norm2(x)
            x = self.wnet3(x, x_gap)
            x = self.norm3(x)
            x = self.relu3(x)

            if self.stride == 2:
                x_proj = self.wnet_proj_1(x_proj, x_gap)
                x_proj = self.norm_proj_1(x_proj)
                x_proj = self.wnet_proj_2(x_proj, x_gap)
                x_proj = self.norm_proj_2(x_proj)
                x_proj = self.relu_proj_2(x_proj)
            return torch.cat((x_proj, x), 1)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class WeightNet(nn.Module):
    """WeightNet (adapted from ShuffleNetV2) backbone.

    Args:
        widen_factor (float): Width multiplier - adjusts the number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (3,).
    """

    def __init__(
        self,
        widen_factor=1.0,
        out_indices=(3,),
        with_cp=False,
    ):
        super(WeightNet, self).__init__()
        self.stage_blocks = [4, 8, 4]
        for index in out_indices:
            if index not in range(0, 4):
                raise ValueError(
                    f"the item in out_indices must in range(0, 4). But received {index}"
                )

        self.out_indices = out_indices
        self.with_cp = with_cp

        if widen_factor == 0.5:
            channels = [48, 96, 192, 1024]
        elif widen_factor == 1.0:
            channels = [116, 232, 464, 1024]
        elif widen_factor == 1.5:
            channels = [176, 352, 704, 1024]
        elif widen_factor == 2.0:
            channels = [244, 488, 976, 2048]
        else:
            raise ValueError(
                f"widen_factor must be in [0.5, 1.0, 1.5, 2.0]. But received {widen_factor}"
            )

        self.in_channels = 24
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            layer = self._make_layer(channels[i], num_blocks)
            self.layers.append(layer)

        output_channels = channels[-1]
        self.layers.append(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=output_channels,
                kernel_size=1,
            )
        )

    def _make_layer(self, out_channels, num_blocks):
        """Stack blocks to make a layer."""
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(
                    InvertedResidual(
                        in_channels=self.in_channels,
                        out_channels=out_channels,
                        stride=2,
                        with_cp=self.with_cp,
                    )
                )
                self.in_channels = out_channels
            else:
                layers.append(
                    InvertedResidual(
                        in_channels=self.in_channels // 2,
                        out_channels=out_channels,
                        stride=1,
                        with_cp=self.with_cp,
                    )
                )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)


def build_weightnet():
    model = WeightNet(widen_factor=0.5, out_indices=(3,))
    model.eval()
    return model


def example_input_weightnet():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 64, 64),)


MENAGERIE_ENTRIES = [
    ("WeightNet", "build_weightnet", "example_input_weightnet", 2020, "vendored"),
]
