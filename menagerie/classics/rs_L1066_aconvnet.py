# SOURCE: vendored from https://github.com/jangsoopark/AConvNet-pytorch @ main
# (src/model/network.py + src/model/_blocks.py)
#
# A-ConvNets: All-Convolutional Networks for SAR Automatic Target Recognition
# (Chen, Wang, Zhu & Sun, IEEE GRSL 2016). The classes below are the REAL
# PyTorch reimplementation used in this widely-cited MSTAR-ATR reference repo
# (the original paper's network was Caffe; this repo is the standard PyTorch
# port the community treats as canonical). All-convolutional (no FC layers):
# 5 conv blocks with 'valid' padding, ReLU, interleaved max-pooling, and a
# dropout layer before the final 3x3 classifier conv, ending in Flatten.
# No architecture was altered; only the module-relative "from . import
# _blocks" import was flattened into this single file.

import collections

import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"

_activations = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "leaky_relu": nn.LeakyReLU,
}


class BaseBlock(nn.Module):
    def __init__(self):
        super(BaseBlock, self).__init__()
        self._layer: nn.Sequential

    def forward(self, x):
        return self._layer(x)


class DenseBlock(BaseBlock):
    def __init__(self, shape, **params):
        super(DenseBlock, self).__init__()
        in_dims, out_dims = shape
        _seq = collections.OrderedDict(
            [
                ("dense", nn.Linear(in_dims, out_dims)),
            ]
        )
        _act_name = params.get("activation")
        if _act_name:
            _seq.update({_act_name: _activations[_act_name](inplace=True)})

        self._layer = nn.Sequential(_seq)

        w_init = params.get("w_init", None)
        idx = list(dict(self._layer.named_children()).keys()).index("dense")
        if w_init:
            w_init(self._layer[idx].weight)
        b_init = params.get("b_init", None)
        if b_init:
            b_init(self._layer[idx].bias)


class Conv2DBlock(BaseBlock):
    def __init__(self, shape, stride, padding="same", **params):
        super(Conv2DBlock, self).__init__()

        h, w, in_channels, out_channels = shape
        _seq = collections.OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(h, w),
                        stride=stride,
                        padding=padding,
                    ),
                )
            ]
        )

        _bn = params.get("batch_norm")
        if _bn:
            _seq.update({"bn": nn.BatchNorm2d(out_channels)})

        _act_name = params.get("activation")
        if _act_name:
            _seq.update({_act_name: _activations[_act_name](inplace=True)})

        _max_pool = params.get("max_pool")
        if _max_pool:
            _kernel_size = params.get("max_pool_size", 2)
            _stride = params.get("max_pool_stride", _kernel_size)
            _seq.update({"max_pool": nn.MaxPool2d(kernel_size=_kernel_size, stride=_stride)})

        self._layer = nn.Sequential(_seq)

        w_init = params.get("w_init", None)
        idx = list(dict(self._layer.named_children()).keys()).index("conv")
        if w_init:
            w_init(self._layer[idx].weight)
        b_init = params.get("b_init", None)
        if b_init:
            b_init(self._layer[idx].bias)


class Network(nn.Module):
    def __init__(self, **params):
        super(Network, self).__init__()
        self.dropout_rate = params.get("dropout_rate", 0.5)
        self.classes = params.get("classes", 10)
        self.channels = params.get("channels", 1)

        _w_init = params.get("w_init", lambda x: nn.init.kaiming_normal_(x, nonlinearity="relu"))
        _b_init = params.get("b_init", lambda x: nn.init.constant_(x, 0.1))

        self._layer = nn.Sequential(
            Conv2DBlock(
                shape=[5, 5, self.channels, 16],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=_w_init,
                b_init=_b_init,
            ),
            Conv2DBlock(
                shape=[5, 5, 16, 32],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=_w_init,
                b_init=_b_init,
            ),
            Conv2DBlock(
                shape=[6, 6, 32, 64],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=_w_init,
                b_init=_b_init,
            ),
            Conv2DBlock(
                shape=[5, 5, 64, 128],
                stride=1,
                padding="valid",
                activation="relu",
                w_init=_w_init,
                b_init=_b_init,
            ),
            nn.Dropout(p=self.dropout_rate),
            Conv2DBlock(
                shape=[3, 3, 128, self.classes],
                stride=1,
                padding="valid",
                w_init=_w_init,
                b_init=nn.init.zeros_,
            ),
            nn.Flatten(),
        )

    def forward(self, x):
        return self._layer(x)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_MSTAR_CHIP_SIZE = 88  # standard MSTAR SAR chip size used throughout the repo's configs
_NUM_CLASSES = 10  # standard 10-class MSTAR target-recognition setup (SOC benchmark)


def build_aconvnet():
    import torch

    torch.manual_seed(0)
    model = Network(classes=_NUM_CLASSES, channels=1, dropout_rate=0.5)
    model.eval()
    return model


def example_input_aconvnet():
    import torch

    torch.manual_seed(0)
    return torch.randn(1, 1, _MSTAR_CHIP_SIZE, _MSTAR_CHIP_SIZE)


MENAGERIE_ENTRIES = [
    ("A-ConvNets", "build_aconvnet", "example_input_aconvnet", 2016, MENAGERIE_ZOO),
]
