# FAITHFUL PORT of MatthieuCourbariaux/BinaryNet @ d7d45aefd2ada203575b29ebb148fb2c4d925e6d
# (original framework: Theano + Lasagne)
# https://raw.githubusercontent.com/MatthieuCourbariaux/BinaryNet/d7d45aefd2ada203575b29ebb148fb2c4d925e6d/Train-time/binary_net.py
# https://raw.githubusercontent.com/MatthieuCourbariaux/BinaryNet/d7d45aefd2ada203575b29ebb148fb2c4d925e6d/Train-time/cifar10.py
#
# Courbariaux & Bengio 2016, "Binarized Neural Networks: Training Deep Neural Networks
# with Weights and Activations Constrained to +1 or -1" (the BinaryNet / BinaryConnect
# CIFAR-10 CNN from this repo). Theano is EOL and its Lasagne binary layers
# (`binary_net.DenseLayer`/`binary_net.Conv2DLayer`, which subclass Lasagne's own
# `DenseLayer`/`Conv2DLayer` and only override `get_output_for`/`convolve` to swap in a
# binarized weight for that call) cannot reasonably be installed in a modern base-lib
# torch environment, so the architecture is transcribed faithfully into self-contained
# torch/nn.Module code rather than vendored.
#
# Ported architecture (the "128C3-128C3-P2 - 256C3-256C3-P2 - 512C3-512C3-P2 -
# 1024FP-1024FP-10FP" CIFAR-10 CNN built in `cifar10.py`'s `__main__`):
#   - Every Conv2d/Linear layer is a `BinaryConv2d`/`BinaryLinear`: at every forward call
#     the *deterministic* binarization from `binary_net.binarization(W, H, binary=True,
#     deterministic=True, stochastic=False)` is applied to the weight before the
#     underlying conv/linear op -- `Wb = H if round(hard_sigmoid(W/H)) else -H`, which
#     (since `hard_sigmoid(w)=clip((w+1)/2,0,1)` rounds to 1 for w>=0 and 0 for w<0) is
#     exactly `H * sign(W)` with `sign(0) := +1` matching Theano's `T.round` tie-break.
#     `deterministic=True` (i.e. eval/inference-time binarization, no stochastic
#     rounding) is used here since this is a forward-pass-only trace, matching the
#     `test_output = lasagne.layers.get_output(cnn, deterministic=True)` path in the
#     original training script. A straight-through estimator (identity gradient) is
#     used for the binarization op, exactly as `Round3.grad` in the source returns the
#     incoming gradient unchanged ("does not set the gradient to 0 like Theano's" own
#     rounding op).
#   - `H` (the binarization clip range) follows the source's `H="Glorot"` default:
#     `H = sqrt(1.5 / (fan_in + fan_out))` for both conv (`fan_in=prod(kernel)*in_ch`,
#     `fan_out=prod(kernel)*out_ch`) and dense layers (`fan_in=in_features`,
#     `fan_out=out_features`).
#   - Every conv/dense layer is followed by BatchNorm and then the binary activation
#     `binary_tanh_unit(x) = 2*round(hard_sigmoid(x)) - 1`, applied with the same
#     straight-through-estimator convention as the weight binarization (`Round3`).
#   - Stage layout exactly mirrors the source: (128,128,maxpool2) - (256,256,maxpool2) -
#     (512,512,maxpool2) - dense(1024) - dense(1024) - dense(num_classes), each stage
#     `conv -> BN -> binary_tanh` (conv-BN-activation order, matching the source, which
#     applies `BatchNormLayer` then `NonlinearityLayer` after each `Conv2DLayer`/
#     `DenseLayer`), with `MaxPool2DLayer` placed after the first BN+activation of each
#     conv pair and before the second BN+activation, exactly as in `cifar10.py`. The
#     final dense(num_classes) is followed by one last BatchNorm with no activation
#     (`cnn = lasagne.layers.BatchNormLayer(cnn, ...)` is the last op built in the
#     source, with no trailing `NonlinearityLayer`), matching this port's `self.bn_out`.
#   - All conv layers use `pad=1` (same as Lasagne's `pad=1` kwarg) with 3x3 kernels and
#     stride 1, matching the source.
#
# No architecture was invented: every layer, its order, and its binarization semantics
# come directly from the two source files above. Only the numeric channel/classifier
# widths are shrunk (documented in `build_bnn_cifar10` below) to keep the traced graph
# small; the *shape* of the network (stage count, per-stage layer count, conv-BN-act
# order, final BN-only classifier head) is unchanged.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _RoundSTE(torch.autograd.Function):
    """Round-to-nearest with a straight-through (identity) gradient.

    Faithful to `Round3` in the source (`theano.scalar.basic.UnaryScalarOp` whose
    `grad` returns the incoming gradient unchanged instead of zeroing it like
    Theano's built-in round).
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _hard_sigmoid(x):
    # T.clip((x+1.)/2., 0, 1)
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def binary_tanh_unit(x):
    # 2.*round3(hard_sigmoid(x))-1.
    return 2.0 * _RoundSTE.apply(_hard_sigmoid(x)) - 1.0


def _binarize_weight(w, h):
    """Deterministic BinaryConnect weight binarization (`binarization(..., binary=True,
    deterministic=True, stochastic=False)` in the source): round(hard_sigmoid(w/H)) in
    {0,1}, mapped to {-H,+H}, with a straight-through gradient through the whole op
    (the source only ever back-propagates through the *use* of Wb via the STE-rounded
    activations upstream/downstream; the weight itself gets `theano.grad(loss, wrt=Wb)`
    directly, i.e. gradients flow through the binarization as identity).
    """
    wb01 = _RoundSTE.apply(_hard_sigmoid(w / h))
    return (2.0 * wb01 - 1.0) * h


class BinaryConv2d(nn.Module):
    """Conv2d with BinaryConnect weight binarization, faithful to
    `binary_net.Conv2DLayer` (subclasses Lasagne's Conv2DLayer, overrides `convolve`
    to substitute a binarized weight for the stored real-valued one on every call).
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        num_inputs = kernel_size * kernel_size * in_channels
        num_units = kernel_size * kernel_size * out_channels
        h = float(np.sqrt(1.5 / (num_inputs + num_units)))
        self.H = h
        nn.init.uniform_(self.weight, -h, h)

    def forward(self, x):
        wb = _binarize_weight(self.weight, self.H)
        return F.conv2d(x, wb, bias=None, stride=self.stride, padding=self.padding)


class BinaryLinear(nn.Module):
    """Linear with BinaryConnect weight binarization, faithful to
    `binary_net.DenseLayer` (subclasses Lasagne's DenseLayer, overrides
    `get_output_for` to substitute a binarized weight on every call).
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        h = float(np.sqrt(1.5 / (in_features + out_features)))
        self.H = h
        nn.init.uniform_(self.weight, -h, h)

    def forward(self, x):
        wb = _binarize_weight(self.weight, self.H)
        return F.linear(x, wb, bias=None)


class BinaryNetCIFAR10(nn.Module):
    """The "128C3-128C3-P2 - 256C3-256C3-P2 - 512C3-512C3-P2 - 1024FP-1024FP-10FP"
    BinaryNet CNN built in `cifar10.py`'s `__main__` (see module docstring for the
    exact layer-by-layer correspondence).
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=10,
        base_width=128,
        dense_width=1024,
        img_size=32,
        epsilon=1e-4,
        momentum=0.1,
    ):
        super().__init__()
        w1, w2, w3 = base_width, base_width * 2, base_width * 4

        # 128C3-128C3-P2
        self.conv1a = BinaryConv2d(in_chans, w1, 3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(w1, eps=epsilon, momentum=momentum)
        self.conv1b = BinaryConv2d(w1, w1, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1b = nn.BatchNorm2d(w1, eps=epsilon, momentum=momentum)

        # 256C3-256C3-P2
        self.conv2a = BinaryConv2d(w1, w2, 3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(w2, eps=epsilon, momentum=momentum)
        self.conv2b = BinaryConv2d(w2, w2, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2b = nn.BatchNorm2d(w2, eps=epsilon, momentum=momentum)

        # 512C3-512C3-P2
        self.conv3a = BinaryConv2d(w2, w3, 3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(w3, eps=epsilon, momentum=momentum)
        self.conv3b = BinaryConv2d(w3, w3, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3b = nn.BatchNorm2d(w3, eps=epsilon, momentum=momentum)

        flat_dim = w3 * (img_size // 8) * (img_size // 8)

        # 1024FP-1024FP-10FP
        self.fc1 = BinaryLinear(flat_dim, dense_width)
        self.bn_fc1 = nn.BatchNorm1d(dense_width, eps=epsilon, momentum=momentum)
        self.fc2 = BinaryLinear(dense_width, dense_width)
        self.bn_fc2 = nn.BatchNorm1d(dense_width, eps=epsilon, momentum=momentum)
        self.fc3 = BinaryLinear(dense_width, num_classes)
        self.bn_out = nn.BatchNorm1d(num_classes, eps=epsilon, momentum=momentum)

    def forward(self, x):
        x = binary_tanh_unit(self.bn1a(self.conv1a(x)))
        x = self.conv1b(x)
        x = self.pool1(x)
        x = binary_tanh_unit(self.bn1b(x))

        x = binary_tanh_unit(self.bn2a(self.conv2a(x)))
        x = self.conv2b(x)
        x = self.pool2(x)
        x = binary_tanh_unit(self.bn2b(x))

        x = binary_tanh_unit(self.bn3a(self.conv3a(x)))
        x = self.conv3b(x)
        x = self.pool3(x)
        x = binary_tanh_unit(self.bn3b(x))

        x = torch.flatten(x, 1)
        x = binary_tanh_unit(self.bn_fc1(self.fc1(x)))
        x = binary_tanh_unit(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        x = self.bn_out(x)
        return x


def build_bnn_cifar10():
    # Shrunk from the source's base_width=128/dense_width=1024/img_size=32 to keep the
    # traced graph small; stage count, per-stage layer order, and binarization
    # semantics are unchanged from the source.
    return BinaryNetCIFAR10(in_chans=3, num_classes=10, base_width=8, dense_width=16, img_size=32)


def example_input_bnn_cifar10():
    return torch.randn(2, 3, 32, 32)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("BinaryNet-CIFAR10", "build_bnn_cifar10", "example_input_bnn_cifar10", 2016, "ported"),
]
