# FAITHFUL PORT of MatthieuCourbariaux/BinaryNet @ master, Train-time/ (original framework: Theano + Lasagne)
# https://raw.githubusercontent.com/MatthieuCourbariaux/BinaryNet/master/Train-time/binary_net.py
# https://raw.githubusercontent.com/MatthieuCourbariaux/BinaryNet/master/Train-time/cifar10.py
#
# "Binarized Neural Networks: Training Deep Neural Networks with Weights and
# Activations Constrained to +1 or -1" (Courbariaux, Hubara, Soudry, El-Yaniv, Bengio,
# 2016 / NeurIPS 2016 "Binarized Neural Networks"). The official repo is Theano/Lasagne
# with a Python-2-only c_code signature (`def c_code(self, node, name, (x,), (z,),
# sub):` -- tuple-unpacking function args, removed in Python 3) and depends on
# `theano.sandbox.cuda`/`pylearn2` -- it cannot run in a base-lib modern-Python torch
# env (Theano has been unmaintained since ~2017), so this is a faithful transcription
# of the real forward-pass architecture rather than a vendor. BinaryNet extends
# BinaryConnect (same repo family, same author) by binarizing BOTH weights AND
# activations, unlike sibling project BinaryConnect (weights only, real-valued
# activations). Ported verbatim from binary_net.py's real functions:
#   - `binarization(W, H, binary, deterministic, stochastic, srng)`: weight
#     binarization identical to BinaryConnect's (hard_sigmoid(W/H) -> round/Bernoulli
#     -> fold to {-H,+H}); cifar10.py uses `stochastic=False` (deterministic rounding).
#   - `binary_tanh_unit(x) = 2*round3(hard_sigmoid(x)) - 1`: the activation
#     binarization used as the network's nonlinearity everywhere (`round3` behaves like
#     round() forward / straight-through identity gradient backward, matching the
#     repo's custom `Round3` UnaryScalarOp docstring: "does not set the gradient to 0
#     like Theano's [round]"). Algebraically hard_sigmoid clips to [0,1] then rounds to
#     {0,1} then rescales to {-1,+1} -- equivalent to sign(x) with the sub-|x|<1 region
#     of hard_sigmoid providing a nonzero straight-through gradient interval.
# The `cifar10.py` architecture (weights AND activations both binarized) is
# transcribed layer-for-layer: InputLayer(3,32,32) ->
# [BinConv2d(128,3x3,pad=1) -> BN -> binary_tanh] x2 -> MaxPool(2x2) ->
# [BinConv2d(256,3x3,pad=1) -> BN -> binary_tanh] x2 -> MaxPool(2x2) ->
# [BinConv2d(512,3x3,pad=1) -> BN -> binary_tanh] x2 -> MaxPool(2x2) ->
# BinDense(1024) -> BN -> binary_tanh -> BinDense(1024) -> BN -> binary_tanh ->
# BinDense(10) -> BN (identity, no activation -- feeds the squared-hinge loss directly,
# matching the repo's final un-activated `lasagne.layers.BatchNormLayer(cnn, epsilon=
# epsilon, alpha=alpha)` call with no `nonlinearity` kwarg override and BinaryNet's
# raw (non-`batch_norm`-wrapper) `BatchNormLayer` default of identity nonlinearity).
# Training-time-only machinery (compute_grads, clipping_scaling, the train() loop,
# hinge loss) is intentionally omitted -- it isn't part of the forward architecture.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _hard_sigmoid(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def _glorot_H(num_inputs, num_units):
    return math.sqrt(1.5 / (num_inputs + num_units))


class _BinarizeWeight(torch.autograd.Function):
    """Straight-through weight binarization (binary_net.py's `binarization()`)."""

    @staticmethod
    def forward(ctx, weight, H, stochastic, training):
        Wb = _hard_sigmoid(weight / H)
        if stochastic and training:
            Wb = torch.bernoulli(Wb)
        else:
            Wb = torch.round(Wb)
        Wb = torch.where(Wb.bool(), torch.full_like(Wb, H), torch.full_like(Wb, -H))
        return Wb

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class _BinaryTanhUnit(torch.autograd.Function):
    """Straight-through activation binarization: forward computes
    `2*round(hard_sigmoid(x)) - 1` (binary_net.py's `binary_tanh_unit`); backward
    passes the gradient straight through (Round3's documented behavior: "does not
    set the gradient to 0 like Theano's [round]")."""

    @staticmethod
    def forward(ctx, x):
        return 2.0 * torch.round(_hard_sigmoid(x)) - 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def binarize_weight(weight, H, stochastic, training):
    return _BinarizeWeight.apply(weight, H, stochastic, training)


def binary_tanh_unit(x):
    return _BinaryTanhUnit.apply(x)


class BinaryConv2d(nn.Module):
    """Port of binary_net.py's `Conv2DLayer` (extends lasagne.layers.Conv2DLayer)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        binary=True,
        stochastic=False,
        H="Glorot",
        w_lr_scale="Glorot",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.binary = binary
        self.stochastic = stochastic

        num_inputs = kernel_size * kernel_size * in_channels
        num_units = kernel_size * kernel_size * out_channels
        self.H = _glorot_H(num_inputs, num_units) if H == "Glorot" else H
        self.w_lr_scale = (1.0 / self.H) if w_lr_scale == "Glorot" else w_lr_scale

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if self.binary:
            nn.init.uniform_(self.weight, -self.H, self.H)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if self.binary:
            Wb = binarize_weight(self.weight, self.H, self.stochastic, self.training)
        else:
            Wb = self.weight
        return F.conv2d(x, Wb, self.bias, stride=self.stride, padding=self.padding)


class BinaryLinear(nn.Module):
    """Port of binary_net.py's `DenseLayer` (extends lasagne.layers.DenseLayer)."""

    def __init__(
        self,
        in_features,
        out_features,
        binary=True,
        stochastic=False,
        H="Glorot",
        w_lr_scale="Glorot",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.binary = binary
        self.stochastic = stochastic

        self.H = _glorot_H(in_features, out_features) if H == "Glorot" else H
        self.w_lr_scale = (1.0 / self.H) if w_lr_scale == "Glorot" else w_lr_scale

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if self.binary:
            nn.init.uniform_(self.weight, -self.H, self.H)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        if self.binary:
            Wb = binarize_weight(self.weight, self.H, self.stochastic, self.training)
        else:
            Wb = self.weight
        return F.linear(x, Wb, self.bias)


class BinaryTanh(nn.Module):
    def forward(self, x):
        return binary_tanh_unit(x)


class BinaryNetCIFAR10(nn.Module):
    """Port of cifar10.py's CNN builder: 128C3-128C3-P2-256C3-256C3-P2-512C3-512C3-P2
    -1024FC-1024FC-(n_classes)FC, weight-AND-activation-binarized. Every BatchNorm is
    followed by `binary_tanh_unit` except the final classifier BatchNorm (identity --
    the repo's last `BatchNormLayer(cnn, epsilon=epsilon, alpha=alpha)` call has no
    `nonlinearity=` override)."""

    def __init__(self, n_classes=10, binary=True, stochastic=False, H=1.0, epsilon=1e-4, alpha=0.1):
        super().__init__()

        def conv_bn_bintanh(c_in, c_out):
            return nn.Sequential(
                BinaryConv2d(c_in, c_out, 3, padding=1, binary=binary, stochastic=stochastic, H=H),
                nn.BatchNorm2d(c_out, eps=epsilon, momentum=alpha),
                BinaryTanh(),
            )

        self.block1 = nn.Sequential(
            conv_bn_bintanh(3, 128),
            conv_bn_bintanh(128, 128),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = nn.Sequential(
            conv_bn_bintanh(128, 256),
            conv_bn_bintanh(256, 256),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = nn.Sequential(
            conv_bn_bintanh(256, 512),
            conv_bn_bintanh(512, 512),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.fc1 = BinaryLinear(512 * 4 * 4, 1024, binary=binary, stochastic=stochastic, H=H)
        self.bn_fc1 = nn.BatchNorm1d(1024, eps=epsilon, momentum=alpha)
        self.fc2 = BinaryLinear(1024, 1024, binary=binary, stochastic=stochastic, H=H)
        self.bn_fc2 = nn.BatchNorm1d(1024, eps=epsilon, momentum=alpha)
        self.fc3 = BinaryLinear(1024, n_classes, binary=binary, stochastic=stochastic, H=H)
        self.bn_fc3 = nn.BatchNorm1d(n_classes, eps=epsilon, momentum=alpha)

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.pool3(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = binary_tanh_unit(self.bn_fc1(x))
        x = self.fc2(x)
        x = binary_tanh_unit(self.bn_fc2(x))
        x = self.fc3(x)
        x = self.bn_fc3(x)
        return x


def build_binarynet_cifar10():
    model = BinaryNetCIFAR10(n_classes=10, binary=True, stochastic=False)
    model.eval()
    return model


def example_input_binarynet_cifar10():
    torch.manual_seed(0)
    return (torch.randn(2, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    (
        "BinaryNet_CIFAR10",
        "build_binarynet_cifar10",
        "example_input_binarynet_cifar10",
        2016,
        "ported",
    ),
]
