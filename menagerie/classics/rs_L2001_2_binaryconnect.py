# FAITHFUL PORT of MatthieuCourbariaux/BinaryConnect @ lasagne (original framework: Theano + Lasagne)
# https://raw.githubusercontent.com/MatthieuCourbariaux/BinaryConnect/lasagne/binary_connect.py
# https://raw.githubusercontent.com/MatthieuCourbariaux/BinaryConnect/lasagne/cifar10.py
#
# "BinaryConnect: Training Deep Neural Networks with binary weights during
# propagations" (Courbariaux, Bengio, David, NeurIPS 2015). The official repo is
# Theano/Lasagne, gated by `theano.sandbox.cuda`, `MRG_RandomStreams`, and Python-2
# constructor keyword defaults -- it cannot run in a base-lib modern-Python torch env
# (Theano itself has been unmaintained since ~2017), so this is a faithful transcription
# rather than a vendor. Two mechanisms are ported verbatim from `binary_connect.py`'s
# real `binarization()` function: (1) deterministic binarization -- round(hard_sigmoid(
# W/H)) folded to {-H,+H} -- used at inference (deterministic=True), and (2) stochastic
# binarization -- Bernoulli sample with p=hard_sigmoid(W/H), folded to {-H,+H} -- used
# during training forward passes, matching the repo's `stochastic=True` default. Weights
# are STORED full-precision (`self.weight`) and BINARIZED ONLY IN THE FORWARD PASS (the
# repo's straight-through-estimator training scheme: real weight update, binary weight
# used for compute), exactly as `Conv2DLayer.convolve`/`DenseLayer.get_output_for` swap
# in `self.Wb` for the duration of the parent-class forward call and restore `self.W`
# afterward. `H` ("Glorot" mode) and `W_LR_scale` ("Glorot" mode) use the exact Glorot
# formulas from the repo (`H = sqrt(1.5/(fan_in+fan_out))`, `W_LR_scale = 1/H`) --
# `W_LR_scale` is a training-time gradient-scaling factor with no forward-pass effect
# and is retained here only as a stored attribute for fidelity, unused at inference.
# The `cifar10.py` architecture (activations stay REAL-VALUED -- unlike sibling
# project BinaryNet, only the WEIGHTS are binarized) is transcribed layer-for-layer:
# InputLayer(3,32,32) -> [BinConv2d(128,3x3,pad=1) -> BN(ReLU)] x2 -> MaxPool(2x2) ->
# [BinConv2d(256,3x3,pad=1) -> BN(ReLU)] x2 -> MaxPool(2x2) ->
# [BinConv2d(512,3x3,pad=1) -> BN(ReLU)] x2 -> MaxPool(2x2) ->
# BinDense(1024) -> BN(ReLU) -> BinDense(1024) -> BN(ReLU) -> BinDense(10) -> BN(identity).
# Lasagne's Conv2DLayer defaults to `flip_filters=True` (true convolution, i.e.
# cross-correlation with a flipped kernel) matching PyTorch's nn.Conv2d cross-correlation
# convention, so no kernel flip is needed. Training-time-only machinery (compute_grads,
# clipping_scaling, the train() loop, hinge loss) is intentionally omitted -- it isn't
# part of the forward architecture.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _hard_sigmoid(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def _glorot_H(num_inputs, num_units):
    # H = "Glorot" branch of binary_connect.py's Conv2DLayer/DenseLayer __init__.
    return math.sqrt(1.5 / (num_inputs + num_units))


class _Binarize(torch.autograd.Function):
    """Straight-through binarization: forward binarizes, backward passes gradient
    through unchanged (matches the repo's Theano `Round3`/`switch` op used for the
    weight update -- the gradient w.r.t. Wb flows straight to W in `compute_grads`)."""

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


def binarize(weight, H, stochastic, training):
    return _Binarize.apply(weight, H, stochastic, training)


class BinaryConv2d(nn.Module):
    """Port of binary_connect.py's `Conv2DLayer` (extends lasagne.layers.Conv2DLayer)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        binary=True,
        stochastic=True,
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
            Wb = binarize(self.weight, self.H, self.stochastic, self.training)
        else:
            Wb = self.weight
        return F.conv2d(x, Wb, self.bias, stride=self.stride, padding=self.padding)


class BinaryLinear(nn.Module):
    """Port of binary_connect.py's `DenseLayer` (extends lasagne.layers.DenseLayer)."""

    def __init__(
        self,
        in_features,
        out_features,
        binary=True,
        stochastic=True,
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
            Wb = binarize(self.weight, self.H, self.stochastic, self.training)
        else:
            Wb = self.weight
        return F.linear(x, Wb, self.bias)


class BinaryConnectCIFAR10(nn.Module):
    """Port of cifar10.py's CNN builder: 128C3-128C3-P2-256C3-256C3-P2-512C3-512C3-P2
    -1024FC-1024FC-(n_classes)FC, weight-binarized conv/dense layers. Every BatchNorm
    in cifar10.py (all conv blocks AND both hidden FC layers) is built via the
    `batch_norm.BatchNormLayer` wrapper with an explicit `nonlinearity=lasagne
    .nonlinearities.rectify` kwarg, including the last conv block and both hidden FC
    BatchNorms -- so ReLU follows every BatchNorm here except the final classifier
    BatchNorm (fc3 -> bn_fc3), whose call in the repo omits the nonlinearity kwarg
    (defaults to identity), matching the squared-hinge-loss-on-raw-BN-output setup."""

    def __init__(self, n_classes=10, binary=True, stochastic=True, H=1.0, epsilon=1e-4, alpha=0.1):
        super().__init__()

        def conv_bn_relu(c_in, c_out):
            return nn.Sequential(
                BinaryConv2d(c_in, c_out, 3, padding=1, binary=binary, stochastic=stochastic, H=H),
                nn.BatchNorm2d(c_out, eps=epsilon, momentum=alpha),
                nn.ReLU(inplace=True),
            )

        self.block1 = nn.Sequential(
            conv_bn_relu(3, 128),
            conv_bn_relu(128, 128),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.block3 = nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
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
        x = F.relu(self.bn_fc1(x), inplace=True)
        x = self.fc2(x)
        x = F.relu(self.bn_fc2(x), inplace=True)
        x = self.fc3(x)
        x = self.bn_fc3(x)
        return x


def build_binaryconnect_cifar10():
    model = BinaryConnectCIFAR10(n_classes=10, binary=True, stochastic=False)
    model.eval()
    return model


def example_input_binaryconnect_cifar10():
    torch.manual_seed(0)
    return (torch.randn(2, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    (
        "BinaryConnect_CIFAR10",
        "build_binaryconnect_cifar10",
        "example_input_binaryconnect_cifar10",
        2015,
        "ported",
    ),
]
