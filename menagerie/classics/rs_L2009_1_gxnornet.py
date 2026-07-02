# FAITHFUL PORT of AcrossV/Gated-XNOR @ master (original framework: Theano + Lasagne)
# https://raw.githubusercontent.com/AcrossV/Gated-XNOR/master/cifar10_GXNOR.py
#
# "GXNOR-Net: Training deep neural networks with ternary weights and activations
# without full-precision memory under a unified discretization framework" (Deng, Jiao,
# Pei, Wu, Li; Neural Networks 2018, arXiv:1705.09283). The official repo is
# Theano/Lasagne (`theano.sandbox.cuda.use('gpu2')`, `pylearn2.datasets.cifar10`,
# Python-2 `cPickle`/`print` statements) and is unmaintained since ~2018 -- it cannot
# run in a base-lib modern-Python torch env -- so this is a faithful transcription
# rather than a vendor.
#
# The distinctive forward-pass mechanism, ported verbatim from `discrete_neuron_3states`
# in cifar10_GXNOR.py, is the ternary ({-1,0,+1}) ACTIVATION function used after every
# BatchNorm in the network: round(hard_sigmoid(2*(x-1)) + hard_sigmoid(2*(x+1)) - 1),
# where hard_sigmoid(z) = clip((z+1)/2, 0, 1). round() uses a straight-through
# estimator for the backward pass (the repo's custom Theano `round_custom` Op has
# `grad -> gz` i.e. an identity/pass-through gradient), reproduced here as a
# torch.autograd.Function.
#
# The repo's `Conv2DLayer`/`DenseLayer` subclasses (extending lasagne's own
# Conv2DLayer/DenseLayer) override ONLY `__init__` -- to initialize weights in
# [-H, H] and tag them with the 'discrete' param set -- NOT `get_output_for`/
# `convolve`. So at FORWARD-PASS time the conv/dense layers are ordinary
# convolution/affine ops; the ternary discrete-state-transition (DST) constraint on
# stored weight VALUES (2^N+1 states in [-H, H]) is enforced entirely by the custom
# `discrete_grads` Theano gradient-update rule the optimizer runs BETWEEN training
# steps (a training-time weight-update mechanism with no separate forward-pass
# effect, analogous to BinaryConnect's persistent-full-precision-weight scheme but
# GXNOR-Net constrains the weights themselves rather than binarizing on every
# forward call) -- so it is architecturally inert here and intentionally omitted,
# matching how `discrete_grads`/`train()`/the CIFAR10 data pipeline are training-time
# machinery outside the forward architecture.
#
# The cifar10_GXNOR.py `__main__` network topology is transcribed layer-for-layer:
# InputLayer(3,32,32) ->
#   [Conv2d(128,3x3,pad=1) -> BN -> ternary_act] ->
#   [Conv2d(128,3x3,pad=1) -> MaxPool(2x2) -> BN -> ternary_act] ->
#   [Conv2d(256,3x3,pad=1) -> BN -> ternary_act] ->
#   [Conv2d(256,3x3,pad=1) -> MaxPool(2x2) -> BN -> ternary_act] ->
#   [Conv2d(512,3x3,pad=1) -> BN -> ternary_act] ->
#   [Conv2d(512,3x3,pad=1) -> MaxPool(2x2) -> BN -> ternary_act] ->
#   Dense(1024) -> BN -> ternary_act -> Dense(10) -> BN (identity, no activation --
#   matches the repo's final `cnn = lasagne.layers.BatchNormLayer(cnn, ...)` call with
#   no trailing NonlinearityLayer, feeding directly into the squared-hinge loss).
# Lasagne's Conv2DLayer defaults to `flip_filters=True` (true convolution = PyTorch's
# nn.Conv2d cross-correlation convention after accounting for the flip), so no extra
# kernel flip is required here.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _hard_sigmoid(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


class _TernaryActivation(torch.autograd.Function):
    """Port of `discrete_neuron_3states` (cifar10_GXNOR.py): ternary {-1,0,+1}
    activation via two shifted hard-sigmoids, rounded. The repo's custom
    `round_custom` Theano Op passes the incoming gradient straight through
    (`grad -> (gz,)`), reproduced here as a straight-through estimator."""

    @staticmethod
    def forward(ctx, x):
        z = _hard_sigmoid(2 * (x - 1)) + _hard_sigmoid(2 * (x + 1)) - 1
        return torch.round(z)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def discrete_neuron_3states(x):
    return _TernaryActivation.apply(x)


class GXNORNetCIFAR10(nn.Module):
    """Port of cifar10_GXNOR.py's `__main__` CNN builder:
    128C3-128C3-P2-256C3-256C3-P2-512C3-512C3-P2-1024FC-(n_classes)FC, ternary-weight-
    initialized conv/dense layers with the ternary `discrete_neuron_3states` activation
    after every BatchNorm except the final classifier BatchNorm."""

    def __init__(self, n_classes=10, H=1.0, epsilon=1e-4, alpha=0.1):
        super().__init__()
        self.H = H

        def conv(c_in, c_out):
            m = nn.Conv2d(c_in, c_out, 3, padding=1)
            nn.init.uniform_(m.weight, -H, H)
            nn.init.zeros_(m.bias)
            return m

        self.conv1 = conv(3, 128)
        self.bn1 = nn.BatchNorm2d(128, eps=epsilon, momentum=alpha)

        self.conv2 = conv(128, 128)
        self.pool1 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(128, eps=epsilon, momentum=alpha)

        self.conv3 = conv(128, 256)
        self.bn3 = nn.BatchNorm2d(256, eps=epsilon, momentum=alpha)

        self.conv4 = conv(256, 256)
        self.pool2 = nn.MaxPool2d(2)
        self.bn4 = nn.BatchNorm2d(256, eps=epsilon, momentum=alpha)

        self.conv5 = conv(256, 512)
        self.bn5 = nn.BatchNorm2d(512, eps=epsilon, momentum=alpha)

        self.conv6 = conv(512, 512)
        self.pool3 = nn.MaxPool2d(2)
        self.bn6 = nn.BatchNorm2d(512, eps=epsilon, momentum=alpha)

        def dense(f_in, f_out):
            m = nn.Linear(f_in, f_out)
            nn.init.uniform_(m.weight, -H, H)
            nn.init.zeros_(m.bias)
            return m

        self.fc1 = dense(512 * 4 * 4, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024, eps=epsilon, momentum=alpha)

        self.fc2 = dense(1024, n_classes)
        self.bn_fc2 = nn.BatchNorm1d(n_classes, eps=epsilon, momentum=alpha)

    def forward(self, x):
        x = discrete_neuron_3states(self.bn1(self.conv1(x)))

        x = self.conv2(x)
        x = self.pool1(x)
        x = discrete_neuron_3states(self.bn2(x))

        x = discrete_neuron_3states(self.bn3(self.conv3(x)))

        x = self.conv4(x)
        x = self.pool2(x)
        x = discrete_neuron_3states(self.bn4(x))

        x = discrete_neuron_3states(self.bn5(self.conv5(x)))

        x = self.conv6(x)
        x = self.pool3(x)
        x = discrete_neuron_3states(self.bn6(x))

        x = torch.flatten(x, 1)

        x = discrete_neuron_3states(self.bn_fc1(self.fc1(x)))
        x = self.bn_fc2(self.fc2(x))
        return x


def build_gxnornet_cifar10():
    model = GXNORNetCIFAR10(n_classes=10)
    model.eval()
    return model


def example_input_gxnornet_cifar10():
    torch.manual_seed(0)
    return (torch.randn(2, 3, 32, 32),)


MENAGERIE_ENTRIES = [
    (
        "GXNOR-Net_CIFAR10",
        "build_gxnornet_cifar10",
        "example_input_gxnornet_cifar10",
        2018,
        "ported",
    ),
]
