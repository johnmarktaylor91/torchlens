# FAITHFUL PORT of MatthieuCourbariaux/BinaryNet @ d7d45aefd2ada203575b29ebb148fb2c4d925e6d
# (original framework: Theano / Lasagne, Python 2)
#
# BinaryNet (Courbariaux, Bengio. 2016, "Training Deep Neural Networks with Weights and
# Activations Constrained to +1 or -1" / "BinaryNet"). The canonical BinaryConnect +
# binarized-activation network: fully-connected (or convolutional) layers whose weights
# are binarized to {-H, +H} (deterministic BinaryConnect rounding, straight-through
# estimator) and whose activations are binarized to {-1, +1} via a sign-like unit with
# a hard-tanh straight-through backward. Ported here is the repo's own MNIST MLP
# (`Train-time/mnist.py`'s `__main__` model-construction block: `dropout_in=.2` ->
# 3x [BinaryDenseLayer(4096) -> BatchNorm -> binary_tanh_unit -> Dropout(.5)] ->
# BinaryDenseLayer(10) -> BatchNorm), transcribed faithfully from the real source:
#   https://raw.githubusercontent.com/MatthieuCourbariaux/BinaryNet/master/Train-time/binary_net.py
#   https://raw.githubusercontent.com/MatthieuCourbariaux/BinaryNet/master/Train-time/mnist.py
#
# Cannot run/vendor as-is: the real code is Theano + Lasagne + `cPickle`/`pylearn2`
# Python 2 (its `Round3` custom Theano op even uses Python-2-only tuple-unpacking
# function args, `def c_code(self, node, name, (x,), (z,), sub)`); Theano has been
# unmaintained/dead since ~2017 and pylearn2's MNIST loader is likewise unmaintained.
# None of that stack installs in a modern Python 3 / PyTorch environment. This is a
# from-scratch-in-torch TRANSCRIPTION of the real Theano/Lasagne code's exact
# computation graph, not a paper-only reimplementation.
#
# What is preserved exactly (mechanism-for-mechanism from the real source files):
#   - `hard_sigmoid(x) = clip((x+1)/2, 0, 1)` exactly as in binary_net.py.
#   - `binary_tanh_unit(x) = 2*round3(hard_sigmoid(x)) - 1`, where `round3` is Theano's
#     custom `Round3` scalar op: forward = elementwise round-to-nearest-integer,
#     backward = identity straight-through gradient (`grad` returns `gz` unchanged,
#     "does not set the gradient to 0 like Theano's [built-in round]"). Reproduced here
#     as `BinaryTanhSTE` (a `torch.autograd.Function` with `round()` forward and
#     identity backward), composed exactly as `2*round(clip((x+1)/2,0,1)) - 1`.
#   - `binarization(W, H, binary=True, deterministic, stochastic=False, ...)`'s
#     deterministic (non-stochastic) BinaryConnect path used by both `mnist.py` runs
#     (`stochastic = False`): `Wb = round(hard_sigmoid(W/H))` (i.e. round(clip((W/H+1)/2,
#     0,1)) in {0,1}), then `Wb = where(Wb, H, -H)` in {-H, +H} -- with an identity
#     straight-through gradient back to the real-valued `W` (BinaryConnect's defining
#     property: forward uses `Wb`, backward differentiates through as if `Wb == W`).
#     Reproduced here as `BinaryConnectSTE`.
#   - The real per-layer Glorot-style `H` and `W_LR_scale` formulas from `DenseLayer.
#     __init__` when `H="Glorot"`: `H = sqrt(1.5 / (num_inputs + num_units))` (kept as
#     each `BinaryLinear`'s `self.H`, used inside the forward binarization -- the
#     `W_LR_scale` term only rescales the *optimizer* update and has no effect on the
#     forward/backward computation graph traced here, so it is omitted).
#   - The real layer order and sizes: `InputLayer(1,28,28) -> Dropout(.2) ->
#     [BinaryDense(4096) -> BatchNorm(eps=1e-4) -> binary_tanh_unit -> Dropout(.5)] * 3
#     -> BinaryDense(10) -> BatchNorm(eps=1e-4)` exactly as built in `mnist.py`'s
#     `__main__` (`num_units=4096`, `n_hidden_layers=3`, `dropout_in=.2`,
#     `dropout_hidden=.5`, `epsilon=1e-4`).
#
# What is dropped (training-harness plumbing, not architecture): the squared-hinge
# loss, Adam-with-weight-clipping optimizer (`compute_grads`/`clipping_scaling`), the
# MNIST/pylearn2 data loading, and the epoch loop -- none of that is part of the
# trainable network's forward graph.
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """`hard_sigmoid(x) = clip((x+1)/2, 0, 1)` -- exact port."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


class Round3STE(torch.autograd.Function):
    """Faithful port of Theano's custom `Round3` scalar op: forward rounds to the
    nearest integer, backward is the identity straight-through estimator (the repo's
    own comment: "does not set the gradient to 0 like Theano's [round]")."""

    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def binary_tanh_unit(x: torch.Tensor) -> torch.Tensor:
    """`2*round3(hard_sigmoid(x)) - 1` -- exact port of binary_net.py's activation
    binarizer (forward behaves like sign(x); backward like hard_tanh)."""
    return 2.0 * Round3STE.apply(hard_sigmoid(x)) - 1.0


class BinaryConnectSTE(torch.autograd.Function):
    """Faithful port of `binarization(W, H, binary=True, deterministic=False,
    stochastic=False)`'s deterministic path: Wb = where(round(hard_sigmoid(W/H)), H,
    -H), with an identity straight-through gradient back to the real-valued W (the
    defining BinaryConnect property)."""

    @staticmethod
    def forward(ctx, w, h):
        wb01 = ((w / h + 1.0) / 2.0).clamp(0.0, 1.0).round()
        return torch.where(wb01.bool(), torch.full_like(w, h), torch.full_like(w, -h))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class BinaryLinear(nn.Module):
    """Faithful port of `binary_net.DenseLayer` with `binary=True, stochastic=False,
    H="Glorot"`: a real `nn.Linear`-shaped weight, binarized to {-H,+H} at every
    forward call via `BinaryConnectSTE`, matching the real
    `self.Wb = binarization(self.W, ...); self.W = self.Wb; rvalue = super().
    get_output_for(...); self.W = Wr` swap-and-restore pattern."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # H = sqrt(1.5 / (num_inputs + num_units)), the real Glorot-style formula from
        # DenseLayer.__init__ when H="Glorot".
        self.h = math.sqrt(1.5 / (in_features + out_features))
        # lasagne.init.Uniform((-H, H)) -- the real weight init range.
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-self.h, self.h))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        wb = BinaryConnectSTE.apply(self.weight, self.h)
        return F.linear(x, wb, self.bias)


class BinaryNetMLP(nn.Module):
    """Faithful port of `mnist.py`'s `__main__` MLP construction:
    Dropout(.2) -> [BinaryLinear(4096) -> BatchNorm1d(eps=1e-4) -> binary_tanh_unit ->
    Dropout(.5)] * 3 -> BinaryLinear(10) -> BatchNorm1d(eps=1e-4)."""

    def __init__(
        self,
        in_features: int = 28 * 28,
        num_units: int = 4096,
        n_hidden_layers: int = 3,
        num_classes: int = 10,
        dropout_in: float = 0.2,
        dropout_hidden: float = 0.5,
        bn_eps: float = 1e-4,
    ):
        super().__init__()
        self.dropout_in = nn.Dropout(dropout_in)

        hidden_blocks = []
        dims = in_features
        for _ in range(n_hidden_layers):
            hidden_blocks.append(
                nn.ModuleDict(
                    {
                        "dense": BinaryLinear(dims, num_units),
                        "bn": nn.BatchNorm1d(num_units, eps=bn_eps),
                        "dropout": nn.Dropout(dropout_hidden),
                    }
                )
            )
            dims = num_units
        self.hidden_blocks = nn.ModuleList(hidden_blocks)

        self.out_dense = BinaryLinear(dims, num_classes)
        self.out_bn = nn.BatchNorm1d(num_classes, eps=bn_eps)

    def forward(self, x):
        x = x.flatten(1)
        x = self.dropout_in(x)
        for block in self.hidden_blocks:
            x = block["dense"](x)
            x = block["bn"](x)
            x = binary_tanh_unit(x)
            x = block["dropout"](x)
        x = self.out_dense(x)
        x = self.out_bn(x)
        return x


def build_binarynet():
    # Small num_units for fast tracing; real architecture (layer order, binarization
    # mechanism, BatchNorm/Dropout placement) unchanged from mnist.py.
    return BinaryNetMLP(in_features=28 * 28, num_units=32, n_hidden_layers=3, num_classes=10)


def example_input_binarynet():
    # bc01 MNIST input, real repo scales pixels to [-1,+1]; shape only matters for
    # tracing, values are just random.
    return torch.randn(2, 1, 28, 28)


MENAGERIE_ENTRIES = [
    ("BinaryNet", "build_binarynet", "example_input_binarynet", 2016, "ported-pytorch"),
]
