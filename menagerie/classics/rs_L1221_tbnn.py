# FAITHFUL PORT of sandialabs/tbnn @ 701d57e402f268da047435b0a842e18689f74e14
# https://raw.githubusercontent.com/sandialabs/tbnn/701d57e402f268da047435b0a842e18689f74e14/tbnn/core.py
# (original framework: Theano + Lasagne, Python 2)
#
# Ling, Kurzawski, Templeton 2016 (JFM) "Reynolds averaged turbulence modelling using
# deep neural networks with embedded invariance" -- the Tensor Basis Neural Network
# (TBNN). Maps a set of Galilean-invariant scalar features (the 5 invariants of the
# normalized mean strain-rate/rotation-rate tensors) through an MLP of stacked
# LeakyReLU-activated Dense layers (`_build_NN` in the real `tbnn/core.py`) to
# `num_tensor_basis` linear coefficients (no nonlinearity on the final `linear_layer`,
# matching the source exactly: `DenseLayer(..., nonlinearity=None)`). Those
# coefficients are then merged with a `num_tensor_basis x 9` tensor-basis input
# (the Pope 1975 integrity-basis tensors T^(1)..T^(10) evaluated per-point) via the
# real `TensorLayer.get_output_for`, which the source computes as
# `T.batched_tensordot(coeffs, tensor_basis, axes=[[1], [1]])` -- i.e.
# `output[b, i] = sum_k coeffs[b, k] * tensor_basis[b, k, i]`, producing the predicted
# (flattened 3x3, 9-component) Reynolds-stress anisotropy tensor per point. This is
# the exact mechanism the source uses to enforce Galilean/rotational invariance: the
# network only ever predicts scalar coefficients over a fixed physical tensor basis,
# never the tensor components directly.
#
# The source is Theano/Lasagne (`import theano`, `import lasagne`, Python 2
# `cPickle`/`xrange`/`print` statements) -- Theano/Lasagne are unmaintained
# (Theano dev ceased 2017, Lasagne never reached 1.0) and are not part of this env's
# declared torch/base-lib-only dependency set, so rung 2 (install-and-vendor) does not
# apply. Transcribed faithfully into base-env torch: the hidden-layer stack (LeakyReLU,
# HeUniform-equivalent init via `nn.init.kaiming_uniform_` with the same gain
# `sqrt(2.0)` the source passes to `lasagne.init.HeUniform(gain=np.sqrt(2.0))`), the
# final linear (no-activation) coefficient layer, and the `TensorLayer` batched
# tensordot merge are all ported 1:1. `NetworkStructure` (layer/node count config) and
# the `fit()`/training-loop machinery (SGD optimizer selection, early-stopping,
# Theano `function` compilation) are training-only and not part of the forward
# architecture graph; this module exposes only the forward inference path
# (`_build_NN` + `TensorLayer.get_output_for`), matching what `TBNN.predict` runs.

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TensorBasisMergeLayer(nn.Module):
    """Faithful port of `tbnn.core.TensorLayer.get_output_for`.

    Real source (Theano):
        output = T.batched_tensordot(inputs[0], inputs[1], axes=[[1], [1]])
    where inputs[0] is (batch, num_tensor_basis) coefficients and inputs[1] is
    (batch, num_tensor_basis, 9) tensor-basis values. torch.einsum is the direct
    equivalent of a batched tensordot contracted over the tensor-basis axis.
    """

    def forward(self, coeffs: torch.Tensor, tensor_basis: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bk,bki->bi", coeffs, tensor_basis)


class TBNN(nn.Module):
    """Faithful port of `tbnn.core.TBNN._build_NN` (forward/inference path only).

    Real source builds (Lasagne):
        InputLayer(num_inputs) -> [DenseLayer(num_nodes, LeakyReLU)] * num_layers
            -> DenseLayer(num_tensor_basis, nonlinearity=None)
        merged via TensorLayer with an InputLayer(num_tensor_basis, 9) tensor-basis
        stream. All DenseLayers use `lasagne.init.HeUniform(gain=np.sqrt(2.0))`.
    """

    def __init__(
        self,
        num_inputs: int,
        num_tensor_basis: int,
        num_layers: int = 1,
        num_nodes: int = 10,
        leakiness: float = 0.1,
    ) -> None:
        super().__init__()
        gain = math.sqrt(2.0)

        hidden_layers = []
        in_features = num_inputs
        for _ in range(num_layers):
            linear = nn.Linear(in_features, num_nodes)
            nn.init.kaiming_uniform_(linear.weight, a=leakiness, nonlinearity="leaky_relu")
            linear.weight.data.mul_(gain / math.sqrt(2.0 / (1 + leakiness**2)))
            nn.init.zeros_(linear.bias)
            hidden_layers.append(linear)
            hidden_layers.append(nn.LeakyReLU(leakiness))
            in_features = num_nodes
        self.hidden = nn.Sequential(*hidden_layers)

        # final linear coefficient layer: no nonlinearity, matches source exactly
        self.coeff_layer = nn.Linear(in_features, num_tensor_basis)
        nn.init.kaiming_uniform_(self.coeff_layer.weight, a=0, nonlinearity="relu")
        self.coeff_layer.weight.data.mul_(gain / math.sqrt(2.0))
        nn.init.zeros_(self.coeff_layer.bias)

        self.merge = TensorBasisMergeLayer()

    def forward(self, scalar_features: torch.Tensor, tensor_basis: torch.Tensor) -> torch.Tensor:
        h = self.hidden(scalar_features)
        coeffs = self.coeff_layer(h)
        return self.merge(coeffs, tensor_basis)


def build_tbnn() -> TBNN:
    # Real turbulence example (examples/turbulence) uses 5 scalar invariants and 10
    # tensor-basis members (Pope's integrity basis), kept here as the faithful default.
    return TBNN(num_inputs=5, num_tensor_basis=10, num_layers=2, num_nodes=16).eval()


def example_input_tbnn():
    batch = 4
    scalar_features = torch.randn(batch, 5)
    tensor_basis = torch.randn(batch, 10, 9)
    return (scalar_features, tensor_basis)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "TBNN (Tensor Basis Neural Network)",
        "build_tbnn",
        "example_input_tbnn",
        2016,
        "ported-pytorch",
    ),
]
