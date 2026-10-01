# SOURCE: vendored from autonlab/auton-survival @ 5dde465f7223601717abddc1d075e837707c403b
# https://raw.githubusercontent.com/autonlab/auton-survival/5dde465f7223601717abddc1d075e837707c403b/auton_survival/models/dcm/dcm_torch.py
# https://raw.githubusercontent.com/autonlab/auton-survival/5dde465f7223601717abddc1d075e837707c403b/auton_survival/models/dsm/dsm_torch.py
#
# Nagpal, Yadlowsky, Rostamzadeh & Heller (MLHC 2021), "Deep Cox Mixtures for
# Survival Regression". The paper's original repo
# (chiragnagpal/deep_cox_mixtures) ships only a README + PDF -- no runnable
# code. The same Auton Lab (Carnegie Mellon) later folded the maintained
# implementation into their `auton-survival` package
# (auton_survival/models/dcm/dcm_torch.py), which is the real, unmodified
# PyTorch architecture: `DeepCoxMixturesTorch` is a mixture-of-K-Cox-models
# survival model -- an MLP representation `Phi(X)` (via the sibling
# `create_representation` helper from `dsm_torch.py`, shared with Deep
# Survival Machines), followed by two linear heads: a `gate` (log-softmax
# mixture-assignment logits over K subgroups) and an `expert` (per-subgroup
# log-hazard-ratio, clamped or tanh-scaled by `gamma`).
#
# `DeepCoxMixturesTorch` and `create_representation` are the real,
# unmodified classes/functions from the files above (layer composition and
# forward-pass control flow untouched). Only mechanical staging edits:
#   - Merged `create_representation` (originally imported from the sibling
#     `auton_survival.models.dsm.dsm_torch` module) directly into this file
#     since we vendor a single self-contained module, not the whole
#     `auton_survival` package tree.
#   - Dropped the unused `import numpy as np` (not referenced by the model
#     forward pass) and the module docstring.
#   - Added `build_deep_cox_mixtures()` / `example_input_deep_cox_mixtures()`
#     staging entry points at a small representation size (inputdim=16,
#     layers=[32], k=3 subgroups), matching the constructor's real
#     `DeepCoxMixturesTorch(inputdim, k, layers=...)` signature.

import torch
import torch.nn as nn


def create_representation(inputdim, layers, activation, bias=False):
    r"""Helper function to generate the representation function for DSM/DCM.

    Deep Survival Machines / Deep Cox Mixtures learn a representation
    (\ Phi(X) \) for the input data. This representation is parameterized
    using a Non Linear Multilayer Perceptron (`torch.nn.Module`). This is a
    helper function designed to instantiate the representation.

    .. warning::
      Not designed to be used directly.

    Parameters
    ----------
    inputdim: int
        Dimensionality of the input features.
    layers: list
        A list consisting of the number of neurons in each hidden layer.
    activation: str
        Choice of activation function: One of 'ReLU6', 'ReLU' or 'SeLU'.

    Returns
    ----------
    an MLP with torch.nn.Module with the specfied structure.

    """

    if activation == "ReLU6":
        act = nn.ReLU6()
    elif activation == "ReLU":
        act = nn.ReLU()
    elif activation == "SeLU":
        act = nn.SELU()
    elif activation == "Tanh":
        act = nn.Tanh()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=bias))
        modules.append(act)
        prevdim = hidden

    return nn.Sequential(*modules)


class DeepCoxMixturesTorch(nn.Module):
    """PyTorch model definition of the Deep Cox Mixture Survival Model.

    The Cox Mixture involves the assumption that the survival function
    of the individual to be a mixture of K Cox Models. Conditioned on each
    subgroup Z=k; the PH assumptions are assumed to hold and the baseline
    hazard rates is determined non-parametrically using an spline-interpolated
    Breslow's estimator.
    """

    def _init_dcm_layers(self, lastdim):
        self.gate = torch.nn.Linear(lastdim, self.k, bias=False)
        self.expert = torch.nn.Linear(lastdim, self.k, bias=False)

    def __init__(self, inputdim, k, gamma=1, use_activation=False, layers=None, optimizer="Adam"):
        super(DeepCoxMixturesTorch, self).__init__()

        if not isinstance(k, int):
            raise ValueError(f"k must be int, but supplied k is {type(k)}")

        self.k = k
        self.optimizer = optimizer

        if layers is None:
            layers = []
        self.layers = layers

        if len(layers) == 0:
            lastdim = inputdim
        else:
            lastdim = layers[-1]

        self._init_dcm_layers(lastdim)
        self.embedding = create_representation(inputdim, layers, "ReLU6")
        self.gamma = gamma
        self.use_activation = use_activation

    def forward(self, x):
        gamma = self.gamma

        x = self.embedding(x)
        if self.use_activation:
            log_hazard_ratios = gamma * torch.nn.Tanh()(self.expert(x))
        else:
            log_hazard_ratios = torch.clamp(self.expert(x), min=-gamma, max=gamma)
        log_gate_prob = torch.nn.LogSoftmax(dim=1)(self.gate(x))

        return log_gate_prob, log_hazard_ratios


def build_deep_cox_mixtures():
    return DeepCoxMixturesTorch(inputdim=16, k=3, layers=[32])


def example_input_deep_cox_mixtures():
    return torch.randn(4, 16)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepCoxMixtures",
        "build_deep_cox_mixtures",
        "example_input_deep_cox_mixtures",
        2021,
        "vendored",
    ),
]
