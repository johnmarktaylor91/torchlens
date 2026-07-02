# SOURCE: vendored from autonlab/auton-survival @ master
#
# Files combined below (imports/paths adjusted only, architecture untouched):
#   auton_survival/models/dsm/dsm_torch.py (create_representation helper only)
#   auton_survival/models/dcm/dcm_torch.py -> DeepCoxMixturesTorch
#
# The queued candidate repo (chiragnagpal/deep_cox_mixtures, the official MLHC 2021
# "Deep Cox Mixtures for Survival Regression" code) explicitly deprecates its own
# TensorFlow implementation in its README and redirects to the "more stable pytorch
# implementation" shipped in autonlab/auton-survival (same first author, same paper).
# That pytorch module (`DeepCoxMixturesTorch`) has no dependency beyond torch/numpy
# (the `create_representation` MLP-builder helper it imports is vendored alongside
# it below) so no extra packages are needed to run it.
#
# Deep Cox Mixtures assumes the population is a latent mixture of K subgroups, each
# obeying its own Cox proportional-hazards model; an embedding MLP feeds a "gate"
# head (soft cluster assignment, log-softmax) and an "expert" head (per-cluster log
# hazard ratio, clamped or tanh-scaled).
#
# Repo: https://github.com/autonlab/auton-survival @ master
#   auton_survival/models/dsm/dsm_torch.py
#   auton_survival/models/dcm/dcm_torch.py

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# auton_survival/models/dsm/dsm_torch.py (create_representation helper only)
# ---------------------------------------------------------------------------
def create_representation(inputdim, layers, activation, bias=False):
    r"""Helper function to generate the representation function for DSM/DCM.

    Instantiates a Non Linear Multilayer Perceptron (torch.nn.Module) representation
    for the input data.
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


# ---------------------------------------------------------------------------
# auton_survival/models/dcm/dcm_torch.py
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
_INPUT_DIM = 16
_K = 3
_LAYERS = [32, 16]


def build_deep_cox_mixtures():
    return DeepCoxMixturesTorch(inputdim=_INPUT_DIM, k=_K, layers=_LAYERS)


def example_input_deep_cox_mixtures():
    return torch.randn(4, _INPUT_DIM)


MENAGERIE_ENTRIES = [
    (
        "Deep Cox Mixtures",
        build_deep_cox_mixtures,
        example_input_deep_cox_mixtures,
        2021,
        "VENDOR",
    ),
]
