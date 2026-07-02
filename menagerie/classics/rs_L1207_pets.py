# FAITHFUL PORT of kchua/handful-of-trials @ master (original framework: TensorFlow 1.x)
#
# https://raw.githubusercontent.com/kchua/handful-of-trials/master/dmbrl/modeling/models/BNN.py
# https://raw.githubusercontent.com/kchua/handful-of-trials/master/dmbrl/modeling/layers/FC.py
# https://raw.githubusercontent.com/kchua/handful-of-trials/master/dmbrl/config/cartpole.py
#
# PETS ("Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics
# Models", Chua et al. NeurIPS 2018) is a model-based RL method: a bootstrap ensemble of
# probabilistic neural networks models the environment dynamics (predicting a Gaussian over
# next-state deltas), driving an MPC planner (CEM/random shooting). The official repo
# (kchua/handful-of-trials, despite the queue's "PyTorch" note -- verified by fetching the
# actual source: it is `tensorflow-gpu>=1.9.0` + `gpflow==1.1.0`, both TF1-only APIs
# (`tf.Session`, `tf.get_variable`, `tf.variable_scope`) removed from modern TensorFlow) and
# cannot be reasonably installed alongside this repo's torch stack. The learnable-NN
# architecture (not the MPC/CEM/TSinf trajectory-sampling controller, which is algorithm
# plumbing, not the network) is faithfully transcribed here into base-env torch.
#
# Ported faithfully from `dmbrl/modeling/models/BNN.py`'s `BNN` class (the paper's
# Probabilistic-Ensemble ("PE") model, the repo's flagship/default model type) and
# `dmbrl/modeling/layers/FC.py`'s `FC` layer:
#  - Each of the `num_networks` ensemble members is a fully-connected stack of batched-linear
#    layers (`FC.compute_output_tensor`'s `einsum("ij,ajk->aik", x, weights) + biases`, i.e. one
#    independent weight matrix per ensemble member, computed as one batched matmul -- ported
#    here as `nn.Parameter` tensors of shape `(ensemble_size, in, out)` and `torch.baddbmm`,
#    the same batched-linear-per-ensemble-member computation).
#  - Swish activations (`x * sigmoid(x)`, per `FC._activations["swish"]`) between hidden
#    layers; the final layer has its activation removed (`BNN.finalize`'s
#    `self.layers[-1].unset_activation()`) and its output width doubled to emit both a mean
#    and a log-variance per output dimension (`BNN.finalize`: "Add variance output").
#  - `max_logvar`/`min_logvar` learnable scalars-per-output-dim that softly clamp the
#    predicted log-variance (`BNN._compile_outputs`'s two-sided softplus clamp), exactly as in
#    the source.
# Hyperparameters (hidden width 500, 3 hidden layers, `MODEL_IN=6`, `MODEL_OUT=4`) taken from
# `dmbrl/config/cartpole.py`'s `nn_constructor`, the repo's own default/flagship config;
# `num_networks=5`, the ensemble size used throughout the PETS paper's PE-TSinf configuration.
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class EnsembleFC(nn.Module):
    """Faithful port of `FC` in dmbrl/modeling/layers/FC.py: one fully-connected layer applied
    independently per ensemble member via a single batched matmul
    (`einsum("ij,ajk->aik", input, weights) + biases` in the source, `torch.baddbmm` here),
    with weights truncated-normal initialized at std `1/(2*sqrt(input_dim))` matching
    `FC.construct_vars`'s `tf.truncated_normal_initializer`.
    """

    def __init__(
        self, input_dim: int, output_dim: int, ensemble_size: int, activation: str | None = "swish"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size
        self.activation = activation

        self.weights = nn.Parameter(torch.empty(ensemble_size, input_dim, output_dim))
        self.biases = nn.Parameter(torch.zeros(ensemble_size, 1, output_dim))
        nn.init.trunc_normal_(self.weights, std=1.0 / (2.0 * (input_dim**0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (ensemble_size, batch, input_dim)
        raw_output = torch.baddbmm(self.biases, x, self.weights)
        if self.activation == "swish":
            return raw_output * torch.sigmoid(raw_output)
        return raw_output


class ProbabilisticEnsembleBNN(nn.Module):
    """Faithful port of `BNN` in dmbrl/modeling/models/BNN.py, configured as
    `dmbrl/config/cartpole.py`'s `nn_constructor` builds it: 3 hidden `EnsembleFC` layers of
    width `hidden_dim` with swish activations, then one `EnsembleFC` output layer (no
    activation, per `BNN.finalize`'s `unset_activation()`) emitting `2 * output_dim` values
    (mean, log-variance), with the log-variance passed through the learnable two-sided
    softplus clamp from `BNN._compile_outputs`.
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 4,
        hidden_dim: int = 500,
        num_hidden_layers: int = 3,
        ensemble_size: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size

        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(EnsembleFC(in_dim, hidden_dim, ensemble_size, activation="swish"))
            in_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)
        # Final layer: no activation, output doubled for mean + logvar (BNN.finalize).
        self.output_layer = EnsembleFC(in_dim, 2 * output_dim, ensemble_size, activation=None)

        self.max_logvar = nn.Parameter(torch.ones(1, output_dim) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, output_dim) * 10.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (ensemble_size, batch, input_dim) -- mirrors BNN._compile_outputs's cur_out.
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        cur_out = self.output_layer(x)

        mean = cur_out[:, :, : self.output_dim]
        raw_logvar = cur_out[:, :, self.output_dim :]

        # Two-sided softplus clamp, exactly as BNN._compile_outputs.
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return torch.cat([mean, logvar], dim=-1)


# ---------------------------------------------------------------------------
# Menagerie staging entrypoints.
# ---------------------------------------------------------------------------


def build_pets():
    torch.manual_seed(0)
    # dmbrl/config/cartpole.py: MODEL_IN=6, MODEL_OUT=4, 3 hidden layers of width 500;
    # num_networks=5 (PETS paper's flagship PE-TSinf ensemble size).
    return ProbabilisticEnsembleBNN(
        input_dim=6, output_dim=4, hidden_dim=500, num_hidden_layers=3, ensemble_size=5
    )


def example_input_pets():
    torch.manual_seed(0)
    ensemble_size, batch, input_dim = 5, 8, 6
    return (torch.randn(ensemble_size, batch, input_dim),)


MENAGERIE_ENTRIES = [
    (
        "PETS (probabilistic ensemble dynamics)",
        "build_pets",
        "example_input_pets",
        2018,
        MENAGERIE_ZOO,
    ),
]
