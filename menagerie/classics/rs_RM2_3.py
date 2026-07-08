# SOURCE: vendored from ThGaskin/NeuralABM @ main (include/neural_net.py)
# (only import path adjusted -- the `from .utils import random_tensor` import was
#  dropped since it is only reached when a `prior` config is supplied, which we
#  do not use here; no architecture change)
#
# NeuralABM -- "Neural parameter calibration for multi-agent models" (Gaskin,
# Pavliotis, Girolami; Nature Communications 2023-adjacent line of work).
# https://github.com/ThGaskin/NeuralABM
# `FeedForwardNN` is the generic configurable feed-forward neural surrogate used
# throughout the repo's ABM case studies (SIR, Covid, Kuramoto, HarrisWilson) to
# learn a differentiable mapping from ABM parameters to simulation summary
# statistics, which is then used for Bayesian/gradient-based ABM calibration --
# i.e. exactly the "ABM calibration neural surrogate" architecture family.
from typing import Any, List, Union

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ----------------------------------------------------------------------------------------------------------------------
# -- NN utility functions ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def sigmoid(beta=torch.tensor(1.0)):
    """Extends the torch.nn.sigmoid activation function by allowing for a slope parameter."""

    return lambda x: torch.sigmoid(beta * x)


# Pytorch activation functions.
# Pairs of activation functions and whether they are part of the torch.nn module, in which case they must be called
# via func(*args, **kwargs)(x).


ACTIVATION_FUNCS = {
    "abs": [torch.abs, False],
    "celu": [torch.nn.CELU, True],
    "cos": [torch.cos, False],
    "cosine": [torch.cos, False],
    "elu": [torch.nn.ELU, True],
    "gelu": [torch.nn.GELU, True],
    "hardshrink": [torch.nn.Hardshrink, True],
    "hardsigmoid": [torch.nn.Hardsigmoid, True],
    "hardswish": [torch.nn.Hardswish, True],
    "hardtanh": [torch.nn.Hardtanh, True],
    "leakyrelu": [torch.nn.LeakyReLU, True],
    "linear": [None, False],
    "logsigmoid": [torch.nn.LogSigmoid, True],
    "mish": [torch.nn.Mish, True],
    "prelu": [torch.nn.PReLU, True],
    "relu": [torch.nn.ReLU, True],
    "rrelu": [torch.nn.RReLU, True],
    "selu": [torch.nn.SELU, True],
    "sigmoid": [sigmoid, True],
    "silu": [torch.nn.SiLU, True],
    "sin": [torch.sin, False],
    "sine": [torch.sin, False],
    "softplus": [torch.nn.Softplus, True],
    "softshrink": [torch.nn.Softshrink, True],
    "softsign": [torch.nn.Softsign, True],
    "swish": [torch.nn.SiLU, True],
    "tanh": [torch.nn.Tanh, True],
    "tanhshrink": [torch.nn.Tanhshrink, True],
    "threshold": [torch.nn.Threshold, True],
}

OPTIMIZERS = {
    "Adagrad": torch.optim.Adagrad,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SparseAdam": torch.optim.SparseAdam,
    "Adamax": torch.optim.Adamax,
    "ASGD": torch.optim.ASGD,
    "LBFGS": torch.optim.LBFGS,
    "NAdam": torch.optim.NAdam,
    "RAdam": torch.optim.RAdam,
    "RMSprop": torch.optim.RMSprop,
    "Rprop": torch.optim.Rprop,
    "SGD": torch.optim.SGD,
}


def get_architecture(input_size: int, output_size: int, n_layers: int, cfg: dict) -> List[int]:
    # Apply default to all hidden layers
    _nodes = [cfg.get("default")] * n_layers

    # Update layer-specific settings
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_size in _layer_specific.items():
        _nodes[layer_id] = layer_size

    return [input_size] + _nodes + [output_size]


def get_single_layer_func(layer_cfg: Union[str, dict]) -> callable:
    """Return the activation function from an entry for a single layer"""

    # Entry is a single string
    if isinstance(layer_cfg, str):
        _f = ACTIVATION_FUNCS[layer_cfg.lower()]
        if _f[1]:
            return _f[0]()
        else:
            return _f[0]

    # Entry is a dictionary containing args and kwargs
    elif isinstance(layer_cfg, dict):
        _f = ACTIVATION_FUNCS[layer_cfg.get("name").lower()]
        if _f[1]:
            return _f[0](*layer_cfg.get("args", ()), **layer_cfg.get("kwargs", {}))
        else:
            return _f[0]

    elif layer_cfg is None:
        _f = ACTIVATION_FUNCS["linear"][0]

    else:
        raise ValueError(f"Unrecognized activation function {layer_cfg}!")


def get_activation_funcs(n_layers: int, cfg: dict) -> List[callable]:
    """Extracts the activation functions from the config. The config is a dictionary containing the
    default activation function, and a layer-specific entry detailing exceptions from the default. 'None' entries
    are interpreted as linear layers.

    .. Example:
        activation_funcs:
          default: relu
          layer_specific:
            0: ~
            2: tanh
            3:
              name: HardTanh
              args:
                - -2  # min_value
                - +2  # max_value
    """

    # Use default activation function on all layers
    _funcs = [get_single_layer_func(cfg.get("default"))] * (n_layers + 1)

    # Change activation functions on specified layers
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_cfg in _layer_specific.items():
        _funcs[layer_id] = get_single_layer_func(layer_cfg)

    return _funcs


def get_bias(n_layers: int, cfg: dict) -> List[Any]:
    """Extracts the bias initialisation settings from the config. The config is a dictionary containing the
    default, and a layer-specific entry detailing exceptions from the default. 'None' entries
    are interpreted as unbiased layers.

    .. Example:
        biases:
          default: ~
          layer_specific:
            0: [-1, 1]
            3: [2, 3]
    """

    # Use the default value on all layers
    biases = [cfg.get("default")] * (n_layers + 1)

    # Amend bias on specified layers
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_bias in _layer_specific.items():
        biases[layer_id] = layer_bias

    return biases


# -----------------------------------------------------------------------------
# -- Neural net class ---------------------------------------------------------
# -----------------------------------------------------------------------------
class BaseNN(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: dict,
        activation_funcs: dict,
        biases: dict,
        optimizer: str = "Adam",
        learning_rate: float = 0.002,
        optimizer_kwargs: dict = {},
        **__,
    ):
        """Base neural network architecture class.

        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: a dictionary specifying the number of nodes per layer
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param biases: a dictionary containing the initialisation parameters for the bias
        :param prior (optional): initial prior distribution of the parameters. If given, the neural net will
            initially output a random value within that distribution.
        :param prior_tol (optional): the tolerance with which the prior distribution should be met
        :param prior_max_iter (optional): maximum number of training iterations to hit the prior target
        :param optimizer: the name of the optimizer to use. Default is the torch.optim.Adam optimizer.
        :param learning_rate: the learning rate of the optimizer. Default is 1e-3.
        :param __: Additional model parameters (ignored)
        """

        super().__init__()
        self.flatten = nn.Flatten()

        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = num_layers

        # Get architecture, activation functions, and layer bias
        self.architecture = get_architecture(input_size, output_size, num_layers, nodes_per_layer)
        self.activation_funcs = get_activation_funcs(num_layers, activation_funcs)
        self.bias = get_bias(num_layers, biases)

        # Add the neural net layers
        self.layers = nn.ModuleList()
        for i in range(len(self.architecture) - 1):
            layer = nn.Linear(
                self.architecture[i],
                self.architecture[i + 1],
                bias=self.bias[i] is not None,
            )

            # Initialise the biases of the layers with a uniform distribution
            if self.bias[i] is not None:
                # Use the pytorch default if indicated
                if self.bias[i] == "default":
                    pass

                # Initialise the bias on explicitly provided intervals
                else:
                    with torch.no_grad():
                        torch.nn.init.uniform_(layer.bias, a=self.bias[i][0], b=self.bias[i][1])

            self.layers.append(layer)

        # Get the optimizer
        self.optimizer = OPTIMIZERS[optimizer](
            self.parameters(), lr=learning_rate, **optimizer_kwargs
        )


class FeedForwardNN(BaseNN):
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: dict,
        activation_funcs: dict,
        biases: dict,
        prior: Union[list, dict] = None,
        prior_max_iter: int = 500,
        prior_tol: float = 1e-5,
        optimizer: str = "Adam",
        learning_rate: float = 0.002,
        optimizer_kwargs: dict = {},
        **__,
    ):
        """Standard feed-forward architecture neural network class.

        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: a dictionary specifying the number of nodes per layer
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param biases: a dictionary containing the initialisation parameters for the bias
        :param prior (optional): initial prior distribution of the parameters. If given, the neural net will
            initially output a random value within that distribution.
        :param prior_tol (optional): the tolerance with which the prior distribution should be met
        :param prior_max_iter (optional): maximum number of training iterations to hit the prior target
        :param optimizer: the name of the optimizer to use. Default is the torch.optim.Adam optimizer.
        :param learning_rate: the learning rate of the optimizer. Default is 1e-3.
        :param __: Additional model parameters (ignored)
        """

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            num_layers=num_layers,
            nodes_per_layer=nodes_per_layer,
            activation_funcs=activation_funcs,
            biases=biases,
            optimizer=optimizer,
            learning_rate=learning_rate,
            optimizer_kwargs=optimizer_kwargs,
        )

        # Get the initial distribution and initialise
        self.prior_distribution = prior
        self.initialise_to_prior(tol=prior_tol, max_iter=prior_max_iter)

    def initialise_to_prior(self, *, tol: float = 1e-5, max_iter: int = 500) -> None:
        """Initialises the neural net to output values following a prior distribution. The random tensor is drawn
        following a prior distribution and the neural network trained to output that value. Training is performed
        until the neural network output matches the drawn value (which typically only takes a few seconds), or until
        a maximum iteration count is reached.

        :param tol: the target error on the neural net initial output and drawn value.
        :param max_iter: maximum number of training steps to perform in the while loop
        """

        # If not initial distribution is given, nothing happens
        if self.prior_distribution is None:
            return

        # NOTE: the real repo draws `target` via a `random_tensor(prior_distribution, ...)` utility here
        # (include/utils.py) that is not vendored since this staging build always uses prior=None.
        raise NotImplementedError(
            "prior-distribution initialisation requires include/utils.py:random_tensor, "
            "not vendored here; construct with prior=None."
        )

    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
        for i in range(len(self.layers)):
            if self.activation_funcs[i] is None:
                x = self.layers[i](x)
            else:
                x = self.activation_funcs[i](self.layers[i](x))
        return x


# ---------------------------------------------------------------------------
# menagerie staging glue
# ---------------------------------------------------------------------------
def build_abm_surrogate() -> nn.Module:
    """Tiny FeedForwardNN: input=ABM parameter vector, output=predicted simulation
    summary statistics, matching the real repo's config-driven constructor exactly
    (prior=None -- the optional prior-initialisation feature is not exercised)."""
    return FeedForwardNN(
        input_size=4,
        output_size=3,
        num_layers=2,
        nodes_per_layer={"default": 8},
        activation_funcs={"default": "relu"},
        biases={"default": None},
        prior=None,
    )


def example_input_abm_surrogate() -> torch.Tensor:
    """Real model's forward() takes a batch of ABM parameter vectors."""
    return torch.randn(1, 4)


MENAGERIE_ENTRIES = [
    (
        "ABM Calibration Neural Surrogate",
        "build_abm_surrogate",
        "example_input_abm_surrogate",
        2022,
        MENAGERIE_ZOO,
    ),
]
