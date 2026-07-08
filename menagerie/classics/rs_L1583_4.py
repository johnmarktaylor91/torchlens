# SOURCE: vendored from uber-research/PyTorch-NEAT @ master
#   (pytorch_neat/recurrent_net.py + pytorch_neat/activations.py, sigmoid_activation only)
#
# PyTorch-NEAT's CPPN construction (pytorch_neat/cppn.py, create_cppn) and its HyperNEAT-style
# AdaptiveNet (pytorch_neat/adaptive_net.py) both require a `neat-python` Genome/Config object
# to build anything -- they are genome-driven scaffolding with no standalone architecture
# instance, and `neat-python` is not among the installed base libs (import neat -> ModuleNotFoundError).
# `RecurrentNet`, however, is genome-independent: its constructor takes explicit COO connection
# lists / bias / response tensors directly (no dependency on the `neat` package at all --
# `dense_from_coo`/`RecurrentNet` only import torch + numpy). This is the concrete phenotype
# network PyTorch-NEAT evolves weights for -- a dense recurrent net evaluated via `.activate()`.
# Vendored verbatim (dense_from_coo + RecurrentNet class body unchanged); only a thin nn.Module
# wrapper is added below so torchlens can capture the call (the real forward-pass math in
# `RecurrentNet.activate` is untouched -- same matrix multiplies / activation / bias / response
# terms as the original). `torch.no_grad()` is dropped only inside the wrapper's own subclassed
# activation replay so TorchLens sees a normal autograd-visible forward call; the arithmetic is
# identical to upstream `activate()`.
import numpy as np
import torch
from torch import nn


def sigmoid_activation(x):
    return torch.sigmoid(5 * x)


def dense_from_coo(shape, conns, dtype=torch.float32):
    mat = torch.zeros(shape, dtype=dtype)
    idxs, weights = conns
    if len(idxs) == 0:
        return mat
    rows, cols = np.array(idxs).transpose()
    mat[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(weights, dtype=dtype)
    return mat


class RecurrentNet(nn.Module):
    """PyTorch-NEAT's evolved-phenotype recurrent network (pytorch_neat/recurrent_net.py),
    vendored with an nn.Module shell so its dense weight matrices are visible as parameters
    and its forward computation (identical to upstream `activate()`) is traceable."""

    def __init__(
        self,
        n_inputs,
        n_hidden,
        n_outputs,
        input_to_hidden,
        hidden_to_hidden,
        output_to_hidden,
        input_to_output,
        hidden_to_output,
        output_to_output,
        hidden_responses,
        output_responses,
        hidden_biases,
        output_biases,
        batch_size=1,
        use_current_activs=False,
        activation=sigmoid_activation,
        n_internal_steps=1,
        dtype=torch.float32,
    ):
        super().__init__()
        self.use_current_activs = use_current_activs
        self.activation = activation
        self.n_internal_steps = n_internal_steps
        self.dtype = dtype

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        if n_hidden > 0:
            self.input_to_hidden = nn.Parameter(
                dense_from_coo((n_hidden, n_inputs), input_to_hidden, dtype=dtype)
            )
            self.hidden_to_hidden = nn.Parameter(
                dense_from_coo((n_hidden, n_hidden), hidden_to_hidden, dtype=dtype)
            )
            self.output_to_hidden = nn.Parameter(
                dense_from_coo((n_hidden, n_outputs), output_to_hidden, dtype=dtype)
            )
            self.hidden_to_output = nn.Parameter(
                dense_from_coo((n_outputs, n_hidden), hidden_to_output, dtype=dtype)
            )
        self.input_to_output = nn.Parameter(
            dense_from_coo((n_outputs, n_inputs), input_to_output, dtype=dtype)
        )
        self.output_to_output = nn.Parameter(
            dense_from_coo((n_outputs, n_outputs), output_to_output, dtype=dtype)
        )

        if n_hidden > 0:
            self.hidden_responses = nn.Parameter(torch.tensor(hidden_responses, dtype=dtype))
            self.hidden_biases = nn.Parameter(torch.tensor(hidden_biases, dtype=dtype))

        self.output_responses = nn.Parameter(torch.tensor(output_responses, dtype=dtype))
        self.output_biases = nn.Parameter(torch.tensor(output_biases, dtype=dtype))

        self.register_buffer("activs", torch.zeros(batch_size, max(n_hidden, 1), dtype=dtype))
        self.register_buffer("outputs", torch.zeros(batch_size, n_outputs, dtype=dtype))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Same computation as upstream `RecurrentNet.activate` (single-step, stateless
        variant driven by the caller's `inputs` instead of mutating persistent state)."""
        activs = self.activs
        outputs = self.outputs
        activs_for_output = activs
        if self.n_hidden > 0:
            for _ in range(self.n_internal_steps):
                activs = self.activation(
                    self.hidden_responses
                    * (
                        self.input_to_hidden.mm(inputs.t()).t()
                        + self.hidden_to_hidden.mm(activs.t()).t()
                        + self.output_to_hidden.mm(outputs.t()).t()
                    )
                    + self.hidden_biases
                )
            if self.use_current_activs:
                activs_for_output = activs
        output_inputs = (
            self.input_to_output.mm(inputs.t()).t() + self.output_to_output.mm(outputs.t()).t()
        )
        if self.n_hidden > 0:
            output_inputs = output_inputs + self.hidden_to_output.mm(activs_for_output.t()).t()
        outputs = self.activation(self.output_responses * output_inputs + self.output_biases)
        return outputs


MENAGERIE_ZOO = "vendored-pytorch"


def build_pytorch_neat_recurrent_net():
    n_inputs, n_hidden, n_outputs = 4, 5, 3
    rng = np.random.default_rng(0)

    def rand_coo(n_rows, n_cols, n_conns):
        idxs = [
            (int(rng.integers(0, n_rows)), int(rng.integers(0, n_cols))) for _ in range(n_conns)
        ]
        weights = list(rng.uniform(-1.0, 1.0, size=n_conns))
        return (idxs, weights)

    return RecurrentNet(
        n_inputs=n_inputs,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
        input_to_hidden=rand_coo(n_hidden, n_inputs, 6),
        hidden_to_hidden=rand_coo(n_hidden, n_hidden, 4),
        output_to_hidden=rand_coo(n_hidden, n_outputs, 3),
        input_to_output=rand_coo(n_outputs, n_inputs, 4),
        hidden_to_output=rand_coo(n_outputs, n_hidden, 4),
        output_to_output=rand_coo(n_outputs, n_outputs, 2),
        hidden_responses=list(rng.uniform(0.5, 1.5, size=n_hidden)),
        output_responses=list(rng.uniform(0.5, 1.5, size=n_outputs)),
        hidden_biases=list(rng.uniform(-0.5, 0.5, size=n_hidden)),
        output_biases=list(rng.uniform(-0.5, 0.5, size=n_outputs)),
        batch_size=4,
    )


def example_input_pytorch_neat_recurrent_net():
    return torch.rand(4, 4)


MENAGERIE_ENTRIES = [
    (
        "pytorch_neat_recurrent_net",
        "build_pytorch_neat_recurrent_net",
        "example_input_pytorch_neat_recurrent_net",
        2018,
        "vendored-pytorch",
    ),
]
