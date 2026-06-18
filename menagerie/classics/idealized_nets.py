"""Idealized random-feature and infinite-width neural nets, 2006-2018.

Paper: Rahimi and Recht 2007, "Random Features for Large-Scale Kernel
Machines"; Jacot, Gabriel, and Hongler 2018, "Neural Tangent Kernel".
These small modules preserve the core forward-pass parameterizations while
omitting training solvers except for traceable inference paths.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RandomFeaturesNet(nn.Module):
    """Random Fourier features with a trainable linear readout."""

    def __init__(
        self, d_in: int = 10, n_features: int = 24, d_out: int = 3, gamma: float = 0.5
    ) -> None:
        """Initialize frozen RBF-kernel Fourier features.

        Parameters
        ----------
        d_in
            Input dimensionality.
        n_features
            Number of random Fourier features.
        d_out
            Output dimensionality.
        gamma
            RBF bandwidth coefficient.
        """
        super().__init__()
        self.register_buffer("omega", torch.randn(d_in, n_features) * math.sqrt(2.0 * gamma))
        self.register_buffer("phase", torch.rand(n_features) * 2.0 * math.pi)
        self.readout = nn.Linear(n_features, d_out)

    def forward(self, x: Tensor) -> Tensor:
        """Compute random Fourier features and readout.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Readout tensor.
        """
        features = math.sqrt(2.0 / self.omega.shape[1]) * torch.cos(x @ self.omega + self.phase)
        return self.readout(features)


class ExtremeLearningMachine(nn.Module):
    """Frozen random hidden layer with trainable linear readout."""

    def __init__(self, d_in: int = 10, n_hidden: int = 18, d_out: int = 3) -> None:
        """Initialize frozen hidden features and readout.

        Parameters
        ----------
        d_in
            Input dimensionality.
        n_hidden
            Number of frozen hidden units.
        d_out
            Output dimensionality.
        """
        super().__init__()
        self.hidden = nn.Linear(d_in, n_hidden)
        for parameter in self.hidden.parameters():
            parameter.requires_grad_(False)
        self.readout = nn.Linear(n_hidden, d_out, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Compute ELM hidden activations and readout.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Readout tensor.
        """
        return self.readout(torch.tanh(self.hidden(x)))


class DeepLinearNetwork(nn.Module):
    """Bias-free deep linear chain."""

    def __init__(self, dims: tuple[int, ...] = (12, 10, 8, 4)) -> None:
        """Initialize bias-free linear layers.

        Parameters
        ----------
        dims
            Layer widths.
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in zip(dims, dims[1:])])

    def forward(self, x: Tensor) -> Tensor:
        """Apply the linear chain.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, dims[0])``.

        Returns
        -------
        Tensor
            Network output.
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class NTKParameterizedNet(nn.Module):
    """Wide MLP using explicit NTK-style fan-in scaling."""

    def __init__(self, d_in: int = 10, width: int = 32, depth: int = 3, d_out: int = 2) -> None:
        """Initialize NTK-parameterized weights.

        Parameters
        ----------
        d_in
            Input dimensionality.
        width
            Hidden width.
        depth
            Number of hidden layers.
        d_out
            Output dimensionality.
        """
        super().__init__()
        dims = (d_in, *([width] * depth), d_out)
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(b, a)) for a, b in zip(dims, dims[1:])]
        )
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(b)) for b in dims[1:]])

    def forward(self, x: Tensor) -> Tensor:
        """Apply an NTK-scaled MLP.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Network output.
        """
        out = x
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            out = out @ weight.T / math.sqrt(weight.shape[1]) + bias
            if index < len(self.weights) - 1:
                out = torch.relu(out)
        return out


class RVFLNet(nn.Module):
    """Random Vector Functional Link net with direct input skip."""

    def __init__(self, d_in: int = 10, n_hidden: int = 18, d_out: int = 3) -> None:
        """Initialize frozen random hidden features and skip readout.

        Parameters
        ----------
        d_in
            Input dimensionality.
        n_hidden
            Number of random hidden units.
        d_out
            Output dimensionality.
        """
        super().__init__()
        self.hidden = nn.Linear(d_in, n_hidden)
        for parameter in self.hidden.parameters():
            parameter.requires_grad_(False)
        self.readout = nn.Linear(d_in + n_hidden, d_out)

    def forward(self, x: Tensor) -> Tensor:
        """Compute random hidden features, concatenate the input, and read out.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Readout tensor.
        """
        hidden = torch.relu(self.hidden(x))
        return self.readout(torch.cat((x, hidden), dim=-1))


class MuPMLP(nn.Module):
    """Maximal-update-parameterized MLP with width-scaled hidden updates."""

    def __init__(self, d_in: int = 10, width: int = 32, d_out: int = 2) -> None:
        """Initialize a small muP-style MLP.

        Parameters
        ----------
        d_in
            Input dimensionality.
        width
            Hidden width.
        d_out
            Output dimensionality.
        """
        super().__init__()
        self.input = nn.Linear(d_in, width)
        self.hidden = nn.Linear(width, width)
        self.output = nn.Linear(width, d_out)
        self.width = width

    def forward(self, x: Tensor) -> Tensor:
        """Apply width-normalized hidden and output transformations.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Network output.
        """
        h = torch.relu(self.input(x))
        h = torch.relu(self.hidden(h) / self.width)
        return self.output(h) / self.width


class NNGPFiniteWidthProxyNet(nn.Module):
    """Finite random-feature proxy for an NNGP kernel draw."""

    def __init__(self, d_in: int = 10, width: int = 64, d_out: int = 2) -> None:
        """Initialize frozen Gaussian-process feature weights.

        Parameters
        ----------
        d_in
            Input dimensionality.
        width
            Number of finite-width random features.
        d_out
            Output dimensionality.
        """
        super().__init__()
        self.register_buffer("hidden_weight", torch.randn(width, d_in) / math.sqrt(d_in))
        self.register_buffer("readout_weight", torch.randn(d_out, width) / math.sqrt(width))

    def forward(self, x: Tensor) -> Tensor:
        """Compute a finite-width NNGP sample proxy.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Random-function output.
        """
        features = torch.relu(x @ self.hidden_weight.T)
        return features @ self.readout_weight.T


class MeanFieldTwoLayerNet(nn.Module):
    """Mean-field two-layer net averaging many neuron contributions."""

    def __init__(self, d_in: int = 10, width: int = 40, d_out: int = 2) -> None:
        """Initialize mean-field first-layer particles and coefficients.

        Parameters
        ----------
        d_in
            Input dimensionality.
        width
            Number of hidden particles.
        d_out
            Output dimensionality.
        """
        super().__init__()
        self.features = nn.Linear(d_in, width)
        self.coefficients = nn.Parameter(torch.randn(width, d_out) * 0.1)
        self.width = width

    def forward(self, x: Tensor) -> Tensor:
        """Average hidden particle contributions.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Mean-field output.
        """
        hidden = torch.relu(self.features(x))
        return hidden @ self.coefficients / self.width


class BarronRandomBasisNet(nn.Module):
    """Barron-space random basis network with bounded readout."""

    def __init__(self, d_in: int = 10, n_basis: int = 28, d_out: int = 2) -> None:
        """Initialize random cosine bases and l1-normalized coefficients.

        Parameters
        ----------
        d_in
            Input dimensionality.
        n_basis
            Number of random basis functions.
        d_out
            Output dimensionality.
        """
        super().__init__()
        self.register_buffer("basis_weight", torch.randn(n_basis, d_in))
        self.register_buffer("basis_bias", torch.rand(n_basis) * 2.0 * math.pi)
        self.coefficients = nn.Parameter(torch.randn(n_basis, d_out) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate random Barron bases and normalized coefficients.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, d_in)``.

        Returns
        -------
        Tensor
            Basis expansion output.
        """
        basis = torch.cos(x @ self.basis_weight.T + self.basis_bias)
        coeff = self.coefficients / self.coefficients.abs().sum(dim=0, keepdim=True).clamp_min(1e-6)
        return basis @ coeff


def _example_features() -> Tensor:
    """Return a shared dense feature example.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return torch.randn(2, 10)


def build_random_features() -> nn.Module:
    """Build a random Fourier features net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return RandomFeaturesNet()


def example_input_random_features() -> Tensor:
    """Return an example random-features input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return _example_features()


def build_elm() -> nn.Module:
    """Build an extreme learning machine.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return ExtremeLearningMachine()


def example_input_elm() -> Tensor:
    """Return an example ELM input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return _example_features()


def build_deep_linear() -> nn.Module:
    """Build a deep linear network.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return DeepLinearNetwork()


def example_input_deep_linear() -> Tensor:
    """Return an example deep-linear input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return torch.randn(2, 12)


def build_ntk_parameterized() -> nn.Module:
    """Build an NTK-parameterized wide net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return NTKParameterizedNet()


def example_input_ntk_parameterized() -> Tensor:
    """Return an example NTK net input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return _example_features()


def build_rvfl() -> nn.Module:
    """Build a Random Vector Functional Link net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return RVFLNet()


def example_input_rvfl() -> Tensor:
    """Return an example RVFL input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return _example_features()


def build_mup_mlp() -> nn.Module:
    """Build a muP-style MLP.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return MuPMLP()


def example_input_mup_mlp() -> Tensor:
    """Return an example muP MLP input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return _example_features()


def build_nngp_proxy() -> nn.Module:
    """Build an NNGP finite-width proxy net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return NNGPFiniteWidthProxyNet()


def example_input_nngp_proxy() -> Tensor:
    """Return an example NNGP proxy input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return _example_features()


def build_mean_field_two_layer() -> nn.Module:
    """Build a mean-field two-layer net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return MeanFieldTwoLayerNet()


def example_input_mean_field_two_layer() -> Tensor:
    """Return an example mean-field input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return _example_features()


def build_barron_random_basis() -> nn.Module:
    """Build a Barron random-basis net.

    Returns
    -------
    nn.Module
        Configured module.
    """
    return BarronRandomBasisNet()


def example_input_barron_random_basis() -> Tensor:
    """Return an example Barron basis input.

    Returns
    -------
    Tensor
        Example tensor.
    """
    return _example_features()


MENAGERIE_ENTRIES = [
    (
        "Random Features net (Random Kitchen Sinks)",
        "build_random_features",
        "example_input_random_features",
        "2007",
        "CH-A",
    ),
    ("Extreme Learning Machine (ELM)", "build_elm", "example_input_elm", "2006", "CH-A"),
    (
        "Deep linear network (Saxe-McClelland-Ganguli)",
        "build_deep_linear",
        "example_input_deep_linear",
        "2014",
        "CH-A",
    ),
    (
        "NTK-parameterized wide net",
        "build_ntk_parameterized",
        "example_input_ntk_parameterized",
        "2018",
        "CH-A",
    ),
    (
        "Random Vector Functional Link (RVFL) net",
        "build_rvfl",
        "example_input_rvfl",
        "1994",
        "CH-A",
    ),
    (
        "muP / maximal-update-parameterized MLP",
        "build_mup_mlp",
        "example_input_mup_mlp",
        "2021",
        "CH-A",
    ),
    ("NNGP finite-width proxy net", "build_nngp_proxy", "example_input_nngp_proxy", "2018", "CH-A"),
    (
        "mean-field two-layer net",
        "build_mean_field_two_layer",
        "example_input_mean_field_two_layer",
        "2018",
        "CH-A",
    ),
    (
        "Barron random-basis net",
        "build_barron_random_basis",
        "example_input_barron_random_basis",
        "1993",
        "CH-A",
    ),
]
