"""Neuroevolution topology phenotypes, 2012-2019, WANN, CoDeepNEAT, ES-HyperNEAT, DENSER.

Paper: Risi 2012, "ES-HyperNEAT"; Miikkulainen 2017, "CoDeepNEAT"; Gaier 2019,
"Weight Agnostic Neural Networks"; Assuncao 2019, "DENSER." The evolutionary search,
speciation, CPPN expansion, and grammar mutation are omitted; each class is a fixed
representative differentiable phenotype.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class WeightAgnosticNetwork(nn.Module):
    """Fixed DAG where all enabled edges share one scalar weight."""

    def __init__(self, n_in: int = 4, n_hidden: int = 6, n_out: int = 3) -> None:
        """Initialize topology masks and output skip weights.

        Parameters
        ----------
        n_in
            Number of input features.
        n_hidden
            Number of hidden topology nodes.
        n_out
            Number of output features.
        """
        super().__init__()
        self.shared_weight = nn.Parameter(torch.tensor(0.7))
        self.input_mask = nn.Parameter(torch.randn(n_in, n_hidden) * 0.0 + 1.0)
        self.hidden_mask = nn.Parameter(torch.tril(torch.ones(n_hidden, n_hidden), diagonal=-1))
        self.output_mask = nn.Parameter(torch.randn(n_hidden, n_out) * 0.0 + 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the weight-agnostic topology.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, 4)``.

        Returns
        -------
        Tensor
            Output features.
        """
        hidden = torch.tanh(self.shared_weight * (x @ self.input_mask))
        for _ in range(2):
            hidden = torch.tanh(hidden + self.shared_weight * (hidden @ self.hidden_mask))
        return torch.tanh(self.shared_weight * (hidden @ self.output_mask))


class CoDeepNEAT(nn.Module):
    """Fixed blueprint assembled from evolved-module-style convolution blocks."""

    def __init__(self) -> None:
        """Initialize a representative CoDeepNEAT phenotype."""
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.module_a = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU())
        self.module_b = nn.Sequential(nn.Conv2d(8, 8, 1), nn.Tanh())
        self.head = nn.Linear(8, 10)

    def forward(self, image: Tensor) -> Tensor:
        """Run the assembled blueprint/module phenotype.

        Parameters
        ----------
        image
            Image tensor with shape ``(batch, 3, 32, 32)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        x = F.relu(self.stem(image))
        x = self.module_a(x) + self.module_b(x)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class ESHyperNEAT(nn.Module):
    """Coordinate-generated substrate weights from a small CPPN surrogate."""

    def __init__(self, n_in: int = 8, n_hidden: int = 10, n_out: int = 4) -> None:
        """Initialize substrate coordinates and CPPN weight generator.

        Parameters
        ----------
        n_in
            Number of substrate input nodes.
        n_hidden
            Number of hidden substrate nodes.
        n_out
            Number of output substrate nodes.
        """
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.cppn = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
        self.register_buffer("in_coords", torch.linspace(-1.0, 1.0, n_in).unsqueeze(-1))
        self.register_buffer("hid_coords", torch.linspace(-0.8, 0.8, n_hidden).unsqueeze(-1))
        self.register_buffer("out_coords", torch.linspace(-1.0, 1.0, n_out).unsqueeze(-1))

    def _weights(self, src: Tensor, dst: Tensor) -> Tensor:
        """Generate substrate edge weights from coordinate pairs.

        Parameters
        ----------
        src
            Source coordinates.
        dst
            Destination coordinates.

        Returns
        -------
        Tensor
            Generated weight matrix.
        """
        src_grid = src.repeat_interleave(dst.shape[0], dim=0)
        dst_grid = dst.repeat(src.shape[0], 1)
        radius = (src_grid - dst_grid).abs()
        inputs = torch.cat((src_grid, dst_grid, radius, torch.ones_like(radius)), dim=-1)
        return self.cppn(inputs).reshape(src.shape[0], dst.shape[0])

    def forward(self, x: Tensor) -> Tensor:
        """Run the coordinate-generated feedforward substrate.

        Parameters
        ----------
        x
            Input tensor with shape ``(batch, 8)``.

        Returns
        -------
        Tensor
            Output features.
        """
        hidden = torch.tanh(x @ self._weights(self.in_coords, self.hid_coords))
        return hidden @ self._weights(self.hid_coords, self.out_coords)


class DENSER(nn.Module):
    """Grammar-decoded representative CNN phenotype."""

    def __init__(self) -> None:
        """Initialize a fixed DSGE/grammar-decoded layer stack."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AvgPool2d(2),
            nn.Conv2d(8, 12, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Linear(12 * 8 * 8, 10)

    def forward(self, image: Tensor) -> Tensor:
        """Run the decoded DENSER phenotype.

        Parameters
        ----------
        image
            Image tensor with shape ``(batch, 3, 32, 32)``.

        Returns
        -------
        Tensor
            Class logits.
        """
        return self.classifier(self.features(image).flatten(1))


MENAGERIE_ENTRIES = [
    (
        "WANN (Weight-Agnostic Neural Network)",
        "build_wann",
        "example_input_wann",
        "2019",
        "DA",
    ),
    ("CoDeepNEAT", "build_codeepneat", "example_input_codeepneat", "2017", "DA"),
    ("ES-HyperNEAT", "build_es_hyperneat", "example_input_es_hyperneat", "2012", "DA"),
    ("DENSER", "build_denser", "example_input_denser", "2019", "DA"),
]


def build_wann() -> nn.Module:
    """Build a WANN phenotype.

    Returns
    -------
    nn.Module
        Configured WANN module.
    """
    return WeightAgnosticNetwork()


def example_input_wann() -> Tensor:
    """Create a WANN input example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 4)``.
    """
    return torch.randn(1, 4)


def build_codeepneat() -> nn.Module:
    """Build a CoDeepNEAT phenotype.

    Returns
    -------
    nn.Module
        Configured CoDeepNEAT module.
    """
    return CoDeepNEAT()


def example_input_codeepneat() -> Tensor:
    """Create a CoDeepNEAT image example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)


def build_es_hyperneat() -> nn.Module:
    """Build an ES-HyperNEAT phenotype.

    Returns
    -------
    nn.Module
        Configured ES-HyperNEAT module.
    """
    return ESHyperNEAT()


def example_input_es_hyperneat() -> Tensor:
    """Create an ES-HyperNEAT input example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 8)``.
    """
    return torch.randn(1, 8)


def build_denser() -> nn.Module:
    """Build a DENSER phenotype.

    Returns
    -------
    nn.Module
        Configured DENSER module.
    """
    return DENSER()


def example_input_denser() -> Tensor:
    """Create a DENSER image example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 3, 32, 32)``.
    """
    return torch.randn(1, 3, 32, 32)
