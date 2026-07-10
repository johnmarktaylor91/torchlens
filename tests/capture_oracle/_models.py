"""Deterministic model zoo used by the capture characterization oracle."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PlainCNN(nn.Module):
    """Small convolutional network for ordinary feed-forward capture."""

    def __init__(self) -> None:
        """Initialize the convolution and side-effect counters."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.forward_invocations = 0
        self.pre_hook_invocations = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one convolution followed by ReLU.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Convolved activation.
        """

        self.forward_invocations += 1
        return torch.relu(self.conv(x))


class TrainBatchNorm(nn.Module):
    """Train-mode BatchNorm model whose running buffers mutate per forward."""

    def __init__(self) -> None:
        """Initialize BatchNorm and side-effect counters."""

        super().__init__()
        self.bn = nn.BatchNorm2d(2)
        self.forward_invocations = 0
        self.pre_hook_invocations = 0
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize an image batch in training mode.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Normalized activation.
        """

        self.forward_invocations += 1
        return torch.relu(self.bn(x))


class RecurrentModel(nn.Module):
    """Tiny recurrent model that calls one module three times."""

    def __init__(self) -> None:
        """Initialize the recurrent cell and side-effect counters."""

        super().__init__()
        self.cell = nn.Linear(4, 4)
        self.forward_invocations = 0
        self.pre_hook_invocations = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared cell for three recurrent steps.

        Parameters
        ----------
        x:
            Initial recurrent state.

        Returns
        -------
        torch.Tensor
            Final recurrent state.
        """

        self.forward_invocations += 1
        state = x
        for _ in range(3):
            state = torch.tanh(self.cell(state))
        return state


class ConditionalModel(nn.Module):
    """Model with tensor-data-dependent control flow and RNG consumption."""

    def __init__(self) -> None:
        """Initialize side-effect counters."""

        super().__init__()
        self.forward_invocations = 0
        self.pre_hook_invocations = 0
        self.rng_draw_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Choose a branch from live tensor data and consume one RNG draw.

        Parameters
        ----------
        x:
            Branch-control input.

        Returns
        -------
        torch.Tensor
            Result of the selected branch.
        """

        self.forward_invocations += 1
        random_offset = torch.rand((), device=x.device)
        self.rng_draw_count += 1
        shifted = x + random_offset
        if bool((x.sum() > 0).item()):
            return torch.sigmoid(shifted)
        return torch.tanh(shifted)


class InPlaceModel(nn.Module):
    """Model with explicit in-place operations on an internal tensor."""

    def __init__(self) -> None:
        """Initialize side-effect counters."""

        super().__init__()
        self.forward_invocations = 0
        self.pre_hook_invocations = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clone, add, and rectify in place.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Mutated internal tensor.
        """

        self.forward_invocations += 1
        result = x.clone()
        result.add_(2.0)
        return result.relu_()


class MutatingPreHookModel(nn.Module):
    """Model with a root forward pre-hook that mutates its tensor input."""

    def __init__(self) -> None:
        """Initialize the layer, counters, and mutating pre-hook."""

        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.forward_invocations = 0
        self.pre_hook_invocations = 0
        self.register_forward_pre_hook(self._mutating_pre_hook)

    def _mutating_pre_hook(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Mutate the first tensor argument before forward.

        Parameters
        ----------
        _module:
            Hooked module, unused.
        args:
            Positional forward arguments.

        Returns
        -------
        tuple[Any, ...]
            Forward arguments containing the mutated tensor.
        """

        self.pre_hook_invocations += 1
        tensor = args[0]
        if isinstance(tensor, torch.Tensor):
            tensor.add_(0.5)
        return args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the pre-hook-mutated input.

        Parameters
        ----------
        x:
            Mutated input tensor.

        Returns
        -------
        torch.Tensor
            Projected activation.
        """

        self.forward_invocations += 1
        return torch.relu(self.linear(x))


class TinyTransformer(nn.Module):
    """Minimal Transformer encoder layer with deterministic dropout settings."""

    def __init__(self) -> None:
        """Initialize the encoder and side-effect counters."""

        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=8,
            dropout=0.0,
            batch_first=True,
        )
        self.forward_invocations = 0
        self.pre_hook_invocations = 0
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a short sequence.

        Parameters
        ----------
        x:
            Sequence batch.

        Returns
        -------
        torch.Tensor
            Encoded sequence.
        """

        self.forward_invocations += 1
        return self.encoder(x)


class FailingConditionalModel(nn.Module):
    """Conditional model that raises after producing capturable operations."""

    def __init__(self) -> None:
        """Initialize side-effect counters."""

        super().__init__()
        self.forward_invocations = 0
        self.pre_hook_invocations = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a branch and then raise a stable user exception.

        Parameters
        ----------
        x:
            Branch-control input.

        Returns
        -------
        torch.Tensor
            This function never returns.

        Raises
        ------
        RuntimeError
            Always, after the selected branch has executed.
        """

        self.forward_invocations += 1
        shifted = x + 1.0
        if bool((x.sum() > 0).item()):
            shifted = torch.relu(shifted)
        raise RuntimeError("capture-oracle intentional forward failure")


def build_model_case(model_axis: str) -> tuple[nn.Module, torch.Tensor]:
    """Build one deterministic model and input pair.

    Parameters
    ----------
    model_axis:
        Model-axis identifier from the oracle matrix.

    Returns
    -------
    tuple[nn.Module, torch.Tensor]
        Fresh model and input objects.

    Raises
    ------
    KeyError
        If the model-axis identifier is unknown.
    """

    if model_axis == "plain_cnn":
        return PlainCNN(), torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)
    if model_axis == "train_batchnorm":
        return TrainBatchNorm(), torch.linspace(-1.0, 1.0, 64).reshape(2, 2, 4, 4)
    if model_axis == "recurrent":
        return RecurrentModel(), torch.linspace(-1.0, 1.0, 8).reshape(2, 4)
    if model_axis == "conditional":
        return ConditionalModel(), torch.tensor([[1.0, -0.25], [0.5, 0.25]])
    if model_axis == "in_place":
        return InPlaceModel(), torch.linspace(-1.0, 1.0, 8).reshape(2, 4)
    if model_axis == "mutating_pre_hook":
        return MutatingPreHookModel(), torch.linspace(-1.0, 1.0, 8).reshape(2, 4)
    if model_axis == "tiny_transformer":
        return TinyTransformer(), torch.linspace(-1.0, 1.0, 24).reshape(2, 3, 4)
    if model_axis == "failing_conditional":
        return FailingConditionalModel(), torch.tensor([[1.0, -0.25], [0.5, 0.25]])
    raise KeyError(f"unknown capture-oracle model axis: {model_axis}")
