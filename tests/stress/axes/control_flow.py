"""Control-flow axis: ten synthetic models whose Python control flow depends on data or shape.

Each row is a ``ModelSpec``; add a model by appending a row with a one-line reason.
"""

from __future__ import annotations

import torch
from torch import nn

from ..generator import Inputs, ModelSpec

FLOAT_DTYPES = ("f32", "f16", "bf16", "f64")
WIDTH = 8


def _vector_input(batch: int = 3) -> Inputs:
    return ((torch.randn(batch, WIDTH),), {})


def _input_for(_model: nn.Module) -> Inputs:
    return _vector_input()


class DataDependentIf(nn.Module):
    """Branch on the sign of the input sum."""

    def __init__(self) -> None:
        super().__init__()
        self.pos = nn.Linear(WIDTH, WIDTH)
        self.neg = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return torch.relu(self.pos(x))
        return torch.tanh(self.neg(x))


class WhileOnItem(nn.Module):
    """Loop while a data-dependent scalar read through ``.item()`` stays above a threshold."""

    def __init__(self) -> None:
        super().__init__()
        self.step = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trips = 0
        while x.abs().mean().item() > 0.05 and trips < 6:
            x = self.step(x) * 0.5
            trips += 1
        return x


class Recursion(nn.Module):
    """Recursively halve the feature dimension until two features remain."""

    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Linear(WIDTH, WIDTH)

    def _fold(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] <= 2:
            return x
        half = x.shape[-1] // 2
        return self._fold(x[..., :half] + x[..., half : 2 * half])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fold(torch.relu(self.mix(x)))


class EarlyReturn(nn.Module):
    """Return before the second layer when the first activation is all non-positive."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(WIDTH, WIDTH)
        self.second = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.first(x))
        if bool((h <= 0).all()):
            return h
        return self.second(h)


class ShapeDependentTrips(nn.Module):
    """Run one step per pair of input features (trip count from the shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.step = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(x.shape[-1] // 2):
            x = torch.tanh(self.step(x))
        return x


class NestedIfElif(nn.Module):
    """Three-way branch on the input mean."""

    def __init__(self) -> None:
        super().__init__()
        self.low = nn.Linear(WIDTH, WIDTH)
        self.mid = nn.Linear(WIDTH, WIDTH)
        self.high = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        if mean < -0.1:
            return self.low(x)
        if mean < 0.1:
            return self.mid(x) + 1.0
        return torch.sigmoid(self.high(x))


class LoopWithBreak(nn.Module):
    """Fixed loop that breaks on a data-dependent condition."""

    def __init__(self) -> None:
        super().__init__()
        self.step = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(5):
            x = self.step(x)
            if x.norm() > 10.0:
                break
        return x


class RecurrentStepsFromItem(nn.Module):
    """Recurrent cell stepped a number of times read from the data."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.GRUCell(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps = int(x[0, 0].abs().item() * 3) + 1
        h = torch.zeros_like(x)
        for _ in range(steps):
            h = self.cell(x, h)
        return h


class ConditionalModuleSkip(nn.Module):
    """Skip a submodule entirely when a data-dependent gate is closed."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(WIDTH, 1)
        self.body = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate(x).mean() > 0:
            x = x + self.body(x)
        return x


class PythonIfVersusWhere(nn.Module):
    """Choose a layer with a Python ``if`` and blend with ``torch.where`` on the same data."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(WIDTH, WIDTH)
        self.b = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chosen = self.a(x) if x[0, 0] > 0 else self.b(x)
        return torch.where(x > 0, chosen, -chosen)


def _row(row_id: str, factory: type[nn.Module], reason: str) -> ModelSpec:
    return ModelSpec(
        id=row_id,
        build=factory,
        example_input=_input_for,
        reason=reason,
        dtypes=FLOAT_DTYPES,
        cost=1,
        tags=frozenset({"control_flow"}),
    )


MODELS: tuple[ModelSpec, ...] = (
    _row("cf_if_sign", DataDependentIf, "data-dependent if on a scalar comparison"),
    _row("cf_while_item", WhileOnItem, "while loop whose condition reads .item()"),
    _row("cf_recursion", Recursion, "recursive method with a shape-dependent base case"),
    _row("cf_early_return", EarlyReturn, "early return skips the second layer on some inputs"),
    _row("cf_shape_trips", ShapeDependentTrips, "for-loop trip count taken from the shape"),
    _row("cf_if_elif", NestedIfElif, "three-way if/elif/else on the input mean"),
    _row("cf_loop_break", LoopWithBreak, "fixed loop with a data-dependent break"),
    _row("cf_recurrent_item", RecurrentStepsFromItem, "GRU cell stepped .item() times"),
    _row("cf_module_skip", ConditionalModuleSkip, "submodule called zero or one times by a gate"),
    _row("cf_if_vs_where", PythonIfVersusWhere, "Python if choosing a layer, then torch.where"),
)
