"""Checkpointing axis: ``torch.utils.checkpoint`` segments inside a captured forward.

Reentrant and non-reentrant checkpoints, a checkpoint nested inside another,
and a segment the hook options rows intervene inside (every segment holds a
linear, the op the hook rows in ``hooks.py`` aim at, so crossing a hook row
with these models puts the intervention inside the checkpointed segment).
The forward runs the segment once; its recompute happens only in backward,
which these rows do not run.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..generator import Inputs, ModelSpec

WIDTH = 8


class _Block(nn.Module):
    """Linear, relu, linear: the segment every row checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(WIDTH, WIDTH)
        self.outer = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outer(torch.relu(self.inner(x)))


class Checkpointed(nn.Module):
    """A stem, one checkpointed block, a head."""

    def __init__(self, *, use_reentrant: bool) -> None:
        super().__init__()
        self.stem = nn.Linear(WIDTH, WIDTH)
        self.block = _Block()
        self.head = nn.Linear(WIDTH, WIDTH)
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.stem(x))
        h = checkpoint(self.block, h, use_reentrant=self.use_reentrant)
        return self.head(h)


class Reentrant(Checkpointed):
    """The reentrant (autograd.Function) checkpoint."""

    def __init__(self) -> None:
        super().__init__(use_reentrant=True)


class NonReentrant(Checkpointed):
    """The non-reentrant (saved-tensor hooks) checkpoint."""

    def __init__(self) -> None:
        super().__init__(use_reentrant=False)


class NestedCheckpoint(nn.Module):
    """A non-reentrant checkpoint whose segment holds a second checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.first = _Block()
        self.second = _Block()

    def _outer_segment(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.first(x))
        return checkpoint(self.second, h, use_reentrant=False) + h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._outer_segment, x, use_reentrant=False)


def _input(_model: nn.Module) -> Inputs:
    return ((torch.randn(3, WIDTH),), {})


def _grad_input(_model: nn.Module) -> Inputs:
    return ((torch.randn(3, WIDTH).requires_grad_(True),), {})


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="ck_reentrant",
        build=Reentrant,
        example_input=_grad_input,
        reason="reentrant checkpoint segment; the input requires grad as reentrant needs",
        tags=frozenset({"checkpointing"}),
    ),
    ModelSpec(
        id="ck_nonreentrant",
        build=NonReentrant,
        example_input=_input,
        reason="non-reentrant checkpoint segment (saved-tensor hooks during forward)",
        tags=frozenset({"checkpointing"}),
    ),
    ModelSpec(
        id="ck_nested",
        build=NestedCheckpoint,
        example_input=_input,
        reason="a checkpoint nested inside another checkpointed segment",
        tags=frozenset({"checkpointing"}),
    ),
)
