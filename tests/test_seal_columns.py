"""The seal-pass column bodies of steps 2 and 6 equal the bodies they replace.

Each model is traced twice from one instance: once with the raw-cell column
bodies (``_postprocess._seal_columns`` for step 2, ``_buffer_fixup`` for step
6) and once with them declined, which runs the column-view or facade bodies
they replace. Every field the two steps write, and the graph fields they read,
must be equal, and the column bodies must actually run on the models they
cover.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._postprocess import _buffer_fixup, _seal_columns


class _Chain(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(12):
            x = torch.relu(x + 1)
        return x


class _TwoOutputs(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.sin(x)
        b = torch.cos(x)
        dead = torch.exp(x)  # reaches no output
        del dead
        return a * 2, a + b


class _Norms(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 1)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn2(torch.relu(self.bn1(self.conv(x))))


class _BufferReread(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = x * self.scale
        return x


class _BufferWrite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("state", torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.state[:2].copy_(x[:2])
        return self.state.sum() + x


_MODELS: dict[str, Callable[[], tuple[nn.Module, torch.Tensor]]] = {
    "chain": lambda: (_Chain(), torch.zeros(3)),
    "two_outputs": lambda: (_TwoOutputs(), torch.ones(3)),
    "norms": lambda: (_Norms().eval(), torch.randn(1, 2, 4, 4)),
    "buffer_reread": lambda: (_BufferReread(), torch.ones(3)),
    "buffer_write": lambda: (_BufferWrite(), torch.ones(3)),
}

#: Step 2's column body covers every model here. Step 6's covers a model whose
#: buffers are each read once from a registered address with no recorded
#: source, and declines a written buffer (it has a source to wire).
_STEP2_COVERED = set(_MODELS)
_STEP6_RUNS = {"norms"}
_STEP6_DECLINES = {"buffer_write"}


def _snapshot(trace: Any) -> dict[str, Any]:
    """Return every field steps 2 and 6 write, plus the graph they read."""

    ops = {}
    for op in trace:
        ops[op.label] = {
            "has_output_descendant": op.graph.has_output_descendant,
            "output_descendants": sorted(op.output_descendants),
            "input_ancestors": sorted(op.input_ancestors),
            "parents": list(op.parents),
            "children": list(op.children),
            "buffer_pass": op.buffer_pass,
            "buffer_address": op.buffer_address,
        }
    return {"ops": ops, "buffer_num_calls": dict(trace._buffer_num_calls)}


def _trace(model: nn.Module, x: torch.Tensor, *, column: bool) -> tuple[dict[str, Any], set[str]]:
    """Trace ``model``; return its snapshot and which column bodies ran."""

    ran: set[str] = set()
    real_step2 = _seal_columns.find_output_ancestors
    real_step6 = _buffer_fixup._fix_by_column

    def step2(trace: Any) -> bool:
        done = real_step2(trace) if column else False
        if done:
            ran.add("step2")
        return done

    def step6(trace: Any) -> bool:
        done = real_step6(trace) if column else False
        if done and trace.buffer_layers:
            ran.add("step6")
        return done

    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(_seal_columns, "find_output_ancestors", step2)
        patch.setattr(_buffer_fixup, "_fix_by_column", step6)
        traced = tl.trace(model, x)
    finally:
        patch.undo()
    return _snapshot(traced), ran


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_column_bodies_equal_the_bodies_they_replace(name: str) -> None:
    """Steps 2 and 6 write the same fields with and without the column bodies."""

    torch.manual_seed(0)
    model, x = _MODELS[name]()
    by_column, ran = _trace(model, x, column=True)
    by_existing, ran_existing = _trace(model, x, column=False)
    assert by_column == by_existing
    assert ran_existing == set()
    assert ("step2" in ran) == (name in _STEP2_COVERED)
    if name in _STEP6_RUNS:
        assert "step6" in ran
    if name in _STEP6_DECLINES:
        assert "step6" not in ran


def test_seal_rows_declines_mixed_records() -> None:
    """A mapping holding a non-``Op`` record gets no row view."""

    assert _seal_columns.seal_rows(["a"], [object()]) is None
    assert _seal_columns.seal_rows([], []) is None
