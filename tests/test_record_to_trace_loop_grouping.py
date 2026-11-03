"""Cooked recordings group recurrent layers the way ``tl.trace`` does.

Predicate-mode records carry the same capture-time equivalence class and
parameter identity as exhaustive ops, so ``Recording.to_trace()`` hands the
structural grouper what it needs: a ``Linear`` reused three times becomes
one layer with three passes on both paths, with the same layer labels and
pass numbering.
"""

from __future__ import annotations

from typing import Any

import example_models
import pytest
import torch
from torch import nn

import torchlens as tl


class _ModuleLoop(nn.Module):
    """One Linear reused three times with a function between passes."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(self.cell(x))
        return x


class _Cell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(x))


class _SubmoduleLoop(nn.Module):
    """A whole submodule reused three times."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = _Cell()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.cell(x)
        return x


def _both_ways(model_cls: type[nn.Module], x: torch.Tensor) -> tuple[Any, Any]:
    """Capture the same model exhaustively and through a cooked recording."""

    torch.manual_seed(0)
    exhaustive = tl.trace(model_cls().eval(), x)
    torch.manual_seed(0)
    recording = tl.record(model_cls().eval(), x, save=lambda ctx: ctx.kind == "op")
    return exhaustive, recording.to_trace()


def _grouping(trace: Any) -> list[tuple[str, int, int]]:
    """Execution-ordered (layer label, pass number, pass count) per op."""

    return [(op.layer_label, op.pass_number, op.num_passes) for op in trace.ops]


@pytest.mark.parametrize(
    "model_cls, shape",
    [
        (_ModuleLoop, (2, 4)),
        (_SubmoduleLoop, (2, 4)),
        (example_models.RecurrentParamsSimple, (5, 5)),
        (example_models.InterleavedSharedParamLoops, (5, 5)),
        (example_models.VaryingLoopWithParam, (5, 5)),
    ],
)
def test_cooked_recording_groups_loops_like_trace(
    model_cls: type[nn.Module], shape: tuple[int, ...]
) -> None:
    """Layer labels, pass numbers and pass counts agree op for op across both paths."""

    exhaustive, cooked = _both_ways(model_cls, torch.rand(*shape))

    assert _grouping(cooked) == _grouping(exhaustive)
    assert len(cooked.layers) == len(exhaustive.layers)
    assert [op.has_params for op in cooked.ops] == [op.has_params for op in exhaustive.ops]
    assert all(op.equivalence_class for op in cooked.ops)


def test_cooked_recording_resolves_parameters_without_holding_them() -> None:
    """Cooked ops resolve their parameters to the model's inventory by address."""

    exhaustive, cooked = _both_ways(_ModuleLoop, torch.rand(2, 4))

    assert sorted(cooked.params.keys()) == sorted(exhaustive.params.keys())
    linear_ops = [op for op in cooked.ops if op.layer_type == "linear"]
    assert linear_ops and all(op.has_params for op in linear_ops)
    assert all(len(op._param_barcodes) == 2 for op in linear_ops)
    # A recording never pins the model's parameters: the inventory carries
    # metadata only, and cooking twice gives each Trace its own copies.
    assert all(param._param_ref is None for param in cooked.params)
    second = tl.record(_ModuleLoop().eval(), torch.rand(2, 4), save=lambda ctx: ctx.kind == "op")
    first_cook = second.to_trace()
    second_cook = second.to_trace()
    assert first_cook.params["cell.weight"] is not second_cook.params["cell.weight"]


def test_parameter_free_loop_keeps_classes_but_does_not_group() -> None:
    """A loop with no parameters has nothing for the structural grouper to anchor on.

    Exhaustive capture groups it by call site; a recording records no call
    sites, so its cooked ops stay separate single-pass layers while still
    carrying argument-hash equivalence classes.
    """

    _exhaustive, cooked = _both_ways(example_models.SimpleLoopNoParam, (5, 5))

    assert all(op.equivalence_class for op in cooked.ops)
    assert all(op.num_passes == 1 for op in cooked.ops)
