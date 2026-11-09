"""Cooked recordings group recurrent layers the way ``tl.trace`` does.

Predicate-mode records carry the same capture-time call-site class,
equivalence class and parameter identity as exhaustive ops, so
``Recording.to_trace()`` runs the same call-site grouper: a ``Linear`` reused
three times from one loop line becomes one layer with three passes on both
paths, an unrolled repeat stays separate layers on both, with the same layer
labels and pass numbering.
"""

from __future__ import annotations

from typing import Any

import example_models
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture._cooked_inventory import recording_projection_inventory


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


class _FunctionLoop(nn.Module):
    """A parameter-free loop body: only the call site tells the passes apart."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 2
        for _ in range(3):
            x = torch.log(x)
            x = torch.sin(x)
        return x + 3


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
        (_FunctionLoop, (5, 5)),
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
    assert all(op.graph.equivalence_class for op in cooked.ops)
    assert cooked.options.grouping_policy["detector"] == "callsite_v2"
    assert (
        cooked.options.grouping_policy["detector"] == exhaustive.options.grouping_policy["detector"]
    )


def test_cooked_recording_resolves_parameters_without_holding_them() -> None:
    """Cooked ops resolve their parameters to the model's inventory by address."""

    exhaustive, cooked = _both_ways(_ModuleLoop, torch.rand(2, 4))

    assert sorted(cooked.params.keys()) == sorted(exhaustive.params.keys())
    linear_ops = [op for op in cooked.ops if op.func_name == "linear"]
    assert linear_ops and all(op.has_params for op in linear_ops)
    assert all(len(op._param_barcodes) == 2 for op in linear_ops)
    # A recording never pins the model's parameters: its inventory carries
    # metadata only, and cooking twice gives each Trace its own copies.
    second = tl.record(_ModuleLoop().eval(), torch.rand(2, 4), save=lambda ctx: ctx.kind == "op")
    _records, inventory, _call_sites = recording_projection_inventory(second)
    assert inventory and all(log._param_ref is None for log in inventory.values())
    first_cook = second.to_trace()
    second_cook = second.to_trace()
    assert first_cook.params["cell.weight"] is not inventory["cell.weight"]
    assert first_cook.params["cell.weight"] is not second_cook.params["cell.weight"]


def _torchvision_model(name: str) -> nn.Module:
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    return getattr(torchvision.models, name)(weights=None).eval()


@pytest.mark.heavy
@pytest.mark.parametrize("name", ["resnet18", "mobilenet_v3_small"])
def test_cooked_torchvision_recording_matches_trace(name: str) -> None:
    """A cooked recording of a torchvision classic has the exhaustive capture's layers.

    Layer count, ordered layer labels, pass grouping and the multiset of module
    addresses all agree; the builder, never the expectation, is what a mismatch
    indicts.
    """

    x = torch.rand(1, 3, 64, 64)
    exhaustive = tl.trace(_torchvision_model(name), x)
    cooked = tl.record(_torchvision_model(name), x, save=lambda ctx: ctx.kind == "op").to_trace()

    assert len(cooked.layers) == len(exhaustive.layers)
    assert [layer.layer_label for layer in cooked.layers] == [
        layer.layer_label for layer in exhaustive.layers
    ]
    assert _grouping(cooked) == _grouping(exhaustive)
    assert sorted(str(op.buffer_address) for op in cooked.ops) == sorted(
        str(op.buffer_address) for op in exhaustive.ops
    )
