"""Structural grouping (``grouper_v1``) through the public recording path.

Torch exhaustive captures carry call-site classes and group by call site
(``callsite_v1``). A recording cooked with ``Recording.to_trace()`` carries
none, so step 7 runs the backend-neutral structural grouper on it. These tests
drive that grouper through the public ``tl.record`` door on recurrent and
shared-module models and check the layers and passes it produces.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import example_models
import pytest
import torch

import torchlens as tl

pytestmark = pytest.mark.smoke


def _cooked(model: torch.nn.Module, x: torch.Tensor) -> Any:
    """Record one forward pass and cook it into a Trace."""

    model.eval()
    recording = tl.record(model, x, save=lambda ctx: ctx.kind == "op")
    return recording.to_trace()


def _layer_passes(trace: Any) -> dict[str, tuple[str, int]]:
    """Map each layer (label without the pass suffix) to (func_name, num_passes)."""

    rows: dict[str, tuple[str, int]] = {}
    for op in trace.ops:
        rows.setdefault(op.layer_label.split(":")[0], (op.func_name, op.num_passes))
    return rows


def _multi_pass(trace: Any) -> dict[str, int]:
    """Func name and pass count of every multi-pass layer."""

    return {
        f"{func}:{label}": passes
        for label, (func, passes) in _layer_passes(trace).items()
        if passes > 1
    }


def _pass_indices(trace: Any) -> dict[str, list[int]]:
    """Pass indices seen per layer, in capture order."""

    seen: dict[str, list[int]] = defaultdict(list)
    for op in trace.ops:
        seen[op.layer_label.split(":")[0]].append(op.pass_index)
    return seen


@pytest.mark.parametrize(
    "model_cls",
    [
        example_models.RecurrentParamsSimple,
        example_models.SharedParamLoopExternal,
        example_models.InterleavedSharedParamLoops,
        example_models.ParallelLoops,
        example_models.NestedParamFreeLoops,
    ],
)
def test_cooked_recording_runs_structural_grouper_with_consistent_passes(model_cls: type) -> None:
    """The cooked recording groups with ``grouper_v1`` and numbers passes 1..n per layer."""

    torch.manual_seed(0)
    x = torch.rand(5, 5)
    cooked = _cooked(model_cls(), x)
    assert cooked.grouping_policy["detector"] == "grouper_v1"
    for label, indices in _pass_indices(cooked).items():
        passes = _layer_passes(cooked)[label][1]
        assert sorted(indices) == list(range(1, passes + 1)), label
    assert _multi_pass(cooked), "a recurrent model must produce at least one multi-pass layer"
