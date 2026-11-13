"""Layer grouping on the public recording path.

Torch exhaustive captures carry call-site classes and group by call site
(``callsite_v2``). Predicate records mint the same call-site classes, so a
recording cooked with ``Recording.to_trace()`` groups by call site too and
stamps ``callsite_v2`` (RC-50/272); the backend-neutral structural grouper
(``grouper_v1``) is for captures without that evidence. These tests pin the
cooked stamp and the pass numbering it leaves behind.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import example_models
import pytest
import torch

import torchlens as tl


def _cooked(model: torch.nn.Module, x: torch.Tensor) -> Any:
    """Record one forward pass of every op and cook it into a Trace."""

    model.eval()
    recording = tl.record(model, x, save=lambda ctx: ctx.kind == "op")
    return recording.to_trace()


def _pass_indices(trace: Any) -> dict[str, tuple[int, list[int]]]:
    """Per layer (label without the pass suffix): declared pass count and the indices seen."""

    seen: dict[str, list[int]] = defaultdict(list)
    declared: dict[str, int] = {}
    for op in trace.ops:
        layer = op.layer_label.split(":")[0]
        seen[layer].append(op.pass_number)
        declared[layer] = op.num_passes
    return {layer: (declared[layer], indices) for layer, indices in seen.items()}


@pytest.mark.parametrize(
    "model_cls",
    [example_models.RecurrentParamsSimple, example_models.InterleavedSharedParamLoops],
)
def test_cooked_recording_stamps_structural_grouper_and_numbers_passes(model_cls: type) -> None:
    """Cooking a recording and the exhaustive capture of the same model both run
    ``callsite_v2``; every layer's passes are numbered 1..n with no gap or repeat."""

    torch.manual_seed(0)
    x = torch.rand(5, 5)
    cooked = _cooked(model_cls(), x)
    exhaustive = tl.trace(model_cls().eval(), x)

    assert cooked.options.grouping_policy["detector"] == "callsite_v2"
    assert cooked.options.grouping_policy["requested"] == "structural"
    assert exhaustive.options.grouping_policy["detector"] == "callsite_v2"
    passes = _pass_indices(cooked)
    assert passes, "the cooked trace must contain ops"
    for layer, (declared, indices) in passes.items():
        assert sorted(indices) == list(range(1, declared + 1)), layer
    assert cooked._num_ops == exhaustive._num_ops
