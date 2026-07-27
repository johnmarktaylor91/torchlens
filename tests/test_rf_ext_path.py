"""Graph-path and source-seeded receptive-field extension tests."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _engine
from torchlens.receptive_field._errors import NoInfluencePathError, ReceptiveFieldError
from torchlens.receptive_field._path import (
    ancestor_labels,
    between_labels,
    descendant_labels,
    require_path,
    resolve_graph_point,
)
from torchlens.receptive_field._types import ReceptiveFieldDirection


class _Branches(nn.Module):
    """Two independent branches with distinct graph endpoints."""

    def __init__(self) -> None:
        """Create two single-output convolution modules."""

        super().__init__()
        self.left = nn.Conv2d(1, 1, 3, padding=1)
        self.right = nn.Conv2d(1, 1, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return branch outputs without merging them."""

        return self.left(inputs), self.right(inputs)


def _op(trace: object, name: str, index: int = 0) -> object:
    """Return one captured operation with the requested normalized function name.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    name:
        Function name to select.
    index:
        Zero-based match position.

    Returns
    -------
    object
        Selected operation.
    """

    matches = [op for op in trace.layer_list if op.func_name == name]  # type: ignore[attr-defined]
    return matches[index]


def test_resolve_graph_point_accepts_exact_labels_and_entity_handles() -> None:
    """Resolve Op, Layer, ModuleCall, and Module handles to the same output Op."""

    trace = tl.trace(_Branches(), torch.randn(1, 1, 8, 8))
    left = _op(trace, "conv2d")
    layer = trace.layers[left.layer_label]
    module = trace.modules["left"]
    module_call = module.calls[0]

    assert resolve_graph_point(trace, left.label) is left
    assert resolve_graph_point(trace, left) is left
    assert resolve_graph_point(trace, layer) is left
    assert resolve_graph_point(trace, module_call) is left
    assert resolve_graph_point(trace, module) is left

    other_trace = tl.trace(nn.Identity(), torch.randn(1, 1, 8, 8))
    with pytest.raises(ReceptiveFieldError, match="does not belong"):
        resolve_graph_point(other_trace, left)


def test_reachability_and_between_slice_follow_only_directed_paths() -> None:
    """Compute ancestor, descendant, and between sets on the executed DAG."""

    trace = tl.trace(_Branches(), torch.randn(1, 1, 8, 8))
    source = trace.input_ops[0]
    left = _op(trace, "conv2d", 0)
    right = _op(trace, "conv2d", 1)

    descendants = descendant_labels(trace, source)
    ancestors = ancestor_labels(trace, left)
    between = between_labels(trace, source, left)

    assert {source.label, left.label} <= descendants
    assert {source.label, left.label} <= ancestors
    assert source.label in between
    assert left.label in between
    assert right.label not in between
    assert between <= descendants
    assert between <= ancestors


def test_require_path_rejects_unrelated_ops_without_a_swap_hint() -> None:
    """Raise the typed path error and avoid a false reverse-path suggestion."""

    trace = tl.trace(_Branches(), torch.randn(1, 1, 8, 8))
    left = _op(trace, "conv2d", 0)
    right = _op(trace, "conv2d", 1)

    with pytest.raises(NoInfluencePathError) as error:
        require_path(left, right, ReceptiveFieldDirection.RECEPTIVE)

    message = str(error.value)
    assert "directed A -> B path" in message
    assert left.label in message
    assert right.label in message
    assert "swap" not in message


@pytest.mark.parametrize(
    ("model", "shape"),
    [
        (nn.Sequential(nn.Conv1d(2, 3, 3, padding=1), nn.ReLU()), (1, 2, 12)),
        (
            nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.MaxPool2d(2), nn.ReLU()),
            (1, 2, 12, 12),
        ),
    ],
)
def test_input_seeded_solve_from_is_behavior_equivalent(
    model: nn.Module, shape: tuple[int, ...]
) -> None:
    """Keep source-seeding at a model input exactly equal to the sealed default solve."""

    trace = tl.trace(model, torch.randn(*shape))
    default = _engine.solve(trace)
    from_input = _engine.solve_from(trace, trace.input_ops[0])

    assert from_input == default
    assert from_input.descriptors == default.descriptors
    assert from_input.per_op == default.per_op
    assert from_input.states == default.states


def test_solve_from_seeds_only_the_source_and_reaches_its_descendants() -> None:
    """Key a layer source by label and omit states on operations before that source."""

    trace = tl.trace(
        nn.Sequential(nn.Conv1d(2, 3, 3, padding=1), nn.MaxPool1d(2)),
        torch.randn(1, 2, 12),
    )
    source = _op(trace, "conv1d")
    target = _op(trace, "max_pool1d")
    solution = _engine.solve_from(trace, source)

    assert tuple(solution.per_op[source.label]) == (source.label,)
    assert tuple(solution.per_op[target.label]) == (source.label,)
    assert solution.per_op[trace.input_ops[0].label] == {}


def test_source_solution_cache_is_trace_owned_lru_eight() -> None:
    """Reuse source solutions and evict the least-recently-used ninth entry."""

    trace = tl.trace(nn.Sequential(*(nn.ReLU() for _ in range(10))), torch.randn(1, 2, 12))
    sources = tuple(trace.compute_ops)[:9]
    first = _engine.solve_from(trace, sources[0])
    assert _engine.solve_from(trace, sources[0]) is first

    for source in sources[1:]:
        _engine.solve_from(trace, source)

    cache = trace.__dict__["_rf_source_solutions"]
    assert isinstance(cache, OrderedDict)
    assert len(cache) == 8
    assert sources[0].label not in cache
    assert tuple(cache) == tuple(source.label for source in sources[1:])
