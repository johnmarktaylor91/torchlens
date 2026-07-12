"""Projective per-unit membership-transpose and geometric-adjoint goldens."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _rules
from torchlens.receptive_field._engine_forward import solve_projective
from torchlens.receptive_field._forward_query import box_for_source_unit
from torchlens.receptive_field._path import forward_index_image
from torchlens.receptive_field._query import _IndexSet
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import ReceptiveFieldDirection, ReceptiveFieldStatus


_PACK: dict[str, object] | None = None


@pytest.fixture(autouse=True)
def isolated_rule_registry() -> Iterator[None]:
    """Install the built-in RF rules per test and restore the registry afterward."""

    import importlib

    global _PACK
    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    _rules._RF_RULES.clear()
    if _PACK is None:
        module = importlib.import_module("torchlens.receptive_field.rules")
        if not _rules._RF_RULES:
            for name in module.__all__:
                importlib.reload(getattr(module, name))
        _PACK = dict(_rules._RF_RULES)
    else:
        _rules._RF_RULES.update(_PACK)
    yield
    _rules._RF_RULES.clear()
    _rules._RF_RULES.update(saved_rules)
    _rules._RF_RULES_EPOCH = saved_epoch


def _spatial_bounds(box: object) -> tuple[int | None, int | None]:
    """Return the sole spatial axis bounds from a one-dimensional RF box."""

    axes = getattr(box, "axes")
    spatial = next(axis for axis in axes if axis.kind == "windowed")
    return spatial.clipped_start, spatial.clipped_stop


def _jacobian_support_bounds(layer: nn.Module, extent: int, source_index: int) -> tuple[int, int]:
    """Return the exact output-support hull from a brute-force autograd Jacobian."""

    inputs = torch.ones(1, 1, extent, requires_grad=True)

    def apply(value: torch.Tensor) -> torch.Tensor:
        """Apply one deterministic linear test layer."""

        return layer(value)

    jacobian = torch.autograd.functional.jacobian(apply, inputs)
    column = jacobian[0, 0, :, 0, 0, source_index]
    reached = torch.nonzero(column != 0, as_tuple=False).flatten().tolist()
    if not reached:
        return 0, 0
    return int(min(reached)), int(max(reached)) + 1


def _trace_projective_box(layer: nn.Module, extent: int, source_index: int) -> object:
    """Capture a layer and return one internal projective per-unit answer."""

    trace = tl.trace(layer, torch.ones(1, 1, extent))
    source = trace.input_ops[0]
    target = trace.output_ops[0]
    solution = solve_projective(trace, [target])
    return box_for_source_unit(solution, source, (source_index,), target=target)


@pytest.mark.parametrize(
    ("factory", "extent"),
    [
        (lambda: nn.Conv1d(1, 1, 3, stride=2, padding=1, bias=False), 9),
        (lambda: nn.AvgPool1d(3, stride=2, padding=1), 9),
        (lambda: nn.Conv1d(1, 1, 3, dilation=2, padding=2, bias=False), 9),
        (lambda: nn.ConvTranspose1d(1, 1, 3, stride=2, padding=1, bias=False), 7),
    ],
)
def test_forward_support_matches_brute_force_connectivity(
    factory: Callable[[], nn.Module], extent: int
) -> None:
    """Match conv, pool, stride, dilation, and transposed-conv support hulls exactly."""

    layer = factory()
    for parameter in layer.parameters():
        nn.init.ones_(parameter)
    for source_index in range(extent):
        expected = _jacobian_support_bounds(layer, extent, source_index)
        box = _trace_projective_box(layer, extent, source_index)
        assert _spatial_bounds(box) == expected
        assert box.direction is ReceptiveFieldDirection.PROJECTIVE
        assert box.empty is (expected == (0, 0))


@pytest.mark.parametrize(
    "layer",
    [
        nn.Conv1d(1, 1, 3, stride=2, padding=1, bias=False),
        nn.AvgPool1d(3, stride=2, padding=1),
        nn.ConvTranspose1d(1, 1, 3, stride=2, padding=1, bias=False),
        nn.AdaptiveAvgPool1d(5),
    ],
)
def test_integer_geometric_adjoint_battery_has_zero_tolerance(layer: nn.Module) -> None:
    """Agree exactly with sealed backward membership for several geometric families."""

    extent = 8
    trace = tl.trace(layer, torch.ones(1, 1, extent))
    source = trace.input_ops[0]
    target = trace.output_ops[0]
    solution = solve_projective(trace, [target])
    backward_boxes = [target.receptive_field.at((unit,)) for unit in range(target.shape[-1])]
    for source_index in range(extent):
        forward = box_for_source_unit(solution, source, (source_index,), target=target)
        reached = [
            unit
            for unit, backward in enumerate(backward_boxes)
            if (
                (start := _spatial_bounds(backward)[0]) is not None
                and (stop := _spatial_bounds(backward)[1]) is not None
                and start <= source_index < stop
            )
        ]
        expected = (0, 0) if not reached else (min(reached), max(reached) + 1)
        assert _spatial_bounds(forward) == expected


class _AddZero(nn.Module):
    """Produce a captured add operation without changing tensor values."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add a scalar zero to the input tensor."""

        return inputs + 0


def test_membership_candidate_budget_widens_to_upper_bound() -> None:
    """Widen an oversized membership envelope honestly instead of truncating it."""

    @_rules.register_rf_rule("add", replace=True)
    def constant_broadcast(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Make source index zero potentially reach every target coordinate."""

        def backward(axis: int, output_set: _IndexSet) -> tuple[_IndexSet, bool]:
            """Map every candidate target coordinate back to source coordinate zero."""

            _ = axis, output_set
            return _IndexSet.singleton(0), True

        edges = (((Fraction(0), Fraction(0)), (Fraction(0), Fraction(0))),)
        return context.window_edges(edges, exact=True, map_index_set=backward)

    trace = tl.trace(_AddZero(), torch.ones(1, 1, 5000))
    source = trace.input_ops[0]
    target = trace.output_ops[0]
    box = box_for_source_unit(solve_projective(trace, [target]), source, (0,), target=target)

    assert box.status is ReceptiveFieldStatus.UPPER_BOUND
    assert not box.exact
    assert _spatial_bounds(box) == (0, 5000)


def test_unit_box_only_rule_degrades_to_upper_bound() -> None:
    """Treat a unit-box-only callback as an inexact transposed envelope."""

    @_rules.register_rf_rule("add", replace=True)
    def unit_box_only(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Expose an exact-looking edge descriptor without a transposable set callback."""

        def unit_box(axis: int, unit: int) -> tuple[int, int]:
            """Return a local identity box that the forward walker must not call."""

            _ = axis
            return unit, unit + 1

        edges = (((Fraction(1), Fraction(0)), (Fraction(1), Fraction(0))),)
        return context.window_edges(edges, exact=True, unit_box=unit_box)

    trace = tl.trace(_AddZero(), torch.ones(1, 1, 8))
    source = trace.input_ops[0]
    target = trace.output_ops[0]
    box = box_for_source_unit(solve_projective(trace, [target]), source, (4,), target=target)

    assert _spatial_bounds(box) == (4, 5)
    assert box.status is ReceptiveFieldStatus.UPPER_BOUND
    assert not box.exact


def test_projective_data_dependent_rule_preserves_its_status() -> None:
    """Keep runtime-routed projective geometry distinct from missing facts."""

    @_rules.register_rf_rule("add", replace=True)
    def runtime_routed(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Model a route whose destination depends on captured runtime data."""

        return context.data_dependent("destination depends on runtime data")

    trace = tl.trace(_AddZero(), torch.ones(1, 1, 8))
    source = trace.input_ops[0]
    target = trace.output_ops[0]
    descriptors = solve_projective(trace, [target]).per_op[source.label]

    assert next(iter(descriptors.values())).status is ReceptiveFieldStatus.DATA_DEPENDENT


def test_forward_accelerator_has_exhaustive_membership_duality_golden() -> None:
    """Require an optional accelerator to equal membership transpose on small extents."""

    accelerator_calls = 0

    def backward(axis: int, output_set: _IndexSet) -> tuple[_IndexSet, bool]:
        """Map each child coordinate to its two-element parent bin."""

        _ = axis
        return _IndexSet.from_values(
            value for output in output_set.values() for value in (2 * output, 2 * output + 1)
        ), True

    def forward(axis: int, source_set: _IndexSet) -> tuple[_IndexSet, bool]:
        """Accelerate the exact transpose of the two-element-bin relation."""

        nonlocal accelerator_calls
        _ = axis
        accelerator_calls += 1
        return _IndexSet.from_values(value // 2 for value in source_set.values()), True

    for source_index in range(8):
        candidates = _IndexSet.interval(0, 3)
        membership, exact = forward_index_image(
            _IndexSet.singleton(source_index),
            candidates,
            lambda candidate_set: backward(0, candidate_set),
        )
        accelerated, accelerated_exact = forward(0, _IndexSet.singleton(source_index))
        assert accelerated == membership
        assert exact and accelerated_exact

    @_rules.register_rf_rule("add", replace=True)
    def binned(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Expose the duality-proved callback pair to the projective walker."""

        edges = (((Fraction(2), Fraction(0)), (Fraction(2), Fraction(1))),)
        return context.window_edges(
            edges,
            exact=True,
            map_index_set=backward,
            map_index_set_forward=forward,
        )

    trace = tl.trace(_AddZero(), torch.ones(1, 1, 8))
    source = trace.input_ops[0]
    target = trace.output_ops[0]
    box = box_for_source_unit(solve_projective(trace, [target]), source, (5,), target=target)

    assert _spatial_bounds(box) == (2, 3)
    assert box.status is ReceptiveFieldStatus.EXACT
    assert accelerator_calls == 9
