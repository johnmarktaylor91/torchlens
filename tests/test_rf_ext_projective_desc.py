"""Projective descriptor goldens for the target-anchored reverse solve."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import FieldPolicy
from torchlens.receptive_field import _rules
from torchlens.receptive_field._engine_forward import solve_projective
from torchlens.receptive_field._engine_geometry import (
    _Affine,
    _Mapped,
    _transpose_mapped,
)
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import (
    ReceptiveFieldDirection,
    ReceptiveFieldStatus,
)


@pytest.fixture(autouse=True)
def isolated_rule_registry() -> Iterator[None]:
    """Restore the process-global RF rule registry after every projective golden."""

    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    _rules._RF_RULES.clear()
    _rules._RF_RULES_EPOCH += 1
    yield
    _rules._RF_RULES.clear()
    _rules._RF_RULES.update(saved_rules)
    _rules._RF_RULES_EPOCH = saved_epoch


def _register_exact_rules() -> None:
    """Register the exact affine families used by focused projective tests."""

    @_rules.register_rf_rule("conv2d")
    def convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit an exact captured convolution recurrence."""

        kernel = context.cfg("kernel_size")
        assert isinstance(kernel, tuple)
        return context.window(
            kernel=kernel,
            stride=context.cfg("stride", (1, 1)),
            padding=context.cfg("padding", (0, 0)),
            dilation=context.cfg("dilation", (1, 1)),
        )

    @_rules.register_rf_rule("add", "relu")
    def passthrough(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit an exact elementwise relation."""

        return context.passthrough()


@pytest.mark.parametrize(
    ("mapping", "extent", "expected_lo", "expected_hi", "exact"),
    [
        (
            _Mapped(_Affine(Fraction(2), Fraction(-1)), _Affine(Fraction(2), Fraction(1))),
            6,
            _Affine(Fraction(1, 2), Fraction(-1, 2)),
            _Affine(Fraction(1, 2), Fraction(1, 2)),
            True,
        ),
        (
            _Mapped(_Affine(Fraction(-1), Fraction(4)), _Affine(Fraction(-1), Fraction(4))),
            5,
            _Affine(Fraction(-1), Fraction(4)),
            _Affine(Fraction(-1), Fraction(4)),
            True,
        ),
        (
            _Mapped(_Affine(Fraction(0), Fraction(0)), _Affine(Fraction(0), Fraction(0))),
            5,
            _Affine(Fraction(0), Fraction(0)),
            _Affine(Fraction(0), Fraction(4)),
            True,
        ),
        (
            _Mapped(_Affine(Fraction(0), Fraction(0)), _Affine(Fraction(1), Fraction(0))),
            5,
            _Affine(Fraction(1), Fraction(0)),
            _Affine(Fraction(0), Fraction(4)),
            True,
        ),
        (
            _Mapped(_Affine(Fraction(2), Fraction(0)), _Affine(Fraction(3), Fraction(0))),
            5,
            _Affine(Fraction(0), Fraction(0)),
            _Affine(Fraction(0), Fraction(4)),
            False,
        ),
    ],
)
def test_transpose_mapped_discharge_per_case_envelopes(
    mapping: _Mapped,
    extent: int,
    expected_lo: _Affine,
    expected_hi: _Affine,
    exact: bool,
) -> None:
    """Lock affine, flip, broadcast, cumulative, and conservative mixed transposes."""

    transposed = _transpose_mapped(mapping, extent)

    assert transposed.lo == expected_lo
    assert transposed.hi == expected_hi
    assert transposed.exact is exact


def test_transpose_mapped_never_recovers_exactness() -> None:
    """Keep an inexact chord envelope inexact after affine transposition."""

    mapping = _Mapped(
        _Affine(Fraction(1), Fraction(-1)),
        _Affine(Fraction(1), Fraction(1)),
        exact=False,
    )

    assert not _transpose_mapped(mapping, 5).exact


def test_projective_descriptor_reports_input_pixel_output_spread() -> None:
    """Compose two transposed 3x3 windows into a 5x5 target-space footprint."""

    _register_exact_rules()
    model = nn.Sequential(
        nn.Conv2d(1, 1, 3, padding=1),
        nn.Conv2d(1, 1, 3, padding=1),
    )
    trace = tl.trace(model, torch.randn(1, 1, 9, 9))
    source = trace.input_ops[0]
    target = trace.output_ops[0]

    descriptor = next(iter(solve_projective(trace, [target]).per_op[source.label].values()))

    assert descriptor.direction is ReceptiveFieldDirection.PROJECTIVE
    assert descriptor.unit_shape == tuple(source.shape)
    assert descriptor.input_shape == tuple(target.shape)
    assert descriptor.input_op_label == target.label
    assert descriptor.size == (5, 5)
    assert descriptor.jump == (Fraction(1), Fraction(1))
    assert descriptor.center0 == (Fraction(0), Fraction(0))
    assert descriptor.status is ReceptiveFieldStatus.EXACT


class _GlobalMean(nn.Module):
    """Keep-dimensional global reduction used to test target saturation."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reduce both spatial axes to a global output."""

        return inputs.mean(dim=(-2, -1), keepdim=True)


def test_projective_global_rule_uses_frozen_whole_input_status() -> None:
    """Store WHOLE_INPUT when every non-pointwise target axis is exactly full."""

    @_rules.register_rf_rule("mean")
    def mean_rule(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Mark both reduced spatial axes as exact whole-extent relations."""

        return context.full(axes=(-2, -1), exact=True)

    trace = tl.trace(_GlobalMean(), torch.randn(2, 3, 7, 7))
    descriptor = next(
        iter(
            solve_projective(trace, [trace.output_ops[0]]).per_op[trace.input_ops[0].label].values()
        )
    )

    assert descriptor.layout.axis_kinds == ("pointwise", "pointwise", "full", "full")
    assert descriptor.status is ReceptiveFieldStatus.WHOLE_INPUT


def test_projective_data_dependent_rule_refuses_static_geometry() -> None:
    """Preserve DATA_DEPENDENT (not UNKNOWN) while refusing a static forward routing descriptor.

    A data-dependent routing family has no static transpose geometry (axes stay None), but the
    public status must keep the "routing depends on runtime data" distinction rather than collapsing
    to "facts were not captured" (UNKNOWN).
    """

    @_rules.register_rf_rule("relu")
    def routed(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Stand in for a captured data-dependent routing family."""

        return context.data_dependent("routing depends on captured values")

    trace = tl.trace(nn.ReLU(), torch.randn(1, 2, 5, 5))
    descriptor = next(
        iter(
            solve_projective(trace, [trace.output_ops[0]]).per_op[trace.input_ops[0].label].values()
        )
    )

    assert descriptor.axes is None
    assert descriptor.status is ReceptiveFieldStatus.DATA_DEPENDENT
    assert any("no static transpose" in note for note in descriptor.notes)


class _Merge(nn.Module):
    """Aligned narrow and wide branches used for projective union."""

    def __init__(self) -> None:
        """Create one pointwise and one 3x3 convolution branch."""

        super().__init__()
        self.narrow = nn.Conv2d(1, 1, 1)
        self.wide = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Union both branch footprints at an elementwise merge."""

        return self.narrow(inputs) + self.wide(inputs)


def test_projective_forward_merge_unions_reached_targets() -> None:
    """Use the exact 3x3 hull of aligned pointwise and wide forward branches."""

    _register_exact_rules()
    trace = tl.trace(_Merge(), torch.randn(1, 1, 9, 9))
    descriptor = next(
        iter(
            solve_projective(trace, [trace.output_ops[0]]).per_op[trace.input_ops[0].label].values()
        )
    )

    assert descriptor.size == (3, 3)
    assert descriptor.status is ReceptiveFieldStatus.EXACT


def test_projective_inexact_family_stays_provenance_marked_upper_bound() -> None:
    """Keep an unproved window transpose as a provenance-marked upper bound."""

    @_rules.register_rf_rule("max_pool2d")
    def pooling(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit a deliberately unproved pooling envelope."""

        return context.window(kernel=(2, 2), stride=(2, 2), exact=False)

    trace = tl.trace(nn.MaxPool2d(2), torch.randn(1, 1, 8, 8))
    descriptor = next(
        iter(
            solve_projective(trace, [trace.output_ops[0]]).per_op[trace.input_ops[0].label].values()
        )
    )

    assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
    assert any("upper-bound envelope" in note for note in descriptor.notes)


def test_projective_target_set_cache_is_canonical_lru_and_epoch_guarded() -> None:
    """Reuse canonical target sets, evict at eight, and invalidate after rule changes."""

    _register_exact_rules()
    trace = tl.trace(nn.Sequential(*(nn.ReLU() for _ in range(9))), torch.randn(1, 2, 4, 4))
    targets = [op for op in trace.layer_list if op.func_name == "relu"]
    paired = solve_projective(trace, [targets[1], targets[0]])

    assert solve_projective(trace, [targets[0], targets[1]]) is paired
    for target in targets:
        solve_projective(trace, [target])

    cache = trace.__dict__["_rf_target_solutions"]
    assert len(cache) == 8
    assert (targets[0].label, targets[1].label) not in cache
    assert type(trace).PORTABLE_STATE_SPEC["_rf_target_solutions"] is FieldPolicy.DROP

    latest = solve_projective(trace, [targets[-1]])

    @_rules.register_rf_rule("relu", replace=True)
    def replacement(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Advance the registry epoch with equivalent geometry."""

        return context.passthrough()

    assert solve_projective(trace, [targets[-1]]) is not latest
