"""Projective descriptor goldens for the target-anchored reverse solve."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import pytest
import torch
from support.rf_isolation import preserved_rf_registry
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


@pytest.fixture(autouse=True)
def isolated_rule_registry() -> Iterator[None]:
    """Restore the process-global RF rule registry after every projective golden."""

    with preserved_rf_registry(bump_epoch=True):
        yield


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


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("mapping", "extent", "expected_lo", "expected_hi", "exact"),
    [
        (
            _Mapped(_Affine(Fraction(-1), Fraction(4)), _Affine(Fraction(-1), Fraction(4))),
            5,
            _Affine(Fraction(-1), Fraction(4)),
            _Affine(Fraction(-1), Fraction(4)),
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


def test_projective_target_set_cache_is_canonical_lru_and_epoch_guarded() -> None:
    """Reuse canonical target sets, evict at eight, and invalidate after rule changes."""

    _register_exact_rules()
    trace = tl.trace(nn.Sequential(*(nn.ReLU() for _ in range(9))), torch.randn(1, 2, 4, 4))
    targets = [op for op in trace._layer_list if op.func_name == "relu"]
    paired = solve_projective(trace, [targets[1], targets[0]])

    assert solve_projective(trace, [targets[0], targets[1]]) is paired
    for target in targets:
        solve_projective(trace, [target])

    cache = trace.__dict__["_rf_directional_solutions"]["target"]
    assert len(cache) == 8
    assert (targets[0].label, targets[1].label) not in cache
    assert type(trace)._PORTABLE_STATE_SPEC["_rf_directional_solutions"] is FieldPolicy.DROP

    latest = solve_projective(trace, [targets[-1]])

    @_rules.register_rf_rule("relu", replace=True)
    def replacement(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Advance the registry epoch with equivalent geometry."""

        return context.passthrough()

    assert solve_projective(trace, [targets[-1]]) is not latest
