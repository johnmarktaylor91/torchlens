"""Golden tests for the receptive-field geometric engine."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import FieldPolicy
from torchlens.receptive_field import _engine, _rules
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import (
    ReceptiveFieldAlignment,
    ReceptiveFieldStatus,
)


@pytest.fixture(autouse=True)
def isolated_rule_registry() -> Iterator[None]:
    """Restore the process-global RF rule registry after every golden."""

    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    yield
    _rules._RF_RULES.clear()
    _rules._RF_RULES.update(saved_rules)
    _rules._RF_RULES_EPOCH = saved_epoch


def _register_standard_rules() -> None:
    """Register only the operation families proved by these focused goldens."""

    @_rules.register_rf_rule("conv1d", "conv2d", "conv3d")
    def convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit captured convolution recurrence parameters."""

        kernel = context.cfg("kernel_size")
        if isinstance(kernel, int):
            kernel = (kernel,)
        assert isinstance(kernel, tuple)
        rank = len(kernel)
        return context.window(
            kernel=kernel,
            stride=context.cfg("stride", (1,) * rank),
            padding=context.cfg("padding", (0,) * rank),
            dilation=context.cfg("dilation", (1,) * rank),
        )

    @_rules.register_rf_rule("max_pool1d", "max_pool2d", "max_pool3d")
    def pooling(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit captured max-pool recurrence parameters, including stride defaulting."""

        kernel = context.cfg("kernel_size")
        rank = len(context.out_shape) - 2
        if isinstance(kernel, int):
            kernel = (kernel,) * rank
        assert isinstance(kernel, tuple)
        return context.window(
            kernel=kernel,
            stride=context.cfg("stride", kernel),
            padding=context.cfg("padding", (0,) * rank),
            dilation=context.cfg("dilation", (1,) * rank),
        )

    @_rules.register_rf_rule("add", "mul", "relu")
    def elementwise(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit an elementwise passthrough/merge rule."""

        return context.passthrough()


def _last_op(trace: object, func_name: str) -> object:
    """Return the last captured operation with a requested function name."""

    operations = [op for op in trace.layer_list if op.func_name == func_name]  # type: ignore[attr-defined]
    assert operations
    return operations[-1]


@pytest.mark.parametrize(
    ("model", "input_shape", "target_name", "expected_size", "expected_jump", "expected_center"),
    [
        (
            nn.Sequential(nn.Conv1d(2, 4, 3, stride=2, padding=1), nn.MaxPool1d(3, stride=2)),
            (1, 2, 32),
            "max_pool1d",
            (7,),
            (Fraction(4),),
            (Fraction(2),),
        ),
        (
            nn.Sequential(nn.Conv2d(2, 4, 3, stride=2, padding=1), nn.MaxPool2d(2)),
            (1, 2, 32, 32),
            "max_pool2d",
            (5, 5),
            (Fraction(4), Fraction(4)),
            (Fraction(1), Fraction(1)),
        ),
        (
            nn.Sequential(nn.Conv3d(2, 4, 3), nn.MaxPool3d(2)),
            (1, 2, 12, 12, 12),
            "max_pool3d",
            (4, 4, 4),
            (Fraction(2), Fraction(2), Fraction(2)),
            (Fraction(3, 2), Fraction(3, 2), Fraction(3, 2)),
        ),
    ],
)
def test_hand_computed_conv_pool_stacks(
    model: nn.Module,
    input_shape: tuple[int, ...],
    target_name: str,
    expected_size: tuple[int, ...],
    expected_jump: tuple[Fraction, ...],
    expected_center: tuple[Fraction, ...],
) -> None:
    """Compose 1-D, 2-D, and 3-D stacks exactly against hand calculations."""

    _register_standard_rules()
    trace = tl.trace(model, torch.randn(*input_shape))
    target = _last_op(trace, target_name)
    descriptor = next(iter(_engine.lookup(trace, target).values()))

    assert descriptor.size == expected_size
    assert descriptor.jump == expected_jump
    assert descriptor.center0 == expected_center
    assert descriptor.status is ReceptiveFieldStatus.EXACT


class _Residual(nn.Module):
    """Aligned residual branch used by the merge golden."""

    def __init__(self) -> None:
        """Create a 3x3 branch and a pointwise branch."""

        super().__init__()
        self.wide = nn.Conv2d(2, 2, 3, padding=1)
        self.point = nn.Conv2d(2, 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Merge aligned branches."""

        return self.wide(inputs) + self.point(inputs)


class _Misaligned(nn.Module):
    """Equal-output-shape branches with offset sampling centers."""

    def __init__(self) -> None:
        """Create stride-two branches centered at 1/2 and 0."""

        super().__init__()
        self.offset = nn.Conv2d(2, 2, 2, stride=2)
        self.origin = nn.Conv2d(2, 2, 1, stride=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Merge the intentionally misaligned branches."""

        return self.offset(inputs) + self.origin(inputs)


def test_residual_merge_unions_aligned_branches_exactly() -> None:
    """Use the union hull of aligned residual branches without losing exactness."""

    _register_standard_rules()
    trace = tl.trace(_Residual(), torch.randn(1, 2, 16, 16))
    descriptor = next(iter(_engine.lookup(trace, _last_op(trace, "__add__")).values()))

    assert descriptor.size == (3, 3)
    assert descriptor.jump == (Fraction(1), Fraction(1))
    assert descriptor.center0 == (Fraction(0), Fraction(0))
    assert descriptor.alignment is ReceptiveFieldAlignment.ALIGNED
    assert descriptor.status is ReceptiveFieldStatus.EXACT


def test_misaligned_merge_is_a_sound_upper_bound() -> None:
    """Trip on unequal centers instead of silently claiming exact aligned geometry."""

    _register_standard_rules()
    trace = tl.trace(_Misaligned(), torch.randn(1, 2, 16, 16))
    descriptor = next(iter(_engine.lookup(trace, _last_op(trace, "__add__")).values()))

    assert descriptor.size == (2, 2)
    assert descriptor.jump == (Fraction(2), Fraction(2))
    assert descriptor.center0 == (Fraction(1, 2), Fraction(1, 2))
    assert descriptor.alignment is ReceptiveFieldAlignment.MISALIGNED
    assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
    assert all(axis.sparse_possible for axis in descriptor.axes or () if axis.kind == "windowed")


class _SqueezeExciteShape(nn.Module):
    """Minimal global-pool-and-broadcast arithmetic graph."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Broadcast a spatial global mean back over its source."""

        return inputs * inputs.mean(dim=(-2, -1), keepdim=True)


def test_global_branch_broadcast_becomes_whole_input_by_arithmetic() -> None:
    """Compose global spatial dependence through a slope-zero broadcast merge."""

    _register_standard_rules()

    @_rules.register_rf_rule("mean")
    def mean_rule(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Mark the captured reduction dimensions full."""

        return context.full(axes=context.arg("dim", (-2, -1)))

    trace = tl.trace(_SqueezeExciteShape(), torch.randn(2, 3, 8, 8))
    descriptor = next(iter(_engine.lookup(trace, _last_op(trace, "__mul__")).values()))

    assert descriptor.layout.axis_kinds == ("pointwise", "pointwise", "full", "full")
    assert descriptor.status is ReceptiveFieldStatus.WHOLE_INPUT
    assert descriptor.axes is not None
    assert descriptor.axes[2].exact and descriptor.axes[3].exact


@pytest.mark.parametrize(
    ("module", "shape", "expected_kinds", "expected_windowed"),
    [
        (nn.Conv2d(3, 4, 3), (2, 3, 12, 12), ("pointwise", "full", "windowed", "windowed"), (2, 3)),
        (nn.Conv1d(3, 4, 3), (2, 3, 12), ("pointwise", "full", "windowed"), (2,)),
    ],
)
def test_layout_is_derived_from_window_rule_semantics(
    module: nn.Module,
    shape: tuple[int, ...],
    expected_kinds: tuple[str, ...],
    expected_windowed: tuple[int, ...],
) -> None:
    """Derive NCHW and NCL roles from proved window rank and consistent shapes."""

    _register_standard_rules()
    trace = tl.trace(module, torch.randn(*shape))
    target = next(op for op in trace.layer_list if op.func_name.startswith("conv"))
    descriptor = next(iter(_engine.lookup(trace, target).values()))

    assert descriptor.layout.axis_kinds == expected_kinds
    assert descriptor.layout.windowed_axes == expected_windowed
    assert descriptor.layout.source == "derived"
    assert all(label == target.label for label in descriptor.layout.provenance)


def test_unconsumed_input_layout_remains_unknown() -> None:
    """Refuse to invent axis roles before registered consumption supplies evidence."""

    trace = tl.trace(nn.Identity(), torch.randn(2, 3, 8, 8))
    input_op = next(op for op in trace.layer_list if op.is_input)
    descriptor = next(iter(_engine.lookup(trace, input_op).values()))

    assert descriptor.layout.axis_kinds == ("unknown",) * 4
    assert descriptor.status is ReceptiveFieldStatus.UNKNOWN


def test_batch_coupled_full_axis_can_never_be_exact() -> None:
    """Represent batch coupling geometrically and derive a non-EXACT status."""

    _register_standard_rules()

    @_rules.register_rf_rule("batch_norm")
    def batch_norm_rule(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Couple batch and spatial axes for the stand-in training rule."""

        return context.full(axes=(0, 2, 3))

    model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), nn.BatchNorm2d(3))
    model.train()
    trace = tl.trace(model, torch.randn(2, 3, 8, 8))
    descriptor = next(iter(_engine.lookup(trace, _last_op(trace, "batch_norm")).values()))

    assert descriptor.batch_coupled
    assert descriptor.status is ReceptiveFieldStatus.WHOLE_INPUT
    assert descriptor.status is not ReceptiveFieldStatus.EXACT
    assert descriptor.axes is not None and descriptor.axes[0].kind == "full"


def test_solution_cache_uses_epoch_and_graph_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse a stable solution and recompute after a registration advances the epoch."""

    _register_standard_rules()
    trace = tl.trace(nn.Conv1d(2, 3, 3), torch.randn(1, 2, 12))
    calls = 0
    original = _engine._solve_uncached

    def counted(*args: object, **kwargs: object) -> object:
        """Count uncached solves while preserving the implementation."""

        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_engine, "_solve_uncached", counted)
    first = _engine.solve(trace)
    second = _engine.solve(trace)
    assert first is second
    assert calls == 1

    @_rules.register_rf_rule("rf_cache_epoch_probe")
    def epoch_probe(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Advance the registry epoch without changing this trace's operations."""

        return context.passthrough()

    third = _engine.solve(trace)
    assert third is not first
    assert calls == 2
    assert type(trace).PORTABLE_STATE_SPEC["_receptive_field_solution"] is FieldPolicy.DROP

    trace._batch_remove_log_entries([trace.layer_list[-1]])
    fourth = _engine.solve(trace)
    assert fourth is not third
    assert calls == 3
