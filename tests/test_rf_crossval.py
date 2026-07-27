"""Gradient-inside-geometric receptive-field tripwire tests."""

from __future__ import annotations

from collections.abc import Iterator
import importlib

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _rules
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import ReceptiveFieldValidationStatus
from torchlens.receptive_field._validation import cross_validate, validate_receptive_field_trace


_PACK: dict[str, object] | None = None


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install the built-in RF rules while preserving registry isolation."""

    global _PACK
    original = dict(_rules._RF_RULES)
    original_epoch = _rules._RF_RULES_EPOCH
    _rules._RF_RULES.clear()
    if _PACK is None:
        module = importlib.import_module("torchlens.receptive_field.rules")
        if not _rules._RF_RULES:
            for name in module.__all__:
                importlib.reload(getattr(module, name))
        _PACK = dict(_rules._RF_RULES)
    else:
        _rules._RF_RULES.update(_PACK)
    try:
        yield
    finally:
        _rules._RF_RULES.clear()
        _rules._RF_RULES.update(original)
        _rules._RF_RULES_EPOCH = original_epoch


def _op(trace: object, name: str) -> object:
    """Return the last captured operation with a raw function name."""

    matches = [item for item in trace.layer_list if item.func_name == name]  # type: ignore[union-attr]
    assert matches
    return matches[-1]


def _input_op(trace: object) -> object:
    """Return the sole model-input operation from a one-input trace."""

    return next(item for item in trace.layer_list if item.is_input)  # type: ignore[union-attr]


def _trace(model: nn.Module, inputs: torch.Tensor) -> object:
    """Capture a graph-connected model suitable for repeated RF probes."""

    capture = tl.options.CaptureOptions(backward_ready=True)
    return tl.trace(model, inputs, capture=capture, save_mode="reference")


def test_positive_conv_is_contained_and_tight() -> None:
    """Require exact support equality so gratuitously loose rules cannot pass."""

    @_rules.register_rf_rule("softplus", replace=True)
    def softplus_passthrough(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Describe the smooth pointwise activation used by the tightness oracle."""

        return context.passthrough()

    model = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1, bias=False), nn.Softplus())
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    trace = _trace(model, torch.ones(1, 1, 7, 7, requires_grad=True))
    result = _op(trace, "softplus").receptive_field.check((0, 0, 3, 3), input=_input_op(trace))

    assert result.status is ReceptiveFieldValidationStatus.PASS
    assert result.slack_per_axis == (0, 0, 0, 0)
    gradient = next(iter(result.gradient.values()))
    box = next(iter(result.geometric.values()))
    assert gradient.support_ranges == tuple(
        (axis.clipped_start, axis.clipped_stop)
        if axis.clipped_start is not None
        else (result.unit[index], result.unit[index] + 1)
        for index, axis in enumerate(box.axes)
    )


def test_deliberately_undersized_rule_fires_negative_tripwire() -> None:
    """Fail loudly when a false 1x1 rule describes a real 3x3 convolution."""

    @_rules.register_rf_rule("conv2d", replace=True)
    def undersized_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Deliberately lie about convolution support for the negative oracle."""

        return context.window(kernel=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1))

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trace = _trace(model, torch.ones(1, 1, 5, 5, requires_grad=True))
    result = _op(trace, "conv2d").receptive_field.check((0, 0, 2, 2), input=_input_op(trace))

    assert result.status is ReceptiveFieldValidationStatus.FAIL
    assert result.n_violations == 8
    assert len(result.violations) == 8
    assert result.per_axis_excess == (0, 0, 1, 1)
    assert result.slack_per_axis is None


@pytest.mark.parametrize("validation_surface", ["check", "cross_validate", "validate_trace"])
def test_validation_thresholds_cannot_suppress_real_support(validation_surface: str) -> None:
    """Keep a false convolution rule failing despite caller-supplied thresholds."""

    @_rules.register_rf_rule("conv2d", replace=True)
    def undersized_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Deliberately report a one-by-one field for a positive three-by-three kernel."""

        return context.window(kernel=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1))

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trace = _trace(model, torch.ones(1, 1, 5, 5, requires_grad=True))
    target = _op(trace, "conv2d")
    input_op = _input_op(trace)
    kwargs = {"atol": 2.0, "rtol": 2.0}
    if validation_surface == "check":
        results = [target.receptive_field.check((0, 0, 2, 2), input=input_op, **kwargs)]
    elif validation_surface == "cross_validate":
        results = cross_validate(trace, ops=[target], units=(0, 0, 2, 2), inputs=input_op, **kwargs)
    else:
        results = validate_receptive_field_trace(
            trace,
            ops=[target],
            units=(0, 0, 2, 2),
            inputs=input_op,
            **kwargs,
        )

    assert len(results) == 1
    assert results[0].status is ReceptiveFieldValidationStatus.FAIL
    assert results[0].n_violations == 8


@pytest.mark.parametrize("direction", ["receptive", "projective"])
def test_verify_reports_containment_and_empirical_adjoint_equality(direction: str) -> None:
    """Compose both model-facing RF diagnostics in the consolidated verifier."""

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trace = _trace(model, torch.ones(1, 1, 5, 5, requires_grad=True))
    target = _op(trace, "conv2d")

    if direction == "receptive":
        report = tl.receptive_field.verify(
            trace,
            ops=[target],
            units=(0, 0, 2, 2),
            inputs=_input_op(trace),
        )
    else:
        report = tl.receptive_field.verify(
            trace,
            ops=[_input_op(trace)],
            units=(0, 0, 2, 2),
            direction=direction,
            target=target,
        )

    assert len(report.containment) == 1
    assert report.containment[0].status is ReceptiveFieldValidationStatus.PASS
    assert len(report.empirical_adjoint) == 1
    assert report.empirical_adjoint[0].passed
    assert report.passed


class _InstanceNorm(nn.Module):
    """InstanceNorm fixture with selectable running-stat behavior."""

    def __init__(self, *, running: bool) -> None:
        """Create an affine-free normalization fixture."""

        super().__init__()
        self.norm = nn.InstanceNorm2d(2, affine=False, track_running_stats=running)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply instance normalization."""

        return self.norm(value)


@pytest.mark.parametrize(
    ("model", "name", "training", "expected_cross_batch"),
    [
        (nn.BatchNorm2d(2, affine=False), "batch_norm", False, "none"),
        (_InstanceNorm(running=True), "instance_norm", False, "none"),
        (nn.BatchNorm2d(2, affine=False), "batch_norm", True, "geometric"),
        (nn.GroupNorm(1, 2, affine=False), "group_norm", True, "none"),
        (nn.LayerNorm((3, 3), elementwise_affine=False), "layer_norm", True, "none"),
    ],
)
def test_normalization_matrix_contains_empirical_support(
    model: nn.Module, name: str, training: bool, expected_cross_batch: str
) -> None:
    """Validate transparent and statistics-global normalization rules uniformly."""

    @_rules.register_rf_rule(name, replace=True)
    def normalization(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Declare the fixture's known captured-mode normalization geometry."""

        if name in {"batch_norm", "instance_norm"} and not training:
            return context.passthrough()
        axes = (0, 2, 3) if name == "batch_norm" else (1, 2, 3)
        return context.full(axes=axes)

    model = nn.Sequential(nn.Conv2d(2, 2, 1, bias=False), model)
    with torch.no_grad():
        model[0].weight.fill_(0.5)
    model.train(training)
    inputs = torch.linspace(0.1, 7.2, 72).reshape(4, 2, 3, 3).requires_grad_()
    trace = _trace(model, inputs)
    result = _op(trace, name).receptive_field.check((0, 0, 1, 1), input=_input_op(trace))

    assert result.status is ReceptiveFieldValidationStatus.PASS
    assert result.cross_batch == expected_cross_batch
    if expected_cross_batch == "geometric":
        gradient = next(iter(result.gradient.values()))
        box = next(iter(result.geometric.values()))
        assert gradient.cross_batch_influence
        assert box.axes[0].clipped_start == 0
        assert box.axes[0].clipped_stop == 4


class _SqueezeExcite(nn.Module):
    """Local-looking multiply with a spatially global broadcast branch."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Modulate each site by its channel's global mean."""

        scale = functional.softplus(value.mean(dim=(-2, -1), keepdim=True))
        return value * scale


def test_squeeze_excite_broadcast_branch_is_contained() -> None:
    """Contain global branch influence at a local-looking arithmetic merge."""

    inputs = torch.linspace(0.1, 3.2, 32).reshape(1, 2, 4, 4).requires_grad_()
    trace = _trace(_SqueezeExcite(), inputs)
    result = _op(trace, "__mul__").receptive_field.check((0, 0, 2, 2), input=_input_op(trace))

    assert result.status is ReceptiveFieldValidationStatus.PASS
    gradient = next(iter(result.gradient.values()))
    assert gradient.support_ranges == ((0, 1), (0, 1), (0, 4), (0, 4))


class _UnknownAttention(nn.Module):
    """Attention-like operation assigned deliberately unknown mask geometry."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply a graph-connected softmax mixing operation."""

        return torch.softmax(value, dim=-1)


@pytest.mark.parametrize("taint", ["unknown", "data_dependent"])
def test_unknown_and_data_dependent_geometry_are_indeterminate(taint: str) -> None:
    """Never convert unavailable geometric truth into either pass or fail."""

    @_rules.register_rf_rule("softmax", replace=True)
    def unknown_mask(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Represent an unavailable attention mask as unknown geometry."""

        if taint == "unknown":
            return context.unknown("attention mask geometry was not captured")
        return context.data_dependent("attention mask geometry depends on runtime data")

    trace = _trace(_UnknownAttention(), torch.randn(1, 2, 4, requires_grad=True))
    result = _op(trace, "softmax").receptive_field.check((0, 0, 1), input=_input_op(trace))

    assert result.status is ReceptiveFieldValidationStatus.INDETERMINATE
    assert result.n_violations == 0
    assert not result.passed


def test_undeclared_training_batch_coupling_fails() -> None:
    """Reject cross-sample influence when a deliberately false rule claims transparency."""

    @_rules.register_rf_rule("batch_norm", replace=True)
    def false_transparent_batch_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Deliberately omit training-statistics coupling for the tripwire oracle."""

        return context.passthrough()

    model = nn.Sequential(
        nn.Conv2d(1, 1, 1, bias=False),
        nn.BatchNorm2d(1, affine=False, track_running_stats=False),
    ).train()
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    inputs = torch.linspace(0.1, 3.2, 32).reshape(2, 1, 4, 4).requires_grad_()
    trace = _trace(model, inputs)
    result = _op(trace, "batch_norm").receptive_field.check((0, 0, 1, 1), input=_input_op(trace))

    assert result.status is ReceptiveFieldValidationStatus.FAIL
    assert result.cross_batch == "undeclared"
    assert result.n_violations > 0


class _SmallConvNet(nn.Module):
    """Small residual-free model-zoo convolution fixture."""

    def __init__(self) -> None:
        """Create two positive convolutions separated by smooth activations."""

        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 2, 3, padding=1, bias=False),
            nn.Softplus(),
            nn.Conv2d(2, 2, 3, padding=1, bias=False),
            nn.Softplus(),
        )
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.fill_(0.25)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Run the convolution stack."""

        return self.layers(value)


def test_cross_validate_small_model_center_and_corners() -> None:
    """Sweep complete center and corner indices while retaining the backward graph."""

    @_rules.register_rf_rule("softplus", replace=True)
    def softplus_passthrough(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Preserve geometry through the model-zoo fixture's smooth activation."""

        return context.passthrough()

    trace = _trace(_SmallConvNet(), torch.ones(1, 1, 7, 7, requires_grad=True))
    target = _op(trace, "conv2d")
    results = cross_validate(trace, ops=[target], units="corners", batch_index=0)

    assert len(results) == 4
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in results)


def test_torchvision_resnet18_basic_block_tripwire() -> None:
    """Exercise a real residual basic block through capture and backward probing."""

    @_rules.register_rf_rule("relu_", "__iadd__", replace=True)
    def residual_arithmetic(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Preserve and merge geometry through the block's in-place arithmetic."""

        return context.passthrough()

    @_rules.register_rf_rule("batch_norm", replace=True)
    def eval_batch_norm(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Describe the fixture's frozen-statistics BatchNorm calls as transparent."""

        return context.passthrough()

    torchvision = pytest.importorskip("torchvision")
    block = torchvision.models.resnet.BasicBlock(4, 4).eval()
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.fill_(0.05)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.fill_(1.0)
                module.bias.zero_()
    trace = _trace(block, torch.ones(1, 4, 8, 8, requires_grad=True))
    target = _op(trace, "__iadd__")
    result = target.receptive_field.check((0, 0, 4, 4), input=_input_op(trace))

    assert result.status is ReceptiveFieldValidationStatus.PASS
