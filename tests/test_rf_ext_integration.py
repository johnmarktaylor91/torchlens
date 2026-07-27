"""Final end-to-end integration gates for influence geometry."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
import importlib
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import FieldPolicy
from torchlens.receptive_field import _rules
from torchlens.receptive_field._errors import ReceptiveFieldUnavailableError
from torchlens.receptive_field._query import _IndexSet
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import (
    ReceptiveFieldDirection,
    ReceptiveFieldStatus,
    ReceptiveFieldValidationStatus,
)
from torchlens.receptive_field._validation import check_geometric_metadata_invariants


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


def _capture(model: nn.Module, inputs: torch.Tensor, *, backward_ready: bool = False) -> object:
    """Capture a model with the payload policy needed by influence queries."""

    return tl.trace(
        model.eval(),
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=backward_ready),
        save_mode="reference" if backward_ready else "copy",
    )


def _ops(trace: object, name: str) -> list[object]:
    """Return captured operations matching one raw function name."""

    return [op for op in trace.layer_list if op.func_name == name]  # type: ignore[union-attr]


def _window_bounds(box: object) -> tuple[tuple[int, int], ...]:
    """Return clipped bounds for every windowed axis in a geometric box."""

    bounds = []
    for axis in box.axes:  # type: ignore[union-attr]
        if axis.kind != "windowed":
            continue
        assert axis.clipped_start is not None
        assert axis.clipped_stop is not None
        bounds.append((int(axis.clipped_start), int(axis.clipped_stop)))
    return tuple(bounds)


def _positive_spatial_chain(depth: int) -> nn.Module:
    """Build a positive convolutional chain with an optional pooling stage."""

    layers: list[nn.Module] = [nn.Conv2d(1, 2, 3, padding=1, bias=False), nn.ReLU()]
    if depth == 3:
        layers.append(nn.AvgPool2d(3, stride=1, padding=1))
    layers.append(nn.Conv2d(2, 1, 3, padding=1, bias=False))
    model = nn.Sequential(*layers)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    return model


@pytest.mark.parametrize(
    ("depth", "input_rf_size", "layer_rf_size"),
    [(2, 5, 3), (3, 7, 5)],
)
def test_real_model_matrix_covers_receptive_projective_and_layer_to_layer(
    depth: int, input_rf_size: int, layer_rf_size: int
) -> None:
    """Exercise both directions and an internal source across spatial model variants."""

    trace = _capture(_positive_spatial_chain(depth), torch.ones(1, 1, 9, 9))
    source = trace.input_ops[0]  # type: ignore[union-attr]
    first_conv, target = _ops(trace, "conv2d")

    receptive = target.receptive_field.at("center", input=source)  # type: ignore[union-attr]
    layer_to_layer = target.receptive_field.at(  # type: ignore[union-attr]
        "center", source=first_conv
    )
    projective = source.projective_field.at((4, 4), target=target)  # type: ignore[union-attr]

    assert receptive.status is ReceptiveFieldStatus.EXACT
    assert layer_to_layer.status is ReceptiveFieldStatus.EXACT
    assert projective.status is ReceptiveFieldStatus.EXACT
    assert tuple(stop - start for start, stop in _window_bounds(receptive)) == (
        input_rf_size,
        input_rf_size,
    )
    assert tuple(stop - start for start, stop in _window_bounds(layer_to_layer)) == (
        layer_rf_size,
        layer_rf_size,
    )
    assert tuple(stop - start for start, stop in _window_bounds(projective)) == (
        input_rf_size,
        input_rf_size,
    )
    assert not trace.receptive_fields(level="layer").to_pandas().empty  # type: ignore[union-attr]
    assert not trace.projective_fields(level="layer").to_pandas().empty  # type: ignore[union-attr]


def _register_resnet_passthrough_rules() -> None:
    """Make torchvision's normalized in-place residual operations explicitly exact."""

    @_rules.register_rf_rule("relu", "iadd", "none", replace=True)
    def residual_passthrough(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Preserve geometry through residual arithmetic and the output wrapper."""

        return context.passthrough()


def _resnet18_merge_goldens() -> tuple[tuple[int, int], ...]:
    """Return independently derived ResNet-18 merge size/jump goldens."""

    size, jump = 7, 2
    size, jump = size + 2 * jump, jump * 2
    goldens: list[tuple[int, int]] = []
    for stage, blocks in enumerate((2, 2, 2, 2)):
        for block in range(blocks):
            stride = 2 if stage > 0 and block == 0 else 1
            main_size = size + 2 * jump
            main_jump = jump * stride
            main_size += 2 * main_jump
            size, jump = main_size, main_jump
            goldens.append((size, jump))
    return tuple(goldens)


@pytest.mark.heavy
def test_torchvision_resnet18_matrix_and_branch_differentiator() -> None:
    """Keep real ResNet receptive, internal, and input-to-output projective geometry live."""

    torchvision_models = pytest.importorskip("torchvision.models")
    _register_resnet_passthrough_rules()
    trace = _capture(
        torchvision_models.resnet18(weights=None),
        torch.ones(1, 3, 32, 32),
    )
    source = trace.input_ops[0]  # type: ignore[union-attr]
    target = trace.output_ops[0]  # type: ignore[union-attr]
    merges = _ops(trace, "__iadd__")
    observed = []
    for merge in merges:
        descriptor = next(iter(merge.receptive_field.per_input.values()))  # type: ignore[union-attr]
        observed.append((descriptor.size[0], int(descriptor.jump[0])))

    expected = _resnet18_merge_goldens()
    assert expected == (
        (27, 4),
        (43, 4),
        (67, 8),
        (99, 8),
        (147, 16),
        (211, 16),
        (307, 32),
        (435, 32),
    )
    assert tuple(observed) == expected

    stage_input = _ops(trace, "max_pool2d")[0]
    within_stage = merges[0].receptive_field.at((4, 4), source=stage_input)  # type: ignore[union-attr]
    assert within_stage.status is ReceptiveFieldStatus.EXACT
    assert _window_bounds(within_stage) == ((2, 7), (2, 7))

    descriptor = next(iter(source.projective_field.per_input.values()))  # type: ignore[union-attr]
    assert descriptor.direction is ReceptiveFieldDirection.PROJECTIVE
    assert descriptor.source_key == source.label  # type: ignore[union-attr]
    assert descriptor.target_key == target.io_role  # type: ignore[union-attr]
    assert descriptor.status not in {
        ReceptiveFieldStatus.UNKNOWN,
        ReceptiveFieldStatus.UNSUPPORTED,
    }


def _torchvision_basic_block(*, backward_ready: bool) -> tuple[nn.Module, object]:
    """Capture a deterministic torchvision BasicBlock with in-place ReLU enabled."""

    torchvision = pytest.importorskip("torchvision")
    block = torchvision.models.resnet.BasicBlock(4, 4).eval()
    with torch.no_grad():
        for module in block.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.fill_(0.05)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.fill_(1.0)
                module.bias.zero_()
    inputs = torch.ones(1, 4, 8, 8, requires_grad=backward_ready)
    return block, _capture(block, inputs, backward_ready=backward_ready)


@pytest.mark.heavy
def test_inplace_relu_projective_double_vjp_succeeds_on_basic_block() -> None:
    """Require a nonempty projective gradient through torchvision's in-place ReLU path."""

    block, trace = _torchvision_basic_block(backward_ready=True)
    source = trace.input_ops[0]  # type: ignore[union-attr]
    target = trace.output_ops[0]  # type: ignore[union-attr]

    result = source.projective_field.gradient(  # type: ignore[union-attr]
        (0, 0, 4, 4), target=target
    )

    assert block.relu.inplace  # type: ignore[union-attr]
    assert result.direction is ReceptiveFieldDirection.PROJECTIVE
    assert result.grad.shape == torch.Size((1, 4, 8, 8))
    assert torch.count_nonzero(result.support_mask).item() > 0
    assert torch.isfinite(result.grad).all()


@pytest.mark.heavy
def test_basic_block_geometric_adjoint_battery_is_exact() -> None:
    """Check forward and backward membership equivalence with zero coordinate tolerance."""

    _register_resnet_passthrough_rules()
    _, trace = _torchvision_basic_block(backward_ready=False)
    source = trace.input_ops[0]  # type: ignore[union-attr]
    target = _ops(trace, "__iadd__")[-1]
    source_units = ((0, 0), (1, 6), (4, 4), (7, 7))
    target_units = ((0, 0), (2, 5), (4, 4), (7, 7))

    for source_unit in source_units:
        forward = source.projective_field.at(source_unit, target=target)  # type: ignore[union-attr]
        assert forward.status is ReceptiveFieldStatus.EXACT
        forward_bounds = _window_bounds(forward)
        for target_unit in target_units:
            backward = target.receptive_field.at(target_unit, input=source)  # type: ignore[union-attr]
            assert backward.status is ReceptiveFieldStatus.EXACT
            backward_bounds = _window_bounds(backward)
            source_in_backward = all(
                start <= coordinate < stop
                for coordinate, (start, stop) in zip(source_unit, backward_bounds, strict=True)
            )
            target_in_forward = all(
                start <= coordinate < stop
                for coordinate, (start, stop) in zip(target_unit, forward_bounds, strict=True)
            )
            assert source_in_backward is target_in_forward


def test_default_validation_runs_cheap_rf_metadata_without_autograd() -> None:
    """Keep always-on RF metadata checks in ordinary forward validation."""

    model = nn.Conv2d(1, 1, 3, padding=1)
    with (
        mock.patch(
            "torchlens.receptive_field._validation.check_geometric_metadata_invariants",
            wraps=check_geometric_metadata_invariants,
        ) as geometry_check,
        mock.patch("torch.autograd.grad", side_effect=AssertionError("autograd invoked")),
    ):
        result = tl.validate(model, torch.ones(1, 1, 5, 5), scope="forward")

    assert result is True
    geometry_check.assert_called_once()


def test_validate_rf_scope_runs_sampled_real_model_cross_checks() -> None:
    """Run both sampled influence directions through the consolidated validator."""

    model = _positive_spatial_chain(2)
    results = tl.validate(model, torch.ones(1, 1, 7, 7), scope="receptive_field")

    assert results
    assert {result.direction for result in results} == {
        ReceptiveFieldDirection.RECEPTIVE,
        ReceptiveFieldDirection.PROJECTIVE,
    }
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in results)


@pytest.mark.heavy
def test_validate_rf_scope_gracefully_skips_transformer_non_grid_results() -> None:
    """Classify inapplicable transformer projective geometry without crashing or failing."""

    model = nn.TransformerEncoderLayer(d_model=8, nhead=2, batch_first=True).eval()
    results = tl.validate(model, torch.randn(1, 5, 8), scope="receptive_field")

    assert results
    assert any(result.status is ReceptiveFieldValidationStatus.INDETERMINATE for result in results)
    assert all(result.status is not ReceptiveFieldValidationStatus.FAIL for result in results)


def test_folded_validate_tripwire_catches_injected_inconsistency() -> None:
    """Make the consolidated RF scope fail on a deliberately undersized rule."""

    @_rules.register_rf_rule("conv2d", replace=True)
    def undersized_convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Falsely describe a 3x3 convolution as pointwise."""

        return context.window(kernel=1, stride=1, padding=0, dilation=1)

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    results = tl.validate(model, torch.ones(1, 1, 5, 5), scope="receptive_field")
    failures = [
        result for result in results if result.status is ReceptiveFieldValidationStatus.FAIL
    ]

    assert {failure.direction for failure in failures} == {
        ReceptiveFieldDirection.RECEPTIVE,
        ReceptiveFieldDirection.PROJECTIVE,
    }
    assert all(failure.n_violations > 0 for failure in failures)


class _AddZero(nn.Module):
    """Expose a captured add operation for membership-budget testing."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Add a scalar zero without changing values."""

        return value + 0


def test_projective_membership_budget_widens_instead_of_blowing_up() -> None:
    """Return an honest upper bound when a per-unit candidate sweep exceeds its budget."""

    @_rules.register_rf_rule("add", replace=True)
    def broadcast_from_zero(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Map every target candidate back to source coordinate zero."""

        def backward(axis: int, output_set: _IndexSet) -> tuple[_IndexSet, bool]:
            """Return the constant source coordinate for every target set."""

            _ = axis, output_set
            return _IndexSet.singleton(0), True

        edges = (((Fraction(0), Fraction(0)), (Fraction(0), Fraction(0))),)
        return context.window_edges(edges, exact=True, map_index_set=backward)

    trace = tl.trace(_AddZero(), torch.ones(1, 1, 5000))
    source = trace.input_ops[0]
    target = trace.output_ops[0]
    result = source.projective_field.at((0,), target=target)

    assert result.status is ReceptiveFieldStatus.UPPER_BOUND
    assert not result.exact
    assert _window_bounds(result) == ((0, 5000),)


def test_source_target_caches_reuse_and_drop_across_portable_round_trip(
    tmp_path: Path,
) -> None:
    """Reuse both endpoint LRUs live, then drop and rebuild them after portable load."""

    trace = tl.trace(_positive_spatial_chain(2), torch.ones(1, 1, 7, 7))
    source, target = _ops(trace, "conv2d")

    target.receptive_field.at((3, 3), source=source)  # type: ignore[union-attr]
    source_cache = trace.__dict__["_rf_source_solutions"]
    first_source_solution = source_cache[source.label][2]  # type: ignore[union-attr]
    target.receptive_field.at((3, 3), source=source)  # type: ignore[union-attr]
    assert source_cache[source.label][2] is first_source_solution  # type: ignore[union-attr]

    source.projective_field.at((3, 3), target=target)  # type: ignore[union-attr]
    target_cache = trace.__dict__["_rf_target_solutions"]
    target_key = (target.label,)  # type: ignore[union-attr]
    first_target_solution = target_cache[target_key][2]
    source.projective_field.at((3, 3), target=target)  # type: ignore[union-attr]
    assert target_cache[target_key][2] is first_target_solution
    assert type(trace).PORTABLE_STATE_SPEC["_rf_source_solutions"] is FieldPolicy.DROP
    assert type(trace).PORTABLE_STATE_SPEC["_rf_target_solutions"] is FieldPolicy.DROP

    path = tmp_path / "rf-endpoint-caches.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    loaded_source = loaded.ops[source.label]  # type: ignore[union-attr]
    loaded_target = loaded.ops[target.label]  # type: ignore[union-attr]

    assert loaded.__dict__.get("_rf_source_solutions") is None
    assert loaded.__dict__.get("_rf_target_solutions") is None
    assert loaded_target.receptive_field.at((3, 3), source=loaded_source).exact
    assert loaded_source.projective_field.at((3, 3), target=loaded_target).exact
    with pytest.raises(ReceptiveFieldUnavailableError, match="backward_ready=True"):
        loaded_source.projective_field.gradient((0, 0, 3, 3), target=loaded_target)


@pytest.mark.backend_tinygrad
def test_non_torch_backend_keeps_geometry_and_gates_both_gradients() -> None:
    """Keep rule-backed geometry backend-neutral and empirical methods torch-only."""

    tinygrad = pytest.importorskip("tinygrad")

    @_rules.register_rf_rule("add", replace=True)
    def tinygrad_add(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Describe tinygrad scalar addition as pointwise geometry."""

        return context.passthrough()

    def add_one(value: object) -> object:
        """Add one through the tinygrad backend."""

        return value + 1.0  # type: ignore[operator]

    trace = tl.trace(add_one, tinygrad.Tensor([1.0, 2.0, 3.0]), backend="tinygrad")
    source = trace.input_ops[0]
    target = _ops(trace, "add")[-1]
    receptive = next(iter(target.receptive_field.per_input.values()))
    projective = next(iter(source.projective_field.per_input.values()))

    assert receptive.status is ReceptiveFieldStatus.EXACT
    assert projective.status is ReceptiveFieldStatus.EXACT
    assert receptive.layout.axis_kinds == ("pointwise",)
    assert projective.layout.axis_kinds == ("pointwise",)
    with pytest.raises(tl.receptive_field.BackendUnsupportedError, match="tinygrad"):
        target.receptive_field.gradient((0,))
    with pytest.raises(tl.receptive_field.BackendUnsupportedError, match="tinygrad"):
        source.projective_field.gradient((0,), target=target)
