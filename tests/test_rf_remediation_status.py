"""Status-honesty goldens for receptive-field rule remediation."""

from __future__ import annotations

from collections.abc import Iterator
import importlib

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

import torchlens as tl
from torchlens.receptive_field._types import (
    ReceptiveFieldStatus,
    ReceptiveFieldValidationStatus,
)


_rf_package = importlib.import_module("torchlens.receptive_field")
_rules = importlib.import_module("torchlens.receptive_field._rules")
setattr(_rf_package, "_rules", _rules)
cross_validate = importlib.import_module("torchlens.receptive_field._validation").cross_validate
_PACK: dict[str, object] | None = None


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install built-in RF rules while preserving registry isolation."""

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
    """Return the last captured operation with the requested function name."""

    return next(
        item
        for item in reversed(trace.layer_list)  # type: ignore[union-attr]
        if item.func_name == name
    )


def _input_op(trace: object) -> object:
    """Return the canonical input operation from a one-input trace."""

    return next(item for item in trace.layer_list if item.is_input)  # type: ignore[union-attr]


def _trace(model: nn.Module, inputs: torch.Tensor) -> object:
    """Capture a model with autograd history retained for RF cross-validation."""

    return tl.trace(
        model,
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )


def test_depthwise_convolution_has_tight_exact_channel_and_spatial_geometry() -> None:
    """Keep depthwise channels pointwise and spatial support exactly windowed."""

    model = nn.Conv2d(4, 4, 3, padding=1, groups=4, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trace = _trace(model, torch.ones(2, 4, 5, 5, requires_grad=True))
    target = _op(trace, "conv2d")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert tuple(axis.kind for axis in descriptor.axes) == (
        "pointwise",
        "pointwise",
        "windowed",
        "windowed",
    )
    result = cross_validate(
        trace,
        ops=[target],
        units=(0, 0, 2, 2),
        inputs=_input_op(trace),
    )[0]
    assert result.status is ReceptiveFieldValidationStatus.PASS


@pytest.mark.parametrize("pool", [nn.MaxPool2d(2), nn.AvgPool2d(2)])
def test_pooling_preserves_exact_pointwise_channels(pool: nn.Module) -> None:
    """Keep bare max and average pooling channel-pointwise and gradient-valid."""

    trace = _trace(pool, torch.randn(1, 3, 6, 6, requires_grad=True))
    target = next(item for item in trace.layer_list if "pool2d" in str(item.func_name))
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert tuple(axis.kind for axis in descriptor.axes) == (
        "pointwise",
        "pointwise",
        "windowed",
        "windowed",
    )
    result = cross_validate(trace, ops=[target], units=(0, 2, 1, 1))[0]
    assert result.status is ReceptiveFieldValidationStatus.PASS


def test_depthwise_convolution_then_pooling_remains_channel_pointwise() -> None:
    """Preserve a single-channel route through depthwise convolution and max pooling."""

    model = nn.Sequential(
        nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False),
        nn.MaxPool2d(2),
    )
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    trace = _trace(model, torch.randn(1, 3, 6, 6, requires_grad=True))
    target = next(item for item in trace.layer_list if item.func_name == "max_pool2d")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert tuple(axis.kind for axis in descriptor.axes) == (
        "pointwise",
        "pointwise",
        "windowed",
        "windowed",
    )
    result = cross_validate(trace, ops=[target], units=(0, 2, 1, 1))[0]
    assert result.status is ReceptiveFieldValidationStatus.PASS


def test_grouped_convolution_keeps_spatial_windows_with_channel_upper_bound() -> None:
    """Limit grouped-convolution degradation to a containing channel-axis bound."""

    model = nn.Conv2d(4, 4, 3, padding=1, groups=2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    trace = _trace(model, torch.ones(1, 4, 5, 5, requires_grad=True))
    target = _op(trace, "conv2d")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
    assert tuple(axis.kind for axis in descriptor.axes) == (
        "pointwise",
        "full",
        "windowed",
        "windowed",
    )
    assert descriptor.axes[1].exact is False
    assert descriptor.axes[2].size == 3
    result = target.receptive_field.check((0, 1, 2, 2))
    assert result.status is ReceptiveFieldValidationStatus.PASS


class _DistinctAttention(nn.Module):
    """Scaled dot-product attention with separately traceable Q, K, and V inputs."""

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Apply unmasked scaled dot-product attention."""

        return functional.scaled_dot_product_attention(query, key, value)


class _GroupedQueryAttention(nn.Module):
    """Scaled dot-product attention with fewer key/value heads than query heads."""

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Apply grouped-query scaled dot-product attention."""

        return functional.scaled_dot_product_attention(query, key, value, enable_gqa=True)


def test_sdpa_feature_axis_upper_bound_contains_query_gradients() -> None:
    """Include every query feature contributing through attention-score dot products."""

    inputs = tuple(torch.randn(1, 2, 4, 3, requires_grad=True) for _ in range(3))
    trace = tl.trace(
        _DistinctAttention(),
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    target = _op(trace, "scaled_dot_product_attention")
    descriptor = target.receptive_field._descriptor(trace.input_ops[0])

    assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
    assert tuple(axis.kind for axis in descriptor.axes) == (
        "pointwise",
        "pointwise",
        "full",
        "full",
    )
    results = cross_validate(
        trace,
        ops=[target],
        units=(0, 0, 1, 0),
        inputs=list(trace.input_ops),
    )
    assert len(results) == 3
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in results)


def test_sdpa_grouped_query_attention_widens_key_value_heads() -> None:
    """Contain key/value support when grouped-query attention reuses fewer heads."""

    inputs = (
        torch.randn(1, 4, 3, 2, requires_grad=True),
        torch.randn(1, 2, 3, 2, requires_grad=True),
        torch.randn(1, 2, 3, 2, requires_grad=True),
    )
    trace = tl.trace(
        _GroupedQueryAttention(),
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    target = _op(trace, "scaled_dot_product_attention")
    key_value_inputs = list(trace.input_ops)[1:]

    for source in key_value_inputs:
        descriptor = target.receptive_field._descriptor(source)
        assert tuple(axis.kind for axis in descriptor.axes) == (
            "pointwise",
            "full",
            "full",
            "full",
        )
    results = cross_validate(
        trace,
        ops=[target],
        units=(0, 3, 1, 0),
        inputs=key_value_inputs,
    )
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in results)


def test_sdpa_broadcast_widens_key_value_batch_and_head_axes() -> None:
    """Contain key/value support broadcast across query batches and heads."""

    inputs = (
        torch.randn(2, 4, 3, 2, requires_grad=True),
        torch.randn(1, 1, 3, 2, requires_grad=True),
        torch.randn(1, 1, 3, 2, requires_grad=True),
    )
    trace = tl.trace(
        _DistinctAttention(),
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    target = _op(trace, "scaled_dot_product_attention")
    key_value_inputs = list(trace.input_ops)[1:]

    for source in key_value_inputs:
        descriptor = target.receptive_field._descriptor(source)
        assert tuple(axis.kind for axis in descriptor.axes) == (
            "full",
            "full",
            "full",
            "full",
        )
    results = cross_validate(
        trace,
        ops=[target],
        units=(1, 3, 1, 0),
        inputs=key_value_inputs,
    )
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in results)


def test_embedding_bag_validation_returns_tri_state_rows_without_raising() -> None:
    """Report integer-index EmbeddingBag validation as honest indeterminate rows."""

    rows = tl.validate(
        nn.EmbeddingBag(10, 4, mode="mean"),
        (torch.tensor([1, 2, 4, 5, 3]), torch.tensor([0, 3])),
        scope="receptive_field",
    )

    assert rows
    assert all(
        row.status
        in {
            ReceptiveFieldValidationStatus.PASS,
            ReceptiveFieldValidationStatus.INDETERMINATE,
        }
        for row in rows
    )
    assert any(row.status is ReceptiveFieldValidationStatus.INDETERMINATE for row in rows)


class _TrailingReduction(nn.Module):
    """Convolution followed by a reduction after one surviving spatial axis."""

    def __init__(self) -> None:
        """Create a positive convolution for exact empirical support."""

        super().__init__()
        self.convolution = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            self.convolution.weight.fill_(1.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Reduce the final spatial dimension after convolution."""

        return self.convolution(value).mean(dim=-1)


class _PaddedConvolution(nn.Module):
    """Positive convolution with an explicitly empty constant-pad corner."""

    def __init__(self) -> None:
        """Create a positive convolution for exact empirical support."""

        super().__init__()
        self.convolution = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        with torch.no_grad():
            self.convolution.weight.fill_(1.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Pad the convolution result by one cell on every spatial side."""

        return functional.pad(self.convolution(value), (1, 1, 1, 1))


def test_trailing_reduction_and_pad_corner_validate_without_raising() -> None:
    """Return bounded PASS results for reduced-window and empty-pad geometries."""

    reduction_results = tl.validate(
        _TrailingReduction(), torch.ones(1, 1, 8, 8), scope="receptive_field"
    )
    assert reduction_results
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in reduction_results)

    trace = _trace(_PaddedConvolution(), torch.ones(1, 1, 5, 5, requires_grad=True))
    target = _op(trace, "pad")
    box = target.receptive_field.at((0, 0))
    assert box.empty
    result = target.receptive_field.check((0, 0, 0, 0))
    assert result.status is ReceptiveFieldValidationStatus.PASS

    padded_results = tl.validate(
        _PaddedConvolution(), torch.ones(1, 1, 5, 5), scope="receptive_field"
    )
    assert padded_results
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in padded_results)


def test_embedding_resolves_exact_index_position_geometry() -> None:
    """Resolve embedding rank expansion without unknown passthrough taint."""

    trace = tl.trace(nn.Embedding(10, 4), torch.tensor([[1, 2, 3], [4, 5, 6]]))
    target = _op(trace, "embedding")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.EXACT
    assert tuple((axis.kind, axis.output_axis) for axis in descriptor.axes) == (
        ("pointwise", 0),
        ("pointwise", 1),
    )


class _Reduction(nn.Module):
    """Middle-axis reduction fixture."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Reduce channels while preserving batch and spatial identities."""

        return value.mean(dim=1)


def test_reduction_preserves_surviving_axis_coordinates() -> None:
    """Keep batch and spatial coordinate languages after a removed middle axis."""

    trace = _trace(_Reduction(), torch.ones(2, 4, 3, 5, requires_grad=True))
    target = _op(trace, "mean")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.WHOLE_INPUT
    assert tuple(axis.kind for axis in descriptor.axes) == (
        "pointwise",
        "full",
        "pointwise",
        "pointwise",
    )
    assert tuple(axis.output_axis for axis in descriptor.axes) == (0, None, 1, 2)
    result = cross_validate(
        trace,
        ops=[target],
        units=(1, 2, 4),
        inputs=_input_op(trace),
    )[0]
    assert result.status is ReceptiveFieldValidationStatus.PASS


class _Cumulative(nn.Module):
    """Dimensioned cumulative-operation fixture."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Accumulate over height rather than the trailing axis."""

        return value.cumsum(dim=2)


def test_cumulative_reads_dimension_and_bounds_that_axis() -> None:
    """Use a containing bound on the selected cumulative axis only."""

    trace = _trace(_Cumulative(), torch.ones(2, 4, 3, 5, requires_grad=True))
    target = _op(trace, "cumsum")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.UPPER_BOUND
    assert tuple(axis.kind for axis in descriptor.axes) == (
        "pointwise",
        "pointwise",
        "full",
        "pointwise",
    )
    result = cross_validate(
        trace,
        ops=[target],
        units=(1, 3, 2, 4),
        inputs=_input_op(trace),
    )[0]
    assert result.status is ReceptiveFieldValidationStatus.PASS


def test_bare_training_batch_norm_declares_batch_coupling() -> None:
    """Recognize positional-None BatchNorm captures as batch-coupled whole input."""

    model = nn.BatchNorm2d(2, affine=False, track_running_stats=False).train()
    trace = _trace(model, torch.randn(3, 2, 4, 4, requires_grad=True))
    target = _op(trace, "batch_norm")
    descriptor = target.receptive_field._descriptor()

    assert descriptor.status is ReceptiveFieldStatus.WHOLE_INPUT
    assert descriptor.batch_coupled is True
    result = cross_validate(
        trace,
        ops=[target],
        units=(0, 0, 2, 2),
        inputs=_input_op(trace),
    )[0]
    assert result.status is ReceptiveFieldValidationStatus.PASS
    assert result.cross_batch == "geometric"
