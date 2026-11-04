"""Status-honesty goldens for receptive-field rule remediation."""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as functional
from support.rf_isolation import preserved_rf_registry
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


@pytest.fixture(autouse=True)
def built_in_rule_pack() -> Iterator[None]:
    """Install built-in RF rules while preserving registry isolation."""

    with preserved_rf_registry(install_builtin=True):
        yield


def _op(trace: object, name: str) -> object:
    """Return the last captured operation with the requested function name."""

    return next(
        item
        for item in reversed(trace._layer_list)  # type: ignore[union-attr]
        if item.func_name == name
    )


def _input_op(trace: object) -> object:
    """Return the canonical input operation from a one-input trace."""

    return next(item for item in trace._layer_list if item.is_input)  # type: ignore[union-attr]


def _trace(model: nn.Module, inputs: torch.Tensor) -> object:
    """Capture a model with autograd history retained for RF cross-validation."""

    return tl.trace(
        model,
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )


# Kept: sole branch coverage of receptive_field/__init__.py, receptive_field/_validation.py, receptive_field/_view.py.
@pytest.mark.parametrize("surface", ["check", "verify"])
def test_poisoned_trace_refuses_receptive_field_verdicts(surface: str) -> None:
    """RF verdict producers share the monotonic poisoned-trace refusal gate."""
    from torchlens._runnable import PathFaithfulness, mark_trace_path_status
    from torchlens.errors import PoisonedRunError

    model = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    trace = _trace(model, torch.ones(1, 1, 5, 5, requires_grad=True))
    target = _op(trace, "conv2d")
    mark_trace_path_status(trace, PathFaithfulness.DIVERGED, None)

    with pytest.raises(PoisonedRunError, match="receptive field verification"):
        if surface == "check":
            target.receptive_field.check((0, 0, 2, 2))
        else:
            tl.receptive_field.verify(trace)


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


# Kept: sole branch coverage of receptive_field/rules/attention.py.
def test_sdpa_grouped_query_attention_widens_key_value_heads() -> None:
    """Contain key/value support under grouped-query attention; value stays feature-pointwise."""

    inputs = (
        torch.randn(1, 4, 3, 2, requires_grad=True),
        torch.randn(1, 2, 3, 2, requires_grad=True),
        torch.randn(1, 2, 3, 2, requires_grad=True),
    )
    trace = tl.trace(
        _GroupedQueryAttention(),
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    target = _op(trace, "scaled_dot_product_attention")
    key_value_inputs = list(trace.input_ops)[1:]
    expected_kinds = {
        0: ("pointwise", "full", "full", "full"),
        1: ("pointwise", "full", "full", "pointwise"),
    }

    for index, source in enumerate(key_value_inputs):
        descriptor = target.receptive_field._descriptor(source)
        assert tuple(axis.kind for axis in descriptor.axes) == expected_kinds[index]
    results = cross_validate(
        trace,
        ops=[target],
        units=(0, 3, 1, 0),
        inputs=key_value_inputs,
    )
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in results)


# Kept: sole branch coverage of receptive_field/rules/attention.py.
def test_sdpa_broadcast_widens_key_value_batch_and_head_axes() -> None:
    """Contain broadcast key/value support; the value feature axis stays pointwise."""

    inputs = (
        torch.randn(2, 4, 3, 2, requires_grad=True),
        torch.randn(1, 1, 3, 2, requires_grad=True),
        torch.randn(1, 1, 3, 2, requires_grad=True),
    )
    trace = tl.trace(
        _DistinctAttention(),
        inputs,
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    target = _op(trace, "scaled_dot_product_attention")
    key_value_inputs = list(trace.input_ops)[1:]
    expected_kinds = {
        0: ("full", "full", "full", "full"),
        1: ("full", "full", "full", "pointwise"),
    }

    for index, source in enumerate(key_value_inputs):
        descriptor = target.receptive_field._descriptor(source)
        assert tuple(axis.kind for axis in descriptor.axes) == expected_kinds[index]
    results = cross_validate(
        trace,
        ops=[target],
        units=(1, 3, 1, 0),
        inputs=key_value_inputs,
    )
    assert all(result.status is ReceptiveFieldValidationStatus.PASS for result in results)


# Kept: sole branch coverage of receptive_field/_gradient.py, receptive_field/_gradient_forward.py, receptive_field/_validation.py and more.
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


# Kept: sole branch coverage of receptive_field/_engine.py, receptive_field/_gradient.py, receptive_field/_validation.py and more; a sole mutation kill.
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


class _Reduction(nn.Module):
    """Middle-axis reduction fixture."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Reduce channels while preserving batch and spatial identities."""

        return value.mean(dim=1)


class _Cumulative(nn.Module):
    """Dimensioned cumulative-operation fixture."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Accumulate over height rather than the trailing axis."""

        return value.cumsum(dim=2)


# Kept: sole branch coverage of receptive_field/_validation.py; a sole mutation kill.
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
