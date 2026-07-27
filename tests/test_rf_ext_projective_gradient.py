"""Saved-graph double-VJP projective gradient regression tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import is_decorated_function
from torchlens.backends.torch.wrappers import (
    DIRECT_TRANSFORM_SITES,
    TRANSFORM_BUILDER_SITES,
    wrap_torch,
)
from torchlens.receptive_field._errors import ReceptiveFieldUnavailableError
from torchlens.receptive_field._gradient import gradient_for_unit
from torchlens.receptive_field._gradient_forward import projective_gradient_for_unit
from torchlens.receptive_field._types import ReceptiveFieldDirection


def _trace_model(
    model: nn.Module, inputs: torch.Tensor, *, save_grads: bool = False
) -> tuple[object, object, object]:
    """Capture a backward-ready model and return its trace endpoints.

    Parameters
    ----------
    model:
        Module to capture.
    inputs:
        Graph-connected model input.
    save_grads:
        Whether to install all supported TorchLens gradient hooks.

    Returns
    -------
    tuple[object, object, object]
        Trace, source input operation, and model-output operation.
    """

    capture = tl.options.CaptureOptions(
        backward_ready=True,
        save_grads="all" if save_grads else False,
        capture_tensor_grad_hooks=save_grads,
    )
    trace = tl.trace(model, inputs, capture=capture, save_mode="reference")
    source = next(op for op in trace.layer_list if op.is_input)
    target = next(op for op in trace.layer_list if op.is_output)
    return trace, source, target


def _forward_column_oracle(
    function: Callable[[torch.Tensor], torch.Tensor],
    source: torch.Tensor,
    unit: tuple[int, ...],
) -> torch.Tensor:
    """Measure one exact affine Jacobian column by forward perturbation.

    Parameters
    ----------
    function:
        Affine function evaluated outside TorchLens capture.
    source:
        Primal input tensor.
    unit:
        Complete source-element index.

    Returns
    -------
    torch.Tensor
        Absolute output change caused by adding one at ``unit``.
    """

    primal = source.detach()
    perturbation = torch.zeros_like(primal)
    perturbation[unit] = 1
    return (function(primal + perturbation) - function(primal)).abs()


def _probe_snapshot(trace: object, model: nn.Module) -> tuple[object, ...]:
    """Snapshot every TorchLens and parameter state guarded for RF probes.

    Parameters
    ----------
    trace:
        Captured trace.
    model:
        Captured module whose parameter gradients must remain untouched.

    Returns
    -------
    tuple[object, ...]
        Comparable immutable probe-state snapshot.
    """

    return (
        trace.num_backward_passes,  # type: ignore[attr-defined]
        trace.has_gradients,  # type: ignore[attr-defined]
        tuple(trace.event_stream.backward_events),  # type: ignore[attr-defined]
        tuple(tuple(op.grads) for op in trace.layer_list),  # type: ignore[attr-defined]
        (
            "_active_backward_pass_index" in trace.__dict__,
            trace.__dict__.get("_active_backward_pass_index"),
        ),
        trace.__dict__.get("_tl_backward_triggers_disarmed"),
        tuple(
            None if parameter.grad is None else parameter.grad.clone()
            for parameter in model.parameters()
        ),
    )


def test_double_vjp_support_matches_forward_influence_oracle() -> None:
    """Match a brute-force forward perturbation column on an affine convolution."""

    model = nn.Conv2d(1, 2, 3, padding=1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.arange(18, dtype=torch.float64).reshape(2, 1, 3, 3) / 9 + 0.25)
    inputs = torch.randn(1, 1, 5, 5, dtype=torch.float64, requires_grad=True)
    trace, source, target = _trace_model(model, inputs)
    unit = (0, 0, 2, 2)

    result = projective_gradient_for_unit(source, unit, target=target)
    oracle = _forward_column_oracle(model, inputs, unit)

    assert result.direction is ReceptiveFieldDirection.PROJECTIVE
    assert result.unit == unit
    assert result.unit_shape == tuple(inputs.shape)
    assert result.grad.shape == oracle.shape
    assert torch.allclose(result.grad, oracle, atol=1e-12, rtol=1e-12)
    assert torch.equal(result.support_mask, oracle != 0)


def test_double_vjp_works_through_inplace_relu() -> None:
    """Cover the in-place activation crater that a replay-based jvp would create."""

    convolution = nn.Conv2d(1, 1, 3, padding=1, bias=False).double()
    with torch.no_grad():
        convolution.weight.fill_(0.5)
    model = nn.Sequential(convolution, nn.ReLU(inplace=True))
    inputs = torch.rand(1, 1, 5, 5, dtype=torch.float64, requires_grad=True) + 0.5
    trace, source, target = _trace_model(model, inputs)
    unit = (0, 0, 2, 2)

    result = projective_gradient_for_unit(source, unit, target=target)
    oracle = _forward_column_oracle(model, inputs, unit)

    assert model[1].inplace
    assert torch.allclose(result.grad, oracle, atol=1e-12, rtol=1e-12)
    assert torch.equal(result.support_mask, oracle != 0)
    assert result.support_mask.sum().item() == 9


def test_torch_func_jvp_wrapping_analysis() -> None:
    """Lock why saved-graph double-VJP is used instead of jvp-over-replay.

    Forward-mode tangents cannot be attached retrospectively to the captured
    primal graph. A jvp would therefore require re-execution or certified replay,
    measuring a potentially different RNG/aliasing/in-place program and imposing
    an intervention-ready capture asymmetry.
    """

    assert not any(
        namespace == "torch.func" and name == "jvp"
        for namespace, name, _ in TRANSFORM_BUILDER_SITES
    )
    assert not any(
        namespace == "torch.func" and name == "jvp"
        for namespace, name, _, _ in DIRECT_TRANSFORM_SITES
    )
    wrap_torch()
    assert not is_decorated_function(torch.func.jvp)

    trace = tl.trace(nn.Identity(), torch.randn(2, 3))
    before = (
        len(trace.layer_list),
        trace.num_backward_passes,
        len(trace.event_stream.backward_events),
    )
    torch.func.jvp(lambda value: value.square(), (torch.ones(3),), (torch.ones(3),))
    after = (
        len(trace.layer_list),
        trace.num_backward_passes,
        len(trace.event_stream.backward_events),
    )
    assert after == before

    rf_directory = Path(__file__).parents[1] / "torchlens" / "receptive_field"
    implementations = "\n".join(path.read_text() for path in rf_directory.glob("*.py"))
    assert "torch.func.jvp" not in implementations
    assert "torch.autograd.functional.jvp" not in implementations


def test_empirical_vjp_row_equals_double_vjp_column() -> None:
    """Confirm the same Jacobian entry through both toleranced contractions."""

    model = nn.Conv2d(2, 3, 3, padding=1, bias=False).double()
    inputs = torch.randn(1, 2, 5, 5, dtype=torch.float64, requires_grad=True)
    trace, source, target = _trace_model(model, inputs)
    source_unit = (0, 1, 2, 3)
    target_unit = (0, 2, 1, 4)

    row = gradient_for_unit(
        target,
        target_unit,
        input=source,
        retain_graph=True,
    )
    column = projective_gradient_for_unit(source, source_unit, target=target)

    assert torch.allclose(
        row.grad[source_unit],
        column.grad[target_unit],
        atol=1e-12,
        rtol=1e-12,
    )


def test_projective_probe_does_not_pollute_trace_state() -> None:
    """Extend the RF side-effect suite across both suppressed autograd calls."""

    model = nn.Conv2d(1, 1, 3, padding=1)
    inputs = torch.randn(2, 1, 5, 5, requires_grad=True)
    trace, source, target = _trace_model(model, inputs, save_grads=True)
    trace._tl_backward_triggers_disarmed = False
    before = _probe_snapshot(trace, model)

    projective_gradient_for_unit(
        source,
        (0, 0, 2, 2),
        target=target,
        retain_graph=True,
    )
    projective_gradient_for_unit(
        source,
        (1, 0, 2, 2),
        target=target,
        retain_graph=True,
    )

    after = _probe_snapshot(trace, model)
    assert after[:6] == before[:6]
    for current, previous in zip(after[6], before[6], strict=True):
        if previous is None:
            assert current is None
        else:
            assert current is not None and torch.equal(current, previous)
    assert "_tl_rf_probe_active" not in trace.__dict__

    trace.log_backward(target.out.sum())
    assert trace.num_backward_passes == 1
    assert trace.has_gradients
    assert trace.event_stream.backward_events


def test_complex_source_is_typed_unavailable() -> None:
    """Refuse one-basis projective probing for complex source elements in v1."""

    inputs = torch.randn(2, 3, dtype=torch.complex64, requires_grad=True)
    trace, source, target = _trace_model(nn.Identity(), inputs)

    with pytest.raises(ReceptiveFieldUnavailableError, match="two-basis support"):
        projective_gradient_for_unit(source, (0, 0), target=target)


@pytest.mark.parametrize("failure_call", [1, 2])
def test_cotangent_linearization_failure_is_typed_and_restores_state(
    monkeypatch: pytest.MonkeyPatch, failure_call: int
) -> None:
    """Classify either-grad failures and restore their shared suppression scope."""

    model = nn.Conv2d(1, 1, 1)
    inputs = torch.randn(1, 1, 2, 2, requires_grad=True)
    trace, source, target = _trace_model(model, inputs)
    original_grad = torch.autograd.grad
    call_count = 0

    def fail_second_grad(*args: object, **kwargs: object) -> object:
        """Raise from the selected VJP after confirming suppression stays active."""

        nonlocal call_count
        call_count += 1
        assert trace._tl_rf_probe_active is True
        if call_count == failure_call:
            raise RuntimeError("SyntheticBackward cotangent detached")
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", fail_second_grad)
    with pytest.raises(
        ReceptiveFieldUnavailableError,
        match="cotangent linearization unavailable.*SyntheticBackward",
    ):
        projective_gradient_for_unit(source, (0, 0, 0, 0), target=target)

    assert call_count == failure_call
    assert "_tl_rf_probe_active" not in trace.__dict__
    assert trace.num_backward_passes == 0
    assert not trace.has_gradients
    assert not trace.event_stream.backward_events
