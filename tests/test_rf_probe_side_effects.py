"""Non-polluting receptive-field probe suppression tripwires."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field._gradient import _probe_suppressed, gradient_for_unit


def _conv_trace() -> tuple[nn.Conv2d, object, object, object]:
    """Build a graph-connected trace with every gradient-retention hook enabled."""

    model = nn.Conv2d(1, 1, 3, padding=1)
    inputs = torch.randn(2, 1, 5, 5, requires_grad=True)
    capture = tl.options.CaptureOptions(
        backward_ready=True,
        save_grads="all",
        capture_tensor_grad_hooks=True,
    )
    trace = tl.trace(
        model,
        inputs,
        capture=capture,
        save_mode="reference",
    )
    target = next(op for op in trace.layer_list if op.func_name == "conv2d")
    input_op = next(op for op in trace.layer_list if op.is_input)
    return model, trace, target, input_op


def _gradient_record_snapshot(trace: object) -> tuple[tuple[object, ...], ...]:
    """Snapshot every operation's retained gradient record sequence."""

    return tuple(tuple(op.grads) for op in trace.layer_list)  # type: ignore[attr-defined]


def test_probe_does_not_mutate_any_backward_or_gradient_state() -> None:
    """Guarantee no event, Op grad, parameter grad, or disarm-state pollution."""

    model, trace, target, input_op = _conv_trace()
    trace._tl_backward_triggers_disarmed = False
    before_records = _gradient_record_snapshot(trace)
    before_events = tuple(trace.event_stream.backward_events)
    before_disarmed = trace._tl_backward_triggers_disarmed
    before_passes = trace.num_backward_passes
    before_has_gradients = trace.has_gradients
    before_active_index = (
        "_active_backward_pass_index" in trace.__dict__,
        trace.__dict__.get("_active_backward_pass_index"),
    )
    before_param_grads = tuple(
        None if parameter.grad is None else parameter.grad.clone()
        for parameter in model.parameters()
    )

    gradient_for_unit(target, (0, 0, 2, 2), input=input_op)

    assert trace.num_backward_passes == before_passes
    assert trace.has_gradients is before_has_gradients
    assert tuple(trace.event_stream.backward_events) == before_events
    assert _gradient_record_snapshot(trace) == before_records
    assert trace._tl_backward_triggers_disarmed is before_disarmed
    assert (
        "_active_backward_pass_index" in trace.__dict__,
        trace.__dict__.get("_active_backward_pass_index"),
    ) == before_active_index
    assert "_tl_rf_probe_active" not in trace.__dict__
    for parameter, before_grad in zip(model.parameters(), before_param_grads):
        if before_grad is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.equal(parameter.grad, before_grad)


def test_suppression_restores_exact_flag_state_when_probe_raises() -> None:
    """Restore probe and disarm flags exactly when the suppressed operation raises."""

    _, trace, _, _ = _conv_trace()
    trace._tl_backward_triggers_disarmed = True
    trace._tl_rf_probe_active = "prior-sentinel"

    with pytest.raises(RuntimeError, match="probe failed"):
        with _probe_suppressed(trace):
            assert trace._tl_rf_probe_active is True
            raise RuntimeError("probe failed")

    assert trace._tl_rf_probe_active == "prior-sentinel"
    assert trace._tl_backward_triggers_disarmed is True
    assert trace.num_backward_passes == 0
    assert not trace.has_gradients
    assert not trace.event_stream.backward_events


def test_suppression_restores_absent_flag_after_mid_probe_autograd_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore an absent flag when autograd raises after the probe has begun."""

    model, trace, target, input_op = _conv_trace()
    trace._tl_backward_triggers_disarmed = False
    original_grad = torch.autograd.grad

    def failing_grad(*args: object, **kwargs: object) -> object:
        """Simulate an autograd-engine failure after verifying suppression is active."""

        assert trace._tl_rf_probe_active is True
        raise RuntimeError("mid-probe autograd failure")

    monkeypatch.setattr(torch.autograd, "grad", failing_grad)
    with pytest.raises(RuntimeError, match="mid-probe autograd failure"):
        gradient_for_unit(target, (0, 0, 2, 2), input=input_op)
    monkeypatch.setattr(torch.autograd, "grad", original_grad)

    assert "_tl_rf_probe_active" not in trace.__dict__
    assert trace._tl_backward_triggers_disarmed is False
    assert trace.num_backward_passes == 0
    assert not trace.has_gradients
    assert not trace.event_stream.backward_events
    assert all(parameter.grad is None for parameter in model.parameters())
