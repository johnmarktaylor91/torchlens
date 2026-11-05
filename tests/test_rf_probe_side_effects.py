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
        capture=tl.options.merge_capture_options(capture=capture, save_mode="reference"),
    )
    target = next(op for op in trace._layer_list if op.func_name == "conv2d")
    input_op = next(op for op in trace._layer_list if op.is_input)
    return model, trace, target, input_op


def test_suppression_restores_exact_flag_state_when_probe_raises() -> None:
    """Restore probe and disarm flags exactly when the suppressed operation raises."""

    _, trace, _, _ = _conv_trace()
    trace._tl_backward_triggers_disarmed = True
    trace._tl_rf_probe_active = "prior-sentinel"

    with pytest.raises(RuntimeError, match="probe failed"), _probe_suppressed(trace):
        assert trace._tl_rf_probe_active is True
        raise RuntimeError("probe failed")

    assert trace._tl_rf_probe_active == "prior-sentinel"
    assert trace._tl_backward_triggers_disarmed is True
    assert len(trace.autograd.backward_passes) == 0
    assert not trace.autograd.has_grads
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
        gradient_for_unit(target, (0, 0, 2, 2), input_op=input_op)
    monkeypatch.setattr(torch.autograd, "grad", original_grad)

    assert "_tl_rf_probe_active" not in trace.__dict__
    assert trace._tl_backward_triggers_disarmed is False
    assert len(trace.autograd.backward_passes) == 0
    assert not trace.autograd.has_grads
    assert not trace.event_stream.backward_events
    assert all(parameter.grad is None for parameter in model.parameters())
