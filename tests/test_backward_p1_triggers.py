"""P1 backward sidecar trigger and registry tests."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import backward as backward_mod
from torchlens.backends.torch.backward import (
    _BACKWARD_GRAD_FN_REGISTRY,
    _close_implicit_backward_pass_if_open,
)
from torchlens.ir.events import BackwardPassEnd, BackwardPassStart, OpGradObserved


class _PlainBackwardModel(nn.Module):
    """Small model with a nontrivial autograd graph."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return self.fc2(torch.relu(self.fc1(x)))


def _trace_for_plain_backward() -> tl.Trace:
    """Return a trace whose live output can still drive backward.

    Returns
    -------
    tl.Trace
        Captured trace with all activations and gradients enabled.
    """

    torch.manual_seed(0)
    model = _PlainBackwardModel()
    x = torch.randn(3, 4, requires_grad=True)
    return tl.trace(model, x, layers_to_save="all", save_grads="all")


def _backward_starts(trace: tl.Trace) -> list[BackwardPassStart]:
    """Return backward pass start events for ``trace``.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    list[BackwardPassStart]
        Captured ``BackwardPassStart`` events.
    """

    return [
        event
        for event in getattr(trace, "_capture_events").backward_events
        if isinstance(event, BackwardPassStart)
    ]


def _backward_ends(trace: tl.Trace) -> list[BackwardPassEnd]:
    """Return backward pass end events for ``trace``.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    list[BackwardPassEnd]
        Captured ``BackwardPassEnd`` events.
    """

    return [
        event
        for event in getattr(trace, "_capture_events").backward_events
        if isinstance(event, BackwardPassEnd)
    ]


def _op_grad_events(trace: tl.Trace) -> list[OpGradObserved]:
    """Return observed op-gradient events for ``trace``.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    list[OpGradObserved]
        Captured op-gradient events.
    """

    return [
        event
        for event in getattr(trace, "_capture_events").backward_events
        if isinstance(event, OpGradObserved)
    ]


def test_plain_tensor_backward_triggers_capture() -> None:
    """A plain ``loss.backward()`` on a known graph opens a managed bracket."""

    trace = _trace_for_plain_backward()
    loss = trace[trace.output_layers[0]].out.sum()

    loss.backward()

    starts = _backward_starts(trace)
    assert trace.has_backward_pass
    assert trace.num_backward_passes == 1
    assert starts[-1].trigger == "autograd_backward"
    assert starts[-1].implicit is False
    assert trace.grad_fn_logs
    assert any(call for grad_fn in trace.grad_fn_logs.values() for call in grad_fn.calls)
    trace.cleanup()


def test_disarm_triggers_suppresses_plain_backward_capture() -> None:
    """``Trace.disarm_triggers`` removes registry entries and disables hooks."""

    trace = _trace_for_plain_backward()
    loss = trace[trace.output_layers[0]].out.sum()

    trace.disarm_triggers()
    loss.backward()

    assert trace.num_backward_passes == 0
    assert _backward_starts(trace) == []
    trace.cleanup()


def test_registered_grad_fn_ids_are_pinned_by_trace() -> None:
    """Every registry id for the trace must refer to a retained wrapper object."""

    trace = _trace_for_plain_backward()
    registered_ids = {
        grad_fn_object_id
        for grad_fn_object_id, trace_ref in _BACKWARD_GRAD_FN_REGISTRY.items()
        if trace_ref() is trace
    }
    pinned_ids = {id(grad_fn_handle) for grad_fn_handle in trace._backward_gradfn_refs}

    assert registered_ids
    assert registered_ids <= pinned_ids
    trace.cleanup()


def test_fast_pass_refreshes_grad_fn_identity_and_hook_registry() -> None:
    """Selective two-pass capture refreshes live grad-fn ids and hooks."""

    torch.manual_seed(0)
    model = _PlainBackwardModel()
    x = torch.randn(3, 4, requires_grad=True)
    trace = tl.trace(model, x, layers_to_save=["linear_2"], save_grads="all")
    grad_fn_object_id = trace["linear_2"].grad_fn_object_id
    assert grad_fn_object_id is not None
    registered_ids = {
        grad_fn_object_id
        for grad_fn_object_id, trace_ref in _BACKWARD_GRAD_FN_REGISTRY.items()
        if trace_ref() is trace
    }

    assert grad_fn_object_id in registered_ids
    assert id(trace["linear_2"].grad_fn_handle) == grad_fn_object_id
    trace[trace.output_layers[0]].out.sum().backward()
    _close_implicit_backward_pass_if_open(trace)
    assert trace.num_backward_passes == 1
    assert any(event.op_label == trace["linear_2"].layer_label for event in _op_grad_events(trace))
    assert trace["linear_2"].grad is not None
    trace.cleanup()


def test_implicit_backward_pass_lazy_close_records_end() -> None:
    """Implicit brackets close with unobservable duration and peak memory."""

    trace = _trace_for_plain_backward()
    loss = trace[trace.output_layers[0]].out.sum()
    original_backward = backward_mod._ORIGINAL_AUTOGRAD_BACKWARD
    assert original_backward is not None

    original_backward((loss,), grad_tensors=(torch.ones_like(loss),))
    _close_implicit_backward_pass_if_open(trace)

    starts = _backward_starts(trace)
    ends = _backward_ends(trace)
    assert starts[-1].trigger == "implicit"
    assert starts[-1].implicit is True
    assert ends[-1].duration is None
    assert ends[-1].peak_memory is None
    assert ends[-1].status == "ok"
    trace.cleanup()
