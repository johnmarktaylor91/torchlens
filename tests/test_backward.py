"""Smoke tests for first-class backward-pass capture."""

import dataclasses
import warnings
from types import MethodType
from unittest import mock

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.validation as tl_validation
import torchlens.validation.backward as backward_validation
import torchlens.validation.consolidated as consolidated_validation
from torchlens.data_classes.grad_fn import GradFn
from torchlens.ir.events import BackwardPassStart, OpGradObserved
from torchlens.options import CaptureOptions, SaveOptions

_NO_GRAD_AUTOGRAD_ERROR = "element 0 of tensors does not require grad and does not have a grad_fn"


class _TinyBackwardModel(nn.Module):
    """Small MLP with view op coverage."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        hidden = torch.relu(self.fc1(x))
        viewed = hidden.view(hidden.shape[0], 4)
        return self.fc2(viewed)


class _DoubleFn(torch.autograd.Function):
    """Custom autograd function for grad_fn_handle classification tests."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """Return doubled input."""
        return x * 2

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
        """Return doubled upstream grad."""
        return grad * 2


class _CustomModel(nn.Module):
    """Model using a custom autograd Function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return _DoubleFn.apply(x).sum()


class _SquareFunction(torch.autograd.Function):
    """Custom autograd function with non-trivial gradient."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor) -> torch.Tensor:
        """Square the input tensor."""

        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor) -> torch.Tensor:
        """Return the analytical square gradient."""

        (x,) = ctx.saved_tensors
        return 2 * x * grad


class _CustomParamModel(nn.Module):
    """Parameterized model that routes through a custom autograd Function."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))
        self.output = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the custom function on a parameterized activation."""

        return self.output(_SquareFunction.apply(x @ self.weight)).sum()


class _WeightTiedModel(nn.Module):
    """Model that applies one module twice."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a shared layer twice."""

        return self.linear(torch.relu(self.linear(x))).sum()


class _DropoutBackwardModel(nn.Module):
    """Dropout model for seeded backward validation."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout and a linear layer."""

        return self.linear(self.dropout(x)).sum()


class _BatchNormBackwardModel(nn.Module):
    """BatchNorm model for state restoration checks."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.linear = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BatchNorm and a linear layer."""

        return self.linear(self.bn(x)).sum()


def _logged_model(
    *,
    layers_to_save: str | list[str] | None = "all",
    save_grads: str | list[str] | None = "all",
) -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    """Create a logged tiny model.

    Returns
    -------
    tuple[nn.Module, torch.Tensor, tl.Trace]
        Model, input tensor, and model log.
    """
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save=layers_to_save, save_grads=save_grads),
    )
    return model, x, trace


def _output_loss(trace: tl.Trace) -> torch.Tensor:
    """Return scalar sum loss from the logged output out."""
    return trace[trace.output_layers[0]].out.sum()


def _saved_relu(trace: tl.Trace) -> torch.Tensor:
    """Return the selectively retained ReLU activation from a trace."""

    return next(op.out for op in trace if op.has_saved_activation and op.func_name == "relu")


def test_detached_saved_activation_backward_names_backward_ready_remedy() -> None:
    """A TorchLens-detached retained activation gives targeted backward guidance."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, save=tl.func("relu"))
    saved = _saved_relu(trace)

    with pytest.raises(RuntimeError, match=r"Re-trace with backward_ready=True") as exc_info:
        saved.sum().backward()

    assert type(exc_info.value) is RuntimeError
    assert str(exc_info.value).startswith(_NO_GRAD_AUTOGRAD_ERROR)
    assert x.grad is None


def test_unrelated_detached_tensor_backward_keeps_native_error() -> None:
    """An unrelated user-detached tensor keeps the unmodified PyTorch error."""

    detached = torch.randn(3, requires_grad=True).detach()

    with pytest.raises(RuntimeError) as exc_info:
        detached.sum().backward()

    assert str(exc_info.value) == _NO_GRAD_AUTOGRAD_ERROR

    trace = tl.trace(_TinyBackwardModel(), torch.randn(2, 3), save=tl.func("relu"))
    explicitly_detached = _saved_relu(trace).detach()
    with pytest.raises(RuntimeError) as marked_exc_info:
        explicitly_detached.sum().backward()

    assert str(marked_exc_info.value) == _NO_GRAD_AUTOGRAD_ERROR

    metadata_only = torch.randn_like(_saved_relu(trace)).view_as(_saved_relu(trace))
    with pytest.raises(RuntimeError) as metadata_exc_info:
        metadata_only.sum().backward()

    assert str(metadata_exc_info.value) == _NO_GRAD_AUTOGRAD_ERROR


def test_connected_output_and_backward_ready_saved_activation_are_unchanged() -> None:
    """Connected model outputs and backward-ready retained activations still backpropagate."""

    output_model = _TinyBackwardModel()
    output_x = torch.randn(2, 3, requires_grad=True)
    output_trace = tl.trace(output_model, output_x)
    output = output_trace.output_ops[0].out
    output.sum().backward()
    assert output_x.grad is not None

    ready_model = _TinyBackwardModel()
    ready_x = torch.randn(2, 3, requires_grad=True)
    ready_trace = tl.trace(
        ready_model,
        ready_x,
        save=tl.func("relu"),
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    ready_saved = _saved_relu(ready_trace)
    ready_saved.sum().backward()
    assert ready_x.grad is not None


def test_detached_log_backward_does_not_poison_later_capture() -> None:
    """Reject detached losses without leaking capture state or a start event."""
    model = nn.Linear(2, 1)
    trace = tl.trace(model, torch.ones(1, 2))

    with pytest.raises(ValueError, match="loss has no grad_fn / is detached"):
        trace.log_backward(torch.tensor(1.0))

    assert not any(isinstance(event, BackwardPassStart) for event in trace.backward_events)
    fresh_trace = tl.trace(model, torch.ones(1, 2))
    assert fresh_trace.output_layers

    recording = tl.record(model, torch.ones(1, 2), save=tl.func("linear"))
    sparse_trace = recording.to_trace()
    detached_output = sparse_trace[sparse_trace.output_layers[0]].out
    assert detached_output is not None
    assert detached_output.grad_fn is None

    with pytest.raises(ValueError, match="loss has no grad_fn / is detached"):
        sparse_trace.log_backward(detached_output)

    assert not any(isinstance(event, BackwardPassStart) for event in sparse_trace.backward_events)
    assert tl.trace(model, torch.ones(1, 2)).output_layers


@pytest.mark.smoke
def test_log_backward_captures_per_layer_grads() -> None:
    """log_backward captures saved per-layer grads."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.has_gradients
    assert len(trace.saved_grad_ops) > 0
    assert all(trace[label].grad is not None for label in trace.saved_grad_ops.keys())


@pytest.mark.smoke
def test_recording_backward_context_manager() -> None:
    """recording_backward accumulates multiple backward calls."""
    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    with trace.recording_backward():
        loss.backward(retain_graph=True)
        (loss * 2).backward()
    assert trace.num_backward_passes == 2


def _backward_projection_snapshot(trace: tl.Trace) -> dict:
    """Return a value-level snapshot of every backward projection surface."""

    grad_fns = {
        object_id: (
            grad_fn.label,
            grad_fn.type,
            grad_fn.type_index,
            grad_fn.ordinal_index,
            grad_fn.step_index,
            grad_fn.order,
            grad_fn.has_op,
            grad_fn.op_label,
            list(grad_fn.children),
            list(grad_fn.parents),
            grad_fn.module_address,
            grad_fn.module_membership_source,
            list(grad_fn.next_grad_fn_ids),
            sorted(grad_fn.calls),
        )
        for object_id, grad_fn in trace.grad_fn_logs.items()
    }
    calls = {
        (object_id, ordinal): (call.label, call.backward_pass_index)
        for object_id, grad_fn in trace.grad_fn_logs.items()
        for ordinal, call in grad_fn.calls.items()
    }
    passes = {
        pass_index: (
            record.trigger,
            record.status,
            record.order,
            len(record.grad_fn_calls),
            record.order_attribution_coverage,
        )
        for pass_index, record in trace.backward_pass_logs.items()
    }
    op_grads = {
        op.layer_label: (
            [
                (record.backward_pass_index, record.grad is not None, record.shape)
                for record in op._slot("_grad_records")
            ],
            int(op.gradient_memory),
        )
        for op in trace.layer_list
        if getattr(op, "has_grad", False)
    }
    param_grads = {
        address: [
            (record.ordinal, record.backward_pass_index, record.shape, record.memory)
            for record in param_log._grad_records
        ]
        for address, param_log in trace.param_logs.items()
        if param_log._grad_records
    }
    return {
        "grad_fns": grad_fns,
        "calls": calls,
        "passes": passes,
        "op_grads": op_grads,
        "param_grads": param_grads,
        "order": list(trace.grad_fn_order),
        "num_passes": trace.num_backward_passes,
        "num_calls": trace.num_saved_grad_fn_calls,
        "num_fns": trace.num_saved_grad_fns,
        "saved_labels": set(trace._saved_grad_labels),
        "total_grad_mem": int(trace.total_gradient_memory),
        "total_bwd_mem": int(trace.total_backward_memory),
    }


@pytest.mark.smoke
def test_backward_reprojection_folds_incrementally() -> None:
    """Repeated passes over an unchanged graph fold O(tail), not full rebuilds."""
    from torchlens.backends.torch import backward as backward_mod

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    full_rebuild_sizes: list[int] = []
    real_impl = backward_mod._materialize_backward_projections_impl

    def counting_impl(trace_arg: tl.Trace, events: list, stream: object = None) -> None:
        full_rebuild_sizes.append(len(events))
        real_impl(trace_arg, events, stream=stream)

    with mock.patch.object(backward_mod, "_materialize_backward_projections_impl", counting_impl):
        with trace.recording_backward():
            loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            loss.backward()
    assert trace.num_backward_passes == 3
    assert len(full_rebuild_sizes) == 1, (
        f"later same-graph passes must fold incrementally, not rebuild: {full_rebuild_sizes}"
    )

    incremental_snapshot = _backward_projection_snapshot(trace)
    trace.__dict__.pop("_backward_projection_fold_state", None)
    trace.__dict__.pop("_backward_projection_revision", None)
    trace.__dict__.pop("_backward_projection_event_count", None)
    backward_mod._materialize_backward_projections(trace)
    assert _backward_projection_snapshot(trace) == incremental_snapshot


@pytest.mark.smoke
def test_backward_events_share_one_monotonic_seq_domain() -> None:
    """Every backward event kind carries one writer-stamped monotonic seq."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd as _End

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    events = _ensure_backward_event_stream(trace).backward_events
    seqs = [event.seq for event in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
    assert all(seq > 0 for seq in seqs)
    start = next(e for e in events if isinstance(e, BackwardPassStart))
    end = next(e for e in events if isinstance(e, _End))
    assert start.seq < end.seq
    for event in events:
        if getattr(event, "pass_index", None) == 1 and event is not start and event is not end:
            assert start.seq < event.seq < end.seq


def _invariant_check(trace: tl.Trace) -> None:
    """Run the backward event-flow invariant directly."""
    from torchlens.validation.invariants import _check_backward_event_flow_invariants

    _check_backward_event_flow_invariants(trace, "backward_graph_invariants")


@pytest.mark.smoke
def test_backward_seq_invariants_fire_on_planted_mutations() -> None:
    """Each rewritten exact-seq assertion still fails on a planted misorder."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd as _End
    from torchlens.validation.invariants import MetadataInvariantError

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    events = _ensure_backward_event_stream(trace).backward_events
    _invariant_check(trace)  # positive control: the real stream passes

    start = next(e for e in events if isinstance(e, BackwardPassStart))
    end = next(e for e in events if isinstance(e, _End))
    op_grad = next(e for e in events if isinstance(e, OpGradObserved))

    original_op_grad_seq = op_grad.seq
    object.__setattr__(op_grad, "seq", end.seq + 1)
    with pytest.raises(MetadataInvariantError, match="follows its pass|unique and monotonic"):
        _invariant_check(trace)
    object.__setattr__(op_grad, "seq", start.seq - 1 if start.seq > 1 else 0)
    with pytest.raises(MetadataInvariantError, match="precedes its pass|unique and monotonic"):
        _invariant_check(trace)
    object.__setattr__(op_grad, "seq", original_op_grad_seq)
    _invariant_check(trace)

    original_end_seq = end.seq
    object.__setattr__(end, "seq", start.seq)
    with pytest.raises(MetadataInvariantError, match="unique and monotonic|does not"):
        _invariant_check(trace)
    object.__setattr__(end, "seq", original_end_seq)
    _invariant_check(trace)

    original_start_seq = start.seq
    object.__setattr__(start, "seq", original_end_seq + 5)
    with pytest.raises(MetadataInvariantError):
        _invariant_check(trace)
    object.__setattr__(start, "seq", original_start_seq)
    _invariant_check(trace)

    # A monotonic-but-misbracketed stream (End reordered before the pass's
    # facts, all seqs renumbered in list order) must fail on bracketing
    # alone, proving the exact-bracket assertion is independently armed.
    original_order = list(events)
    original_seqs = [event.seq for event in events]
    events.remove(end)
    events.insert(events.index(start) + 1, end)
    for renumbered_seq, event in enumerate(events, start=1):
        object.__setattr__(event, "seq", renumbered_seq)
    with pytest.raises(MetadataInvariantError, match="follows its pass"):
        _invariant_check(trace)
    events[:] = original_order
    for original_seq, event in zip(original_seqs, events):
        object.__setattr__(event, "seq", original_seq)
    _invariant_check(trace)


@pytest.mark.smoke
def test_backward_capture_refuses_missing_event_stream() -> None:
    """A trace that lost its event stream gets a typed refusal, not a silent buffer."""
    from torchlens._errors import BackwardStreamUnavailableError
    from torchlens.backends.torch.backward import _ensure_backward_event_stream

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    del trace._capture_events
    assert trace.event_stream is None
    with pytest.raises(BackwardStreamUnavailableError):
        _ensure_backward_event_stream(trace)
    with pytest.raises(BackwardStreamUnavailableError):
        trace.log_backward(loss)


@pytest.mark.smoke
def test_param_gradients_enter_the_backward_event_stream() -> None:
    """Every recorded AccumulateGrad increment has a ParamGradObserved event."""
    from torchlens.ir.events import ParamGradObserved

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))

    param_events = [
        event for event in trace.backward_events if isinstance(event, ParamGradObserved)
    ]
    assert param_events
    event_records = {(event.param_address, event.pass_index) for event in param_events}
    projected_records = {
        (address, record.backward_pass_index)
        for address, param_log in trace.param_logs.items()
        for record in param_log._grad_records
    }
    assert projected_records
    assert event_records == projected_records
    for event in param_events:
        assert event.payload_ref is not None
        assert event.shape is not None
        assert event.memory
        assert event.seq > 0


def test_param_gradient_payloads_honor_per_call_retention_policy() -> None:
    """Per-call ``save_grads=False`` keeps param-grad facts but drops payloads."""
    from torchlens.ir.events import ParamGradObserved

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace), save_grads=False)

    param_events = [
        event for event in trace.backward_events if isinstance(event, ParamGradObserved)
    ]
    assert param_events
    assert all(event.payload_ref is None for event in param_events)
    assert all(record.grad is None for param in trace.param_logs.values() for record in param.grads)


def test_param_gradient_payloads_use_trace_save_mode() -> None:
    """Parameter gradients route through the shared save-mode copy chokepoint."""
    _model, _x, trace = _logged_model()
    trace.save_mode = "reference"
    trace.log_backward(_output_loss(trace))

    payloads = [record.grad for param in trace.param_logs.values() for record in param.grads]
    assert payloads
    assert all(payload is not None for payload in payloads)
    assert all(payload.grad_fn is None for payload in payloads)


def test_grad_fn_event_payloads_are_detached_snapshots() -> None:
    """Projection rebuild never exposes graph-connected autograd hook buffers."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace), create_graph=True)

    tensors = [
        tensor
        for grad_fn in trace.grad_fns
        for call in grad_fn.calls.values()
        for payload in (call.grad_inputs, call.grad_outputs)
        for tensor in (payload or ())
        if isinstance(tensor, torch.Tensor)
    ]
    assert tensors
    assert all(tensor.grad_fn is None for tensor in tensors)


def test_per_call_grad_policy_does_not_mutate_trace_selection() -> None:
    """A backward call keeps the capture-time gradient selection unchanged."""
    _model, _x, trace = _logged_model(save_grads=None)
    selection_before = trace._grad_op_nums_to_save

    trace.log_backward(_output_loss(trace), save_grads=False)

    assert trace._grad_op_nums_to_save == selection_before
    assert all(
        call.grad_inputs is None and call.grad_outputs is None
        for grad_fn in trace.grad_fns
        for call in grad_fn.calls.values()
    )


@pytest.mark.parametrize(
    "failure_site",
    ["_clear_forward_grad_fn_refs", "_rewalk_higher_order_grad_fns"],
)
def test_backward_tail_failure_restores_globals_and_closes_journal(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    """Tail finalizer failures cannot strand globals or a start-only pass."""
    from torchlens import _state
    from torchlens.backends.torch import backward
    from torchlens.ir.events import BackwardPassEnd

    _model, _x, trace = _logged_model()

    def fail_tail(_trace: tl.Trace) -> None:
        """Inject one backward-tail failure."""
        raise RuntimeError(f"injected {failure_site}")

    monkeypatch.setattr(backward, failure_site, fail_tail)
    with pytest.raises(RuntimeError, match=failure_site):
        trace.log_backward(_output_loss(trace))

    assert _state._active_trace is None
    assert _state._active_hook_plan is None
    assert _state._active_intervention_spec is None
    assert "_tl_active_backward_bracket" not in trace.__dict__
    assert "_active_backward_pass_index" not in trace.__dict__
    assert "_active_save_grads_policy" not in trace.__dict__
    assert any(isinstance(event, BackwardPassEnd) for event in trace.backward_events)


def test_backward_walk_failure_removes_partial_hooks_and_disarms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A BaseException during graph-hook arming removes its registered prefix."""
    from torchlens import _state
    from torchlens.backends.torch import backward

    _model, _x, trace = _logged_model()

    class _Handle:
        """Minimal removable hook-handle probe."""

        removed = False

        def remove(self) -> None:
            """Record that cleanup reached this partial handle."""
            self.removed = True

    handle = _Handle()

    def fail_walk(_trace: tl.Trace, _loss: torch.Tensor, handles: list[object]) -> list[object]:
        """Register one synthetic handle and interrupt graph walking."""
        handles.append(handle)
        raise KeyboardInterrupt

    monkeypatch.setattr(backward, "_walk_and_hook_backward_graph", fail_walk)
    with pytest.raises(KeyboardInterrupt):
        trace.log_backward(_output_loss(trace))

    assert handle.removed is True
    assert trace._tl_backward_triggers_disarmed is True
    assert _state._active_trace is None


def test_rewalk_failure_removes_hooks_and_clears_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A higher-order rewalk failure still unwinds hooks and grad-fn refs.

    The rewalk error used to re-raise from the pass finalizer BEFORE handle
    removal, pending-record clearing, and forward grad-fn ref clearing, so a
    failure in that phase stranded live TorchLens hooks on the user's
    autograd graph.
    """
    from torchlens.backends.torch import backward

    _model, _x, trace = _logged_model()

    class _Handle:
        """Minimal removable hook-handle probe."""

        removed = False

        def remove(self) -> None:
            """Record that cleanup reached this handle."""
            self.removed = True

    handle = _Handle()
    real_walk = backward._walk_and_hook_backward_graph

    def spy_walk(
        trace_arg: tl.Trace,
        loss_arg: torch.Tensor,
        handles: list[object] | None = None,
    ) -> list[object]:
        """Run the real walk, then append one spy handle."""
        result = real_walk(trace_arg, loss_arg, handles)
        target = handles if handles is not None else result
        target.append(handle)
        return result

    def fail_rewalk(_trace: tl.Trace) -> None:
        """Inject a rewalk-phase failure."""
        raise RuntimeError("injected rewalk failure")

    monkeypatch.setattr(backward, "_walk_and_hook_backward_graph", spy_walk)
    monkeypatch.setattr(backward, "_rewalk_higher_order_grad_fns", fail_rewalk)
    with pytest.raises(RuntimeError, match="injected rewalk failure"):
        trace.log_backward(_output_loss(trace))

    assert handle.removed is True
    assert all(layer.grad_fn_handle is None for layer in trace.layer_list)


def test_ordinary_tail_pending_clear_failure_still_removes_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ORDINARY finalization tail is fenced like the rewalk-error arm.

    grind-p5 b3-sol-R14-1: the ordinary tail called
    ``_clear_pending_accumulate_grad_records`` unfenced BEFORE the
    handle-removal loop and the forward grad-fn ref clear -- an injected
    failure there stranded every grad-fn hook and strong graph ref (the
    unswept sibling of the fenced rewalk-error branch 20 lines up). The
    failure must still propagate (never be silently swallowed), but every
    later cleanup step must run first.
    """
    from torchlens.backends.torch import backward

    _model, _x, trace = _logged_model()

    class _Handle:
        """Minimal removable hook-handle probe."""

        removed = False

        def remove(self) -> None:
            """Record that cleanup reached this handle."""
            self.removed = True

    handle = _Handle()
    real_walk = backward._walk_and_hook_backward_graph

    def spy_walk(
        trace_arg: tl.Trace,
        loss_arg: torch.Tensor,
        handles: list[object] | None = None,
    ) -> list[object]:
        """Run the real walk, then append one spy handle."""
        result = real_walk(trace_arg, loss_arg, handles)
        target = handles if handles is not None else result
        target.append(handle)
        return result

    def fail_pending_clear(_trace: tl.Trace) -> None:
        """Inject a pending-record clear failure in the ordinary tail."""
        raise RuntimeError("injected pending-clear failure")

    monkeypatch.setattr(backward, "_walk_and_hook_backward_graph", spy_walk)
    monkeypatch.setattr(backward, "_clear_pending_accumulate_grad_records", fail_pending_clear)
    with pytest.raises(RuntimeError, match="injected pending-clear failure"):
        trace.log_backward(_output_loss(trace))

    assert handle.removed is True, "pending-clear failure stranded grad-fn hooks"
    assert all(layer.grad_fn_handle is None for layer in trace.layer_list), (
        "pending-clear failure stranded strong forward grad-fn refs"
    )


@pytest.mark.parametrize("entrypoint", ["log_backward", "recording_backward"])
def test_backward_entrypoints_finalize_streaming_on_exception(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    """Both backward APIs run the deferred streaming finalizer on failure."""
    from torchlens.backends.torch import backward

    _model, _x, trace = _logged_model()
    finalized: list[tl.Trace] = []

    def record_finalize(finalized_trace: tl.Trace) -> None:
        """Record one streaming-finalizer invocation."""
        finalized.append(finalized_trace)

    monkeypatch.setattr(backward, "_finalize_grad_streaming", record_finalize)
    if entrypoint == "log_backward":

        def fail_capture(*_args: object, **_kwargs: object) -> None:
            """Inject a backward-capture failure."""
            raise RuntimeError("injected backward failure")

        monkeypatch.setattr(backward, "_run_backward_with_capture", fail_capture)
        with pytest.raises(RuntimeError, match="injected backward failure"):
            trace.log_backward(_output_loss(trace))
    else:
        with pytest.raises(RuntimeError, match="injected block failure"):
            with trace.recording_backward():
                raise RuntimeError("injected block failure")

    assert finalized == [trace]


@pytest.mark.slow
def test_separate_unmanaged_engine_calls_get_separate_implicit_passes() -> None:
    """Graph-task identity prevents adjacent unmanaged backwards from merging."""
    from torchlens.backends.torch import backward
    from torchlens.ir.events import BackwardPassStart

    if not hasattr(torch._C, "_current_graph_task_id"):
        pytest.skip("torch build does not expose autograd graph-task identity")
    _model, _x, trace = _logged_model()
    output = trace[trace.output_layers[0]].out
    original_backward = backward._ORIGINAL_AUTOGRAD_BACKWARD
    assert original_backward is not None

    original_backward((output.sum(),), retain_graph=True)
    original_backward(((output * 2).sum(),))
    backward._close_implicit_backward_pass_if_open(trace)

    starts = [
        event
        for event in trace.backward_events
        if isinstance(event, BackwardPassStart) and event.implicit
    ]
    assert [event.pass_index for event in starts] == [1, 2]


@pytest.mark.smoke
def test_replay_fork_does_not_inherit_gradient_state() -> None:
    """A replay fork starts with no captured gradient state; the source keeps its own."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            layers_to_save="all", save_grads="all", intervention_ready=True, backward_ready=True
        ),
    )
    trace.log_backward(_output_loss(trace))
    assert trace.has_gradients
    assert trace._saved_grad_labels
    assert any(param_log._grad_records for param_log in trace.param_logs.values())

    def _identity_hook(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        return out

    fork = trace.push(
        replay=tl.options.ReplayOptions(
            hooks={tl.func("relu"): _identity_hook}, differentiable=True
        )
    )

    assert fork.has_gradients is False
    assert fork._saved_grad_labels == set()
    assert not any(param_log._grad_records for param_log in fork.param_logs.values())
    assert int(fork.total_param_gradient_memory) == 0
    assert fork.num_backward_passes == 0

    assert trace.has_gradients
    assert trace._saved_grad_labels
    assert any(param_log._grad_records for param_log in trace.param_logs.values())


@pytest.mark.smoke
def test_replay_fork_cannot_resurrect_has_grad_from_derived_payload() -> None:
    """A seeded ``_derived_grad_payload`` never survives onto a replay fork.

    ``_check_param_grad`` treats a surviving derived payload as proof of a
    gradient, so a fork that inherited it would report ``has_grad=True`` for a
    fork that never ran a backward — the stale-True channel the replay-fork
    reset exists to kill. No backward runs here, so the live-model
    read-through cannot legitimately supply a gradient either.
    """
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", intervention_ready=True, backward_ready=True),
    )
    address, param_log = next(iter(trace.param_logs.items()))
    param_log._derived_grad_payload = torch.ones(1)
    assert param_log.has_grad is True  # positive control: the seed arms the channel

    def _identity_hook(out: torch.Tensor, *, hook: object) -> torch.Tensor:
        return out

    fork = trace.push(
        replay=tl.options.ReplayOptions(
            hooks={tl.func("relu"): _identity_hook}, differentiable=True
        )
    )

    fork_param = fork.param_logs[address]
    assert fork_param._derived_grad_payload is None
    assert fork_param.has_grad is False
    # The source keeps its own (seeded) state untouched.
    assert param_log._derived_grad_payload is not None


@pytest.mark.smoke
def test_backward_reprojection_guard_survives_count_preserving_mutation() -> None:
    """A count-preserving event mutation still triggers reprojection."""
    from dataclasses import replace as dataclass_replace

    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.backward_pass_logs[1].status == "ok"

    stream = _ensure_backward_event_stream(trace)
    old_end = stream.backward_events[-1]
    assert isinstance(old_end, BackwardPassEnd)
    stream.backward_events.pop()
    stream.note_backward_event_removal()
    stream.append_backward(dataclass_replace(old_end, status="error"))

    trace._sync_backward_projection_if_needed()
    assert trace.backward_pass_logs[1].status == "error"


@pytest.mark.smoke
def test_recording_backward_delegates_foreign_graphs() -> None:
    """A backward on an unrelated graph inside the context never enters the trace."""
    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    foreign = torch.randn(3, 3, requires_grad=True)
    with trace.recording_backward():
        (foreign * 2).sum().backward()
        loss.backward()
    assert foreign.grad is not None
    assert trace.num_backward_passes == 1

    control_model, _control_x, control_trace = _logged_model()
    control_loss = _output_loss(control_trace)
    with control_trace.recording_backward():
        control_loss.backward()
    trace_labels = sorted(grad_fn.label for grad_fn in trace.grad_fn_logs.values())
    control_labels = sorted(grad_fn.label for grad_fn in control_trace.grad_fn_logs.values())
    assert trace_labels == control_labels


@pytest.mark.smoke
def test_recording_backward_foreign_only_block_stays_empty() -> None:
    """A foreign-only block records no passes and warns once, not silently."""
    _model, _x, trace = _logged_model()
    foreign = torch.randn(2, 2, requires_grad=True)
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        with trace.recording_backward():
            (foreign * foreign).sum().backward()
            (foreign * 2.0).sum().backward()
    unmatched_warnings = [
        record
        for record in warning_records
        if issubclass(record.category, RuntimeWarning)
        and "did not reach any grad-fn" in str(record.message)
    ]
    assert len(unmatched_warnings) == 1, "unmatched-backward warning must fire exactly once"
    assert foreign.grad is not None
    assert trace.num_backward_passes == 0
    assert len(trace.grad_fn_logs) == 0


@pytest.mark.smoke
def test_recording_backward_exit_preserves_interleaved_patch() -> None:
    """__exit__ never clobbers a Tensor.backward patch installed inside the block."""
    _model, _x, trace = _logged_model()

    def interloper_backward(tensor_self: torch.Tensor, *args: object, **kwargs: object) -> None:
        raise AssertionError("interloper should never run in this test")

    context = trace.recording_backward()
    context.__enter__()
    try:
        torch.Tensor.backward = interloper_backward  # type: ignore[assignment, method-assign]
        with pytest.warns(UserWarning, match="Tensor.backward"):
            context.__exit__(None, None, None)
        assert torch.Tensor.backward is interloper_backward
    finally:
        torch.Tensor.backward = context._original_backward  # type: ignore[assignment, method-assign]


@pytest.mark.smoke
def test_backward_graph_walk_includes_intervening_grad_fns() -> None:
    """The backward DAG includes grad_fns without forward Layer matches."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert any(not grad_fn_handle.has_op for grad_fn_handle in trace.grad_fn_logs.values())


def test_has_op_storage_field() -> None:
    """GradFn.has_op records whether a forward op was captured."""

    grad_fn_handle = GradFn(
        grad_fn_object_id=1,
        class_name="AddBackward0",
        class_qualname="torch.autograd.AddBackward0",
        is_custom=False,
        label="addbackward0_1_1",
        type="addbackward0",
        type_index=1,
        ordinal_index=1,
        step_index=1,
        has_op=True,
    )
    assert grad_fn_handle.has_op is True


@pytest.mark.smoke
def test_grad_fn_log_back_pointer() -> None:
    """Forward LayerLogs link to corresponding GradFnLogs by identity."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert any(layer.grad_fn is not None for layer in trace.layer_list)
    assert any(
        grad_fn_handle.op is not None and grad_fn_handle.op.grad_fn is grad_fn_handle
        for grad_fn_handle in trace.grad_fns
    )


@pytest.mark.smoke
def test_grad_fn_naming_and_indexing() -> None:
    """GradFn labels and accessor indexing mirror layer lookup patterns."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    first_grad_fn = trace.grad_fns[0]
    assert "_back_" in first_grad_fn.label
    assert first_grad_fn.label == first_grad_fn.label.lower()
    assert trace.grad_fns[first_grad_fn.label] is first_grad_fn
    assert trace.grad_fns[first_grad_fn.type] is first_grad_fn
    if first_grad_fn.num_calls:
        assert trace.grad_fns[f"{first_grad_fn.label}:1"] is first_grad_fn
        assert trace.grad_fn_calls[f"{first_grad_fn.label}:1"] is first_grad_fn.calls[0]
    assert list(trace.grad_fns)


@pytest.mark.smoke
def test_save_grads_true_captures_all_grads() -> None:
    """save_grads=True captures all gradients independent of layers_to_save."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=CaptureOptions(layers_to_save=["relu"], save_grads=True))
    trace.log_backward(_output_loss(trace))
    assert trace.saved_grad_ops
    assert any("relu" in label for label in trace.saved_grad_ops.keys())
    assert any("linear" in label for label in trace.saved_grad_ops.keys())


@pytest.mark.smoke
def test_grads_to_save_independent_override() -> None:
    """save_grads selectors are independent from layers_to_save."""
    _model, _x, trace = _logged_model(layers_to_save="all", save_grads=["relu"])
    trace.log_backward(_output_loss(trace))
    assert trace.saved_grad_ops
    assert all("relu" in label for label in trace.saved_grad_ops.keys())


@pytest.mark.smoke
def test_auto_train_mode_when_backward_opted_in() -> None:
    """Explicit save_grads selectors auto-enable backward_ready."""
    _model, _x, trace = _logged_model()
    assert trace.backward_ready is True


@pytest.mark.smoke
def test_auto_train_mode_conflict_with_explicit_false() -> None:
    """Explicit backward_ready=False conflicts with backward capture."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(ValueError, match="requires backward_ready=True"):
        tl.trace(model, x, capture=CaptureOptions(save_grads="all", backward_ready=False))


@pytest.mark.smoke
def test_grad_transform_applied() -> None:
    """grad_transform writes transformed grads separately."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(save_grads="all"),
        save=SaveOptions(grad_transform=lambda grad: torch.zeros_like(grad)),
    )
    trace.log_backward(_output_loss(trace))
    assert all(
        torch.equal(
            trace[label].transformed_grad,
            torch.zeros_like(trace[label].grad),
        )
        for label in trace.saved_grad_ops.keys()
    )


@pytest.mark.smoke
def test_flat_transform_kwargs_populate_transformed_payloads() -> None:
    """Flat activation_transform and grad_transform kwargs are applied."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        grad_transform=lambda grad: grad.half(),
        capture=tl.options.CaptureOptions(save_grads=True, backward_ready=True),
        save=tl.options.SaveOptions(activation_transform=lambda out: out.half()),
    )
    relu_op = next(op for op in trace.ops if op.func_name == "relu")

    trace.log_backward(_output_loss(trace), retain_graph=True)

    assert relu_op.transformed_out is not None
    assert relu_op.transformed_out.dtype == torch.float16
    assert relu_op.transformed_grad is not None
    assert relu_op.transformed_grad.dtype == torch.float16


@pytest.mark.smoke
def test_module_log_grad_aggregation() -> None:
    """Module exposes aggregated grads for contained layers."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.modules["fc2"].grad is not None


@pytest.mark.smoke
def test_input_layer_grad_access() -> None:
    """Input layers expose saved grads after backward."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace[trace.input_layers[0]].grad is not None


@pytest.mark.smoke
def test_param_layer_grad_access() -> None:
    """Param grad metadata still works through the existing hook path."""
    model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert any(param_log.has_grad for param_log in trace.params)
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.smoke
def test_custom_autograd_function_captured_with_is_custom_flag() -> None:
    """Custom autograd.Function grad_fns are captured and flagged."""
    model = _CustomModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=CaptureOptions(save_grads="all"))
    trace.log_backward(_output_loss(trace))
    assert any(grad_fn_handle.is_custom for grad_fn_handle in trace.grad_fn_logs.values())


@pytest.mark.smoke
def test_implicit_hook_firing_preserved() -> None:
    """Calling backward outside log_backward still populates Layer grads."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=CaptureOptions(save_grads=True))
    _output_loss(trace).backward()
    assert trace.saved_grad_ops


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:`layers_to_save` is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:`random_seed` is deprecated:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:`save_grads` is deprecated:DeprecationWarning")
def test_validate_backward_pass_correct() -> None:
    """validate_backward_pass returns True for correct capture."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    assert tl_validation.validate_backward_pass(model, x)


def test_validate_backward_default_detects_corrupted_captured_op_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default backward validation must read captured ``Op.grad`` payloads."""

    original_log_backward = tl.Trace.log_backward
    corrupted_labels: list[str] = []

    def corrupt_captured_grad(trace: tl.Trace, *args: object, **kwargs: object) -> object:
        """Corrupt one captured module-output grad after backward logging."""

        result = original_log_backward(trace, *args, **kwargs)
        for op in trace.layer_list:
            grad = getattr(op, "grad", None)
            if not isinstance(grad, torch.Tensor) or not getattr(op, "modules", None):
                continue
            record = op.grads.for_pass(1)
            record.grad = torch.full_like(grad, 12345.0)
            corrupted_labels.append(op.label)
            break
        return result

    monkeypatch.setattr(tl.Trace, "log_backward", corrupt_captured_grad)

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    assert not tl_validation.validate_backward_pass(model, x, random_seed=42)
    assert corrupted_labels


def test_validate_backward_pass_random_seed_kwarg_public_wrapper() -> None:
    """The public backward wrapper forwards ``random_seed`` into the dispatcher."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with mock.patch(
        "torchlens.validation.consolidated.validate_backward_pass",
        wraps=consolidated_validation.validate_backward_pass,
    ) as validator:
        assert tl.validate(model, x, scope="backward", random_seed=42)
    assert validator.call_args is not None
    assert validator.call_args.kwargs["random_seed"] == 42


def test_validation_validate_backward_pass_random_seed_kwarg_user_funcs_path() -> None:
    """The validation subpackage path forwards ``random_seed`` to the source validator."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with mock.patch(
        "torchlens.validation.backward.validate_backward_pass",
        wraps=backward_validation.validate_backward_pass,
    ) as validator:
        assert tl_validation.validate_backward_pass(model, x, random_seed=42)
    assert validator.call_args is not None
    assert validator.call_args.kwargs["random_seed"] == 42


def test_validate_backward_random_seed_deterministic() -> None:
    """A fixed seed produces repeatable backward validation outcomes."""

    torch.manual_seed(0)
    model = _DropoutBackwardModel().train()
    x = torch.randn(6, 3, requires_grad=True)
    first = tl_validation.validate_backward_pass(model, x, random_seed=42)
    second = tl_validation.validate_backward_pass(model, x, random_seed=42)
    assert first is True
    assert second is first


def test_validate_backward_zero_grad_between_passes() -> None:
    """Backward validation zeros parameter grads before, between, and after passes."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    calls: list[bool | None] = []
    original_zero_grad = model.zero_grad

    def counted_zero_grad(self: nn.Module, set_to_none: bool | None = True) -> None:
        """Record zero_grad calls and delegate to the original method."""

        calls.append(set_to_none)
        original_zero_grad(set_to_none=set_to_none)

    model.zero_grad = MethodType(counted_zero_grad, model)
    assert tl_validation.validate_backward_pass(model, x, random_seed=42)
    assert calls == [True, True, True]


def test_validate_backward_dropout_train_mode_reproducible() -> None:
    """Seeded validation handles train-mode Dropout."""

    torch.manual_seed(0)
    model = _DropoutBackwardModel().train()
    x = torch.randn(6, 3, requires_grad=True)
    assert tl_validation.validate_backward_pass(model, x, random_seed=42)


def test_validate_backward_batchnorm_train_mode_state_restored() -> None:
    """Seeded validation handles train-mode BatchNorm without leaking state."""

    torch.manual_seed(0)
    model = _BatchNormBackwardModel().train()
    x = torch.randn(6, 3, requires_grad=True)
    running_mean = model.bn.running_mean.detach().clone()
    running_var = model.bn.running_var.detach().clone()

    assert tl_validation.validate_backward_pass(model, x, random_seed=42)
    assert torch.equal(model.bn.running_mean, running_mean)
    assert torch.equal(model.bn.running_var, running_var)


def test_validate_backward_train_mode_audit() -> None:
    """Backward validation preserves the model train/eval flag."""

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    for training in (True, False):
        model.train(training)
        assert tl_validation.validate_backward_pass(model, x, random_seed=42)
        assert model.training is training


def test_validate_backward_multipass_weight_tied() -> None:
    """Parameter-grad comparison handles a shared module used twice."""

    torch.manual_seed(0)
    model = _WeightTiedModel()
    x = torch.randn(2, 3, requires_grad=True)
    assert tl_validation.validate_backward_pass(model, x, random_seed=42)


def test_validate_backward_custom_autograd_function() -> None:
    """Parameter-grad comparison covers custom autograd.Function nodes."""

    torch.manual_seed(0)
    model = _CustomParamModel()
    x = torch.randn(2, 3, requires_grad=True)
    assert tl_validation.validate_backward_pass(model, x, random_seed=42)


def test_validate_backward_hygiene_data_parallel() -> None:
    """Backward validation unwraps DataParallel wrappers."""

    model = nn.DataParallel(_TinyBackwardModel())
    model.module.cpu()
    x = torch.randn(2, 3, requires_grad=True)
    assert backward_validation.validate_backward_pass(model, x, random_seed=42)


def test_validate_backward_hygiene_opaque_wrapper() -> None:
    """Backward validation rejects opaque TorchScript wrappers."""

    model = torch.jit.trace(_TinyBackwardModel(), torch.randn(2, 3))
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(RuntimeError, match="torch.jit"):
        backward_validation.validate_backward_pass(model, x, random_seed=42)


def test_accumulategrad_labels_deterministic_across_captures() -> None:
    """AccumulateGrad labels are stable across seeded captures."""

    torch.manual_seed(0)
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace1 = tl.trace(model, x, capture=CaptureOptions(save_grads="all", random_seed=42))
    trace1.log_backward(_output_loss(trace1))
    labels1 = {
        grad_fn_handle.label
        for grad_fn_handle in trace1.grad_fn_logs.values()
        if grad_fn_handle.type == "accumulategrad"
    }
    trace1.cleanup()

    trace2 = tl.trace(model, x, capture=CaptureOptions(save_grads="all", random_seed=42))
    trace2.log_backward(_output_loss(trace2))
    labels2 = {
        grad_fn_handle.label
        for grad_fn_handle in trace2.grad_fn_logs.values()
        if grad_fn_handle.type == "accumulategrad"
    }
    trace2.cleanup()

    assert labels1 == labels2


@pytest.mark.smoke
def test_validate_backward_pass_perturb_kwarg_removed() -> None:
    """The inert saved-grad perturbation kwarg is deleted outright."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    with pytest.raises(TypeError):
        tl_validation.validate_backward_pass(model, x, perturb_saved_grads=True)


@pytest.mark.smoke
def test_peak_memory_tracking_populated() -> None:
    """Trace stores flat backward peak-memory tracking metadata."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace.has_backward_pass
    assert isinstance(trace.backward_peak_memory, int)
    assert trace.backward_memory_backend in {"cpu", "cuda", "mps"}


@pytest.mark.smoke
def test_higher_order_grads_basic_support() -> None:
    """create_graph=True backward calls run through capture."""
    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace), create_graph=True)
    assert trace.num_backward_passes == 1


# ---------------------------------------------------------------------------
# Dual-review fix round: param-grad event authority, event immutability,
# exact bracketing, restored-trace streams, and cleanup disarm.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_param_grad_records_rebuild_from_event_spine() -> None:
    """A forced scratch rebuild reconstructs Param._grad_records from events."""
    from torchlens.backends.torch import backward as backward_mod

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    with trace.recording_backward():
        loss.backward(retain_graph=True)
        loss.backward()

    before = {
        address: [
            (record.ordinal, record.backward_pass_index, record.shape, record.memory)
            for record in param_log._grad_records
        ]
        for address, param_log in trace.param_logs.items()
    }
    payloads_before = {
        address: [record.grad for record in param_log._grad_records]
        for address, param_log in trace.param_logs.items()
    }
    assert any(before.values()), "expected captured param gradient records"
    assert all(len(records) == 2 for records in before.values() if records)

    for param_log in trace.param_logs.values():
        param_log._grad_records = []
    trace.__dict__.pop("_backward_projection_fold_state", None)
    trace.__dict__.pop("_backward_projection_revision", None)
    trace.__dict__.pop("_backward_projection_event_count", None)
    backward_mod._materialize_backward_projections(trace)

    after = {
        address: [
            (record.ordinal, record.backward_pass_index, record.shape, record.memory)
            for record in param_log._grad_records
        ]
        for address, param_log in trace.param_logs.items()
    }
    assert after == before
    for address, param_log in trace.param_logs.items():
        for record, payload in zip(param_log._grad_records, payloads_before[address]):
            assert record.grad is payload, "rebuild must reuse the event-held payload"


@pytest.mark.smoke
def test_param_grad_incremental_fold_matches_scratch_rebuild() -> None:
    """The param-inclusive snapshot proves fold == scratch across passes."""
    from torchlens.backends.torch import backward as backward_mod

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    with trace.recording_backward():
        loss.backward(retain_graph=True)
        loss.backward(retain_graph=True)
        loss.backward()

    incremental_snapshot = _backward_projection_snapshot(trace)
    assert incremental_snapshot["param_grads"], "oracle must include parameter state"
    # Armed by construction: erase every projected param record first, so the
    # forced scratch rebuild can only match the incremental snapshot by
    # re-deriving the records from the ParamGradObserved event spine. Under
    # direct-write param capture (no event folding) the cleared records never
    # come back and this comparison fails.
    for param_log in trace.param_logs.values():
        param_log._grad_records = []
    trace.__dict__.pop("_backward_projection_fold_state", None)
    trace.__dict__.pop("_backward_projection_revision", None)
    trace.__dict__.pop("_backward_projection_event_count", None)
    backward_mod._materialize_backward_projections(trace)
    assert _backward_projection_snapshot(trace) == incremental_snapshot


@pytest.mark.smoke
def test_param_grad_reconciliation_counts_multiplicity() -> None:
    """Duplicating one projected param record now fails reconciliation."""
    from torchlens.validation.invariants import MetadataInvariantError

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    _invariant_check(trace)  # positive control

    param_log = next(
        param_log for param_log in trace.param_logs.values() if param_log._grad_records
    )
    duplicated = param_log._grad_records[0]
    param_log._grad_records.append(duplicated)
    with pytest.raises(MetadataInvariantError, match="by multiplicity"):
        _invariant_check(trace)
    param_log._grad_records.pop()
    _invariant_check(trace)


@pytest.mark.smoke
def test_op_grad_reconciliation_counts_multiplicity() -> None:
    """Duplicating one projected op record now fails reconciliation."""
    from torchlens.validation.invariants import MetadataInvariantError

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    _invariant_check(trace)  # positive control

    victim = next(op for op in trace.layer_list if op._slot("_grad_records"))
    records = victim._slot("_grad_records")
    records.append(records[0])
    with pytest.raises(MetadataInvariantError, match="by multiplicity"):
        _invariant_check(trace)
    records.pop()
    _invariant_check(trace)


@pytest.mark.smoke
def test_grad_fn_discovered_source_is_frozen_by_the_writer() -> None:
    """In-place mutation of GradFnDiscovered.source raises instead of biting."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import GradFnDiscovered

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    events = _ensure_backward_event_stream(trace).backward_events
    discovered = next(event for event in events if isinstance(event, GradFnDiscovered))
    with pytest.raises(TypeError):
        discovered.source["class_source_file"] = "/tmp/planted-mutation.py"  # type: ignore[index]
    assert dict(discovered.source) is not discovered.source  # copies still work


@pytest.mark.smoke
def test_higher_order_discovery_bracketing_is_armed() -> None:
    """A created_in_pass discovery moved past its pass end fails the invariant."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd, GradFnDiscovered
    from torchlens.validation.invariants import MetadataInvariantError

    class _HigherOrderModel(nn.Module):
        """Tiny nonlinear model with differentiable first gradients."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a scalar nonlinear output."""

            return (torch.tanh(x) ** 3).sum()

    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    trace = tl.trace(_HigherOrderModel(), x, capture=tl.options.CaptureOptions(save_grads="all"))
    loss = trace[trace.output_layers[0]].out
    first_grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
    torch.autograd.grad(first_grad.sum(), x, retain_graph=True)
    events = _ensure_backward_event_stream(trace).backward_events
    _invariant_check(trace)  # positive control

    created = [
        event
        for event in events
        if isinstance(event, GradFnDiscovered) and event.created_in_pass is not None
    ]
    assert created, "create_graph autograd.grad must discover higher-order grad-fns"
    victim = created[0]
    end = next(
        event
        for event in events
        if isinstance(event, BackwardPassEnd) and event.pass_index == victim.created_in_pass
    )
    # Move ONLY the discovery after its pass end and renumber every event
    # monotonically in list order (the sol probe): all other pass-scoped
    # events stay inside their brackets, so a failure here proves the new
    # created_in_pass bracket check specifically is armed.
    original_order = list(events)
    original_seqs = [event.seq for event in events]
    events.remove(victim)
    events.insert(events.index(end) + 1, victim)
    for renumbered_seq, event in enumerate(events, start=1):
        object.__setattr__(event, "seq", renumbered_seq)
    with pytest.raises(MetadataInvariantError, match="higher-order GradFnDiscovered"):
        _invariant_check(trace)
    events[:] = original_order
    for original_seq, event in zip(original_seqs, events):
        object.__setattr__(event, "seq", original_seq)
    _invariant_check(trace)


@pytest.mark.smoke
def test_pass_brackets_reject_partial_interleaving() -> None:
    """Two pass brackets that partially overlap fail the invariant."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd
    from torchlens.validation.invariants import MetadataInvariantError

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    with trace.recording_backward():
        loss.backward(retain_graph=True)
        loss.backward()
    events = _ensure_backward_event_stream(trace).backward_events
    _invariant_check(trace)  # positive control

    end_one = next(
        event for event in events if isinstance(event, BackwardPassEnd) and event.pass_index == 1
    )
    start_two = next(
        event for event in events if isinstance(event, BackwardPassStart) and event.pass_index == 2
    )
    # Interleave: end(1) slides just after start(2) and every seq is
    # renumbered monotonically -> [1 .. [2 .. 1] .. 2]. Each pass-scoped fact
    # still sits inside its own bracket, so only the new partial-overlap
    # check can fire.
    original_order = list(events)
    original_seqs = [event.seq for event in events]
    events.remove(end_one)
    events.insert(events.index(start_two) + 1, end_one)
    for renumbered_seq, event in enumerate(events, start=1):
        object.__setattr__(event, "seq", renumbered_seq)
    with pytest.raises(MetadataInvariantError, match="partially overlaps"):
        _invariant_check(trace)
    events[:] = original_order
    for original_seq, event in zip(original_seqs, events):
        object.__setattr__(event, "seq", original_seq)
    _invariant_check(trace)


@pytest.mark.smoke
def test_restored_trace_supports_backward_capture() -> None:
    """A pickled-and-restored trace records a fresh backward correctly."""
    import pickle

    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(backward_ready=True, save_grads="all"),
        save_mode="reference",
    )
    loss = _output_loss(trace)
    restored = pickle.loads(pickle.dumps(trace))
    assert len(restored._capture_events.backward_events) == 0
    assert "_backward_projection_revision" not in restored.__dict__
    assert "_backward_projection_fold_state" not in restored.__dict__

    restored.log_backward(loss)
    assert restored.num_backward_passes == 1
    assert restored.grad_fn_logs
    assert restored.backward_pass_logs[1].status == "ok"
    _invariant_check(restored)


@pytest.mark.smoke
def test_double_restore_replaces_stale_stream() -> None:
    """__setstate__ replaces a reused object's stream based on incoming state."""
    import pickle

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    assert trace._capture_events.backward_events

    pickled_state = pickle.dumps(trace)
    restored = pickle.loads(pickled_state)
    # Plant an event on the fresh stream, then restore AGAIN onto the same
    # object: the stale stream (and its planted event) must not survive.
    restored._capture_events.backward_events.append("SENTINEL")
    restored.__setstate__(trace.__getstate__())
    assert len(restored._capture_events.backward_events) == 0

    # And the stream never leaks into later pickle state.
    assert "_capture_events" not in restored.__getstate__()
    assert "_capture_events" not in trace.__getstate__()
    assert "_backward_projection_revision" not in trace.__getstate__()
    assert "_backward_projection_fold_state" not in trace.__getstate__()


def _assert_pass_one_calls_survive_rebase(trace, pass_one_call_snapshot) -> None:
    """Assert every rediscovered grad-fn keeps its pass-1 calls VERBATIM.

    The same GradFnCall objects, at the same call ordinals, with the pass-2
    call appended after them — a rebase must never renumber the retained
    pass's call tuples.
    """
    for object_id, snapshot in pass_one_call_snapshot.items():
        record = trace.grad_fn_logs[object_id]
        merged_calls = dict(record.calls._dict)
        for call_ordinal, pass_one_call in snapshot.items():
            assert merged_calls.get(call_ordinal) is pass_one_call
            assert pass_one_call.backward_pass_index == 1
        if snapshot:
            call_passes = sorted(call.backward_pass_index for call in merged_calls.values())
            assert call_passes == [1, 2], (
                f"grad-fn {record.label} call passes {call_passes}: the retained "
                "pass-1 call must survive next to the new pass-2 call"
            )


def _assert_pass_record_calls_owned(trace, pass_record) -> None:
    """Assert the pass record's calls agree with the grad-fn projections.

    Every call the retained pass record lists must be a call some grad-fn
    record still owns — the two projections must never diverge.
    """
    owned_call_ids = {
        id(call) for record in trace.grad_fn_logs.values() for call in record.calls._dict.values()
    }
    assert pass_record.grad_fn_calls
    for call in pass_record.grad_fn_calls:
        assert id(call) in owned_call_ids
        assert call.backward_pass_index == pass_record.pass_index


@pytest.mark.smoke
def test_restored_trace_with_prior_backward_extends_pass_numbering() -> None:
    """A post-restore backward numbers itself after the preserved pass.

    The restored stream is DETACHED (pass-index base = passes materialized
    before pickling), so the new pass is pass 2, the invariants stay green,
    and pass 1's projection (pass record, per-param records, grad-fn records)
    survives the full rebuild instead of being silently erased.
    """
    import pickle

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    trace.log_backward(loss, retain_graph=True)
    assert trace.num_backward_passes == 1

    restored = pickle.loads(pickle.dumps(trace))
    assert restored._capture_events.pass_index_base == 1
    pass_one_record = restored.backward_pass_logs[1]
    preserved_grad_fn_ids = set(restored.grad_fn_logs)
    assert preserved_grad_fn_ids
    # Full pass-1 projection content, not just ID presence: the exact
    # GradFnCall objects each grad-fn record holds before the rebase.
    pass_one_call_snapshot = {
        object_id: dict(record.calls._dict) for object_id, record in restored.grad_fn_logs.items()
    }
    assert any(pass_one_call_snapshot.values())
    pass_one_param_counts = {
        address: len(param_log._grad_records) for address, param_log in restored.param_logs.items()
    }
    assert any(pass_one_param_counts.values())

    restored.log_backward(loss, retain_graph=True)

    assert restored.num_backward_passes == 2
    assert set(restored.backward_pass_logs) == {1, 2}
    # The earlier pass's projection is intact — the SAME preserved record,
    # not a lookalike rebuilt from a stream that never saw pass 1.
    assert restored.backward_pass_logs[1] is pass_one_record
    assert preserved_grad_fn_ids <= set(restored.grad_fn_logs)
    _assert_pass_one_calls_survive_rebase(restored, pass_one_call_snapshot)
    # And the retained pass record agrees with the grad-fn records: every
    # call it lists is one the merged grad-fn projection still owns.
    _assert_pass_record_calls_owned(restored, pass_one_record)
    all_labels = [record.label for record in restored.grad_fn_logs.values()]
    assert len(all_labels) == len(set(all_labels)), "merged grad-fn labels must stay unique"
    for address, param_log in restored.param_logs.items():
        pass_indices = sorted(record.backward_pass_index for record in param_log._grad_records)
        preserved = [index for index in pass_indices if index == 1]
        assert len(preserved) == pass_one_param_counts[address]
        if pass_one_param_counts[address]:
            assert 2 in pass_indices, f"{address} missing the new pass's record"
    _invariant_check(restored)

    # Re-restoring after the second pass and running a THIRD exercises a
    # repeat rebuild of a detached stream (base = 2): both earlier passes'
    # calls survive verbatim and the ordinals stay contiguous — the new
    # fire must continue after the pre-base window, never gap past it.
    restored_again = pickle.loads(pickle.dumps(restored))
    assert restored_again._capture_events.pass_index_base == 2
    two_pass_call_snapshot = {
        object_id: dict(record.calls._dict)
        for object_id, record in restored_again.grad_fn_logs.items()
    }
    assert any(two_pass_call_snapshot.values())
    restored_again.log_backward(loss)
    assert restored_again.num_backward_passes == 3
    assert set(restored_again.backward_pass_logs) == {1, 2, 3}
    for object_id, snapshot in two_pass_call_snapshot.items():
        record = restored_again.grad_fn_logs[object_id]
        merged_calls = dict(record.calls._dict)
        for call_ordinal, prior_call in snapshot.items():
            assert merged_calls.get(call_ordinal) is prior_call
        if snapshot:
            assert sorted(merged_calls) == [1, 2, 3], (
                f"grad-fn {record.label} call ordinals {sorted(merged_calls)} "
                "must stay contiguous across a repeat rebuild"
            )
            assert [
                merged_calls[ordinal].backward_pass_index for ordinal in sorted(merged_calls)
            ] == [1, 2, 3]
    _invariant_check(restored_again)


@pytest.mark.smoke
def test_fork_after_backward_gets_detached_event_stream() -> None:
    """``Trace.fork()`` never shares the parent's backward event stream.

    The fork starts a detached stream (fresh lists, pass-index base = the
    parent's materialized passes) with the stream-derived projection guards
    dropped, and a managed backward on the fork leaves the parent's stream
    and projection completely untouched.
    """
    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)
    trace.log_backward(loss, retain_graph=True)

    fork = trace.fork()
    assert fork._capture_events is not trace._capture_events
    assert fork._capture_events.backward_events is not trace._capture_events.backward_events
    assert fork._capture_events.backward_events == []
    assert fork._capture_events.pass_index_base == 1
    assert "_backward_projection_revision" not in fork.__dict__
    assert "_backward_projection_fold_state" not in fork.__dict__
    assert "_backward_projection_event_count" not in fork.__dict__

    parent_event_count = len(trace._capture_events.backward_events)
    parent_revision = trace._capture_events.backward_revision
    fork_pass_one_record = fork.backward_pass_logs[1]
    fork_pass_one_call_snapshot = {
        object_id: dict(record.calls._dict) for object_id, record in fork.grad_fn_logs.items()
    }
    assert any(fork_pass_one_call_snapshot.values())
    with warnings.catch_warnings():
        # The parent's tensor hooks must stand down for the fork's managed
        # pass — an implicit-pass warning here means the parent absorbed it.
        warnings.simplefilter("error")
        fork.log_backward(loss, retain_graph=True)

    assert len(trace._capture_events.backward_events) == parent_event_count
    assert trace._capture_events.backward_revision == parent_revision
    assert trace.num_backward_passes == 1
    assert set(trace.backward_pass_logs) == {1}
    assert fork.num_backward_passes == 2
    assert set(fork.backward_pass_logs) == {1, 2}
    # The parent's persistent tensor hooks redirect their observations to the
    # bracket-holding fork (shared label space), so the fork's pass records
    # op gradients instead of silently losing them.
    fork_op_grad_passes = {
        event.pass_index
        for event in fork._capture_events.backward_events
        if isinstance(event, OpGradObserved)
    }
    assert fork_op_grad_passes == {2}
    # The detached-fork rebuild preserves the inherited pass's projection
    # content the same way the pickle-restore path does.
    assert fork.backward_pass_logs[1] is fork_pass_one_record
    _assert_pass_one_calls_survive_rebase(fork, fork_pass_one_call_snapshot)
    _assert_pass_record_calls_owned(fork, fork_pass_one_record)
    _invariant_check(trace)
    _invariant_check(fork)


@pytest.mark.smoke
def test_grad_fn_discovered_source_snapshot_defeats_caller_proxy() -> None:
    """The writer re-snapshots an already-proxied source's backing dict.

    A caller-supplied ``MappingProxyType`` aliases the caller's mutable dict;
    trusting it would let an in-place mutation of that backing dict change the
    event without moving ``backward_revision``.
    """
    from types import MappingProxyType

    from torchlens.ir.capture_events import CaptureEvents
    from torchlens.ir.events import GradFnDiscovered

    backing = {
        "class_source_file": "/original/source.py",
        "class_source_line": 1,
    }
    event = GradFnDiscovered(
        object_id=1,
        class_name="AddBackward0",
        class_qualname="AddBackward0",
        is_custom=False,
        op_label=None,
        param_ref=None,
        created_in_pass=None,
        creator_object_id=None,
        source=MappingProxyType(backing),
        topology=(),
    )
    stream = CaptureEvents()
    stream.append_backward(event)
    backing["class_source_file"] = "/tmp/planted-mutation.py"
    assert event.source["class_source_file"] == "/original/source.py"


@pytest.mark.smoke
def test_fork_deep_copy_debug_mode_rethrows(monkeypatch: pytest.MonkeyPatch) -> None:
    """TORCHLENS_DEBUG_FORK_COPY=1 surfaces silently-degraded fork copies.

    The fork copier intentionally degrades opaque values to ``on_failure``
    (several routine Trace fields rely on it), which is exactly how a genuine
    copy bug — a field that MUST fork independently silently becoming shared —
    stays invisible. The debug channel re-raises instead; this exercises the
    single choke point every fork field copy routes through.
    """
    import copy

    from torchlens.data_classes import _trace_intervention as trace_intervention

    class _PoisonDeepCopy:
        """Object whose deepcopy always fails (shallow copy still works)."""

        def __deepcopy__(self, memo: dict) -> "_PoisonDeepCopy":
            raise RuntimeError("planted deepcopy failure")

    monkeypatch.delenv("TORCHLENS_DEBUG_FORK_COPY", raising=False)
    degraded = trace_intervention._memoized_deep_copy(_PoisonDeepCopy(), None, on_failure=copy.copy)
    assert isinstance(degraded, _PoisonDeepCopy)  # default: silent degradation

    monkeypatch.setenv("TORCHLENS_DEBUG_FORK_COPY", "1")
    with pytest.raises(RuntimeError, match="planted deepcopy failure"):
        trace_intervention._memoized_deep_copy(_PoisonDeepCopy(), None, on_failure=copy.copy)

    # And a whole-trace fork under a poisoned field degrades (not raises) by
    # default — the historical behavior stays available.
    monkeypatch.delenv("TORCHLENS_DEBUG_FORK_COPY", raising=False)
    _model, _x, trace = _logged_model()
    trace.__dict__["_tl_test_poison"] = _PoisonDeepCopy()
    fork = trace.fork()
    assert isinstance(fork.__dict__.get("_tl_test_poison"), _PoisonDeepCopy)


@pytest.mark.smoke
def test_cleanup_disarms_backward_triggers() -> None:
    """A user backward after cleanup() must not raise from lingering hooks."""
    model = _TinyBackwardModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    out = trace[trace.output_layers[0]].out
    trace.log_backward(out.sum(), retain_graph=True)
    trace.cleanup()

    # The tensor hooks and grad_fn hooks registered on the user's still-live
    # graph fire during this backward; disarmed, they must silently no-op.
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.smoke
def test_journal_seq_spans_forward_and_backward_lanes() -> None:
    """Every retained lane is writer-stamped from ONE run-monotonic counter."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    stream = _ensure_backward_event_stream(trace)
    lanes = (
        stream.op_events,
        stream.module_prep_events,
        stream.module_enter_events,
        stream.module_exit_events,
        stream.pre_hook_events,
        stream.output_version_events,
        stream.backward_events,
    )
    all_seqs = [event.seq for lane in lanes for event in lane]
    assert all(isinstance(seq, int) and seq >= 1 for seq in all_seqs)
    assert len(all_seqs) == len(set(all_seqs)), "journal seq values must be unique across lanes"
    for lane in lanes:
        lane_seqs = [event.seq for event in lane]
        assert lane_seqs == sorted(lane_seqs)
    assert max(all_seqs) <= stream.event_seq
    # Forward ops were observed before this post-hoc backward pass, so the
    # recorded order must say so exactly.
    max_op_seq = max(event.seq for event in stream.op_events)
    assert all(event.seq > max_op_seq for event in stream.backward_events)


@pytest.mark.smoke
def test_journal_seq_invariant_fires_on_planted_mutations() -> None:
    """The journal-wide seq invariant is independently armed per failure mode."""
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.validation.invariants import (
        MetadataInvariantError,
        _check_journal_seq_invariants,
    )

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    stream = _ensure_backward_event_stream(trace)

    def check() -> None:
        _check_journal_seq_invariants(trace, "backward_graph_invariants")

    check()  # positive control: the real stream passes

    first_op = stream.op_events[0]
    second_op = stream.op_events[1]

    # ``OpRecord.seq`` is a read-only view over the frozen core, so a plant
    # forges a whole record (``dataclasses.replace`` on the core) and splices
    # it into the journal lane -- ``core.seq`` is unpatchable through the
    # amendment channel by design, and the invariant must still catch a
    # forged record that bypassed the writer entirely.
    def plant_op_seq(index: int, seq_value: int) -> None:
        record = stream.op_events[index]
        forged_core = dataclasses.replace(record.core, seq=seq_value)
        stream.op_events[index] = dataclasses.replace(record, core=forged_core)

    # Unstamped event (bypassed the writer).
    original_seq = first_op.seq
    plant_op_seq(0, 0)
    with pytest.raises(MetadataInvariantError, match="missing a writer-stamped seq"):
        check()
    stream.op_events[0] = first_op
    check()

    # Lane reorder (strictly-increasing violated).
    plant_op_seq(1, original_seq)
    with pytest.raises(MetadataInvariantError, match="appears in both|does not increase"):
        check()
    plant_op_seq(1, original_seq - 1 if original_seq > 1 else 0)
    with pytest.raises(MetadataInvariantError):
        check()
    stream.op_events[1] = second_op
    check()

    # Cross-lane duplicate (module lane forging an op's seq).
    enter_event = stream.module_enter_events[0]
    original_enter_seq = enter_event.seq
    object.__setattr__(enter_event, "seq", original_seq)
    with pytest.raises(MetadataInvariantError, match="appears in both"):
        check()
    object.__setattr__(enter_event, "seq", original_enter_seq)
    check()

    # Counter bypass (an event stamped past the writer counter).
    backward_event = stream.backward_events[-1]
    original_backward_seq = backward_event.seq
    object.__setattr__(backward_event, "seq", stream.event_seq + 7)
    with pytest.raises(MetadataInvariantError, match="exceeds the writer counter"):
        check()
    object.__setattr__(backward_event, "seq", original_backward_seq)
    check()


@pytest.mark.smoke
def test_journal_seq_invariant_fires_on_event_deletion() -> None:
    """FINDING B1-08: DELETING an event used to pass every journal check.

    Stamped / lane-monotone / journal-unique / counter-bounded are all preserved
    by removing an event, so popping ``op_events[1]`` from a real capture passed
    both ``_check_journal_seq_invariants`` AND the full ``check_metadata_
    invariants`` gate -- the omission a tampered or lossy journal produces was
    the one mutation the tripwire could not see. ``next_seq()`` is called only by
    the nine single-writer appenders, so a retaining journal's seq domain is
    DENSE and every hole is a deletion.
    """

    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.validation.invariants import (
        MetadataInvariantError,
        _check_journal_seq_invariants,
        check_metadata_invariants,
    )

    _model, _x, trace = _logged_model()
    trace.log_backward(_output_loss(trace))
    stream = _ensure_backward_event_stream(trace)

    def check() -> None:
        _check_journal_seq_invariants(trace, "backward_graph_invariants")

    check()  # positive control
    check_metadata_invariants(trace)

    # Sol's exact repro: pop one op event out of the middle of the lane.
    popped_op = stream.op_events.pop(1)
    with pytest.raises(MetadataInvariantError, match="not retained by any lane"):
        check()
    with pytest.raises(MetadataInvariantError, match="not retained by any lane"):
        check_metadata_invariants(trace)
    stream.op_events.insert(1, popped_op)
    check()

    # Every lane is covered, not just the op lane.
    for lane_name in ("module_enter_events", "module_exit_events", "backward_events"):
        lane = getattr(stream, lane_name)
        if not lane:
            continue
        removed = lane.pop(0)
        with pytest.raises(MetadataInvariantError, match="not retained by any lane"):
            check()
        lane.insert(0, removed)
        check()

    # A counter advanced past the events it stamped is the same hole seen from
    # the other side (a writer that consumed a seq without appending).
    stream.event_seq += 3
    with pytest.raises(MetadataInvariantError, match="not retained by any lane"):
        check()
    stream.event_seq -= 3
    check()


@pytest.mark.smoke
def test_amendment_lane_deletion_is_visible_to_the_journal_invariant() -> None:
    """The amendment lane rides its own counter and is checked the same way.

    Companion half of B1-08: the typed post-commit lane was absent from the
    invariant entirely. An EMPTY lane stays legitimate (released, never amended,
    or fully folded at the seal) but a hole inside a retained run, an unstamped
    amendment, or a start offset the seal watermark does not explain is red.
    """

    from torchlens.ir.capture_events import CaptureEvents
    from torchlens.validation.invariants import (
        MetadataInvariantError,
        _check_journal_seq_invariants,
    )

    _model, _x, trace = _logged_model()
    stream = trace._capture_events

    class _FakeAmendment:
        """Minimal stand-in carrying only the lane's sequencing facts."""

        def __init__(self, seq: int) -> None:
            self.seq = seq

    def check() -> None:
        _check_journal_seq_invariants(trace, "backward_graph_invariants")

    original_lane = list(stream.op_amendments)
    original_counter = stream.amendment_seq
    original_watermark = stream.core_seal_watermark
    try:
        # A contiguous unfiltered run passes.
        stream.op_amendments[:] = [_FakeAmendment(1), _FakeAmendment(2), _FakeAmendment(3)]
        stream.amendment_seq = 3
        stream.core_seal_watermark = None
        check()

        # A middle deletion is a hole in the retained run.
        del stream.op_amendments[1]
        with pytest.raises(MetadataInvariantError, match="not contiguous"):
            check()

        # A projection legitimately starts one past the seal watermark.
        stream.op_amendments[:] = [_FakeAmendment(3), _FakeAmendment(4)]
        stream.amendment_seq = 4
        stream.core_seal_watermark = 2
        check()

        # An unexplained start offset (no watermark to justify it) is red.
        stream.core_seal_watermark = None
        with pytest.raises(MetadataInvariantError, match="starts at seq"):
            check()

        # An unstamped amendment bypassed the lane's single writer.
        stream.op_amendments[:] = [_FakeAmendment(0)]
        stream.amendment_seq = 1
        with pytest.raises(MetadataInvariantError, match="no writer-stamped seq"):
            check()

        # A stamp beyond the lane counter is a bypass too.
        stream.op_amendments[:] = [_FakeAmendment(1), _FakeAmendment(9)]
        stream.amendment_seq = 2
        with pytest.raises(MetadataInvariantError, match="exceeds the amendment writer counter"):
            check()

        # An empty lane stays legitimate.
        stream.op_amendments[:] = []
        check()
    finally:
        stream.op_amendments[:] = original_lane
        stream.amendment_seq = original_counter
        stream.core_seal_watermark = original_watermark
    check()
    assert isinstance(stream, CaptureEvents)


@pytest.mark.smoke
def test_aliased_label_registrations_emit_one_grad_event_per_pass() -> None:
    """One logical op output emits exactly ONE OpGradObserved per pass.

    An identity-output module relabels the SAME live tensor under a second
    raw label and also hooks its own logged entry, so the same logical
    gradient used to be emitted twice under one final label — caught by the
    events<->projection multiplicity reconciliation. The label-owner rule
    (one gradient owner per raw label; live registrations take ownership)
    keeps exactly one emission.
    """
    import collections

    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import OpGradObserved
    from torchlens.validation.invariants import check_metadata_invariants

    class IdentityWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self.identity = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.identity(self.linear(x))

    torch.manual_seed(42)
    trace = tl.trace(
        IdentityWrapper(),
        torch.randn(2, 3),
        capture=CaptureOptions(layers_to_save="all", save_grads="all", random_seed=42),
    )
    # Arming proof for the owner rule itself: the identity label is owned by
    # exactly one hooked tensor even though two registrations happened.
    owners = trace.__dict__.get("_tl_grad_hook_owner_by_label", {})
    identity_labels = [label for label in owners if label.startswith("identity")]
    assert identity_labels, "identity op label must be gradient-hooked"

    trace.log_backward(trace[trace.output_layers[0]].out.sum())
    events = _ensure_backward_event_stream(trace).backward_events
    op_grad_counts = collections.Counter(
        (event.op_label, event.pass_index) for event in events if isinstance(event, OpGradObserved)
    )
    assert op_grad_counts, "backward must observe op gradients"
    duplicated = {key: count for key, count in op_grad_counts.items() if count > 1}
    assert not duplicated, f"duplicate OpGradObserved emissions: {duplicated}"
    # The reconciliation tripwire stays green on the fixed producer.
    check_metadata_invariants(trace)


@pytest.mark.smoke
def test_failed_backward_walk_keeps_start_and_gains_failed_end() -> None:
    """A failed graph walk is evidence: Start stays, a failed End closes it."""
    from torchlens.backends.torch import backward as backward_mod
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardPassEnd as _End
    from torchlens.validation.invariants import check_metadata_invariants

    _model, _x, trace = _logged_model()
    loss = _output_loss(trace)

    with (
        mock.patch.object(
            backward_mod,
            "_walk_and_hook_backward_graph",
            side_effect=RuntimeError("planted walk failure"),
        ),
        pytest.raises(RuntimeError, match="planted walk failure"),
    ):
        trace.log_backward(loss)

    events = _ensure_backward_event_stream(trace).backward_events
    starts = [e for e in events if isinstance(e, BackwardPassStart)]
    ends = [e for e in events if isinstance(e, _End)]
    assert [s.pass_index for s in starts] == [1], "the attempted pass keeps its start"
    assert [e.pass_index for e in ends] == [1], "the attempted pass gains a terminal end"
    assert ends[0].status == "error"
    assert trace.num_backward_passes == 1
    check_metadata_invariants(trace)

    # A later real backward numbers itself after the failed attempt and the
    # whole stream still satisfies the exact bracketing invariants.
    trace.log_backward(_output_loss(trace))
    assert trace.num_backward_passes == 2
    check_metadata_invariants(trace)


@pytest.mark.smoke
def test_hook_registration_failure_records_typed_coverage_gap() -> None:
    """A registration skip is a typed BackwardCoverageGap, and validation fails closed."""
    from torchlens.backends.torch import backward as backward_mod
    from torchlens.backends.torch.backward import _ensure_backward_event_stream
    from torchlens.ir.events import BackwardCoverageGap
    from torchlens.validation.invariants import check_metadata_invariants

    _model, _x, trace = _logged_model()
    with mock.patch.object(
        backward_mod,
        "_make_grad_fn_hook",
        side_effect=RuntimeError("planted registration failure"),
    ):
        trace.log_backward(_output_loss(trace))

    events = _ensure_backward_event_stream(trace).backward_events
    gaps = [e for e in events if isinstance(e, BackwardCoverageGap)]
    assert gaps, "every skipped registration must record a typed gap"
    assert {gap.reason for gap in gaps} == {"registration_error"}
    assert {gap.pass_index for gap in gaps} == {1}
    for gap in gaps:
        assert gap.class_qualname
        assert "planted registration failure" in (gap.detail or "")
    # Gaps sit inside their pass bracket and the stream stays invariant-green.
    check_metadata_invariants(trace)


@pytest.mark.smoke
def test_validate_backward_fails_closed_on_coverage_gaps() -> None:
    """validate_backward_pass returns False when any unexplained gap exists."""
    from torchlens.backends.torch import backward as backward_mod
    from torchlens.validation import backward as backward_validation

    model = _TinyBackwardModel()
    x = torch.randn(2, 3)
    with (
        mock.patch.object(
            backward_mod,
            "_make_grad_fn_hook",
            side_effect=RuntimeError("planted registration failure"),
        ),
        pytest.warns(RuntimeWarning, match="coverage gap"),
    ):
        passed = backward_validation.validate_backward_pass(
            model,
            x,
            loss_fn=lambda output: output.sum(),
            random_seed=11,
        )
    assert passed is False


class _ForeachInplaceModel(nn.Module):
    """Model whose live tensor is mutated by a list-returning in-place op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Double, foreach-increment in place, then scale."""

        values = [x * 2.0]
        torch._foreach_add_(values, 1.0)
        return values[0] * 3.0


def test_foreach_inplace_live_member_owns_its_gradient() -> None:
    """The live member of a foreach in-place op emits its real gradient.

    Regression: the logged safe copy hooked first and owned the label, and the
    genuinely live (mutated) member was hooked WITHOUT ownership transfer --
    its hook fired and returned without emitting, so a real ``_foreach_add_``
    gradient produced zero journal events and ``has_grad`` stayed False while
    native autograd delivered the gradient to the live tensor.
    """

    model = _ForeachInplaceModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", save_grads="all", random_seed=4),
    )
    layer = next(op for op in trace.layer_list if "_foreach_add_" in str(op.func_name))
    trace.log_backward(trace[trace.output_layers[0]].out.sum())

    observed = [
        event
        for event in trace._capture_events.backward_events
        if isinstance(event, OpGradObserved) and event.op_label == layer.layer_label
    ]
    assert len(observed) == 1
    assert layer.has_grad
    assert torch.equal(layer.grad, torch.full_like(x, 3.0))


class _InplaceReluModel(nn.Module):
    """Model with a same-object in-place mutation on the live path."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear, in-place relu, then scale."""

        hidden = self.fc(x)
        torch.relu_(hidden)
        return hidden * 2.0


def test_refresh_projection_with_inplace_op_keeps_one_grad_owner_per_label() -> None:
    """After save_new_outs, each label emits exactly one gradient observation.

    Regression: the one-owner check ran against the hook's OWN trace's owner
    map BEFORE the refresh-projection redirect, so a stale source-trace hook
    on the in-place live tensor and the rebound target hook could both pass
    their own maps and emit duplicate OpGradObserved events for one label on
    the same final target.
    """

    from collections import Counter

    model = _InplaceReluModel()
    trace = tl.trace(
        model,
        torch.randn(2, 3, requires_grad=True),
        capture=CaptureOptions(layers_to_save="all", save_grads="all", random_seed=0),
    )
    trace.save_new_outs(model, torch.randn(2, 3, requires_grad=True), random_seed=1)
    trace.log_backward(trace[trace.output_layers[0]].out.sum())

    pass_index = trace.num_backward_passes
    per_label = Counter(
        event.op_label
        for event in trace._capture_events.backward_events
        if isinstance(event, OpGradObserved) and event.pass_index == pass_index
    )
    assert per_label, "backward produced no gradient observations"
    duplicated = {label: count for label, count in per_label.items() if count > 1}
    assert not duplicated, duplicated


class _NaNGradParamModel(nn.Module):
    """Model whose ``scale`` parameter deterministically earns a NaN gradient.

    ``(scale * 0.0) * inf`` contributes NaN to ``scale``'s grad through the
    identical graph in BOTH the stock-autograd and TorchLens-captured
    pipelines, so the two NaN patterns agree exactly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nan_term = (self.scale * 0.0) * torch.tensor(float("inf"))
        return self.lin(x) + nan_term


def test_backward_validation_nan_bearing_grads_do_not_false_fail() -> None:
    """A CORRECT NaN-bearing gradient must not fail backward validation.

    Both pipelines compute the same NaN pattern for ``scale``; the comparison
    follows tensor_nanequal's doctrine (identical NaN patterns are agreement,
    ``equal_nan=True``). The bare ``torch.allclose`` this replaces returned
    False for ANY NaN, so validation false-FAILED gradients TorchLens had
    captured perfectly (NaN-vs-number still fails elementwise).
    """

    model = _NaNGradParamModel().eval()
    x = torch.randn(2, 4)
    assert backward_validation.validate_backward_pass(model, (x,), random_seed=0) is True


def test_gradient_validation_tolerances_are_named_constants() -> None:
    """The grad tolerance pairs are spelled ONCE and shared by every consumer.

    Two backward checks of one capture used to disagree 10x via bare literals
    in three files; the pairs now live in torchlens.utils.tensor_utils with a
    documented error model (param grads = batch/position reductions, layer
    grads and RF adjoint probes = elementwise comparisons).

    Dtype-aware update (round 3): every verdict site's ``atol``/``rtol`` now
    DEFAULTS TO ``None`` -- the sentinel meaning "derive the band per output
    dtype" via :func:`derive_float_tolerances`. The named fp32-baseline
    constants remain the SINGLE SOURCE those derivations key on (fp32 headroom),
    so the no-magic-number / spelled-once invariant is preserved; the params
    just no longer hard-code the fp32 pair as their default.
    """

    import inspect

    from torchlens.receptive_field import verify as rf_verify
    from torchlens.utils.tensor_utils import (
        LAYER_GRAD_VALIDATION_ATOL,
        LAYER_GRAD_VALIDATION_RTOL,
        PARAM_GRAD_VALIDATION_ATOL,
        PARAM_GRAD_VALIDATION_RTOL,
        derive_float_tolerances,
    )
    from torchlens.validation._layer_grad_report import _compare_module_output_grads

    # New contract: every tolerance param defaults to the dtype-aware sentinel
    # (None -> derived per output dtype at the verdict site), not a hard-coded pair.
    backward_params = inspect.signature(backward_validation.validate_backward_pass).parameters
    assert backward_params["atol"].default is None
    assert backward_params["rtol"].default is None

    layer_params = inspect.signature(_compare_module_output_grads).parameters
    assert layer_params["atol"].default is None
    assert layer_params["rtol"].default is None

    rf_params = inspect.signature(rf_verify).parameters
    assert rf_params["empirical_adjoint_atol"].default is None
    assert rf_params["empirical_adjoint_rtol"].default is None

    # The named constants remain the SINGLE fp32-baseline source (spelled once):
    # the elementwise (layer/RF adjoint) pair is 10x tighter than the reduction
    # (param) pair by design, and both are real named constants, never literals.
    assert pytest.approx(PARAM_GRAD_VALIDATION_RTOL / 10) == LAYER_GRAD_VALIDATION_RTOL
    assert pytest.approx(PARAM_GRAD_VALIDATION_ATOL / 10) == LAYER_GRAD_VALIDATION_ATOL

    # And the fp32 derivation keys on those same named constants (single source):
    # deriving at float32 reproduces the named fp32 pair rather than a fresh literal.
    fp32_atol, fp32_rtol = derive_float_tolerances(torch.float32, ulp_headroom=0.0)
    assert isinstance(fp32_atol, float) and isinstance(fp32_rtol, float)


def test_success_path_handle_removal_never_swallows_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """grind-r6 b8 R63 (fable, probe): SIGINT during success-path cleanup.

    The success-path finally tail folds every cleanup step's failure into
    ``cleanup_error`` and re-raises the FIRST at the end -- except the
    hook-handle removal loop, which used ``suppress(BaseException)`` and
    discarded a KeyboardInterrupt delivered during handle removal after a
    SUCCESSFUL backward: ``log_backward`` returned normally and the interrupt
    vanished. There is no primary exception on this path whose precedence
    could justify the swallow.
    """
    from torchlens import _state
    from torchlens.backends.torch import backward

    _model, _x, trace = _logged_model()

    class _InterruptedHandle:
        """Handle whose removal is hit by a pending SIGINT exactly once."""

        fired = False

        def remove(self) -> None:
            """Raise KeyboardInterrupt on the first removal attempt."""
            if not self.fired:
                self.fired = True
                raise KeyboardInterrupt

    hostile = _InterruptedHandle()
    real_walk = backward._walk_and_hook_backward_graph

    def walk_then_plant(trace_arg: tl.Trace, loss: torch.Tensor, handles: list[object]) -> None:
        """Run the real walk, then plant the interrupt-carrying handle."""
        real_walk(trace_arg, loss, handles)
        handles.append(hostile)

    monkeypatch.setattr(backward, "_walk_and_hook_backward_graph", walk_then_plant)
    with pytest.raises(KeyboardInterrupt):
        trace.log_backward(_output_loss(trace))

    assert hostile.fired is True
    assert _state._active_trace is None
