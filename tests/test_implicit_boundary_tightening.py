"""Implicit-boundary tightening pins (L9 memo 1.2).

The close routine is a journal/scavenge/finalize split whose FINALIZE guard
lives INSIDE ``_close_implicit_backward_pass_if_open`` so no caller can
bypass it: JOURNAL + SCAVENGE run on every close on any thread; FINALIZE
(the R36-1 D2H fence + full projection) runs only outside an engine
invocation, deferring with a sticky pending flag otherwise. Implicit passes
additionally journal their close at the engine-drain boundary via a queued
final callback that identity-checks (pass_index, graph_task_id); the
sync-point path stays armed as the guaranteed backstop. The two starred
tests are the MERGE PRECONDITIONS from the memo. Spellings
(``close_path`` values) DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends.torch import backward as backward_mod, tensor_tracking as tracking_mod
from torchlens.ir.events import BackwardPassEnd, BackwardPassStart

_HAS_GRAPH_TASK_ID = hasattr(torch._C, "_current_graph_task_id")


class _PlainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _plain_trace() -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        _PlainModel(),
        torch.randn(3, 4, requires_grad=True),
        capture=tl.options.CaptureOptions(layers_to_save="all", save_grads="all"),
    )


def _ends(trace: tl.Trace) -> list[BackwardPassEnd]:
    return [event for event in trace.backward_events if isinstance(event, BackwardPassEnd)]


def _run_unmanaged_backward(trace: tl.Trace, *, retain_graph: bool = False) -> None:
    loss = trace[trace.output_layers[0]].out.sum()
    original_backward = backward_mod._ORIGINAL_AUTOGRAD_BACKWARD
    assert original_backward is not None
    original_backward((loss,), retain_graph=retain_graph)


# ---------------------------------------------------------------------------
# (STAR) merge precondition 1: journal-only close then IMMEDIATE read --
# the D2H fence runs before any record becomes reachable AND the pending
# accumulate-grad table is empty (the no-bypass pin).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_GRAPH_TASK_ID, reason="no graph-task identity on this torch")
def test_star_journal_only_close_then_immediate_read_fences_before_records(monkeypatch) -> None:
    trace = _plain_trace()
    # Suppress the engine drain so the pass is still open after backward.
    monkeypatch.setattr(backward_mod, "_enqueue_implicit_pass_drain_callback", lambda *args: None)
    _run_unmanaged_backward(trace)
    assert trace._implicit_backward_pass_open

    # Journal-only close: simulate being inside an engine invocation so the
    # in-routine guard defers FINALIZE.
    monkeypatch.setattr(backward_mod, "_current_backward_graph_task_id", lambda: 7)
    backward_mod._close_implicit_backward_pass_if_open(trace)
    assert not trace._implicit_backward_pass_open
    assert backward_mod._backward_finalize_pending(trace)
    # SCAVENGE ran at journal time on this (engine-simulated) close.
    assert trace.__dict__.get("_tl_pending_accumulate_grad_fire_records", {}) == {}

    # Mid-engine read: journals, but must NOT materialize while finalize is
    # pending (records would become reachable ahead of the D2H fence).
    materialize_calls: list[str] = []
    real_materialize = backward_mod._materialize_backward_projections

    def recording_materialize(target) -> None:
        materialize_calls.append("materialize")
        real_materialize(target)

    real_fence = backward_mod.synchronize_pending_cpu_async_copies

    def recording_fence() -> None:
        materialize_calls.append("fence")
        real_fence()

    monkeypatch.setattr(backward_mod, "_materialize_backward_projections", recording_materialize)
    monkeypatch.setattr(backward_mod, "synchronize_pending_cpu_async_copies", recording_fence)
    assert len(trace.backward_passes) == 0
    assert materialize_calls == []
    assert backward_mod._backward_finalize_pending(trace)

    # First post-pass read: FINALIZE runs fully, fence strictly before the
    # projection makes records reachable.
    monkeypatch.setattr(backward_mod, "_current_backward_graph_task_id", lambda: None)
    passes = trace.backward_passes
    assert len(passes) == 1
    assert materialize_calls[0] == "fence"
    assert "materialize" in materialize_calls
    assert materialize_calls.index("fence") < materialize_calls.index("materialize")
    assert not backward_mod._backward_finalize_pending(trace)
    trace.cleanup()


# ---------------------------------------------------------------------------
# (STAR) merge precondition 2: engine-drain vs sync-point close produce
# byte-identical projected records INCLUDING gradient payload values, on a
# cpu_async-device grad capture.
# ---------------------------------------------------------------------------


def _cpu_async_capture(*, engine_drain: bool, monkeypatch) -> tuple[tl.Trace, nn.Module]:
    if not engine_drain:
        monkeypatch.setattr(
            backward_mod, "_enqueue_implicit_pass_drain_callback", lambda *args: None
        )
    torch.manual_seed(0)
    model = _PlainModel()
    trace = tl.trace(
        model,
        torch.randn(3, 4, requires_grad=True),
        save_mode="cpu_async",
        capture=tl.options.CaptureOptions(layers_to_save="all", save_grads="all"),
    )
    _run_unmanaged_backward(trace)
    backward_mod._close_implicit_backward_pass_if_open(trace)
    # The model rides along so lazy Param.grad reads stay live.
    return trace, model


@pytest.mark.skipif(not _HAS_GRAPH_TASK_ID, reason="no graph-task identity on this torch")
def test_star_engine_drain_and_sync_point_project_identical_records(monkeypatch) -> None:
    with pytest.MonkeyPatch.context() as drain_patch:
        drained, drained_model = _cpu_async_capture(engine_drain=True, monkeypatch=drain_patch)
    with pytest.MonkeyPatch.context() as sync_patch:
        synced, synced_model = _cpu_async_capture(engine_drain=False, monkeypatch=sync_patch)
    del monkeypatch

    drain_paths = [event.close_path for event in _ends(drained)]
    sync_paths = [event.close_path for event in _ends(synced)]
    assert "engine_drain" in drain_paths
    assert drain_paths.count("engine_drain") == 1
    assert sync_paths == ["sync_point"]

    from torchlens.ir.events import OpGradObserved

    # Implicit (unmanaged) passes carry no walked grad-fn hooks; their
    # projected records are the pass records plus op/param gradient events.
    drained_passes = list(drained.backward_passes)
    synced_passes = list(synced.backward_passes)
    assert len(drained_passes) == len(synced_passes) == 1

    def _op_grad_payloads(trace: tl.Trace) -> dict[tuple[str, int], torch.Tensor]:
        return {
            (event.op_label, event.pass_index): event.payload_ref
            for event in trace.backward_events
            if isinstance(event, OpGradObserved) and isinstance(event.payload_ref, torch.Tensor)
        }

    drained_ops = _op_grad_payloads(drained)
    synced_ops = _op_grad_payloads(synced)
    assert drained_ops.keys() == synced_ops.keys()
    assert drained_ops, "cpu_async capture retained no op gradient payloads"
    for key, left_payload in drained_ops.items():
        assert torch.equal(left_payload, synced_ops[key])
    drained_params = {
        address: record.grad
        for address, record in drained.param_logs.items()
        if record.grad is not None
    }
    synced_params = {
        address: record.grad
        for address, record in synced.param_logs.items()
        if record.grad is not None
    }
    assert drained_params.keys() == synced_params.keys()
    assert drained_params, "cpu_async capture saved no parameter gradients"
    for address, left_grad in drained_params.items():
        assert torch.equal(left_grad, synced_params[address])
    del drained_model, synced_model
    drained.cleanup()
    synced.cleanup()


# ---------------------------------------------------------------------------
# Engine-drain journal close on a plain implicit backward.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_GRAPH_TASK_ID, reason="no graph-task identity on this torch")
def test_engine_drain_journals_close_without_any_read() -> None:
    trace = _plain_trace()
    _run_unmanaged_backward(trace)
    # No explicit close, no read: the drain callback already journaled.
    assert not trace._implicit_backward_pass_open
    ends = _ends(trace)
    assert len(ends) == 1
    assert ends[0].close_path == "engine_drain"
    assert trace not in tracking_mod._IMPLICIT_BACKWARD_TASK_IDS
    # FINALIZE deferred by construction (the drain runs inside the engine
    # invocation); the first read settles it.
    assert backward_mod._backward_finalize_pending(trace)
    assert len(trace.backward_passes) == 1
    assert not backward_mod._backward_finalize_pending(trace)
    trace.cleanup()


def test_sync_point_fallback_when_engine_handle_unavailable(monkeypatch) -> None:
    trace = _plain_trace()
    real_engine = torch.autograd.Variable._execution_engine

    class _NoQueueProxy:
        """Real engine with only the final-callback queue unavailable."""

        def __getattr__(self, name):
            return getattr(real_engine, name)

        def queue_callback(self, callback) -> None:
            del callback
            raise RuntimeError("final callbacks unavailable")

    monkeypatch.setattr(
        torch.autograd.Variable, "_execution_engine", _NoQueueProxy(), raising=False
    )
    _run_unmanaged_backward(trace)
    # Enqueue degraded: current behavior unchanged, pass still open.
    assert trace._implicit_backward_pass_open
    backward_mod._close_implicit_backward_pass_if_open(trace)
    ends = _ends(trace)
    assert ends[-1].close_path == "sync_point"
    assert trace not in tracking_mod._IMPLICIT_BACKWARD_TASK_IDS
    trace.cleanup()


@pytest.mark.smoke
@pytest.mark.skipif(not _HAS_GRAPH_TASK_ID, reason="no graph-task identity on this torch")
def test_stale_drain_callback_never_closes_a_newer_pass(monkeypatch) -> None:
    callbacks: list = []
    real_enqueue = backward_mod._enqueue_implicit_pass_drain_callback

    def recording_enqueue(trace, pass_index, graph_task_id):
        callback = real_enqueue(trace, pass_index, graph_task_id)
        if callback is not None:
            callbacks.append(callback)
        return callback

    # tensor_tracking resolves the enqueue helper from the backward module at
    # call time, so patching the module attribute covers the open path.
    monkeypatch.setattr(backward_mod, "_enqueue_implicit_pass_drain_callback", recording_enqueue)
    trace = _plain_trace()
    _run_unmanaged_backward(trace, retain_graph=True)
    _run_unmanaged_backward(trace)
    backward_mod._close_implicit_backward_pass_if_open(trace)
    assert len(callbacks) == 2
    end_count = len(_ends(trace))
    # Fire the first (stale) callback again after everything settled: the
    # identity check must make it a no-op.
    callbacks[0]()
    assert len(_ends(trace)) == end_count
    trace.cleanup()


@pytest.mark.skipif(not _HAS_GRAPH_TASK_ID, reason="no graph-task identity on this torch")
def test_back_to_back_engine_calls_close_and_reopen_cleanly() -> None:
    trace = _plain_trace()
    _run_unmanaged_backward(trace, retain_graph=True)
    _run_unmanaged_backward(trace)
    backward_mod._close_implicit_backward_pass_if_open(trace)
    starts = [
        event
        for event in trace.backward_events
        if isinstance(event, BackwardPassStart) and event.implicit
    ]
    assert [event.pass_index for event in starts] == [1, 2]
    ends = _ends(trace)
    assert sorted(event.pass_index for event in ends) == [1, 2]
    # No stale accumulate-grad record crossed passes (SCAVENGE at journal
    # time) and the task-id table never leaks.
    assert trace.__dict__.get("_tl_pending_accumulate_grad_fire_records", {}) == {}
    assert trace not in tracking_mod._IMPLICIT_BACKWARD_TASK_IDS
    trace.cleanup()


def test_error_path_backstop_sync_point_still_closes(monkeypatch) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4, requires_grad=True)
    trace = tl.trace(
        _PlainModel(),
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_grads="all"),
    )
    loss = trace[trace.output_layers[0]].out.sum()
    original_backward = backward_mod._ORIGINAL_AUTOGRAD_BACKWARD
    assert original_backward is not None

    def exploding_hook(grad: torch.Tensor) -> torch.Tensor:
        del grad
        raise RuntimeError("hook boom")

    # Hook the LIVE leaf (not a saved copy) so the engine hits it mid-drain,
    # after TorchLens tensor hooks already opened the implicit pass.
    handle = x.register_hook(exploding_hook)
    try:
        with pytest.raises(RuntimeError, match="hook boom"):
            original_backward((loss,))
    finally:
        handle.remove()
    # Final callbacks skip the error path: the pass may still be open; the
    # sync-point backstop closes it.
    backward_mod._close_implicit_backward_pass_if_open(trace)
    assert not trace._implicit_backward_pass_open
    assert trace not in tracking_mod._IMPLICIT_BACKWARD_TASK_IDS
    trace.cleanup()


@pytest.mark.skipif(not _HAS_GRAPH_TASK_ID, reason="no graph-task identity on this torch")
def test_finalize_never_runs_inside_an_engine_invocation(monkeypatch) -> None:
    observed: list[int | None] = []
    real_materialize = backward_mod._materialize_backward_projections

    def guarded_materialize(target) -> None:
        observed.append(backward_mod._current_backward_graph_task_id())
        real_materialize(target)

    monkeypatch.setattr(backward_mod, "_materialize_backward_projections", guarded_materialize)
    trace = _plain_trace()
    _run_unmanaged_backward(trace)
    backward_mod._close_implicit_backward_pass_if_open(trace)
    assert len(trace.backward_passes) == 1
    assert observed, "materialize never observed"
    assert all(task_id is None for task_id in observed)
    trace.cleanup()
