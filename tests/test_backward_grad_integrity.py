"""Backward/grad integrity regression tests (fix-bwgrad lane, grind-p3 T2).

Each test here is red-capable: it FAILS against the defect it pins and passes
only with the corresponding fix in place.
"""

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class _TinyModel(nn.Module):
    """Small MLP for backward capture tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""
        return self.fc2(torch.relu(self.fc1(x)))


def _armed_trace(save_mode: str = "copy") -> tl.Trace:
    """Capture a backward-ready trace of the tiny model."""

    torch.manual_seed(0)
    model = _TinyModel()
    x = torch.randn(2, 3, requires_grad=True)
    return tl.trace(
        model,
        x,
        capture=CaptureOptions(backward_ready=True, save_grads="all"),
        save_mode=save_mode,
    )


def _loss(trace: tl.Trace) -> torch.Tensor:
    """Return a scalar loss built from the trace's saved output."""

    return trace[trace.output_layers[0]].out.sum()


@pytest.mark.parametrize("save_mode", ["reference", "view"])
def test_second_backward_does_not_rewrite_recorded_grads(save_mode: str) -> None:
    """Recorded pass-1 grad payloads stay frozen when a second backward runs.

    Under save_mode="reference"/"view" the grad payload chokepoint used to
    store a live ALIAS of the observed gradient; AccumulateGrad steals that
    tensor as ``.grad`` and accumulates into it in place, so the pass-1
    record silently became the running sum (2x truth) after a second
    backward.
    """

    trace = _armed_trace(save_mode=save_mode)
    loss = _loss(trace)
    trace.log_backward(loss, retain_graph=True)

    param_payloads = [record.grad for param in trace.param_logs.values() for record in param.grads]
    op_payloads = [
        record.out if hasattr(record, "out") else record.grad
        for layer in trace.layer_list
        for record in layer.grads
    ]
    payloads = [p for p in param_payloads + op_payloads if isinstance(p, torch.Tensor)]
    assert payloads, "expected recorded pass-1 gradient payloads"
    snapshots = [p.clone() for p in payloads]

    trace.log_backward(loss)

    for payload, snapshot in zip(payloads, snapshots):
        assert torch.equal(payload, snapshot), (
            "pass-1 gradient record was rewritten by a second backward: "
            f"{snapshot.flatten()[:4]} -> {payload.flatten()[:4]}"
        )


def test_reentered_recording_backward_restores_tensor_backward() -> None:
    """Re-entering one RecordingBackward context leaves no persistent wrapper.

    A second ``__enter__`` on the same context object used to capture the
    first entry's wrapper as the "original", so the outer exit could never
    match its own wrapper and permanently leaked a TorchLens wrapper on the
    process-global ``torch.Tensor.backward``.
    """

    trace = _armed_trace()
    original_backward = torch.Tensor.backward
    context = trace.recording_backward()
    try:
        with context:
            with context:
                _loss(trace).backward()
    finally:
        if torch.Tensor.backward is not original_backward:
            torch.Tensor.backward = original_backward  # type: ignore[method-assign]
            pytest.fail("re-entered recording_backward() leaked a wrapper on torch.Tensor.backward")
    # The backward inside the nested block is still recorded exactly once.
    assert trace.num_backward_passes == 1


def test_op_grad_payloads_charge_the_save_budget() -> None:
    """Retained op-gradient payloads charge the save-budget accountant.

    Frozen parameters isolate the op/output gradient path (no param grads
    exist): the committed footprint must grow when backward retains op grad
    payloads, otherwise the budget's committed-footprint claim is false
    after any backward.
    """

    torch.manual_seed(0)
    model = _TinyModel()
    for param in model.parameters():
        param.requires_grad_(False)
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(backward_ready=True, save_grads="all"),
    )
    budget = trace._save_budget_accountant
    assert budget is not None
    committed_before = sum(ledger.committed_bytes for ledger in budget.ledgers.values())

    trace.log_backward(_loss(trace))

    grad_bytes = sum(
        int(record.grad.nelement() * record.grad.element_size())
        for layer in trace.layer_list
        for record in layer.grads
        if isinstance(record.grad, torch.Tensor)
    )
    assert grad_bytes > 0, "expected retained op gradient payloads"
    committed_after = sum(ledger.committed_bytes for ledger in budget.ledgers.values())
    assert committed_after >= committed_before + grad_bytes, (
        "op gradient payloads bypassed the save-budget accountant: "
        f"committed {committed_before} -> {committed_after}, grads {grad_bytes}"
    )


@pytest.mark.smoke
def test_selective_save_grads_consults_predicate_for_params() -> None:
    """Callable save_grads policies see parameter gradients, not a silent drop.

    A selector/callable policy used to bypass parameters entirely (no param
    context existed), so every param grad payload silently dropped no matter
    what the predicate asked for.
    """

    seen_kinds: set[str] = set()

    def keep_param_grads(ctx: object) -> bool:
        """Retain exactly the parameter gradients."""

        kind = getattr(ctx, "grad_kind", None)
        if isinstance(kind, str):
            seen_kinds.add(kind)
        return kind == "param_grad"

    trace = _armed_trace()
    trace.log_backward(_loss(trace), save_grads=keep_param_grads)

    param_payloads = [record.grad for param in trace.param_logs.values() for record in param.grads]
    assert param_payloads, "expected parameter gradient records"
    assert "param_grad" in seen_kinds, "predicate never saw a parameter context"
    assert all(isinstance(p, torch.Tensor) for p in param_payloads), (
        "selective save_grads silently dropped parameter gradients"
    )
    # The same predicate rejected op gradients, so none retain payloads.
    op_payloads = [
        record.out if hasattr(record, "out") else record.grad
        for layer in trace.layer_list
        for record in layer.grads
    ]
    assert all(not isinstance(p, torch.Tensor) for p in op_payloads)


def test_backward_finalize_drains_pending_cpu_async_copies() -> None:
    """The backward finalize seam fences pending cpu_async D2H copies.

    The R36-1 drain only ran at the FORWARD finalize seam, so a cpu_async
    grad payload recorded during log_backward stayed on the pending-event
    list and a host read could observe partial bytes from the unfinished
    non_blocking copy. A planted fence object stands in for an in-flight
    CUDA copy event (CPU-only hosts record no real events).
    """

    from torchlens.utils import tensor_utils

    class _FenceSpy:
        """Pending-copy stand-in recording whether it was synchronized."""

        def __init__(self) -> None:
            self.synchronized = False

        def synchronize(self) -> None:
            """Mark the pending copy as fenced."""

            self.synchronized = True

    trace = _armed_trace(save_mode="cpu_async")
    fence = _FenceSpy()
    tensor_utils._CPU_ASYNC_PENDING_EVENTS.append(fence)
    try:
        trace.log_backward(_loss(trace))
        assert fence.synchronized, (
            "log_backward finished without draining pending cpu_async copy fences"
        )
        assert fence not in tensor_utils._CPU_ASYNC_PENDING_EVENTS
    finally:
        if fence in tensor_utils._CPU_ASYNC_PENDING_EVENTS:
            tensor_utils._CPU_ASYNC_PENDING_EVENTS.remove(fence)


def test_backward_graph_task_id_routes_through_torch_compat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The graph-task-id probe routes through the _torch_compat chokepoint.

    The LOCKED compat doctrine requires every fragile torch-private-API probe
    to flip a named HAS_* capability flag; the inline
    ``getattr(torch._C, "_current_graph_task_id", None)`` probe bypassed it.
    """

    from torchlens.backends.torch import tensor_tracking
    from torchlens.utils import _torch_compat

    assert hasattr(_torch_compat, "HAS_CURRENT_GRAPH_TASK_ID")
    assert "HAS_CURRENT_GRAPH_TASK_ID" in _torch_compat._CAPABILITY_ATTRS

    # Routing: the hook-side helper must consume the compat accessor, not a
    # private inline torch._C probe.
    monkeypatch.setattr(tensor_tracking, "get_current_graph_task_id_fn", lambda: lambda: 42)
    assert tensor_tracking._current_backward_graph_task_id() == 42
    monkeypatch.setattr(tensor_tracking, "get_current_graph_task_id_fn", lambda: None)
    assert tensor_tracking._current_backward_graph_task_id() is None


@pytest.mark.parametrize("save_mode", ["reference", "view"])
def test_legacy_grad_slot_stores_snapshot_not_alias(save_mode: str) -> None:
    """``Op.log_tensor_grad`` snapshots the observed gradient under all modes.

    The legacy backward-hook write went through ``safe_copy(save_mode=...)``,
    which returns a storage ALIAS under ``reference``/``view``. Output-layer
    ops receive their grad ONLY via this legacy propagation, so the recorded
    grad aliased the caller's seed gradient and later user mutation of the
    seed silently rewrote the recorded value with no disclosure.
    """

    trace = _armed_trace(save_mode=save_mode)
    out_op = trace[trace.output_layers[0]]
    seed = torch.full_like(out_op.out, 2.0)
    trace.log_backward(out_op.out, gradient=seed)

    seed_ptr = seed.untyped_storage().data_ptr()
    recorded: dict[str, torch.Tensor] = {}
    for label in trace.layer_labels:
        payload = getattr(trace[label], "grad", None)
        if isinstance(payload, torch.Tensor):
            recorded[label] = payload
    assert recorded, "expected recorded gradient payloads"
    aliased = [
        label
        for label, payload in recorded.items()
        if payload.untyped_storage().data_ptr() == seed_ptr
    ]
    assert aliased == [], f"recorded grads alias the caller's seed gradient: {aliased}"

    snapshots = {label: payload.clone() for label, payload in recorded.items()}
    seed.mul_(1234.5)
    rewritten = [
        label for label, payload in recorded.items() if not torch.equal(payload, snapshots[label])
    ]
    assert rewritten == [], f"seed mutation rewrote recorded grads: {rewritten}"


def test_per_call_save_grads_false_gates_legacy_layer_slot() -> None:
    """``log_backward(..., save_grads=False)`` disables legacy slot retention.

    The legacy layer-slot write was gated on the deprecated ``save_grads``
    trace attribute instead of the active per-call policy, so a capture armed
    with ``save_grads="all"`` kept retaining full grad payloads on a call
    that explicitly disabled retention.
    """

    trace = _armed_trace()
    trace.log_backward(_loss(trace), save_grads=False)
    retained = [
        label
        for label in trace.layer_labels
        if isinstance(getattr(trace[label], "grad", None), torch.Tensor)
    ]
    assert retained == [], f"save_grads=False call retained legacy grad payloads: {retained}"


def test_saved_grad_hook_clones_once_and_transforms_once() -> None:
    """grind-r6 b5 R34-N1/R35-N1 (fable+opus, probe-corroborated).

    The tensor hook fired BOTH retention paths: the charged event-sidecar
    build AND the legacy layer-slot write, which minted a second independent
    clone and executed ``grad_transform`` a SECOND time per firing --
    uncharged by the save budget (saved grads retained ~2x while the budget
    saw 1x). The layer slots must now REUSE the event-sidecar payload
    objects: one transform execution per firing, identical payload objects
    in both surfaces.
    """

    transform_calls: list[int] = []

    def counting_transform(grad: torch.Tensor) -> torch.Tensor:
        transform_calls.append(1)
        return grad * 2.0

    torch.manual_seed(0)
    model = _TinyModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(backward_ready=True, save_grads="all"),
        grad_transform=counting_transform,
    )
    trace.log_backward(_loss(trace))

    from torchlens.ir.events import OpGradObserved

    grad_events = [e for e in trace.backward_events if isinstance(e, OpGradObserved)]
    fired = [
        e for e in grad_events if e.payload_ref is not None or e.transformed_payload_ref is not None
    ]
    assert fired, "no gradient events retained payloads; test premise broken"
    assert len(transform_calls) == len(fired), (
        f"grad_transform executed {len(transform_calls)}x for {len(fired)} retained "
        "gradient events -- the legacy layer slot re-ran the transform"
    )

    # The layer slots hold the SAME payload objects the event sidecar charged
    # (identity reuse, not a second uncharged clone).
    events_by_label = {e.op_label: e for e in fired}
    reused = 0
    for label, event in events_by_label.items():
        layer = trace.layer_dict_all_keys.get(label)
        if layer is None or not getattr(layer, "has_grad", False):
            continue
        transformed_slot = getattr(layer, "transformed_grad", None)
        if transformed_slot is not None:
            assert transformed_slot is event.transformed_payload_ref, (
                f"{label}: layer transformed_grad is a second clone, not the charged event payload"
            )
            reused += 1
        raw_slot = getattr(layer, "grad", None)
        if raw_slot is not None and event.payload_ref is not None:
            assert raw_slot is event.payload_ref, (
                f"{label}: layer grad is a second clone, not the charged event payload"
            )
    assert reused, "no layer slot carried a transformed payload; premise broken"
