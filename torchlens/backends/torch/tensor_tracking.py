"""Track torch tensor provenance, family links, and equivalence classes.

This module owns tensor backward hooks, parent argument positions, parameter pass
tracking, and structural fingerprints used by loop detection.
"""

import time
import warnings
import weakref
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from ... import _state
from ..._capture_state_helpers import _is_uninitialized_param
from ..._state import pause_logging
from ...data_classes.op import Op
from ...fastlog.types import CaptureSpec
from ...intervention.selectors import BaseSelector
from ...ir.events import BackwardPassStart, OpGradObserved
from ...utils._torch_compat import get_current_graph_task_id_fn
from ...utils.display import _record_phase_timing
from ...utils.hashing import make_random_barcode, make_short_barcode_from_input
from ...utils.tensor_utils import SaveMode, safe_copy
from ._tl import (
    get_param_meta,
    get_tensor_label,
    increment_param_call_index,
    mutated_parameter_label,
    set_param_meta,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


_IMPLICIT_BACKWARD_TASK_IDS: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()


def _is_fork_relative(trace: "Trace", other: "Trace") -> bool:
    """Return whether two traces are related through the fork parent chain.

    A fork keeps a ``parent_run`` weakref to its source; two traces are
    relatives when either appears on the other's (bounded) parent chain.
    Structural corollary relied on by callers: fork relatives share one op
    label space, because a fork is a structural copy of its parent.

    Parameters
    ----------
    trace:
        First trace.
    other:
        Second trace.

    Returns
    -------
    bool
        ``True`` when one trace is a fork ancestor of the other.
    """

    for start, target in ((trace, other), (other, trace)):
        current: Any = start
        for _ in range(64):
            parent_ref = getattr(current, "parent_run", None)
            parent = parent_ref() if callable(parent_ref) else None
            if parent is None:
                break
            if parent is target:
                return True
            current = parent
    return False


def _add_tensor_backward_hook(
    trace: "Trace", t: torch.Tensor, tensor_label: str, *, take_ownership: bool = False
) -> None:
    """Register a backward hook on ``t`` that captures its grad into Trace.

    The hook closure captures a ``weakref`` to Trace (not a strong reference)
    so that the hook doesn't prevent GC of the Trace after the user drops it
    (GC-8).  The closure also captures ``tensor_label`` (a string) rather than
    the tensor itself, avoiding circular references.

    Only tensors that participate in autograd (have grad_fn_handle or require_grad)
    get hooks — others would never receive grads.

    One label, one gradient OWNER: a raw label names exactly one logical op
    output, so exactly one hooked tensor may emit ``OpGradObserved`` for it —
    otherwise aliased registrations (an identity module's relabeled live
    tensor, an in-place op's live result) double-emit the same logical fact
    and trip the events<->projection multiplicity reconciliation. The first
    registration owns the label; a later registration with
    ``take_ownership=True`` (the live-tensor path, whose premise is that the
    logged object is a graph dead end) transfers ownership. Non-owner hooks
    stay registered but drop their fire.

    Args:
        t: The tensor to hook.
        tensor_label: Raw tensor label (e.g. ``"conv2d_3_47_raw"``) used to
            look up the corresponding log entry when the grad arrives.
        take_ownership: Transfer gradient-emission ownership of
            ``tensor_label`` to this tensor even if another tensor already
            holds it.
    """
    # r65: TorchLens's OWN hook-bookkeeping ``grad_fn``/``requires_grad`` reads, hoisted
    # under the explicit internal-read marker so the r65 state-metadata property observer
    # never mistakes them for user autograd reads on a registered buffer/param receiver.
    from .completeness_witness import internal_scalar_read

    with internal_scalar_read():
        _grad_fn = t.grad_fn
        _requires_grad = bool(t.requires_grad)
    if _grad_fn is not None:
        from .backward import _register_forward_grad_fn

        _register_forward_grad_fn(trace, _grad_fn, tensor_label)
    # Deferred gradient selectors (all selective grad selections, including
    # positive integer ordinals — FINAL layer numbers unknowable during the
    # forward) install their hooks post-postprocess from the reference escrow.
    should_defer_hook = getattr(
        trace, "_deferred_gradient_selector", None
    ) is not None and not getattr(trace, "_installing_deferred_gradient_hooks", False)
    if (
        not getattr(trace, "capture_tensor_grad_hooks", True)
        or should_defer_hook
        or (_grad_fn is None and not _requires_grad)
    ):
        return

    hooked_tensors = trace.__dict__.setdefault("_tl_backward_hooked_tensor_keys", set())
    hook_key = (tensor_label, id(t))
    grad_hook_owners = trace.__dict__.setdefault("_tl_grad_hook_owner_by_label", {})
    if take_ownership or tensor_label not in grad_hook_owners:
        grad_hook_owners[tensor_label] = id(t)
    if hook_key in hooked_tensors:
        return
    hooked_tensors.add(hook_key)

    # Weak reference prevents Trace -> tensor -> hook -> Trace ref cycle.
    trace_ref = weakref.ref(trace)
    hooked_tensor_id = id(t)

    def log_grad_to_model_history(grad: torch.Tensor) -> None:
        """Emit and optionally retain one gradient observed by a tensor hook."""
        active_trace = trace_ref()
        # One-owner-per-label: a non-owner alias registration drops its fire
        # so one logical op output emits exactly one OpGradObserved per pass.
        if active_trace is not None:
            owner_map = active_trace.__dict__.get("_tl_grad_hook_owner_by_label")
            if (
                owner_map is not None
                and owner_map.get(tensor_label, hooked_tensor_id) != hooked_tensor_id
            ):
                return
        refresh_target_ref = getattr(active_trace, "_refresh_projection_target_ref", None)
        if refresh_target_ref is not None:
            active_trace = refresh_target_ref()
        if _state._rf_probe_depth > 0:
            # A fork probe resolves here to the BASE trace whose per-trace
            # flag is unset; the global depth suppresses grad recording on
            # every trace while any RF/PF probe runs.
            return
        if active_trace is not None and getattr(active_trace, "_tl_rf_probe_active", False):
            return
        # A managed backward directed at a FORK RELATIVE (a fork's
        # ``log_backward`` over the shared forward tensors) owns this
        # gradient: recording it here would silently mutate the hook trace's
        # projection with a pass the user directed at the fork. Fork
        # relatives share one label space, so the observation redirects to
        # the bracket-holding relative instead. Unrelated traces (e.g. two
        # composed models) keep the historical implicit-pass recording.
        managed_trace = _state._active_trace
        if (
            active_trace is not None
            and managed_trace is not None
            and managed_trace is not active_trace
            and getattr(managed_trace, "_tl_active_backward_bracket", False)
            and _is_fork_relative(active_trace, managed_trace)
        ):
            active_trace = managed_trace
        if active_trace is not None:
            # One-owner-per-label must hold on the FINAL emission target, not
            # just the hook's own trace: after a refresh-projection or fork
            # redirect, a stale source-trace hook (e.g. on an in-place live
            # tensor) can own the label in the SOURCE map while the target's
            # rebound map names a different tensor -- both passing their own
            # map would emit duplicate observations for one label. A target
            # map with no entry for the label stays permissive so redirected
            # gradients are not silently dropped.
            if active_trace is not trace_ref():
                target_owner_map = active_trace.__dict__.get("_tl_grad_hook_owner_by_label")
                if (
                    target_owner_map is not None
                    and target_owner_map.get(tensor_label, hooked_tensor_id) != hooked_tensor_id
                ):
                    return
            prebuilt = _emit_tensor_grad_event(active_trace, grad, tensor_label)
            # Gate the legacy layer-slot write on the ACTIVE per-call policy,
            # not the deprecated ``save_grads`` attribute: a per-call
            # ``log_backward(..., save_grads=False)`` sets the policy while
            # the attribute can stay truthy, and the attribute-keyed gate
            # kept retaining full grad payloads the caller disabled.
            # The event-sidecar payloads are handed down for REUSE: the
            # legacy layer-slot write used to mint a second independent
            # clone and run grad_transform a second time, uncharged by the
            # save budget (grind-r6 b5 R34-N1/R35-N1, fable+opus probe).
            if _active_save_grads_policy(active_trace) not in (None, False, "none", []):
                _log_tensor_grad(active_trace, grad, tensor_label, prebuilt=prebuilt)

    # TorchLens bookkeeping: torch's ``register_hook`` reads ``self.grad_fn``
    # internally, and ``t`` can be the user's registered state receiver (an
    # in-place op output), so the read is marked internal (r65 unread-bit).
    from .completeness_witness import internal_scalar_read

    with internal_scalar_read():
        t.register_hook(log_grad_to_model_history)


def _ensure_backward_event_stream(trace: "Trace") -> Any:
    """Return the mutable capture event bundle for backward sidecar emission.

    Raises
    ------
    BackwardStreamUnavailableError
        If the trace no longer owns a capture event stream. Fabricating a
        fresh empty buffer here would let post-hoc backward capture append
        into a container nothing reads and report success.
    """

    events = getattr(trace, "event_stream", None)
    if events is None:
        events = getattr(trace, "_capture_events", None)
    if events is None:
        events = getattr(trace, "capture_events", None)
    if events is not None:
        return events
    from ..._errors import BackwardStreamUnavailableError

    raise BackwardStreamUnavailableError(
        "This trace no longer owns a capture event stream, so backward "
        "capture cannot record events. The stream is released by "
        "trace.cleanup() and is not part of portable artifacts; capture a "
        "fresh trace before calling backward-capture APIs."
    )


def _forward_op_count_at_backward_trigger(trace: "Trace") -> int | None:
    """Return the number of forward ops created when a backward trigger starts.

    Parameters
    ----------
    trace:
        Trace being backward-captured.

    Returns
    -------
    int | None
        Active forward op count when available, otherwise the highest finalized
        layer ``step_index``. ``None`` means no structural count is available.
    """

    from ...capture import projections

    active_state = getattr(projections, "_active_recording_state", None)
    if active_state is not None and getattr(active_state, "runtime_trace", None) is trace:
        return max(0, int(active_state.step_index) - 1)

    step_indices = [
        int(step_index)
        for layer in getattr(trace, "layer_list", ())
        if isinstance((step_index := getattr(layer, "step_index", None)), int) and step_index > 0
    ]
    if step_indices:
        return max(step_indices)
    raw_graph_ws = trace.__dict__.get("_raw_graph_ws")
    raw_count = getattr(raw_graph_ws, "layer_counter", None)
    return raw_count if isinstance(raw_count, int) else None


def _ensure_backward_pass_for_tensor_hook(trace: "Trace") -> int:
    """Return an active backward pass index, opening an implicit pass if needed."""

    current_task_id = _current_backward_graph_task_id()
    pass_index = getattr(trace, "_active_backward_pass_index", None)
    if pass_index is not None:
        prior_task_id = _IMPLICIT_BACKWARD_TASK_IDS.get(trace)
        if (
            getattr(trace, "_implicit_backward_pass_open", False)
            and current_task_id is not None
            and prior_task_id is not None
            and current_task_id != prior_task_id
        ):
            from .backward import _close_implicit_backward_pass_if_open

            _IMPLICIT_BACKWARD_TASK_IDS.pop(trace, None)
            _close_implicit_backward_pass_if_open(trace)
        else:
            return int(pass_index)
    pass_index = int(getattr(trace, "num_backward_passes", 0)) + 1
    trace._active_backward_pass_index = pass_index
    trace._implicit_backward_pass_open = True
    if current_task_id is not None:
        _IMPLICIT_BACKWARD_TASK_IDS[trace] = current_task_id
        # Engine-drain close (L9 memo 1.2): opening inside a tensor hook is
        # provably in-backward, so queue the drain callback for THIS graph
        # task now. Opportunistic -- the sync-point backstop stays armed.
        from .backward import _enqueue_implicit_pass_drain_callback

        _enqueue_implicit_pass_drain_callback(trace, pass_index, current_task_id)
    if "implicit_backward_pass" not in trace._warned_once:
        warnings.warn(
            "TorchLens observed gradients outside a managed backward trigger; recording an "
            "implicit backward pass. Use trace.log_backward(), trace.backward(), or a TorchLens "
            "autograd trigger for precise pass boundaries.",
            RuntimeWarning,
            stacklevel=3,
        )
        trace._warned_once.add("implicit_backward_pass")
    events = _ensure_backward_event_stream(trace)
    events.append_backward(
        BackwardPassStart(
            pass_index=pass_index,
            trigger="implicit",
            implicit=True,
            outer_context=None,
            call_context_ref=None,
            root_meta=(),
            root_grad_arguments=None,
            inputs_subset=(),
            order=None,
            origin_backward_pass=None,
            save_grads_policy_repr=repr(_active_save_grads_policy(trace)),
            engine_flags=None,
            forward_op_count_at_trigger=_forward_op_count_at_backward_trigger(trace),
            timestamp=time.time(),
        )
    )
    return pass_index


def _current_backward_graph_task_id() -> int | None:
    """Return PyTorch's current autograd-engine invocation id when available.

    Returns
    -------
    int | None
        Engine graph-task id inside a backward hook, or ``None`` when the
        installed torch build exposes no such capability.
    """

    resolver = get_current_graph_task_id_fn()
    if resolver is None:
        return None
    try:
        task_id = resolver()
    except (AttributeError, RuntimeError):
        return None
    return int(task_id) if isinstance(task_id, int) and task_id >= 0 else None


def _emit_tensor_grad_event(
    trace: "Trace", grad: torch.Tensor, tensor_label: str
) -> tuple[torch.Tensor | None, Any | None, bool]:
    """Append an ``OpGradObserved`` event for a tensor hook firing.

    Returns
    -------
    tuple[torch.Tensor | None, Any | None, bool]
        ``(raw_payload, transformed_payload, built)`` — the payloads this
        event retained and whether the build actually ran past the policy
        gate. The legacy layer-slot writer REUSES these instead of minting a
        second uncharged clone and re-running ``grad_transform`` (grind-r6
        b5 R34-N1/R35-N1).
    """

    if getattr(trace, "_tl_backward_triggers_disarmed", False):
        return None, None, False
    stream_start = time.perf_counter()
    events = _ensure_backward_event_stream(trace)
    _record_phase_timing(
        trace,
        "backward_grad_event:ensure_stream",
        time.perf_counter() - stream_start,
    )
    pass_start = time.perf_counter()
    pass_index = _ensure_backward_pass_for_tensor_hook(trace)
    _record_phase_timing(
        trace,
        "backward_grad_event:ensure_pass",
        time.perf_counter() - pass_start,
    )
    payload_start = time.perf_counter()
    final_label = getattr(trace, "_raw_to_final_layer_labels", {}).get(tensor_label, tensor_label)
    with pause_logging():
        memory = int(grad.nelement() * grad.element_size())
        payload, transformed_payload, built = _build_grad_payloads(trace, grad, final_label)
    _record_phase_timing(
        trace,
        "backward_grad_event:payload",
        time.perf_counter() - payload_start,
    )
    append_start = time.perf_counter()
    events.append_backward(
        OpGradObserved(
            op_label=final_label,
            pass_index=pass_index,
            payload_ref=payload,
            transformed_payload_ref=transformed_payload,
            shape=tuple(grad.shape),
            dtype=str(grad.dtype),
            memory=memory,
            timestamp=time.time(),
        )
    )
    _record_phase_timing(
        trace,
        "backward_grad_event:append",
        time.perf_counter() - append_start,
    )
    return payload, transformed_payload, built


def _build_grad_payloads(
    trace: "Trace", grad: torch.Tensor, layer_label: str
) -> tuple[torch.Tensor | None, Any | None, bool]:
    """Return raw and transformed payloads for one observed op gradient.

    Parameters
    ----------
    trace:
        Trace owning the observed gradient.
    grad:
        Raw gradient tensor emitted by autograd.
    layer_label:
        Final operation label for the hook firing.

    Returns
    -------
    tuple[torch.Tensor | None, Any | None, bool]
        Raw payload, transformed payload, and whether the build ran past the
        policy gate (``built=True`` means the payload pair — including a
        legitimately-``None`` raw slot under ``save_raw_gradients=False`` —
        is THE charged retention for this grad+label and safe to reuse).
    """

    if not _should_save_grad_payload(trace, layer_label):
        return None, None, False
    if layer_label not in getattr(trace, "layer_dict_all_keys", {}):
        raw_payload, transformed_payload = _build_fastlog_grad_payloads(trace, grad)
        return raw_payload, transformed_payload, True
    op = trace.layer_dict_all_keys[layer_label]
    grad_transform = getattr(trace, "grad_transform", None)
    save_raw_gradients = getattr(trace, "save_raw_gradients", True)
    save_mode = _trace_grad_save_mode(trace)
    reservation = _admit_grad_payload_budget(trace, grad, layer_label, save_mode)
    raw_payload = (
        _copy_grad_payload(grad, save_mode=save_mode)
        if save_raw_gradients or grad_transform is None
        else None
    )
    if grad_transform is None:
        _commit_grad_payload_budget(trace, reservation, (raw_payload,))
        return raw_payload, None, True
    writer = getattr(trace, "_out_writer", None)
    transformed_payload = op._apply_transform(
        grad,
        grad_transform,
        transform_kind="grad",
        streaming_active=writer is not None,
    )
    op._validate_train_mode_transform_output(
        grad,
        transformed_payload,
        transform_kind="grad",
        transform=grad_transform,
    )
    op._validate_streaming_transform_output(
        transformed_payload,
        transform_kind="grad",
        streaming_active=writer is not None,
    )
    _commit_grad_payload_budget(trace, reservation, (raw_payload, transformed_payload))
    return raw_payload, transformed_payload, True


def _should_save_grad_payload(trace: "Trace", layer_label: str) -> bool:
    """Return whether a tensor-hook gradient should retain its payload."""

    policy = _active_save_grads_policy(trace)
    if policy in [None, False, "none", []]:
        return False
    if policy is True or policy == "all":
        return True
    if layer_label not in getattr(trace, "layer_dict_all_keys", {}):
        param_log = _param_log_for_exact_address(trace, layer_label)
        if param_log is not None:
            # Parameter gradients honor selector/callable policies through a
            # param-shaped context (grad_kind="param_grad") instead of being
            # silently dropped. Ordinal/label-string selections name OPS;
            # parameters are outside that vocabulary and stay unsaved there.
            if callable(policy) or isinstance(policy, BaseSelector):
                decision = policy(
                    _ParamGradPayloadContext(
                        param=param_log, pass_index=_current_backward_pass(trace)
                    )
                )
                return _grad_payload_decision_saves_out(decision)
            return False
        ctx = getattr(trace, "_fastlog_grad_contexts", {}).get(layer_label)
        if ctx is None:
            return False
        if callable(policy) or isinstance(policy, BaseSelector):
            decision = policy(
                _FastlogGradPayloadContext(ctx=ctx, pass_index=_current_backward_pass(trace))
            )
            return _grad_payload_decision_saves_out(decision)
        return False
    op = trace.layer_dict_all_keys[layer_label]
    if isinstance(policy, BaseSelector):
        decision = policy(_GradPayloadContext(op=op, pass_index=_current_backward_pass(trace)))
        return _grad_payload_decision_saves_out(decision)
    if callable(policy):
        decision = policy(_GradPayloadContext(op=op, pass_index=_current_backward_pass(trace)))
        return _grad_payload_decision_saves_out(decision)
    selection = getattr(trace, "_grad_op_nums_to_save", "all")
    if selection in [None, "none", []]:
        return False
    if selection == "all":
        return True
    return op.raw_index in selection


def _build_fastlog_grad_payloads(
    trace: "Trace",
    grad: torch.Tensor,
) -> tuple[torch.Tensor | None, Any | None]:
    """Return gradient payloads for predicate-mode recording contexts."""

    grad_transform = getattr(trace, "grad_transform", None)
    save_raw_gradients = getattr(trace, "save_raw_gradients", True)
    save_mode = _trace_grad_save_mode(trace)
    reservation = _admit_grad_payload_budget(trace, grad, "<fastlog grad>", save_mode)
    raw_payload = (
        _copy_grad_payload(grad, save_mode=save_mode)
        if save_raw_gradients or grad_transform is None
        else None
    )
    if grad_transform is None:
        _commit_grad_payload_budget(trace, reservation, (raw_payload,))
        return raw_payload, None
    transformed_payload = grad_transform(grad)
    if not isinstance(transformed_payload, torch.Tensor):
        raise TypeError("grad_transform must return a torch.Tensor for fastlog gradients")
    _commit_grad_payload_budget(trace, reservation, (raw_payload, transformed_payload))
    return raw_payload, transformed_payload


def _admit_grad_payload_budget(
    trace: "Trace", grad: torch.Tensor, label: str, save_mode: SaveMode
) -> Any:
    """Pre-admit one retained gradient payload against the save budget.

    Gradient payloads are RAM-retained copies exactly like forward primary
    payloads, so they charge the same per-device accountant; skipping them
    would falsify the budget's committed-footprint claim after any backward.

    Parameters
    ----------
    trace:
        Trace carrying the optional ``_save_budget_accountant``.
    grad:
        Observed gradient tensor whose copy would be retained.
    label:
        Operation label named in a refusal.
    save_mode:
        Active gradient save mode, used to project the retention device.

    Returns
    -------
    Any
        Opaque reservation reconciled after allocation, or ``None``.
    """

    budget = getattr(trace, "_save_budget_accountant", None)
    if budget is None:
        return None
    target_device = torch.device("cpu") if save_mode == "cpu_async" else grad.device
    num_bytes = int(grad.nelement() * grad.element_size())
    return budget.admit(str(label), target_device, num_bytes)


def _commit_grad_payload_budget(
    trace: "Trace", reservation: Any, payloads: tuple[Any, ...]
) -> None:
    """Reconcile a gradient-payload admission against retained storage."""

    budget = getattr(trace, "_save_budget_accountant", None)
    if budget is None or reservation is None:
        return
    budget.commit(reservation, payloads)


def _trace_grad_save_mode(trace: "Trace") -> SaveMode:
    """Return the active save mode for trace-side gradient payloads."""

    policy = _active_save_grads_policy(trace)
    if isinstance(policy, CaptureSpec):
        return policy.save_mode
    return cast(SaveMode, getattr(trace, "save_mode", "copy"))


def _copy_grad_payload(grad: torch.Tensor, *, save_mode: SaveMode = "copy") -> torch.Tensor:
    """Return a detached gradient payload snapshot through the copy chokepoint.

    Parameters
    ----------
    grad:
        Gradient tensor observed by an autograd hook.

    Returns
    -------
    torch.Tensor
        Detached tensor copy suitable for storage in gradient records.
    """

    # Gradient payloads are ALWAYS genuine snapshots, even under
    # save_mode="reference"/"view": autograd's AccumulateGrad may steal the
    # observed gradient as the leaf's ``.grad`` and accumulate into it IN
    # PLACE on the next backward, so an aliased payload silently rewrites the
    # recorded pass-N value (pass-1 record becomes the running sum). Unlike
    # the forward path, no wrapped in-place op exists here to stamp and
    # fail-close the mutation, so aliasing cannot be disclosed -- clone.
    if save_mode in ("reference", "view"):
        save_mode = "copy"
    copied = safe_copy(grad, detach_tensor=True, save_mode=save_mode)
    if not isinstance(copied, torch.Tensor):
        raise TypeError("safe_copy returned a non-tensor gradient payload")
    return copied


def _grad_payload_decision_saves_out(decision: Any) -> bool:
    """Return whether a gradient predicate decision retains tensor payload."""

    if isinstance(decision, CaptureSpec):
        return decision.save_out
    return bool(decision)


class _GradPayloadContext:
    """Minimal predicate context for trace-side op gradient retention."""

    def __init__(self, *, op: Op, pass_index: int | None) -> None:
        """Initialize a trace gradient predicate context.

        Parameters
        ----------
        op:
            Operation whose output gradient was observed.
        pass_index:
            One-based backward pass number, when known.
        """

        self.label = op.label
        self.layer_label = op.layer_label
        self.op_label = op.label
        self.raw_label = getattr(op, "raw_tensor_label", None)
        self.func_name = op.func_name
        self.layer_type = op.layer_type
        self.type = op.layer_type
        self.module_stack = tuple(getattr(op, "module_stack", ()) or ())
        self.modules = tuple(getattr(op, "modules", ()) or ())
        self.output_of_module_calls = tuple(getattr(op, "output_of_module_calls", ()) or ())
        self.has_forward_op = True
        self.has_op = True
        self.grad_kind = "grad_output"
        self.pass_index = pass_index
        self.backward_pass_index = pass_index
        self.shape = op.shape
        self.dtype = op.dtype
        self.tensor_device = getattr(op, "output_device", None)


def _param_log_for_exact_address(trace: "Trace", address: str) -> Any | None:
    """Return the Param record for an exact address, or None.

    Membership is checked against the exact address mapping, never the
    accessor's fuzzy short-name/substring resolution, so an op label can
    never accidentally resolve to a parameter.
    """

    param_logs = getattr(trace, "param_logs", None)
    if param_logs is None:
        return None
    exact = getattr(param_logs, "_dict", None)
    if exact is not None:
        return exact.get(address)
    if isinstance(param_logs, Mapping):
        return param_logs.get(address)
    return None


class _ParamGradPayloadContext:
    """Minimal predicate context for parameter gradient retention."""

    def __init__(self, *, param: Any, pass_index: int | None) -> None:
        """Initialize a parameter gradient predicate context.

        Parameters
        ----------
        param:
            Param record whose gradient was observed.
        pass_index:
            One-based backward pass number, when known.
        """

        self.label = param.address
        self.layer_label = param.address
        self.op_label = param.address
        self.raw_label = None
        self.param_address = param.address
        self.param_name = param.name
        self.func_name = None
        self.layer_type = "param"
        self.type = "param"
        self.module_stack = ()
        module_address = getattr(param, "module_address", None)
        self.modules = (module_address,) if module_address else ()
        self.output_of_module_calls = ()
        self.has_forward_op = False
        self.has_op = False
        self.grad_kind = "param_grad"
        self.pass_index = pass_index
        self.backward_pass_index = pass_index
        self.shape = param.shape
        self.dtype = param.dtype
        self.tensor_device = None


class _FastlogGradPayloadContext:
    """Minimal predicate context for fastlog op gradient retention."""

    def __init__(self, *, ctx: Any, pass_index: int | None) -> None:
        """Initialize a fastlog gradient predicate context."""

        self.label = ctx.label
        self.layer_label = _public_fastlog_label(ctx)
        self.op_label = self.layer_label
        self.raw_label = ctx.raw_label
        self.func_name = ctx.func_name
        self.layer_type = ctx.layer_type
        self.type = ctx.layer_type
        self.module_stack = tuple(getattr(ctx, "module_stack", ()) or ())
        self.modules = tuple(frame.address for frame in self.module_stack)
        self.output_of_module_calls = self.modules
        self.has_forward_op = True
        self.has_op = True
        self.grad_kind = "grad_output"
        self.pass_index = pass_index
        self.backward_pass_index = pass_index
        self.shape = ctx.shape
        self.dtype = ctx.dtype
        self.tensor_device = ctx.tensor_device


def _public_fastlog_label(ctx: Any) -> str:
    """Return the compact fastlog op label for a record context."""

    if ctx.kind == "op" and ctx.layer_type is not None and ctx.type_index is not None:
        return f"{ctx.layer_type}_{ctx.type_index}"
    return ctx.label


def _current_backward_pass(trace: "Trace") -> int | None:
    """Return the currently active backward pass index, if any."""

    pass_index = getattr(trace, "_active_backward_pass_index", None)
    return None if pass_index is None else int(pass_index)


def _active_save_grads_policy(trace: "Trace") -> Any:
    """Return the current gradient retention policy for tensor hooks."""

    if hasattr(trace, "_active_save_grads_policy"):
        return getattr(trace, "_active_save_grads_policy")
    return getattr(trace, "save_grads", None)


def _log_tensor_grad(
    self: "Trace",
    grad: torch.Tensor,
    _label_raw: str,
    prebuilt: tuple[torch.Tensor | None, Any | None, bool] | None = None,
) -> None:
    """Callback invoked during backward pass to save a tensor's grad.

    Resolves the raw label to a final label, then saves the grad on the
    layer entry.  If the layer is an output parent, its output-layer children
    also receive the grad (since output layers are identity wrappers that
    share the same grad).

    Args:
        grad: The grad tensor from autograd.
        _label_raw: Raw tensor label used to look up the final label.
        prebuilt: The event-sidecar ``(raw_payload, transformed_payload,
            built)`` triple from ``_emit_tensor_grad_event``. When built, the
            layer slots REUSE those exact (budget-charged) objects instead of
            minting a second uncharged clone and running ``grad_transform``
            a second time (grind-r6 b5 R34-N1/R35-N1); output-layer children
            share the parent's payloads, matching their identity-wrapper
            contract. When the event build did not run (narrower event
            selection, disarmed sidecar), the slot's payloads are built and
            charged per layer through the same chokepoint.
    """
    self.has_gradients = True
    if _label_raw not in self._raw_to_final_layer_labels:
        return
    if not self.layer_dict_all_keys:
        # A fastlog/raw runtime trace never builds the legacy layer surface
        # (postprocess is skipped), so the layer-slot write is structurally
        # inapplicable there -- its gradients ride the backward event stream.
        # The per-call save_grads policy gate (b9937876) armed this write on
        # recording backward, where every lookup would KeyError. Full traces
        # keep the populated dict, so a genuinely missing individual key
        # still raises below (mapping/dict divergence stays a tripwire).
        return
    tensor_label = self._raw_to_final_layer_labels[_label_raw]
    layer_log_entry = self.layer_dict_all_keys[tensor_label]
    layers_to_update = [tensor_label]
    # Output layers are identity wrappers; propagate grad to them too.
    if layer_log_entry.is_output_parent:
        for child_layer in layer_log_entry.children:
            if self.layer_dict_all_keys[child_layer].is_output:
                layers_to_update.append(child_layer)

    for layer_label in layers_to_update:
        layer = self.layer_dict_all_keys[layer_label]
        selection = getattr(self, "_grad_op_nums_to_save", "all")
        if selection != "all":
            if selection in [None, "none", []] or layer.raw_index not in selection:
                continue
        if layer_label not in self._saved_grad_labels:
            self._saved_grad_labels.add(layer_label)
        payload_source = (
            prebuilt
            if prebuilt is not None and prebuilt[2]
            else _build_grad_payloads(self, grad, layer_label)
        )
        if payload_source[2]:
            layer.log_tensor_grad(grad, prebuilt=(payload_source[0], payload_source[1]))
        else:
            # Retention policy denies a payload for this label through the
            # charged chokepoint; keep the historical bare slot write.
            layer.log_tensor_grad(grad)
        self.saved_gradient_memory += layer.gradient_memory
        self.total_gradient_memory += layer.gradient_memory


def _locate_parent_tensors_in_args(
    self: "Trace",
    parent_log_entries: list[Op],
    args: tuple[Any, ...],
    kwargs: dict[Any, Any],
) -> dict[str, dict[Any, str]]:
    """Map each parent tensor to its position in the function's args/kwargs.

    Supports up to 2 levels of nesting:
      - Top-level: ``args[i]`` maps to key ``i``
      - Nested: ``args[i][j]`` maps to key ``(i, j)``
    Deeper nesting is not tracked (would require recursive search).

    This mapping is stored as ``parent_arg_positions`` on the child's log
    entry, and is used by:
      - ``_get_parent_contents``: to retrieve pre-call parent values from arg_copies
      - Validation replay: to reconstruct the function call

    Returns:
        ``{"args": {pos: label, ...}, "kwargs": {key: label, ...}}``
    """
    tensor_all_arg_positions: dict[str, dict[Any, str]] = {"args": {}, "kwargs": {}}
    if not parent_log_entries:
        return tensor_all_arg_positions

    positions_by_label: dict[str, dict[str, list[Any]]] = {
        parent_entry._label_raw: {"args": [], "kwargs": []} for parent_entry in parent_log_entries
    }

    for arg_type, arg_struct in (("args", args), ("kwargs", kwargs)):
        for arg_key, arg in _iter_arg_container_items(arg_type, arg_struct):
            arg_label = (
                mutated_parameter_label(arg)
                if isinstance(arg, torch.nn.Parameter)
                else get_tensor_label(arg)
            )
            if arg_label in positions_by_label:
                positions_by_label[arg_label][arg_type].append(arg_key)

            if not _is_supported_parent_arg_container(arg):
                continue
            # Second level of nesting (e.g., torch.cat([tensor_a, tensor_b])).
            for sub_arg_key, sub_arg in _iter_arg_container_items(arg, arg):
                sub_arg_label = (
                    mutated_parameter_label(sub_arg)
                    if isinstance(sub_arg, torch.nn.Parameter)
                    else get_tensor_label(sub_arg)
                )
                # The former parent-first scan stopped at a top-level match for that
                # parent, while still inspecting the container for every other parent.
                if sub_arg_label in positions_by_label and sub_arg_label != arg_label:
                    positions_by_label[sub_arg_label][arg_type].append((arg_key, sub_arg_key))

    # Preserve the historical insertion order: positions were emitted parent by
    # parent, even though finding them required rescanning every argument each time.
    for parent_entry in parent_log_entries:
        parent_label = parent_entry._label_raw
        for arg_type in ("args", "kwargs"):
            for position in positions_by_label[parent_label][arg_type]:
                tensor_all_arg_positions[arg_type][position] = parent_label

    return tensor_all_arg_positions


def _is_supported_parent_arg_container(value: object) -> bool:
    """Return whether ``value`` participates in parent arg-position mapping.

    Parameters
    ----------
    value:
        Candidate nested argument value.

    Returns
    -------
    bool
        True when ``value`` is a supported list/tuple or mapping container.
    """

    return isinstance(value, (Mapping, tuple, list))


def _iter_arg_container_items(
    container_kind: str | object,
    container: object,
) -> Iterable[tuple[Any, Any]]:
    """Yield items from an args/kwargs structure or nested supported container.

    Parameters
    ----------
    container_kind:
        Either ``"args"``, ``"kwargs"``, or the nested container object itself.
    container:
        Args tuple, kwargs mapping, or a nested supported container.

    Returns
    -------
    Iterable[tuple[Any, Any]]
        Position keys paired with container values.

    Raises
    ------
    TypeError
        If ``container`` is not a supported args/kwargs or nested container shape.
    """

    if container_kind == "args":
        return enumerate(cast(tuple[Any, ...], container))
    if container_kind == "kwargs":
        return cast(dict[Any, Any], container).items()
    if isinstance(container, Mapping):
        return container.items()
    if isinstance(container, (list, tuple)):
        return enumerate(container)
    raise TypeError(f"Unsupported parent-arg container: {type(container)!r}")


def _get_ancestors_from_parents(
    parent_entries: list[Op],
) -> tuple[set[str], set[str]]:
    """Utility function to get the ancestors of a tensor based on those of its parent tensors.

    Args:
        parent_entries: list of parent entries

    Returns:
        List of input ancestors and internally initialized ancestors.
    """
    input_ancestors = set()
    internal_source_ancestors = set()

    for parent_entry in parent_entries:
        input_ancestors.update(parent_entry.input_ancestors)
        internal_source_ancestors.update(parent_entry.internal_source_ancestors)
    return input_ancestors, internal_source_ancestors


def _process_parent_param_ops(
    arg_parameters: list[torch.nn.Parameter],
) -> dict[str, int]:
    """Assign persistent barcodes to parameters and track their pass number.

    On first encounter, each parameter gets a random barcode
    and pass number 1.  On subsequent encounters (same parameter used again in
    a later loop iteration), the pass number is incremented.

    The barcode is a random string that uniquely identifies the parameter tensor
    across the entire logging session.  It's used (together with layer_type) to
    build the ``equivalence_class`` for parameterized operations, which
    is how loop detection recognizes "conv2d with weights W" as the same layer
    on pass 1 and pass 2.

    Args:
        arg_parameters: Parameter tensors found in the function's arguments.

    Returns:
        Dict mapping each parameter's barcode to its current pass number.
    """
    parent_param_ops = {}
    for param in arg_parameters:
        meta = get_param_meta(param)
        if meta is None or meta.param_barcode is None:
            # First time seeing this parameter — assign a unique barcode.
            param_barcode = make_random_barcode()
            set_param_meta(
                param,
                barcode=param_barcode,
                address="",
                requires_grad_before=param.requires_grad,
            )
        else:
            param_barcode = meta.param_barcode
        call_index = increment_param_call_index(param)
        parent_param_ops[param_barcode] = call_index
    return parent_param_ops


def _make_raw_param_group_barcode(
    indiv_param_barcodes: list[str],
    layer_type: str,
    *,
    output_index: int | None = None,
) -> str:
    """Build an equivalence_class string for a parameterized operation.

    Combines the layer type with sorted parameter barcodes to produce a
    canonical fingerprint.  Sorting ensures order-independence (e.g., weight
    and bias can appear in either order).

    The layer_type prefix is critical: different operations using the same
    parameters (e.g., ``isinf(weight)`` vs ``expand(weight)``) must NOT be
    grouped as the same layer.

    Example: ``"conv2d_abc123_def456"``

    Parameters
    ----------
    indiv_param_barcodes:
        Barcodes for each parameter tensor.
    layer_type:
        The normalized operation name.
    output_index:
        Zero-based output index for iterable or structured multi-output
        operations, or ``None`` for single-output operations.

    Returns
    -------
    str
        Canonical fingerprint string for this parameterized operation.
    """
    param_group_barcode = f"{layer_type}_{'_'.join(sorted(indiv_param_barcodes))}"
    if output_index is not None:
        param_group_barcode += f"_outindex{output_index}"
    return param_group_barcode


def _append_module_suffix_to_equivalence_class(
    equivalence_class: str,
    modules: list[tuple[str, int]] | tuple[tuple[str, int], ...],
) -> str:
    """Append the canonical module-stack suffix used by loop detection.

    Parameters
    ----------
    equivalence_class:
        Base operation equivalence string.
    modules:
        Module stack snapshot for the captured op, as ``(address, pass)``
        pairs ordered outer-to-inner.

    Returns
    -------
    str
        ``equivalence_class`` plus the historical module-path suffix. Empty
        module stacks append an empty suffix.
    """

    return equivalence_class + "_".join(module_pass[0] for module_pass in modules)


def _get_equivalence_class(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    i: int,
    layer_type: str,
    fields_dict: dict[str, Any],
) -> str:
    """Build an equivalence_class string for a NON-parameterized operation.

    For ops that don't use parameters (e.g., ``relu``, ``cat``, ``add``), the
    fingerprint is built from:
      1. ``layer_type``: the normalized function name.
      2. ``arg_hash``: hash of non-tensor arguments (shapes, scalar values, etc.).
      3. ``outindex`` suffix: disambiguates outputs of multi-output functions.
      4. ``module`` suffix: disambiguates identical ops in different submodules.

    This fingerprint is used by loop detection.  Two operations with the same
    fingerprint are candidates for being "the same layer on different ops."

    Note: non-parameterized ops default to ``call_index=1`` because without
    parameters to track reuse, there's no reliable way to count ops.

    Args:
        args: Positional arguments to the function call.
        kwargs: Keyword arguments to the function call.
        i: Index of this output tensor within a multi-output call.
        layer_type: The normalized operation name.
        fields_dict: Must contain ``in_multi_output`` and
            ``module``.

    Returns:
        A string key identifying this operation's equivalence class.
    """
    arg_hash = _get_hash_from_args(args, kwargs)
    equivalence_class = f"{layer_type}_{arg_hash}"
    if fields_dict["in_multi_output"]:
        equivalence_class += f"_outindex{i}"
    if fields_dict["module"] is not None:
        module_str = fields_dict["module"][0]
        equivalence_class += f"_module{module_str}"
    return equivalence_class


def _get_hash_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Compute a structural hash of non-tensor arguments for equivalence fingerprinting.

    Tensor arguments are excluded (they define graph edges, not structural identity).
    Parameters are also excluded (they have their own barcode system).
    The hash preserves positional indices and kwarg keys to avoid collisions
    between e.g. ``f(a=1, b=2)`` and ``f(a=2, b=1)``.

    Returns:
        A short deterministic hash string, or ``"no_args"`` if no non-tensor
        arguments are present.
    """
    args_to_hash: list[Any] = []
    for a, arg in enumerate(args):
        _append_arg_hash(arg, f"pos{a}", args_to_hash)
    for key, arg in kwargs.items():
        _append_arg_hash(arg, f"kw_{key}", args_to_hash)

    if len(args_to_hash) == 0:
        return "no_args"
    return make_short_barcode_from_input(args_to_hash)


def _tensor_hash_token(prefix: str, arg: torch.Tensor) -> str:
    """Return the barcode hash token for one tensor argument.

    Uses shape/dtype only — formatting a tensor can trigger wrapped methods
    (item, __format__) which re-enter logging and cause infinite recursion.
    F20 lazy-buffer completion: a pending UninitializedBuffer is a plain
    Tensor subclass (unlike UninitializedParameter, which rides the
    Parameter carve-out at the caller) and raises on any shape read until
    torch's lazy pre-hook materializes it, so it hashes as a fixed token.
    """

    if _is_uninitialized_param(arg):
        return f"{prefix}_tensor_uninitialized"
    return f"{prefix}_tensor{arg.shape}"


def _append_arg_hash(arg: Any, prefix: str, args_to_hash: list[Any], _depth: int = 0) -> None:
    """Append structural fingerprint tokens for a single argument to the accumulator list.

    Builds an ``equivalence_class`` -- a structural fingerprint of the operation's
    argument types and shapes (not a content hash). This fingerprint is used by loop
    detection to identify operations that are structurally identical across ops.

    For tensors, only shape and dtype are recorded (not values). Containers (dicts, lists,
    tuples, sets) are recursed into with depth-limited traversal. Parameters are excluded.

    Args:
        arg: The argument value to fingerprint.
        prefix: String prefix encoding the argument's position/key path.
        args_to_hash: Accumulator list that fingerprint tokens are appended to.
        _depth: Recursion depth guard; stops at 10 to prevent infinite recursion.
    """
    if _depth > 10:
        args_to_hash.append(f"{prefix}_deep")
        return
    if isinstance(arg, torch.nn.Parameter):
        pass  # exclude parameters from hash — must check before Tensor (Parameter is a subclass)
    elif isinstance(arg, torch.Tensor):
        args_to_hash.append(_tensor_hash_token(prefix, arg))
    elif isinstance(arg, (torch.TypedStorage, torch.UntypedStorage)):
        # Same hazard as torch.Tensor above: str()/repr() on a Storage walks
        # every element via wrapped __getitem__ (and even constructs a fresh
        # wrapped tensor per element on some torch builds), re-entering
        # logging and causing infinite/runaway recursion. Use size/dtype only.
        dtype = getattr(arg, "dtype", None)
        args_to_hash.append(f"{prefix}_storage{arg.size()}_{dtype}")
    elif isinstance(arg, dict):
        for k, v in arg.items():
            _append_arg_hash(v, f"{prefix}_dk{k}", args_to_hash, _depth + 1)
    elif isinstance(arg, (list, tuple)):
        for i, elem in enumerate(arg):
            _append_arg_hash(elem, f"{prefix}_i{i}", args_to_hash, _depth + 1)
    elif isinstance(arg, (set, frozenset)):
        _append_set_arg_hash(arg, prefix, args_to_hash, _depth)
    else:
        args_to_hash.append(_leaf_arg_token(arg, prefix))


def _leaf_arg_token(arg: Any, prefix: str) -> str:
    """Return the fingerprint token for one non-container leaf argument.

    grind-r5 b7 R21-C (P4): the default object repr IS the memory address,
    so ``str(arg)`` here persisted a process-local address into
    ``equivalence_class`` -- cross-process keys diverged for any op holding
    an object arg (``torch.randn(..., generator=g)``), the SAME key could
    collide for two distinct objects after address reuse, and recurrence
    grouping split when a semantically-identical fresh object was passed per
    call. Same rule as ``_capture_fingerprint``: never repr an address; use
    the address-free type token.
    """

    arg_type = type(arg)
    # mypy sees bound-descriptor types diverge here; the identity comparison
    # against the object slots is exactly the intended check.
    default_repr = arg_type.__repr__ is object.__repr__  # type: ignore[comparison-overlap]
    default_str = arg_type.__str__ is object.__str__  # type: ignore[comparison-overlap]
    if default_repr and default_str:
        return f"{prefix}_obj:{arg_type.__module__}.{arg_type.__qualname__}"
    token = f"{prefix}_{arg}"
    if " at 0x" in token:
        # Custom reprs that still embed a memory address (C types,
        # functools.partial interiors) are equally id()-derived.
        return f"{prefix}_obj:{arg_type.__module__}.{arg_type.__qualname__}"
    return token


def _append_set_arg_hash(
    arg: "set[Any] | frozenset[Any]", prefix: str, args_to_hash: list[Any], _depth: int
) -> None:
    """Fingerprint one set/frozenset arg in seed-independent member order.

    grind-r5 b7 R21/P4 rider: sets iterate in hash order, so the old
    position-encoded ``_i{i}`` prefixes made the persisted fingerprint
    PYTHONHASHSEED-dependent (and frozenset fell to the str() tail).
    Fingerprint each member independently, then fold the members in sorted
    token order -- deterministic for any seed and identical for
    set/frozenset of equal members.
    """

    member_tokens: list[tuple[str, ...]] = []
    for elem in arg:
        elem_tokens: list[Any] = []
        _append_arg_hash(elem, "", elem_tokens, _depth + 1)
        member_tokens.append(tuple(str(token) for token in elem_tokens))
    member_tokens.sort()
    for i, tokens in enumerate(member_tokens):
        args_to_hash.extend(f"{prefix}_s{i}{token}" for token in tokens)
