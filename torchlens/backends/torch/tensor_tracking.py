"""Track torch tensor provenance, family links, and equivalence classes.

This module owns tensor backward hooks, parent argument positions, parameter pass
tracking, and structural fingerprints used by loop detection.
"""

import time
import warnings
import weakref
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from ... import _state
from ..._data_classes._trace_grad_sites import apply_gradient_site_edits
from ..._data_classes.op import Op
from ..._ir.events import BackwardPassStart, OpGradObserved
from ..._selection.selectors import BaseSelector
from ..._state import pause_logging
from ..._utils._record_group_table import record_member
from ..._utils._tensor_label import (
    get_param_meta,
    get_tensor_label,
    increment_param_call_index,
    mutated_parameter_label,
    set_param_meta,
)
from ..._utils.display import _record_phase_timing
from ..._utils.hashing import make_random_barcode
from ..._utils.tensor_utils import SaveMode, safe_copy
from ...recording._types import CaptureSpec
from ._arg_hash import (
    _get_hash_from_args as _get_hash_from_args,  # re-exported
    observe_arg_inputs,
)
from ._backward_markers import _current_backward_graph_task_id
from ._completeness_origins import internal_scalar_read
from ._equivalence_resolve import resolve_equivalence_class
from ._graph_pins import _STATE_BOUNDARY_GRAD_FNS
from ._ops_arg_slots import current_arg_slots, parent_arg_positions_from_slots
from ._parent_arg_containers import _is_supported_parent_arg_container, _iter_arg_container_items
from ._tensor_grad_hook import (
    _TENSOR,
    _foreign_label_owner,
    _fork_redirect_target,
    _logged_copy_source_slot,
    add_plain_tensor_hook,
    raw_grad_fn,
    raw_requires_grad,
    register_tensor_grad_hook,
)

if TYPE_CHECKING:
    from ..._capture.session import LiveGradientReference
    from ..._data_classes.trace import Trace


_IMPLICIT_BACKWARD_TASK_IDS: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
# Removable handles of gradient hooks that sit on model state (a mutated prepared
# Parameter's autograd history), keyed weakly by the owning trace. A key marks the
# trace STATE-ENTANGLED: later forwards on the model chain onto that history, so
# every later backward reaches this trace's hooks. cleanup() removes the handles
# but keeps the key (its remaining op hooks stay gated); keys die with the trace.
_OWNED_STATE_GRAD_HOOK_HANDLES: weakref.WeakKeyDictionary[Any, list[Any]] = (
    weakref.WeakKeyDictionary()
)


def _register_state_boundary_grad_fn(trace: Any, grad_fn_handle: Any) -> None:
    """Register a model-state grad-fn node as a root-matching boundary of ``trace``.

    Parameters
    ----------
    trace:
        Trace that hooked the node.
    grad_fn_handle:
        Capture-pass autograd node of a mutated prepared Parameter. It is pinned
        for the trace's lifetime so its id cannot be reused by another node.
    """

    from .backward import _backward_registry_slot, _strong_grad_fn_refs

    _strong_grad_fn_refs(trace).append(grad_fn_handle)
    trace_ref, owned_ids = _backward_registry_slot(trace)
    owned_ids.add(id(grad_fn_handle))
    _STATE_BOUNDARY_GRAD_FNS[id(grad_fn_handle)] = trace_ref


def _defer_at_state_boundary(
    grad_fn_handle: Any, matched_ids: set[int], deferred: dict[int, list[Any]]
) -> bool:
    """Hold a root walk at a model-state node whose owner has not matched yet.

    A node reached through the model's history from outside its owner's own graph
    must not make the owner match (every later forward chains onto it). The walk
    parks it under the owner's id and resumes past it only once the owner matches
    through its own nodes.

    Parameters
    ----------
    grad_fn_handle:
        Node the walk just reached.
    matched_ids:
        Ids of the traces matched so far.
    deferred:
        Parked boundary nodes by owner id (updated in place).

    Returns
    -------
    bool
        Whether the walk must not descend past the node now.
    """

    trace_ref = _STATE_BOUNDARY_GRAD_FNS.get(id(grad_fn_handle))
    owner = None if trace_ref is None else trace_ref()
    if owner is None or id(owner) in matched_ids:
        return False
    deferred.setdefault(id(owner), []).append(grad_fn_handle)
    return True


def _resume_state_boundaries(
    trace: Any, deferred: dict[int, list[Any]], seen: set[int]
) -> list[Any]:
    """Return the boundary nodes parked for ``trace`` once it matched, unseen again.

    Parameters
    ----------
    trace:
        Trace that just matched through its own nodes.
    deferred:
        Parked boundary nodes by owner id (the trace's entry is removed).
    seen:
        Ids the walk visited (the resumed nodes are removed so they are walked).

    Returns
    -------
    list[Any]
        Nodes to put back on the walk queue.
    """

    resumed = deferred.pop(id(trace), [])
    seen.difference_update(id(node) for node in resumed)
    return resumed


def _is_state_entangled(trace: Any) -> bool:
    """Return whether ``trace`` registered a gradient hook on model state."""

    return bool(_OWNED_STATE_GRAD_HOOK_HANDLES) and (
        trace is not None and trace in _OWNED_STATE_GRAD_HOOK_HANDLES
    )


def _owning_backward_is_running(*candidates: Any) -> bool:
    """Return whether one of ``candidates`` holds its managed backward bracket now.

    A state-entangled trace records only its own managed passes (``backward``,
    ``trace.backward()``, or a plain ``.backward()`` whose roots match the trace):
    any other backward that reaches its hooks came in through the model's
    autograd history and is not this trace's.

    Parameters
    ----------
    *candidates:
        Traces to check (``None`` entries are skipped).

    Returns
    -------
    bool
        Whether a candidate's managed backward is running.
    """

    return any(
        candidate is not None and getattr(candidate, "_tl_active_backward_bracket", False)
        for candidate in candidates
    )


def remove_owned_state_grad_hooks(trace: Any) -> None:
    """Remove the model-state gradient hooks ``trace`` registered.

    Parameters
    ----------
    trace:
        Trace being cleaned up.
    """

    handles = _OWNED_STATE_GRAD_HOOK_HANDLES.get(trace)
    if handles is None:
        return
    for handle in handles:
        handle.remove()
    handles.clear()


def _add_tensor_backward_hook(
    trace: "Trace",
    t: torch.Tensor,
    tensor_label: str,
    *,
    take_ownership: bool = False,
    owning_backward_only: bool = False,
) -> None:
    """Register a backward hook on ``t`` that captures its grad into Trace.

    The hook is a small function holding a ``weakref`` to Trace and the
    ``tensor_label`` (never the tensor), so it keeps neither alive.

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
        owning_backward_only: The hook sits on MODEL STATE (a mutated
            prepared Parameter's autograd history), which later forwards on
            the same model chain onto, so every later backward reaches it and
            the op hooks upstream of it. The trace becomes state-entangled:
            its hooks record only inside its own managed backward
            (``_owning_backward_is_running``), the node is a root-matching
            boundary rather than a trigger (``_register_state_boundary_grad_fn``),
            and ``cleanup()`` removes this hook. Plain traces keep the
            historical implicit-pass recording.
    """
    # TorchLens's OWN bookkeeping reads: a plain tensor reads the C getsets (no
    # TorchLens frame for the state observer to see); a subclass, whose read may run
    # its ``__torch_function__``, keeps the read marked internal.
    if type(t) is _TENSOR:
        _grad_fn = raw_grad_fn(t)
        _requires_grad = bool(raw_requires_grad(t))
    else:
        with internal_scalar_read():
            _grad_fn = t.grad_fn
            _requires_grad = bool(t.requires_grad)
    if _grad_fn is not None and owning_backward_only:
        _register_state_boundary_grad_fn(trace, _grad_fn)
    elif _grad_fn is not None:
        _backward_module._register_forward_grad_fn(trace, _grad_fn, tensor_label)
    elif not _requires_grad:
        # Outside autograd: never hooked, whatever the policy reads below say.
        return
    # Deferred gradient selectors (all selective grad selections, including
    # positive integer ordinals — FINAL layer numbers unknowable during the
    # forward) install their hooks post-postprocess from the reference escrow.
    should_defer_hook = trace.__dict__.get(
        "_deferred_gradient_selector"
    ) is not None and not getattr(trace, "_installing_deferred_gradient_hooks", False)
    if not getattr(trace, "capture_tensor_grad_hooks", True):
        return
    if should_defer_hook:
        # An in-place op's live result is the only tensor its gradient flows
        # through (the logged copy is a dead end), and nothing else holds it
        # at its current version for the deferred install: escrow it now.
        if take_ownership and _grad_fn is not None:
            _escrow_live_gradient_reference(trace, t, tensor_label, _grad_fn, owning_backward_only)
        return
    _install_tensor_grad_hook(
        trace,
        t,
        tensor_label,
        take_ownership=take_ownership,
        owning_backward_only=owning_backward_only,
    )


def _escrow_live_gradient_reference(
    trace: "Trace",
    t: torch.Tensor,
    tensor_label: str,
    grad_fn: Any,
    owning_backward_only: bool,
) -> None:
    """Keep an in-place op's live result for the deferred gradient install.

    Parameters
    ----------
    trace:
        Trace being captured under a deferred gradient selector.
    t:
        Live tensor taking gradient ownership of ``tensor_label``.
    tensor_label:
        Raw label of the in-place op.
    grad_fn:
        ``t.grad_fn`` right after the op.
    owning_backward_only:
        Whether the hook sits on model state (see ``_add_tensor_backward_hook``).
    """

    from ..._capture.session import LiveGradientReference, capture_session_for

    session = capture_session_for(trace)
    if session is None:
        return
    with pause_logging(), internal_scalar_read():
        output_nr = int(t.output_nr)
    session.escrow_live_gradient_reference(
        tensor_label,
        LiveGradientReference(
            tensor=t,
            grad_fn=grad_fn,
            output_nr=output_nr,
            owning_backward_only=owning_backward_only,
        ),
    )


def _add_live_gradient_reference_hook(
    trace: "Trace", reference: "LiveGradientReference", tensor_label: str
) -> None:
    """Install the deferred gradient hook on an in-place op's live result.

    The forward-time path hooks the live tensor right after the op, so the
    hook fires with the gradient of that version. When the tensor was written
    in place again later, its current ``grad_fn`` belongs to the later write;
    the hook then goes on the node the op produced, which receives exactly
    the gradient that tensor hook would have seen.

    Parameters
    ----------
    trace:
        Trace whose deferred gradient hooks are being installed.
    reference:
        Live tensor pinned at the op's autograd node.
    tensor_label:
        Raw label of the in-place op.
    """

    with internal_scalar_read():
        current_grad_fn = reference.tensor.grad_fn
    _install_tensor_grad_hook(
        trace,
        reference.tensor if current_grad_fn is reference.grad_fn else reference,
        tensor_label,
        take_ownership=True,
        owning_backward_only=reference.owning_backward_only,
    )


def _add_rebound_gradient_hook(
    trace: "Trace", logged: torch.Tensor, grad_fn_handle: Any, tensor_label: str
) -> None:
    """Rebind a refreshed op's gradient hook onto the projection target.

    A refresh projection keeps only the logged copy of each op output, and
    that copy is a graph dead end: its ``CloneBackward`` node was cut from the
    live tensor the model kept using. For an in-place op the live tensor may
    even have been written again since. The copy's clone nodes still lead to
    the autograd node the op produced (``grad_fn_handle``, pinned for the
    trace's lifetime) and to the live tensor's output slot on it, so the hook
    goes on that node's slot, the path the deferred in-place install uses
    (``_add_live_gradient_reference_hook``). A copy without that link keeps
    the plain tensor hook.

    Parameters
    ----------
    trace:
        Projection target whose gradient hooks are being rebound.
    logged:
        Logged copy of the op output; its id keys label ownership.
    grad_fn_handle:
        Autograd node the op produced, or ``None``.
    tensor_label:
        Raw label of the op.
    """

    from ..._capture.session import LiveGradientReference

    output_nr = _logged_copy_source_slot(logged, grad_fn_handle)
    if output_nr is None or not getattr(trace, "capture_tensor_grad_hooks", True):
        _add_tensor_backward_hook(trace, logged, tensor_label)
        return
    _install_tensor_grad_hook(
        trace,
        LiveGradientReference(
            tensor=logged,
            grad_fn=grad_fn_handle,
            output_nr=output_nr,
            owning_backward_only=False,
        ),
        tensor_label,
        take_ownership=False,
        owning_backward_only=False,
    )


def _install_tensor_grad_hook(
    trace: "Trace",
    site: "torch.Tensor | LiveGradientReference",
    tensor_label: str,
    *,
    take_ownership: bool,
    owning_backward_only: bool,
) -> None:
    """Register the gradient-recording hook for ``tensor_label`` (see ``_add_tensor_backward_hook``).

    Parameters
    ----------
    trace:
        Trace that records the gradient.
    site:
        Tensor whose gradient is recorded (its id keys label ownership), or a
        live reference to a tensor written in place again since the op: the
        hook then receives the incoming gradient of the node the op produced,
        for the tensor's slot, instead of the current version's gradient.
    tensor_label:
        Raw label the gradient is recorded under.
    take_ownership:
        Transfer gradient-emission ownership of ``tensor_label`` to ``t``.
    owning_backward_only:
        Whether the hook sits on model state.
    """

    t = site if isinstance(site, torch.Tensor) else site.tensor
    trace_fields = trace.__dict__
    hooked_tensors = trace_fields.get("_tl_backward_hooked_tensor_keys")
    if hooked_tensors is None:
        hooked_tensors = trace_fields["_tl_backward_hooked_tensor_keys"] = set()
    hooked_tensor_id = id(t)
    hook_key = (tensor_label, hooked_tensor_id)
    grad_hook_owners = trace_fields.get("_tl_grad_hook_owner_by_label")
    if grad_hook_owners is None:
        grad_hook_owners = trace_fields["_tl_grad_hook_owner_by_label"] = {}
    if take_ownership or tensor_label not in grad_hook_owners:
        grad_hook_owners[tensor_label] = hooked_tensor_id
    if hook_key in hooked_tensors:
        return
    hooked_tensors.add(hook_key)
    # Kept by name: torch's "hook will not be serialized" warning prints its repr. It holds
    # a weak Trace reference (no Trace -> tensor -> hook cycle) and this registration's
    # label (a label moves on with in-place ops); the work is in _record_hooked_gradient.
    trace_ref = weakref.ref(trace)

    def log_grad_to_model_history(grad: torch.Tensor | None) -> torch.Tensor | None:
        return _record_hooked_gradient(
            trace_ref, tensor_label, hooked_tensor_id, owning_backward_only, grad
        )

    slot_prehook = _backward_module._register_slot_gradient_prehook
    if owning_backward_only:
        handle = register_tensor_grad_hook(site, log_grad_to_model_history, slot_prehook)
        _OWNED_STATE_GRAD_HOOK_HANDLES.setdefault(trace, []).append(handle)
    elif not add_plain_tensor_hook(site, log_grad_to_model_history):
        register_tensor_grad_hook(site, log_grad_to_model_history, slot_prehook)


def _record_hooked_gradient(
    trace_ref: "weakref.ReferenceType[Trace]",
    tensor_label: str,
    hooked_tensor_id: int,
    owning_backward_only: bool,
    grad: torch.Tensor | None,
) -> torch.Tensor | None:
    """Edit, emit and optionally retain one gradient observed by a tensor hook.

    Autograd hands a hook ``None`` when no gradient reached the tensor (an
    unused output of a multi-output op: one half of ``torch.split``, the
    ``S`` of ``torch.svd``, the eigenvalues of ``torch.linalg.eig``). That
    records nothing: no gradient stored, no memory counted, no pass opened,
    so the op reads no gradient rather than an invented zero tensor.

    This is the TorchLens boundary for the gradient of the op's output:
    ``tl.grad(site)`` edits run here first (in declared order), the saved
    gradient is the chain's final value, and the edited tensor is
    returned so autograd feeds it into the op's backward function. The first
    four arguments are bound per registration by ``_install_tensor_grad_hook``.
    """
    if grad is None:
        return None
    target = _hooked_gradient_target(
        trace_ref, tensor_label, hooked_tensor_id, owning_backward_only
    )
    if target is None:
        return None
    active_trace, edits_live = target
    if edits_live:
        grad = apply_gradient_site_edits(active_trace, grad, tensor_label)
    prebuilt = _emit_tensor_grad_event(active_trace, grad, tensor_label)
    # Gate the legacy layer-slot write on the ACTIVE per-call policy,
    # not the deprecated ``save_grads`` attribute: a per-call
    # ``backward(..., save_grads=False)`` sets the policy while
    # the attribute can stay truthy, and the attribute-keyed gate
    # kept retaining full grad payloads the caller disabled.
    # The event-sidecar payloads are handed down for REUSE: the
    # legacy layer-slot write used to mint a second independent
    # clone and run grad_transform a second time, uncharged by the
    # save budget.
    if _active_save_grads_policy(active_trace) not in (None, False, "none", []):
        _log_tensor_grad(active_trace, grad, tensor_label, prebuilt=prebuilt)
    return grad


def _hooked_gradient_target(
    trace_ref: "weakref.ReferenceType[Trace]",
    tensor_label: str,
    hooked_tensor_id: int,
    owning_backward_only: bool,
) -> "tuple[Trace, bool] | None":
    """Return the trace a hooked gradient records into and whether edits run, or ``None``.

    ``None`` drops the fire: a non-owner alias, an active fork probe, a
    state-entangled trace outside its own managed backward, or a target whose
    owner map gives the label to another tensor. The gates run in this order.
    """
    # Essential complexity: the ownership, probe, fork and policy gates form one
    # fire-time decision (moved unchanged from the hook closure).
    active_trace = trace_ref()
    # One-owner-per-label: a non-owner alias registration drops its fire
    # so one logical op output emits exactly one OpGradObserved per pass.
    if active_trace is not None and _foreign_label_owner(
        active_trace, tensor_label, hooked_tensor_id
    ):
        return None
    refresh_target_ref = getattr(active_trace, "_refresh_projection_target_ref", None)
    if refresh_target_ref is not None:
        active_trace = refresh_target_ref()
    # A fork probe resolves here to the BASE trace whose per-trace flag
    # is unset; the global depth suppresses grad recording on every trace
    # while any RF/PF probe runs.
    if _state._rf_probe_depth > 0 or (
        active_trace is not None and getattr(active_trace, "_tl_rf_probe_active", False)
    ):
        return None
    # A managed backward directed at a FORK RELATIVE (a fork's
    # ``backward`` over the shared forward tensors) owns this
    # gradient: recording it here would silently mutate the hook trace's
    # projection with a pass the user directed at the fork. Fork
    # relatives share one label space, so the observation redirects to
    # the bracket-holding relative instead. Unrelated traces (e.g. two
    # composed models) keep the historical implicit-pass recording.
    active_trace = _fork_redirect_target(active_trace)
    if (owning_backward_only or _is_state_entangled(trace_ref())) and not (
        _owning_backward_is_running(trace_ref(), active_trace)
    ):
        return None
    # One-owner-per-label must hold on the FINAL emission target, not
    # just the hook's own trace: after a refresh-projection or fork
    # redirect, a stale source-trace hook (e.g. on an in-place live
    # tensor) can own the label in the SOURCE map while the target's
    # rebound map names a different tensor -- both passing their own
    # map would emit duplicate observations for one label. A target
    # map with no entry for the label stays permissive so redirected
    # gradients are not silently dropped.
    edits_live = _state.active_capture()[0] is active_trace
    if active_trace is None or (
        active_trace is not trace_ref()
        and _foreign_label_owner(active_trace, tensor_label, hooked_tensor_id)
    ):
        return None
    return active_trace, edits_live


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

    from ..._capture import projections

    active_state = getattr(projections, "_active_recording_state", None)
    if active_state is not None and getattr(active_state, "runtime_trace", None) is trace:
        return max(0, int(active_state.step_index) - 1)

    step_indices = [
        int(step_index)
        for layer in trace._layer_list
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
    pass_index = getattr(trace, "_active_backward_pass_number", None)
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
    pass_index = int(trace._num_backward_passes) + 1
    trace._active_backward_pass_number = pass_index
    trace._implicit_backward_pass_open = True
    if current_task_id is not None:
        _IMPLICIT_BACKWARD_TASK_IDS[trace] = current_task_id
        # Engine-drain close: opening inside a tensor hook is
        # provably in-backward, so queue the drain callback for THIS graph
        # task now. Opportunistic -- the sync-point backstop stays armed.
        from .backward import _enqueue_implicit_pass_drain_callback

        _enqueue_implicit_pass_drain_callback(trace, pass_index, current_task_id)
    if "implicit_backward_pass" not in trace._warned_once:
        warnings.warn(
            "TorchLens observed gradients outside a managed backward trigger; recording an "
            "implicit backward pass. Use trace.backward() or a TorchLens "
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
        second uncharged clone and re-running ``grad_transform``.
    """

    if getattr(trace, "_tl_backward_triggers_disarmed", False):
        return None, None, False
    stream_start = time.perf_counter()
    events = _ensure_backward_event_stream(trace)
    _record_phase_timing(
        trace,
        "backward_grad_event_ensure_stream",
        time.perf_counter() - stream_start,
    )
    pass_start = time.perf_counter()
    pass_index = _ensure_backward_pass_for_tensor_hook(trace)
    _record_phase_timing(
        trace,
        "backward_grad_event_ensure_pass",
        time.perf_counter() - pass_start,
    )
    payload_start = time.perf_counter()
    final_label = getattr(trace, "_raw_to_final_layer_labels", {}).get(tensor_label, tensor_label)
    with pause_logging():
        memory = int(grad.nelement() * grad.element_size())
        payload, transformed_payload, built = _build_grad_payloads(trace, grad, final_label)
    _record_phase_timing(
        trace,
        "backward_grad_event_payload",
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
        "backward_grad_event_append",
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
        legitimately-``None`` raw slot under ``save_raw_grads=False`` —
        is THE charged retention for this grad+label and safe to reuse).
    """

    if not _should_save_grad_payload(trace, layer_label):
        return None, None, False
    if layer_label not in trace._layer_dict_all_keys:
        raw_payload, transformed_payload = _build_fastlog_grad_payloads(trace, grad)
        return raw_payload, transformed_payload, True
    op = trace._layer_dict_all_keys[layer_label]
    grad_transform = trace._options_grad_transform
    save_raw_gradients = trace._options_save_raw_grads
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
    if layer_label not in trace._layer_dict_all_keys:
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
    op = trace._layer_dict_all_keys[layer_label]
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

    grad_transform = trace._options_grad_transform
    save_raw_gradients = trace._options_save_raw_grads
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
    return cast(SaveMode, trace._options_save_mode)


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
        self.layer_type = op.type
        self.type = op.type
        self.module_stack = tuple(getattr(op, "module_stack", ()) or ())
        self.modules = tuple(getattr(op, "modules", ()) or ())
        self.output_of_module_calls = tuple(record_member(op, "output_of_module_calls", ()) or ())
        self.has_forward_op = True
        self.has_op = True
        self.grad_kind = "grad_output"
        self.pass_index = pass_index
        self.backward_pass_number = pass_index
        self.shape = op.shape
        self.dtype = op.dtype
        self.tensor_device = record_member(op, "output_device", None)


def _param_log_for_exact_address(trace: "Trace", address: str) -> Any | None:
    """Return the Param record for an exact address, or None.

    Membership is checked against the exact address mapping, never the
    accessor's fuzzy short-name/substring resolution, so an op label can
    never accidentally resolve to a parameter.
    """

    param_logs = trace._param_logs
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
        self.backward_pass_number = pass_index
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
        self.backward_pass_number = pass_index
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

    pass_index = getattr(trace, "_active_backward_pass_number", None)
    return None if pass_index is None else int(pass_index)


def _active_save_grads_policy(trace: "Trace") -> Any:
    """Return the current gradient retention policy for tensor hooks."""

    if hasattr(trace, "_active_save_grads_policy"):
        return getattr(trace, "_active_save_grads_policy")
    return None if trace.__dict__.get("_tl_cleaned_up", False) else trace._options_save_grads


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
            a second time; output-layer children
            share the parent's payloads, matching their identity-wrapper
            contract. When the event build did not run (narrower event
            selection, disarmed sidecar), the slot's payloads are built and
            charged per layer through the same chokepoint.
    """
    self._autograd_has_grads = True
    if _label_raw not in self._raw_to_final_layer_labels:
        return
    if not self._layer_dict_all_keys:
        # A fastlog/raw runtime trace never builds the legacy layer surface
        # (postprocess is skipped), so the layer-slot write is structurally
        # inapplicable there -- its gradients ride the backward event stream.
        # The per-call save_grads policy gate armed this write on
        # recording backward, where every lookup would KeyError. Full traces
        # keep the populated dict, so a genuinely missing individual key
        # still raises below (mapping/dict divergence stays a tripwire).
        return
    tensor_label = self._raw_to_final_layer_labels[_label_raw]
    layer_log_entry = self._layer_dict_all_keys[tensor_label]
    layers_to_update = [tensor_label]
    # Output layers are identity wrappers; propagate grad to them too.
    if layer_log_entry._graph_is_output_parent:
        for child_layer in layer_log_entry.children:
            if self._layer_dict_all_keys[child_layer].is_output:
                layers_to_update.append(child_layer)

    for layer_label in layers_to_update:
        layer = self._layer_dict_all_keys[layer_label]
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
            layer._log_tensor_grad(grad, prebuilt=(payload_source[0], payload_source[1]))
        else:
            # Retention policy denies a payload for this label through the
            # charged chokepoint; keep the historical bare slot write.
            layer._log_tensor_grad(grad)
        self._cost_saved_grad_memory += layer.gradient_memory
        self._cost_total_grad_memory += layer.gradient_memory


def _locate_parent_tensors_in_args(
    self: "Trace",
    parent_log_entries: list[Any],
    args: tuple[Any, ...],
    kwargs: dict[Any, Any],
) -> dict[str, dict[Any, str]]:
    """Map each parent tensor (its indexed journal record) to its position in args/kwargs.

    Two levels of nesting: ``args[i]`` maps to key ``i``, ``args[i][j]`` to ``(i, j)``.
    Stored as the child's ``parent_arg_positions``; ``_get_parent_contents`` reads
    pre-call parent values with it and validation replay rebuilds the call from it.
    Labels come from the op's argument slot snapshot when one was taken.

    Returns:
        ``{"args": {pos: label, ...}, "kwargs": {key: label, ...}}``
    """
    tensor_all_arg_positions: dict[str, dict[Any, str]] = {"args": {}, "kwargs": {}}
    if not parent_log_entries:
        return tensor_all_arg_positions
    if (slots := current_arg_slots(self, args, kwargs)) is not None:
        return parent_arg_positions_from_slots(slots, parent_log_entries)

    positions_by_label: dict[str, dict[str, list[Any]]] = {
        parent_record.label_raw: {"args": [], "kwargs": []} for parent_record in parent_log_entries
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
    for parent_record in parent_log_entries:
        parent_label = parent_record.label_raw
        for arg_type in ("args", "kwargs"):
            for position in positions_by_label[parent_label][arg_type]:
                tensor_all_arg_positions[arg_type][position] = parent_label

    return tensor_all_arg_positions


def _get_ancestors_from_parents(
    parent_entries: list[Any],
) -> tuple[set[str], set[str]]:
    """Utility function to get the ancestors of a tensor based on those of its parent tensors.

    Args:
        parent_entries: the parents' indexed journal records

    Returns:
        List of input ancestors and internally initialized ancestors.
    """
    input_ancestors = set()
    internal_source_ancestors = set()

    for parent_record in parent_entries:  # set(...) as the live view's getter returns it
        input_ancestors.update(set(parent_record.input_ancestors))
        internal_source_ancestors.update(set(parent_record.internal_source_ancestors))
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
    It is the resolver applied to the call's recorded inputs
    (:mod:`._equivalence_resolve`), so resolving later gives the same string.

    Args:
        args: Positional arguments to the function call.
        kwargs: Keyword arguments to the function call.
        i: Index of this output tensor within a multi-output call.
        layer_type: The normalized operation name.
        fields_dict: Must contain ``in_multi_output`` and ``module``.

    Returns:
        A string key identifying this operation's equivalence class.
    """
    output_index = i if fields_dict["in_multi_output"] else None
    record = observe_arg_inputs(args, kwargs)
    return resolve_equivalence_class(layer_type, record, output_index, fields_dict["module"])


# ``backward`` imports this module at its top; bound here, at the bottom, so the cycle
# resolves in either import order (the hook install reads its names at call time).
from . import backward as _backward_module  # noqa: E402
