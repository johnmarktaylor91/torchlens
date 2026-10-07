"""Capture torch backward execution and autograd graph metadata.

This module installs backward/grad wrappers, walks grad_fn graphs, records hook
events, and exposes Trace/Recording backward helpers.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import os
import re
import threading
import time
import warnings
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import torch

from ... import _state
from ..._deprecations import MISSING, MissingType
from ..._state import pause_logging
from ...data_classes.backward_pass import BackwardPass
from ...data_classes.func_call_location import FuncCallLocation
from ...data_classes.grad_fn import GradFn
from ...data_classes.grad_fn_call import GradFnCall
from ...data_classes.op import _dtype_or_none, _memory_or_none, _shape_or_none
from ...errors import ConfigurationError
from ...ir.events import (
    BackwardCoverageGap,
    BackwardPassEnd,
    BackwardPassStart,
    CheckpointInvocationObserved,
    GradFnDiscovered,
    GradFnFired,
    OpGradObserved,
    ParamGradObserved,
)
from ...quantities import Bytes, Duration
from ...utils._torch_compat import (
    HAS_SAVED_TENSORS_HOOKS_PATCHABLE,
    get_accumulate_grad_class,
)
from ...utils._torch_symbols import torch_attr
from ...utils.introspection import _get_code_qualname, _get_col_offset
from ...utils.tensor_utils import synchronize_pending_cpu_async_copies
from ._fire_timing import (
    _clear_fire_timing_stamps,
    _fire_timing_stamp_list,
    _pop_matching_fire_start,
    _register_fire_timing_prehook,
)
from ._gradfn_markers import (
    _drain_gradfn_markers,
    _enter_internal_gradfn_bracket,
    _exit_gradfn_marker,
    _gradfn_marker_list,
)
from ._tl import detached_saved_activation_label, get_tensor_label
from .escape_detection import expected_original_call
from .tensor_tracking import (
    _STATE_BOUNDARY_GRAD_FNS,
    _copy_grad_payload,
    _current_backward_graph_task_id,
    _defer_at_state_boundary,
    _ensure_backward_event_stream,
    _forward_op_count_at_backward_trigger,
    _resume_state_boundaries,
    _should_save_grad_payload,
    _trace_grad_save_mode,
)

_BACKWARD_GRAD_FN_REGISTRY: dict[int, weakref.ReferenceType[Any]] = {}
_ORIGINAL_AUTOGRAD_BACKWARD: Callable[..., Any] | None = None
_ORIGINAL_AUTOGRAD_GRAD: Callable[..., Any] | None = None
# Exact installed patch objects: teardown restores a slot only when it still
# holds OUR patch (the identity-checked standard every sibling teardown in
# wrappers.py/belt.py/identity_shims.py/rescue.py follows). A foreign patch
# layered on top is preserved and disclosed, never clobbered.
_INSTALLED_AUTOGRAD_BACKWARD: Callable[..., Any] | None = None
_INSTALLED_AUTOGRAD_GRAD: Callable[..., Any] | None = None
_AUTOGRAD_WRAPPERS_INSTALLED = False
_ORIGINAL_SAVED_TENSORS_HOOKS_INIT: Callable[..., Any] | None = None
_ORIGINAL_SAVED_TENSORS_HOOKS_ENTER: Callable[..., Any] | None = None
_INSTALLED_SAVED_TENSORS_HOOKS_INIT: Callable[..., Any] | None = None
_INSTALLED_SAVED_TENSORS_HOOKS_ENTER: Callable[..., Any] | None = None
_SAVED_TENSORS_HOOKS_INIT_PATCHED = False
_TORCHLENS_PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_INFERENCE_ONLY_BACKWARD_ERROR = (
    "Cannot run log_backward on a trace captured with inference_only=True: the autograd "
    "graph was discarded during capture. Re-capture without inference_only (optionally with "
    "backward_ready=True) to enable deferred backward."
)
_CHUNKED_FORWARD_BACKWARD_ERROR = (
    "Cannot run log_backward on a trace captured with chunk_size: chunked forward capture "
    "concatenates forward payloads across independent sub-batches and does not retain a "
    "single full-batch autograd graph. Re-capture without chunk_size to enable deferred backward."
)
_NO_GRAD_AUTOGRAD_ERROR = "element 0 of tensors does not require grad and does not have a grad_fn"


def _run_with_detached_activation_guidance(
    roots: Any,
    engine_callable: Callable[[], Any],
) -> Any:
    """Run autograd and augment only a proven TorchLens-detached-root failure.

    Parameters
    ----------
    roots:
        Tensor roots passed to the PyTorch autograd engine.
    engine_callable:
        Zero-argument invocation of the original PyTorch entry point.

    Returns
    -------
    Any
        Original autograd return value.

    Raises
    ------
    RuntimeError
        The original PyTorch exception object, with guidance appended to its message only
        when an exact weak-identity marker and the canonical no-grad failure both match.
    """

    labels = [
        label
        for root in _root_tensors(roots)
        if (label := detached_saved_activation_label(root)) is not None
    ]
    try:
        return engine_callable()
    except RuntimeError as exc:
        if labels and str(exc) == _NO_GRAD_AUTOGRAD_ERROR:
            guidance = (
                f"TorchLens retained activation {labels[0]!r} after detaching it from autograd "
                "in the default capture mode. Re-trace with backward_ready=True before "
                "building a loss from this saved activation."
            )
            exc.args = (f"{exc}\n\n{guidance}", *exc.args[1:])
        raise


def _ensure_not_inference_only_backward(trace: Any) -> None:
    """Reject deferred backward capture for inference-only traces.

    Parameters
    ----------
    trace:
        Trace receiving backward metadata.

    Raises
    ------
    ConfigurationError
        If the trace was captured with ``inference_only=True``.
    """

    if getattr(trace, "inference_only", False):
        raise ConfigurationError(_INFERENCE_ONLY_BACKWARD_ERROR)


def _ensure_not_chunked_forward_backward(trace: Any) -> None:
    """Reject deferred backward capture for chunked-forward traces.

    Parameters
    ----------
    trace:
        Trace receiving backward metadata.

    Raises
    ------
    ConfigurationError
        If the trace was captured with ``chunk_size``.
    """

    if getattr(trace, "chunked_forward", False):
        raise ConfigurationError(_CHUNKED_FORWARD_BACKWARD_ERROR)


def _capture_backward_call_context(trace: Any) -> FuncCallLocation | None:
    """Return the first non-TorchLens caller location for a backward API call.

    Parameters
    ----------
    trace:
        Trace whose source-loading preference controls lazy source context.

    Returns
    -------
    FuncCallLocation | None
        User-visible call-site location, or ``None`` if no external frame is found.

    Notes
    -----
    Contract: an IMPLICIT backward (orphan hook-only autograd flow that never passes
    through a wrapped TorchLens backward trigger) has no user call site, so its
    ``backward_call_context`` must stay ``None``. No stable public fixture currently
    forces that path, so the contract lives here rather than as a skipped test
    (disputed-r2 b10/R79); if such a fixture is ever found, pin it in
    ``tests/test_backward_call_context.py``.
    """

    frame = inspect.currentframe()
    if frame is None:
        return None
    frame = frame.f_back
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        if not filename.startswith(_TORCHLENS_PKG_DIR):
            source_loading_enabled = bool(getattr(trace, "save_code_context", False))
            func_name = frame.f_code.co_name
            return FuncCallLocation(
                file=frame.f_code.co_filename,
                line_number=frame.f_lineno,
                func_name=func_name,
                num_context_lines_requested=int(getattr(trace, "num_context_lines", 7)),
                _frame_func_obj=(
                    frame.f_locals.get(func_name) or frame.f_globals.get(func_name)
                    if source_loading_enabled
                    else None
                ),
                code_firstlineno=frame.f_code.co_firstlineno,
                func_qualname=_get_code_qualname(frame),
                col_offset=_get_col_offset(frame),
                source_loading_enabled=source_loading_enabled,
            )
        frame = frame.f_back
    return None


def _strong_grad_fn_refs(trace: Any) -> list[Any]:
    """Return the trace-owned grad-fn strong-reference list.

    Parameters
    ----------
    trace:
        Trace that owns runtime autograd node references.

    Returns
    -------
    list[Any]
        Mutable list holding grad-fn wrapper objects for the trace lifetime.
    """

    return trace.__dict__.setdefault("_backward_gradfn_refs", [])


_BACKWARD_TRACE_SLOTS: weakref.WeakKeyDictionary[Any, tuple[weakref.ReferenceType[Any], set[int]]]
_BACKWARD_TRACE_SLOTS = weakref.WeakKeyDictionary()
"""Per-trace registration slot: its ONE self-evicting weakref plus its owned keys.

Values hold the trace only WEAKLY, so this table never pins its own keys.
"""

_PENDING_BACKWARD_FINALIZE: weakref.WeakKeyDictionary[Any, bool] = weakref.WeakKeyDictionary()
"""Traces whose implicit-close FINALIZE step (D2H fence + projection) is owed.

Set by the journal step of :func:`_close_implicit_backward_pass_if_open` and
cleared only after BOTH finalize sub-steps complete outside any engine
invocation (L9 memo 1.2). Weak-keyed so it never pins a trace.
"""


def _backward_registry_slot(trace: Any) -> tuple[weakref.ReferenceType[Any], set[int]]:
    """Return (and lazily build) one trace's backward-registry slot.

    A single weak reference per trace is created here, carrying an eviction
    callback: when the trace dies, every ``_BACKWARD_GRAD_FN_REGISTRY`` key it
    owns is removed immediately. Before this, dead entries only left the table
    opportunistically -- when some LATER, unrelated backward happened to walk
    past that grad-fn id, or when an explicit purge ran -- so a process that
    dropped traces without ``cleanup()`` accreted entries indefinitely.

    Reusing one weakref for all of a trace's ops also removes a per-op
    ``weakref.ref`` allocation from the capture path.

    Parameters
    ----------
    trace:
        Trace registering a backward trigger.

    Returns
    -------
    tuple[weakref.ReferenceType[Any], set[int]]
        The trace's shared weak reference and the set of registry keys it owns.
    """

    slot = _BACKWARD_TRACE_SLOTS.get(trace)
    if slot is not None:
        return slot
    owned_ids: set[int] = set()

    def _evict(dead_reference: weakref.ReferenceType[Any]) -> None:
        """Drop every registry key this now-dead trace owned."""

        for grad_fn_object_id in owned_ids:
            # Only evict keys still owned by THIS reference: a new grad-fn
            # object can reuse the id of a collected one.
            if _BACKWARD_GRAD_FN_REGISTRY.get(grad_fn_object_id) is dead_reference:
                _BACKWARD_GRAD_FN_REGISTRY.pop(grad_fn_object_id, None)
            if _STATE_BOUNDARY_GRAD_FNS.get(grad_fn_object_id) is dead_reference:
                _STATE_BOUNDARY_GRAD_FNS.pop(grad_fn_object_id, None)
        owned_ids.clear()

    slot = (weakref.ref(trace, _evict), owned_ids)
    _BACKWARD_TRACE_SLOTS[trace] = slot
    return slot


def _active_forward_op_count_at_trigger() -> int | None:
    """Return active forward op count before autograd graph walking.

    Returns
    -------
    int | None
        Last forward ``step_index`` that exists before the active autograd
        trigger's graph walk, or ``None`` when no forward capture is active.
    """

    from ...capture import projections

    active_state = getattr(projections, "_active_recording_state", None)
    if active_state is None:
        return None
    return max(0, int(active_state.step_index) - 1)


def _register_forward_grad_fn(trace: Any, grad_fn_handle: Any, _raw_label: str | None) -> None:
    """Register a pinned forward grad-fn object for later autograd triggers.

    Parameters
    ----------
    trace:
        Trace that recorded the forward operation.
    grad_fn_handle:
        Live PyTorch autograd node wrapper from the operation output.
    _raw_label:
        Raw TorchLens op label associated with the output tensor, if known.

    Returns
    -------
    None
        The process registry and trace strong-ref set are updated in place.
    """

    if grad_fn_handle is None:
        return
    refs = _strong_grad_fn_refs(trace)
    refs.append(grad_fn_handle)
    grad_fn_object_id = id(grad_fn_handle)
    trace_ref, owned_ids = _backward_registry_slot(trace)
    owned_ids.add(grad_fn_object_id)
    _BACKWARD_GRAD_FN_REGISTRY[grad_fn_object_id] = trace_ref


def _purge_trace_from_backward_registry(trace: Any) -> None:
    """Remove every registry key currently owned by ``trace``.

    Parameters
    ----------
    trace:
        Trace being cleaned up or disarmed.

    Returns
    -------
    None
        Matching process-registry entries are removed.
    """

    stale_ids = [
        grad_fn_object_id
        for grad_fn_object_id, trace_ref in _BACKWARD_GRAD_FN_REGISTRY.items()
        if trace_ref() is trace or trace_ref() is None
    ]
    for grad_fn_object_id in stale_ids:
        _BACKWARD_GRAD_FN_REGISTRY.pop(grad_fn_object_id, None)
    for grad_fn_object_id, ref in list(_STATE_BOUNDARY_GRAD_FNS.items()):
        if ref() is trace or ref() is None:
            _STATE_BOUNDARY_GRAD_FNS.pop(grad_fn_object_id, None)
    slot = _BACKWARD_TRACE_SLOTS.get(trace)
    if slot is not None:
        # Keep the owned-key set in step with the table, so a re-armed trace
        # never carries keys it no longer owns into its eviction callback.
        slot[1].clear()


def _close_implicit_backward_pass_if_open(trace: Any, *, _close_path: str = "sync_point") -> None:
    """Close an implicit backward bracket: journal + scavenge + guarded finalize.

    L9 memo 1.2 journal/scavenge/finalize split. The public shape stays ONE
    function with the same name and signature, so all three shipped call
    sites (the ``_run_backward_with_capture`` sync point on the owner thread,
    the tensor-hook task-id-change close on the ENGINE thread, and the lazy
    read path on whatever thread reads) keep calling it unchanged and no
    caller can bypass FINALIZE -- its deferral guard lives INSIDE this
    routine:

    - (J) JOURNAL-if-open: append the implicit ``BackwardPassEnd`` (with the
      close-path disclosure on the sidecar event only, wave 2), bump the pass
      counters, pop the implicit task-id entry, and set the pending-finalize
      flag. Cheap; engine-safe.
    - (S) SCAVENGE: clear the pending AccumulateGrad prehook records and the
      per-fire timing stamp lists. MUST run at close time, before any later
      pass can open -- deferring it is exactly the cross-pass
      ``(grad_fn_object_id, call_index)`` stale-pop mis-attribution.
    - (F) FINALIZE-if-pending: the R36-1 D2H fence + full projection, run iff
      the pending flag is set AND ``_current_backward_graph_task_id()`` is
      ``None`` (torch's own engine-invocation witness) -- NEVER inside an
      engine invocation, hence never on the engine thread mid-drain. When
      the guard fails the flag STAYS SET and the next qualifying call
      finalizes; the flag clears only after BOTH steps complete.

    Parameters
    ----------
    trace:
        Trace that may have an orphan tensor-hook bracket open.
    _close_path:
        Close-path disclosure value for the journaled End event (private;
        the engine-drain callback passes ``"engine_drain"``, every backstop
        path keeps the ``"sync_point"`` default). Provisional vocabulary,
        DOCUMENTED-UNSTABLE (E-L9-4 routing).
    """

    from .tensor_tracking import _IMPLICIT_BACKWARD_TASK_IDS

    if getattr(trace, "_implicit_backward_pass_open", False):
        pass_index = getattr(trace, "_active_backward_pass_index", None)
        if pass_index is not None:
            events = _ensure_backward_event_stream(trace)
            events.append_backward(
                BackwardPassEnd(
                    pass_index=int(pass_index),
                    duration=None,
                    peak_memory=None,
                    status="ok",
                    order_attribution_coverage=None,
                    close_path=_close_path,
                )
            )
            trace.num_backward_passes = max(
                int(getattr(trace, "num_backward_passes", 0)), int(pass_index)
            )
            trace.__dict__.pop("_active_backward_pass_index", None)
            # Close the open-pass flag BEFORE the fallible record clear
            # (R14-1 sibling ordering): a clear failure propagates loudly
            # either way, but must not strand the implicit pass marked open
            # after its End event was journaled.
            trace._implicit_backward_pass_open = False
            _IMPLICIT_BACKWARD_TASK_IDS.pop(trace, None)
            _PENDING_BACKWARD_FINALIZE[trace] = True
            # (S) SCAVENGE on every journaled close, on any thread.
            _clear_pending_accumulate_grad_records(trace)
            _clear_fire_timing_stamps(trace)
            _drain_gradfn_markers(trace)
    # (F) FINALIZE-if-pending, guarded in-routine so no caller can bypass it.
    if _PENDING_BACKWARD_FINALIZE.get(trace) and _current_backward_graph_task_id() is None:
        # Fence in-flight cpu_async D2H grad copies before projections make
        # the payloads reachable: the forward finalize seam already ran, so
        # backward is the only remaining producer of pending copies.
        synchronize_pending_cpu_async_copies()
        _materialize_backward_projections(trace)
        _PENDING_BACKWARD_FINALIZE.pop(trace, None)


def _backward_finalize_pending(trace: Any) -> bool:
    """Return whether an implicit close's FINALIZE step is still owed."""

    return bool(_PENDING_BACKWARD_FINALIZE.get(trace))


def _enqueue_implicit_pass_drain_callback(
    trace: Any, pass_index: int, graph_task_id: int | None
) -> Callable[[], None] | None:
    """Queue the engine-drain close for a freshly opened implicit pass.

    Called at implicit-pass OPEN inside the tensor grad hook, i.e. provably
    in-backward (``queue_callback`` raises outside one, so the availability
    probe is the enqueue attempt itself). The callback binds to the graph
    task CURRENT at open: on fire it calls the split close routine only if
    BOTH captured values still match the trace's current open implicit state,
    so a stale callback can never close a newer pass. Final callbacks do not
    run on the engine's error path -- the drain is opportunistic, never
    presumed, and the sync-point path stays armed as the guaranteed backstop.

    Returns
    -------
    Callable[[], None] | None
        The enqueued callback (returned for tests), or ``None`` when the
        private engine handle is unavailable or the enqueue raises (current
        behavior unchanged; disclosure reports the sync-point path).
    """

    if graph_task_id is None:
        return None
    from .tensor_tracking import _IMPLICIT_BACKWARD_TASK_IDS

    trace_ref = weakref.ref(trace)

    def _drain_callback() -> None:
        """Close the implicit backward pass once the autograd engine drains.

        Registered as a final callback; torch runs these with the graph task
        still live, so the close is deferred off the engine thread. No-ops when
        the trace is gone or the pass is already closed.
        """
        live_trace = trace_ref()
        if live_trace is None:
            return
        if not getattr(live_trace, "_implicit_backward_pass_open", False):
            return
        if getattr(live_trace, "_active_backward_pass_index", None) != pass_index:
            return
        if _IMPLICIT_BACKWARD_TASK_IDS.get(live_trace) != graph_task_id:
            return
        _close_implicit_backward_pass_if_open(live_trace, _close_path="engine_drain")

    from ...utils._torch_compat import get_autograd_engine_queue_callback

    queue_callback = get_autograd_engine_queue_callback()
    if queue_callback is None:
        return None
    try:
        queue_callback(_drain_callback)
    except (RuntimeError, TypeError):
        return None
    return _drain_callback


def _root_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    """Flatten autograd root arguments into tensors.

    Parameters
    ----------
    value:
        Tensor or nested sequence passed to an autograd engine entry.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Tensor roots in left-to-right order.
    """

    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(_root_tensors(item))
        return tuple(tensors)
    return ()


def _traces_for_roots(roots: Any) -> tuple[Any, ...]:
    """Find live traces whose pinned grad-fn ids appear under ``roots``.

    Parameters
    ----------
    roots:
        Tensor or nested tensor sequence passed to an autograd entry point.

    Returns
    -------
    tuple[Any, ...]
        Matched traces in discovery order, without duplicates.
    """

    if not _BACKWARD_GRAD_FN_REGISTRY:
        return ()
    if _state._rf_probe_depth > 0:
        # An RF/PF gradient probe is in flight: probes are pure measurements
        # and must never mint a managed pass on ANY trace. The per-trace
        # ``_tl_rf_probe_active`` check below cannot cover a probe on a FORK,
        # whose registry entries resolve to the base trace.
        return ()
    matched: list[Any] = []
    matched_ids: set[int] = set()
    stale_ids: list[int] = []
    queue: deque[Any] = deque(
        root.grad_fn for root in _root_tensors(roots) if root.grad_fn is not None
    )
    seen: set[int] = set()
    # Model-state boundary nodes whose owner has not matched yet, by owner id.
    deferred: dict[int, list[Any]] = {}
    while queue:
        grad_fn_handle = queue.popleft()
        grad_fn_object_id = id(grad_fn_handle)
        if grad_fn_object_id in seen:
            continue
        seen.add(grad_fn_object_id)
        if _defer_at_state_boundary(grad_fn_handle, matched_ids, deferred):
            continue
        trace_ref = _BACKWARD_GRAD_FN_REGISTRY.get(grad_fn_object_id)
        if trace_ref is not None:
            trace = trace_ref()
            if trace is None or not hasattr(trace, "layer_list"):
                stale_ids.append(grad_fn_object_id)
            elif (
                id(trace) not in matched_ids
                and not getattr(trace, "_tl_backward_triggers_disarmed", False)
                and not getattr(trace, "_tl_rf_probe_active", False)
            ):
                matched.append(trace)
                matched_ids.add(id(trace))
                queue.extend(_resume_state_boundaries(trace, deferred, seen))
        queue.extend(_iter_next_grad_fns(grad_fn_handle))
    for stale_id in stale_ids:
        _BACKWARD_GRAD_FN_REGISTRY.pop(stale_id, None)
    return tuple(matched)


def _autograd_roots_from_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    root_kwarg: str,
) -> Any:
    """Return root tensors from an autograd wrapper call.

    Parameters
    ----------
    args:
        Positional arguments passed to the wrapped function.
    kwargs:
        Keyword arguments passed to the wrapped function.
    root_kwarg:
        Keyword name that carries roots for this autograd entry.

    Returns
    -------
    Any
        Original root argument object, or ``None`` when absent.
    """

    if args:
        return args[0]
    return kwargs.get(root_kwarg)


def _first_root_tensor(roots: Any) -> torch.Tensor | None:
    """Return the first tensor in an autograd root object.

    Parameters
    ----------
    roots:
        Tensor or nested sequence of tensors.

    Returns
    -------
    torch.Tensor | None
        First tensor root, if one exists.
    """

    tensors = _root_tensors(roots)
    return tensors[0] if tensors else None


def _root_forward_op_count(trace: Any, roots: Any) -> int | None:
    """Return the highest forward ``step_index`` among autograd root tensors.

    Parameters
    ----------
    trace:
        Trace that owns the root tensors.
    roots:
        Tensor or nested tensor sequence passed to an autograd entry point.

    Returns
    -------
    int | None
        Highest resolved root forward position, or ``None`` when no root label
        can be resolved.
    """

    root_steps = [
        step_index
        for root in _root_tensors(roots)
        if (step_index := _root_tensor_step_index(trace, root)) is not None
    ]
    return max(root_steps) if root_steps else None


def _root_tensor_step_index(trace: Any, root: torch.Tensor) -> int | None:
    """Return a TorchLens ``step_index`` for an autograd root tensor.

    Parameters
    ----------
    trace:
        Trace that owns the root tensor.
    root:
        Root tensor passed to an autograd entry point.

    Returns
    -------
    int | None
        Root layer ``step_index`` when resolvable.
    """

    label = get_tensor_label(root)
    if label is None:
        return None
    lookup_labels = (getattr(trace, "_raw_to_final_layer_labels", {}).get(label, label), label)
    layer_lookup = getattr(trace, "layer_dict_all_keys", {})
    for lookup_label in lookup_labels:
        layer = layer_lookup.get(lookup_label)
        step_index = getattr(layer, "step_index", None)
        if isinstance(step_index, int):
            return step_index
    raw_match = re.search(r"_(\d+)_raw$", label)
    if raw_match is not None:
        return int(raw_match.group(1))
    final_match = re.search(r"_[1-9]\d*_(\d+)(?::[1-9]\d*)?$", label)
    if final_match is not None:
        return int(final_match.group(1))
    return None


def _grad_fn_is_custom(grad_fn_handle: Any) -> bool:
    """Return whether a grad_fn_handle appears to come from user/custom autograd code.

    Parameters
    ----------
    grad_fn_handle:
        Autograd function object.

    Returns
    -------
    bool
        True when the type's module path is outside PyTorch's built-in autograd
        and nn namespaces.
    """
    class_module = type(grad_fn_handle).__module__
    if class_module == "torch.autograd.function":
        return True
    builtin_prefixes = ("torch.autograd", "torch.nn", "torch", "builtins")
    return not class_module.startswith(builtin_prefixes)


def _safe_source_file_and_line(obj: Any) -> tuple[str | None, int | None]:
    """Return source file and first line for an object when introspection works.

    Parameters
    ----------
    obj:
        Object to inspect.

    Returns
    -------
    tuple[str | None, int | None]
        Source file and line, or ``(None, None)`` when unavailable.
    """

    try:
        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        source_line = inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        return None, None
    return source_file, source_line


def _safe_signature(obj: Any) -> str | None:
    """Return an inspect signature string when available.

    Parameters
    ----------
    obj:
        Callable object to inspect.

    Returns
    -------
    str | None
        Signature string, or ``None`` when unavailable.
    """

    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _grad_fn_source_metadata(grad_fn_cls: type[Any]) -> dict[str, Any]:
    """Return best-effort source metadata for a grad-fn class.

    Parameters
    ----------
    grad_fn_cls:
        Runtime class of the autograd grad-fn object.

    Returns
    -------
    dict[str, Any]
        Keyword arguments accepted by ``GradFn`` for source metadata fields.
    """

    class_source_file, class_source_line = _safe_source_file_and_line(grad_fn_cls)
    init_method = getattr(grad_fn_cls, "__init__", None)
    forward_method = getattr(grad_fn_cls, "forward", None)
    backward_method = getattr(grad_fn_cls, "backward", None)
    init_source_file, init_source_line = _safe_source_file_and_line(init_method)
    forward_source_file, forward_source_line = _safe_source_file_and_line(forward_method)
    backward_source_file, backward_source_line = _safe_source_file_and_line(backward_method)
    return {
        "class_source_file": class_source_file,
        "class_source_line": class_source_line,
        "class_docstring": grad_fn_cls.__doc__,
        "init_source_file": init_source_file,
        "init_source_line": init_source_line,
        "init_signature": _safe_signature(init_method),
        "init_docstring": getattr(init_method, "__doc__", None),
        "forward_source_file": forward_source_file,
        "forward_source_line": forward_source_line,
        "forward_signature": _safe_signature(forward_method),
        "forward_docstring": getattr(forward_method, "__doc__", None),
        "backward_source_file": backward_source_file,
        "backward_source_line": backward_source_line,
        "backward_signature": _safe_signature(backward_method),
        "backward_docstring": getattr(backward_method, "__doc__", None),
    }


def _iter_next_grad_fns(grad_fn_handle: Any) -> Iterator[Any]:
    """Yield non-null child grad_fns from ``grad_fn_handle.next_functions``.

    Parameters
    ----------
    grad_fn_handle:
        Autograd function object.

    Yields
    ------
    Any
        Reachable child grad_fn_handle object.
    """
    for next_fn, _input_num in getattr(grad_fn_handle, "next_functions", ()):
        if next_fn is not None:
            yield next_fn


def _selected_for_grad_save(trace: Any, layer_label: str | None) -> bool:
    """Return whether a forward layer's grad should be saved.

    Parameters
    ----------
    trace:
        Trace being updated.
    layer_label:
        Final layer label, or ``None`` for intervening grad_fns.

    Returns
    -------
    bool
        True if this layer is selected by the trace's gradient-retention policy.
    """
    return layer_label is not None and _should_save_grad_payload(trace, layer_label)


def _sync_grad_fn_graph_relations(trace: Any) -> None:
    """Populate backward-oriented GradFn graph relation labels.

    Parameters
    ----------
    trace:
        Trace whose ``grad_fn_logs`` should be synchronized.
    """

    id_to_label = {
        grad_fn_object_id: grad_fn.label
        for grad_fn_object_id, grad_fn in trace.grad_fn_logs.items()
    }
    child_map: dict[str, list[str]] = {}
    parent_map: dict[str, list[str]] = {
        grad_fn.label: [] for grad_fn in trace.grad_fn_logs.values()
    }
    parent_membership: dict[str, set[str]] = {
        grad_fn.label: set() for grad_fn in trace.grad_fn_logs.values()
    }

    for grad_fn in trace.grad_fn_logs.values():
        children = [
            id_to_label[next_grad_fn_id]
            for next_grad_fn_id in grad_fn.next_grad_fn_ids
            if next_grad_fn_id in id_to_label
        ]
        child_map[grad_fn.label] = children
        for child_label in children:
            if grad_fn.label not in parent_membership[child_label]:
                parent_map[child_label].append(grad_fn.label)
                parent_membership[child_label].add(grad_fn.label)

    for grad_fn in trace.grad_fn_logs.values():
        grad_fn.children = child_map[grad_fn.label]
        grad_fn.parents = parent_map[grad_fn.label]
        sibling_labels: list[str] = []
        sibling_membership: set[str] = set()
        for parent_label in grad_fn.parents:
            for sibling_label in child_map.get(parent_label, []):
                if sibling_label != grad_fn.label and sibling_label not in sibling_membership:
                    sibling_labels.append(sibling_label)
                    sibling_membership.add(sibling_label)
        grad_fn.siblings = sibling_labels

        co_parent_labels: list[str] = []
        co_parent_membership: set[str] = set()
        for child_label in grad_fn.children:
            for co_parent_label in parent_map.get(child_label, []):
                if co_parent_label != grad_fn.label and co_parent_label not in co_parent_membership:
                    co_parent_labels.append(co_parent_label)
                    co_parent_membership.add(co_parent_label)
        grad_fn.co_parents = co_parent_labels


def _param_module_address(trace: Any, param_ref: object | None) -> str | None:
    """Return the owning module address for a parameter reference.

    Parameters
    ----------
    trace:
        Trace containing parameter logs.
    param_ref:
        Parameter address recorded on an ``AccumulateGrad`` discovery event.

    Returns
    -------
    str | None
        Owning module address, or ``None`` when the parameter is unknown.
    """

    if not isinstance(param_ref, str):
        return None
    param_logs = getattr(trace, "param_logs", {})
    if param_ref not in param_logs:
        return None
    param_log = param_logs[param_ref]
    return None if param_log is None else getattr(param_log, "module_address", None)


def _op_module_address(trace: Any, op_label: str | None) -> str | None:
    """Return the containing module address for a paired forward op label.

    Parameters
    ----------
    trace:
        Trace containing op logs.
    op_label:
        Forward op label paired to a grad-fn record.

    Returns
    -------
    str | None
        Module address for the paired op, if known.
    """

    if op_label is None or op_label not in getattr(trace, "layer_dict_all_keys", {}):
        return None
    op = trace.layer_dict_all_keys[op_label]
    module_address = getattr(op, "module_address", None)
    if module_address is not None:
        return cast(str, module_address)
    atomic_module_address = getattr(op, "atomic_module_address", None)
    if atomic_module_address is not None:
        return cast(str, atomic_module_address)
    output_module_calls = list(getattr(op, "output_of_module_calls", []) or [])
    if output_module_calls:
        return str(output_module_calls[-1]).rsplit(":", 1)[0]
    module_calls = list(getattr(op, "modules", []) or [])
    if not module_calls:
        return None
    return str(module_calls[-1]).rsplit(":", 1)[0]


def _resolve_op_grad_event_label(trace: Any, op_label: str) -> str:
    """Return the final lookup label for an ``OpGradObserved`` event.

    Parameters
    ----------
    trace:
        Trace that owns the event.
    op_label:
        Raw or final op label stored on the event.

    Returns
    -------
    str
        Final lookup label when postprocess mappings know it; otherwise the
        original label.
    """

    raw_to_final_layer = getattr(trace, "_raw_to_final_layer_labels", {})
    if isinstance(raw_to_final_layer, dict) and op_label in raw_to_final_layer:
        return str(raw_to_final_layer[op_label])
    raw_to_final_op = getattr(trace, "_raw_to_final_op_labels", {})
    if isinstance(raw_to_final_op, dict) and op_label in raw_to_final_op:
        return str(raw_to_final_op[op_label])
    return op_label


def _op_raw_index(trace: Any, grad_fn_record: GradFn) -> int | None:
    """Return the forward raw index backing a grad-fn's module attribution.

    Parameters
    ----------
    trace:
        Trace containing op logs.
    grad_fn_record:
        GradFn whose attributed op position should be inspected.

    Returns
    -------
    int | None
        Forward raw index, or ``None`` for parameter-only and unresolved records.
    """

    op_label = grad_fn_record.op_label
    if op_label is None or op_label not in getattr(trace, "layer_dict_all_keys", {}):
        return None
    return int(getattr(trace.layer_dict_all_keys[op_label], "raw_index"))


def _post_forward_grad_fn_ids(grad_fn_logs: dict[int, GradFn]) -> set[int]:
    """Return loss-construction grad-fn ids before the first op-anchored node.

    Parameters
    ----------
    grad_fn_logs:
        Projected GradFn records in backward discovery order.

    Returns
    -------
    set[int]
        Unpaired, non-creator-attributed ids that should not receive inferred
        module containment.
    """

    op_steps = [
        grad_fn_record.step_index
        for grad_fn_record in grad_fn_logs.values()
        if grad_fn_record.has_op
    ]
    if not op_steps:
        return set()
    first_op_step = min(op_steps)
    return {
        object_id
        for object_id, grad_fn_record in grad_fn_logs.items()
        if (
            grad_fn_record.step_index < first_op_step
            and not grad_fn_record.has_op
            and grad_fn_record.creator_object_id is None
        )
    }


def _candidate_module_sort_key(trace: Any, candidate: GradFn) -> tuple[int, int, str]:
    """Return the deterministic P5 tie-break key for a containment candidate.

    Parameters
    ----------
    trace:
        Trace containing forward op logs.
    candidate:
        Neighbor GradFn with resolved module containment.

    Returns
    -------
    tuple[int, int, str]
        Sort key where smaller values are preferred.
    """

    raw_index = _op_raw_index(trace, candidate)
    consumer_rank = -(raw_index if raw_index is not None else -1)
    return (consumer_rank, candidate.step_index, candidate.label)


def _infer_grad_fn_module_membership(trace: Any) -> None:
    """Populate inferred module containment for intervening backward nodes.

    Parameters
    ----------
    trace:
        Trace whose ``grad_fn_logs`` already have graph relations synchronized.

    Returns
    -------
    None
        GradFn ``modules``, ``module_address``, and ``module_membership_source``
        fields are updated in place.
    """

    grad_fn_logs = getattr(trace, "grad_fn_logs", {})
    label_to_id = {grad_fn.label: object_id for object_id, grad_fn in grad_fn_logs.items()}
    carve_out_ids = _post_forward_grad_fn_ids(grad_fn_logs)
    changed = True
    while changed:
        changed = False
        for object_id, grad_fn_record in grad_fn_logs.items():
            if (
                object_id in carve_out_ids
                or grad_fn_record.module_membership_source is not None
                or grad_fn_record.creator_object_id is not None
            ):
                continue
            neighbor_ids = [
                label_to_id[label]
                for label in [*grad_fn_record.parents, *grad_fn_record.children]
                if label in label_to_id
            ]
            candidates = [
                grad_fn_logs[neighbor_id]
                for neighbor_id in neighbor_ids
                if grad_fn_logs[neighbor_id].module_membership_source is not None
            ]
            if not candidates:
                continue
            winner = min(
                candidates, key=lambda candidate: _candidate_module_sort_key(trace, candidate)
            )
            if winner.module_address is None:
                continue
            grad_fn_record.module_address = winner.module_address
            grad_fn_record.modules = list(winner.modules) or [winner.module_address]
            grad_fn_record.module_membership_source = "inferred"
            changed = True


def _prefer_direct_pairing_source_for_paired_ops(trace: Any) -> None:
    """Mark direct op pairings as paired after containment inference.

    Parameters
    ----------
    trace:
        Trace whose ``grad_fn_logs`` should be normalized.

    Returns
    -------
    None
        Directly paired GradFn records keep any inferred module address but expose
        ``module_membership_source="paired"``.
    """

    for grad_fn_record in getattr(trace, "grad_fn_logs", {}).values():
        op = grad_fn_record.op
        if (
            op is not None
            and (getattr(op, "is_compute_op", False) or getattr(op, "is_compute_layer", False))
            and grad_fn_record.module_membership_source == "inferred"
        ):
            grad_fn_record.module_membership_source = "paired"


def _resolved_order_for_creator(
    creator_object_id: int | None,
    grad_fn_logs: dict[int, GradFn],
) -> int | None:
    """Return the derived order for a creator-attributed grad-fn node."""

    if creator_object_id is None:
        return 1
    creator = grad_fn_logs.get(creator_object_id)
    creator_order = None if creator is None else creator.order
    if creator_order is None:
        return None
    return creator_order + 1


def _pass_order_from_roots(
    root_grad_fn_ids: Sequence[int],
    grad_fn_logs: dict[int, GradFn],
    pass_orders: dict[int, int],
) -> int | None:
    """Derive a backward pass order from its root grad-fn nodes."""

    root_orders = [
        _root_based_order(grad_fn_logs[root_id], pass_orders)
        for root_id in root_grad_fn_ids
        if root_id in grad_fn_logs
    ]
    if not root_orders:
        return None
    if any(root_order is None for root_order in root_orders):
        return None
    if len(root_orders) != len(root_grad_fn_ids):
        return None
    unique_orders = set(root_orders)
    if len(unique_orders) != 1:
        return None
    return cast(int, root_orders[0])


def _root_based_order(root_grad_fn: GradFn, pass_orders: dict[int, int]) -> int | None:
    """Return the ROOT-based pass order for one root grad-fn.

    Parameters
    ----------
    root_grad_fn:
        Root autograd node for the backward pass.
    pass_orders:
        Already materialized pass orders keyed by pass index.

    Returns
    -------
    int | None
        Root-based order, or ``None`` when the root cannot be attributed.
    """

    origin_backward_pass = root_grad_fn.origin_backward_pass
    if origin_backward_pass is None:
        return root_grad_fn.order
    origin_order = pass_orders.get(origin_backward_pass)
    if origin_order is None:
        return root_grad_fn.order
    return origin_order + 1


def _order_attribution_coverage(
    calls: Sequence[GradFnCall],
    grad_fn_logs: dict[int, GradFn],
) -> float | None:
    """Return the fraction of calls with resolved grad-fn order metadata."""

    if not calls:
        return None
    labels_with_order = {
        grad_fn.label for grad_fn in grad_fn_logs.values() if grad_fn.order is not None
    }
    covered = sum(1 for call in calls if call.label in labels_with_order)
    return covered / len(calls)


def _max_call_order(
    calls: Sequence[GradFnCall],
    grad_fn_logs: dict[int, GradFn],
) -> int | None:
    """Return the maximum resolved GradFn order among calls in one pass.

    Parameters
    ----------
    calls:
        Materialized GradFn calls for a backward pass.
    grad_fn_logs:
        GradFn records keyed by object id.

    Returns
    -------
    int | None
        Highest order observed among the pass calls, or ``None`` when no call has
        resolved order metadata.
    """

    label_to_order = {
        grad_fn.label: grad_fn.order
        for grad_fn in grad_fn_logs.values()
        if grad_fn.order is not None
    }
    call_orders = [label_to_order[call.label] for call in calls if call.label in label_to_order]
    return max(call_orders) if call_orders else None


def _assign_grad_fn_orders(grad_fn_logs: dict[int, GradFn]) -> None:
    """Populate resolved GradFn order values from creators and known topology."""

    for grad_fn_record in grad_fn_logs.values():
        grad_fn_record.order = _resolved_order_for_creator(
            grad_fn_record.creator_object_id,
            grad_fn_logs,
        )

    changed = True
    while changed:
        changed = False
        for grad_fn_record in grad_fn_logs.values():
            if grad_fn_record.creator_object_id is not None or grad_fn_record.has_op:
                continue
            child_orders: list[int] = []
            for child_id in grad_fn_record.next_grad_fn_ids:
                child = grad_fn_logs.get(child_id)
                if child is not None and child.order is not None:
                    child_orders.append(child.order)
            if not child_orders:
                continue
            inferred_order = max(child_orders)
            if inferred_order > 1 and grad_fn_record.order != inferred_order:
                grad_fn_record.order = inferred_order
                changed = True


def _normalize_grad_fn_type(grad_fn_handle: Any) -> str:
    """Normalize an autograd grad_fn_handle class name for TorchLens labels.

    Parameters
    ----------
    grad_fn_handle:
        Autograd function object.

    Returns
    -------
    str
        Lowercased class name with a trailing ``Backward<digits>`` suffix removed.
    """
    return re.sub(r"Backward\d*$", "", type(grad_fn_handle).__name__).lower()


def _grad_fn_label_parts(
    trace: Any,
    type: str,
    _layer_label: str | None,
    type_counter: dict[str, int],
    total_num: int,
) -> tuple[int, int, str]:
    """Build numeric label fields for one grad_fn_handle.

    Parameters
    ----------
    trace:
        Trace being updated.
    type:
        Normalized grad_fn_handle type.
    _layer_label:
        Matching forward layer label, or ``None`` for intervening grad_fns.
    type_counter:
        Running per-type counter for intervening grad_fns.
    total_num:
        One-based discovery index in the backward graph.

    Returns
    -------
    tuple[int, int, str]
        GradFn type index, total index, and user-facing label.
    """
    del trace, _layer_label
    type_counter[type] = type_counter.get(type, 0) + 1
    type_num = type_counter[type]
    label = f"{type}_back_{type_num}_{total_num}"
    return type_num, total_num, label


def _grad_fn_type_from_class_name(class_name: str) -> str:
    """Normalize an autograd class name for native backward labels."""

    return re.sub(r"Backward\d*$", "", class_name).lower()


@dataclass
class _BackwardFoldState:
    """Cross-fold accumulators for incremental backward projection.

    Session-only state (never portable): holds exactly what the full rebuild
    computes along the way so that a clean appended event tail can be folded
    in O(tail) instead of re-reading every backward event ever emitted.
    """

    discovered: OrderedDict[int, GradFnDiscovered] = field(default_factory=OrderedDict)
    latest_topology: dict[int, tuple[int, ...]] = field(default_factory=dict)
    starts: dict[int, BackwardPassStart] = field(default_factory=dict)
    ends: dict[int, BackwardPassEnd] = field(default_factory=dict)
    per_object_ordinals: dict[int, int] = field(default_factory=dict)
    unique_payload_ids: set[int] = field(default_factory=set)
    total_gradient_memory: int = 0
    total_backward_memory: int = 0
    saved_grad_labels: set[str] = field(default_factory=set)
    resolved_pass_orders: dict[int, int] = field(default_factory=dict)
    built_pass_indices: set[int] = field(default_factory=set)
    op_grad_passes: set[int] = field(default_factory=set)
    param_grad_passes: set[int] = field(default_factory=set)


def _backward_tail_is_foldable(state: _BackwardFoldState, tail: list[Any]) -> bool:
    """Return whether an appended event tail can fold without a full rebuild.

    Foldable means the tail introduces no new grad-fn node, no merge that
    would retroactively change an already-assigned label or field, no
    topology change (which would require re-running the global order and
    module-inference passes), and no event for an already-built pass. Any
    other tail falls back to the byte-identical full rebuild.
    """

    max_built = max(state.built_pass_indices, default=0)
    for event in tail:
        if isinstance(event, BackwardPassStart):
            if event.pass_index <= max_built or event.pass_index in state.starts:
                return False
        elif isinstance(event, BackwardPassEnd):
            if event.pass_index <= max_built or event.pass_index in state.ends:
                return False
        elif isinstance(event, GradFnDiscovered):
            existing = state.discovered.get(event.object_id)
            if existing is None:
                return False
            if existing.op_label is None and event.op_label is not None:
                return False
            if existing.param_ref is None and event.param_ref is not None:
                return False
            if existing.created_in_pass is None and event.created_in_pass is not None:
                return False
            if existing.creator_object_id is None and event.creator_object_id is not None:
                return False
            if tuple(event.topology) != tuple(state.latest_topology.get(event.object_id, ())):
                return False
        elif isinstance(event, (GradFnFired, OpGradObserved, ParamGradObserved)):
            if event.pass_index <= max_built:
                return False
        elif isinstance(event, CheckpointInvocationObserved):
            # Projection-neutral: token evidence lives in runtime state and
            # the witness field, never in the folded records.
            continue
        else:
            return False
    return True


def _materialize_backward_projections(trace: Any) -> None:
    """Rebuild backward-facing Trace records from sidecar events.

    Parameters
    ----------
    trace:
        Trace whose runtime backward event sidecar is authoritative.

    Returns
    -------
    None
        ``grad_fn_logs``, ``grad_fn_order``, ``backward_pass_logs``, and
        related counters are replaced in place.
    """

    if getattr(trace, "_tl_active_backward_bracket", False):
        return
    if getattr(trace, "_tl_materializing_backward_projection", False):
        return
    stream = _ensure_backward_event_stream(trace)
    events = list(getattr(stream, "backward_events", ()))
    if not events:
        return
    # Checkpoint witness (memo 2.3): every backward-capturing trace gets the
    # projected summary, so the affirmative zero-token verdict is available
    # even when no checkpoint enter or degrade flag ever fired.
    _refresh_checkpoint_witness(trace)
    revision = getattr(stream, "backward_revision", None)
    if revision is None:
        # Legacy stream (e.g. unpickled from an older version) without a
        # revision counter: fall back to the event count as a change signal.
        revision = len(events)
    if getattr(trace, "_backward_projection_revision", None) == revision:
        from ...postprocess._primitive_profile import _materialize_backward_primitive_profile

        _materialize_backward_primitive_profile(trace)
        return
    trace._tl_materializing_backward_projection = True
    try:
        state = getattr(trace, "_backward_projection_fold_state", None)
        watermark = getattr(trace, "_backward_projection_event_count", None)
        if (
            isinstance(state, _BackwardFoldState)
            and isinstance(watermark, int)
            and 0 < watermark < len(events)
            and _backward_tail_is_foldable(state, events[watermark:])
        ):
            _fold_backward_projection_tail(trace, state, events[watermark:])
            full_rebuild = False
        else:
            _materialize_backward_projections_impl(trace, events, stream=stream)
            full_rebuild = True
        _bind_backward_epoch(
            trace,
            full_rebuild=full_rebuild,
            revision=revision,
            watermark=len(events),
        )
        # Publication is the LAST step: the lazy-invalidation stamps only
        # advance once the facade projection AND the core epoch both hold
        # the complete result, so a failure anywhere above cannot suppress
        # the retry (sol review finding 2).
        trace._backward_projection_event_count = len(events)
        trace._backward_projection_revision = revision
        from ...postprocess._primitive_profile import _materialize_backward_primitive_profile

        _materialize_backward_primitive_profile(trace)
    except BaseException:
        # Poison every lazy-invalidation surface: the projection or epoch
        # may be partially mutated, so the next access must take the full
        # scratch rebuild (events are all retained by the stream), never a
        # fold over inconsistent state.
        trace._backward_projection_revision = None
        trace._backward_projection_event_count = None
        trace._backward_projection_fold_state = None
        core = trace.__dict__.get("_trace_core")
        if core is not None:
            core.backward_epochs = []
        raise
    finally:
        trace.__dict__.pop("_tl_materializing_backward_projection", None)


def _bind_backward_epoch(trace: Any, *, full_rebuild: bool, revision: Any, watermark: Any) -> None:
    """Bind the projected backward records to the core's backward epoch (M9).

    Runs AFTER a successful projection and BEFORE the trace-side stamp
    publication. The swap is genuinely atomic: a full rebuild adopts every
    row into a STAGED fresh epoch and only then replaces the epoch list, so
    a failed adoption leaves ``core.backward_epochs`` untouched (the caller
    poisons the lazy-invalidation stamps and the next access rebuilds from
    scratch). A clean tail fold extends the live epoch; ``adopt_rows``
    skips already-adopted records, so folds only add the new tail rows and
    a failed fold is healed by the caller-forced full rebuild. The epoch
    stamps ``revision``/``watermark`` (its generation) last, after every
    adoption succeeded. Non-core-backed traces (loaded artifacts, previews)
    keep detached-backed records.
    """

    core = trace.__dict__.get("_trace_core")
    if core is None:
        return
    from torchlens._trace_core.record_rows import BackwardEpoch, adopt_rows

    epochs = core.backward_epochs
    staged = full_rebuild or not epochs
    epoch = BackwardEpoch() if staged else epochs[-1]
    grad_fn_logs = getattr(trace, "grad_fn_logs", None) or {}
    grad_fns = list(grad_fn_logs.values())
    calls: list[Any] = []
    for grad_fn_record in grad_fns:
        call_map = getattr(grad_fn_record.calls, "_dict", grad_fn_record.calls)
        calls.extend(call_map.values())
    passes = list((getattr(trace, "backward_pass_logs", None) or {}).values())
    adopt_rows(epoch.stores, "grad_fn", grad_fns)
    adopt_rows(epoch.stores, "grad_fn_call", calls)
    adopt_rows(epoch.stores, "backward_pass", passes)
    epoch.revision = revision
    epoch.watermark = watermark
    if staged:
        core.backward_epochs = [epoch]


def _materialize_backward_projections_impl(
    trace: Any, events: list[Any], stream: Any = None
) -> None:
    """Rebuild backward projections from an already-snapshotted event list.

    A DETACHED stream (installed by pickle restore or ``Trace.fork()`` over an
    already-materialized projection) carries ``pass_index_base`` plus the
    cumulative baselines of the projection it extends. The rebuild is then
    stream-authoritative only for passes strictly above the base: pass records,
    per-op/per-param gradient records, and grad-fn records from at-or-below
    the base are preserved facts whose source events were dropped with the old
    stream by design (event streams never serialize). A live capture stream
    has base 0 and this function behaves exactly as a full scratch rebuild.
    """

    pass_index_base = int(getattr(stream, "pass_index_base", 0) or 0)
    state = _BackwardFoldState()
    if pass_index_base:
        state.total_gradient_memory = int(getattr(stream, "base_total_gradient_memory", 0) or 0)
        state.total_backward_memory = int(getattr(stream, "base_total_backward_memory", 0) or 0)
        state.saved_grad_labels = set(getattr(stream, "base_saved_grad_labels", ()) or ())
        # Pre-base passes are already materialized; marking them built keeps
        # ``num_backward_passes`` honest and stops any later fold from ever
        # re-building them.
        state.built_pass_indices.update(range(1, pass_index_base + 1))
    starts = state.starts
    ends = state.ends
    discovered = state.discovered
    latest_topology = state.latest_topology
    fired_events: list[GradFnFired] = []
    op_grad_passes = state.op_grad_passes
    op_grad_events: list[OpGradObserved] = []
    param_grad_events: list[ParamGradObserved] = []
    for event in events:
        if isinstance(event, BackwardPassStart):
            starts[event.pass_index] = event
        elif isinstance(event, BackwardPassEnd):
            ends[event.pass_index] = event
        elif isinstance(event, GradFnDiscovered):
            if event.object_id not in discovered:
                discovered[event.object_id] = event
            else:
                existing = discovered[event.object_id]
                if existing.op_label is None and event.op_label is not None:
                    existing = replace(existing, op_label=event.op_label)
                if existing.param_ref is None and event.param_ref is not None:
                    existing = replace(existing, param_ref=event.param_ref)
                if existing.created_in_pass is None and event.created_in_pass is not None:
                    existing = replace(existing, created_in_pass=event.created_in_pass)
                if existing.creator_object_id is None and event.creator_object_id is not None:
                    existing = replace(existing, creator_object_id=event.creator_object_id)
                discovered[event.object_id] = existing
            latest_topology[event.object_id] = event.topology
        elif isinstance(event, GradFnFired):
            fired_events.append(event)
        elif isinstance(event, OpGradObserved):
            op_grad_passes.add(event.pass_index)
            op_grad_events.append(event)
        elif isinstance(event, ParamGradObserved):
            state.param_grad_passes.add(event.pass_index)
            param_grad_events.append(event)

    # Preserved grad-fn records: on a detached stream, every existing record
    # whose discovery is NOT in this stream predates the detach (its events
    # were dropped with the old stream) and is kept verbatim. A record whose
    # object id IS rediscovered here rebuilds fresh from the stream. Label and
    # ordinal counters continue after the preserved records so merged labels
    # stay unique.
    preserved_grad_fn_logs: OrderedDict[int, GradFn] = OrderedDict()
    prior_grad_fn_logs: dict[int, GradFn] = {}
    if pass_index_base:
        prior_grad_fn_logs = dict(getattr(trace, "grad_fn_logs", {}) or {})
        for object_id, grad_fn_record in prior_grad_fn_logs.items():
            if object_id not in discovered:
                preserved_grad_fn_logs[object_id] = grad_fn_record
            # Ordinal counters continue after the PRE-BASE calls (preserved OR
            # rediscovered) so a new fire never reuses a pre-base call index.
            # Above-base calls are excluded: the stream still holds their fire
            # events and a later re-rebuild replays them onto this same seed.
            prior_calls = getattr(grad_fn_record.calls, "_dict", grad_fn_record.calls)
            base_ordinals = [
                call_ordinal
                for call_ordinal, prior_call in prior_calls.items()
                # A call with no recorded pass index counts as pre-base so its
                # ordinal is reserved (never reused), the conservative reading.
                if (prior_call.backward_pass_index or 0) <= pass_index_base
            ]
            if base_ordinals:
                state.per_object_ordinals[object_id] = max(
                    max(base_ordinals), state.per_object_ordinals.get(object_id, 0)
                )

    grad_fn_logs: OrderedDict[int, GradFn] = OrderedDict()
    type_counter: dict[str, int] = {}
    for preserved_record in preserved_grad_fn_logs.values():
        type_counter[preserved_record.type] = max(
            type_counter.get(preserved_record.type, 0), preserved_record.type_index
        )
    ordinal_offset = len(preserved_grad_fn_logs)
    object_to_label: dict[int, str] = {}
    for ordinal_index, (object_id, event) in enumerate(discovered.items(), start=ordinal_offset):
        grad_fn_type = _grad_fn_type_from_class_name(event.class_name)
        step_index = ordinal_index + 1
        type_index, step_index, label = _grad_fn_label_parts(
            trace,
            grad_fn_type,
            event.op_label,
            type_counter,
            step_index,
        )
        source_fields = cast(dict[str, Any], dict(event.source))
        modules: list[str] = []
        module_address = None
        module_membership_source = None
        op_module_address = _op_module_address(trace, event.op_label)
        param_module_address = _param_module_address(trace, event.param_ref)
        if op_module_address is not None:
            module_address = op_module_address
            modules = [module_address]
            module_membership_source = "paired"
        elif param_module_address is not None:
            module_address = param_module_address
            modules = [module_address]
            module_membership_source = "paired"
        grad_fn_record = GradFn(
            grad_fn_object_id=object_id,
            class_name=event.class_name,
            class_qualname=event.class_qualname,
            label=label,
            type=grad_fn_type,
            type_index=type_index,
            ordinal_index=ordinal_index,
            step_index=step_index,
            is_custom=event.is_custom,
            has_op=event.op_label is not None,
            op_label=event.op_label,
            order=None,
            origin_backward_pass=event.created_in_pass,
            creator_object_id=event.creator_object_id,
            modules=modules,
            module_address=module_address,
            module_membership_source=module_membership_source,
            next_grad_fn_ids=list(latest_topology.get(object_id, event.topology)),
            **source_fields,
        )
        grad_fn_record.source_trace = trace
        prior_record = prior_grad_fn_logs.get(object_id)
        if prior_record is not None:
            # A rediscovered node predating the detach keeps its at-or-below-
            # base calls verbatim (the SAME GradFnCall objects the preserved
            # pass records hold): the stream is authoritative only above the
            # base, and dropping these calls would strand the retained pass's
            # projection with calls its grad-fn record no longer knows.
            prior_calls = getattr(prior_record.calls, "_dict", prior_record.calls)
            for call_ordinal, prior_call in sorted(prior_calls.items()):
                if prior_call.backward_pass_index <= pass_index_base:
                    grad_fn_record.calls[call_ordinal] = prior_call
            if grad_fn_record.origin_backward_pass is None:
                grad_fn_record.origin_backward_pass = prior_record.origin_backward_pass
        grad_fn_logs[object_id] = grad_fn_record
        object_to_label[object_id] = label

    _assign_grad_fn_orders(grad_fn_logs)

    for object_id, grad_fn_record in grad_fn_logs.items():
        if grad_fn_record.creator_object_id is not None:
            grad_fn_record.differentiates = object_to_label.get(grad_fn_record.creator_object_id)

    if preserved_grad_fn_logs:
        merged_grad_fn_logs: OrderedDict[int, GradFn] = OrderedDict(preserved_grad_fn_logs)
        merged_grad_fn_logs.update(grad_fn_logs)
        grad_fn_logs = merged_grad_fn_logs
    trace.grad_fn_logs = grad_fn_logs
    trace.grad_fn_order = list(grad_fn_logs)

    pass_to_calls: dict[int, list[GradFnCall]] = {}
    _fold_fired_events(trace, state, fired_events, pass_to_calls)

    _sync_grad_fn_graph_relations(trace)
    _infer_grad_fn_module_membership(trace)
    _prefer_direct_pairing_source_for_paired_ops(trace)

    # Gradient records at or below the pass-index base are preserved facts
    # from before the stream detach; only the stream's own window rebuilds.
    for op in getattr(trace, "layer_list", []):
        if pass_index_base:
            op_records = op._slot("_grad_records") or []
            op._internal_set(
                "_grad_records",
                [record for record in op_records if record.backward_pass_index <= pass_index_base],
            )
        else:
            op._clear_gradient_records()
    _fold_op_grad_events(trace, state, op_grad_events)

    for param_log in getattr(trace, "param_logs", {}).values():
        if pass_index_base:
            param_log._grad_records = [
                record
                for record in param_log._grad_records
                if record.backward_pass_index <= pass_index_base
            ]
        else:
            param_log._clear_gradient_records()
    _fold_param_grad_events(trace, param_grad_events)

    backward_pass_logs: OrderedDict[int, BackwardPass] = OrderedDict()
    if pass_index_base:
        for preserved_pass_index, preserved_pass in getattr(
            trace, "backward_pass_logs", {}
        ).items():
            if preserved_pass_index <= pass_index_base:
                backward_pass_logs[preserved_pass_index] = preserved_pass
    pass_indices = sorted(
        set(starts) | set(ends) | set(pass_to_calls) | op_grad_passes | state.param_grad_passes
    )
    _build_backward_pass_records(trace, state, pass_indices, pass_to_calls, backward_pass_logs)
    trace.backward_pass_logs = backward_pass_logs
    _refresh_backward_pass_counters(trace, state)

    for grad_fn_record in trace.grad_fn_logs.values():
        if grad_fn_record.op is not None:
            grad_fn_record.op.grad_fn = grad_fn_record
            parent_layer = trace.layer_logs.get(grad_fn_record.op.layer_label)
            if parent_layer is not None:
                parent_layer.grad_fn = grad_fn_record
    for layer in getattr(trace, "layer_list", ()):
        grad_fn_object_id = getattr(layer, "grad_fn_object_id", None)
        if grad_fn_object_id in trace.grad_fn_logs:
            layer.grad_fn = trace.grad_fn_logs[grad_fn_object_id]
    for layer in getattr(trace, "layer_logs", {}).values():
        grad_fn_object_id = getattr(layer, "grad_fn_object_id", None)
        if grad_fn_object_id in trace.grad_fn_logs:
            layer.grad_fn = trace.grad_fn_logs[grad_fn_object_id]

    trace._backward_projection_fold_state = state


def _fold_backward_projection_tail(trace: Any, state: _BackwardFoldState, tail: list[Any]) -> None:
    """Fold a clean appended event tail into the existing projections.

    Only called after ``_backward_tail_is_foldable`` proved the tail contains
    no new grad-fn nodes, no retroactive merges, no topology changes, and no
    events for already-built passes, so the global order / graph-relation /
    module-inference passes provably do not need to re-run.
    """

    fired_events: list[GradFnFired] = []
    op_grad_events: list[OpGradObserved] = []
    param_grad_events: list[ParamGradObserved] = []
    new_pass_candidates: set[int] = set()
    for event in tail:
        if isinstance(event, BackwardPassStart):
            state.starts[event.pass_index] = event
            new_pass_candidates.add(event.pass_index)
        elif isinstance(event, BackwardPassEnd):
            state.ends[event.pass_index] = event
            new_pass_candidates.add(event.pass_index)
        elif isinstance(event, GradFnDiscovered):
            state.latest_topology[event.object_id] = event.topology
        elif isinstance(event, GradFnFired):
            fired_events.append(event)
        elif isinstance(event, OpGradObserved):
            state.op_grad_passes.add(event.pass_index)
            new_pass_candidates.add(event.pass_index)
            op_grad_events.append(event)
        elif isinstance(event, ParamGradObserved):
            state.param_grad_passes.add(event.pass_index)
            new_pass_candidates.add(event.pass_index)
            param_grad_events.append(event)

    pass_to_calls: dict[int, list[GradFnCall]] = {}
    _fold_fired_events(trace, state, fired_events, pass_to_calls)
    new_pass_candidates.update(pass_to_calls)
    _fold_op_grad_events(trace, state, op_grad_events)
    _fold_param_grad_events(trace, param_grad_events)
    new_pass_indices = sorted(new_pass_candidates - state.built_pass_indices)
    _build_backward_pass_records(
        trace, state, new_pass_indices, pass_to_calls, trace.backward_pass_logs
    )
    _refresh_backward_pass_counters(trace, state)


def _fold_fired_events(
    trace: Any,
    state: _BackwardFoldState,
    fired_events: list[GradFnFired],
    pass_to_calls: dict[int, list[GradFnCall]],
) -> None:
    """Fold grad-fn fire events into ``GradFn.calls`` and per-pass call lists."""

    for event in sorted(fired_events, key=lambda item: (item.pass_index, item.timestamp, item.seq)):
        grad_fn_record = trace.grad_fn_logs.get(event.object_id)
        if grad_fn_record is None:
            continue
        ordinal = state.per_object_ordinals.get(event.object_id, 0) + 1
        state.per_object_ordinals[event.object_id] = ordinal
        # tlspec v8 (L9 bump-time flip): the persisted stamps carry the
        # per-fire perf_counter pair, discriminated by the persisted
        # Trace.grad_fn_timing_provenance; an untimed fire carries
        # (None, None). The wall ``timestamp`` stays the ordering stamp.
        call = GradFnCall(
            call_index=ordinal,
            ordinal=ordinal,
            backward_pass_index=event.pass_index,
            label=grad_fn_record.label,
            grad_inputs=event.grad_input_refs,
            grad_outputs=event.grad_output_refs,
            intervention_fire_ref=event.intervention_fire_ref,
            timestamp=event.timestamp,
            _time_started=event.fire_started_monotonic,
            _time_finished=event.fire_finished_monotonic,
        )
        call.source_trace = trace
        grad_fn_record.calls[ordinal] = call
        pass_to_calls.setdefault(event.pass_index, []).append(call)


def _fold_op_grad_events(
    trace: Any,
    state: _BackwardFoldState,
    op_grad_events: list[OpGradObserved],
) -> None:
    """Fold op-gradient events into Op records and cumulative totals."""

    for event in sorted(
        op_grad_events, key=lambda item: (item.pass_index, item.timestamp, item.seq)
    ):
        event_label = _resolve_op_grad_event_label(trace, event.op_label)
        if event_label not in getattr(trace, "layer_dict_all_keys", {}):
            continue
        op = trace.layer_dict_all_keys[event_label]
        payload = event.payload_ref if isinstance(event.payload_ref, torch.Tensor) else None
        transformed_payload = event.transformed_payload_ref
        op._record_gradient(
            backward_pass_index=event.pass_index,
            grad=payload,
            transformed_grad=transformed_payload,
            shape=event.shape,
            dtype=event.dtype,
            memory=event.memory,
            timestamp=event.timestamp,
        )
        if payload is None and transformed_payload is None:
            continue
        op.has_grad = True
        if event.shape is not None:
            op.grad_shape = tuple(event.shape)
        if event.dtype is not None:
            op.grad_dtype = _torch_dtype_from_string(event.dtype)
        op.gradient_memory = Bytes(event.memory or 0)
        op._internal_set("grad", payload)
        op._internal_set("transformed_grad", transformed_payload)
        op.transformed_grad_shape = _shape_or_none(transformed_payload)
        op.transformed_grad_dtype = _dtype_or_none(transformed_payload)
        op.transformed_gradient_memory = _memory_or_none(transformed_payload)
        state.saved_grad_labels.add(op.layer_label)
        for payload_ref, payload_memory in _retained_grad_payload_refs(
            payload, transformed_payload, raw_memory=event.memory
        ):
            payload_id = id(payload_ref)
            if payload_id not in state.unique_payload_ids:
                state.unique_payload_ids.add(payload_id)
                state.total_backward_memory += payload_memory
        state.total_gradient_memory += int(event.memory or 0)
    # Assign a copy: the live hook path also mutates trace._saved_grad_labels
    # mid-pass, and every fold must overwrite those live writes with the
    # events-derived set exactly like the historical full rebuild did.
    trace._saved_grad_labels = set(state.saved_grad_labels)
    trace.saved_gradient_memory = Bytes(state.total_gradient_memory)
    trace.total_gradient_memory = Bytes(state.total_gradient_memory)
    trace.total_backward_memory = Bytes(state.total_backward_memory)


def _fold_param_grad_events(
    trace: Any,
    param_grad_events: list[ParamGradObserved],
) -> None:
    """Fold parameter-gradient events into per-Param accumulating records.

    The ``ParamGradObserved`` event spine is the single authoritative source
    for ``Param._grad_records``: the full rebuild clears every Param's records
    before calling this, and the incremental fold hands only strictly-newer
    passes (guaranteed by ``_backward_tail_is_foldable``), so folded and
    scratch-rebuilt records are identical.
    """

    param_logs = getattr(trace, "param_logs", None)
    if param_logs is None:
        return
    for event in sorted(
        param_grad_events, key=lambda item: (item.pass_index, item.timestamp, item.seq)
    ):
        if event.param_address not in param_logs:
            continue
        param_logs[event.param_address]._append_gradient_record(
            backward_pass_index=event.pass_index,
            grad=event.payload_ref,
            shape=event.shape,
            dtype=event.dtype,
            memory=event.memory,
            timestamp=event.timestamp,
        )


def _build_backward_pass_records(
    trace: Any,
    state: _BackwardFoldState,
    pass_indices: list[int],
    pass_to_calls: dict[int, list[GradFnCall]],
    backward_pass_logs: OrderedDict[int, BackwardPass],
) -> None:
    """Build BackwardPass records for ``pass_indices`` into ``backward_pass_logs``."""

    grad_fn_logs = trace.grad_fn_logs
    roots_by_pass = getattr(trace, "_backward_roots_by_pass", {})
    resolved_pass_orders = state.resolved_pass_orders
    for pass_index in pass_indices:
        start = state.starts.get(pass_index)
        end = state.ends.get(pass_index)
        root_grad_fn_ids = list(roots_by_pass.get(pass_index, ()))
        pass_order = (
            _pass_order_from_roots(root_grad_fn_ids, grad_fn_logs, resolved_pass_orders)
            if root_grad_fn_ids
            else start.order
            if start is not None
            else None
        )
        if pass_order is None and start is not None:
            pass_order = start.order
        grad_fn_calls = pass_to_calls.get(pass_index, [])
        max_call_order = _max_call_order(grad_fn_calls, grad_fn_logs)
        if max_call_order is not None and (pass_order is None or max_call_order > pass_order):
            pass_order = max_call_order
        order_attribution_coverage = (
            end.order_attribution_coverage
            if end is not None and end.order_attribution_coverage is not None
            else _order_attribution_coverage(grad_fn_calls, grad_fn_logs)
        )
        pass_record = BackwardPass(
            pass_index=pass_index,
            trigger=start.trigger if start is not None else "implicit",
            implicit=start.implicit if start is not None else True,
            outer_context=start.outer_context if start is not None else None,
            backward_call_context=start.call_context_ref if start is not None else None,
            root_grad_fn_ids=root_grad_fn_ids,
            root_meta=tuple(start.root_meta) if start is not None else (),
            root_grad_arguments=start.root_grad_arguments if start is not None else None,
            inputs_subset=tuple(start.inputs_subset) if start is not None else (),
            order=pass_order,
            origin_backward_pass=start.origin_backward_pass if start is not None else None,
            engine_flags=start.engine_flags if start is not None else None,
            save_grads_policy=start.save_grads_policy_repr if start is not None else None,
            duration=None if end is None or end.duration is None else Duration(end.duration),
            peak_memory=end.peak_memory if end is not None else None,
            status=end.status if end is not None else "ok",
            order_attribution_coverage=order_attribution_coverage,
            grad_fn_calls=grad_fn_calls,
        )
        pass_record.source_trace = trace
        backward_pass_logs[pass_index] = pass_record
        state.built_pass_indices.add(pass_index)
        if pass_order is not None:
            resolved_pass_orders[pass_index] = pass_order


def _refresh_backward_pass_counters(trace: Any, state: _BackwardFoldState) -> None:
    """Refresh pass-derived Trace counters after a full or incremental fold."""

    roots_by_pass = getattr(trace, "_backward_roots_by_pass", {})
    # ``_backward_roots_by_pass`` is session-only; a detached stream carries
    # the pre-detach root ids so preserved passes keep their roots listed.
    base_root_ids = tuple(
        getattr(getattr(trace, "_capture_events", None), "base_root_grad_fn_object_ids", ()) or ()
    )
    trace.backward_root_grad_fn_object_ids = [
        *base_root_ids,
        *(root_id for roots in roots_by_pass.values() for root_id in roots),
    ]
    trace.num_backward_passes = max(state.built_pass_indices, default=0)
    trace.has_backward_pass = bool(state.built_pass_indices)
    trace.num_saved_grad_fn_calls = len(trace.saved_grad_fn_calls)
    trace.num_saved_grad_fns = len(trace.saved_grad_fns)


def _torch_dtype_from_string(dtype_name: str) -> torch.dtype | str:
    """Return a ``torch.dtype`` for canonical dtype strings when possible."""

    if dtype_name.startswith("torch."):
        dtype_attr = dtype_name.removeprefix("torch.")
        dtype = torch_attr(dtype_attr)  # r47 secD_1: no lazy ``torch.__getattr__``
        if isinstance(dtype, torch.dtype):
            return dtype
    return dtype_name


def _retained_grad_payload_refs(
    raw_payload: torch.Tensor | None,
    transformed_payload: Any | None,
    *,
    raw_memory: int | None,
) -> list[tuple[Any, int]]:
    """Return retained gradient payload refs paired with their memory cost."""

    payloads: list[tuple[Any, int]] = []
    if raw_payload is not None:
        payloads.append((raw_payload, int(raw_memory or 0)))
    transformed_memory = _memory_or_none(transformed_payload)
    if transformed_payload is not None and transformed_memory is not None:
        payloads.append((transformed_payload, int(transformed_memory)))
    return payloads


def _resolve_live_grad_fn(trace_ref: Any, grad_fn_object_id: int) -> tuple[Any, Any] | None:
    """Resolve the live (trace, grad_fn record) target for one hook firing.

    A cleaned-up trace disarms its triggers but cannot remove hooks already
    registered on the user's live graph; firing must no-op rather than raise
    inside the user's autograd engine.
    """

    live_trace = trace_ref()
    if live_trace is None:
        return None
    if getattr(live_trace, "_tl_backward_triggers_disarmed", False):
        return None
    grad_fn_handle = getattr(live_trace, "grad_fn_logs", {}).get(grad_fn_object_id)
    if grad_fn_handle is None:
        return None
    return live_trace, grad_fn_handle


def _make_grad_fn_hook(  # noqa: PLR0913 -- one keyword per carrier the ONE registration path threads (aten LIFO, timing LIFO, FLIP-2 marker LIFO); packing them would hide which hook owns which stack
    trace: Any,
    grad_fn_object_id: int,
    *,
    is_accumulate_grad: bool = False,
    aten_marker_tokens: list[Any] | None = None,
    fire_start_stamps: list[tuple[int, float]] | None = None,
    gradfn_marker_tokens: list[Any] | None = None,
) -> Callable[..., tuple[torch.Tensor | None, ...] | None]:
    """Build a runtime hook for one autograd grad_fn_handle.

    Parameters
    ----------
    trace:
        Trace whose flat backward fields should receive runtime data.
    grad_fn_object_id:
        ``id()`` of the hooked grad_fn_handle.
    is_accumulate_grad:
        Whether this hook is attached to an AccumulateGrad node.
    aten_marker_tokens:
        LIFO marker tokens installed by the private ATen GradFn prehook.
    fire_start_stamps:
        Keyed per-node start-stamp LIFO fed by the timing prehook. ``None``
        when timing registration failed for this node (untimed fires).
    gradfn_marker_tokens:
        Per-node open ``record_function`` marker LIFO fed by the timing
        prehook's FLIP-2 sink; the posthook pops it at the finish-stamp line
        so the marker span equals the recorded host span by construction.

    Returns
    -------
    Callable[..., tuple[torch.Tensor | None, ...] | None]
        Hook compatible with ``grad_fn_handle.register_hook``.
    """

    trace_ref = weakref.ref(trace)

    def hook(*hook_args: Any) -> tuple[torch.Tensor | None, ...] | None:
        """Record one autograd grad_fn hook firing and apply live interventions."""
        # Finish stamp FIRST, before any logging work, so TorchLens overhead
        # stays outside the measured span (L9 memo 1.3). Same clock as the
        # prehook stamp -- perf_counter, never the wall `timestamp`.
        fire_finished_monotonic = time.perf_counter()
        # FLIP-2 (F27): the node marker closes AT the finish stamp, and
        # TorchLens's OWN logging below runs inside a TYPED internal bracket
        # (TN-D18) so the complement bucket can never publish our
        # gradient-saving work under torch's name.
        if gradfn_marker_tokens:
            _exit_gradfn_marker(gradfn_marker_tokens.pop())
        internal_bracket = _enter_internal_gradfn_bracket(grad_fn_object_id)
        try:
            return _hook_body(fire_finished_monotonic, *hook_args)
        finally:
            _exit_gradfn_marker(internal_bracket)

    def _hook_body(
        fire_finished_monotonic: float, *hook_args: Any
    ) -> tuple[torch.Tensor | None, ...] | None:
        """The posthook logging body (runs inside the typed internal bracket)."""
        if aten_marker_tokens:
            from ._aten_capture import _end_backward_grad_fn

            _end_backward_grad_fn(aten_marker_tokens.pop())
        resolved = _resolve_live_grad_fn(trace_ref, grad_fn_object_id)
        if resolved is None:
            return None
        live_trace, grad_fn_handle = resolved
        grad_inputs = hook_args[0] if len(hook_args) >= 1 else None
        grad_outputs = hook_args[1] if len(hook_args) >= 2 else None
        layer_label = grad_fn_handle.op.layer_label if grad_fn_handle.has_op else None
        stored_grad_inputs = grad_inputs
        stored_grad_outputs = grad_outputs
        if not _selected_for_grad_save(live_trace, layer_label):
            stored_grad_inputs = None
            stored_grad_outputs = None
        with pause_logging():
            grad_fn_handle._log_call(stored_grad_inputs, stored_grad_outputs, time.time())
        call_index = len(grad_fn_handle.calls)
        logged_call = grad_fn_handle.calls[-1]
        stored_grad_inputs = logged_call.grad_inputs
        stored_grad_outputs = logged_call.grad_outputs
        # Pair the fire span from the keyed LIFO: match -> timed fire; empty,
        # key mismatch, or failed timing registration -> (None, None), never a
        # stale stamp or a cross-fire pair.
        fire_started_monotonic: float | None = None
        if fire_start_stamps is not None:
            fire_started_monotonic = _pop_matching_fire_start(fire_start_stamps, call_index)
        paired_finished_monotonic = (
            fire_finished_monotonic if fire_started_monotonic is not None else None
        )
        events = _ensure_backward_event_stream(live_trace)
        event_timestamp = time.time()
        pass_index = int(
            getattr(
                live_trace,
                "_active_backward_pass_index",
                getattr(live_trace, "num_backward_passes", 0) + 1,
            )
        )
        _set_live_grad_fn_call_backward_pass_index(grad_fn_handle, call_index, pass_index)
        if is_accumulate_grad:
            fire_records = _pop_pending_accumulate_grad_records(
                live_trace, grad_fn_object_id, call_index
            )
            fire_ref = _intervention_fire_ref(fire_records)
            _set_live_grad_fn_call_fire_ref(grad_fn_handle, call_index, fire_ref)
            _record_higher_order_terminals_from_tuple(
                live_trace,
                tuple(grad_inputs or ()),
                creator_object_id=grad_fn_object_id,
                pass_index=pass_index,
            )
            events.append_backward(
                GradFnFired(
                    object_id=grad_fn_object_id,
                    pass_index=pass_index,
                    grad_input_refs=stored_grad_inputs,
                    grad_output_refs=stored_grad_outputs,
                    intervention_fire_ref=fire_ref,
                    timestamp=event_timestamp,
                    fire_started_monotonic=fire_started_monotonic,
                    fire_finished_monotonic=paired_finished_monotonic,
                )
            )
            param_address = getattr(live_trace, "_grad_fn_param_refs_by_object_id", {}).get(
                grad_fn_object_id
            )
            param_log = (
                live_trace.param_logs[param_address]
                if param_address is not None and param_address in live_trace.param_logs
                else None
            )
            picked = _first_tensor_from_hook_args(hook_args)
            if param_log is not None and param_address is not None and picked is not None:
                # Event-only emission: the projection fold is the single writer
                # of Param._grad_records, so a forced scratch rebuild from the
                # event spine reconstructs the exact same records.
                observed_grad = picked[2]
                memory = int(observed_grad.nelement() * observed_grad.element_size())
                saved_grad = None
                with pause_logging():
                    if _should_save_grad_payload(live_trace, param_address):
                        save_mode = _trace_grad_save_mode(live_trace)
                        target_device = (
                            torch.device("cpu")
                            if save_mode == "cpu_async"
                            else observed_grad.device
                        )
                        budget = getattr(live_trace, "_save_budget_accountant", None)
                        reservation = (
                            None
                            if budget is None
                            else budget.admit(param_address, target_device, memory)
                        )
                        saved_grad = _copy_grad_payload(observed_grad, save_mode=save_mode)
                        if budget is not None:
                            budget.commit(reservation, (saved_grad,))
                events.append_backward(
                    ParamGradObserved(
                        param_address=param_address,
                        pass_index=pass_index,
                        payload_ref=saved_grad,
                        shape=tuple(observed_grad.shape),
                        dtype=str(observed_grad.dtype),
                        memory=memory,
                        timestamp=event_timestamp,
                    )
                )
            return None
        from ...intervention.runtime import _apply_live_backward_hooks

        result, fire_records = _apply_live_backward_hooks(
            grad_inputs, grad_outputs, grad_fn_handle, call_index
        )
        fire_ref = _intervention_fire_ref(fire_records)
        _set_live_grad_fn_call_fire_ref(grad_fn_handle, call_index, fire_ref)
        terminal_grad_inputs = result if result is not None else grad_inputs
        _record_higher_order_terminals_from_tuple(
            live_trace,
            tuple(terminal_grad_inputs or ()),
            creator_object_id=grad_fn_object_id,
            pass_index=pass_index,
        )
        events.append_backward(
            GradFnFired(
                object_id=grad_fn_object_id,
                pass_index=pass_index,
                grad_input_refs=stored_grad_inputs,
                grad_output_refs=stored_grad_outputs,
                intervention_fire_ref=fire_ref,
                timestamp=event_timestamp,
                fire_started_monotonic=fire_started_monotonic,
                fire_finished_monotonic=paired_finished_monotonic,
            )
        )
        return result

    return hook


def _make_aten_grad_fn_prehook(
    trace: Any,
    grad_fn_object_id: int,
    marker_tokens: list[Any],
) -> Callable[..., None]:
    """Build a prehook that brackets one GradFn execution for ATen attribution.

    Parameters
    ----------
    trace
        Trace receiving primitive events.
    grad_fn_object_id
        Captured GradFn object identity.
    marker_tokens
        Per-node LIFO token stack shared with the posthook.

    Returns
    -------
    Callable[..., None]
        Framework-compatible prehook that leaves gradient inputs unchanged.
    """

    trace_ref = weakref.ref(trace)

    def prehook(*hook_args: Any) -> None:
        """Publish the predicted GradFn-call witness immediately before execution."""

        del hook_args
        live_trace = trace_ref()
        if live_trace is None:
            return None
        grad_fn_record = getattr(live_trace, "grad_fn_logs", {}).get(grad_fn_object_id)
        if grad_fn_record is None:
            return None
        pass_index = int(getattr(live_trace, "_active_backward_pass_index", 0) or 0)
        call_index = len(grad_fn_record.calls) + 1
        from ._aten_capture import _begin_backward_grad_fn

        marker_tokens.append(_begin_backward_grad_fn(grad_fn_object_id, call_index, pass_index))
        return None

    return prehook


def _make_grad_fn_prehook(
    trace: Any,
    grad_fn_object_id: int,
) -> Callable[..., tuple[torch.Tensor | None, ...] | None]:
    """Build an AccumulateGrad prehook for mutating incoming gradients.

    Parameters
    ----------
    trace:
        Trace whose backward hook plan should dispatch.
    grad_fn_object_id:
        ``id()`` of the hooked grad_fn_handle.

    Returns
    -------
    Callable[..., tuple[torch.Tensor | None, ...] | None]
        Hook compatible with ``grad_fn_handle.register_prehook``.
    """

    trace_ref = weakref.ref(trace)

    def prehook(*hook_args: Any) -> tuple[torch.Tensor | None, ...] | None:
        """Apply pending AccumulateGrad prehook interventions for one call."""
        live_trace = trace_ref()
        if live_trace is None:
            return None
        grad_fn_handle = live_trace.grad_fn_logs.get(grad_fn_object_id)
        if grad_fn_handle is None:
            return None
        grad_inputs = hook_args[0] if len(hook_args) >= 1 else ()
        call_index = len(grad_fn_handle.calls) + 1
        from ...intervention.runtime import _apply_live_backward_prehooks

        result, fire_records = _apply_live_backward_prehooks(
            grad_inputs, grad_fn_handle, call_index
        )
        if fire_records:
            _store_pending_accumulate_grad_records(
                live_trace, grad_fn_object_id, call_index, fire_records
            )
        return result

    return prehook


def _intervention_fire_ref(records: tuple[Any, ...]) -> Any | None:
    """Return the compact event-side reference for backward fire records.

    Parameters
    ----------
    records:
        Fire records produced for one backward callback.

    Returns
    -------
    Any | None
        ``None`` for no fires, the single record for one fire, or a tuple for
        multiple helpers fired at the same callback.
    """

    if not records:
        return None
    if len(records) == 1:
        return records[0]
    return records


def _set_live_grad_fn_call_fire_ref(
    grad_fn_handle: Any,
    call_index: int,
    fire_ref: Any | None,
) -> None:
    """Set the fire reference on the just-logged runtime GradFnCall.

    Parameters
    ----------
    grad_fn_handle:
        Runtime GradFn record whose call accessor was just appended.
    call_index:
        One-based callback index.
    fire_ref:
        FireRecord, tuple of records, or ``None``.

    Returns
    -------
    None
        Mutates the runtime call record when a fire reference exists.
    """

    if fire_ref is None:
        return
    calls = getattr(grad_fn_handle, "calls", None)
    call = getattr(calls, "_dict", {}).get(call_index)
    if call is not None:
        call.intervention_fire_ref = fire_ref


def _set_live_grad_fn_call_backward_pass_index(
    grad_fn_handle: Any,
    call_index: int,
    pass_index: int,
) -> None:
    """Set the active backward pass index on the just-logged GradFnCall.

    Parameters
    ----------
    grad_fn_handle:
        Runtime GradFn record whose call accessor was just appended.
    call_index:
        One-based callback index.
    pass_index:
        One-based active backward pass index.

    Returns
    -------
    None
        Mutates the runtime call record when present.
    """

    calls = getattr(grad_fn_handle, "calls", None)
    call = getattr(calls, "_dict", {}).get(call_index)
    if call is not None:
        call.backward_pass_index = pass_index


def _pending_accumulate_grad_record_key(
    grad_fn_object_id: int,
    call_index: int,
) -> tuple[int, int]:
    """Return the trace-local key for pending AccumulateGrad prehook records.

    Parameters
    ----------
    grad_fn_object_id:
        Hooked grad_fn object id.
    call_index:
        One-based callback index.

    Returns
    -------
    tuple[int, int]
        Stable pending-record key.
    """

    return grad_fn_object_id, call_index


def _store_pending_accumulate_grad_records(
    trace: Any,
    grad_fn_object_id: int,
    call_index: int,
    records: tuple[Any, ...],
) -> None:
    """Store AccumulateGrad prehook fire records until the posthook logs.

    Parameters
    ----------
    trace:
        Active trace.
    grad_fn_object_id:
        Hooked grad_fn object id.
    call_index:
        One-based callback index.
    records:
        Fire records emitted by the prehook.

    Returns
    -------
    None
        Mutates a private trace-local queue.
    """

    pending = trace.__dict__.setdefault("_tl_pending_accumulate_grad_fire_records", {})
    key = _pending_accumulate_grad_record_key(grad_fn_object_id, call_index)
    pending[key] = list(records)


def _pop_pending_accumulate_grad_records(
    trace: Any,
    grad_fn_object_id: int,
    call_index: int,
) -> tuple[Any, ...]:
    """Pop AccumulateGrad prehook fire records for the matching posthook.

    Parameters
    ----------
    trace:
        Active trace.
    grad_fn_object_id:
        Hooked grad_fn object id.
    call_index:
        One-based callback index.

    Returns
    -------
    tuple[Any, ...]
        Pending records for this callback, if any.
    """

    pending = trace.__dict__.get("_tl_pending_accumulate_grad_fire_records")
    if not pending:
        return ()
    key = _pending_accumulate_grad_record_key(grad_fn_object_id, call_index)
    records = tuple(pending.pop(key, ()))
    if not pending:
        trace.__dict__.pop("_tl_pending_accumulate_grad_fire_records", None)
    return records


def _clear_pending_accumulate_grad_records(trace: Any) -> None:
    """Drop unpaired AccumulateGrad prehook records at backward pass teardown.

    Parameters
    ----------
    trace:
        Active trace whose pending prehook records should be cleared.

    Returns
    -------
    None
        Removes the private trace-local pending queue if present.
    """

    trace.__dict__.pop("_tl_pending_accumulate_grad_fire_records", None)


def _record_higher_order_terminals_from_tuple(
    trace: Any,
    grad_values: tuple[Any, ...],
    *,
    creator_object_id: int,
    pass_index: int,
) -> None:
    """Register higher-order terminals found in a gradient tuple.

    Parameters
    ----------
    trace:
        Active trace.
    grad_values:
        Gradient tuple to inspect after live intervention mutation.
    creator_object_id:
        Backward grad_fn id that produced the tuple.
    pass_index:
        Active backward pass index.

    Returns
    -------
    None
        Mutates trace higher-order terminal state.
    """

    for grad_value in grad_values:
        if isinstance(grad_value, torch.Tensor) and grad_value.grad_fn is not None:
            _record_higher_order_terminal(
                trace,
                grad_value.grad_fn,
                creator_object_id=creator_object_id,
                pass_index=pass_index,
            )


def _memory_snapshot(device: torch.device) -> tuple[str, int]:
    """Return backend name and current allocated memory for a device.

    Parameters
    ----------
    device:
        Device associated with the backward loss tensor.

    Returns
    -------
    tuple[str, int]
        Backend label and memory snapshot in bytes.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        return "cuda", int(torch.cuda.max_memory_allocated(device))
    if device.type == "mps" and hasattr(torch, "mps"):
        return "mps", int(torch.mps.current_allocated_memory())
    try:
        import psutil
    except ImportError:
        return "cpu", 0
    return "cpu", int(psutil.Process().memory_info().rss)


def _peak_memory_baseline(device: torch.device) -> tuple[str, int]:
    """Return the pre-pass memory baseline WITHOUT resetting peak tracking.

    R36-2: this used to call ``torch.cuda.reset_peak_memory_stats`` on every
    ``log_backward``, clobbering the caller's process-wide high-water counter.
    The baseline is now the pre-existing ``max_memory_allocated`` snapshot:
    a backward that pushes a new device peak reports the exact excess over
    the prior high-water mark, and one that stays under it honestly reads
    ``0`` (the same contract as the forward bracket and the CPU/MPS deltas).

    Parameters
    ----------
    device:
        Device associated with the backward loss tensor.

    Returns
    -------
    tuple[str, int]
        Backend label and starting memory snapshot in bytes.
    """
    return _memory_snapshot(device)


def _layer_grad_fn_pairing_rank(layer: Any) -> int:
    """Return the preference rank for a forward op grad-fn pairing candidate.

    Parameters
    ----------
    layer:
        Forward Op/Layer candidate sharing a grad-fn object id.

    Returns
    -------
    int
        Lower rank is preferred; compute ops outrank boundary sentinels.
    """

    is_compute = getattr(layer, "is_compute_op", False) or getattr(
        layer,
        "is_compute_layer",
        False,
    )
    return 0 if is_compute else 1


def _layer_by_grad_fn_id(trace: Any) -> dict[int, str]:
    """Build a mapping from forward grad_fn_handle identity to final layer label.

    Parameters
    ----------
    trace:
        Trace whose layers were captured during the forward pass.

    Returns
    -------
    dict[int, str]
        Mapping from ``id(grad_fn_handle)`` to final layer label.
    """
    mapping: dict[int, str] = {}
    ranks: dict[int, int] = {}
    for layer in trace.layer_list:
        grad_fn_object_id = getattr(layer, "grad_fn_object_id", None)
        if grad_fn_object_id is None:
            continue
        candidate_rank = _layer_grad_fn_pairing_rank(layer)
        current_rank = ranks.get(grad_fn_object_id)
        if current_rank is not None and current_rank <= candidate_rank:
            continue
        mapping[grad_fn_object_id] = layer.layer_label
        ranks[grad_fn_object_id] = candidate_rank
    return mapping


def _walk_and_hook_backward_graph(
    trace: Any,
    loss: torch.Tensor,
    handles: list[Any] | None = None,
) -> list[Any]:
    """Walk ``loss.grad_fn`` and register hooks on every reachable grad_fn_handle.

    Parameters
    ----------
    trace:
        Trace that owns the flat backward fields.
    loss:
        Scalar or tensor loss whose backward graph should be captured.
    handles:
        Caller-owned hook-handle list. Passing this list lets the caller
        remove partially registered hooks when graph walking is interrupted.

    Returns
    -------
    list[Any]
        Hook handles to remove after backward finishes.
    """
    if loss.grad_fn is None:
        raise ValueError("log_backward requires a loss tensor with a grad_fn_handle.")

    layer_lookup = _layer_by_grad_fn_id(trace)
    queue: deque[Any] = deque([loss.grad_fn])
    seen: set[int] = set()
    if handles is None:
        handles = []
    type_counter: dict[str, int] = {}
    source_metadata_by_class: dict[type[Any], dict[str, Any]] = {}
    # Keep strong refs to every discovered grad_fn_handle for the trace's lifetime so
    # Python cannot recycle their memory addresses. ``id()`` is used as the
    # primary key for ``grad_fn_logs`` and ``next_grad_fn_ids``; if leaf nodes
    # like AccumulateGrad were gc'd, their ids could be reused by later-created
    # grad_fns (e.g. the output clone wrapper), creating phantom cycles.
    strong_refs = trace.__dict__.setdefault("_backward_gradfn_refs", [])
    strong_refs.append(loss.grad_fn)
    root_grad_fn_id = id(loss.grad_fn)
    if root_grad_fn_id not in trace.backward_root_grad_fn_object_ids:
        trace.backward_root_grad_fn_object_ids.append(root_grad_fn_id)
    pass_index = int(
        getattr(trace, "_active_backward_pass_index", getattr(trace, "num_backward_passes", 0) + 1)
    )
    roots_by_pass = trace.__dict__.setdefault("_backward_roots_by_pass", {})
    roots_by_pass.setdefault(pass_index, [])
    if root_grad_fn_id not in roots_by_pass[pass_index]:
        roots_by_pass[pass_index].append(root_grad_fn_id)
    trace.has_backward_pass = True

    while queue:
        grad_fn_handle = queue.popleft()
        grad_fn_object_id = id(grad_fn_handle)
        if grad_fn_object_id in seen:
            continue
        seen.add(grad_fn_object_id)
        next_grad_fns = list(_iter_next_grad_fns(grad_fn_handle))
        strong_refs.extend(next_grad_fns)
        queue.extend(next_grad_fns)
        layer_label = layer_lookup.get(grad_fn_object_id)
        grad_fn_record = trace.grad_fn_logs.get(grad_fn_object_id)
        if grad_fn_record is None:
            grad_fn_type = _normalize_grad_fn_type(grad_fn_handle)
            step_index = len(trace.grad_fn_order) + 1
            type_index, step_index, label = _grad_fn_label_parts(
                trace,
                grad_fn_type,
                layer_label,
                type_counter,
                step_index,
            )
            grad_fn_cls = type(grad_fn_handle)
            source_metadata = source_metadata_by_class.get(grad_fn_cls)
            if source_metadata is None:
                source_metadata = _grad_fn_source_metadata(grad_fn_cls)
                source_metadata_by_class[grad_fn_cls] = source_metadata
            grad_fn_record = GradFn(
                grad_fn_object_id=grad_fn_object_id,
                class_name=grad_fn_cls.__name__,
                class_qualname=f"{grad_fn_cls.__module__}.{grad_fn_cls.__qualname__}",
                label=label,
                type=grad_fn_type,
                type_index=type_index,
                ordinal_index=len(trace.grad_fn_order),
                step_index=step_index,
                is_custom=_grad_fn_is_custom(grad_fn_handle),
                has_op=layer_label is not None,
                op_label=layer_label,
                next_grad_fn_ids=[id(next_fn) for next_fn in next_grad_fns],
                **source_metadata,
            )
            grad_fn_record.source_trace = trace
            trace.grad_fn_logs[grad_fn_object_id] = grad_fn_record
            trace.grad_fn_order.append(grad_fn_object_id)
        else:
            grad_fn_record.next_grad_fn_ids = [id(next_fn) for next_fn in next_grad_fns]
            grad_fn_record.source_trace = trace
        is_accumulate_grad = _is_accumulate_grad(grad_fn_handle)
        if is_accumulate_grad:
            param = getattr(grad_fn_handle, "variable", None)
            param_address = trace._param_log_by_pid.get(id(param)) if param is not None else None
            if param_address is not None:
                trace._grad_fn_param_refs[grad_fn_record.label] = param_address
                trace.__dict__.setdefault("_grad_fn_param_refs_by_object_id", {})[
                    grad_fn_object_id
                ] = param_address
        source_fields = {
            key: getattr(grad_fn_record, key)
            for key in (
                "class_source_file",
                "class_source_line",
                "init_source_file",
                "init_source_line",
                "forward_source_file",
                "forward_source_line",
                "backward_source_file",
                "backward_source_line",
            )
        }
        _ensure_backward_event_stream(trace).append_backward(
            GradFnDiscovered(
                object_id=grad_fn_object_id,
                class_name=grad_fn_record.class_name,
                class_qualname=grad_fn_record.class_qualname,
                is_custom=grad_fn_record.is_custom,
                op_label=layer_label,
                param_ref=trace._grad_fn_param_refs.get(grad_fn_record.label),
                created_in_pass=None,
                creator_object_id=None,
                source=source_fields,
                topology=tuple(id(next_fn) for next_fn in next_grad_fns),
            )
        )
        # D5 reentrant sentinel: the discovery stream affirmatively witnesses
        # reentrant checkpointing (token-free by definition, memo 2.3).
        _flag_reentrant_checkpoint_if_sentinel(trace, grad_fn_record.class_qualname)
        if layer_label is not None:
            layer = trace.layer_dict_all_keys[layer_label]
            layer.grad_fn = grad_fn_record
            parent_layer = trace.layer_logs.get(layer.layer_label)
            if parent_layer is not None:
                parent_layer.grad_fn = grad_fn_record
        try:
            aten_marker_tokens: list[Any] = []
            fire_start_stamps = _fire_timing_stamp_list(trace, grad_fn_object_id)
            gradfn_marker_tokens = _gradfn_marker_list(trace, grad_fn_object_id)
            with pause_logging():
                handles.append(
                    grad_fn_handle.register_hook(
                        _make_grad_fn_hook(
                            trace,
                            grad_fn_object_id,
                            is_accumulate_grad=is_accumulate_grad,
                            aten_marker_tokens=aten_marker_tokens,
                            fire_start_stamps=fire_start_stamps,
                            gradfn_marker_tokens=gradfn_marker_tokens,
                        )
                    )
                )
            if is_accumulate_grad:
                handles.append(
                    grad_fn_handle.register_prehook(_make_grad_fn_prehook(trace, grad_fn_object_id))
                )
            # Timing prehook registers BEFORE the aten marker prehook so the
            # measured span covers the whole fire including its aten
            # dispatches; a registration failure degrades this node to
            # untimed fires only (own try/except inside the helper).
            timing_handle = _register_fire_timing_prehook(
                trace, grad_fn_handle, grad_fn_object_id, fire_start_stamps, gradfn_marker_tokens
            )
            if timing_handle is not None:
                handles.append(timing_handle)
                if getattr(trace, "grad_fn_timing_provenance", None) in (None, "unmeasured"):
                    trace.grad_fn_timing_provenance = "perf_counter"
            event_stream = _ensure_backward_event_stream(trace)
            if getattr(event_stream, "aten_recording_enabled", False):
                handles.append(
                    grad_fn_handle.register_prehook(
                        _make_aten_grad_fn_prehook(
                            trace,
                            grad_fn_object_id,
                            aten_marker_tokens,
                        )
                    )
                )
        except RuntimeError as exc:
            # A node the walk discovered but could not observe is a typed
            # journal fact, never a silent skip: validation fails closed on
            # every reason that is not a proven framework-contract exclusion.
            _ensure_backward_event_stream(trace).append_backward(
                BackwardCoverageGap(
                    pass_index=int(getattr(trace, "_active_backward_pass_index", 0) or 0),
                    object_id=grad_fn_object_id,
                    class_qualname=grad_fn_record.class_qualname,
                    reason="registration_error",
                    detail=str(exc)[:200],
                    timestamp=time.time(),
                )
            )
            continue
    _sync_grad_fn_graph_relations(trace)
    return handles


def _record_higher_order_terminal(
    trace: Any,
    grad_fn_handle: Any,
    *,
    creator_object_id: int,
    pass_index: int,
) -> None:
    """Queue a terminal grad-fn created by a differentiable backward op."""

    _strong_grad_fn_refs(trace).append(grad_fn_handle)
    terminals = trace.__dict__.setdefault("_higher_order_grad_fn_terminals", [])
    terminals.append((grad_fn_handle, creator_object_id, pass_index))


def _emit_discovered_grad_fn(
    trace: Any,
    grad_fn_handle: Any,
    *,
    created_in_pass: int | None,
    creator_object_id: int | None,
    type_counter: dict[str, int],
) -> None:
    """Append a discovery event and runtime record for one grad-fn object.

    ``type_counter`` is rewalk-scoped scratch owned by the caller
    (grind-r6 b5 R45: parking it on ``trace.__dict__`` left an undeclared
    private field on the Trace that broke ``tl.save`` after two
    differentiable grad passes -- ``_backward_grad_fn_type_counter``
    missing from ``PORTABLE_STATE_SPEC``).
    """

    grad_fn_object_id = id(grad_fn_handle)
    next_grad_fns = list(_iter_next_grad_fns(grad_fn_handle))
    _strong_grad_fn_refs(trace).extend(next_grad_fns)
    grad_fn_record = trace.grad_fn_logs.get(grad_fn_object_id)
    if grad_fn_record is None:
        grad_fn_type = _normalize_grad_fn_type(grad_fn_handle)
        step_index = len(trace.grad_fn_order) + 1
        type_index, step_index, label = _grad_fn_label_parts(
            trace,
            grad_fn_type,
            None,
            type_counter,
            step_index,
        )
        grad_fn_cls = type(grad_fn_handle)
        grad_fn_record = GradFn(
            grad_fn_object_id=grad_fn_object_id,
            class_name=grad_fn_cls.__name__,
            class_qualname=f"{grad_fn_cls.__module__}.{grad_fn_cls.__qualname__}",
            label=label,
            type=grad_fn_type,
            type_index=type_index,
            ordinal_index=len(trace.grad_fn_order),
            step_index=step_index,
            is_custom=_grad_fn_is_custom(grad_fn_handle),
            has_op=False,
            op_label=None,
            order=_resolved_order_for_creator(creator_object_id, trace.grad_fn_logs),
            origin_backward_pass=created_in_pass,
            creator_object_id=creator_object_id,
            next_grad_fn_ids=[id(next_fn) for next_fn in next_grad_fns],
            **_grad_fn_source_metadata(grad_fn_cls),
        )
        grad_fn_record.source_trace = trace
        trace.grad_fn_logs[grad_fn_object_id] = grad_fn_record
        trace.grad_fn_order.append(grad_fn_object_id)
    else:
        grad_fn_record.next_grad_fn_ids = [id(next_fn) for next_fn in next_grad_fns]
        grad_fn_record.source_trace = trace

    source_fields = {
        key: getattr(grad_fn_record, key)
        for key in (
            "class_source_file",
            "class_source_line",
            "init_source_file",
            "init_source_line",
            "forward_source_file",
            "forward_source_line",
            "backward_source_file",
            "backward_source_line",
        )
    }
    _ensure_backward_event_stream(trace).append_backward(
        GradFnDiscovered(
            object_id=grad_fn_object_id,
            class_name=grad_fn_record.class_name,
            class_qualname=grad_fn_record.class_qualname,
            is_custom=grad_fn_record.is_custom,
            op_label=None,
            param_ref=None,
            created_in_pass=created_in_pass,
            creator_object_id=creator_object_id,
            source=source_fields,
            topology=tuple(id(next_fn) for next_fn in next_grad_fns),
        )
    )
    _flag_reentrant_checkpoint_if_sentinel(trace, grad_fn_record.class_qualname)


def _rewalk_higher_order_grad_fns(trace: Any) -> None:
    """Discover grad-fn nodes created by a differentiable backward pass."""

    terminals = list(trace.__dict__.pop("_higher_order_grad_fn_terminals", ()))
    if not terminals:
        return
    existing_ids = set(trace.grad_fn_logs)

    # Rewalk-scoped label allocation state, seeded from every existing record
    # so a fresh discovery can never collide with an already-assigned
    # ``type_index`` (same seeding idiom as the projection rebuild path).
    # Holding this on the trace between rewalks left an undeclared private
    # field that broke tl.save (grind-r6 b5 R45).
    type_counter: dict[str, int] = {}
    for existing_record in trace.grad_fn_logs.values():
        type_counter[existing_record.type] = max(
            type_counter.get(existing_record.type, 0), existing_record.type_index
        )

    discovered_ids: set[int] = set()
    for terminal, creator_object_id, pass_index in terminals:
        queue: deque[Any] = deque([terminal])
        while queue:
            grad_fn_handle = queue.popleft()
            grad_fn_object_id = id(grad_fn_handle)
            if grad_fn_object_id in existing_ids or grad_fn_object_id in discovered_ids:
                continue
            discovered_ids.add(grad_fn_object_id)
            _emit_discovered_grad_fn(
                trace,
                grad_fn_handle,
                created_in_pass=pass_index,
                creator_object_id=creator_object_id,
                type_counter=type_counter,
            )
            queue.extend(_iter_next_grad_fns(grad_fn_handle))
    _sync_grad_fn_graph_relations(trace)


def _is_static_truthy_keep_grad(value: Any) -> bool:
    """Return whether a gradient capture decision is statically keep-grad true."""

    from ...fastlog.types import CaptureSpec

    return value is True or (isinstance(value, CaptureSpec) and value.keep_grad)


def _selected_fastlog_forward_labels(recording: Any) -> set[str]:
    """Return labels for predicate-selected forward records."""

    from ...capture.predicates import _evaluate_keep_op
    from ...ir.predicate import RetroactiveCaptureDecision

    recording._ensure_records()  # noqa: SLF001
    labels: set[str] = set()
    for record in recording.records:
        labels.add(record.ctx.label)
        if (
            record.ctx.kind == "op"
            and record.ctx.layer_type is not None
            and record.ctx.type_index is not None
        ):
            labels.add(f"{record.ctx.layer_type}_{record.ctx.type_index}")
        if record.ctx.raw_label is not None:
            labels.add(record.ctx.raw_label)
    recording_state = getattr(recording, "_recording_state", None)
    if recording_state is None:
        return labels
    for ctx in recording_state.all_contexts:
        spec = _evaluate_keep_op(ctx, recording_state.options)
        if isinstance(spec, RetroactiveCaptureDecision):
            continue
        if not spec.save_out and not spec.save_metadata:
            continue
        labels.add(ctx.label)
        if ctx.kind == "op" and ctx.layer_type is not None and ctx.type_index is not None:
            labels.add(f"{ctx.layer_type}_{ctx.type_index}")
        if ctx.raw_label is not None:
            labels.add(ctx.raw_label)
    return labels


def _first_tensor_from_hook_args(
    hook_args: tuple[Any, ...],
) -> tuple[str, int | None, torch.Tensor] | None:
    """Pick the first tensor gradient from grad output or input hook arguments."""

    grad_outputs = hook_args[1] if len(hook_args) >= 2 else ()
    if grad_outputs is not None:
        for index, grad in enumerate(grad_outputs):
            if isinstance(grad, torch.Tensor):
                return "grad_output", index, grad
    grad_inputs = hook_args[0] if len(hook_args) >= 1 else ()
    if grad_inputs is not None:
        for index, grad in enumerate(grad_inputs):
            if isinstance(grad, torch.Tensor):
                return "grad_input", index, grad
    return None


def log_recording_backward(
    recording: Any,
    loss: torch.Tensor,
    *,
    save_grads: Any = None,
    default_grad: Any = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
) -> Any:
    """Run backward while capturing gradients for a fastlog ``Recording``."""

    from ...capture.projections import sync_recording_grad_records_from_sidecar
    from ...fastlog.exceptions import InvalidStorageError, RecorderStateError

    recording_state = getattr(recording, "_recording_state", None)
    if recording_state is None:
        raise RecorderStateError("Recording.log_backward() requires a live recording state")
    trace = getattr(recording_state, "runtime_trace", None)
    if trace is None:
        raise RecorderStateError(
            "Recording.log_backward() requires a live runtime trace from Recorder.log()."
        )
    effective_save_grads = (
        save_grads if save_grads is not None else recording_state.options.save_grads
    )
    effective_default_grad = (
        default_grad if default_grad is not None else recording_state.options.default_grad
    )
    if (
        recording_state.storage_intent.on_disk
        and not recording_state.storage_intent.in_ram
        and _is_static_truthy_keep_grad(effective_save_grads)
    ):
        raise InvalidStorageError(
            "save_grads=True with CaptureSpec(keep_grad=True) is incompatible with "
            "disk-only fastlog gradient storage"
        )
    selected_forward_labels = _selected_fastlog_forward_labels(recording)

    def selected_save_grads(ctx: Any) -> Any:
        """Apply fastlog default and selected-forward-label semantics."""

        layer_label = getattr(ctx, "layer_label", None)
        if callable(effective_save_grads):
            decision = effective_save_grads(ctx)
            if (
                recording_state.storage_intent.on_disk
                and not recording_state.storage_intent.in_ram
                and _is_static_truthy_keep_grad(decision)
            ):
                raise InvalidStorageError(
                    "save_grads returned CaptureSpec(keep_grad=True), which is incompatible "
                    "with disk-only fastlog gradient storage"
                )
            return decision
        if effective_save_grads is True and layer_label not in selected_forward_labels:
            return False
        if effective_save_grads is None:
            return effective_default_grad
        return effective_save_grads

    backward_kwargs: dict[str, Any] = {"create_graph": create_graph}
    if retain_graph is not None:
        backward_kwargs["retain_graph"] = retain_graph

    def run() -> Any:
        """Run the requested recording backward call."""

        return loss.backward(**backward_kwargs)

    recording_state.active_save_grads_record_policy = selected_save_grads
    try:
        _run_backward_with_capture(
            trace,
            loss,
            run,
            trigger="recording_backward",
            engine_flags=backward_kwargs,
            save_grads=selected_save_grads,
        )
        sync_recording_grad_records_from_sidecar(recording_state)
    finally:
        recording_state.active_save_grads_record_policy = None
    return recording


def _is_accumulate_grad(grad_fn_handle: Any) -> bool:
    """Return whether an autograd node is an AccumulateGrad leaf.

    Parameters
    ----------
    grad_fn_handle
        Autograd node to inspect.

    Returns
    -------
    bool
        True when ``grad_fn_handle`` is an AccumulateGrad node.
    """

    accumulate_grad_cls = get_accumulate_grad_class()
    return type(grad_fn_handle).__name__ == "AccumulateGrad" or isinstance(
        grad_fn_handle, accumulate_grad_cls
    )


def _clear_forward_grad_fn_refs(trace: Any) -> None:
    """Clear strong forward references to autograd nodes after hook registration.

    Parameters
    ----------
    trace:
        Trace whose Op and Layer grad_fn_handle object refs should be
        released.
    """
    for layer in trace.layer_list:
        layer.grad_fn_handle = None
    for layer_log in trace.layer_logs.values():
        layer_log.grad_fn_handle = None


def _warn_zero_match_backward_interventions(trace: Any) -> None:
    """Warn when an armed capture-time backward selector fired nowhere.

    Parameters
    ----------
    trace:
        Trace whose intervention spec and deferred fire counter should be
        reconciled after a completed backward pass.
    """

    spec = getattr(trace, "_intervention_spec", None)
    hook_specs = getattr(spec, "hook_specs", ())
    selector_hooks = [
        hook_spec
        for hook_spec in hook_specs
        if hook_spec.metadata.get("created_by") == "intervene_backward_selector"
    ]
    if not selector_hooks:
        return
    try:
        if int(getattr(trace, "_tl_intervene_selector_fire_count", 0)) == 0:
            targets = [hook_spec.site_target for hook_spec in selector_hooks]
            warnings.warn(
                f"Capture-time backward intervention selector {targets[0]!r} matched zero "
                "sites; no intervention fired.",
                UserWarning,
                stacklevel=3,
            )
    finally:
        trace.__dict__.pop("_tl_intervene_selector_fire_count", None)


def _run_backward_with_capture(
    trace: Any,
    loss: torch.Tensor,
    backward_callable: Callable[[], Any],
    *,
    trigger: str = "backward",
    outer_context: str | None = None,
    engine_flags: dict[str, object] | None = None,
    save_grads: Any | MissingType = MISSING,
    forward_op_count_at_trigger: int | None = None,
    backward_call_context: FuncCallLocation | None = None,
) -> Any:
    """Capture a backward graph, run backward, and record memory delta.

    Parameters
    ----------
    trace:
        Trace to mutate.
    loss:
        Loss tensor that roots the backward graph.
    backward_callable:
        Zero-argument callable that performs the actual backward pass.

    Returns
    -------
    Any
        Return value from ``backward_callable``.
    """
    from ...intervention.hooks import normalize_hooks_from_spec

    _ensure_not_inference_only_backward(trace)
    _ensure_not_chunked_forward_backward(trace)
    if loss.grad_fn is None:
        raise ValueError("cannot run backward: loss has no grad_fn / is detached")
    if getattr(trace, "_tl_active_backward_bracket", False):
        return backward_callable()
    _close_implicit_backward_pass_if_open(trace)
    # Resolve the event stream BEFORE installing any trace or global capture
    # state: a stream-less trace raises typed here, and the refusal must not
    # leave half-installed state that poisons later captures.
    events = _ensure_backward_event_stream(trace)
    previous_save_grads_policy = getattr(trace, "_active_save_grads_policy", None)
    previous_had_save_grads_policy = "_active_save_grads_policy" in trace.__dict__
    active_save_grads_policy = (
        getattr(trace, "save_grads", None) if save_grads is MISSING else save_grads
    )
    # Publish the backward window through the admission lock (R54: the raw
    # unlocked save/swap here was the one site left when tf/paddle converted;
    # a cross-thread interleave could snapshot a concurrent capture's trace as
    # "previous" and republish it after that capture finished, wedging every
    # later admission). Same-thread nesting (multi-trace bracket, inner
    # backward inside a traced forward) keeps its save/restore semantics; a
    # foreign thread's live window refuses typed BEFORE any trace scratch is
    # written.
    intervention_spec = getattr(trace, "_intervention_spec", None)
    publication = _state.publish_backward_capture(
        trace,
        hook_plan=[*normalize_hooks_from_spec(intervention_spec)],
        intervention_spec=intervention_spec,
    )
    trace._active_save_grads_policy = active_save_grads_policy
    pass_index = int(getattr(trace, "num_backward_passes", 0)) + 1
    trace._active_backward_pass_index = pass_index
    trace._implicit_backward_pass_open = False
    start_event = BackwardPassStart(
        pass_index=pass_index,
        trigger=cast(Any, trigger),
        implicit=False,
        outer_context=outer_context,
        call_context_ref=backward_call_context,
        root_meta=(
            {
                "shape": tuple(loss.shape),
                "dtype": str(loss.dtype),
                "device": str(loss.device),
            },
        ),
        root_grad_arguments=None,
        inputs_subset=(),
        order=None,
        origin_backward_pass=None,
        save_grads_policy_repr=repr(active_save_grads_policy),
        engine_flags=engine_flags,
        forward_op_count_at_trigger=(
            forward_op_count_at_trigger
            if forward_op_count_at_trigger is not None
            else _forward_op_count_at_backward_trigger(trace)
        ),
        timestamp=time.time(),
    )
    events.append_backward(start_event)
    handles: list[Any] = []
    try:
        _walk_and_hook_backward_graph(trace, loss, handles)
    except BaseException:
        # The graph walk can fail after the start event and global capture
        # state have been installed. Restore the global state so a failed
        # backward cannot poison later traces — but NEVER delete the start
        # record: a failed attempted pass is evidence, and it closes with a
        # terminal failed End so the bracket invariant holds exactly (the
        # same convention the engine-failure path below already follows).
        # Restore process-global and per-pass scratch state before any fallible
        # cleanup. The graph walk can be interrupted after registering only a
        # prefix of hooks, so unwind those handles and disarm fail-closed.
        publication.restore()
        trace.__dict__.pop("_tl_active_backward_bracket", None)
        trace.__dict__.pop("_active_backward_pass_index", None)
        if previous_had_save_grads_policy:
            trace.__dict__["_active_save_grads_policy"] = previous_save_grads_policy
        else:
            trace.__dict__.pop("_active_save_grads_policy", None)
        events.append_backward(
            BackwardPassEnd(
                pass_index=pass_index,
                duration=None,
                peak_memory=None,
                status="error",
                order_attribution_coverage=None,
            )
        )
        trace.num_backward_passes = max(int(getattr(trace, "num_backward_passes", 0)), pass_index)
        for handle in handles:
            with contextlib.suppress(BaseException):
                handle.remove()
        with contextlib.suppress(BaseException):
            _clear_forward_grad_fn_refs(trace)
        with contextlib.suppress(BaseException):
            disarm_triggers(trace)
        with contextlib.suppress(BaseException):
            synchronize_pending_cpu_async_copies()
        with contextlib.suppress(BaseException):
            _materialize_backward_projections(trace)
        raise
    backend, before = _peak_memory_baseline(loss.device)
    backward_start_time = time.time()
    status = "ok"
    result = None
    trace._tl_active_backward_bracket = True
    try:
        from ._aten_capture import _capture_backward_aten

        with _capture_backward_aten(trace, pass_index):
            result = backward_callable()
    except BaseException:
        # BaseException (including KeyboardInterrupt/SystemExit) is stamped
        # as a failed attempt without being swallowed or translated: the
        # terminal End record in ``finally`` must never report "ok" for a
        # pass the engine did not complete.
        status = "error"
        raise
    finally:
        # Restore globals and per-pass scratch FIRST. Every operation below is
        # user/framework code or non-trivial bookkeeping and may raise.
        publication.restore()
        trace.__dict__.pop("_tl_active_backward_bracket", None)
        trace.__dict__.pop("_active_backward_pass_index", None)
        if previous_had_save_grads_policy:
            trace.__dict__["_active_save_grads_policy"] = previous_save_grads_policy
        else:
            trace.__dict__.pop("_active_save_grads_policy", None)

        # Close the journal bracket before fallible projection/accounting
        # finalizers. A later failure can leave derived fields stale, but never
        # leaves a start-only pass that violates the event-stream invariant.
        duration = time.time() - backward_start_time
        memory_error: BaseException | None = None
        try:
            _backend, after = _memory_snapshot(loss.device)
        except BaseException as exc:
            memory_error = exc
            after = before
        peak_delta = max(0, after - before)
        # The higher-order rewalk EMITS journal events (GradFnDiscovered), so it
        # must run inside the pass bracket; the bracket close is guaranteed even
        # if the rewalk fails, preserving the never-start-only-pass invariant.
        rewalk_error: BaseException | None = None
        try:
            _rewalk_higher_order_grad_fns(trace)
        except BaseException as exc:
            rewalk_error = exc
        events.append_backward(
            BackwardPassEnd(
                pass_index=pass_index,
                duration=duration,
                peak_memory=peak_delta,
                status=cast(Any, status),
                order_attribution_coverage=None,
            )
        )
        if rewalk_error is not None:
            # The graph walk already registered hooks and strong-pinned
            # grad-fn refs; propagating before cleanup strands live
            # TorchLens hooks on the user's autograd graph (the next plain
            # backward re-enters hook code and appends phantom events).
            # Mirror the walk-failure arm's unconditional cleanup, then
            # re-raise the rewalk error.
            trace.num_backward_passes = max(
                int(getattr(trace, "num_backward_passes", 0)), pass_index
            )
            with contextlib.suppress(BaseException):
                _clear_pending_accumulate_grad_records(trace)
            with contextlib.suppress(BaseException):
                _clear_fire_timing_stamps(trace)
            with contextlib.suppress(BaseException):
                _drain_gradfn_markers(trace)
            for handle in handles:
                with contextlib.suppress(BaseException):
                    handle.remove()
            with contextlib.suppress(BaseException):
                _clear_forward_grad_fn_refs(trace)
            with contextlib.suppress(BaseException):
                synchronize_pending_cpu_async_copies()
            with contextlib.suppress(BaseException):
                _materialize_backward_projections(trace)
            raise rewalk_error
        trace.num_backward_passes = max(int(getattr(trace, "num_backward_passes", 0)), pass_index)
        # Fenced like the rewalk-error arm above (b3-sol-R14-1): an unfenced
        # failure in the pending-record clear stranded every grad-fn hook and
        # strong forward graph ref registered by the walk. Each cleanup step
        # runs regardless of its siblings; the FIRST failure still propagates
        # at the end of the tail (never silently swallowed).
        cleanup_error: BaseException | None = None
        try:
            _clear_pending_accumulate_grad_records(trace)
        except BaseException as exc:
            cleanup_error = exc
        try:
            _clear_fire_timing_stamps(trace)
        # Cleanup must preserve the first failure, including cancellation.
        except BaseException as exc:  # noqa: BLE001
            cleanup_error = cleanup_error if cleanup_error is not None else exc
        try:
            _drain_gradfn_markers(trace)
        except BaseException as exc:  # noqa: BLE001
            cleanup_error = cleanup_error if cleanup_error is not None else exc
        # SUCCESS path: there is no primary exception whose precedence would
        # justify discarding a failure here, so fold it into cleanup_error
        # like the sibling steps. suppress(BaseException) silently discarded
        # a KeyboardInterrupt delivered during handle removal after a
        # successful backward and log_backward returned normally (grind-r6
        # b8 R63, fable probe). Already-removed / framework-debris handles
        # still never mask a real first failure: the fold keeps the FIRST.
        for handle in handles:
            try:
                handle.remove()
            except BaseException as exc:
                cleanup_error = cleanup_error if cleanup_error is not None else exc
        try:
            _clear_forward_grad_fn_refs(trace)
        except BaseException as exc:
            cleanup_error = cleanup_error if cleanup_error is not None else exc
        trace.backward_memory_backend = backend
        trace.backward_peak_memory += Bytes(peak_delta)
        trace.backward_durations.append(Duration(duration))
        trace.total_param_gradient_memory = Bytes(
            sum(int(param_log.gradient_memory) for param_log in getattr(trace, "param_logs", []))
        )
        # Fence in-flight cpu_async D2H grad copies before projections make
        # the payloads reachable (R36-1 for the backward seam): the forward
        # finalize drain already ran at capture time, so without this drain a
        # host read of a backward-retained cpu_async payload could observe
        # partial bytes from an unfinished non_blocking copy.
        synchronize_pending_cpu_async_copies()
        _materialize_backward_projections(trace)
        if status == "ok":
            _warn_zero_match_backward_interventions(trace)
        if cleanup_error is not None:
            # Pre-fence, the pending-clear failure raised here-abouts anyway
            # (before the later memory raise), so first-cleanup-error keeps
            # precedence over the snapshot error.
            raise cleanup_error
        if memory_error is not None:
            raise memory_error
    return result


def disarm_triggers(trace: Any) -> None:
    """Detach a trace from global autograd trigger interception.

    Parameters
    ----------
    trace:
        Trace whose future tensor-hook and autograd-wrapper capture should be
        disabled.

    Returns
    -------
    None
        Registry state is updated in place.
    """

    trace._tl_backward_triggers_disarmed = True
    _purge_trace_from_backward_registry(trace)


def _capture_autograd_engine_call(
    roots: Any,
    engine_callable: Callable[[], Any],
    *,
    trigger: Literal["autograd_backward", "autograd_grad"],
    engine_flags: dict[str, object] | None,
    forward_op_count_at_trigger: int | None = None,
) -> Any:
    """Run a global autograd entry through TorchLens capture when roots match.

    Parameters
    ----------
    roots:
        Tensor roots passed to the autograd engine.
    engine_callable:
        Zero-argument callable invoking the original PyTorch autograd function.
    trigger:
        Trigger label to store on ``BackwardPassStart``.
    engine_flags:
        Best-effort keyword metadata from the engine invocation.

    Returns
    -------
    Any
        Return value from ``engine_callable``.
    """

    from .tensor_tracking import _is_fork_relative

    # A backward the user directed at one trace (open managed bracket) must
    # not be re-captured by that trace's FORK RELATIVES just because they
    # share the root tensors: a fork's ``log_backward`` would otherwise
    # silently append a pass to its parent's projection (and vice versa).
    # Unrelated root-owning traces keep the historical multi-trace capture.
    directing_trace, _ = _state.active_capture()
    if directing_trace is not None and not getattr(
        directing_trace, "_tl_active_backward_bracket", False
    ):
        directing_trace = None
    matched_traces = tuple(
        trace
        for trace in _traces_for_roots(roots)
        if not getattr(trace, "_tl_active_backward_bracket", False)
        and not (
            directing_trace is not None
            and trace is not directing_trace
            and _is_fork_relative(trace, directing_trace)
        )
    )
    loss = _first_root_tensor(roots)
    if loss is None or not matched_traces:
        return _run_with_detached_activation_guidance(roots, engine_callable)
    if len(matched_traces) == 1:
        matched_trace = matched_traces[0]
        root_forward_op_count = _root_forward_op_count(matched_trace, roots)
        return _run_backward_with_capture(
            matched_trace,
            loss,
            engine_callable,
            trigger=trigger,
            engine_flags=engine_flags,
            forward_op_count_at_trigger=(
                root_forward_op_count
                if root_forward_op_count is not None
                else forward_op_count_at_trigger
            ),
        )

    # P1 supports multi-trace bracket opening by nesting setup and running the
    # engine once through the innermost capture. Later projection phases will
    # refine per-trace outer_context metadata.
    def nested_call(index: int) -> Any:
        """Run nested capture setup for all matched traces."""

        if index == len(matched_traces):
            return engine_callable()
        matched_trace = matched_traces[index]
        root_forward_op_count = _root_forward_op_count(matched_trace, roots)
        return _run_backward_with_capture(
            matched_trace,
            loss,
            lambda: nested_call(index + 1),
            trigger=trigger,
            outer_context=None if index == 0 else f"{trigger}:multi_trace",
            engine_flags=engine_flags,
            forward_op_count_at_trigger=(
                root_forward_op_count
                if root_forward_op_count is not None
                else forward_op_count_at_trigger
            ),
        )

    return nested_call(0)


# ---------------------------------------------------------------------------
# Checkpoint invocation tokens (L9 memo 2.3). Every spelling below is
# DOCUMENTED-UNSTABLE; the typed ambiguity REFUSAL awaits a pending contract amendment and
# NOT raised anywhere in this module -- only the capture machinery (classifier,
# per-instance token-bearing closures, witness bookkeeping) ships pre-amendment.
# ---------------------------------------------------------------------------

_CHECKPOINT_TOKEN_STATE: weakref.WeakKeyDictionary[Any, dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)
"""Per-trace runtime token state: next ordinal, token records, degrade flags.

Tokens die with the process (the witness summary on the DROP-gated Trace
field is the only projected surface). Minting happens only on the armed owner
thread (classifier condition 2), so the ordinal needs no lock; engine-thread
unpack evidence appends go through the per-trace lock.
"""

#: Degrade-flag vocabulary (provisional strings, E-L9-4 routing): D1-D7.
_CHECKPOINT_FLAG_CLASSIFIER_UNAVAILABLE = "classifier_unavailable"  # D1
_CHECKPOINT_FLAG_PATCH_UNAVAILABLE = "patch_unavailable"  # D2
_CHECKPOINT_FLAG_EXOTIC_SUBCLASS = "exotic_subclass"  # D3
_CHECKPOINT_FLAG_UNMATCHED_BACKWARD_WARN = "unmatched_backward_warn"  # D4
_CHECKPOINT_FLAG_REENTRANT_NODE_DISCOVERED = "reentrant_node_discovered"  # D5
_CHECKPOINT_FLAG_UNWITNESSED_ENTER = "unwitnessed_checkpoint_enter"  # D6
_CHECKPOINT_FLAG_HOOK_IDENTITY_UNPRESERVED = "hook_identity_unpreserved"  # D7

#: Attribute-name prefix reserved for TorchLens's own hook-wrapper bookkeeping
#: (``__tl_saved_tensors_hook_scoped__``, ``__tl_checkpoint_token__``,
#: ``__tl_token_inner__``); every other attribute found on a pack/unpack hook
#: callable is foreign state the token wrapper must carry forward verbatim.
_TL_HOOK_ATTR_PREFIX = "__tl_"

#: Reentrant checkpoint node qualname sentinel (venv-verified on torch 2.13,
#: memo claim A22). GradFn class_qualname is stored module-qualified, so the
#: sentinel matches on the trailing qualname segment.
_REENTRANT_CHECKPOINT_QUALNAME = "CheckpointFunctionBackward"


def _resolve_checkpoint_hook_cls() -> type | None:
    """Return torch's private non-reentrant checkpoint hook class, or ``None``.

    A torch without the private name yields ``None`` -> classifier
    unavailable -> NO tokens (degrade class D1, fail-closed: an unrecognized
    checkpoint variant can never mint a false token). Routed through the
    compat chokepoint; absence flips ``HAS_CHECKPOINT_HOOK_CLASS`` there.
    """

    from ...utils._torch_compat import get_checkpoint_hook_class

    return get_checkpoint_hook_class()


def _checkpoint_hook_identity_attrs_in_play() -> bool:
    """Return whether this torch runtime's checkpoint hooks need attr carry-over.

    torch >= 2.14 interposes ``_checkpoint_internal_hook`` (routed through the
    named ``HAS_CHECKPOINT_INTERNAL_HOOK_CLASS`` capability flag): its
    ``__enter__``/``__exit__`` stash a private ``_user_hooks`` attribute
    directly on the pack-hook callable across the enter/exit pair. Absence
    (torch 2.13 and earlier) means no such state exists to preserve.
    """

    from ...utils._torch_compat import get_checkpoint_internal_hook_class

    return get_checkpoint_internal_hook_class() is not None


def _carry_foreign_hook_attrs(new_hook: Callable[[Any], Any], old_hook: Any) -> bool:
    """Copy non-TorchLens attributes from ``old_hook`` onto ``new_hook``.

    The checkpoint-token classifier replaces ``context.pack_hook`` /
    ``context.unpack_hook`` with freshly built token-bearing wrappers. On
    torch >= 2.14, torch's own ``_checkpoint_internal_hook.__enter__`` has
    already stashed a private ``_user_hooks`` attribute on the hook object
    BEFORE the classifier runs (memo 2.3 addendum); dropping it by installing
    an unrelated wrapper object would make the matching ``__exit__`` raise
    ``AttributeError`` reaching for state that moved to a different function
    object -- an exception INSIDE ``__exit__`` that skips torch's own
    ``_pop_saved_tensors_default_hooks()`` call and permanently corrupts the
    global saved-tensors-hooks stack for the rest of the process (the
    261-test cascade this guards against). Returns ``True`` once every
    foreign attribute copied cleanly; ``False`` means the caller must skip
    the hook-object replacement entirely (fail closed) rather than install a
    wrapper torch's own internals cannot find their state on.
    """

    try:
        foreign_items = [
            (key, value)
            for key, value in vars(old_hook).items()
            if not key.startswith(_TL_HOOK_ATTR_PREFIX)
        ]
    except TypeError:
        # No __dict__ to read (e.g. a C-level callable): nothing to carry, and
        # nothing torch's chain could have stashed either -- safe to proceed.
        return True
    try:
        for key, value in foreign_items:
            setattr(new_hook, key, value)
    except AttributeError:
        return False
    return True


def _checkpoint_token_state(trace: Any) -> dict[str, Any]:
    """Return (and lazily build) one trace's runtime checkpoint-token state."""

    state = _CHECKPOINT_TOKEN_STATE.get(trace)
    if state is None:
        state = {
            "next_ordinal": 1,
            "tokens": {},
            "flags": set(),
            "lock": threading.Lock(),
        }
        _CHECKPOINT_TOKEN_STATE[trace] = state
    return state


def _flag_checkpoint_degrade(trace: Any, flag: str) -> None:
    """Set one degrade flag and refresh the projected witness summary."""

    state = _checkpoint_token_state(trace)
    if flag not in state["flags"]:
        state["flags"].add(flag)
        _refresh_checkpoint_witness(trace)


def _flag_reentrant_checkpoint_if_sentinel(trace: Any, class_qualname: str | None) -> None:
    """Set D5 when a discovered grad-fn is the reentrant checkpoint node."""

    if class_qualname and class_qualname.rsplit(".", 1)[-1] == _REENTRANT_CHECKPOINT_QUALNAME:
        _flag_checkpoint_degrade(trace, _CHECKPOINT_FLAG_REENTRANT_NODE_DISCOVERED)


def _token_bearing_pack_hook(
    trace: Any, token: int, inner: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Wrap the (already-scoped) pack hook with count-only token evidence.

    Runs on the owner thread during forward with logging paused by the inner
    scoped body; this wrapper does dict bookkeeping only (no tensor ops), so
    the r34-A/r33-F2 fences are untouched. The forward-side slot->op binding
    is NOT claimed: pack evidence is a COUNT, never a producer-op join (the
    pack hook runs before TorchLens logs the producing op).
    """

    trace_ref = weakref.ref(trace)

    def pack_hook(value: Any) -> Any:
        """Count one checkpoint pack for this token, then pass the value through.

        Held via a weakref so an abandoned trace cannot keep the hook alive; a
        dead referent degrades to pass-through rather than raising inside torch's
        saved-tensor machinery.
        """
        live_trace = trace_ref()
        if live_trace is not None:
            state = _CHECKPOINT_TOKEN_STATE.get(live_trace)
            record = None if state is None else state["tokens"].get(token)
            if record is not None:
                record["pack_count"] += 1
        return inner(value)

    pack_hook.__tl_saved_tensors_hook_scoped__ = True  # type: ignore[attr-defined]
    pack_hook.__tl_checkpoint_token__ = token  # type: ignore[attr-defined]
    pack_hook.__tl_token_inner__ = inner  # type: ignore[attr-defined]
    return pack_hook


def _token_bearing_unpack_hook(
    trace: Any, token: int, inner: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    """Wrap the (already-scoped) unpack hook with window-evidence recording.

    Unpack fires on the ENGINE thread during backward: each firing records an
    evidence point (graph task id, active grad-fn fire bracket if any, pass
    coordinates) under the per-trace lock. An unpack outside any fire bracket
    contributes a task-id-only point, which can only widen ambiguity, never
    resolve it (dependent reads refuse rather than guess).
    """

    trace_ref = weakref.ref(trace)

    def unpack_hook(value: Any) -> Any:
        """Count one checkpoint unpack (recomputation) for this token.

        Mirrors :func:`pack_hook`: weakref-held, pass-through on a dead trace.
        """
        live_trace = trace_ref()
        if live_trace is not None:
            state = _CHECKPOINT_TOKEN_STATE.get(live_trace)
            record = None if state is None else state["tokens"].get(token)
            if state is not None and record is not None:
                from ._aten_capture import _ACTIVE_BACKWARD_GRAD_FN_REFS

                active_refs = _ACTIVE_BACKWARD_GRAD_FN_REFS.get()
                point = {
                    "graph_task_id": _current_backward_graph_task_id(),
                    "fire_bracket": active_refs[-1] if active_refs else None,
                    "pass_index": getattr(live_trace, "_active_backward_pass_index", None),
                }
                with state["lock"]:
                    record["unpack_evidence"].append(point)
        return inner(value)

    unpack_hook.__tl_saved_tensors_hook_scoped__ = True  # type: ignore[attr-defined]
    unpack_hook.__tl_checkpoint_token__ = token  # type: ignore[attr-defined]
    unpack_hook.__tl_token_inner__ = inner  # type: ignore[attr-defined]
    return unpack_hook


def _observe_saved_tensors_hooks_enter(context: Any) -> None:
    """Classify one ``saved_tensors_hooks`` enter; mint or flag (memo 2.3).

    Classifier (exact, conservative, fail-closed) -- a token is minted ONLY
    when ALL of:

    1. the context is a ``torch.utils.checkpoint._checkpoint_hook`` instance
       (lazy private-name resolution; absence -> D1, no tokens);
    2. capture-armed on the OWNER thread (same gate as the scoped hook body);
    3. NOT inside an engine invocation (torch's graph-task witness).

    ``_recomputation_hook`` instances, ordinary ``saved_tensors_hooks``,
    ``save_on_cpu``, and user subclasses fail (1) and NEVER mint or flag.
    A ``_checkpoint_hook`` enter failing (2)/(3) sets D6 instead of staying
    silent: every checkpoint enter therefore either mints or flags.
    """

    trace, logging_enabled = _state.active_capture()
    if trace is None:
        return
    checkpoint_cls = _resolve_checkpoint_hook_cls()
    if checkpoint_cls is None:
        _flag_checkpoint_degrade(trace, _CHECKPOINT_FLAG_CLASSIFIER_UNAVAILABLE)
        return
    if not isinstance(context, checkpoint_cls):
        return
    armed_on_owner_thread = (
        logging_enabled and _state._active_owner_thread_id == threading.get_ident()
    )
    if not armed_on_owner_thread or _current_backward_graph_task_id() is not None:
        _flag_checkpoint_degrade(trace, _CHECKPOINT_FLAG_UNWITNESSED_ENTER)
        return
    state = _checkpoint_token_state(trace)
    token = state["next_ordinal"]
    state["next_ordinal"] = token + 1
    state["tokens"][token] = {"pack_count": 0, "unpack_evidence": []}
    # Install token-bearing wrappers on THIS instance (the shipped rescoping
    # move already rewraps instance hooks in patched_enter). Re-entry mints a
    # fresh token and installs FRESH wrappers: unwrap any prior token layer
    # through its declared inner so counts never stack across invocations.
    hook_owner: Any = context
    old_pack_hook = hook_owner.pack_hook
    old_unpack_hook = hook_owner.unpack_hook
    new_pack_hook = _token_bearing_pack_hook(
        trace, token, getattr(old_pack_hook, "__tl_token_inner__", old_pack_hook)
    )
    new_unpack_hook = _token_bearing_unpack_hook(
        trace, token, getattr(old_unpack_hook, "__tl_token_inner__", old_unpack_hook)
    )
    if _checkpoint_hook_identity_attrs_in_play() and not (
        _carry_foreign_hook_attrs(new_pack_hook, old_pack_hook)
        and _carry_foreign_hook_attrs(new_unpack_hook, old_unpack_hook)
    ):
        # Fail closed (memo 2.3 addendum, torch >= 2.14): this torch runtime's
        # checkpoint internals keep private cross-call identity state
        # directly on the hook callable that the token wrapper could not
        # carry forward safely. Installing it anyway would desync torch's
        # own __exit__ accounting and corrupt its global hook stack for the
        # rest of the process. Roll back the mint and leave the already-
        # scoped hooks untouched: no token for this enter, degrade instead.
        del state["tokens"][token]
        state["next_ordinal"] = token
        _flag_checkpoint_degrade(trace, _CHECKPOINT_FLAG_HOOK_IDENTITY_UNPRESERVED)
        return
    hook_owner.pack_hook = new_pack_hook
    hook_owner.unpack_hook = new_unpack_hook
    _ensure_backward_event_stream(trace).append_backward(
        CheckpointInvocationObserved(token=token, timestamp=time.time())
    )
    _refresh_checkpoint_witness(trace)


def _checkpoint_token_summary(
    record: dict[str, Any],
    evidence_lock: threading.Lock,
    grad_fn_logs: dict[int, Any],
    layer_lookup: dict[str, Any],
) -> dict[str, Any]:
    """Return one checkpoint token's count, site, and backward-window summary."""

    with evidence_lock:
        evidence = list(record["unpack_evidence"])
    site_candidates: set[str] = set()
    window_refs: list[tuple[int | None, str | None, int | None]] = []
    for point in evidence:
        bracket = point.get("fire_bracket")
        if bracket is None:
            window_refs.append((point.get("pass_index"), None, None))
            continue
        bracket_object_id, bracket_call_index, bracket_pass_index = bracket
        grad_fn_record = grad_fn_logs.get(bracket_object_id)
        label = getattr(grad_fn_record, "label", None)
        window_refs.append((bracket_pass_index, label, bracket_call_index))
        if grad_fn_record is not None and getattr(grad_fn_record, "has_op", False):
            op = layer_lookup.get(grad_fn_record.op_label)
            site_key = getattr(op, "site_key", None)
            if site_key is not None:
                site_candidates.add(site_key)
    return {
        "pack_count": record["pack_count"],
        "unpack_evidence_count": len(evidence),
        "site_key_candidates": sorted(site_candidates),
        "window_evidence": window_refs,
    }


def _checkpoint_witness_verdict(flags: set[str], token_count: int) -> str:
    """Return the evidence-scoped checkpoint witness verdict."""

    if flags:
        return "evidence_incomplete"
    if token_count:
        return "checkpoint_invocations_observed"
    return "no_checkpoint_invocation_observed"


def _refresh_checkpoint_witness(trace: Any) -> None:
    """Project the runtime token state onto the DROP-gated witness field."""

    state = _CHECKPOINT_TOKEN_STATE.get(trace)
    flags = set(state["flags"]) if state is not None else set()
    if not HAS_SAVED_TENSORS_HOOKS_PATCHABLE or not _SAVED_TENSORS_HOOKS_INIT_PATCHED:
        flags.add(_CHECKPOINT_FLAG_PATCH_UNAVAILABLE)
    if state is None:
        tokens_summary: dict[int, dict[str, Any]] = {}
    else:
        tokens_summary = {
            token: _checkpoint_token_summary(
                record,
                state["lock"],
                getattr(trace, "grad_fn_logs", {}),
                getattr(trace, "layer_dict_all_keys", {}),
            )
            for token, record in state["tokens"].items()
        }
    verdict = _checkpoint_witness_verdict(flags, len(tokens_summary))
    trace.checkpoint_invocation_witness = {
        "token_count": len(tokens_summary),
        "tokens": tokens_summary,
        "degrade_flags": sorted(flags),
        "verdict": verdict,
    }


def _scoped_saved_tensors_hook(hook: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap a user pack/unpack hook so its torch ops stay out of the capture.

    Autograd invokes pack hooks synchronously DURING the producing op's
    dispatch, before the wrapper has logged that op. A hook-run torch op
    therefore observes (and can label) the producer's not-yet-logged output:
    a same-object return like ``t.cpu()`` on a CPU tensor stole the producer's
    label slot, deleting the real user op from the graph and leaving the hook
    op parentless, while a pack over an already-labeled input spliced the hook
    op INTO the forward dataflow (round-34 Finding A). Hook results feed
    autograd's saved-for-backward storage, never the forward dataflow, so the
    honest forward-trace model is to scope hook bodies as autograd-internal
    bookkeeping: logging is paused for the hook body on the capture-owner
    thread, exactly like TorchLens's own internal tensor ops.
    """

    if getattr(hook, "__tl_saved_tensors_hook_scoped__", False):
        return hook

    def scoped_hook(value: Any) -> Any:
        """Run the user hook with capture logging paused on the owner thread."""
        active_trace, logging_enabled = _state.active_capture()
        on_owner_thread = _state._active_owner_thread_id == threading.get_ident()
        if logging_enabled and active_trace is not None and on_owner_thread:
            with pause_logging():
                return hook(value)
        return hook(value)

    try:
        functools.update_wrapper(scoped_hook, hook)
    except (AttributeError, TypeError):
        pass
    scoped_hook.__tl_saved_tensors_hook_scoped__ = True  # type: ignore[attr-defined]
    return scoped_hook


def _install_saved_tensors_hooks_scope() -> None:
    """Patch ``saved_tensors_hooks`` to scope user hooks idempotently.

    Covers ``torch.autograd.graph.saved_tensors_hooks``, ``save_on_cpu``, and
    the non-reentrant checkpoint hook (both subclass it and route through
    ``super().__init__``). ``__init__`` scopes hooks at construction time;
    ``__enter__`` re-scopes them at use time so a context CONSTRUCTED BEFORE
    the first capture (e.g. stored on a model in its own ``__init__``, round-35
    R1) is covered too -- construction-time scoping alone left such instances
    carrying raw hooks forever. Re-scoping is idempotent through the
    ``__tl_saved_tensors_hook_scoped__`` marker. When the class shape is not
    patchable on this torch runtime (``HAS_SAVED_TENSORS_HOOKS_PATCHABLE`` is
    False), degrade gracefully to the historical unscoped behavior.
    """

    global _SAVED_TENSORS_HOOKS_INIT_PATCHED
    global _ORIGINAL_SAVED_TENSORS_HOOKS_INIT, _ORIGINAL_SAVED_TENSORS_HOOKS_ENTER
    if _SAVED_TENSORS_HOOKS_INIT_PATCHED or not HAS_SAVED_TENSORS_HOOKS_PATCHABLE:
        return
    hooks_cls = torch.autograd.graph.saved_tensors_hooks
    original_init = hooks_cls.__init__
    original_enter = hooks_cls.__enter__
    _ORIGINAL_SAVED_TENSORS_HOOKS_INIT = original_init
    _ORIGINAL_SAVED_TENSORS_HOOKS_ENTER = original_enter

    @functools.wraps(original_init)
    def patched_init(self: Any, pack_hook: Any, unpack_hook: Any) -> None:
        """Install scoped pack/unpack hooks in place of the raw user hooks."""
        original_init(
            self,
            _scoped_saved_tensors_hook(pack_hook),
            _scoped_saved_tensors_hook(unpack_hook),
        )

    @functools.wraps(original_enter)
    def patched_enter(self: Any) -> Any:
        """Re-scope hooks at use time to cover pre-wrap-constructed contexts."""
        try:
            self.pack_hook = _scoped_saved_tensors_hook(self.pack_hook)
            self.unpack_hook = _scoped_saved_tensors_hook(self.unpack_hook)
        except AttributeError:
            # An exotic subclass without settable hook attributes degrades to
            # the historical unscoped behavior rather than breaking entry --
            # and flags D3 so the checkpoint witness never claims coverage a
            # degraded patch cannot provide.
            active_trace, _ = _state.active_capture()
            if active_trace is not None:
                _flag_checkpoint_degrade(active_trace, _CHECKPOINT_FLAG_EXOTIC_SUBCLASS)
        else:
            # Checkpoint-invocation classifier (L9 memo 2.3): mint-or-flag on
            # every _checkpoint_hook enter, BEFORE original_enter pushes the
            # instance hooks onto torch's stack so the pushed pair is the
            # token-bearing one.
            _observe_saved_tensors_hooks_enter(self)
        return original_enter(self)

    hooks_cls.__init__ = patched_init  # type: ignore[method-assign]
    hooks_cls.__enter__ = patched_enter  # type: ignore[method-assign]
    global _INSTALLED_SAVED_TENSORS_HOOKS_INIT, _INSTALLED_SAVED_TENSORS_HOOKS_ENTER
    _INSTALLED_SAVED_TENSORS_HOOKS_INIT = patched_init
    _INSTALLED_SAVED_TENSORS_HOOKS_ENTER = patched_enter
    _SAVED_TENSORS_HOOKS_INIT_PATCHED = True


def _restore_slot_identity_checked(
    owner: Any,
    attr_name: str,
    installed: Callable[..., Any] | None,
    original: Callable[..., Any] | None,
    site_label: str,
) -> str | None:
    """Restore ``owner.attr_name`` to ``original`` only when it is still OURS.

    A third-party patch layered over the torchlens patch (apex/deepspeed-style
    passthroughs on the autograd entry points) is preserved and its
    ``site_label`` returned for the caller's burial disclosure instead of
    being silently clobbered -- this was the only teardown in the codebase
    with no drift guard (grind-r5 b8 R56). Returns ``None`` when the slot was
    restored (or there was nothing to restore).
    """

    if original is None:
        return None
    current = getattr(owner, attr_name, None)
    if installed is not None and current is not installed and current is not original:
        return site_label
    setattr(owner, attr_name, original)
    return None


def _warn_buried_autograd_sites(buried_sites: list[str]) -> None:
    """Disclose teardown slots left holding a foreign patch."""

    if not buried_sites:
        return
    from ..._errors import TorchLensWarning

    warnings.warn(
        "uninstall of TorchLens autograd wrappers left "
        f"{len(buried_sites)} slot(s) holding a third-party patch layered over "
        f"the torchlens wrapper: {', '.join(buried_sites)}. TorchLens never "
        "clobbers foreign patches, so those slots still run the torchlens "
        "wrapper underneath. Remove or reinstall the outer patch around the "
        "restored original to fully unwrap.",
        TorchLensWarning,
        stacklevel=3,
    )


def _uninstall_saved_tensors_hooks_scope(buried_sites: list[str] | None = None) -> None:
    """Restore the original ``saved_tensors_hooks`` methods when patched."""

    global _SAVED_TENSORS_HOOKS_INIT_PATCHED
    if not _SAVED_TENSORS_HOOKS_INIT_PATCHED:
        return
    own_sites = buried_sites if buried_sites is not None else []
    hooks_cls = torch.autograd.graph.saved_tensors_hooks
    for attr_name, installed, original in (
        ("__init__", _INSTALLED_SAVED_TENSORS_HOOKS_INIT, _ORIGINAL_SAVED_TENSORS_HOOKS_INIT),
        ("__enter__", _INSTALLED_SAVED_TENSORS_HOOKS_ENTER, _ORIGINAL_SAVED_TENSORS_HOOKS_ENTER),
    ):
        buried = _restore_slot_identity_checked(
            hooks_cls,
            attr_name,
            installed,
            original,
            f"torch.autograd.graph.saved_tensors_hooks.{attr_name}",
        )
        if buried is not None:
            own_sites.append(buried)
    _SAVED_TENSORS_HOOKS_INIT_PATCHED = False
    if buried_sites is None:
        _warn_buried_autograd_sites(own_sites)


def install_autograd_wrappers() -> None:
    """Install global autograd trigger wrappers idempotently.

    Returns
    -------
    None
        ``torch.autograd.backward`` and ``torch.autograd.grad`` are patched once.
    """

    global _AUTOGRAD_WRAPPERS_INSTALLED, _ORIGINAL_AUTOGRAD_BACKWARD, _ORIGINAL_AUTOGRAD_GRAD
    _install_saved_tensors_hooks_scope()
    if _AUTOGRAD_WRAPPERS_INSTALLED:
        return
    _ORIGINAL_AUTOGRAD_BACKWARD = torch.autograd.backward
    _ORIGINAL_AUTOGRAD_GRAD = torch.autograd.grad
    # Bind the snapshots into the closures, NEVER a call-time global read: a
    # buried old wrapper (foreign patch layered on top, identity-guarded
    # teardown, then a re-install snapshotting the foreign chain) otherwise
    # re-pointed EVERY live old wrapper at the new global -- a chain that
    # contains the old wrapper itself, i.e. infinite recursion on the first
    # backward (grind-r5 R56 follow-on, caught by this lane's own gate).
    closure_original_backward = cast(Callable[..., Any], _ORIGINAL_AUTOGRAD_BACKWARD)
    closure_original_grad = cast(Callable[..., Any], _ORIGINAL_AUTOGRAD_GRAD)

    def wrapped_backward(*args: Any, **kwargs: Any) -> Any:
        """Route ``torch.autograd.backward`` through TorchLens when roots match."""

        original = closure_original_backward
        forward_op_count_at_trigger = _active_forward_op_count_at_trigger()
        roots = _autograd_roots_from_call(args, kwargs, "tensors")

        def run() -> Any:
            """Invoke the original autograd backward function."""
            if _state._escape_detector_mode == "shadow":
                with expected_original_call(original, "autograd:backward"):
                    return original(*args, **kwargs)
            return original(*args, **kwargs)

        return _capture_autograd_engine_call(
            roots,
            run,
            trigger="autograd_backward",
            engine_flags=dict(kwargs),
            forward_op_count_at_trigger=forward_op_count_at_trigger,
        )

    def wrapped_grad(*args: Any, **kwargs: Any) -> Any:
        """Route ``torch.autograd.grad`` through TorchLens when roots match."""

        original = closure_original_grad
        forward_op_count_at_trigger = _active_forward_op_count_at_trigger()
        roots = _autograd_roots_from_call(args, kwargs, "outputs")

        def run() -> Any:
            """Invoke the original autograd grad function."""
            if (
                _state._escape_detector_mode == "shadow"
                or _state._completeness_witness_mode == "shadow"
            ):
                with expected_original_call(
                    original,
                    "autograd:grad",
                    census_scope="expected_opaque",
                ):
                    return original(*args, **kwargs)
            return original(*args, **kwargs)

        def engine() -> Any:
            """Run the original call inside the backward capture of its engine pass."""
            return _capture_autograd_engine_call(
                roots,
                run,
                trigger="autograd_grad",
                engine_flags=dict(kwargs),
                forward_op_count_at_trigger=forward_op_count_at_trigger,
            )

        active_trace, logging_enabled = _state.active_capture()
        if logging_enabled and active_trace is not None:
            from ._autograd_grad_boundary import record_autograd_grad_boundary

            return record_autograd_grad_boundary(engine, original, args, kwargs)
        return engine()

    # Provenance parity with every namespace wrapper (grind-r5 b8 R56): the
    # entry wrappers carry the original's metadata so introspection reports
    # torch.autograd.backward/grad (not a closure qualname), inspect.signature
    # resolves through __wrapped__, and pickling the wrapped entry by
    # reference succeeds for the whole wrapped epoch.
    functools.update_wrapper(wrapped_backward, _ORIGINAL_AUTOGRAD_BACKWARD)
    functools.update_wrapper(wrapped_grad, _ORIGINAL_AUTOGRAD_GRAD)

    global _INSTALLED_AUTOGRAD_BACKWARD, _INSTALLED_AUTOGRAD_GRAD
    _INSTALLED_AUTOGRAD_BACKWARD = wrapped_backward
    _INSTALLED_AUTOGRAD_GRAD = wrapped_grad
    torch.autograd.backward = wrapped_backward
    torch.autograd.grad = wrapped_grad
    _AUTOGRAD_WRAPPERS_INSTALLED = True


def uninstall_autograd_wrappers() -> None:
    """Restore original global autograd entry points when installed.

    Returns
    -------
    None
        PyTorch autograd functions are restored in place.
    """

    global _AUTOGRAD_WRAPPERS_INSTALLED
    buried_sites: list[str] = []
    _uninstall_saved_tensors_hooks_scope(buried_sites)
    if not _AUTOGRAD_WRAPPERS_INSTALLED:
        _warn_buried_autograd_sites(buried_sites)
        return
    for attr_name, installed, original in (
        ("backward", _INSTALLED_AUTOGRAD_BACKWARD, _ORIGINAL_AUTOGRAD_BACKWARD),
        ("grad", _INSTALLED_AUTOGRAD_GRAD, _ORIGINAL_AUTOGRAD_GRAD),
    ):
        buried = _restore_slot_identity_checked(
            torch.autograd,
            attr_name,
            installed,
            original,
            f"torch.autograd.{attr_name}",
        )
        if buried is not None:
            buried_sites.append(buried)
    _AUTOGRAD_WRAPPERS_INSTALLED = False
    _warn_buried_autograd_sites(buried_sites)


def _finalize_grad_streaming(trace: Any) -> None:
    """Finalize a deferred grad-streaming bundle after backward capture."""

    writer = getattr(trace, "_out_writer", None)
    if writer is None or not getattr(trace, "_defer_streaming_bundle_finalization", False):
        return

    from ...postprocess.finalization import (
        _evict_streamed_grads,
        _evict_streamed_outs,
        _finalize_streamed_bundle,
    )

    _finalize_streamed_bundle(trace)
    if not getattr(trace, "_keep_outs_in_memory", True):
        _evict_streamed_outs(trace)
    if not getattr(trace, "_grad_stream_retain_in_memory", True):
        _evict_streamed_grads(trace)
    for field_name in (
        "_out_writer",
        "_out_sink",
        "_keep_outs_in_memory",
        "_grad_stream_retain_in_memory",
        "_defer_streaming_bundle_finalization",
    ):
        trace.__dict__.pop(field_name, None)


def log_backward(
    self: Any,
    loss: torch.Tensor,
    *,
    save_grads: Any | MissingType = MISSING,
    **backward_kwargs: Any,
) -> Any:
    """Run ``loss.backward`` while capturing the backward graph.

    Parameters
    ----------
    loss:
        Loss tensor whose ``grad_fn_handle`` roots the backward graph.
    save_grads:
        Optional per-call gradient retention override. ``True`` captures all
        observed op gradients, ``False``/``None`` disables retention for this
        call, and selectors/callables are evaluated at hook fire time.
    **backward_kwargs:
        Keyword arguments forwarded to ``torch.Tensor.backward``.

    Returns
    -------
    Any
        The same Trace, for chaining.
    """
    _ensure_not_inference_only_backward(self)
    _ensure_not_chunked_forward_backward(self)
    backward_call_context = _capture_backward_call_context(self)

    def run() -> Any:
        """Run the user's requested backward call."""
        return loss.backward(**backward_kwargs)

    try:
        _run_backward_with_capture(
            self,
            loss,
            run,
            trigger="backward",
            engine_flags=dict(backward_kwargs),
            save_grads=save_grads,
            backward_call_context=backward_call_context,
        )
    finally:
        _finalize_grad_streaming(self)
    return self


class RecordingBackward:
    """Context manager that captures every ``Tensor.backward`` call inside it."""

    def __init__(
        self,
        trace: Any,
        *,
        save_grads: Any | MissingType = MISSING,
    ) -> None:
        """Store context state for temporary ``Tensor.backward`` patching."""
        self.trace = trace
        self.save_grads = save_grads
        self._original_backward: Callable[..., Any] | None = None
        self._wrapped_backward: Callable[..., Any] | None = None
        self._warned_unmatched_backward = False
        self._entry_depth = 0

    def __enter__(self) -> RecordingBackward:
        """Patch ``torch.Tensor.backward`` and return this context object."""
        # Re-entering an already-entered context must not re-patch: a second
        # entry would capture the first entry's wrapper as "original", so the
        # OUTER exit could never match its own wrapper and would leave a
        # TorchLens wrapper installed on the process-global
        # ``torch.Tensor.backward`` permanently. Recording semantics inside
        # the block are unchanged (the one installed wrapper already records
        # for this trace), so nested entry is a counted no-op.
        if self._entry_depth > 0:
            self._entry_depth += 1
            return self
        self._entry_depth = 1
        self._original_backward = torch.Tensor.backward
        trace = self.trace
        original_backward = self._original_backward

        def wrapped_backward(tensor_self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            """Capture the graph rooted at ``tensor_self`` before backward runs."""

            def run() -> Any:
                """Run the original Tensor.backward implementation."""
                if _state._escape_detector_mode == "shadow":
                    with expected_original_call(original_backward, "autograd:tensor_backward"):
                        return original_backward(tensor_self, *args, **kwargs)
                return original_backward(tensor_self, *args, **kwargs)

            # A backward on a graph that does not reach this trace's pinned
            # forward grad-fns is unrelated PyTorch work: delegate it
            # untouched instead of corrupting the selected trace with a
            # foreign backward pass. The user explicitly asked this block to
            # record, so an unmatched backward warns once instead of looking
            # like a successfully-recorded empty pass (e.g. a graph rebuilt
            # under no_grad by reentrant checkpointing pins zero grad-fns).
            if not any(matched is trace for matched in _traces_for_roots(tensor_self)):
                # D4: KEPT as an evidence-incompleteness flag, NOT a reentrant
                # detector (its trigger is zero pinned grad-fns, which a
                # partially-checkpointed model never trips).
                _flag_checkpoint_degrade(trace, _CHECKPOINT_FLAG_UNMATCHED_BACKWARD_WARN)
                if not self._warned_unmatched_backward:
                    self._warned_unmatched_backward = True
                    warnings.warn(
                        "A Tensor.backward() inside recording_backward() did not "
                        "reach any grad-fn pinned by this trace, so no backward "
                        "pass was recorded for it. This happens when the backward "
                        "graph does not include the traced forward (e.g. the model "
                        "ran entirely under torch.no_grad or reentrant "
                        "checkpointing rebuilt the graph outside the capture).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return run()

            return _run_backward_with_capture(
                trace,
                tensor_self,
                run,
                trigger="recording_backward",
                engine_flags=dict(kwargs),
                save_grads=self.save_grads,
            )

        self._wrapped_backward = wrapped_backward
        torch.Tensor.backward = wrapped_backward  # type: ignore[assignment, method-assign]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Restore ``torch.Tensor.backward`` unless someone patched over us."""
        if self._entry_depth > 1:
            # Inner exit of a re-entered context: the outermost exit owns the
            # restore and the streaming finalization.
            self._entry_depth -= 1
            return
        self._entry_depth = 0
        try:
            if self._original_backward is not None:
                if torch.Tensor.backward is self._wrapped_backward:
                    torch.Tensor.backward = self._original_backward  # type: ignore[method-assign]
                else:
                    warnings.warn(
                        "recording_backward() exited while torch.Tensor.backward was "
                        "patched by another party inside the block; leaving the "
                        "interleaved patch in place instead of clobbering it.",
                        UserWarning,
                        stacklevel=2,
                    )
        finally:
            _finalize_grad_streaming(self.trace)


def recording_backward(
    self: Any,
    *,
    save_grads: Any | MissingType = MISSING,
) -> RecordingBackward:
    """Return a context manager that records user-managed backward calls.

    Returns
    -------
    RecordingBackward
        Context manager that patches ``Tensor.backward`` inside the block.
    """
    return RecordingBackward(
        self,
        save_grads=save_grads,
    )
