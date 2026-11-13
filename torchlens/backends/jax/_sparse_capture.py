"""Selective ``save=`` decided at JAX capture time: unselected values are never retained.

A static-label selector (``tl.label``, ``tl.func``, ``tl.label_contains`` and
their boolean composites) names ops by their final labels, which exist only
after the capture pipeline runs. A JAX capture whose ``save=`` is such a
selector therefore runs twice:

1. a structure pass keeps no equation values (each is freed after its last use)
   and yields the final labels; the selector is evaluated on its ops;
2. a value pass keeps the values of the selected equations only.

Both passes interpret the same closed jaxpr with the same concrete inputs, so
labels, edges and taken branches agree. The returned trace holds payloads for
the selected ops only. Its replay validation follows torch's selective-save
rule: an op the selector excluded is ``exempted:not_saved_by_user`` (proved by
the selector's exact negative result on that op, recorded in
``trace.annotations["jax_selective_save"]``), and a selected op whose data
parent was excluded is ``unverified:missing_saved_args`` (torch's code for the same case).
"""

from __future__ import annotations

import functools
import inspect
import math
from collections.abc import Callable, Mapping
from typing import Any

from ..._ir.refs import TensorRef
from .._finalize import value_nbytes
from ._retention import KEEP_NONE, is_released, keep_only, pending_retention
from ._trace_graph_helpers import _data_parent_labels
from ._validation_log import _JaxValidationLog, _JaxVerdict, _record_jax_decision

NOT_SAVED_BY_USER = "not_saved_by_user"
MISSING_SAVED_ARGS = "missing_saved_args"

#: Options whose features read every equation value after capture; a capture
#: that sets one keeps the plain full capture and hides payloads afterwards.
_NEEDS_EVERY_VALUE = ("backward_ready", "save_grads", "grad_options", "intervene")

#: ``trace.annotations`` key holding a sparse trace's selective-save evidence:
#: the selector's repr and the raw labels of the ops it excluded.
EVIDENCE_KEY = "jax_selective_save"


def sparse_save_passes(
    is_static: Callable[[Any], bool],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return a decorator making a static-label ``save=`` capture retain only its ops.

    Parameters
    ----------
    is_static
        Whether a ``save=`` value is a selector of static-label kinds only (the
        test ``apply_static_label_save_policy`` uses to evaluate it on ops).

    Returns
    -------
    Callable
        Decorator for ``JAXBackend.capture_trace``: eligible selective captures
        run as a structure pass plus a value pass, every other capture runs
        unchanged.
    """

    return functools.partial(_wrap_capture, is_static=is_static)


def _wrap_capture(
    capture_trace: Callable[..., Any], *, is_static: Callable[[Any], bool]
) -> Callable[..., Any]:
    """Wrap one capture method (see :func:`sparse_save_passes`)."""

    accepted = frozenset(inspect.signature(capture_trace).parameters)

    @functools.wraps(capture_trace)
    def capture(
        self: Any, model: Any, input_args: Any, input_kwargs: Any = None, **options: Any
    ) -> Any:
        selector: Any = options.get("save")
        if not _eligible(selector, options, accepted, is_static):
            return capture_trace(self, model, input_args, input_kwargs, **options)
        with pending_retention(KEEP_NONE):
            structure = capture_trace(
                self, model, input_args, input_kwargs, **{**options, "save": None}
            )
        keep = selected_capture_indexes(structure, selector)
        del structure
        with pending_retention(keep):
            trace = capture_trace(self, model, input_args, input_kwargs, **options)
        trace.annotations[EVIDENCE_KEY] = {
            "selector": repr(selector),
            "excluded_raw_labels": sorted(
                str(op._label_raw)
                for op in trace._layer_list
                if getattr(op, "_label_raw", None) is not None and not bool(selector(op))
            ),
        }
        return trace

    return capture


def _eligible(
    selector: Any,
    options: Mapping[str, Any],
    accepted: frozenset[str],
    is_static: Callable[[Any], bool],
) -> bool:
    """Return whether a capture can decide retention from a structure pass."""

    return ineligible_reason(selector, options, accepted, is_static) is None


def ineligible_reason(
    selector: Any,
    options: Mapping[str, Any],
    accepted: frozenset[str],
    is_static: Callable[[Any], bool],
) -> str | None:
    """Return why a capture keeps the full capture, or ``None`` when it can run sparse.

    Parameters
    ----------
    selector
        The ``save=`` value.
    options
        Keyword options of the capture call.
    accepted
        Option names the capture method declares.
    is_static
        Whether ``selector`` is a static-label selector.

    Returns
    -------
    str | None
        A short reason, or ``None`` when the two-pass capture applies.
    """

    if selector is None or not is_static(selector):
        return "save= is not a static-label selector"
    unknown = sorted(n for n, v in options.items() if n not in accepted and _is_set(v))
    if unknown:
        return f"options outside the capture signature: {unknown}"
    if _is_set(options.get("lookback")):
        return "lookback keeps payloads of unselected predecessors"
    grouped = options.get("capture")
    for name in _NEEDS_EVERY_VALUE:
        for value in (options.get(name), getattr(grouped, name, None)):
            if _is_set(value):
                return f"{name} reads every equation value"
    return None


def _is_set(value: Any) -> bool:
    """Return whether an option value asks for a feature (``MISSING``, ``None``, ``False``, ``0`` do not)."""

    if value is None or value is False or (type(value) is int and value == 0):
        return False
    return type(value).__name__ != "MissingType"


def selected_capture_indexes(structure: Any, selector: Any) -> Any:
    """Return the retention keeping the captures of the ops ``selector`` selects.

    Parameters
    ----------
    structure
        The structure pass's finished trace.
    selector
        The static-label ``save=`` selector.

    Returns
    -------
    Retention
        Capture indexes whose values the value pass keeps.
    """

    selected = {
        str(op._label_raw)
        for op in structure._layer_list
        if getattr(op, "_label_raw", None) is not None and bool(selector(op))
    }
    raw_by_index = getattr(structure, "_jax_capture_index_to_raw_op_label", {})
    return keep_only(index for index, raw in raw_by_index.items() if raw in selected)


def jax_tensor_ref(value: object, label_raw: str, *, is_tensor: bool) -> TensorRef:
    """Build the tensor reference of one JAX event output.

    Parameters
    ----------
    value
        Captured output: a JAX array, another value, or a released placeholder.
    label_raw
        Raw TorchLens label.
    is_tensor
        Whether ``value`` is a JAX array.

    Returns
    -------
    TensorRef
        Payload-bearing for a retained array; shape, dtype and size without a
        payload for a released value.
    """

    if is_released(value) and not isinstance(value, tuple):
        shape = tuple(getattr(value, "shape", ()))
        dtype = getattr(value, "dtype", None)
        itemsize = int(getattr(dtype, "itemsize", 0) or 0)
        memory = math.prod(shape) * itemsize
        return TensorRef(label_raw, shape, str(dtype), None, None, memory, None, None, None)
    if is_released(value):
        return TensorRef(label_raw, None, None, None, None, 0, None, None, None)
    if not is_tensor:
        return TensorRef(label_raw, None, None, None, None, 0, value, None, str(id(value)))
    array: Any = value
    return TensorRef(
        label_raw=label_raw,
        shape=tuple(array.shape),
        dtype=str(array.dtype),
        device=str(array.device),
        requires_grad=None,
        memory=value_nbytes(array),
        payload=array,
        blob_ref=None,
        backend_handle_id=str(id(array)),
    )


def sparse_save_verdict(
    trace: Any,
    op: Any,
    ops_by_label: Mapping[str, Any],
    hidden: Mapping[str, tuple[Any, ...]],
    log: _JaxValidationLog,
) -> bool:
    """Record the selective-save decision for one equation op of a sparse trace.

    Parameters
    ----------
    trace
        Trace being validated.
    op
        The equation op.
    ops_by_label
        Ops keyed by raw and final labels.
    hidden
        Replay-only payloads keyed by raw and final labels.
    log
        Validation log receiving the decision.

    Returns
    -------
    bool
        ``True`` when a decision was recorded (the op is not replayed);
        ``False`` for a plain trace or an op whose replay data is all present.
    """

    evidence = getattr(trace, "annotations", {}).get(EVIDENCE_KEY)
    if not isinstance(evidence, Mapping):
        return False
    excluded = frozenset(evidence.get("excluded_raw_labels", ()))
    selector = evidence.get("selector")
    label = str(getattr(op, "label", getattr(op, "_label_raw", "")))
    func_name = str(getattr(op, "func_name", ""))
    if not _payload_present(op, hidden):
        if str(getattr(op, "_label_raw", "")) not in excluded:
            return False
        reason = f"selector {selector} excluded op {label} ({func_name}); payload never retained"
        _record_jax_decision(
            log, label, func_name, _JaxVerdict("replay", "exempted", NOT_SAVED_BY_USER), reason
        )
        counts = log.exempted_reason_counts
        counts[NOT_SAVED_BY_USER] = counts.get(NOT_SAVED_BY_USER, 0) + 1
        return True
    missing = sorted(
        parent
        for parent in _data_parent_labels(op)
        if not _payload_present(ops_by_label[parent], hidden)
        and str(getattr(ops_by_label[parent], "_label_raw", "")) in excluded
    )
    if not missing:
        return False
    reason = f"data parents {missing} were excluded by selector {selector}"
    _record_jax_decision(
        log, label, func_name, _JaxVerdict("replay", "unverified", MISSING_SAVED_ARGS), reason
    )
    return True


def _payload_present(op: Any, hidden: Mapping[str, tuple[Any, ...]]) -> bool:
    """Return whether ``op`` has a public or replay-only payload."""

    if getattr(op, "has_saved_activation", False) and getattr(op, "out", None) is not None:
        return True
    return any(
        label is not None and label in hidden
        for label in (getattr(op, "_label_raw", None), getattr(op, "label", None))
    )


def sparse_counts(log: _JaxValidationLog | None) -> tuple[int, int, dict[str, int]]:
    """Return (not replayed, unverified, unverified reason counts) of selective-save decisions."""

    if log is None:
        return 0, 0, {}
    unverified = sum(
        1
        for decision in log.decisions
        if decision["decision"] == "unverified" and decision["reason"] == MISSING_SAVED_ARGS
    )
    exempted = log.exempted_reason_counts.get(NOT_SAVED_BY_USER, 0)
    reasons = {MISSING_SAVED_ARGS: unverified} if unverified else {}
    return exempted + unverified, unverified, reasons
