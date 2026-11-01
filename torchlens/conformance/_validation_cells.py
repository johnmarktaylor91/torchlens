"""Validation cells of the preview-backend conformance suite.

The replay-validation oracles (``c1_validation_passes``,
``c1_validation_names_reason``, ``c2_validation_after_load``) and the
validation-ready capture every tier runs on: torch replays from saved
arguments, so its capture saves them; the preview backends replay from
their own captured programs and take no extra option.
"""

from __future__ import annotations

from typing import Any

from ..backends._backend_protocol import VALIDATION_REASONS
from ..options import CaptureOptions


def _is_input(op: Any) -> bool:
    """Whether the op is an input leaf (record field ``is_input``)."""

    return bool(getattr(op, "is_input", getattr(op, "is_input_layer", False)))


def _label_keys(op: Any) -> set[str]:
    """Every key a parent or child list may name this op by."""

    return {
        str(key)
        for key in (getattr(op, "_label_raw", None), op.label, getattr(op, "layer_label", None))
        if key
    }


def validation_capture_options(protocol: Any) -> dict[str, Any]:
    """Capture options a backend's replay validation needs on the trace it checks.

    Torch replays each op from its saved arguments, which the default
    capture omits (``save_args=False``), so its validation-ready capture
    saves them. The preview backends replay from their own captured
    programs and refuse ``save_args``; they take no extra option.

    Parameters
    ----------
    protocol:
        The backend's protocol view (``name`` selects the row).

    Returns
    -------
    dict[str, Any]
        Keyword arguments for the adapter's ``trace_with``; empty when the
        backend needs none.
    """

    name = str(getattr(protocol, "name", ""))
    if name in _SAVED_ARGS_VALIDATION_BACKENDS:
        return {"capture": CaptureOptions(save_args=True)}
    return {}


#: Backends whose replay validation reads saved arguments.
_SAVED_ARGS_VALIDATION_BACKENDS: frozenset[str] = frozenset({"torch"})


def capture_for_validation(adapter: Any, net: Any, example: Any) -> Any:
    """Capture ``net`` so the adapter's backend can replay-validate the trace.

    Parameters
    ----------
    adapter:
        The resolved adapter.
    net, example:
        The roster row's model and example input.

    Returns
    -------
    Any
        The trace, captured through ``trace_with`` with
        :func:`validation_capture_options` when the backend needs any, else
        through ``trace``.
    """

    import warnings

    from ._protocol_cases import adapter_protocol

    options = validation_capture_options(adapter_protocol(adapter))
    trace_with = getattr(adapter, "trace_with", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if options and callable(trace_with):
            return trace_with(net, example, **options)
        return adapter.trace(net, example)


def validation_status_problems(status: Any) -> tuple[str, ...]:
    """Why a ``validate`` return is not a passed status with registered reasons."""

    state = getattr(status, "state", None)
    if state is None:
        return (f"validate returned {type(status).__name__} ({status!r}), not a status object",)
    problems: list[str] = []
    if state != "passed":
        message = str(getattr(status, "message", ""))[:200]
        problems.append(f"state={state!r}: {message}")
    counts = getattr(status, "unverified_reason_counts", {}) or {}
    unknown = sorted(str(key) for key in counts if str(key) not in VALIDATION_REASONS)
    if unknown:
        problems.append(f"unverified reasons outside VALIDATION_REASONS: {unknown}")
    return tuple(problems)


def tamper_target(trace: Any) -> Any | None:
    """The op whose saved payload the naming cell tampers: an internal op with a consumer."""

    candidates = [
        op
        for op in trace._layer_list
        if not _is_input(op)
        and getattr(op, "out", None) is not None
        and tuple(getattr(op, "children", None) or ())
    ]
    if candidates:
        return candidates[0]
    fallback = [op for op in trace._layer_list if getattr(op, "out", None) is not None]
    return fallback[0] if fallback else None


def tampered_payload(value: Any) -> Any:
    """``value + 1`` in the framework's own arithmetic (every supported array type adds a scalar)."""

    return value + 1


def _consumer_keys(target: Any, trace: Any | None) -> set[str]:
    """Every key the verdict may use for the tampered op's direct consumers.

    Child lists name an op by one key (its layer label); a decision may use
    another (its pass label), so each child resolves to the full key set of
    the trace op it names, falling back to the raw child key.
    """

    children = {str(key) for key in (getattr(target, "children", None) or ())}
    keys = set(children)
    for op in getattr(trace, "_layer_list", None) or ():
        op_keys = _label_keys(op)
        if op_keys & children:
            keys |= op_keys
    return keys


def names_reason_problems(status: Any, target: Any, trace: Any | None = None) -> tuple[str, ...]:
    """Why a failed status does not name the tampered op (or its consumer) with a reason."""

    state = getattr(status, "state", None)
    if state is None:
        return (f"validate returned {type(status).__name__} ({status!r}), not a status object",)
    if state != "failed":
        return (f"tampered payload on {target.label} validated as {state!r}",)
    accepted = _label_keys(target) | _consumer_keys(target, trace)
    decisions = list(getattr(status, "decisions", ()) or ())
    naming = [
        row
        for row in decisions
        if isinstance(row, dict) and str(row.get("op_label", "")) in accepted
    ]
    problems: list[str] = []
    if not naming:
        problems.append(f"no decision names {target.label} or its consumers")
    else:
        reasons = {str(row.get("reason", "")) for row in naming}
        if not reasons & VALIDATION_REASONS:
            problems.append(f"decision reasons {sorted(reasons)} are not in VALIDATION_REASONS")
    message = str(getattr(status, "message", ""))
    if not any(label in message for label in accepted):
        problems.append(f"message does not name {target.label}: {message[:160]!r}")
    return tuple(problems)
