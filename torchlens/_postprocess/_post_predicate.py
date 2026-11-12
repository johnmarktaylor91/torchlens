"""Value-dependent and lookback ``save=`` predicates on a finished capture.

Torch evaluates ``save=`` while the forward runs, against a ``RecordContext``
that carries the op's live output and the contexts of the last ``lookback``
evaluated ops (8 when ``lookback`` is 0). A finished non-torch capture holds
the same facts: every op's realized payload, its parents and the execution
order. This module rebuilds the per-op ``RecordContext`` and that window and
applies the same rules:

- plain callables and ``tl.where`` predicates get the op's ``RecordContext``
  with ``tensor`` set to the realized payload;
- ``tl.preceded_by(inner)`` matches when a parent op inside the window (any op
  inside the window when the op has no parents) matches ``inner``;
- ``candidate & tl.followed_by(successor)`` keeps each candidate parent of a
  matching successor that is still inside the successor's window, and needs a
  payload-retaining ``lookback_payload_policy`` like torch.
"""

from __future__ import annotations

__tl_layer__ = "L3"

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .._utils._record_group_table import record_member
from ..recording._exceptions import PredicateError

#: Torch's predicate history size when ``lookback`` is 0 (``_configure_predicate_options``).
_DEFAULT_HISTORY: Final = 8


@dataclass(frozen=True)
class _Event:
    """One rebuilt capture event: an op and its predicate context."""

    kind: str
    op: Any
    ctx: Any


def post_save_keep_labels(
    trace: Any,
    predicate: Any,
    *,
    static_kinds: frozenset[str],
    split: tuple[Any, Any] | None,
    build_context: Callable[..., Any],
    lookback: int = 0,
    lookback_payload_policy: str = "metadata_only",
) -> set[str]:
    """Return the op labels a value-dependent or lookback predicate saves.

    Parameters
    ----------
    trace:
        Finished full-save trace.
    predicate:
        Callable or selector tree of post-evaluable kinds.
    static_kinds:
        Selector kinds evaluated on the finished ``Op`` (label, func, module,
        ...); every other leaf is evaluated on the rebuilt ``RecordContext``.
    split:
        ``split_followed_by_conjunction(predicate)``: ``(followed_by, candidate)``
        for ``candidate & tl.followed_by(successor)``, else ``None``.
    build_context:
        Torch's ``RecordContext`` builder (``_capture.projections._build_record_context``),
        passed in so this module stays below the capture layer's import cycle.
    lookback:
        Predicate history size for ``tl.preceded_by`` / ``tl.followed_by``
        (0 means torch's default of 8).
    lookback_payload_policy:
        Torch lookback payload policy; ``"metadata_only"`` retains no candidate.

    Returns
    -------
    set[str]
        Final labels of the ops whose payloads stay public.
    """

    history_size = lookback if lookback > 0 else _DEFAULT_HISTORY
    events = _rebuild_events(trace, history_size, build_context)
    if split is not None:
        return _followed_by_keep(trace, events, split, lookback_payload_policy, static_kinds)
    return {event.op.label for event in events if _matches(predicate, event, static_kinds)}


def _rebuild_events(
    trace: Any, history_size: int, build_context: Callable[..., Any]
) -> list[_Event]:
    """Return one event per op in execution order, each context carrying its window.

    Torch appends a context to the predicate history only for the ops its save
    predicate evaluates during the forward (model inputs, buffers and the
    output nodes added at postprocess are not appended), so the window holds
    the last ``history_size`` such ops. Labels follow torch's matcher contract:
    ``raw_label`` and ``parent_labels`` use raw labels, ``label`` the final one.
    """

    raw_of = _raw_label_index(trace)
    events: list[_Event] = []
    history: list[Any] = []
    for index, op in enumerate(trace._layer_list):
        kind = _event_kind(op)
        window = tuple(history[-history_size:])
        ctx = build_context(
            kind=kind, op_log_or_op_data=_op_data(op, raw_of), history=window, event_index=index
        )
        events.append(_Event(kind, op, ctx))
        if kind == "op":
            history.append(ctx)
    return events


def _raw_label_index(trace: Any) -> dict[str, str]:
    """Map each op's final label, with and without its ``:pass`` suffix, to its raw label.

    ``Op.parents`` names parents without the pass suffix that ``Op.label`` carries.
    """

    index: dict[str, str] = {}
    for op in trace._layer_list:
        raw = _raw_label(op)
        label = str(op.label)
        index[label] = raw
        index.setdefault(label.rsplit(":", 1)[0], raw)
    return index


def _raw_label(op: Any) -> str:
    """Return an op's raw capture label, or its final label when none was kept."""

    return str(record_member(op, "_label_raw", None) or op.label)


def _event_kind(op: Any) -> str:
    """Return the ``RecordContext.kind`` torch gives this op's capture event."""

    if bool(record_member(op, "is_output", False)):
        return "output"
    if bool(record_member(op, "is_input", False)):
        return "input"
    if bool(record_member(op, "is_buffer", False)):
        return "buffer"
    return "op"


def _op_data(op: Any, raw_of: dict[str, str]) -> dict[str, Any]:
    """Return the op fields torch's ``RecordContext`` builder reads for one finished op."""

    return {
        "label": op.label,
        "raw_label": raw_of[str(op.label)],
        "raw_index": record_member(op, "raw_index", None),
        "type": record_member(op, "type", None),
        "type_index": record_member(op, "type_index", None),
        "func_name": record_member(op, "func_name", None),
        "parent_labels": tuple(raw_of.get(str(p), str(p)) for p in (op.parents or ())),
        "shape": record_member(op, "shape", None),
        "dtype": record_member(op, "dtype", None),
        "output_index": record_member(op, "multi_output_index", None),
    }


def _matches(selector: Any, event: _Event, static_kinds: frozenset[str]) -> bool:
    """Evaluate a predicate tree: static leaves on the ``Op``, the rest on the context."""

    kind = getattr(selector, "selector_kind", None)
    if kind in ("and", "or"):
        results = (_matches(child, event, static_kinds) for child in selector.selectors)
        return all(results) if kind == "and" else any(results)
    if kind == "not":
        return not _matches(selector.selector, event, static_kinds)
    if kind in static_kinds:
        return bool(selector(event.op))
    return bool(selector(event.ctx))


def _followed_by_keep(
    trace: Any,
    events: Sequence[_Event],
    split: tuple[Any, Any],
    policy: str,
    static_kinds: frozenset[str],
) -> set[str]:
    """Return the candidate labels ``candidate & tl.followed_by(successor)`` keeps."""

    followed, candidate = split
    by_raw = {_raw_label(event.op): event for event in events}
    keep: set[str] = set()
    for event in events:
        if event.kind != "op" or not _matches(followed.inner, event, static_kinds):
            continue
        if policy == "metadata_only":
            raise PredicateError(
                "tl.followed_by(...) requires lookback_payload_policy other than "
                "'metadata_only' so candidate payloads are retained. "
                "Remedy: pass lookback_payload_policy='detached_raw' or another "
                "payload-retaining policy.",
                code="lookback_payload_policy_conflict",
            )
        recent = {prior.raw_label for prior in event.ctx.recent_ops}
        for parent in event.ctx.parent_labels:
            prior_event = by_raw.get(parent)
            if parent not in recent or prior_event is None:
                _warn_outside_window(parent)
            elif _matches(candidate, prior_event, static_kinds):
                keep.add(prior_event.op.label)
    return keep


def _warn_outside_window(parent: str) -> None:
    """Warn like torch when a successor's parent left the lookback window."""

    warnings.warn(
        f"followed_by parent {parent!r} is outside the lookback window; "
        "increase lookback to make this dependency queryable.",
        RuntimeWarning,
        stacklevel=4,
    )
