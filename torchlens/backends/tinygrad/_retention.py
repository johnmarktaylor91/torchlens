"""Retention decided at capture for a tinygrad ``save=`` selector.

A capture with a static-label ``save=`` selector (label, func, module, output,
contains, in_module and their ``&``, ``|``, ``~`` combinations) does not
realize every recorded value. The forward's graph is read and emitted with
lazy payloads; once the trace is finalized the selector picks the ops to keep,
and one ``realize_once`` realizes only those values and the outputs. Every
value the selector drops stays fused into its consumers' kernels and is never
written to a buffer.

Validation of such a sparse trace follows torch's selective-save semantics: an
op the selector dropped is ``exempted:not_saved_by_user``, a kept op with a
dropped parent is ``unverified:parent_payload_missing`` (so the trace is
``unverified``, not passed), and every other kept op replays exactly as in a
full capture.

A capture with pre-forward random values, edits or a backward pass, or with a
shared value tinygrad has not realized yet, realizes everything as before and
filters afterwards (:func:`torchlens._postprocess._selective_save.apply_static_label_save_policy`).
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .._protocols.tinygrad import TinygradProtocol
from ._rules import ReplayVerdict

__tl_layer__ = "L2"

RETENTION_FULL = "full"
"""A capture realized with every recorded value (no deferral)."""
RETENTION_KEPT = "kept"
"""A deferred capture's value the selector kept: realized in the single realize."""
RETENTION_DROPPED = "dropped"
"""A deferred capture's value the selector dropped: never realized, payload gone."""


@dataclass(frozen=True)
class DeferredRealize:
    """What a deferred capture realizes once its trace is finalized.

    Parameters
    ----------
    outputs
        Flat output tensors (and hook tensors) the single realize must include.
    """

    outputs: tuple[Any, ...]


def may_defer(session: Any, plan: Any) -> bool:
    """Whether a capture may decide retention after emission instead of realizing now.

    Only a static-label selector qualifies (its decision needs labels and
    modules, never values), and only a capture without pre-forward random
    values, edits or a backward pass: those need every value in the realize.
    """

    if session is None or not session.is_static_selector(session.save_predicate):
        return False
    if plan.random_values or plan.edit_sites:
        return False
    return not session.backward_ready and session.tinygrad_backward is None


def defer_realize(
    session: Any,
    targets: Mapping[int, Any],
    owned: frozenset[int],
    outputs: Sequence[Any],
) -> bool:
    """Record a deferred realize on ``session`` when every shared value is a buffer.

    ``targets`` and ``owned`` are the lazy realization targets
    (``_realization_targets``). A target the capture does not own shares the
    UOp itself; emission copies it at once. That copy reads no forward value
    when tinygrad already holds its buffer or the UOp is a ``CONST`` (a
    scalar such as a reshape's shape argument, copied the same way in a full
    capture). Any other shared value would compute its ancestors early, so it
    keeps the capture on the realize-everything path.
    """

    for key, tensor in targets.items():
        if key not in owned and not _copy_reads_no_value(tensor.uop):
            return False
    session.deferred = DeferredRealize(outputs=tuple(outputs))
    return True


def _copy_reads_no_value(uop: Any) -> bool:
    """Whether copying a shared ``uop`` needs no forward value computed first."""

    if bool(getattr(uop, "is_realized", False)):
        return True
    return getattr(getattr(uop, "op", None), "name", None) == "CONST"


def realize_retained(trace: Any, session: Any, after_realize: Callable[[Any], None]) -> None:
    """Pick the kept ops, realize them with the outputs once, and drop the rest.

    Runs on the finalized trace of a deferred capture (``session.deferred``);
    a no-op otherwise. ``after_realize`` is the hook run after the single
    realize (random-state counters).
    """

    deferred = getattr(session, "deferred", None)
    if deferred is None:
        return
    session.deferred = None
    predicate = session.save_predicate
    keep = {op.label for op in trace._layer_list if bool(predicate(op))}
    captures = tuple(getattr(trace, "tinygrad_uop_captures", ()) or ())
    kept_tensors = _kept_payload_tensors(trace, captures, keep)
    TinygradProtocol().realize_once([*kept_tensors, *deferred.outputs])
    after_realize(session)
    labels = {op._label_raw: op.label for op in trace._layer_list}
    trace.tinygrad_uop_captures = tuple(
        _retained_capture(capture, labels.get(capture.label_raw) in keep) for capture in captures
    )
    # The zero-match warning is the public filter's to give, once, afterwards.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.apply_keep_set(trace, keep, backend_name="tinygrad")


def _kept_payload_tensors(trace: Any, captures: Sequence[Any], keep: set[str]) -> list[Any]:
    """The distinct lazy payload tensors of kept ops (public payloads and snapshots)."""

    tensors: dict[int, Any] = {}
    kept_raw = set()
    for op in trace._layer_list:
        if op.label in keep:
            kept_raw.add(op._label_raw)
            if op.out is not None:
                tensors[id(op.out)] = op.out
    for capture in captures:
        if capture.label_raw in kept_raw and capture.payload_snapshot is not None:
            tensors[id(capture.payload_snapshot)] = capture.payload_snapshot
    return list(tensors.values())


def _retained_capture(capture: Any, kept: bool) -> Any:
    """``capture`` marked kept, or marked dropped with its lazy payload released."""

    if kept:
        return replace(capture, retention=RETENTION_KEPT)
    return replace(capture, retention=RETENTION_DROPPED, payload_snapshot=None)


def is_sparse(captures: Sequence[Any]) -> bool:
    """Whether the captures come from a deferred (sparse) capture."""

    return any(getattr(c, "retention", RETENTION_FULL) != RETENTION_FULL for c in captures)


def _decision(capture: Any, ops_by_raw_label: Mapping[str, Any]) -> tuple[str, str | None]:
    """Torch's selective-save decision for one capture: replay, exempted or unverified."""

    if getattr(capture, "retention", RETENTION_FULL) == RETENTION_FULL:
        return "replay", None
    op = ops_by_raw_label.get(capture.label_raw)
    if op is None:
        return "replay", None
    if not bool(getattr(op, "has_saved_activation", False)):
        return "exempted", "not_saved_by_user"
    for label in capture.parent_labels:
        parent = ops_by_raw_label.get(label)
        if parent is not None and not bool(getattr(parent, "has_saved_activation", False)):
            return "unverified", "parent_payload_missing"
    return "replay", None


def replay_captures(
    backend: Any,
    captures: Sequence[Any],
    ops_by_raw_label: dict[str, Any],
    hidden_outputs_by_label: dict[str, Any],
) -> ReplayVerdict:
    """Replay every checkable capture; the first failure ends the walk.

    In a full capture every capture is checkable; in a sparse one only kept
    ops whose parents were kept (:func:`_decision`).
    """

    checked = 0
    for index, capture in enumerate(captures):
        if _decision(capture, ops_by_raw_label)[0] != "replay":
            continue
        failure = backend._uop_capture_failure(
            index, capture, ops_by_raw_label, hidden_outputs_by_label
        )
        if failure is not None:
            return failure
        checked += 1
    return ReplayVerdict.ok(checked)


def sparse_status_counts(trace: Any, verdict: ReplayVerdict, source: str) -> dict[str, Any] | None:
    """``ValidationStatus.from_replay_counts`` arguments of a sparse trace, else ``None``.

    The counts give torch's status for the same selective save: exempted ops
    leave the state alone, any unverified op makes it ``unverified``, and a
    replay failure makes it ``failed``. ``source`` is ``"live"`` or
    ``"loaded"``, as the backend reads it from the trace.
    """

    captures = tuple(getattr(trace, "tinygrad_uop_captures", ()) or ())
    if not is_sparse(captures):
        return None
    ops_by_raw_label = {getattr(op, "_label_raw", ""): op for op in trace._layer_list}
    decisions: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {"exempted": Counter(), "unverified": Counter()}
    for capture in captures:
        kind, reason = _decision(capture, ops_by_raw_label)
        if kind == "replay" or reason is None:
            continue
        counts[kind][reason] += 1
        decisions.append(_decision_record(ops_by_raw_label[capture.label_raw], kind, reason))
    return {
        "backend": "tinygrad",
        "source": source,
        "num_replayed_ops": verdict.num_replayed if verdict.passed else 0,
        "num_unverified_ops": sum(counts["unverified"].values()),
        "num_failed_ops": 0 if verdict.passed else 1,
        "payload_load_status": getattr(trace, "payload_load_status", None),
        "unverified_reason_counts": dict(counts["unverified"]),
        "exempted_reason_counts": dict(counts["exempted"]),
        "decisions": (*decisions, *verdict.decisions()),
    }


def _decision_record(op: Any, kind: str, reason: str) -> dict[str, Any]:
    """One per-op decision in torch's ``ValidationDecision.as_dict`` shape."""

    record: dict[str, Any] = {
        "op_label": str(getattr(op, "label", op._label_raw)),
        "func_name": getattr(op, "func_name", None),
        "phase": "replay",
        "decision": kind,
        "reason": reason,
    }
    if kind == "exempted":
        record["justification"] = (
            "the save= selector excluded this op at capture, so its value was never "
            f"realized (op_label={record['op_label']}, func_name={record['func_name']}, "
            "has_saved_activation=False)"
        )
    return record
