"""Per-op decisions the TensorFlow validator takes before replaying an op.

Two cases bypass the replay-and-perturb check, each on recorded evidence only:

* a variable read verified as a parameter source (ruling X4, see
  :mod:`._variable_reads`), which validates or fails on its own;
* a pure-replay op whose own payload, or a parent's payload, is absent because
  the user's ``save=`` selector declined it. The op is ``exempted`` with reason
  ``not_saved_by_user``, the same rule the PyTorch validator applies to a
  predicate-save omission. The proof is the capture's ``payload_declined`` flag
  (the selector's decision at capture); for a constant source parent of an eager
  trace it is the source op's ``has_saved_activation``, which the eager capture
  sets to the selector's decision (not to payload presence). A payload dropped for
  any other reason (the snapshot size cap, an unsupported dtype, a release) keeps
  the ordinary replay check and its failure. A FuncGraph (static) trace sets
  ``has_saved_activation`` from payload presence, so its constant parents are
  never read as declined.

Every other op goes to the validator's ordinary path unchanged.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from typing import Any

from .._validation_shared import replay_payload
from ._variable_reads import parameter_read_decision, read_failures

_REASON_NOT_SAVED = "not_saved_by_user"


def _declined(capture: Any, op: Any) -> bool:
    """Whether ``op``'s payload is absent because the save selector declined it."""

    return (
        capture is not None
        and bool(getattr(capture, "payload_declined", False))
        and replay_payload(op) is None
    )


def _declined_constant(source: Any, eager: bool) -> bool:
    """Whether an eager trace's constant source has no payload because the selector declined it."""

    return (
        eager
        and source is not None
        and str(getattr(source, "func_name", "")) == "constant"
        and not bool(getattr(source, "has_saved_activation", True))
        and replay_payload(source) is None
    )


def _declined_labels(
    trace: Any,
    op: Any,
    capture: Any,
    lookups: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> list[str]:
    """Raw labels of ``op`` and its parents whose payload the selector declined."""

    ops_by_label, captures_by_label = lookups
    eager = not hasattr(trace, "_tf_static_region_labels")
    labels = [capture.label_raw] if _declined(capture, op) else []
    for item in sorted(capture.inputs, key=lambda entry: entry.input_index):
        producer, source = item.producer_label_raw, getattr(item, "source_label_raw", None)
        if producer is not None:
            declined = _declined(captures_by_label.get(producer), ops_by_label.get(producer))
        else:
            declined = source is not None and _declined_constant(ops_by_label.get(source), eager)
        parent = producer or source
        if declined and parent not in labels:
            labels.append(str(parent))
    return labels


def not_saved_decision(
    trace: Any,
    op: Any,
    capture: Any,
    lookups: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """The ``not_saved_by_user`` exemption for ``op``, or ``None`` to replay it.

    Parameters
    ----------
    trace
        TensorFlow trace being validated.
    op
        Materialized TensorFlow op the validator would replay.
    capture
        Its callback capture record.
    lookups
        Materialized ops keyed by raw and public labels, and callback captures
        keyed by raw label.

    Returns
    -------
    Mapping[str, Any] | None
        An ``exempted`` replay decision naming every declined label, or ``None``
        when every payload the replay needs was saved or was dropped for a reason
        other than the save selector.
    """

    if capture is None:
        return None
    declined = _declined_labels(trace, op, capture, lookups)
    if not declined:
        return None
    return {
        "op_label": str(getattr(op, "label", capture.label_raw)),
        "func_name": str(getattr(op, "func_name", capture.op_type)),
        "phase": "replay",
        "decision": "exempted",
        "reason": _REASON_NOT_SAVED,
        "justification": (
            "save selector declined the replay payload of "
            f"{', '.join(declined)} (declined at capture, payload absent)"
        ),
    }


def early_decision(
    trace: Any,
    op: Any,
    op_class: str,
    replayable: Collection[str],
    lookups: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """The decision taken before replay for ``op``, or ``None`` for the ordinary path.

    Parameters
    ----------
    trace
        TensorFlow trace being validated.
    op
        Materialized op.
    op_class
        The validator's class for ``op`` (``value`` ops may be replayed).
    replayable
        Op types the validator replays.
    lookups
        Materialized ops keyed by raw and public labels, and callback captures
        keyed by raw label (as :func:`not_saved_decision` takes them).

    Returns
    -------
    Mapping[str, Any] | None
        A verified variable read's decision for a non-value op, a
        ``not_saved_by_user`` exemption for a replayable value op, else ``None``.
    """

    _ops_by_label, captures_by_label = lookups
    capture = captures_by_label.get(str(getattr(op, "_label_raw", "")))
    if op_class != "value":
        return parameter_read_decision(trace, op, capture)
    if str(getattr(op, "func_name", "")) not in replayable:
        return None
    return not_saved_decision(trace, op, capture, lookups)


def tally_early(
    label: str,
    decision: Mapping[str, Any],
    failures: list[str],
    exempted_reasons: Counter[str],
) -> None:
    """Add an early decision's failure or exemption to the validator's tallies."""

    failures.extend(read_failures(label, decision))
    if decision.get("decision") == "exempted":
        exempted_reasons[str(decision.get("reason"))] += 1
