"""The pristine reading of a Trace, derived from the evidence it carries.

A trace is pristine when nothing ever intervened on it: no capture-time
``intervene=``, no staged hook or value replacement, no ``do()``/``set()``/
region edit, no hooked rerun or run, no direct write. The reading is DERIVED at
read time from records that already persist with the trace (``tl.save`` /
``tl.load``, pickle, ``copy.deepcopy``, ``fork()``), never stored as a flag of
its own, so no later call can make a touched trace read pristine again: the
records it is derived from are append-only, and ``Trace.state`` (the mutable
operational state machine, which an assignment or a fresh ``__init__`` sets to
``TraceState.PRISTINE``) is read only as one more piece of touch evidence.

This is a different sense of "pristine" from ``torchlens.validation``'s
pristine-torch oracle, which means running on unwrapped torch functions.

The evidence, all persisted (``FieldPolicy.KEEP``) unless noted:

- ``Trace.intervention_audit``: every row is written by an intervention door
  (capture ``intervene=``, ``do()``, the experiment and vary engines).
- ``Trace.state_history``: the operation rows of :data:`TOUCH_OPERATIONS`;
  ``rerun`` and ``append`` rows count only when they ran hooks.
- ``Trace.state`` in :data:`TOUCH_STATES` (only mutation paths set them).
- ``Trace._has_direct_writes``.
- Op intervention marks (:data:`OP_TOUCH_FIELDS`): a replaced or edited value
  recorded on the graph itself.
- The staged intervention spec (session-time, dropped at save; the staging
  call's ``state_history`` row is what persists).

Absent evidence is never read as pristine. Artifacts stamped below
:data:`PRISTINE_EVIDENCE_MIN_TLSPEC` predate the persisted intervention audit,
and non-torch backends record none of this evidence for their interventions,
so for them an untouched-looking trace reads ``None`` (not recorded), and any
touch evidence they do carry still reads ``False``. The reading is of what the
artifact carries: a hand-edited artifact that deletes its own records is out of
scope, as it is for every other persisted record.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .._trace_state import TraceState

#: First tlspec version whose writer persists the intervention event audit.
PRISTINE_EVIDENCE_MIN_TLSPEC = 9

#: ``state_history`` operations that always mean the trace was touched.
TOUCH_OPERATIONS = frozenset(
    {"set", "attach_hooks", "do", "region_do", "replay", "intervention_event"}
)

#: ``state_history`` operations that touch only when they ran hooks.
HOOKED_OPERATIONS = frozenset({"rerun", "append"})

#: Operational states that only mutation paths set.
TOUCH_STATES = frozenset(
    {TraceState.REPLAY_PROPAGATED, TraceState.DIRECT_WRITE_DIRTY, TraceState.LIVE_CAPTURED}
)

#: Op fields that record an intervention on the graph itself.
OP_TOUCH_FIELDS = (
    "interventions",
    "intervention_replaced",
    "edge_substitutions",
    "edge_replacement_stamps",
)


def pristine_touch_evidence(trace: Any) -> tuple[str, ...]:
    """Return the kinds of touch evidence a trace carries, in a fixed order.

    Parameters
    ----------
    trace:
        Live or loaded ``Trace``.

    Returns
    -------
    tuple[str, ...]
        Empty when no evidence of a touch is found; otherwise the evidence
        kinds found (``"intervention_audit"``, ``"state_history"``,
        ``"state"``, ``"direct_writes"``, ``"op_marks"``, ``"staged_spec"``).
    """

    found: list[str] = []
    if getattr(trace, "intervention_audit", None):
        found.append("intervention_audit")
    if any(_is_touch_operation(row) for row in getattr(trace, "state_history", None) or ()):
        found.append("state_history")
    if getattr(trace, "state", None) in TOUCH_STATES:
        found.append("state")
    if getattr(trace, "_has_direct_writes", False):
        found.append("direct_writes")
    if _any_op_marked(getattr(trace, "layer_list", None) or ()):
        found.append("op_marks")
    if _spec_has_entries(trace.__dict__.get("_intervention_spec")):
        found.append("staged_spec")
    return tuple(found)


def derive_pristine(trace: Any) -> bool | None:
    """Return the derived pristine reading of a trace.

    Parameters
    ----------
    trace:
        Live or loaded ``Trace``.

    Returns
    -------
    bool | None
        ``False`` when any touch evidence is found; ``None`` when the trace
        cannot carry the evidence (an artifact older than tlspec
        :data:`PRISTINE_EVIDENCE_MIN_TLSPEC`, or a non-torch backend);
        otherwise ``True``.
    """

    if pristine_touch_evidence(trace):
        return False
    if not _evidence_recorded(trace):
        return None
    return True


def _evidence_recorded(trace: Any) -> bool:
    """Return whether this trace's producer persists the full touch evidence."""

    if str(getattr(trace, "backend", "torch")) != "torch":
        return False
    version = getattr(trace, "tlspec_version", None)
    return isinstance(version, int) and version >= PRISTINE_EVIDENCE_MIN_TLSPEC


def _is_touch_operation(row: Any) -> bool:
    """Return whether one ``state_history`` row records a touch."""

    if not isinstance(row, Mapping):
        return False
    op = row.get("op")
    if op in TOUCH_OPERATIONS:
        return True
    if op in HOOKED_OPERATIONS:
        return _positive(row.get("hook_count")) or _positive(row.get("hook_fire_count"))
    return False


def _positive(value: Any) -> bool:
    """Return whether a recorded count is a positive integer."""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _any_op_marked(ops: Iterable[Any]) -> bool:
    """Return whether any op records an intervention on its own fields."""

    return any(getattr(op, name, None) for op in ops for name in OP_TOUCH_FIELDS)


def _spec_has_entries(spec: Any) -> bool:
    """Return whether a staged intervention spec holds any hook or value entry."""

    if spec is None:
        return False
    return bool(getattr(spec, "hook_specs", None) or getattr(spec, "target_value_specs", None))


__all__ = ["derive_pristine", "pristine_touch_evidence"]
