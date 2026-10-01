"""The ONE FireRecord builder and the InterventionEvent envelope (C03).

Ledger memo D1a / edits memo D12 / leverage B6, one substrate: every
intervention write door mints its per-fire :class:`FireRecord` through ONE
builder (uniform seed/notes/timestamp lift -- before this, 1 of 8
construction sites lifted disclosure notes), and every top-level intervention
TRANSACTION writes one envelope through ONE writer, so the trace-level record
is never empty after an attempt (the fire-evidence rule) and two transactions
differing in any edit parameter are distinguishable (the misfire /
distinguishability oracle: ``noise(std=0.1)`` vs ``noise(std=0.9)`` carried
byte-identical audit rows before this module existed).

Persistence boundary (pre-C07, disclosed): the persisted
``trace.intervention_audit`` row family is CLOSED at load validation
(``_io/forgery_validation.py``), so the envelope's rich fields ride the
free-form persisted ``state_history`` stream (``op="intervention_event"``
rows) while the canonical audit gains a closed-schema ACT row per
transaction. The full ``intervention_event_v2`` persisted row family is
declared in ``sprint/field_intent.tsv`` for the C07 coordinated schema write;
nothing here edits schema files. Every spelling DOCUMENTED-UNSTABLE pending
naming-session ratification.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Literal

from .types import FireRecord, HelperSpec

_UNSET: Any = object()

#: Closed transaction-status vocabulary (documented-unstable).
EventStatus = Literal["fired", "no_fire", "error"]

#: Closed lane vocabulary for the envelope (surgery memo lanes; ``bind`` is
#: reserved for F01).
EventLane = Literal["replay", "rerun", "capture", "live_hook", "set_only", "bind"]


def build_fire_record(
    *,
    target_label: str,
    call_label: str | None = None,
    func_call_id: int | None = None,
    container_path: tuple[Any, ...] = (),
    engine: str | None = None,
    helper: HelperSpec | None = None,
    site_label: str | None = None,
    timing: Literal["pre", "post"] | None = None,
    direction: Literal["forward", "backward"] | None = None,
    helper_name: str | None = None,
    run_ctx: dict[str, Any] | None = None,
    previous_notes: tuple[Any, ...] = (),
    seed: Any = _UNSET,
    determinism_note: str | None = None,
    timestamp: float | None = None,
    backward_pass_index: int | None = None,
    call_index: int | None = None,
    grad_kind: Literal["grad_input", "grad_output"] | None = None,
    tuple_index: int | None = None,
    edge_address: tuple | None = None,
    replaced: bool | None = None,
) -> FireRecord:
    """Mint one FireRecord with the uniform disclosure lift (the ONE builder).

    Every write door routes here. The builder owns exactly the disclosures
    the eight historical construction sites applied inconsistently:

    - ``seed``: lifted from the helper's declared kwargs when the caller does
      not pass one (only the live-forward site did this before).
    - ``determinism_note``: new ``ledger_notes`` appended to ``run_ctx``
      during the fire, beyond ``previous_notes`` (only the live-forward site
      lifted these; replay/edge/param fires dropped them).
    - ``timestamp``: always stamped (monotonic clock), which is what lets a
      door bound the records of ONE transaction.

    Parameters mirror :class:`FireRecord`; ``run_ctx``/``previous_notes``
    feed the note lift and are not stored.
    """

    if helper_name is None and helper is not None:
        helper_name = helper.helper_name
    if seed is _UNSET:
        helper_kwargs = dict(helper.kwargs) if helper is not None else {}
        seed = helper_kwargs.get("seed")
    if determinism_note is None and run_ctx is not None:
        new_notes = tuple(run_ctx.get("ledger_notes", ()))[len(previous_notes) :]
        if new_notes:
            determinism_note = "; ".join(str(note) for note in new_notes)
    return FireRecord(
        target_label=target_label,
        call_label=call_label,
        func_call_id=func_call_id,
        container_path=container_path,
        engine=engine,
        helper=helper,
        site_label=site_label,
        timing=timing,
        direction=direction,
        helper_name=helper_name,
        seed=seed,
        determinism_note=determinism_note,
        timestamp=time.monotonic() if timestamp is None else timestamp,
        backward_pass_index=backward_pass_index,
        call_index=call_index,
        grad_kind=grad_kind,
        tuple_index=tuple_index,
        edge_address=edge_address,
        replaced=replaced,
    )


def _prior_event_rows(trace: Any) -> list[dict[str, Any]]:
    """Return the envelope rows already recorded on this trace, in order."""

    return [
        row
        for row in getattr(trace, "state_history", ())
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]


def _event_lineage(trace: Any, prior: list[dict[str, Any]]) -> str:
    """Return this trace's event-lineage id (minted once, in the rows).

    Random at first use, NEVER a hash of timestamps/labels/values (ledger
    D1a). The lineage lives IN the first envelope row -- forks copy
    ``state_history``, so the lineage is inherited unchanged, new edits
    append under it, and no new session field is needed on ``Trace``.
    """

    if prior:
        return str(prior[0]["event_id"]).split(":", 1)[0]
    return uuid.uuid4().hex[:16]


def fire_records_since(trace: Any, t0: float) -> list[FireRecord]:
    """Collect FireRecords minted at or after ``t0`` across retained ops.

    The ONE builder stamps a monotonic timestamp on every record, so a door
    can bound exactly its own transaction's fires. Records without a
    timestamp (legacy artifacts) are never attributed to a live transaction.
    """

    collected: list[FireRecord] = []
    for label in getattr(trace, "op_labels", ()) or ():
        op = trace.ops[label]
        for record in getattr(op, "interventions", None) or ():
            stamp = getattr(record, "timestamp", None)
            if stamp is not None and stamp >= t0:
                collected.append(record)
    return collected


def record_intervention_event(
    trace: Any,
    *,
    lane: EventLane,
    door: str,
    edit_names: tuple[str, ...],
    selection_repr: str,
    status: EventStatus,
    fire_count: int,
    site_keys: tuple[str, ...] = (),
    rules: tuple[dict[str, Any], ...] = (),
    zero_fire_rule_ids: tuple[str, ...] = (),
    error: str | None = None,
    extra: dict[str, Any] | None = None,
    append_audit_row: bool = True,
) -> dict[str, Any]:
    """Write ONE transaction envelope (the fire-evidence chokepoint).

    Appends (a) the full ``intervention_event_v2`` envelope to the persisted
    free-form ``state_history`` stream and (b) one closed-schema ACT row to
    the canonical ``intervention_audit`` -- so the audit is NEVER empty after
    an attempt, including no-fire and error outcomes, and loaded artifacts
    keep validating under the shipped closed row schema.

    Parameters
    ----------
    trace:
        Trace (or fork) the transaction ran against.
    lane:
        Closed execution-lane vocabulary value.
    door:
        Human door name (``"do"`` / ``"push"`` / ``"run"`` / ``"trace"`` /
        ``"set"``).
    edit_names:
        Helper/callable display names, one per rule/edit, in order.
    selection_repr:
        Canonical WHERE disclosure (spec rule reprs or selector repr).
    status:
        ``"fired"`` / ``"no_fire"`` / ``"error"``.
    fire_count:
        FireRecords minted by THIS transaction.
    site_keys:
        Site-key-first resolved target refs (``site_key_v1`` strings where
        derivable, labels as the disclosed fallback).
    rules:
        Per-rule payload rows (rule_id, where/action reprs, address class,
        payload fidelity, helper args rendering).
    zero_fire_rule_ids:
        Rules that never fired (a misfire is DATA, never silence).
    error:
        Stringified failure for ``status="error"`` rows.
    extra:
        Door-specific disclosures folded into the envelope.
    append_audit_row:
        Whether to append the closed-schema ACT row to the canonical audit.
        Doors that already appended their own canonical row THIS transaction
        (the Selection/EDGE/PARAM paths) pass ``False`` -- one transaction,
        one canonical row, never a duplicate.

    Returns
    -------
    dict
        The envelope row (also appended to ``state_history``).
    """

    prior = _prior_event_rows(trace)
    lineage = _event_lineage(trace, prior)
    ordinal = len(prior) + 1
    event_id = f"{lineage}:{ordinal}"
    parent_event_id = prior[-1]["event_id"] if prior else None
    payload_basis = "|".join(
        (
            lineage,
            str(ordinal),
            lane,
            door,
            ",".join(edit_names),
            selection_repr,
            status,
            str(fire_count),
            ",".join(site_keys),
            ",".join(sorted(zero_fire_rule_ids)),
            ",".join(sorted(str(sorted(rule.items())) for rule in rules)),
        )
    )
    digest = hashlib.sha256(payload_basis.encode()).hexdigest()
    envelope: dict[str, Any] = {
        "schema": "intervention_event_v2",
        "event_id": event_id,
        "transaction_id": event_id,
        "parent_event_id": parent_event_id,
        "lane": lane,
        "door": door,
        "edit_names": list(edit_names),
        "selection_repr": selection_repr,
        "status": status,
        "fire_count": int(fire_count),
        "site_keys": list(site_keys),
        "rules": [dict(rule) for rule in rules],
        "zero_fire_rule_ids": list(zero_fire_rule_ids),
        "error": error,
        "event_digest": digest,
    }
    if extra:
        envelope.update(extra)
    trace._record_operation("intervention_event", **envelope)
    if append_audit_row:
        trace.intervention_audit.append(
            {
                "kind": "ACT",
                "edit": ",".join(edit_names) if edit_names else "none",
                "selection_repr": selection_repr,
                "resolve_digest": digest,
                "sites": [
                    {"relation": "exact", "selected": 1, "site_key": repr((key,))}
                    for key in site_keys
                ],
            }
        )
    return envelope


def rules_payload(spec: Any) -> tuple[dict[str, Any], ...]:
    """Render a public InterventionSpec's per-rule envelope payload."""

    from .spec import InterventionSpec

    if not isinstance(spec, InterventionSpec):
        return ()
    return tuple(
        {
            "rule_id": rule.rule_id,
            "where": rule.where_repr,
            "action": rule.action_repr,
            "direction": rule.direction,
            "address_class": rule.address_class,
            "payload_fidelity": rule.payload_fidelity,
        }
        for rule in spec.rules
    )


def site_keys_for_labels(trace: Any, labels: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve labels to site-key-first target refs (labels as fallback).

    Site keys are the join identity (labels are display-only, ME ordinal
    rule); a label whose op carries no key falls back to the label string --
    the ``key_pending`` disclosure rides the envelope's ``rules`` payload
    consumer side, never a guessed key.
    """

    keys: list[str] = []
    for label in labels:
        try:
            op = trace.ops[label]
        except (KeyError, AttributeError, TypeError):
            # The honest lookup-failure set: missing label (accessor KeyError),
            # a trace-like without .ops, or a husked/None accessor.
            keys.append(label)
            continue
        key = getattr(op, "site_key", None)
        keys.append(str(key) if key else label)
    return tuple(keys)


__all__ = [
    "EventLane",
    "EventStatus",
    "build_fire_record",
    "fire_records_since",
    "record_intervention_event",
    "rules_payload",
    "site_keys_for_labels",
]
