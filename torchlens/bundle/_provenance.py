"""The provenance join: ``bundle.why()`` / ``bundle.provenance()`` (F03 item 4).

A PURE DERIVED VIEW over the repaired material record (ledger memo D1b):
nothing here is stored, loads recompute the projection, and ancestry is
EXACT-OR-REFUSED — never inferred from labels, reprs, output equality, or
list position (D1c). The walk is over ORDERED event identities, never set
arithmetic, so applying A, applying A twice, and applying A-then-B stay
distinct.

Evidence sources, in precedence order:

1. **Event lineage** — the trace-side envelope rows
   (``state_history`` ``op="intervention_event"`` rows; the C03 substrate).
   Forks copy ``state_history``, so a shared lineage id with an ordinal
   prefix of digest-equal rows is exact shared history.
2. **Container operations** — the bundle-side chronology (F03 item 3): a
   ``member_construction`` anchor naming the reference as ``source_member``,
   or one operation row that created both members (a sweep/vary that also
   minted the pristine baseline), is container-attested ancestry; the basis
   is disclosed on the report.
3. **Nothing** — emptiness is DISCLOSURE ("no recorded construction
   evidence"), never "identical" (status ``unattested``); two evidenced but
   unlinked histories are ``unrelated`` with a pointer at ``relate()``,
   never a fabricated diff.

Four orthogonal honesty axes, never blurred into one enum (D1c): lineage
status, payload fidelity (declared | opaque — a bare user callable names the
site and the fact, never guesses content), value residual (explained |
unexplained | not_checked — identical recorded chains with differing outputs
report ``unexplained``), and comparability (the shipped structural
relationship plus the evidence lanes, in their own columns). Additive
wording ("reference + these edits") is licensed ONLY when the
reference-side suffix is empty; otherwise both suffixes are shown and the
additive wording is refused.

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

from ..intervention.errors import BundleMemberError

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from . import Bundle

__all__ = [
    "ProvenanceEvent",
    "WhyReport",
    "bundle_provenance",
    "bundle_why",
]

LineageStatus = Literal["exact", "partial", "diverged", "unrelated", "unattested"]
PayloadFidelity = Literal["declared", "opaque", "none"]
ValueResidual = Literal["explained", "unexplained", "not_checked"]
LineageBasis = Literal["event_lineage", "container_operation", "none"]


@dataclass(frozen=True)
class ProvenanceEvent:
    """One suffix event in a member's construction story (ordered, typed)."""

    event_id: str
    door: str
    lane: str
    edit_names: tuple[str, ...]
    selection_repr: str
    status: str
    fire_count: int
    payload_fidelity: PayloadFidelity

    def describe(self) -> str:
        """One-line rendering for report wording."""

        edits = ", ".join(self.edit_names) if self.edit_names else "(no edits)"
        qualifier = "" if self.payload_fidelity == "declared" else f" [{self.payload_fidelity}]"
        return (
            f"{edits} @ {self.selection_repr} ({self.lane}/{self.door}, {self.status}){qualifier}"
        )


@dataclass(frozen=True)
class WhyReport:
    """The derived construction answer for ONE member against a reference.

    Four orthogonal honesty axes (D1c); :meth:`describe` renders the wording
    law (additive phrasing only when ``reference_suffix`` is empty).
    """

    member: str
    reference: str
    lineage_status: LineageStatus
    lineage_basis: LineageBasis
    common_prefix_len: int
    member_suffix: tuple[ProvenanceEvent, ...]
    reference_suffix: tuple[ProvenanceEvent, ...]
    payload_fidelity: PayloadFidelity
    value_residual: ValueResidual
    relationship: str | None
    lanes: tuple[str, ...]

    def describe(self) -> str:
        """Render the wording the lineage status licenses (D1b/D1c)."""

        if self.lineage_status == "exact":
            edits = "; ".join(event.describe() for event in self.member_suffix) or "(no edits)"
            return (
                f"{self.member!r} = {self.reference!r} + [{edits}] "
                f"(lineage exact, basis={self.lineage_basis}, "
                f"fidelity={self.payload_fidelity})"
            )
        if self.lineage_status == "diverged":
            member_edits = "; ".join(event.describe() for event in self.member_suffix)
            reference_edits = "; ".join(event.describe() for event in self.reference_suffix)
            return (
                f"diverged: {self.member!r} adds [{member_edits}] while "
                f"{self.reference!r} adds [{reference_edits}] after "
                f"{self.common_prefix_len} shared event(s) — additive wording refused"
            )
        if self.lineage_status == "partial":
            return (
                f"partially attested: {self.member!r} and {self.reference!r} share a "
                f"lineage id but their recorded chains disagree inside the common "
                "prefix (tampered, trimmed, or drifted evidence)"
            )
        if self.lineage_status == "unrelated":
            return (
                f"no shared construction evidence between {self.member!r} and "
                f"{self.reference!r}: independent capture histories are never "
                "joined by guess — record an explicit claim with bundle.relate()"
            )
        return (
            f"no recorded construction evidence for {self.member!r} vs "
            f"{self.reference!r} (unattested; emptiness is disclosure, never 'identical')"
        )


def _member_events(trace: Any) -> list[dict[str, Any]]:
    """Ordered envelope rows from the member's persisted state_history."""

    return [
        row
        for row in getattr(trace, "state_history", ())
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]


def _lineage_id(events: list[dict[str, Any]]) -> str | None:
    """Extract the lineage prefix of the first event id (None when eventless)."""

    if not events:
        return None
    return str(events[0].get("event_id", "")).split(":", 1)[0] or None


def _event_fidelity(event: dict[str, Any]) -> PayloadFidelity:
    """declared | opaque for one event (D1c axis 2).

    Declared means the per-rule payload carries rendered WHERE/action reprs
    (the spec doors). An edit that fired with no rule payload is a bare user
    callable: the record names the site and the fact, never the content.
    """

    if event.get("rules"):
        return "declared"
    if event.get("fire_count", 0) or event.get("edit_names"):
        return "opaque"
    return "none"


def _to_provenance_event(event: dict[str, Any]) -> ProvenanceEvent:
    """Project one raw state-history row into a frozen ProvenanceEvent."""

    return ProvenanceEvent(
        event_id=str(event.get("event_id", "")),
        door=str(event.get("door", "")),
        lane=str(event.get("lane", "")),
        edit_names=tuple(str(name) for name in event.get("edit_names", ())),
        selection_repr=str(event.get("selection_repr", "")),
        status=str(event.get("status", "")),
        fire_count=int(event.get("fire_count", 0) or 0),
        payload_fidelity=_event_fidelity(event),
    )


def _suffix_fidelity(events: tuple[ProvenanceEvent, ...]) -> PayloadFidelity:
    """Fold per-event fidelity over a suffix: any opaque event makes it opaque."""

    if not events:
        return "none"
    if any(event.payload_fidelity == "opaque" for event in events):
        return "opaque"
    return "declared"


def _common_prefix_len(
    member_events: list[dict[str, Any]], reference_events: list[dict[str, Any]]
) -> int:
    """Count of leading rows equal in BOTH event id and event digest.

    After a fork both branches mint the same next ordinal independently, so
    an id-equal/digest-different row is the DIVERGENCE POINT (it ends the
    prefix and belongs to both suffixes) — never evidence of tampering by
    itself.
    """

    shared = 0
    for member_row, reference_row in zip(member_events, reference_events, strict=False):
        if member_row.get("event_id") != reference_row.get("event_id"):
            break
        if member_row.get("event_digest") != reference_row.get("event_digest"):
            break
        shared += 1
    return shared


def _container_link(bundle: Bundle, member: str, reference: str) -> bool:
    """Container-attested ancestry (item-3 anchors + operation rows)."""

    anchor = bundle._member_construction.get(member, {})
    if anchor.get("source_member") == reference:
        return True
    operation_id = anchor.get("operation_id")
    if operation_id is None:
        return False
    for row in bundle._operations:
        if row.operation_id == operation_id and reference in row.member_names:
            return True
    return False


def _value_residual(
    member_trace: Trace, reference_trace: Trace, *, chains_identical: bool
) -> ValueResidual:
    """explained | unexplained | not_checked (D1c axis 3).

    Only IDENTICAL recorded chains license a value check: equal outputs are
    ``explained``, differing outputs ``unexplained`` (the two failures this
    surface exists to catch: nondeterminism and an unrecorded write). With
    differing chains the output difference is expected and the axis stays
    ``not_checked`` — corroborating causality needs replay, not a diff.
    """

    if not chains_identical:
        return "not_checked"
    member_out = _readable_output(member_trace)
    reference_out = _readable_output(reference_trace)
    if member_out is None or reference_out is None:
        return "not_checked"
    if member_out.shape != reference_out.shape or member_out.dtype != reference_out.dtype:
        return "unexplained"
    return "explained" if torch.equal(member_out, reference_out) else "unexplained"


def _readable_output(trace: Any) -> torch.Tensor | None:
    """The member's retained output tensor, when cheaply readable."""

    try:
        output_layers = getattr(trace, "output_layers", None) or []
        for label in output_layers:
            layer = trace[label]
            value = getattr(layer, "out", None)
            if isinstance(value, torch.Tensor):
                return value
    except Exception:  # noqa: BLE001 - a husked/cleaned member reads as unavailable
        return None
    return None


def _relationship_repr(bundle: Bundle, member: str, reference: str) -> str | None:
    """Best-effort relationship name for the comparability axis (None on failure)."""

    try:
        relationship = bundle.relationship(member, reference)
    except Exception:  # noqa: BLE001 - comparability is disclosure, never a gate here
        return None
    return getattr(relationship, "name", None) or str(relationship)


def bundle_why(
    bundle: Bundle, member: str | Any, relative_to: str | Any | None = None
) -> WhyReport:
    """Answer "member X = reference + WHICH edits", exact-or-refused.

    Parameters
    ----------
    bundle:
        The container whose material record is walked.
    member:
        Member name (or Trace) to explain.
    relative_to:
        Reference member; defaults to the bundle baseline.

    Raises
    ------
    BundleMemberError
        ``provenance_reference_missing`` when no reference is available, or
        ``provenance_self_comparison`` for member == reference (vacuous).
    """

    member_name = bundle._coerce_member_name(member)
    if relative_to is not None:
        reference_name = bundle._coerce_member_name(relative_to)
    elif bundle.baseline_name is not None:
        reference_name = bundle.baseline_name
    else:
        raise BundleMemberError(
            "why() needs a reference: this bundle has no baseline and no "
            "relative_to= was passed. Set baseline= at construction or pass "
            "relative_to=<member>.",
            code="provenance_reference_missing",
            member=member_name,
        )
    if member_name == reference_name:
        raise BundleMemberError(
            f"why({member_name!r}) against itself is vacuous; pass a different "
            "relative_to= member.",
            code="provenance_self_comparison",
            member=member_name,
        )

    member_trace = bundle[member_name]
    reference_trace = bundle[reference_name]
    member_events = _member_events(member_trace)
    reference_events = _member_events(reference_trace)
    member_lineage = _lineage_id(member_events)
    reference_lineage = _lineage_id(reference_events)

    basis: LineageBasis
    prefix_len = 0
    if member_lineage is not None and member_lineage == reference_lineage:
        basis = "event_lineage"
        prefix_len = _common_prefix_len(member_events, reference_events)
        member_suffix = tuple(_to_provenance_event(row) for row in member_events[prefix_len:])
        reference_suffix = tuple(_to_provenance_event(row) for row in reference_events[prefix_len:])
        if prefix_len == 0:
            # A shared lineage id with ZERO verifiable shared rows claims a
            # common origin it cannot evidence (trimmed/edited history).
            status: LineageStatus = "partial"
        elif reference_suffix:
            status = "diverged"
        else:
            status = "exact"
    elif _container_link(bundle, member_name, reference_name):
        basis = "container_operation"
        member_suffix = tuple(_to_provenance_event(row) for row in member_events)
        reference_suffix = tuple(_to_provenance_event(row) for row in reference_events)
        status = "exact" if not reference_suffix else "diverged"
    else:
        basis = "none"
        member_suffix = tuple(_to_provenance_event(row) for row in member_events)
        reference_suffix = tuple(_to_provenance_event(row) for row in reference_events)
        status = "unrelated" if (member_events and reference_events) else "unattested"

    chains_identical = (
        basis == "event_lineage" and not member_suffix and not reference_suffix
    ) or (basis == "container_operation" and not member_suffix and not reference_suffix)
    lanes = tuple(
        sorted({event.lane for event in (*member_suffix, *reference_suffix) if event.lane})
    )
    return WhyReport(
        member=member_name,
        reference=reference_name,
        lineage_status=status,
        lineage_basis=basis,
        common_prefix_len=prefix_len,
        member_suffix=member_suffix,
        reference_suffix=reference_suffix,
        payload_fidelity=_suffix_fidelity(member_suffix),
        value_residual=_value_residual(
            member_trace, reference_trace, chains_identical=chains_identical
        ),
        relationship=_relationship_repr(bundle, member_name, reference_name),
        lanes=lanes,
    )


def bundle_provenance(bundle: Bundle) -> list[dict[str, Any]]:
    """One derived construction row per member (never stored, D1b).

    Rows carry the container anchor, the member's own ordered event count
    and fidelity, and — when a baseline exists — the member's lineage status
    against it. Without a baseline the lineage column reads ``no_baseline``
    (a disclosure, not a default).
    """

    rows: list[dict[str, Any]] = []
    baseline = bundle.baseline_name
    for name, trace in bundle.members.items():
        events = [_to_provenance_event(row) for row in _member_events(trace)]
        row: dict[str, Any] = {
            "member": name,
            "origin": dict(bundle._member_construction.get(name, {"origin": "constructed"})),
            "n_events": len(events),
            "payload_fidelity": _suffix_fidelity(tuple(events)),
            "lanes": tuple(sorted({event.lane for event in events if event.lane})),
        }
        if baseline is None:
            row["lineage_status"] = "no_baseline"
        elif name == baseline:
            row["lineage_status"] = "baseline"
        else:
            row["lineage_status"] = bundle_why(bundle, name).lineage_status
        rows.append(row)
    return rows
