"""Derived relation-row grades: the ladder plus the ``broken`` finding (FK-2, FK-4, FK-5).

A relation row's grade is DERIVED at read from its evidence items and its
declared params; nothing here is persisted (F-REL derives, never stores).
The ladder, weakest to strongest checked rung: ``unchecked`` (no check ran),
``disclosed`` (declared, nothing checked), ``consistent`` (every check that
ran agrees, not proven), ``verified`` (every item verified). Any divergent
item grades the row ``divergent``; ``broken`` fires on EXACTLY two premises
(FK-2):

- ``bulk_origin_failure``: more than half of the member-origin items that
  were checked diverge;
- ``verbatim_carry_failure``: the row declares ``carry_mode="verbatim"`` and a
  ``declared_carry`` item diverges.

A divergence at a DECLARED seam (the item's ``seam`` is in the row's
``seams``) is seam-local: it counts toward neither premise, so a chat-template
revision at a declared seam grades ``divergent`` on record without
escalating a healthy chain to ``broken``.

``carry_mode="none"`` with no evidence grades ``disclosed`` (FK-5): a
declared fresh-start claim, checkable as a class. A ``declared_carry`` item
that diverges on such a row is detected secretly-carried history and grades
the row ``divergent``.

``exogenous`` (FK-3) is the count of items whose class lies outside the
member-measured kinds; it is reported here and never written to an item.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ._relation_evidence import (
    EVIDENCE_CLASS_KEY,
    EVIDENCE_EXOGENOUS_ORIGIN_KINDS,
    EVIDENCE_MEMBER_ORIGIN_KINDS,
    EVIDENCE_SEAM_KEY,
)
from ._relations import RELATION_CLAIM_GRADES, MemberRelationRow, OpaqueRelationRow

if TYPE_CHECKING:
    from . import Bundle

__all__ = ["RELATION_BROKEN_PREMISES", "RELATION_ROW_GRADES", "RelationGrade"]

#: Closed row-grade vocabulary: the item ladder plus the derived ``broken`` finding.
RELATION_ROW_GRADES = RELATION_CLAIM_GRADES | frozenset({"broken"})

#: The two premises on which ``broken`` fires (FK-2); nothing else escalates.
RELATION_BROKEN_PREMISES = frozenset({"bulk_origin_failure", "verbatim_carry_failure"})

_CHECKED_GRADES = frozenset({"verified", "consistent", "divergent"})
_EMPTY_COUNTS: Mapping[str, int] = MappingProxyType({})


@dataclass(frozen=True)
class RelationGrade:
    """The derived grade of one relation row.

    Parameters
    ----------
    kind:
        The row's relation kind.
    members:
        The member names the row references, in schema order.
    grade:
        One of :data:`RELATION_ROW_GRADES`.
    broken_premises:
        The FK-2 premises that fired, in sorted order; empty unless ``grade``
        is ``broken``.
    unchecked_reasons:
        The contracted reasons of the row's unchecked items (or the opaque
        row's ``kind_unregistered``), sorted and distinct.
    seam_local_divergences:
        Divergent items at a declared seam: on record, never a premise.
    exogenous_items:
        Items whose class lies outside the member-measured kinds (FK-3,
        derived).
    items_by_class:
        Item counts per origin class; classless items count under ``""``.
    items_by_grade:
        Item counts per claim grade.
    """

    kind: str
    members: tuple[str, ...]
    grade: str
    broken_premises: tuple[str, ...] = ()
    unchecked_reasons: tuple[str, ...] = ()
    seam_local_divergences: int = 0
    exogenous_items: int = 0
    items_by_class: Mapping[str, int] = field(default=_EMPTY_COUNTS)
    items_by_grade: Mapping[str, int] = field(default=_EMPTY_COUNTS)


def grade_relation_row(row: MemberRelationRow | OpaqueRelationRow) -> RelationGrade:
    """Derive one row's grade from its evidence items and declared params.

    Every item a user authors today is clamped to ``disclosed`` or
    ``unchecked`` at the row's construction (W051-HONESTY M8), so the
    measured rungs and ``broken`` fire only once a TorchLens producer writes
    measured items; the derivation is pinned now so that door grades
    identically when it opens.
    """

    if isinstance(row, OpaqueRelationRow):
        return RelationGrade(
            kind=row.kind,
            members=row.named_members(),
            grade="unchecked",
            unchecked_reasons=("kind_unregistered",),
        )
    return grade_relation_params(row.kind, row.named_members(), row.params)


def grade_relation_params(
    kind: str, members: tuple[str, ...], params: Mapping[str, Any]
) -> RelationGrade:
    """The derivation itself, over a validated params mapping (the grader's one input)."""

    envelope = params.get("evidence")
    items: list[Mapping[str, Any]] = list(envelope["items"]) if envelope else []
    seams = tuple(params.get("seams", ()))
    by_class = Counter(str(item.get(EVIDENCE_CLASS_KEY, "")) for item in items)
    by_grade = Counter(str(item["grade"]) for item in items)
    divergent = [item for item in items if item["grade"] == "divergent"]
    seam_local = [item for item in divergent if item.get(EVIDENCE_SEAM_KEY) in seams]
    escalating = [item for item in divergent if item.get(EVIDENCE_SEAM_KEY) not in seams]
    premises = _broken_premises(items, escalating, carry_mode=params.get("carry_mode"))
    reasons = tuple(sorted({str(item["reason"]) for item in items if item["grade"] == "unchecked"}))
    return RelationGrade(
        kind=kind,
        members=members,
        grade=_ladder(by_grade, premises),
        broken_premises=premises,
        unchecked_reasons=reasons,
        seam_local_divergences=len(seam_local),
        exogenous_items=sum(
            1 for item in items if item.get(EVIDENCE_CLASS_KEY) in EVIDENCE_EXOGENOUS_ORIGIN_KINDS
        ),
        items_by_class=MappingProxyType(dict(sorted(by_class.items()))),
        items_by_grade=MappingProxyType(dict(sorted(by_grade.items()))),
    )


def _broken_premises(
    items: list[Mapping[str, Any]], escalating: list[Mapping[str, Any]], *, carry_mode: Any
) -> tuple[str, ...]:
    """Return the FK-2 premises that fire, given the non-seam-local divergent items."""

    fired: list[str] = []
    checked_origin = [
        item
        for item in items
        if item.get(EVIDENCE_CLASS_KEY) in EVIDENCE_MEMBER_ORIGIN_KINDS
        and item["grade"] in _CHECKED_GRADES
    ]
    diverging_origin = [
        item for item in escalating if item.get(EVIDENCE_CLASS_KEY) in EVIDENCE_MEMBER_ORIGIN_KINDS
    ]
    if diverging_origin and 2 * len(diverging_origin) > len(checked_origin):
        fired.append("bulk_origin_failure")
    if carry_mode == "verbatim" and any(
        item.get(EVIDENCE_CLASS_KEY) == "declared_carry" for item in escalating
    ):
        fired.append("verbatim_carry_failure")
    return tuple(fired)


def _ladder(by_grade: Counter[str], premises: tuple[str, ...]) -> str:
    """Place the row on the grade ladder from its item-grade counts."""

    if premises:
        return "broken"
    if by_grade["divergent"]:
        return "divergent"
    checked = sum(by_grade[grade] for grade in _CHECKED_GRADES)
    total = sum(by_grade.values())
    if checked == 0:
        # A bare row, a declared carry_mode and disclosed-only items are all
        # declarations on record (FK-5: never a blank unavailable).
        return "unchecked" if by_grade["unchecked"] else "disclosed"
    return "verified" if by_grade["verified"] == total else "consistent"


def _bundle_relation_grades(self: Bundle) -> tuple[RelationGrade, ...]:
    """Return the derived grade of every relation row, in table order (``Bundle.relation_grades``).

    A derivation over :attr:`Bundle.relations`, never stored: the same rows
    always grade the same way, and loading a bundle recomputes nothing.
    """

    return tuple(grade_relation_row(row) for row in self.relations)
