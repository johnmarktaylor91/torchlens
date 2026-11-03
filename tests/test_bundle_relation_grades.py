"""Derived relation-row grades (FK-2 broken premises, FK-5 carry_mode none, FK-6 item class).

``Bundle.relation_grades`` is a derivation over the relation table: nothing is
stored, and the same rows grade the same way after a save/load round trip.
User-authored measurement grades stay clamped to ``disclosed`` at the row door
(W051-HONESTY M8), so the measured rungs are pinned through the derivation's
own input, a validated params mapping.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bundle import Bundle, MemberRelationRow, RelationGrade
from torchlens.bundle._relation_evidence import (
    EVIDENCE_EXOGENOUS_ORIGIN_KINDS,
    EVIDENCE_MEMBER_ORIGIN_KINDS,
    EVIDENCE_ORIGIN_KINDS,
    RELATION_CARRY_MODES,
)
from torchlens.bundle._relation_grades import (
    RELATION_BROKEN_PREMISES,
    RELATION_ROW_GRADES,
    grade_relation_params,
)
from torchlens.bundle._relations import RELATION_CLAIM_GRADES

pytestmark = pytest.mark.smoke

MEMBERS = ("turn_2", "turn_1")


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture(scope="module")
def bundle():
    model = _Tiny()
    x = torch.randn(2, 3)
    early = tl.trace(model, x)
    late = tl.trace(model, x)
    try:
        yield Bundle({"turn_1": early, "turn_2": late})
    finally:
        early.cleanup()
        late.cleanup()


def _item(grade: str, cls: str, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"grade": grade, "basis": "digest", "class": cls}
    if grade == "unchecked":
        item["basis"] = None
        item.setdefault("reason", "no_param_snapshot")
    item.update(extra)
    return item


def _params(items: list[dict[str, Any]], **params: Any) -> dict[str, Any]:
    params["evidence"] = {
        "schema": "turn_boundary_v1",
        "items": items,
        "facts_digest": "sha256:00",
    }
    return params


def _grade(items: list[dict[str, Any]], **params: Any) -> RelationGrade:
    return grade_relation_params("successor_of", MEMBERS, _params(items, **params))


# ---------------------------------------------------------------------------
# Vocabularies.
# ---------------------------------------------------------------------------


def test_row_grades_are_the_claim_ladder_plus_broken() -> None:
    assert RELATION_CLAIM_GRADES | {"broken"} == RELATION_ROW_GRADES
    assert {"bulk_origin_failure", "verbatim_carry_failure"} == RELATION_BROKEN_PREMISES


def test_seven_origin_kinds_split_into_member_and_exogenous() -> None:
    assert len(EVIDENCE_ORIGIN_KINDS) == 7
    assert EVIDENCE_MEMBER_ORIGIN_KINDS | EVIDENCE_EXOGENOUS_ORIGIN_KINDS == EVIDENCE_ORIGIN_KINDS
    assert not (EVIDENCE_MEMBER_ORIGIN_KINDS & EVIDENCE_EXOGENOUS_ORIGIN_KINDS)
    assert "exogenous" not in EVIDENCE_ORIGIN_KINDS  # a rollup, never a persisted class
    assert {"none", "verbatim", "summarized"} == RELATION_CARRY_MODES


# ---------------------------------------------------------------------------
# FK-5: carry_mode is closed; ``none`` grades disclosed.
# ---------------------------------------------------------------------------


def test_carry_mode_outside_vocabulary_refuses_at_the_row_door() -> None:
    with pytest.raises(ValueError, match="carry_mode"):
        MemberRelationRow(
            kind="successor_of", from_member="b", to_member="a", params={"carry_mode": "declared"}
        )


def test_carry_mode_none_grades_disclosed_not_unchecked(bundle: Bundle) -> None:
    b = bundle.relate("successor_of", "turn_2", "turn_1", carry_mode="none")
    (grade,) = b.relation_grades()
    assert grade.grade == "disclosed"
    assert grade.unchecked_reasons == ()
    assert grade.members == MEMBERS


def test_carry_mode_none_with_divergent_declared_carry_grades_divergent_never_broken() -> None:
    grade = _grade([_item("divergent", "declared_carry")], carry_mode="none")
    assert grade.grade == "divergent"
    assert grade.broken_premises == ()


# ---------------------------------------------------------------------------
# FK-2: broken fires on exactly two premises; seams suppress the escalation.
# ---------------------------------------------------------------------------


def test_bulk_origin_failure_fires_past_half_of_checked_member_items() -> None:
    items = [
        _item("divergent", "member_state"),
        _item("divergent", "member_output"),
        _item("verified", "member_input"),
    ]
    grade = _grade(items)
    assert grade.grade == "broken"
    assert grade.broken_premises == ("bulk_origin_failure",)


def test_half_of_checked_member_items_diverging_is_divergent_not_broken() -> None:
    items = [_item("divergent", "member_state"), _item("verified", "member_input")]
    grade = _grade(items)
    assert grade.grade == "divergent"
    assert grade.broken_premises == ()


def test_verbatim_carry_failure_fires_only_under_carry_mode_verbatim() -> None:
    items = [_item("divergent", "declared_carry"), _item("verified", "member_input")]
    assert _grade(items, carry_mode="verbatim").broken_premises == ("verbatim_carry_failure",)
    assert _grade(items, carry_mode="summarized").grade == "divergent"


def test_both_premises_report_together() -> None:
    items = [_item("divergent", "declared_carry"), _item("divergent", "member_state")]
    grade = _grade(items, carry_mode="verbatim")
    assert grade.grade == "broken"
    assert grade.broken_premises == ("bulk_origin_failure", "verbatim_carry_failure")


def test_exogenous_divergence_alone_never_escalates() -> None:
    items = [_item("divergent", "harness_record"), _item("divergent", "declared_transform")]
    grade = _grade(items, carry_mode="summarized")
    assert grade.grade == "divergent"
    assert grade.broken_premises == ()
    assert grade.exogenous_items == 2


def test_divergence_at_a_declared_seam_stays_on_record_without_escalating() -> None:
    items = [
        _item("divergent", "member_state", seam="chat_template_v3"),
        _item("divergent", "declared_carry", seam="chat_template_v3"),
    ]
    grade = _grade(items, carry_mode="verbatim", seams=["chat_template_v3"])
    assert grade.grade == "divergent"
    assert grade.broken_premises == ()
    assert grade.seam_local_divergences == 2


def test_undeclared_seam_refuses_at_the_row_door() -> None:
    with pytest.raises(ValueError, match="does not declare"):
        MemberRelationRow(
            kind="successor_of",
            from_member="b",
            to_member="a",
            params=_params([_item("disclosed", "member_state", seam="tokenizer_swap")]),
        )


def test_seams_must_be_distinct_non_empty_names() -> None:
    for bad in (["a", "a"], [""], "a", [1]):
        with pytest.raises(ValueError, match="seams"):
            MemberRelationRow(
                kind="successor_of", from_member="b", to_member="a", params={"seams": bad}
            )


# ---------------------------------------------------------------------------
# Ladder rungs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([], "disclosed"),
        ([_item("disclosed", "member_state")], "disclosed"),
        ([_item("unchecked", "member_params")], "unchecked"),
        ([_item("unchecked", "member_params"), _item("disclosed", "member_state")], "unchecked"),
        ([_item("verified", "member_state"), _item("verified", "member_input")], "verified"),
        ([_item("verified", "member_state"), _item("consistent", "member_input")], "consistent"),
        ([_item("verified", "member_state"), _item("disclosed", "declared_carry")], "consistent"),
    ],
)
def test_ladder(items: list[dict[str, Any]], expected: str) -> None:
    grade = _grade(items)
    assert grade.grade == expected
    assert grade.grade in RELATION_ROW_GRADES


def test_unchecked_reasons_are_listed_sorted_and_distinct() -> None:
    items = [
        _item("unchecked", "member_params", reason="no_param_snapshot"),
        _item("unchecked", "member_state", reason="incomparable_basis"),
        _item("unchecked", "member_input", reason="no_param_snapshot"),
    ]
    assert _grade(items).unchecked_reasons == ("incomparable_basis", "no_param_snapshot")


# ---------------------------------------------------------------------------
# FK-6: item class is a load-validated closed vocabulary, scoped per schema.
# ---------------------------------------------------------------------------


def test_reserved_schema_requires_a_class_on_every_item() -> None:
    with pytest.raises(ValueError, match="missing the 'class' key"):
        MemberRelationRow(
            kind="successor_of",
            from_member="b",
            to_member="a",
            params=_params([{"grade": "disclosed", "basis": "digest"}]),
        )


def test_class_outside_the_schema_scope_refuses() -> None:
    # declared_carry is a turn-boundary kind, not a version-boundary one.
    envelope = {
        "schema": "version_boundary_v1",
        "items": [_item("disclosed", "declared_carry")],
        "facts_digest": "sha256:00",
    }
    with pytest.raises(ValueError, match="outside the closed origin vocabulary"):
        MemberRelationRow(
            kind="successor_of", from_member="b", to_member="a", params={"evidence": envelope}
        )


def test_rollup_names_are_never_a_persisted_class() -> None:
    with pytest.raises(ValueError, match="outside the closed origin vocabulary"):
        MemberRelationRow(
            kind="successor_of",
            from_member="b",
            to_member="a",
            params=_params([_item("disclosed", "exogenous")]),
        )


def test_user_schema_leaves_class_optional_but_validated() -> None:
    envelope = {
        "schema": "lab.custom_v1",
        "items": [{"grade": "disclosed", "basis": "x"}],
        "facts_digest": "d",
    }
    row = MemberRelationRow(
        kind="successor_of", from_member="b", to_member="a", params={"evidence": envelope}
    )
    assert row.params["evidence"]["items"][0].get("class") is None
    envelope["items"] = [{"grade": "disclosed", "basis": "x", "class": "made_up"}]
    with pytest.raises(ValueError, match="outside the closed origin vocabulary"):
        MemberRelationRow(
            kind="successor_of", from_member="b", to_member="a", params={"evidence": envelope}
        )


# ---------------------------------------------------------------------------
# Derived at read: Bundle.relation_grades, save/load, opaque rows.
# ---------------------------------------------------------------------------


def test_bundle_relation_grades_are_derived_and_survive_a_round_trip(bundle: Bundle) -> None:
    b = bundle.relate(
        "successor_of",
        "turn_2",
        "turn_1",
        carry_mode="verbatim",
        seams=["chat_template_v3"],
        evidence=_params(
            [
                _item("divergent", "member_state", seam="chat_template_v3"),
                _item("disclosed", "declared_carry"),
            ]
        )["evidence"],
    )
    (grade,) = b.relation_grades()
    # The user-authored 'divergent' is clamped to disclosed at the door, so the
    # row is a declaration on record: disclosed, with the exogenous rollup derived.
    assert grade.grade == "disclosed"
    assert grade.exogenous_items == 1
    assert dict(grade.items_by_class) == {"declared_carry": 1, "member_state": 1}
    assert "exogenous" not in b.relations[0].params["evidence"]["items"][0]
    assert "grade" not in b.relations[0].to_payload()["params"]
    rows = [MemberRelationRow.from_payload(row.to_payload()) for row in b.relations]
    reloaded = Bundle({"turn_1": b["turn_1"], "turn_2": b["turn_2"]}, relations=rows)
    assert reloaded.relation_grades() == b.relation_grades()


def test_opaque_row_grades_unchecked_kind_unregistered(bundle: Bundle) -> None:
    from torchlens.bundle import OpaqueRelationRow

    opaque = OpaqueRelationRow.from_payload(
        {"kind": "lab.custom_link", "from": "turn_2", "to": "turn_1", "params": {}}
    )
    b = Bundle({"turn_1": bundle["turn_1"], "turn_2": bundle["turn_2"]}, relations=[opaque])
    (grade,) = b.relation_grades()
    assert grade.grade == "unchecked"
    assert grade.unchecked_reasons == ("kind_unregistered",)
