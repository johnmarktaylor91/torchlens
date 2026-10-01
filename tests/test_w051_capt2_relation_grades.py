"""User-authored relation evidence never mints TRUST (W051-HONESTY M8).

``RELATION_CLAIM_GRADES`` reserves ``verified`` / ``consistent`` / ``divergent``
for TorchLens MEASUREMENTS, yet every evidence item that reaches a row today is
user-authored (no producer writes envelopes). A user ``"verified"`` used to be
stored and rendered verbatim. Now a measurement grade is clamped to
``disclosed`` at row construction -- ``Bundle.relate`` and the payload rebuild
at load share the constructor -- with the declared grade preserved under
``declared_grade``. ``disclosed`` and ``unchecked`` (contracted reason) pass
through; out-of-vocabulary grades still refuse.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.bundle import Bundle
from torchlens.bundle._relations import (
    RELATION_CLAIM_GRADES,
    RELATION_DECLARED_GRADE_KEY,
    RELATION_MEASURED_GRADES,
    MemberRelationRow,
    MemberRelationTable,
)

pytestmark = pytest.mark.smoke


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _envelope(*items: dict[str, object]) -> dict[str, object]:
    return {"schema": "version_boundary_v1", "items": list(items), "facts_digest": "sha256:00"}


def _row(**params: object) -> MemberRelationRow:
    return MemberRelationRow(
        kind="successor_of", from_member="ckpt_200", to_member="ckpt_100", params=params
    )


def test_measured_grades_are_the_reserved_trust_subset() -> None:
    assert {"verified", "consistent", "divergent"} == RELATION_MEASURED_GRADES
    assert RELATION_MEASURED_GRADES < RELATION_CLAIM_GRADES
    assert RELATION_DECLARED_GRADE_KEY == "declared_grade"


@pytest.mark.parametrize("declared", sorted(RELATION_MEASURED_GRADES))
def test_relate_time_measurement_grade_is_clamped_to_disclosed(declared: str) -> None:
    row = _row(evidence=_envelope({"grade": declared, "basis": "digest", "extra": [1]}))
    item = row.params["evidence"]["items"][0]
    assert item["grade"] == "disclosed"
    assert item[RELATION_DECLARED_GRADE_KEY] == declared
    assert item["extra"] == [1]  # schema-specific keys preserved opaque
    assert row.to_payload()["params"]["evidence"]["items"][0]["grade"] == "disclosed"


def test_user_authored_disclosed_and_unchecked_pass_through() -> None:
    row = _row(
        evidence=_envelope(
            {"grade": "disclosed", "basis": None},
            {"grade": "unchecked", "basis": None, "reason": "no_param_snapshot"},
        )
    )
    items = row.params["evidence"]["items"]
    assert [item["grade"] for item in items] == ["disclosed", "unchecked"]
    assert all(RELATION_DECLARED_GRADE_KEY not in item for item in items)


def test_load_time_normalization_of_a_persisted_verified_item() -> None:
    """A payload persisted with a raw user 'verified' (pre-clamp writer) reads 'disclosed'."""

    payload = {
        "kind": "successor_of",
        "from": "ckpt_200",
        "to": "ckpt_100",
        "params": {"evidence": _envelope({"grade": "verified", "basis": "digest"})},
    }
    table = MemberRelationTable.from_payload([payload])
    rebuilt = table.rows[0]
    assert isinstance(rebuilt, MemberRelationRow)
    item = rebuilt.params["evidence"]["items"][0]
    assert item["grade"] == "disclosed"
    assert item[RELATION_DECLARED_GRADE_KEY] == "verified"
    # Idempotent: re-loading the clamped payload changes nothing.
    again = MemberRelationTable.from_payload([rebuilt.to_payload()]).rows[0]
    assert again.to_payload() == rebuilt.to_payload()


def test_forged_declared_grade_never_upgrades() -> None:
    """A payload carrying grade='verified' AND a stale declared_grade keeps the
    earliest declaration and still reads disclosed."""

    row = _row(
        evidence=_envelope(
            {"grade": "verified", "basis": "digest", RELATION_DECLARED_GRADE_KEY: "consistent"}
        )
    )
    item = row.params["evidence"]["items"][0]
    assert item["grade"] == "disclosed"
    assert item[RELATION_DECLARED_GRADE_KEY] == "consistent"


def test_out_of_vocabulary_grade_still_refuses() -> None:
    with pytest.raises(ValueError, match="outside the closed vocabulary"):
        _row(evidence=_envelope({"grade": "certified", "basis": None}))


def test_bundle_relate_and_artifact_round_trip_read_disclosed(tmp_path) -> None:
    model = _Tiny()
    x = torch.randn(2, 3)
    early = tl.trace(model, x)
    late = tl.trace(model, x)
    try:
        bundle = Bundle({"ckpt_100": early, "ckpt_200": late})
        bundle.relate(_row(evidence=_envelope({"grade": "verified", "basis": "digest"})))
        stored = [r for r in bundle.member_relations if r.kind == "successor_of"][0]
        assert stored.params["evidence"]["items"][0]["grade"] == "disclosed"
        target = tmp_path / "bundle.tlspec"
        bundle.save(target)
        loaded = tl.load(str(target))
        assert isinstance(loaded, Bundle)
        row = [r for r in loaded.member_relations if r.kind == "successor_of"][0]
        item = row.params["evidence"]["items"][0]
        assert item["grade"] == "disclosed"
        assert item[RELATION_DECLARED_GRADE_KEY] == "verified"
    finally:
        early.cleanup()
        late.cleanup()
