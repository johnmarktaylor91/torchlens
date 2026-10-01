"""C07X amendment, payload B: bundle relation grammar v2 + loader doctrine.

Covers foldB s4.4 items 1, 2, 6, 7, 8, 9, 10, 11 riding the open tlspec-v9
window (TLSPEC_VERSION stays 9):

- required/optional param split with the undeclared-key refusal untouched;
- ``successor_of`` optional evidence envelope ({schema, items[], facts_digest})
  + ``carry_mode`` / ``state_source`` (D16 spellings);
- graded evidence items over the closed grade vocabulary with the contracted
  unchecked-reason menu (``no_param_snapshot`` admitted, D15);
- the per-row evidence canonical-JSON byte budget (1 MiB, typed refusal);
- the successor_of direction pin (from = the LATER member);
- three-leg preserve-and-disclose loading: (a) unknown NAMESPACED relation
  kinds load opaque/disclosed (bare unknown refuses; ``tl.load(
  unknown_relations="refuse")`` restores strictness), (b) unknown evidence
  schema ids load opaque, (c) unknown namespaced bundle.json sections
  preserve through the Bundle carriage while bare-unknown sections refuse —
  with byte-preserving re-save in every leg;
- reserved evidence schema-id strings and reserved TorchLens sidecar family
  ids (registrations, not slots).

Every protective claim gets a test that TRIES the forbidden thing (D18).
"""

from __future__ import annotations

import json

import pytest
import torch

import torchlens as tl
from torchlens._io.sidecar import (
    RESERVED_SIDECAR_FAMILY_IDS,
    SidecarError,
    SidecarFamily,
    register_sidecar_family,
    unregister_sidecar_family,
)
from torchlens._registry import ProviderInfo
from torchlens.bundle import Bundle
from torchlens.bundle._relations import (
    RELATION_CLAIM_GRADES,
    RELATION_EVIDENCE_BUDGET_BYTES,
    RELATION_UNCHECKED_REASONS,
    RESERVED_EVIDENCE_SCHEMA_IDS,
    MemberRelationRow,
    MemberRelationTable,
    OpaqueRelationRow,
)
from torchlens.errors import TorchLensWarning
from torchlens.errors.episode import BundleRelationError

pytestmark = pytest.mark.smoke


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture(scope="module")
def two_member_bundle():
    model = _Tiny()
    x = torch.randn(2, 3)
    early = tl.trace(model, x)
    late = tl.trace(model, x)
    try:
        yield Bundle({"ckpt_100": early, "ckpt_200": late})
    finally:
        early.cleanup()
        late.cleanup()


def _successor_row(**params: object) -> MemberRelationRow:
    # Direction pin: from = the LATER member (the successor).
    return MemberRelationRow(
        kind="successor_of", from_member="ckpt_200", to_member="ckpt_100", params=params
    )


def _evidence(**overrides: object) -> dict[str, object]:
    envelope: dict[str, object] = {
        "schema": "version_boundary_v1",
        "items": [{"grade": "consistent", "basis": "step_count"}],
        "facts_digest": "sha256:00",
    }
    envelope.update(overrides)
    return envelope


# ---------------------------------------------------------------------------
# Item 1: required/optional split; successor_of optional params.
# ---------------------------------------------------------------------------


def test_successor_of_bare_row_still_valid() -> None:
    """v8-era successor_of rows (empty params) stay valid under grammar v2."""

    row = _successor_row()
    assert row.params == {}
    rebuilt = MemberRelationRow.from_payload(row.to_payload())
    assert rebuilt == row


def test_successor_of_admits_optional_evidence_carry_mode_state_source() -> None:
    row = _successor_row(
        evidence=_evidence(), carry_mode="declared", state_source="optimizer_checkpoint"
    )
    payload = row.to_payload()
    rebuilt = MemberRelationRow.from_payload(payload)
    assert rebuilt.params["carry_mode"] == "declared"
    assert rebuilt.params["state_source"] == "optimizer_checkpoint"
    assert rebuilt.params["evidence"]["schema"] == "version_boundary_v1"


def test_undeclared_param_key_still_refuses() -> None:
    """The required/optional split does NOT weaken the unknown-key check (D4)."""

    with pytest.raises(ValueError, match="undeclared param keys"):
        _successor_row(surprise=1)


def test_optional_keys_are_per_kind_closed() -> None:
    """successor_of's optional set does not leak onto other kinds."""

    with pytest.raises(ValueError, match="undeclared param keys"):
        MemberRelationRow(
            kind="alternative_of",
            from_member="ckpt_200",
            to_member="ckpt_100",
            params={"evidence": _evidence()},
        )


def test_required_keys_still_required() -> None:
    with pytest.raises(ValueError, match="missing required param keys"):
        MemberRelationRow(kind="forked_from", from_member="ckpt_200", to_member="ckpt_100")


def test_carry_mode_and_state_source_type_envelope() -> None:
    with pytest.raises(ValueError, match="carry_mode"):
        _successor_row(carry_mode="")
    with pytest.raises(ValueError, match="state_source"):
        _successor_row(state_source=7)


# ---------------------------------------------------------------------------
# Items 9 + 10: graded evidence items; contracted unchecked-reason menu.
# ---------------------------------------------------------------------------


def test_grade_vocabulary_reserved() -> None:
    assert {
        "verified",
        "consistent",
        "disclosed",
        "unchecked",
        "divergent",
    } == RELATION_CLAIM_GRADES
    assert "no_param_snapshot" in RELATION_UNCHECKED_REASONS
    assert "incomparable_basis" in RELATION_UNCHECKED_REASONS


def test_evidence_item_requires_contracted_grade() -> None:
    with pytest.raises(ValueError, match="outside the closed vocabulary"):
        _successor_row(evidence=_evidence(items=[{"grade": "excellent", "basis": None}]))


def test_evidence_item_requires_basis_key() -> None:
    """Item 9: the basis field is always present (expressible non-null)."""

    with pytest.raises(ValueError, match="missing the 'basis' key"):
        _successor_row(evidence=_evidence(items=[{"grade": "verified"}]))


def test_unchecked_grade_requires_contracted_reason() -> None:
    """D15: no lane may invent an uncontracted reason string."""

    with pytest.raises(ValueError, match="outside the contracted menu"):
        _successor_row(
            evidence=_evidence(items=[{"grade": "unchecked", "basis": None, "reason": "vibes"}])
        )


def test_no_param_snapshot_is_admitted() -> None:
    row = _successor_row(
        evidence=_evidence(
            items=[{"grade": "unchecked", "basis": None, "reason": "no_param_snapshot"}]
        )
    )
    assert row.params["evidence"]["items"][0]["reason"] == "no_param_snapshot"


def test_evidence_envelope_exact_keys() -> None:
    with pytest.raises(ValueError, match="exactly the keys"):
        _successor_row(evidence={"schema": "x", "items": []})
    with pytest.raises(ValueError, match="exactly the keys"):
        _successor_row(evidence=_evidence(extra="k"))


# ---------------------------------------------------------------------------
# Item 7: the per-row evidence byte budget.
# ---------------------------------------------------------------------------


def test_evidence_over_budget_refuses_typed() -> None:
    oversized = _evidence(
        items=[
            {"grade": "disclosed", "basis": "bulk", "blob": "x" * RELATION_EVIDENCE_BUDGET_BYTES}
        ]
    )
    with pytest.raises(BundleRelationError) as excinfo:
        _successor_row(evidence=oversized)
    assert excinfo.value.fields["code"] == "bundle_relation_evidence_over_budget"


def test_evidence_over_budget_code_survives_bundle_door(two_member_bundle: Bundle) -> None:
    """The distinct code must not flatten to schema_invalid at Bundle.relate."""

    oversized = _evidence(
        items=[
            {"grade": "disclosed", "basis": "bulk", "blob": "x" * RELATION_EVIDENCE_BUDGET_BYTES}
        ]
    )
    with pytest.raises(BundleRelationError) as excinfo:
        two_member_bundle.relate(
            {
                "kind": "successor_of",
                "from": "ckpt_200",
                "to": "ckpt_100",
                "params": {"evidence": oversized},
            }
        )
    assert excinfo.value.fields["code"] == "bundle_relation_evidence_over_budget"


# ---------------------------------------------------------------------------
# Item 8: direction pin.
# ---------------------------------------------------------------------------


def test_successor_of_direction_pin_documented() -> None:
    """from = the LATER member, pinned in the row docstring and module doc."""

    from torchlens.bundle import _relations as relations_module

    for text in (MemberRelationRow.__doc__ or "", relations_module.__doc__ or ""):
        assert "LATER member" in text, "successor_of direction pin missing from docs"


# ---------------------------------------------------------------------------
# Item 11 + 6: reserved id strings (registrations, not slots).
# ---------------------------------------------------------------------------


def test_reserved_evidence_schema_ids() -> None:
    assert {"version_boundary_v1", "turn_boundary_v1"} == RESERVED_EVIDENCE_SCHEMA_IDS


def test_reserved_sidecar_family_ids_registered() -> None:
    assert {
        "torchlens.input_origin",
        "torchlens.input_digest",
        "torchlens.boundary_facts",
    } == RESERVED_SIDECAR_FAMILY_IDS


def test_torchlens_sidecar_namespace_refuses_foreign_provider() -> None:
    """Squat prevention: only the TorchLens provider registers torchlens.*."""

    foreign = ProviderInfo(provider_id="someorg", distribution="someorg-pkg")
    family = SidecarFamily(
        family_id="torchlens.input_origin",
        schema_id="someorg.squat.v1",
        version=1,
        owner="someorg",
    )
    with pytest.raises(SidecarError) as excinfo:
        register_sidecar_family(family, provider=foreign)
    assert excinfo.value.fields["code"] == "sidecar_family_namespace_reserved"


def test_torchlens_provider_can_register_reserved_family() -> None:
    """The reservation is a squat guard, not a slot: the owner still can."""

    family = SidecarFamily(
        family_id="torchlens.input_origin",
        schema_id="torchlens.input_origin.v1",
        version=1,
        owner="torchlens",
    )
    register_sidecar_family(family)
    unregister_sidecar_family("torchlens.input_origin")


# ---------------------------------------------------------------------------
# Item 2 leg (a): unknown relation kinds — opaque vs refuse.
# ---------------------------------------------------------------------------


def _foreign_kind_payload() -> dict[str, object]:
    return {"kind": "someorg.custody_of", "anchor": "ckpt_100", "note": {"chain": [1, 2]}}


def test_unknown_namespaced_kind_loads_opaque_and_preserves() -> None:
    payload = [_successor_row().to_payload(), _foreign_kind_payload()]
    table = MemberRelationTable.from_payload(payload, unknown_kinds="opaque")
    assert table.opaque_kinds == ("someorg.custody_of",)
    opaque = table.rows[1]
    assert isinstance(opaque, OpaqueRelationRow)
    assert opaque.named_members() == ()
    # Verbatim re-save: the raw payload round-trips value-exact.
    assert table.to_payload()[1] == _foreign_kind_payload()


def test_unknown_namespaced_kind_refuses_under_strict_policy() -> None:
    payload = [_foreign_kind_payload()]
    with pytest.raises(ValueError, match="outside the closed vocabulary"):
        MemberRelationTable.from_payload(payload, unknown_kinds="refuse")


def test_bare_unknown_kind_refuses_under_both_policies() -> None:
    payload = [{"kind": "custody_of", "from": "a", "to": "b", "params": {}}]
    for policy in ("opaque", "refuse"):
        with pytest.raises(ValueError, match="outside the closed vocabulary"):
            MemberRelationTable.from_payload(payload, unknown_kinds=policy)  # type: ignore[arg-type]


def test_construction_always_refuses_unknown_kinds(two_member_bundle: Bundle) -> None:
    """Opaque rows enter from artifacts only; relate() refuses unknown kinds."""

    with pytest.raises(BundleRelationError) as excinfo:
        two_member_bundle.relate(_foreign_kind_payload())
    assert excinfo.value.fields["code"] == "bundle_relation_schema_invalid"


def test_default_from_payload_policy_is_refuse() -> None:
    """Construction semantics by default; the artifact door opts into opaque."""

    with pytest.raises(ValueError, match="outside the closed vocabulary"):
        MemberRelationTable.from_payload([_foreign_kind_payload()])


# ---------------------------------------------------------------------------
# Item 2 legs (a) + (c) end to end through tl.save / tl.load.
# ---------------------------------------------------------------------------


def _saved_bundle_dir(tmp_path, bundle: Bundle) -> str:
    target = tmp_path / "bundle.tlspec"
    bundle.save(target)
    return str(target)


def test_loader_leg_a_end_to_end(tmp_path, two_member_bundle: Bundle) -> None:
    """Foreign namespaced relation row: opaque load, disclosed, re-saved."""

    two_member_bundle.relate(
        {"kind": "successor_of", "from": "ckpt_200", "to": "ckpt_100", "params": {}}
    )
    path = _saved_bundle_dir(tmp_path, two_member_bundle)
    metadata_path = tmp_path / "bundle.tlspec" / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["member_relations"].append(_foreign_kind_payload())
    metadata_path.write_text(json.dumps(metadata))

    with pytest.warns(TorchLensWarning, match="unknown namespaced kinds") as caught:
        loaded = tl.load(path)
    opaque_warnings = [
        w.message
        for w in caught
        if getattr(w.message, "fields", {}).get("code") == "bundle_relation_unknown_kinds_opaque"
    ]
    assert opaque_warnings, "the opaque-kinds disclosure must carry its warning code"
    assert isinstance(loaded, Bundle)
    opaque_rows = [row for row in loaded.member_relations if isinstance(row, OpaqueRelationRow)]
    assert [row.kind for row in opaque_rows] == ["someorg.custody_of"]

    resaved = tmp_path / "resaved.tlspec"
    loaded.save(resaved)
    resaved_metadata = json.loads((resaved / "bundle.json").read_text())
    assert _foreign_kind_payload() in resaved_metadata["member_relations"]

    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(path, unknown_relations="refuse")
    assert excinfo.value.fields["code"] == "bundle_relation_schema_invalid"


def test_loader_leg_c_namespaced_section_preserves(tmp_path, two_member_bundle: Bundle) -> None:
    path = _saved_bundle_dir(tmp_path, two_member_bundle)
    metadata_path = tmp_path / "bundle.tlspec" / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["someorg.experiment_index"] = {"runs": [1, 2, 3]}
    metadata_path.write_text(json.dumps(metadata))

    with pytest.warns(TorchLensWarning, match="unknown namespaced sections") as caught:
        loaded = tl.load(path)
    section_warnings = [
        w.message
        for w in caught
        if getattr(w.message, "fields", {}).get("code") == "bundle_unknown_sections_preserved"
    ]
    assert section_warnings, "the preserved-sections disclosure must carry its warning code"
    assert isinstance(loaded, Bundle)
    assert loaded.preserved_sections == {"someorg.experiment_index": {"runs": [1, 2, 3]}}

    resaved = tmp_path / "resaved.tlspec"
    loaded.save(resaved)
    resaved_metadata = json.loads((resaved / "bundle.json").read_text())
    assert resaved_metadata["someorg.experiment_index"] == {"runs": [1, 2, 3]}


def test_loader_leg_c_bare_unknown_section_refuses(tmp_path, two_member_bundle: Bundle) -> None:
    """The historical silent load-then-destroy is banned in both directions."""

    from torchlens._io.format_errors import TorchLensIOError

    path = _saved_bundle_dir(tmp_path, two_member_bundle)
    metadata_path = tmp_path / "bundle.tlspec" / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["experiment_index"] = {"runs": [1]}
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(TorchLensIOError) as excinfo:
        tl.load(path)
    assert excinfo.value.fields["code"] == "bundle_section_unknown"


def test_unknown_relations_policy_validates(tmp_path, two_member_bundle: Bundle) -> None:
    from torchlens._errors import InvalidArgumentError

    path = _saved_bundle_dir(tmp_path, two_member_bundle)
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.load(path, unknown_relations="maybe")  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "unknown_relations_policy_invalid"


# ---------------------------------------------------------------------------
# Item 2 leg (b): unknown evidence schema ids load opaque.
# ---------------------------------------------------------------------------


def test_unknown_evidence_schema_id_loads_opaque(tmp_path, two_member_bundle: Bundle) -> None:
    """Leg (b): future torchlens.* (and foreign) schema ids must not break
    earlier readers — the envelope validates, items preserve, and the row
    stays a first-class validated row."""

    envelope = {
        "schema": "torchlens.future_boundary_v9",
        "items": [{"grade": "verified", "basis": "digest", "extra_future_key": [1]}],
        "facts_digest": "sha256:ff",
    }
    row = _successor_row(evidence=envelope)
    table = MemberRelationTable.from_payload([row.to_payload()])
    rebuilt = table.rows[0]
    assert isinstance(rebuilt, MemberRelationRow)
    assert rebuilt.params["evidence"]["items"][0]["extra_future_key"] == [1]
    # And end-to-end through the artifact door.
    two_member_bundle.relate(row)
    path = _saved_bundle_dir(tmp_path, two_member_bundle)
    loaded = tl.load(path)
    assert isinstance(loaded, Bundle)
    kinds = [r.kind for r in loaded.member_relations]
    assert "successor_of" in kinds
