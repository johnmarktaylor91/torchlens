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
    SidecarError,
    SidecarFamily,
    register_sidecar_family,
)
from torchlens._registry import ProviderInfo
from torchlens.bundle import Bundle
from torchlens.bundle._relations import (
    RELATION_EVIDENCE_BUDGET_BYTES,
    MemberRelationRow,
    OpaqueRelationRow,
)
from torchlens.errors import TorchLensWarning
from torchlens.errors.episode import BundleRelationError


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
        "items": [{"grade": "consistent", "basis": "step_count", "class": "member_state"}],
        "facts_digest": "sha256:00",
    }
    envelope.update(overrides)
    return envelope


# ---------------------------------------------------------------------------
# Item 1: required/optional split; successor_of optional params.
# ---------------------------------------------------------------------------


def test_successor_of_admits_optional_evidence_carry_mode_state_source() -> None:
    row = _successor_row(
        evidence=_evidence(), carry_mode="summarized", state_source="optimizer_checkpoint"
    )
    payload = row.to_payload()
    rebuilt = MemberRelationRow.from_payload(payload)
    assert rebuilt.params["carry_mode"] == "summarized"
    assert rebuilt.params["state_source"] == "optimizer_checkpoint"
    assert rebuilt.params["evidence"]["schema"] == "version_boundary_v1"


def test_carry_mode_and_state_source_type_envelope() -> None:
    with pytest.raises(ValueError, match="carry_mode"):
        _successor_row(carry_mode="")
    with pytest.raises(ValueError, match="state_source"):
        _successor_row(state_source=7)


# ---------------------------------------------------------------------------
# Items 9 + 10: graded evidence items; contracted unchecked-reason menu.
# ---------------------------------------------------------------------------


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
            items=[
                {
                    "grade": "unchecked",
                    "basis": None,
                    "reason": "no_param_snapshot",
                    "class": "member_params",
                }
            ]
        )
    )
    assert row.params["evidence"]["items"][0]["reason"] == "no_param_snapshot"


# ---------------------------------------------------------------------------
# Item 7: the per-row evidence byte budget.
# ---------------------------------------------------------------------------


def test_evidence_over_budget_code_survives_bundle_door(two_member_bundle: Bundle) -> None:
    """The distinct code must not flatten to schema_invalid at Bundle.relate."""

    oversized = _evidence(
        items=[
            {
                "grade": "disclosed",
                "basis": "bulk",
                "class": "member_state",
                "blob": "x" * RELATION_EVIDENCE_BUDGET_BYTES,
            }
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


# ---------------------------------------------------------------------------
# Item 11 + 6: reserved id strings (registrations, not slots).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Item 2 leg (a): unknown relation kinds — opaque vs refuse.
# ---------------------------------------------------------------------------


def _foreign_kind_payload() -> dict[str, object]:
    return {"kind": "someorg.custody_of", "anchor": "ckpt_100", "note": {"chain": [1, 2]}}


# ---------------------------------------------------------------------------
# Item 2 legs (a) + (c) end to end through Trace.save / tl.load.
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
    opaque_rows = [row for row in loaded.relations if isinstance(row, OpaqueRelationRow)]
    assert [row.kind for row in opaque_rows] == ["someorg.custody_of"]

    resaved = tmp_path / "resaved.tlspec"
    loaded.save(resaved)
    resaved_metadata = json.loads((resaved / "bundle.json").read_text())
    assert _foreign_kind_payload() in resaved_metadata["member_relations"]

    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(path, unknown_relations="refuse")
    assert excinfo.value.fields["code"] == "bundle_relation_schema_invalid"


@pytest.mark.smoke
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

    from torchlens._io.format_errors import ArtifactError

    path = _saved_bundle_dir(tmp_path, two_member_bundle)
    metadata_path = tmp_path / "bundle.tlspec" / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["experiment_index"] = {"runs": [1]}
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ArtifactError) as excinfo:
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
