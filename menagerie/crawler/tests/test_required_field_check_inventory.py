"""The machine-derived required-field-check inventory and its exact coverage.

The rung-2 census (2026-08) proved the metadata accuracy gate was a wall, not a
tripwire: the engine demanded one check per authored schema leaf (200+ paths per
proposal) while the checker was told nothing about the set and the author's own
generated claim vocabulary tags evidence at claim granularity, so no checker
verdict -- however correct -- could reach a canonical write. These tests pin the
repaired contract from both sides: the required set is derived by ONE machine
function from the proposal bytes, every checker envelope item is stamped with
it, a caller-invented inventory is refused, full coverage passes, and an
omitted, duplicated, grouped, or leaf-expanded check still refuses.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checker_dispatch import (
    CheckerDispatchError,
    build_metadata_vet_envelope,
)
from menagerie.crawler.metadata import (
    MetadataValidationError,
    validate_authored_facts_for_write,
)
from menagerie.crawler.proposal import (
    DEFAULT_GATED_CLAIMS,
    KEYWORD_CLAIM,
    required_metadata_field_checks,
)
from menagerie.crawler.tests.conftest import _model_facts, make_gate, make_model


def _accepted_gate_item(stable_id: str = "m_example") -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one accepted model's facts and its bound metadata gate item."""

    model = make_model(stable_id, accepted=True)
    gate = make_gate([stable_id], vet_identity=model["accuracy_gate"]["vet_identity"])
    return _model_facts(model), deepcopy(gate["items"][0])


@pytest.mark.smoke
def test_inventory_is_deterministic_sorted_and_claim_shaped() -> None:
    """The derivation is stable, ordered, and spans the closed claim vocabulary."""

    facts = _model_facts(make_model("m_example", accepted=True))
    first = required_metadata_field_checks(facts)
    second = required_metadata_field_checks(deepcopy(dict(facts)))
    assert first == second
    assert list(first) == sorted(first)
    assert KEYWORD_CLAIM in first
    assert set(first) - {KEYWORD_CLAIM, "external_metadata.citation"} == set(
        DEFAULT_GATED_CLAIMS - {"external_metadata.citation"}
    )
    # No per-leaf expansions and no section names ever appear.
    assert all("[" not in claim for claim in first)
    assert "identity" not in first


@pytest.mark.smoke
def test_citation_claim_follows_presence_and_paper_source() -> None:
    """The citation claim is conditional exactly like the author-side gate."""

    facts = dict(_model_facts(make_model("m_example", accepted=True)))
    assert "external_metadata.citation" in required_metadata_field_checks(facts)

    bare: dict[str, Any] = {"implementation": {"code_path": None}}
    assert "external_metadata.citation" not in required_metadata_field_checks(bare)

    paper_only = {
        "implementation": {"code_path": None},
        "source_resolution": {
            "sources": [
                {
                    "source_id": "source-paper",
                    "role": "introducing-paper",
                    "content_sha256": "sha256:" + "a" * 64,
                }
            ]
        },
    }
    assert "external_metadata.citation" in required_metadata_field_checks(paper_only)


@pytest.mark.smoke
def test_full_claim_coverage_passes_canonical_write_validation() -> None:
    """A checker that covers exactly the shipped inventory passes the gate."""

    facts, item = _accepted_gate_item()
    report = validate_authored_facts_for_write(facts, item)
    assert report.gated_fields == frozenset(required_metadata_field_checks(facts))


@pytest.mark.smoke
def test_omitting_one_required_claim_still_refuses_as_ungated() -> None:
    """Shipping the inventory makes coverage satisfiable, never optional."""

    facts, item = _accepted_gate_item()
    dropped = item["field_checks"].pop()
    with pytest.raises(MetadataValidationError, match="ungated authored facts"):
        validate_authored_facts_for_write(facts, item)
    assert dropped["field"] not in {check["field"] for check in item["field_checks"]}


@pytest.mark.smoke
def test_duplicate_grouped_and_leaf_expanded_checks_still_refuse() -> None:
    """Exactness holds in the other direction: nothing outside the list passes."""

    facts, item = _accepted_gate_item()
    duplicated = deepcopy(item)
    duplicated["field_checks"].append(deepcopy(duplicated["field_checks"][0]))
    with pytest.raises(MetadataValidationError, match="duplicate authored field check"):
        validate_authored_facts_for_write(facts, duplicated)

    for bad_field in (
        "identity",
        "citation; dates",
        "external_metadata.citation.year",
    ):
        extraneous = deepcopy(item)
        extraneous["field_checks"].append(
            {**deepcopy(item["field_checks"][0]), "field": bad_field}
        )
        with pytest.raises(
            MetadataValidationError, match="outside the required gated-claim checks"
        ):
            validate_authored_facts_for_write(facts, extraneous)


@pytest.mark.smoke
def test_envelope_items_are_stamped_with_the_machine_derivation(tmp_path: Path) -> None:
    """Every metadata envelope item carries the inventory the machine derived."""

    gate = make_gate([f"m_{index}" for index in range(10)])
    items = [_request_item(item) for item in gate["items"]]
    envelope = build_metadata_vet_envelope(
        items,
        gate_round=1,
        output_path=tmp_path / "result.json",
        checker_model="codex",
        checker_version="test",
        request_nonce="nonce-1",
    )
    for envelope_item in envelope["items"]:
        expected = list(
            required_metadata_field_checks(envelope_item["proposal"]["proposed_facts"])
        )
        assert envelope_item["required_field_checks"] == expected


@pytest.mark.smoke
def test_envelope_refuses_a_caller_invented_inventory(tmp_path: Path) -> None:
    """The inventory is machine-owned: a conflicting supplied list refuses."""

    gate = make_gate([f"m_{index}" for index in range(10)])
    items = [_request_item(item) for item in gate["items"]]
    items[0]["required_field_checks"] = ["external_metadata.description"]
    with pytest.raises(CheckerDispatchError, match="machine derivation"):
        build_metadata_vet_envelope(
            items,
            gate_round=1,
            output_path=tmp_path / "result.json",
            checker_model="codex",
            checker_version="test",
            request_nonce="nonce-1",
        )


def _request_item(item: dict[str, Any]) -> dict[str, Any]:
    """Build one minimal identity-bound checker request item."""

    return {
        "work_id": item["work_id"],
        "campaign_root_work_id": item["campaign_root_work_id"],
        "stable_id": item["stable_id"],
        "family_representative_id": item["family_representative_id"],
        "fidelity_identity": item["fidelity_identity"],
        "vet_identity": item["vet_identity"],
        "verified_hashes": deepcopy(item["verified_hashes"]),
        "proposal": {
            "description": "scoped test proposal",
            "proposed_facts": {"implementation": {"code_path": None}},
        },
        "source_manifest": {"sources": []},
        "model_dir": f"/menagerie-checker-test/{item['stable_id']}/author/model",
    }
