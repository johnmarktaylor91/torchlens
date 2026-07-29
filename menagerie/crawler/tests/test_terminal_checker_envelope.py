"""Terminal checker verdict routing and grounded-evidence envelope tripwires.

Two live campaign-killing defects are pinned here (rung 1d, 2026-07-29):

1. A legitimate checker verdict was discarded because the checker model omitted
   ``gate_id`` -- a machine-owned identity it could not observe -- and the
   resulting typed contract rejection propagated as an undifferentiated
   ``DriverIntegrationError`` that ended the whole run.
2. The frozen terminal envelope carried evidence *identifiers* plus synthesized
   excerpt rows (source IDs assigned by round-robin index, ``supports`` stamped
   with the typed predicate), so no verdict on it could ever be grounded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import AuthorResultBinding, BlockedRecommendation
from menagerie.crawler.checker_dispatch import (
    CheckerDispatchError,
    apply_machine_owned_gate_fields,
    build_terminal_disposition_envelope,
    machine_owned_gate_fields,
    validate_checker_result_mapping,
)
from menagerie.crawler.driver_admission import _raise_for_checker_exit
from menagerie.crawler.driver_contracts import AuthorArtifact, DriverIntegrationError
from menagerie.crawler.driver_contracts import RetryableOperatorError
from menagerie.crawler.driver_models import _terminal_checker_item
from menagerie.crawler.identity import compute_evidence_identity, stable_hash
from menagerie.crawler.operator_checker import TERMINAL_CHECKER_MODEL
from menagerie.crawler.terminal_evidence import (
    GROUNDED,
    TERMINAL_EVIDENCE_FILENAME,
    TERMINAL_LICENSE_FILENAME,
    UNRESOLVED,
    resolve_terminal_evidence,
)
from menagerie.crawler.tests.conftest import HASH, make_gate

EXCERPT_TEXT = "activation=leaky\n"
SECOND_EXCERPT_TEXT = "filters=16\n"


def _excerpts() -> list[dict[str, Any]]:
    """Return two literal, locator-bearing excerpt records."""

    return [
        {
            "evidence_id": "ev-one",
            "source_id": "source-1",
            "locator": "cfg/yolov3-tiny.cfg lines 25-31",
            "text": EXCERPT_TEXT,
            "text_sha256": stable_hash(EXCERPT_TEXT),
            "supports": ["blocked-prerequisite"],
        },
        {
            "evidence_id": "ev-two",
            "source_id": "source-2",
            "locator": "cfg/yolov3-tiny.cfg lines 32-38",
            "text": SECOND_EXCERPT_TEXT,
            "text_sha256": stable_hash(SECOND_EXCERPT_TEXT),
            "supports": ["blocked-prerequisite"],
        },
    ]


def _blocked_artifact(
    tmp_path: Path,
    *,
    excerpts: list[dict[str, Any]] | None,
    evidence_identity: str | None = None,
    license_record: dict[str, Any] | None = None,
) -> AuthorArtifact:
    """Stage one BLOCKED recommendation with optional frozen author records.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory used as the campaign work root.
    excerpts:
        Frozen excerpt records to publish, or ``None`` to publish none.
    evidence_identity:
        Declared evidence identity; defaults to the identity of ``excerpts``.
    license_record:
        Frozen license record to publish, or ``None`` to publish none.

    Returns
    -------
    AuthorArtifact
        Privately staged terminal artifact.
    """

    author_root = tmp_path / "work" / "m_example" / "author"
    model_dir = author_root / "model"
    model_dir.mkdir(parents=True)
    if excerpts is not None:
        (author_root / TERMINAL_EVIDENCE_FILENAME).write_text(
            json.dumps({"excerpts": excerpts}), encoding="utf-8"
        )
    if license_record is not None:
        (author_root / TERMINAL_LICENSE_FILENAME).write_text(
            json.dumps(license_record), encoding="utf-8"
        )
    resolved_identity = (
        evidence_identity
        if evidence_identity is not None
        else compute_evidence_identity(excerpts or [])
    )
    payload = {
        "arm": "BLOCKED",
        "stage": "source",
        "reason_code": "missing-material-source",
        "prerequisite_ids": ["prereq-1"],
        "evidence_ids": ["ev-one", "ev-two"],
        "evidence_identity": resolved_identity,
        "license_identity": (
            stable_hash(license_record) if license_record is not None else HASH
        ),
    }
    payload["recommendation_sha256"] = stable_hash(payload)
    raw_fields = {
        "result_id": "result-blocked",
        "result_sha256": HASH,
        "stable_id": "m_example",
        "work_id": "work-m_example",
        "campaign_id": "work-m_example",
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": HASH,
        "intake_snapshot_id": "intake-1",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": HASH,
        "created_at": "2026-07-29T00:00:00Z",
    }
    binding = AuthorResultBinding(raw_result={**raw_fields, "payload": payload}, **raw_fields)
    result = BlockedRecommendation(
        binding=binding,
        stage="source",
        reason_code="missing-material-source",
        prerequisite_ids=("prereq-1",),
        evidence_ids=("ev-one", "ev-two"),
        evidence_identity=resolved_identity,
        license_identity=str(payload["license_identity"]),
        recommendation_sha256=str(payload["recommendation_sha256"]),
    )
    return AuthorArtifact(
        author_result=result,
        source_manifest={
            "manifest_sha256": HASH,
            "sources": [{"source_id": "source-1"}, {"source_id": "source-2"}],
        },
        model_dir=model_dir,
    )


def _terminal_envelope(item: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """Build one terminal-disposition envelope around a checker item pack."""

    return build_terminal_disposition_envelope(
        item,
        gate_round=1,
        output_path=tmp_path / "checker" / "result.json",
        checker_model=TERMINAL_CHECKER_MODEL,
        checker_version="test",
        request_nonce="terminal-m_example",
    )


def _rejected_verdict_body() -> dict[str, Any]:
    """Return a complete terminal verdict body with no machine-owned scaffold."""

    gate = make_gate(["m_example"], gate_kind="terminal_disposition")
    gate["items"][0]["terminal_disposition"] = {
        "author_result_id": "result-blocked",
        "author_result_sha256": HASH,
        "handoff_proposal_id": None,
        "handoff_sha256": None,
        "kind": "BLOCKED",
        "predicate": "blocked-prerequisite",
        "verdict": "rejected",
        "source_manifest_identity": HASH,
        "source_ids": ["source-1", "source-2"],
        "evidence_identity": HASH,
        "evidence_ids": ["ev-one", "ev-two"],
        "license_identity": HASH,
        "findings": ["evidence excerpts are not inspectable"],
    }
    gate["items"][0]["verdict"] = "inaccurate"
    gate["items"][0]["integrity"]["verdict"] = "inaccurate"
    gate["items"][0]["required_repairs"] = ["supply the frozen evidence records"]
    return gate


# -- 1. a verdict is a per-model result, not an infrastructure failure ---------


def test_verdict_missing_machine_owned_scaffold_is_not_discarded(tmp_path: Path) -> None:
    """A complete rejection survives even when the checker omits every machine field.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, excerpts=_excerpts())
    envelope = _terminal_envelope(_terminal_checker_item(artifact), tmp_path)
    verdict = _rejected_verdict_body()
    item = verdict["items"][0]
    item["work_id"] = envelope["items"][0]["work_id"]
    item["stable_id"] = envelope["items"][0]["stable_id"]
    item["family_representative_id"] = envelope["items"][0]["family_representative_id"]
    item["fidelity_identity"] = envelope["items"][0]["fidelity_identity"]
    item["vet_identity"] = envelope["items"][0]["vet_identity"]
    item["verified_hashes"] = envelope["items"][0]["verified_hashes"]
    for machine_field in machine_owned_gate_fields(envelope):
        # ``schema_version`` is the one machine field the checker can read off
        # the envelope, so the live omission was exactly the rest of them.
        if machine_field != "schema_version":
            verdict.pop(machine_field, None)
    verdict.pop("checker", None)

    # Pre-fix behaviour: the raw verdict is refused for a field the checker
    # could not observe, and the whole campaign died on that refusal.
    with pytest.raises(CheckerDispatchError, match="gate_id"):
        validate_checker_result_mapping(verdict, envelope)

    stamped = apply_machine_owned_gate_fields(
        verdict,
        envelope,
        started_at="2026-07-29T00:00:00Z",
        finished_at="2026-07-29T00:05:00Z",
    )
    validated = validate_checker_result_mapping(stamped, envelope)
    assert validated["items"][0]["terminal_disposition"]["verdict"] == "rejected"
    assert validated["gate_identity"] == envelope["envelope_sha256"]


def test_machine_owned_gate_identity_is_never_taken_from_the_checker(tmp_path: Path) -> None:
    """A checker-supplied identity is overwritten, never believed.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, excerpts=_excerpts())
    envelope = _terminal_envelope(_terminal_checker_item(artifact), tmp_path)
    verdict = _rejected_verdict_body()
    verdict["gate_id"] = "gate-fabricated"
    verdict["dispatcher_identity"] = "sha256:" + "f" * 64
    stamped = apply_machine_owned_gate_fields(
        verdict,
        envelope,
        started_at="2026-07-29T00:00:00Z",
        finished_at="2026-07-29T00:05:00Z",
    )
    expected = machine_owned_gate_fields(envelope)
    assert stamped["gate_id"] == expected["gate_id"] != "gate-fabricated"
    assert stamped["dispatcher_identity"] == expected["dispatcher_identity"]


# -- 2. only a genuine execution failure is an integration error ---------------


def test_retryable_and_unavailable_checker_exits_do_not_end_the_campaign() -> None:
    """Declared-transient wrapper exits are bounded transport retries."""

    for returncode in (75, 78):
        with pytest.raises(RetryableOperatorError):
            _raise_for_checker_exit(returncode, "", "transport failed")


def test_genuine_checker_execution_failure_still_raises_integration_error() -> None:
    """A crash, a missing binary, or a refused contract stays an integration error."""

    with pytest.raises(DriverIntegrationError) as crashed:
        _raise_for_checker_exit(127, "", "codex: command not found")
    assert not isinstance(crashed.value, RetryableOperatorError)
    with pytest.raises(DriverIntegrationError, match="rejected the gate contract"):
        _raise_for_checker_exit(64, "", "no valid gate was produced")
    _raise_for_checker_exit(0, "", "")


# -- 3. the envelope carries literal excerpts and locators ---------------------


def test_terminal_envelope_carries_literal_excerpts_and_locators(tmp_path: Path) -> None:
    """Every referenced evidence ID arrives with its verbatim text and locator.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    license_record = {"source_dispositions": [{"source_id": "source-1", "spdx": "MIT"}]}
    artifact = _blocked_artifact(
        tmp_path, excerpts=_excerpts(), license_record=license_record
    )
    item = _terminal_checker_item(artifact)
    pack = item["evidence_pack"]
    assert pack["resolution"] == GROUNDED
    by_id = {excerpt["evidence_id"]: excerpt for excerpt in pack["excerpts"]}
    assert set(by_id) == {"ev-one", "ev-two"}
    for excerpt in by_id.values():
        assert excerpt["locator"]
        assert excerpt["text"]
        assert excerpt["text_sha256"]
    # Real provenance, not the round-robin index the envelope used to invent.
    assert by_id["ev-one"]["source_id"] == "source-1"
    assert by_id["ev-two"]["source_id"] == "source-2"
    assert compute_evidence_identity(pack["excerpts"]) == artifact.author_result.evidence_identity
    assert item["license_pack"]["resolution"] == GROUNDED
    assert item["license_pack"]["record"] == license_record
    preimage = item["recommendation_preimage"]
    assert stable_hash(preimage) == item["recommendation_sha256"]
    envelope = _terminal_envelope(item, tmp_path)
    assert envelope["items"][0]["evidence_pack"]["resolution"] == GROUNDED


def test_terminal_envelope_refuses_a_grounded_claim_without_literal_excerpts(
    tmp_path: Path,
) -> None:
    """A pack that claims grounding without inspectable text cannot be dispatched.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, excerpts=_excerpts())
    item = _terminal_checker_item(artifact)
    # Exactly the shape the envelope used to ship: identifiers only.
    item["evidence_pack"]["excerpts"] = [
        {"evidence_id": "ev-one", "source_id": "source-1", "supports": ["blocked-prerequisite"]},
        {"evidence_id": "ev-two", "source_id": "source-2", "supports": ["blocked-prerequisite"]},
    ]
    with pytest.raises(CheckerDispatchError, match="no literal excerpt"):
        _terminal_envelope(item, tmp_path)


# -- 4. a genuinely incomplete evidence pack is still rejected -----------------


def test_unbound_evidence_pack_resolves_unresolved_and_cannot_be_accepted(
    tmp_path: Path,
) -> None:
    """Records that do not recompute to the declared identity are not evidence.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(
        tmp_path, excerpts=_excerpts(), evidence_identity="sha256:" + "b" * 64
    )
    item = _terminal_checker_item(artifact)
    pack = item["evidence_pack"]
    assert pack["resolution"] == UNRESOLVED
    assert pack["excerpts"] == []
    assert set(pack["unresolved_evidence_ids"]) == {"ev-one", "ev-two"}
    assert "declared" in str(pack["unresolved_reason"])
    # The envelope still builds -- the gap is stated, not hidden -- so the
    # independent checker sees exactly what it cannot verify, instead of the
    # synthesized rows that used to make the claim look grounded.
    envelope = _terminal_envelope(item, tmp_path)
    shipped = envelope["items"][0]["evidence_pack"]
    assert shipped["resolution"] == UNRESOLVED
    assert shipped["excerpts"] == []
    assert set(shipped["unresolved_evidence_ids"]) == {"ev-one", "ev-two"}
    assert shipped["unresolved_reason"]


def test_absent_evidence_records_are_declared_not_invented(tmp_path: Path) -> None:
    """With no frozen records at all the envelope names every unresolved ID.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    artifact = _blocked_artifact(tmp_path, excerpts=None, evidence_identity=HASH)
    pack = _terminal_checker_item(artifact)["evidence_pack"]
    assert pack["resolution"] == UNRESOLVED
    assert pack["excerpts"] == []
    assert set(pack["unresolved_evidence_ids"]) == {"ev-one", "ev-two"}
    assert pack["checked_source_ids"] == ["source-1", "source-2"]


def test_partial_evidence_records_are_unresolved(tmp_path: Path) -> None:
    """A pack that binds its identity but omits one excerpt is not grounded.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    excerpts = _excerpts()
    del excerpts[1]["locator"]
    artifact = _blocked_artifact(tmp_path, excerpts=excerpts)
    resolution = resolve_terminal_evidence(
        author_root=artifact.model_dir.parent,
        evidence_ids=("ev-one", "ev-two"),
        evidence_identity=artifact.author_result.evidence_identity,
    )
    assert resolution.resolution == UNRESOLVED
    assert resolution.unresolved_evidence_ids == ("ev-two",)
