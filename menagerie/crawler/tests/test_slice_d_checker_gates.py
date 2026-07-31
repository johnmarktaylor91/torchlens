"""Checker dispatch and bounded gate-routing tests for crawler Slice D."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorResultBinding,
    DeferRecommendation,
    HandoffExecution,
)
from menagerie.crawler.authority import AuthorityDerivationError, load_current_gate_proof
from menagerie.crawler.checker_dispatch import (
    LEDGER_ASSIGNED_GATE_FIELDS,
    TERMINAL_VERDICT_LOCKSTEP,
    VERDICT_SEVERITY,
    CheckerDispatchError,
    _validate_item_decision,
    apply_machine_owned_gate_fields,
    build_metadata_vet_envelope,
    classify_checker_response,
    machine_owned_gate_fields,
    validate_checker_result,
)
from menagerie.crawler.constants import (
    GATE_SCHEMA_VERSION_V3,
    AccuracyVerdict,
    CheckerPauseReason,
    FidelityVerdict,
    GateKind,
    GateRoute,
)
from menagerie.crawler.gates import (
    GateRoutingError,
    next_metadata_batch_ids,
    route_fidelity_gate,
    route_metadata_gate,
    validate_terminal_disposition_gate,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import LedgerConflictError
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.schema import PayloadValidationError, validate_payload
from menagerie.crawler.tests.conftest import (
    HASH,
    make_authority_context,
    make_attempt,
    make_author_proposal,
    make_gate,
    make_model,
)


def _stamped(gate: dict[str, Any], envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the machine-owned gate scaffold exactly as the wrapper does.

    The shared ``make_gate`` fixture carries a full placeholder scaffold, and a
    checker is instructed to OMIT every machine-owned field. Handing the
    fixture's scaffold to the stamp modelled a checker that fabricates
    identities and relied on the stamp silently correcting them, which is the
    laundering path the stamp now refuses. Strip them first, as a compliant
    checker would.

    ``LEDGER_ASSIGNED_GATE_FIELDS`` are stripped alongside the scaffold but are
    NOT part of it: the machine derives the scaffold up front, whereas the ledger
    assigns those two at append time, so nothing -- checker or wrapper -- may
    carry a value for them into a not-yet-appended gate.
    """

    checker = gate.get("checker", {})
    candidate = dict(gate)
    for field in (*machine_owned_gate_fields(envelope), *LEDGER_ASSIGNED_GATE_FIELDS):
        candidate.pop(field, None)
    candidate.pop("checker", None)
    return apply_machine_owned_gate_fields(
        candidate,
        envelope,
        started_at=str(checker.get("started_at", "2026-01-01T00:00:00Z")),
        finished_at=str(checker.get("finished_at", "2026-01-01T00:00:01Z")),
    )


def _checker_item_pack(item: dict[str, Any]) -> dict[str, Any]:
    """Build one checker request item from an expected gate item.

    Parameters
    ----------
    item:
        Expected gate result item.

    Returns
    -------
    dict[str, Any]
        Identity/hash-bound checker artifact pack.
    """

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
        "evidence": {"excerpts": []},
        # Every real envelope item names its author directory, because the
        # envelope derives the declared ``source-cas`` read root from it.
        "model_dir": f"/menagerie-checker-test/{item['stable_id']}/author/model",
    }


@pytest.mark.parametrize("rung", ["R2_VENDOR", "R4_REIMPLEMENT"])
def test_typed_proposal_code_manifest_reaches_checker_envelope(tmp_path: Path, rung: str) -> None:
    """Typed R2/R4 recursive manifests enter the checker envelope.

    Parameters
    ----------
    tmp_path:
        Isolated checker result directory.
    rung:
        Typed source rung represented by the proposal.
    """

    gate = make_gate([f"m_{rung.lower()}"])
    item = _checker_item_pack(gate["items"][0])
    manifest = [{"path": "adapter.py", "sha256": HASH}]
    item["proposal"]["proposed_facts"] = {
        "source_resolution": {"rung": rung},
        "implementation": {
            "code_path": "adapter.py",
            "code_sha256": HASH,
            "code_manifest": manifest,
        },
    }
    item["verified_hashes"].update({"code": HASH, "code_manifest": stable_hash(manifest)})

    envelope = build_metadata_vet_envelope(
        [item],
        gate_round=1,
        output_path=tmp_path / "typed" / "result.json",
        checker_model="codex",
        checker_version="test",
        request_nonce=f"typed-{rung}",
        final_tail=True,
    )

    assert envelope["items"][0]["verified_hashes"]["code_manifest"] == stable_hash(manifest)


def test_declarative_proposal_rejects_stray_code_manifest(tmp_path: Path) -> None:
    """A no-code R1 proposal cannot claim a recursive code manifest."""

    gate = make_gate(["m_r1_stray_manifest"])
    item = _checker_item_pack(gate["items"][0])
    item["proposal"]["proposed_facts"]["source_resolution"] = {"rung": "R1_LIBRARY"}
    item["verified_hashes"]["code_manifest"] = HASH

    with pytest.raises(CheckerDispatchError, match="exact proposal/artifact pack"):
        build_metadata_vet_envelope(
            [item],
            gate_round=1,
            output_path=tmp_path / "declarative" / "result.json",
            checker_model="codex",
            checker_version="test",
            request_nonce="declarative-stray-manifest",
            final_tail=True,
        )


def test_typed_proposal_rejects_missing_code_manifest(tmp_path: Path) -> None:
    """A typed proposal cannot reach a checker without its closure digest."""

    gate = make_gate(["m_r2_missing_manifest"])
    item = _checker_item_pack(gate["items"][0])
    item["proposal"]["proposed_facts"] = {
        "source_resolution": {"rung": "R2_VENDOR"},
        "implementation": {"code_path": "adapter.py", "code_sha256": HASH},
    }
    item["verified_hashes"]["code"] = HASH

    with pytest.raises(CheckerDispatchError, match="exact proposal/artifact pack"):
        build_metadata_vet_envelope(
            [item],
            gate_round=1,
            output_path=tmp_path / "typed-missing" / "result.json",
            checker_model="codex",
            checker_version="test",
            request_nonce="typed-missing-manifest",
            final_tail=True,
        )


def test_metadata_batch_envelope_validates_every_item_result(tmp_path: Path) -> None:
    """A fresh 10-item batch round-trips only with all independent bindings."""

    gate = make_gate()
    items = [_checker_item_pack(item) for item in gate["items"]]
    result_path = tmp_path / "result.json"
    envelope = build_metadata_vet_envelope(
        items,
        gate_round=1,
        output_path=result_path,
        checker_model="codex",
        checker_version="test",
        request_nonce="fresh-1",
    )
    gate = _stamped(gate, envelope)
    result_path.write_text(json.dumps(gate))
    validated = validate_checker_result(result_path, envelope)
    assert validated["batch_size"] == 10


def test_metadata_final_tail_requires_explicit_dispatch_flag(tmp_path: Path) -> None:
    """Only an explicitly final dispatcher request may contain fewer than ten items."""

    gate = make_gate(["m_tail"])
    items = [_checker_item_pack(gate["items"][0])]
    with pytest.raises(CheckerDispatchError, match="10--20"):
        build_metadata_vet_envelope(
            items,
            gate_round=1,
            output_path=tmp_path / "rejected" / "result.json",
            checker_model="codex",
            checker_version="test",
            request_nonce="ordinary-short-batch",
        )
    result_path = tmp_path / "accepted" / "result.json"
    envelope = build_metadata_vet_envelope(
        items,
        gate_round=1,
        output_path=result_path,
        checker_model="codex",
        checker_version="test",
        request_nonce="final-short-batch",
        final_tail=True,
    )
    gate = _stamped(gate, envelope)
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(gate), encoding="utf-8")
    assert validate_checker_result(result_path, envelope)["batch_size"] == 1


def test_checker_result_rejects_partial_or_mismatched_item(tmp_path: Path) -> None:
    """One missing or independently mismatched item invalidates the result envelope."""

    gate = make_gate()
    items = [_checker_item_pack(item) for item in gate["items"]]
    result_path = tmp_path / "result.json"
    envelope = build_metadata_vet_envelope(
        items,
        gate_round=1,
        output_path=result_path,
        checker_model="codex",
        checker_version="test",
        request_nonce="fresh-2",
    )
    gate["items"][0]["verified_hashes"]["evidence"] = "sha256:" + "c" * 64
    gate = _stamped(gate, envelope)
    result_path.write_text(json.dumps(gate))
    with pytest.raises(CheckerDispatchError, match="mismatched binding"):
        validate_checker_result(result_path, envelope)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("429 rate limit exceeded; retry after reset", CheckerPauseReason.RATE_LIMIT),
        ("You have hit your usage limit.", CheckerPauseReason.QUOTA_EXHAUSTED),
    ],
)
def test_rate_and_quota_responses_classify_to_typed_pause(
    body: str, reason: CheckerPauseReason
) -> None:
    """Provider capacity responses become wakeup-layer signals, not verdicts.

    Parameters
    ----------
    body:
        Provider response text.
    reason:
        Expected closed pause reason.
    """

    signal = classify_checker_response(429, body, retry_after_seconds=60)
    assert signal is not None
    assert signal.reason is reason
    assert signal.retry_after_seconds == 60


def _inaccurate_metadata_gate() -> dict[str, Any]:
    """Return a metadata gate with one independently inaccurate item.

    Returns
    -------
    dict[str, Any]
        Complete metadata gate.
    """

    gate = make_gate()
    item = gate["items"][0]
    item["integrity"]["verdict"] = "inaccurate"
    item["integrity"]["excerpt_discrepancies"] = ["altered excerpt"]
    item["verdict"] = "inaccurate"
    item["required_repairs"] = ["restore literal excerpt"]
    return gate


def _rung_checked_fidelity_gate(verdict: AccuracyVerdict) -> dict[str, Any]:
    """Return a matching fidelity gate with an independently checked source rung.

    Parameters
    ----------
    verdict:
        Independent source-ladder accuracy verdict.

    Returns
    -------
    dict[str, Any]
        Complete one-model fidelity gate.
    """

    gate = make_gate(["m_example"], gate_kind="fidelity", fidelity_identity=HASH)
    gate["items"][0]["rung_check"] = {
        "selected_rung": "R4_REIMPLEMENT",
        "highest_applicable": (
            "R4_REIMPLEMENT" if verdict is AccuracyVerdict.ACCURATE else "R2_VENDOR"
        ),
        "verdict": verdict.value,
        "findings": (
            ["usable upstream implementation exists"]
            if verdict is AccuracyVerdict.INACCURATE
            else []
        ),
    }
    return gate


def _ledger_paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated reducer ledger paths for a gate regression.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    LedgerPaths
        Three isolated canonical ledger paths.
    """

    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def test_metadata_gate_blocks_write_requeues_then_human_fails() -> None:
    """An inaccurate item never writes and exhausts a bounded next-batch loop."""

    gate = _inaccurate_metadata_gate()
    first = route_metadata_gate(gate, {}, max_repairs=2)
    assert first[0].canonical_write_allowed is False
    assert first[0].route is GateRoute.REQUEUE_NEXT_BATCH
    assert next_metadata_batch_ids(first) == ("m_0",)
    terminal = route_metadata_gate(gate, {"m_0": 2}, max_repairs=2)
    assert terminal[0].route is GateRoute.HUMAN_FAIL
    assert terminal[0].human_review_required is True
    accurate_ids = {decision.stable_id for decision in first if decision.canonical_write_allowed}
    assert "m_0" not in accurate_ids


@pytest.mark.parametrize("verdict", list(FidelityVerdict))
def test_five_way_fidelity_routes_without_proposal_mutation(
    verdict: FidelityVerdict,
) -> None:
    """All frozen fidelity outcomes route independently and leave proposals unchanged.

    Parameters
    ----------
    verdict:
        Frozen five-way fidelity verdict.
    """

    gate = make_gate(["m_example"], gate_kind="fidelity", fidelity_identity=HASH)
    item = gate["items"][0]
    item["fidelity"]["verdict"] = verdict.value
    item["fidelity"]["permanent_scar"] = verdict is FidelityVerdict.SLOP
    if verdict in {FidelityVerdict.MAJOR_DRIFT, FidelityVerdict.SLOP}:
        item["verdict"] = "inaccurate"
    elif verdict is FidelityVerdict.CANNOT_VERIFY:
        item["verdict"] = "cannot-verify"
    proposal = make_author_proposal()
    before = deepcopy(proposal)
    decision = route_fidelity_gate(gate, proposal)
    assert decision.verdict is verdict
    assert decision.accepted_for_fidelity is (
        verdict in {FidelityVerdict.MATCH, FidelityVerdict.MINOR_DRIFT}
    )
    assert proposal == before


def test_inaccurate_rung_check_blocks_matching_fidelity_gate() -> None:
    """R4 reimplementation is refused when the checker finds usable R2 source."""

    gate = _rung_checked_fidelity_gate(AccuracyVerdict.INACCURATE)
    decision = route_fidelity_gate(gate, make_author_proposal())
    metadata_gate = make_gate()
    metadata_gate["items"][0]["rung_check"] = deepcopy(gate["items"][0]["rung_check"])
    metadata_decision = route_metadata_gate(metadata_gate, {}, max_repairs=2)[0]
    assert decision.verdict is FidelityVerdict.MATCH
    assert decision.accepted_for_fidelity is False
    assert decision.canonical_write_allowed is False
    assert decision.route is GateRoute.BLOCK_FIDELITY
    assert metadata_decision.canonical_write_allowed is False
    assert metadata_decision.route is GateRoute.REQUEUE_NEXT_BATCH


def test_accurate_rung_check_allows_matching_fidelity_gate() -> None:
    """A matching fidelity result remains accepted when its rung check is accurate."""

    gate = _rung_checked_fidelity_gate(AccuracyVerdict.ACCURATE)
    decision = route_fidelity_gate(gate, make_author_proposal())
    metadata_gate = make_gate()
    metadata_gate["items"][0]["rung_check"] = deepcopy(gate["items"][0]["rung_check"])
    metadata_decision = route_metadata_gate(metadata_gate, {}, max_repairs=2)[0]
    assert decision.accepted_for_fidelity is True
    assert decision.canonical_write_allowed is True
    assert decision.route is GateRoute.ACCEPT
    assert metadata_decision.canonical_write_allowed is True
    assert metadata_decision.route is GateRoute.ACCEPT


def test_accurate_rung_rejects_highest_applicable_mismatch_at_metadata_admission() -> None:
    """M3 accurate routing cannot select below the highest applicable source rung."""

    gate = make_gate(["m_example"])
    gate["items"][0]["rung_check"].update(
        {
            "selected_rung": "R4_REIMPLEMENT",
            "highest_applicable": "R2_VENDOR",
            "verdict": "accurate",
        }
    )
    with pytest.raises(
        GateRoutingError,
        match="accurate rung check requires highest_applicable == selected_rung",
    ):
        route_metadata_gate(gate, {}, max_repairs=2)


@pytest.mark.parametrize(
    ("verdict", "highest_applicable", "allowed"),
    (
        ("accurate", "R4_REIMPLEMENT", True),
        ("cannot-verify", "R2_VENDOR", False),
    ),
)
def test_highest_applicable_equality_is_accurate_verdict_sensitive(
    verdict: str, highest_applicable: str, allowed: bool
) -> None:
    """Equal accurate and unequal cannot-verify controls take their intended routes.

    Parameters
    ----------
    verdict, highest_applicable, allowed:
        Rung-check control facts and expected canonical-write decision.
    """

    gate = make_gate(["m_example"])
    gate["items"][0]["rung_check"].update(
        {
            "selected_rung": "R4_REIMPLEMENT",
            "highest_applicable": highest_applicable,
            "verdict": verdict,
        }
    )
    decision = route_metadata_gate(gate, {}, max_repairs=2)[0]
    assert decision.canonical_write_allowed is allowed


def test_cannot_verify_rung_check_does_not_silently_accept() -> None:
    """An unresolved source-ladder check fails closed into fidelity repair."""

    gate = _rung_checked_fidelity_gate(AccuracyVerdict.CANNOT_VERIFY)
    decision = route_fidelity_gate(gate, make_author_proposal())
    metadata_gate = make_gate()
    metadata_gate["items"][0]["rung_check"] = deepcopy(gate["items"][0]["rung_check"])
    metadata_decision = route_metadata_gate(metadata_gate, {}, max_repairs=2)[0]
    assert decision.accepted_for_fidelity is False
    assert decision.canonical_write_allowed is False
    assert decision.route is GateRoute.BLOCK_FIDELITY
    assert metadata_decision.canonical_write_allowed is False
    assert metadata_decision.route is GateRoute.REQUEUE_NEXT_BATCH


def test_terminal_disposition_gate_resolves_exact_advisory_references() -> None:
    """A terminal gate checks exact result/source/evidence/license facts but does not award."""

    raw_result = {
        "result_id": "result-defer",
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
        "created_at": "2026-07-16T00:00:00Z",
    }
    binding = AuthorResultBinding(raw_result=raw_result, **raw_result)
    # A deferral MUST carry its executable handoff authority: the deferred platform
    # later runs this proposal, and its licensing is what ``license_identity`` binds.
    # This fixture previously omitted it and so described a deferral the wire format
    # cannot express and the Linux deferred sweep would refuse to start on.
    handoff_proposal = deepcopy(make_author_proposal())
    handoff_proposal["proposal_id"] = "proposal-defer-1"
    result = DeferRecommendation(
        binding=binding,
        platform="cuda",
        source_ids=("source-1",),
        evidence_ids=("evidence-1",),
        evidence_identity=HASH,
        license_identity=HASH,
        recommendation_sha256=HASH,
        handoff_execution=HandoffExecution(
            proposal=handoff_proposal,
            proposal_sha256=HASH,
            code_manifest_identity=HASH,
            source_manifest_identity=HASH,
            handoff_sha256=HASH,
        ),
    )
    gate = make_gate(["m_example"])
    gate.update(
        {
            "schema_version": GATE_SCHEMA_VERSION_V3,
            "gate_kind": "terminal_disposition",
            "batch_size": 1,
            "author_result_schema_identity": HASH,
            "dispatcher_identity": HASH,
        }
    )
    gate["items"][0]["terminal_disposition"] = {
        "author_result_id": "result-defer",
        "author_result_sha256": HASH,
        "kind": "DEFER_RECOMMENDATION",
        "predicate": "needs-cuda",
        "handoff_proposal_id": "proposal-defer-1",
        "handoff_sha256": HASH,
        "verdict": "accepted",
        "source_manifest_identity": HASH,
        "source_ids": ["source-1"],
        "evidence_identity": HASH,
        "evidence_ids": ["evidence-1"],
        "license_identity": HASH,
        "findings": [],
    }
    source_manifest = {
        "manifest_sha256": HASH,
        "sources": [{"source_id": "source-1"}],
    }
    evidence_pack = {
        "evidence_identity": HASH,
        "excerpts": [
            {
                "evidence_id": "evidence-1",
                "source_id": "source-1",
                "supports": ["needs-cuda"],
            }
        ],
    }
    decision = validate_terminal_disposition_gate(
        gate,
        result,
        source_manifest=source_manifest,
        evidence_pack=evidence_pack,
        license_identity=HASH,
    )
    assert decision.accepted is True
    assert decision.predicate == "needs-cuda"

    gate["items"][0]["terminal_disposition"]["source_ids"] = ["source-fabricated"]
    with pytest.raises(GateRoutingError, match="source IDs"):
        validate_terminal_disposition_gate(
            gate,
            result,
            source_manifest=source_manifest,
            evidence_pack=evidence_pack,
            license_identity=HASH,
        )
    gate["items"][0]["terminal_disposition"]["source_ids"] = ["source-1"]

    # Independently of the checker-dispatch decision rule, the routing gate refuses an
    # acceptance over degraded integrity. Every reference above stays exact, so only the
    # integrity clause can decide these two probes.
    for degraded in ("inaccurate", "cannot-verify"):
        gate["items"][0]["integrity"]["verdict"] = degraded
        with pytest.raises(GateRoutingError, match="accurate item integrity"):
            validate_terminal_disposition_gate(
                gate,
                result,
                source_manifest=source_manifest,
                evidence_pack=evidence_pack,
                license_identity=HASH,
            )


#: The exact ``terminal_disposition`` key set the schema admits. Spelled literally rather
#: than read back out of the schema, so a widening of the schema block cannot silently
#: widen the assertion with it.
_TERMINAL_DISPOSITION_KEYS = (
    "author_result_id",
    "author_result_sha256",
    "kind",
    "predicate",
    "handoff_proposal_id",
    "handoff_sha256",
    "verdict",
    "source_manifest_identity",
    "source_ids",
    "evidence_identity",
    "evidence_ids",
    "license_identity",
    "findings",
)


def _terminal_gate(disposition: Mapping[str, Any]) -> dict[str, Any]:
    """Build a schema-complete terminal gate carrying ``disposition``.

    Parameters
    ----------
    disposition:
        Candidate ``terminal_disposition`` block under test.

    Returns
    -------
    dict[str, Any]
        Single-item ``terminal_disposition`` gate.v3 payload.
    """

    gate = make_gate(["m_example"])
    gate.update(
        {
            "schema_version": GATE_SCHEMA_VERSION_V3,
            "gate_kind": "terminal_disposition",
            "batch_size": 1,
            "author_result_schema_identity": HASH,
            "dispatcher_identity": HASH,
        }
    )
    gate["items"][0]["terminal_disposition"] = dict(disposition)
    return gate


def _compliant_disposition() -> dict[str, Any]:
    """Return a terminal disposition carrying exactly the admitted key set.

    Returns
    -------
    dict[str, Any]
        Schema-valid ``terminal_disposition`` block.
    """

    return {
        "author_result_id": "result-blocked",
        "author_result_sha256": HASH,
        "kind": "BLOCKED",
        "predicate": "blocked-prerequisite",
        "handoff_proposal_id": None,
        "handoff_sha256": None,
        "verdict": "rejected",
        "source_manifest_identity": HASH,
        "source_ids": ["source-1"],
        "evidence_identity": HASH,
        "evidence_ids": ["evidence-1"],
        "license_identity": HASH,
        "findings": ["evidence identity does not bind the declared excerpts"],
    }


def test_terminal_disposition_block_admits_exactly_its_declared_keys() -> None:
    """The closed terminal block refuses author-result vocabulary, and admits its own.

    A 10-model pilot rung reached terminal for the first time and lost every verdict
    here: the checker composed ``terminal_disposition`` out of the *author-result's*
    names for the same facts -- ``arm`` for ``kind``, ``result_sha256`` for
    ``author_result_sha256``, a ``reason`` string for the ``findings`` array, plus
    ``recommendation_sha256`` -- and the closed block refused all nine. That refusal is
    the contract working, so this pins BOTH directions: the invented spellings stay
    refused, and a disposition carrying exactly the declared thirteen still passes. A
    guard exercised only where it passes proves nothing, so the negative case asserts on
    the specific unexpected keys rather than on rejection alone.
    """

    compliant = _compliant_disposition()
    assert set(compliant) == set(_TERMINAL_DISPOSITION_KEYS)

    # The passing direction: without it, a schema typo would satisfy the negative case.
    validate_payload(_terminal_gate(compliant), GATE_SCHEMA_VERSION_V3)

    # The failing direction, one invented key at a time, so no single rejection can
    # stand in for the rest and mask a key the block has quietly started admitting.
    for invented, value in (
        ("arm", "BLOCKED"),
        ("result_sha256", HASH),
        ("recommendation_sha256", HASH),
        ("reason", "the recommendation is not validly grounded"),
        ("checked_source_ids", ["source-1"]),
        ("checked_evidence_ids", ["evidence-1"]),
        ("prerequisite_ids", ["prereq-1"]),
    ):
        polluted = dict(compliant, **{invented: value})
        with pytest.raises(PayloadValidationError) as excinfo:
            validate_payload(_terminal_gate(polluted), GATE_SCHEMA_VERSION_V3)
        message = str(excinfo.value)
        assert "additionalProperties" in message, f"{invented} refused for the wrong reason"
        assert f"'{invented}'" in message, f"{invented} was admitted or refused unnamed"

    # The exact shape one pilot model emitted: author-result spellings throughout.
    pilot_shape = {
        "arm": "BLOCKED",
        "verdict": "rejected",
        "result_sha256": HASH,
        "recommendation_sha256": HASH,
        "source_manifest_identity": HASH,
        "evidence_identity": HASH,
        "license_identity": HASH,
        "reason": "the required BLOCKED proof is not validly grounded",
    }
    with pytest.raises(PayloadValidationError) as excinfo:
        validate_payload(_terminal_gate(pilot_shape), GATE_SCHEMA_VERSION_V3)
    assert "Additional properties are not allowed" in str(excinfo.value)


def test_reducer_refuses_run_award_with_inaccurate_rung_check(tmp_path: Path) -> None:
    """The canonical writer rejects a run governed by an inaccurate rung check.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    stable_ids = ["m_example", *(f"m_{index}" for index in range(9))]
    metadata_gate = make_gate(stable_ids)
    fidelity_gate = _rung_checked_fidelity_gate(AccuracyVerdict.INACCURATE)
    fidelity_gate["gate_id"] = "gate-fidelity"
    fidelity_gate["ledger_seq"] = 2
    # Keep the proof envelope authentic so this regression reaches the intended
    # semantic rung-check rejection rather than failing at the proof loader.
    fidelity_gate["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in fidelity_gate.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    model = make_model(accepted=True)
    model["source_resolution"]["rung"] = "R4_REIMPLEMENT"
    model["fidelity"].update(
        {
            "required": True,
            "reason": "independent fidelity required",
            "verdict": "match",
            "fidelity_identity": HASH,
            "gate_id": "gate-fidelity",
        }
    )

    with CanonicalReducer(_ledger_paths(tmp_path), make_authority_context(stable_ids)) as reducer:
        reducer.append_attempt(make_attempt())
        reducer.append_gate(metadata_gate)
        reducer.append_gate(fidelity_gate)
        with pytest.raises(ReductionError, match="rung check"):
            reducer.append_model(reducer.prepare_model(model))


def _stamped_metadata_gate(tmp_path: Path, stable_id: str, *, nonce: str) -> dict[str, Any]:
    """Build one gate exactly as the production stamp produces it.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    stable_id:
        Sole model in the batch, which also makes the gate distinct.
    nonce:
        Fresh request nonce, which seeds a distinct ``gate_id``.

    Returns
    -------
    dict[str, Any]
        Stamped, not-yet-appended gate.
    """

    gate = make_gate([stable_id])
    envelope = build_metadata_vet_envelope(
        [_checker_item_pack(gate["items"][0])],
        gate_round=1,
        output_path=tmp_path / nonce / "result.json",
        checker_model="codex",
        checker_version="test",
        request_nonce=nonce,
        final_tail=True,
    )
    return _stamped(gate, envelope)


def _pre_fix_stamp(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the pre-fix stamp verbatim on one stamped gate.

    Before 2026-07-30 the machine-owned scaffold carried
    ``ledger_seq = PLACEHOLDER_LEDGER_SEQ`` (the constant ``1``) and a zeroed
    ``payload_sha256``, and ``compute_result_envelope_sha256`` excluded only the
    result and payload hashes -- so ``ledger_seq`` was INSIDE the digest.

    Parameters
    ----------
    gate:
        Correctly stamped gate carrying no ledger-assigned fields.

    Returns
    -------
    dict[str, Any]
        The same gate as the pre-fix code would have produced it.
    """

    legacy = dict(gate)
    legacy["ledger_seq"] = 1
    legacy["payload_sha256"] = "sha256:" + "0" * 64
    legacy["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in legacy.items()
            if key not in {"result_envelope_sha256", "payload_sha256"}
        }
    )
    return legacy


def test_two_gates_in_one_run_get_advancing_ledger_assigned_sequences(tmp_path: Path) -> None:
    """A SECOND gate appends, and the ledger -- not the stamp -- numbers both.

    This is the case no test covered. Every prior gate regression appended at
    most ONE gate through the production stamp, and a stamped constant
    ``ledger_seq = 1`` is indistinguishable from a correct assignment at sequence
    one. The defect therefore survived five rungs and killed the first campaign
    that ever recorded a second gate (2026-07-30: ``LedgerConflictError:
    ledger_seq must be next local sequence 2``). A single-gate assertion cannot
    detect it, so this test appends two.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    stable_ids = ["m_seq_first", "m_seq_second"]
    first = _stamped_metadata_gate(tmp_path, "m_seq_first", nonce="ledger-seq-first")
    second = _stamped_metadata_gate(tmp_path, "m_seq_second", nonce="ledger-seq-second")

    # The two gates must be genuinely distinct, or the ledger's idempotent-replay
    # path -- which returns the existing record WITHOUT appending -- would be what
    # this test exercised, and a frozen sequence would read as a pass.
    assert first["gate_id"] != second["gate_id"]
    assert first["items"][0]["stable_id"] != second["items"][0]["stable_id"]

    # The appends come BEFORE the shape assertions on purpose. Under the pre-fix
    # stamp this test must fail on the SECOND APPEND -- the failure the campaign
    # actually hit -- and a "no stamped ledger_seq" assertion placed first would
    # short-circuit it into a shape complaint that never reaches the ledger.
    with CanonicalReducer(_ledger_paths(tmp_path), make_authority_context(stable_ids)) as reducer:
        first_result = reducer.append_gate(first)
        second_result = reducer.append_gate(second)

    # The stamp leaves both ledger-assigned slots empty for the ledger to fill.
    for field in LEDGER_ASSIGNED_GATE_FIELDS:
        assert field not in first
        assert field not in second

    assert first_result.appended is True
    assert second_result.appended is True
    assert first_result.record["ledger_seq"] == 1
    assert second_result.record["ledger_seq"] == 2
    assert first_result.record["payload_sha256"] != second_result.record["payload_sha256"]

    # Both persisted gates replay their own v3 proof. The self-hash is recomputed
    # by the authority with ``ledger_seq`` EXCLUDED, so a digest taken over a body
    # that contained it can never replay -- which is why the pre-fix code left
    # even its single recorded gate unreadable.
    for record in (first_result.record, second_result.record):
        assert load_current_gate_proof(record)["gate_id"] == record["gate_id"]


def test_the_pre_fix_constant_ledger_seq_stamp_fails_this_regression(tmp_path: Path) -> None:
    """The removed behaviour is proven to fail, in both directions it failed.

    A guard shown only where it passes proves nothing. Restoring the exact pre-fix
    stamp must reproduce BOTH defects: the second append conflicts with the
    ledger's own next sequence, and the first gate -- which appended cleanly,
    because a constant ``1`` IS the next sequence exactly once -- cannot replay
    its own result-envelope self-hash.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    stable_ids = ["m_seq_first", "m_seq_second"]
    first = _pre_fix_stamp(_stamped_metadata_gate(tmp_path, "m_seq_first", nonce="legacy-first"))
    second = _pre_fix_stamp(
        _stamped_metadata_gate(tmp_path, "m_seq_second", nonce="legacy-second")
    )

    assert first["ledger_seq"] == 1
    assert second["ledger_seq"] == 1

    with CanonicalReducer(_ledger_paths(tmp_path), make_authority_context(stable_ids)) as reducer:
        persisted = reducer.append_gate(first).record
        # Defect one: the constant was right by accident exactly once.
        assert persisted["ledger_seq"] == 1
        with pytest.raises(LedgerConflictError) as conflict:
            reducer.append_gate(second)

    assert "ledger_seq must be next local sequence 2" in str(conflict.value)

    # Defect two, independent of the first: the digest was taken over a body that
    # contained ``ledger_seq``, so the gate that DID append is still unreadable.
    with pytest.raises(AuthorityDerivationError) as proof:
        load_current_gate_proof(persisted)
    assert "result-envelope self-hash is invalid" in str(proof.value)

    # The correctly stamped form of the SAME gate replays cleanly, so the failure
    # above is attributable to the pre-fix stamp and not to the fixture.
    with CanonicalReducer(
        _ledger_paths(tmp_path / "control"), make_authority_context(stable_ids)
    ) as reducer:
        control = reducer.append_gate(
            _stamped_metadata_gate(tmp_path, "m_seq_first", nonce="legacy-first")
        ).record
    assert load_current_gate_proof(control)["gate_id"] == control["gate_id"]


# The complete (integrity, disposition) product for a terminal-disposition item.
#
# The top-level verdict is pinned to the disposition by TERMINAL_VERDICT_LOCKSTEP, so
# integrity is the only free variable and the product is exactly nine cells.
# ``old_admits`` is the retired ``integrity == verdict`` equality; ``new_admits`` is the
# monotone ``severity(verdict) >= severity(integrity)`` rule.
#
# integrity, disposition, old_admits, new_admits
_TERMINAL_VERDICT_CELLS = (
    ("accurate", "accepted", True, True),
    ("cannot-verify", "accepted", False, False),
    ("inaccurate", "accepted", False, False),
    ("accurate", "cannot-verify", False, True),
    ("cannot-verify", "cannot-verify", True, True),
    ("inaccurate", "cannot-verify", False, False),
    ("accurate", "rejected", False, True),
    ("cannot-verify", "rejected", False, True),
    ("inaccurate", "rejected", True, True),
)


def _terminal_decision_item(
    integrity: str, disposition: str, *, verdict: str | None = None
) -> dict[str, Any]:
    """Build one terminal-disposition gate item at an exact verdict triple.

    Parameters
    ----------
    integrity:
        Integrity verdict to declare.
    disposition:
        Terminal disposition verdict.
    verdict:
        Top-level verdict; defaults to the disposition's locked counterpart.

    Returns
    -------
    dict[str, Any]
        Complete gate item.
    """

    item = deepcopy(make_gate(["m_terminal"])["items"][0])
    item["integrity"]["verdict"] = integrity
    item["verdict"] = verdict or TERMINAL_VERDICT_LOCKSTEP[disposition].value
    item["terminal_disposition"] = {
        "kind": "BLOCKED",
        "predicate": "blocked-prerequisite",
        "verdict": disposition,
        "author_result_id": "result-1",
        "author_result_sha256": HASH,
        "handoff_proposal_id": None,
        "handoff_sha256": None,
        "source_manifest_identity": HASH,
        "evidence_identity": HASH,
        "license_identity": HASH,
        "source_ids": ["source-1"],
        "evidence_ids": ["evidence-1"],
        "findings": [],
    }
    return item


def test_terminal_verdict_cells_are_the_complete_product() -> None:
    """The enumeration is exhaustive over the closed verdict and disposition sets."""

    verdicts = {member.value for member in AccuracyVerdict}
    dispositions = set(TERMINAL_VERDICT_LOCKSTEP)
    assert len(verdicts) == 3 and len(dispositions) == 3
    tabulated = {
        (integrity, disposition) for integrity, disposition, _, _ in _TERMINAL_VERDICT_CELLS
    }
    assert tabulated == {
        (integrity, disposition) for integrity in verdicts for disposition in dispositions
    }


def test_terminal_verdict_cells_match_the_two_stated_rules() -> None:
    """Each tabulated admissibility column is the rule it claims to be.

    The table carries the safety argument, so it is checked against the rules rather than
    trusted.
    """

    for integrity, disposition, old_admits, new_admits in _TERMINAL_VERDICT_CELLS:
        verdict = TERMINAL_VERDICT_LOCKSTEP[disposition]
        assert old_admits is (integrity == verdict.value)
        assert new_admits is (
            VERDICT_SEVERITY[verdict] >= VERDICT_SEVERITY[AccuracyVerdict(integrity)]
        )


def test_terminal_relaxation_never_admits_a_more_lenient_verdict() -> None:
    """Every newly admitted cell judges the item strictly above its integrity findings.

    This is the safety property the relaxation must preserve: no item may be accepted, or
    judged less severely than its own integrity findings warrant. A cell the old rule
    refused and the new rule admits is only ever one where the verdict is *more* severe
    than integrity.
    """

    newly_admitted = [
        (integrity, disposition)
        for integrity, disposition, old_admits, new_admits in _TERMINAL_VERDICT_CELLS
        if new_admits and not old_admits
    ]
    assert newly_admitted == [
        ("accurate", "cannot-verify"),
        ("accurate", "rejected"),
        ("cannot-verify", "rejected"),
    ]
    for integrity, disposition in newly_admitted:
        verdict = TERMINAL_VERDICT_LOCKSTEP[disposition]
        assert VERDICT_SEVERITY[verdict] > VERDICT_SEVERITY[AccuracyVerdict(integrity)]
    # Nothing the old rule admitted is now refused: the change only ever widens.
    assert not [cell for cell in _TERMINAL_VERDICT_CELLS if cell[2] and not cell[3]]
    # Acceptance over degraded integrity stays refused under the new rule.
    assert [
        (integrity, disposition)
        for integrity, disposition, _, new_admits in _TERMINAL_VERDICT_CELLS
        if disposition == "accepted" and new_admits
    ] == [("accurate", "accepted")]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("integrity", "disposition", "old_admits", "new_admits"), _TERMINAL_VERDICT_CELLS
)
def test_terminal_verdict_cell_admissibility_matches_the_production_check(
    integrity: str, disposition: str, old_admits: bool, new_admits: bool
) -> None:
    """The real production check admits exactly the tabulated ``new_admits`` cells.

    Parameters
    ----------
    integrity, disposition:
        Verdict cell under test.
    old_admits, new_admits:
        Tabulated admissibility of the retired equality and the shipped monotone rule.
        ``old_admits`` is documentation here; that it really is the old rule is proven by
        ``test_terminal_verdict_cells_match_the_two_stated_rules``.
    """

    item = _terminal_decision_item(integrity, disposition)
    if new_admits:
        _validate_item_decision(item, GateKind.TERMINAL_DISPOSITION)
        return
    with pytest.raises(CheckerDispatchError) as caught:
        _validate_item_decision(item, GateKind.TERMINAL_DISPOSITION)
    # Exact equality, not a substring: an earlier clause refusing this item for an
    # unrelated reason would otherwise read as a pass.
    assert str(caught.value) == "terminal verdict is less severe than its own integrity verdict"


@pytest.mark.smoke
@pytest.mark.parametrize("disposition", sorted(TERMINAL_VERDICT_LOCKSTEP))
def test_terminal_top_level_verdict_lockstep_still_binds(disposition: str) -> None:
    """The retained clause refuses any top-level verdict other than the locked one.

    Parameters
    ----------
    disposition:
        Terminal disposition under test.
    """

    expected = TERMINAL_VERDICT_LOCKSTEP[disposition]
    for member in AccuracyVerdict:
        if member is expected:
            continue
        # Integrity is set to the offered verdict so the monotone clause is satisfied and
        # only the lockstep clause can decide this probe.
        item = _terminal_decision_item(member.value, disposition, verdict=member.value)
        with pytest.raises(CheckerDispatchError) as caught:
            _validate_item_decision(item, GateKind.TERMINAL_DISPOSITION)
        assert str(caught.value) == "terminal top-level verdict contradicts the disposition"
