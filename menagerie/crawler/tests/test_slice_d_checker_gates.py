"""Checker dispatch and bounded gate-routing tests for crawler Slice D."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checker_dispatch import (
    CheckerDispatchError,
    build_metadata_vet_envelope,
    classify_checker_response,
    compute_result_envelope_sha256,
    validate_checker_result,
)
from menagerie.crawler.constants import (
    AccuracyVerdict,
    CheckerPauseReason,
    FidelityVerdict,
    GateRoute,
)
from menagerie.crawler.gates import (
    next_metadata_batch_ids,
    route_fidelity_gate,
    route_metadata_gate,
)
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.tests.conftest import (
    HASH,
    make_author_proposal,
    make_gate,
    make_model,
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
        "stable_id": item["stable_id"],
        "family_representative_id": item["family_representative_id"],
        "fidelity_identity": item["fidelity_identity"],
        "vet_identity": item["vet_identity"],
        "verified_hashes": deepcopy(item["verified_hashes"]),
        "proposal": {"description": "scoped test proposal"},
        "source_manifest": {"sources": []},
        "evidence": {"excerpts": []},
    }


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
    gate["gate_identity"] = envelope["envelope_sha256"]
    gate["checker"]["prompt_sha256"] = envelope["prompt"]["sha256"]
    gate["result_envelope_sha256"] = compute_result_envelope_sha256(gate)
    result_path.write_text(json.dumps(gate))
    validated = validate_checker_result(result_path, envelope)
    assert validated["batch_size"] == 10


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
    gate["gate_identity"] = envelope["envelope_sha256"]
    gate["checker"]["prompt_sha256"] = envelope["prompt"]["sha256"]
    gate["items"][0]["verified_hashes"]["evidence"] = "sha256:" + "c" * 64
    gate["result_envelope_sha256"] = compute_result_envelope_sha256(gate)
    result_path.write_text(json.dumps(gate))
    with pytest.raises(CheckerDispatchError, match="mismatched binding"):
        validate_checker_result(result_path, envelope)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("429 rate limit exceeded; retry after reset", CheckerPauseReason.RATE_LIMIT),
        ("insufficient_quota for this billing period", CheckerPauseReason.QUOTA_EXHAUSTED),
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
        "highest_applicable": "R2_VENDOR",
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

    with CanonicalReducer(_ledger_paths(tmp_path), stable_ids) as reducer:
        reducer.append_gate(metadata_gate)
        reducer.append_gate(fidelity_gate)
        with pytest.raises(ReductionError, match="rung check"):
            reducer.append_model(model)
