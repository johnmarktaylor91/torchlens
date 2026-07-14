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
from menagerie.crawler.constants import CheckerPauseReason, FidelityVerdict, GateRoute
from menagerie.crawler.gates import (
    next_metadata_batch_ids,
    route_fidelity_gate,
    route_metadata_gate,
)
from menagerie.crawler.tests.conftest import HASH, make_author_proposal, make_gate


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
