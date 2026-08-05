"""Pre-publication gate: wall-claim honesty and the one bounded repair round.

Rung-8 evidence (2026-08-05, intake_head20 @ 19a72e4a) drives every case here:

* Eight of twenty sessions published ``BLOCKED/wall-exceeded`` with
  ``timed_out=false`` and 63-88% of the 30-minute grant UNUSED, and the machine
  recorded each unverified self-report as a trusted effort terminal.
* Every malformed result surfaced exactly ONE validation error per burned
  attempt, with two to four latent violations behind it (the W-8 amplifier).

The gate refuses the first class against the executor's own clock and repairs
the second in one bounded in-session round. Both bounds live inside the
attempt's existing wall grant: the census revoked the session-budget increase,
so these tests also pin that no continuation can outspend the grant.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from menagerie.crawler.author_attempts import latest_attempt, new_attempt
from menagerie.crawler.author_executor import (
    EXIT_OK,
    EXIT_RETRYABLE,
    ExecutorConfig,
    WALL_CLAIM_MIN_OBSERVED_FRACTION,
    _author_result_from_author_payload,
    _prepublication_validation_available,
    _settle_result_prepublication,
    _unverified_wall_claim,
    AuthorExecutorError,
    main,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.tests.conftest import make_author_proposal
from menagerie.crawler.tests.executor_test_support import (
    AUTHOR_IDENTITY_INPUTS,
    DEFAULT_RESULT,
    executor_environment,
    read_invocations,
    write_author_envelope,
    write_broker_fixtures,
    write_fake_claude,
    write_source_request,
)

WALL_CLAIM_RESULT = {
    "kind": "BLOCKED",
    "payload": {
        "stage": "author",
        "reason_code": "wall-exceeded",
        "prerequisite_ids": ["authoring-wall-budget"],
        "evidence_ids": [],
    },
}


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One configured executor rig, exactly like the executor suite's."""

    fake = write_fake_claude(tmp_path / "bin")
    fixtures = write_broker_fixtures(tmp_path / "fixtures")
    log_dir = tmp_path / "log"
    executor_environment(
        monkeypatch, fake_claude=fake, fixtures=fixtures, log_dir=log_dir
    )
    return {
        "root": tmp_path / "work" / "m1" / "author",
        "log": log_dir,
        "monkeypatch": monkeypatch,
        "tmp": tmp_path,
    }


def _author_round(rig, stable_id: str = "m1") -> int:
    """Run the stage-1 round then one author round through the real CLI."""

    root = rig["root"]
    assert main([str(write_source_request(root, stable_id))]) == EXIT_OK
    return main([str(write_author_envelope(root, stable_id))])


def _stages(rig) -> list[str]:
    return [entry["stage"] for entry in read_invocations(rig["log"])]


# -- wall-claim honesty (census S2) -----------------------------------------


def test_an_unverified_wall_claim_is_refused_and_the_session_continues(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rung-8 bail shape converts: refusal round -> real result -> publish.

    The session publishes ``wall-exceeded`` seconds into a 30-minute grant.
    The executor's clock refutes the claim, the still-live session is resumed
    with the machine's own numbers, and its continued work publishes normally.
    """

    monkeypatch.setenv("FAKE_CLAUDE_RESULT", json.dumps(WALL_CLAIM_RESULT))
    monkeypatch.setenv("FAKE_CLAUDE_CONTINUED_RESULT", json.dumps(DEFAULT_RESULT))

    assert _author_round(rig) == EXIT_OK

    published = json.loads((rig["root"] / "result.json").read_text(encoding="utf-8"))
    assert published["payload"]["reason_code"] == "missing-material-source", (
        "the published result must be the CONTINUED work, not the refused claim"
    )
    stages = _stages(rig)
    assert "wall-refusal" in stages, "the refusal round must resume the session"
    refusal_prompt = next(
        entry["prompt"]
        for entry in read_invocations(rig["log"])
        if entry["stage"] == "wall-refusal"
    )
    assert "WALL CLAIM REFUSED" in refusal_prompt
    assert "wall grant consumed" in refusal_prompt, (
        "the refusal must cite the machine's own observation, not scold in prose"
    )
    assert "--clock --deadline" in refusal_prompt, (
        "the refusal must hand the session the granted clock invocation"
    )
    attempt = latest_attempt(rig["root"])
    assert attempt is not None
    rounds = attempt.record.get("continuations")
    assert rounds and rounds[0]["kind"] == "wall-claim-refusal"


def test_an_insisting_unverified_wall_claim_fails_typed_and_publishes_nothing(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeated refuted claim is REFUSED, never laundered into a terminal.

    Census tripwire finding 2: the driver used to record the session's
    self-report as ``effort-exhausted:wall-seconds`` -- a trusted, requeueable
    budget terminal -- while its own record said ``timed_out=false`` with 88%
    of the grant unused. Now nothing is published and the attempt fails with
    the machine's numbers on the record.
    """

    monkeypatch.setenv("FAKE_CLAUDE_RESULT", json.dumps(WALL_CLAIM_RESULT))
    monkeypatch.delenv("FAKE_CLAUDE_CONTINUED_RESULT", raising=False)

    assert _author_round(rig) == EXIT_RETRYABLE

    assert not (rig["root"] / "result.json").is_file(), (
        "a refused wall claim must never be published"
    )
    attempt = latest_attempt(rig["root"])
    assert attempt is not None
    outcome = attempt.record["outcome"]
    assert outcome["failure_reason"] == "unverified-wall-claim"
    detail = outcome["detail"]
    assert detail["reason_code"] == "wall-exceeded"
    assert detail["wall_seconds_grant"] > 0
    assert detail["observed_fraction"] < WALL_CLAIM_MIN_OBSERVED_FRACTION, (
        "the refusal must carry the machine observation that justifies it"
    )


def test_a_corroborated_wall_claim_publishes_without_any_continuation(
    tmp_path: Path,
) -> None:
    """An HONEST exhaustion claim is never refused -- the tripwire direction.

    ``spent`` at or above the corroboration floor publishes exactly as before;
    the config's harness command is a nonexistent binary, so any attempted
    continuation would crash this test loudly.
    """

    root = tmp_path / "author"
    request_path = write_author_envelope(root, "m1")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    attempt = new_attempt(root, stable_id="m1", campaign_id="c1-mech", kind="author")
    result_path = attempt.paths.directory / "result.json"
    result_path.write_text(json.dumps(WALL_CLAIM_RESULT), encoding="utf-8")
    config = ExecutorConfig(
        claude_command=("/nonexistent-claude-binary",),
        author_model=None,
        campaign_id="c1-mech",
        wall_seconds_override=None,
        pause_after=None,
    )
    grant = config.wall_seconds()

    materialized, typed_exit = _settle_result_prepublication(
        attempt,
        request,
        config,
        author_root=root,
        result_path=result_path,
        read_roots=[root],
        spent=WALL_CLAIM_MIN_OBSERVED_FRACTION * grant,
        resume_session=None,
    )

    assert typed_exit is None
    assert materialized is not None
    assert materialized["payload"]["reason_code"] == "wall-exceeded"


def test_the_wall_claim_check_speaks_the_lanes_own_claim_grammar() -> None:
    """Every reserved and grammatical exhaustion spelling is caught; no more.

    The refusal reuses ``_names_effort_exhaustion``, so the rung-7 free-form
    spellings and the reserved family all resolve identically, and a genuine
    prerequisite claim can never transmute into a wall refusal.
    """

    grant = 1800.0

    def claim_for(reason_code: str, *, spent: float = 200.0):
        result = {
            "kind": "BLOCKED",
            "payload": {"reason_code": reason_code},
        }
        return _unverified_wall_claim(result, spent=spent, grant=grant)

    for spelling in (
        "wall-exceeded",
        "effort-exhausted:wall-seconds",
        "authoring-budget-exhausted",
        "author-wall-deadline-before-citation-grounding",
    ):
        refusal = claim_for(spelling)
        assert refusal is not None, spelling
        assert refusal["reason_code"] == spelling
        assert refusal["wall_seconds_observed"] == 200.0

    assert claim_for("missing-material-source") is None
    assert claim_for("needs-source-access") is None
    # Corroborated: at or past the floor the claim publishes untouched.
    assert claim_for("wall-exceeded", spent=0.5 * grant) is None
    assert claim_for("wall-exceeded", spent=grant) is None
    # Only BLOCKED carries a reason_code at all.
    proposed = {"kind": "PROPOSED", "payload": {}}
    assert _unverified_wall_claim(proposed, spent=1.0, grant=grant) is None


# -- the one bounded repair round (census W-8) -------------------------------


BROKEN_RESULT = {
    "kind": "BLOCKED",
    # Three schema violations at once: reason_code, prerequisite_ids, and
    # evidence_ids are all absent. The old first-error-only path would have
    # cost one attempt PER violation.
    "payload": {"stage": "author"},
}


def test_materialization_failure_gets_one_full_enumeration_repair_round(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All violations ride ONE repair brief, and the repaired result publishes."""

    monkeypatch.setenv("FAKE_CLAUDE_RESULT", json.dumps(BROKEN_RESULT))
    monkeypatch.setenv("FAKE_CLAUDE_REPAIRED_RESULT", json.dumps(DEFAULT_RESULT))

    assert _author_round(rig) == EXIT_OK

    stages = _stages(rig)
    assert stages.count("repair") == 1, "exactly one bounded repair round"
    repair_prompt = next(
        entry["prompt"]
        for entry in read_invocations(rig["log"])
        if entry["stage"] == "repair"
    )
    assert "COMPLETE enumerated" in repair_prompt
    assert "1. " in repair_prompt and "2. " in repair_prompt, (
        "the brief must enumerate MULTIPLE violations at once (W-8)"
    )
    assert "reason_code" in repair_prompt
    assert "prerequisite_ids" in repair_prompt
    published = json.loads((rig["root"] / "result.json").read_text(encoding="utf-8"))
    assert published["payload"]["reason_code"] == "missing-material-source"
    attempt = latest_attempt(rig["root"])
    assert attempt is not None
    rounds = attempt.record.get("continuations")
    assert rounds and rounds[0]["kind"] == "validation-repair"


def test_a_repair_round_that_does_not_repair_fails_typed_with_every_error(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The round is bounded: a second identical failure ends the attempt."""

    monkeypatch.setenv("FAKE_CLAUDE_RESULT", json.dumps(BROKEN_RESULT))
    monkeypatch.delenv("FAKE_CLAUDE_REPAIRED_RESULT", raising=False)

    assert _author_round(rig) == EXIT_RETRYABLE

    assert _stages(rig).count("repair") == 1, "never a second repair round"
    attempt = latest_attempt(rig["root"])
    assert attempt is not None
    outcome = attempt.record["outcome"]
    assert outcome["failure_reason"] == "result-contract-invalid"
    detail = outcome["detail"]
    assert detail["repair_round"] == "did-not-repair"
    assert len(detail["errors"]) >= 2, (
        "the durable record must carry the FULL enumeration for the retry ladder"
    )


def test_schema_refusals_enumerate_every_violation_at_once() -> None:
    """The refusal still refuses, and its ``errors`` list is the W-8 killer."""

    request = {
        "identity_inputs": deepcopy(AUTHOR_IDENTITY_INPUTS),
        "expected_result": {
            "schema_version": "menagerie.crawler.author-result.v4",
            "stable_id": "m1",
            "work_id": "work-m1",
            "campaign_id": "c1-mech",
            "author_identity": "sha256:" + "3" * 64,
            "prompt_identity": "sha256:" + "4" * 64,
            "dispatcher_identity": "sha256:" + "5" * 64,
            "source_manifest_identity": "sha256:" + "6" * 64,
            "intake_snapshot_id": "intake-test",
            "intake_snapshot_sha256": "sha256:" + "7" * 64,
            "intake_item_sha256": "sha256:" + "8" * 64,
        },
        "source_manifest": {"sources": [{"source_id": "impl-net"}]},
    }
    with pytest.raises(AuthorExecutorError) as raised:
        _author_result_from_author_payload(deepcopy(BROKEN_RESULT), request)
    errors = raised.value.errors
    assert len(errors) >= 2
    joined = "\n".join(errors)
    assert "reason_code" in joined
    assert "prerequisite_ids" in joined


# -- deep validation replay (production validator, verified envelopes only) --


def _verified_envelope(root: Path, stable_id: str) -> Path:
    """Write an author envelope whose self-hash and author binding are real."""

    path = write_author_envelope(root, stable_id)
    request = json.loads(path.read_text(encoding="utf-8"))
    author = AUTHOR_IDENTITY_INPUTS["author"]
    request["expected_result"]["author_identity"] = stable_hash(dict(author))
    request["expected_result"]["prompt_identity"] = author["prompt_sha256"]
    request.pop("envelope_sha256", None)
    request["envelope_sha256"] = stable_hash(request)
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def test_deep_validation_requires_a_verified_envelope(tmp_path: Path) -> None:
    """The production replay runs only against a self-hash-verified envelope."""

    root = tmp_path / "author"
    stub = json.loads(write_author_envelope(root, "m1").read_text(encoding="utf-8"))
    assert not _prepublication_validation_available(stub)
    verified = json.loads(
        _verified_envelope(root, "m1").read_text(encoding="utf-8")
    )
    assert _prepublication_validation_available(verified)
    tampered = {**verified, "campaign_id": "c9-doctored"}
    assert not _prepublication_validation_available(tampered)


def test_a_verified_envelope_replays_the_production_validator_before_publication(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deep defects reach the live session; publication is never blocked.

    The proposal is schema-valid but carries identities that do not follow
    from its own facts under THIS envelope's checker binding, and a
    ``verified_hashes.source_manifest`` copied from somewhere else -- the m538
    shape. The gate feeds the enumerated defects to the session; when the
    session fails to repair them, the result still publishes and the driver's
    own validation stays the authority.
    """

    proposal = make_author_proposal("m1")
    for machine_owned in (
        "schema_version",
        "campaign_id",
        "stable_id",
        "work_id",
        "intake_snapshot_id",
        "intake_snapshot_sha256",
        "intake_item_sha256",
        "source_manifest_identity",
        "dispatcher_identity",
        "author",
        "proposal_sha256",
    ):
        proposal.pop(machine_owned, None)
    proposal["proposed_facts"]["modes"]["per_mode_run"] = {}
    authored = {"kind": "PROPOSED", "payload": {"proposal": proposal}}
    monkeypatch.setenv("FAKE_CLAUDE_RESULT", json.dumps(authored))
    monkeypatch.setenv("FAKE_CLAUDE_REPAIRED_RESULT", json.dumps(authored))

    root = rig["root"]
    assert main([str(write_source_request(root, "m1"))]) == EXIT_OK
    assert main([str(_verified_envelope(root, "m1"))]) == EXIT_OK

    stages = _stages(rig)
    assert stages.count("repair") == 1, (
        "the deep-validation defects must reach the still-live session once"
    )
    repair_prompt = next(
        entry["prompt"]
        for entry in read_invocations(rig["log"])
        if entry["stage"] == "repair"
    )
    assert "PRE-PUBLICATION VALIDATION FAILED" in repair_prompt
    assert (
        "verified_hashes.source_manifest" in repair_prompt
        or "does not follow from the declared proposed_facts" in repair_prompt
    ), "the m538/identity mirror findings must be in the enumerated feedback"
    # An unrepaired deep failure NEVER blocks publication: the driver's own
    # validation of the published result stays authoritative.
    assert (root / "result.json").is_file()


def test_a_clean_result_publishes_with_no_continuation_rounds(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is invisible for honest, valid work -- no extra sessions."""

    root = rig["root"]
    assert main([str(write_source_request(root, "m1"))]) == EXIT_OK
    assert main([str(_verified_envelope(root, "m1"))]) == EXIT_OK

    stages = _stages(rig)
    assert "repair" not in stages
    assert "wall-refusal" not in stages
    published = json.loads((root / "result.json").read_text(encoding="utf-8"))
    assert published["kind"] == "BLOCKED"
    attempt = latest_attempt(root)
    assert attempt is not None
    assert not attempt.record.get("continuations")
