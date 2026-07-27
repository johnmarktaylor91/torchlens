"""Slice F author subagent pool, capability receipts, and operator wrappers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Optional

import pytest

from menagerie.crawler import author_pool as pool_module
from menagerie.crawler import operator_author, operator_notify
from menagerie.crawler.author_pool import (
    CAMPAIGN_AUTHOR_MODELS,
    AuthorPool,
    AuthorPoolError,
    classify_usage_limit,
    render_dispatch_brief,
)
from menagerie.crawler.author_queue import (
    AUTHOR_JOB_VERSION,
    AUTHOR_QUEUE_DIRECTORIES,
    DEFAULT_EFFORT_GRANT,
    AuthorQueueError,
    Consumption,
    EffortGrant,
    QueueJob,
    author_queue_directories,
    build_failure,
    build_receipt,
    job_paths,
    read_json,
    write_json_atomic,
)
from menagerie.crawler.capability_probe import (
    CAPABILITY_PROBE_FORMAT,
    MIN_FETCH_CONTENT_CHARS,
    REQUIRED_AUTHOR_TOOLS,
    CapabilityProbeError,
    build_probe_request,
    derive_challenge,
    validate_capability_evidence,
)
from menagerie.crawler.constants import (
    AUTHOR_MAX_FETCH_TARGETS,
    AUTHOR_MAX_TOOL_CALLS,
    AUTHOR_SESSION_WALL_SECONDS,
)
from menagerie.crawler.driver_admission import (
    _AUTHOR_JOB_VERSION,
    author_queue_directories as engine_author_queue_directories,
)
from menagerie.crawler.identity import hash_bytes, stable_hash

_REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_NONCE = "a3f1" * 12
REQUESTED_AT = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)


# -- fixtures --------------------------------------------------------------


def _write_request(root: Path, payload: dict[str, Any]) -> Path:
    """Write one request envelope and return its path.

    Parameters
    ----------
    root, payload:
        Destination directory and envelope body.

    Returns
    -------
    Path
        Written request path.
    """

    root.mkdir(parents=True, exist_ok=True)
    path = root / "request.json"
    path.write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return path


def _enqueue(
    queue_root: Path,
    *,
    kind: str = "author",
    campaign_id: str = "c1-mech",
    request: Optional[dict[str, Any]] = None,
    grant: Optional[EffortGrant] = None,
    nonce: str = "nonce-1",
) -> QueueJob:
    """Publish one pending job exactly as the lane would.

    Parameters
    ----------
    queue_root, kind, campaign_id, request, grant, nonce:
        Queue root, round-trip label, campaign, request body, grant, and attempt nonce.

    Returns
    -------
    QueueJob
        Parsed published job.
    """

    work_root = queue_root.parent / "work" / kind
    output_path = work_root / "result.json"
    body = request or {
        "envelope_version": "menagerie.crawler.author-envelope.v3",
        "work_id": "work-m1",
        "stable_id": "m1",
        "campaign_id": campaign_id,
        "required_output_path": str(output_path),
    }
    request_path = _write_request(work_root, body)
    job_id = f"{kind}-fixture"
    paths = job_paths(queue_root, job_id)
    effort = grant or DEFAULT_EFFORT_GRANT
    descriptor = {
        "envelope_version": AUTHOR_JOB_VERSION,
        "job_id": job_id,
        "attempt_nonce": nonce,
        "kind": kind,
        "stable_id": str(body.get("stable_id", "probe")),
        "work_id": str(body.get("work_id", "work-probe")),
        "author_model": CAMPAIGN_AUTHOR_MODELS.get(campaign_id),
        "campaign_id": campaign_id,
        "request_path": str(request_path.resolve()),
        "request_sha256": hash_bytes(request_path.read_bytes()),
        "required_output_path": str(Path(str(body["required_output_path"])).resolve()),
        "receipt_path": str(paths["receipt"]),
        "backoff_path": str(paths["backoff"]),
        "failure_path": str(paths["failure"]),
        "claim_path": str(paths["claim"]),
        "effort_grant": effort.to_dict(),
        "stall_timeout_seconds": 2700.0,
        "enqueued_at": "2026-07-27T12:00:00Z",
    }
    from menagerie.crawler.author_queue import ensure_author_queue

    ensure_author_queue(queue_root)
    write_json_atomic(paths["pending"], descriptor)
    return QueueJob.from_mapping(descriptor)


def _fixed_clock(instant: datetime) -> Any:
    """Return a clock returning one fixed instant.

    Parameters
    ----------
    instant:
        Instant to return.

    Returns
    -------
    Callable[[], datetime]
        Fixed clock.
    """

    return lambda: instant


def _probe_document(version: str, serial: int) -> str:
    """Return a plausible metadata document containing the demanded facts.

    Parameters
    ----------
    version, serial:
        Facts the document must literally contain.

    Returns
    -------
    str
        Document body of at least the minimum fetch length.
    """

    filler = "release history entry. " * 60
    return json.dumps(
        {
            "info": {"version": version, "summary": filler},
            "last_serial": serial,
            "urls": [{"filename": f"pkg-{version}.whl"}],
        }
    )


def _valid_evidence(
    *,
    nonce: str = PROBE_NONCE,
    observed_at: str = "2026-07-27T12:00:30Z",
    version: str = "2.31.0",
    serial: int = 28_450_112,
) -> dict[str, Any]:
    """Build capability evidence that genuinely satisfies every check.

    Parameters
    ----------
    nonce, observed_at, version, serial:
        Probe nonce, observation instant, and the cross-tool facts.

    Returns
    -------
    dict[str, Any]
        Evidence payload.
    """

    challenge = derive_challenge(nonce)
    document = _probe_document(version, serial)
    assert len(document) >= MIN_FETCH_CONTENT_CHARS
    return {
        "challenge_id": challenge.challenge_id,
        "tools": {
            "WebSearch": {
                "nonce": nonce,
                "observed_at": observed_at,
                "query": f"{challenge.package} pypi latest version",
                "reported_version": version,
                "results": [
                    {
                        "url": challenge.project_url,
                        "title": f"{challenge.package} - PyPI",
                        "excerpt": f"latest release {version}",
                    },
                    {
                        "url": f"https://libraries.io/pypi/{challenge.package}",
                        "title": f"{challenge.package} on libraries.io",
                        "excerpt": f"current version {version}",
                    },
                ],
            },
            "web_search_exa": {
                "nonce": nonce,
                "observed_at": observed_at,
                "query": f"{challenge.package} current released version",
                "reported_version": version,
                "results": [
                    {
                        "url": f"{challenge.project_url}history/",
                        "title": f"{challenge.package} release history",
                        "text": "release notes body. " * 20,
                    },
                    {
                        "url": f"https://github.com/psf/{challenge.package}",
                        "title": f"{challenge.package} on GitHub",
                        "text": "readme body",
                    },
                ],
            },
            "web_fetch_exa": {
                "nonce": nonce,
                "observed_at": observed_at,
                "url": challenge.metadata_url,
                "content": document,
                "content_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
                "reported_version": version,
                "reported_last_serial": serial,
            },
        },
    }


# -- wire protocol ---------------------------------------------------------


def test_queue_layout_matches_the_lane_contract() -> None:
    """The pool must speak exactly the layout ``QueueAuthorLane`` publishes."""

    assert AUTHOR_QUEUE_DIRECTORIES == ("pending", "claimed", "receipts", "signals")
    directories = author_queue_directories(Path("/queue"))
    assert directories["pending"] == Path("/queue/pending")
    paths = job_paths(Path("/queue"), "author-abc")
    assert paths["pending"] == Path("/queue/pending/author-abc.json")
    assert paths["claim"] == Path("/queue/claimed/author-abc.json")
    assert paths["receipt"] == Path("/queue/receipts/author-abc.json")
    assert paths["backoff"] == Path("/queue/signals/author-abc.backoff.json")
    assert paths["failure"] == Path("/queue/signals/author-abc.failure.json")


def test_queue_layout_agrees_with_the_engine_lane() -> None:
    """The merged engine and pool halves must agree byte-for-byte."""

    assert engine_author_queue_directories(Path("/q")) == author_queue_directories(Path("/q"))
    assert _AUTHOR_JOB_VERSION == AUTHOR_JOB_VERSION


def test_default_grant_agrees_with_engine_constants() -> None:
    """The wrapper's default grant must not drift from the engine's LP-13.2 ceiling."""

    assert DEFAULT_EFFORT_GRANT.tool_calls == AUTHOR_MAX_TOOL_CALLS
    assert DEFAULT_EFFORT_GRANT.fetch_targets == AUTHOR_MAX_FETCH_TARGETS
    assert DEFAULT_EFFORT_GRANT.wall_seconds == float(AUTHOR_SESSION_WALL_SECONDS)


def test_every_published_file_echoes_the_attempt_nonce(tmp_path: Path) -> None:
    """A stall triggers a retry, so an unstamped pool file could answer the wrong attempt."""

    job = _enqueue(tmp_path / "queue", nonce="nonce-A")
    pool = AuthorPool(tmp_path / "queue", owner="session-1")
    claimed = pool.claim(job.job_id)
    job.required_output_path.parent.mkdir(parents=True, exist_ok=True)
    job.required_output_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    receipt = pool.complete(claimed, tool_calls=4)
    failure = pool.fail(job, reason="x", retryable=True)
    backoff = pool.backoff(job, excerpt="usage limit reached")
    claim = read_json(job.claim_path)
    assert claim is not None
    for payload in (receipt, failure, backoff, claim):
        assert payload["job_id"] == job.job_id
        assert payload["attempt_nonce"] == "nonce-A"


def test_failure_sidecar_requires_an_explicit_retryable_boolean(tmp_path: Path) -> None:
    """The lane refuses a sidecar without one rather than guessing; so does the builder."""

    job = _enqueue(tmp_path / "queue")
    with pytest.raises(AuthorQueueError, match="explicit boolean retryable"):
        build_failure(job, reason="boom", retryable="yes")  # type: ignore[arg-type]


def test_receipt_refuses_to_declare_more_than_the_grant(tmp_path: Path) -> None:
    """An over-grant receipt is refused at the pool, not silently clamped to fit."""

    job = _enqueue(tmp_path / "queue", grant=EffortGrant(4, 2, 60.0))
    with pytest.raises(AuthorQueueError, match="tool_calls consumption 9"):
        build_receipt(job, consumption=Consumption(9, 1, 10.0))


# -- honest consumption ----------------------------------------------------


def test_wall_consumption_is_measured_not_reported(tmp_path: Path) -> None:
    """Wall time comes from the pool's own claim instant, never from the subagent."""

    job = _enqueue(tmp_path / "queue")
    start = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
    clock = {"now": start}
    pool = AuthorPool(tmp_path / "queue", owner="s", clock=lambda: clock["now"])
    claimed = pool.claim(job.job_id)
    clock["now"] = start + timedelta(seconds=93)
    job.required_output_path.parent.mkdir(parents=True, exist_ok=True)
    job.required_output_path.write_text("{}", encoding="utf-8")
    receipt = pool.complete(claimed, tool_calls=7)
    assert receipt["consumption"]["wall_seconds"] == pytest.approx(93.0)
    assert receipt["consumption"]["tool_calls"] == 7


def test_fetch_targets_are_counted_from_the_published_result(tmp_path: Path) -> None:
    """The one dimension the pool can observe exactly is never operator-declared."""

    request = {
        "envelope_version": "menagerie.crawler.author-source-request.v1",
        "work_id": "work-m1",
        "stable_id": "m1",
        "required_output_path": str(tmp_path / "work" / "source-targets.json"),
    }
    job = _enqueue(tmp_path / "queue", kind="source-request", request=request)
    pool = AuthorPool(tmp_path / "queue", owner="s")
    claimed = pool.claim(job.job_id)
    job.required_output_path.parent.mkdir(parents=True, exist_ok=True)
    job.required_output_path.write_text(
        json.dumps({"sources": [{"url": f"https://x/{i}"} for i in range(6)]}), encoding="utf-8"
    )
    receipt = pool.complete(claimed, tool_calls=3)
    assert receipt["consumption"]["fetch_targets"] == 6


def test_over_grant_completion_publishes_a_typed_failure_instead_of_a_receipt(
    tmp_path: Path,
) -> None:
    """Shrinking the number to fit would defeat the audit the lane performs."""

    job = _enqueue(tmp_path / "queue", grant=EffortGrant(30, 20, 30.0))
    start = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
    clock = {"now": start}
    pool = AuthorPool(tmp_path / "queue", owner="s", clock=lambda: clock["now"])
    claimed = pool.claim(job.job_id)
    clock["now"] = start + timedelta(seconds=400)
    job.required_output_path.parent.mkdir(parents=True, exist_ok=True)
    job.required_output_path.write_text("{}", encoding="utf-8")
    with pytest.raises(AuthorPoolError, match="exceeded its grant"):
        pool.complete(claimed, tool_calls=1)
    assert not job.receipt_path.exists()
    failure = read_json(job.failure_path)
    assert failure is not None
    assert failure["reason"] == "effort-cap-exhausted"
    assert failure["retryable"] is False


def test_completion_without_a_result_publishes_no_receipt(tmp_path: Path) -> None:
    """The receipt is the commit point, so it may not exist before the result does."""

    job = _enqueue(tmp_path / "queue")
    pool = AuthorPool(tmp_path / "queue", owner="s")
    claimed = pool.claim(job.job_id)
    with pytest.raises(AuthorPoolError, match="no result at its required output path"):
        pool.complete(claimed, tool_calls=1)
    assert not job.receipt_path.exists()


# -- leasing and typed signals --------------------------------------------


def test_a_live_lease_blocks_another_servicer(tmp_path: Path) -> None:
    """Two sessions must not dispatch the same model twice."""

    job = _enqueue(tmp_path / "queue")
    AuthorPool(tmp_path / "queue", owner="session-a").claim(job.job_id)
    other = AuthorPool(tmp_path / "queue", owner="session-b")
    with pytest.raises(AuthorPoolError, match="leased by session-a"):
        other.claim(job.job_id)
    assert other.claim(job.job_id, force=True).job.job_id == job.job_id


def test_a_lapsed_lease_returns_the_job(tmp_path: Path) -> None:
    """A dead managing session must look like a stalled queue, not a failed model."""

    job = _enqueue(tmp_path / "queue")
    start = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
    clock = {"now": start}
    pool = AuthorPool(
        tmp_path / "queue", owner="s", lease_seconds=60.0, clock=lambda: clock["now"]
    )
    pool.claim(job.job_id)
    assert pool.expired() == []
    clock["now"] = start + timedelta(seconds=120)
    assert [stale.job_id for stale in pool.expired()] == [job.job_id]


def test_usage_limit_text_becomes_a_pause_not_a_model_failure(tmp_path: Path) -> None:
    """Anthropic exhaustion is a scheduler pause with a reset, never a failed model."""

    job = _enqueue(tmp_path / "queue")
    pool = AuthorPool(tmp_path / "queue", owner="s")
    payload = pool.backoff(
        job, excerpt="Claude usage limit reached. try again at 2026-07-27 18:00 PDT."
    )
    assert payload["provider"] == "anthropic"
    assert payload["reason"] == "quota-exhausted"
    assert "2026-07-27 18:00 PDT" in str(payload["reset_at"])
    assert classify_usage_limit("everything is fine") is None


# -- dispatch briefs -------------------------------------------------------


@pytest.mark.parametrize("campaign_id", sorted(CAMPAIGN_AUTHOR_MODELS))
def test_every_campaign_has_a_bound_prompt_and_tier(tmp_path: Path, campaign_id: str) -> None:
    """Tier is a campaign property; a missing prompt must fail loudly, not default."""

    job = _enqueue(tmp_path / "queue", campaign_id=campaign_id)
    brief = render_dispatch_brief(job)
    assert campaign_id in brief
    assert str(job.required_output_path) in brief
    pool = AuthorPool(tmp_path / "queue", owner="s")
    expected = "opus" if campaign_id == "c3-classics" else "sonnet"
    assert pool.subagent_model(job) == expected


def test_an_unknown_campaign_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """Guessing a campaign would author a hard model with the wrong tier and standards."""

    job = _enqueue(tmp_path / "queue", campaign_id="c9-unknown")
    with pytest.raises(AuthorPoolError, match="unknown campaign"):
        render_dispatch_brief(job)


def test_capability_probe_brief_states_the_nonce_derived_challenge(tmp_path: Path) -> None:
    """The subagent must be told the exact unpredictable target it has to answer."""

    request = build_probe_request(
        nonce=PROBE_NONCE,
        required_output_path=str(tmp_path / "work" / "receipt.json"),
        requested_at="2026-07-27T12:00:00Z",
    )
    job = _enqueue(tmp_path / "queue", kind="capability-probe", request=request)
    brief = render_dispatch_brief(job)
    challenge = derive_challenge(PROBE_NONCE)
    assert challenge.metadata_url in brief
    assert challenge.challenge_id in brief
    assert PROBE_NONCE in brief


# -- capability probe: the proof ------------------------------------------


def test_challenge_is_unpredictable_but_reproducible_from_the_nonce() -> None:
    """An auditor with the nonce can recompute exactly which package was demanded."""

    first = derive_challenge(PROBE_NONCE)
    assert derive_challenge(PROBE_NONCE) == first
    others = {derive_challenge(f"{i:064x}").package for i in range(200)}
    assert len(others) > 5, "nonce selection must spread across the corpus"
    assert first.metadata_url == f"https://pypi.org/pypi/{first.package}/json"


def test_valid_evidence_produces_a_nonce_bound_receipt_for_every_required_tool() -> None:
    """The happy path must actually pass; a probe nothing can satisfy proves nothing."""

    receipt = validate_capability_evidence(
        nonce=PROBE_NONCE,
        evidence=_valid_evidence(),
        requested_at=REQUESTED_AT,
        now=REQUESTED_AT + timedelta(seconds=45),
    )
    assert receipt["nonce"] == PROBE_NONCE
    tools = {entry["tool"] for entry in receipt["receipts"]}
    assert tools == set(REQUIRED_AUTHOR_TOOLS)
    for entry in receipt["receipts"]:
        assert entry["nonce"] == PROBE_NONCE
        assert entry["exercised"] is True
        assert isinstance(entry["receipt"], str) and entry["receipt"].strip()


@pytest.mark.parametrize("missing", REQUIRED_AUTHOR_TOOLS)
def test_a_missing_tool_cannot_be_papered_over(missing: str) -> None:
    """Losing any one tool loses one of the three independent reports."""

    evidence = _valid_evidence()
    evidence["tools"].pop(missing)
    with pytest.raises(CapabilityProbeError, match="omits required tools"):
        validate_capability_evidence(
            nonce=PROBE_NONCE,
            evidence=evidence,
            requested_at=REQUESTED_AT,
            now=REQUESTED_AT + timedelta(seconds=45),
        )


def test_tools_that_disagree_about_the_live_fact_are_refused() -> None:
    """Cross-tool agreement on an unpredictable live value is the actual proof."""

    evidence = _valid_evidence()
    evidence["tools"]["WebSearch"]["reported_version"] = "9.9.9"
    with pytest.raises(CapabilityProbeError, match="disagree about"):
        validate_capability_evidence(
            nonce=PROBE_NONCE,
            evidence=evidence,
            requested_at=REQUESTED_AT,
            now=REQUESTED_AT + timedelta(seconds=45),
        )


def test_fetched_document_must_be_self_consistent_with_its_declared_digest() -> None:
    """A document that does not hash to its own claim was not the document received."""

    evidence = _valid_evidence()
    evidence["tools"]["web_fetch_exa"]["content"] += " tampered"
    with pytest.raises(CapabilityProbeError, match="declared sha256"):
        validate_capability_evidence(
            nonce=PROBE_NONCE,
            evidence=evidence,
            requested_at=REQUESTED_AT,
            now=REQUESTED_AT + timedelta(seconds=45),
        )


def test_fetch_must_retrieve_the_demanded_url() -> None:
    """Fetching something else answers a challenge nobody asked."""

    evidence = _valid_evidence()
    evidence["tools"]["web_fetch_exa"]["url"] = "https://pypi.org/pypi/not-demanded/json"
    with pytest.raises(CapabilityProbeError, match="not the demanded"):
        validate_capability_evidence(
            nonce=PROBE_NONCE,
            evidence=evidence,
            requested_at=REQUESTED_AT,
            now=REQUESTED_AT + timedelta(seconds=45),
        )


def test_reported_facts_must_appear_in_the_document_that_was_fetched() -> None:
    """A number recited beside a document is not a number read out of it."""

    version = "2.31.0"
    document = _probe_document(version, 28_450_112)
    evidence = _valid_evidence(version=version)
    evidence["tools"]["web_fetch_exa"]["reported_last_serial"] = 99_000_001
    assert "99000001" not in document
    with pytest.raises(CapabilityProbeError, match="last_serial absent"):
        validate_capability_evidence(
            nonce=PROBE_NONCE,
            evidence=evidence,
            requested_at=REQUESTED_AT,
            now=REQUESTED_AT + timedelta(seconds=45),
        )


def test_one_tools_output_cannot_be_reused_as_anothers() -> None:
    """Two search engines do not return byte-identical result lists."""

    evidence = _valid_evidence()
    evidence["tools"]["web_search_exa"]["results"] = [
        dict(item) for item in evidence["tools"]["WebSearch"]["results"]
    ]
    evidence["tools"]["web_search_exa"]["results"][0]["text"] = "x" * 400
    with pytest.raises(CapabilityProbeError, match="byte-identical result lists"):
        validate_capability_evidence(
            nonce=PROBE_NONCE,
            evidence=evidence,
            requested_at=REQUESTED_AT,
            now=REQUESTED_AT + timedelta(seconds=45),
        )


def test_a_snippet_sized_body_is_not_a_fetched_document() -> None:
    """Structural separation of fetch evidence from search evidence."""

    evidence = _valid_evidence()
    short = json.dumps({"info": {"version": "2.31.0"}, "last_serial": 28_450_112})
    evidence["tools"]["web_fetch_exa"]["content"] = short
    evidence["tools"]["web_fetch_exa"]["content_sha256"] = hashlib.sha256(
        short.encode("utf-8")
    ).hexdigest()
    with pytest.raises(CapabilityProbeError, match="a real metadata document"):
        validate_capability_evidence(
            nonce=PROBE_NONCE,
            evidence=evidence,
            requested_at=REQUESTED_AT,
            now=REQUESTED_AT + timedelta(seconds=45),
        )


def test_stale_evidence_outside_the_probe_window_is_refused() -> None:
    """Freshness is what stops a cached receipt from answering a new nonce."""

    evidence = _valid_evidence(observed_at="2026-07-27T09:00:00Z")
    with pytest.raises(CapabilityProbeError, match="outside the probe window"):
        validate_capability_evidence(
            nonce=PROBE_NONCE,
            evidence=evidence,
            requested_at=REQUESTED_AT,
            now=REQUESTED_AT + timedelta(seconds=45),
        )


def test_evidence_must_echo_the_probe_nonce() -> None:
    """Per-tool nonce echo binds each observation to this exact probe."""

    evidence = _valid_evidence()
    evidence["tools"]["WebSearch"]["nonce"] = "someone-elses-nonce"
    with pytest.raises(CapabilityProbeError, match="does not echo the probe nonce"):
        validate_capability_evidence(
            nonce=PROBE_NONCE,
            evidence=evidence,
            requested_at=REQUESTED_AT,
            now=REQUESTED_AT + timedelta(seconds=45),
        )


def test_unproven_capability_publishes_a_failure_and_no_receipt(tmp_path: Path) -> None:
    """A probe that cannot be proven writes nothing the doctor could accept."""

    request = build_probe_request(
        nonce=PROBE_NONCE,
        required_output_path=str(tmp_path / "work" / "receipt.json"),
        requested_at="2026-07-27T12:00:00Z",
    )
    job = _enqueue(tmp_path / "queue", kind="capability-probe", request=request)
    pool = AuthorPool(
        tmp_path / "queue", owner="s", clock=_fixed_clock(REQUESTED_AT + timedelta(seconds=30))
    )
    claimed = pool.claim(job.job_id)
    evidence = _valid_evidence()
    evidence["tools"].pop("web_fetch_exa")
    with pytest.raises(AuthorPoolError, match="unproven"):
        pool.complete_capability_probe(claimed, evidence, tool_calls=5)
    assert not job.required_output_path.exists()
    assert not job.receipt_path.exists()
    failure = read_json(job.failure_path)
    assert failure is not None and failure["reason"] == "capability-probe-unproven"


def test_proven_capability_writes_the_doctor_shaped_receipt(tmp_path: Path) -> None:
    """The receipt the doctor reads is the one the pool commits at the required path."""

    request = build_probe_request(
        nonce=PROBE_NONCE,
        required_output_path=str(tmp_path / "work" / "receipt.json"),
        requested_at="2026-07-27T12:00:00Z",
    )
    job = _enqueue(tmp_path / "queue", kind="capability-probe", request=request)
    pool = AuthorPool(
        tmp_path / "queue", owner="s", clock=_fixed_clock(REQUESTED_AT + timedelta(seconds=30))
    )
    claimed = pool.claim(job.job_id)
    pool.complete_capability_probe(claimed, _valid_evidence(), tool_calls=6)
    receipt = read_json(job.required_output_path)
    assert receipt is not None and receipt["nonce"] == PROBE_NONCE
    assert {entry["tool"] for entry in receipt["receipts"]} == set(REQUIRED_AUTHOR_TOOLS)
    assert read_json(job.receipt_path) is not None


def test_doctor_accepts_the_pool_receipt_end_to_end(tmp_path: Path) -> None:
    """The real ``SystemDoctorProbes.author_tools`` must accept this exact receipt.

    This is the check that keeps the two halves honest: the receipt is validated by the
    doctor's own parser, not by a test-local reimplementation of it.
    """

    from menagerie.crawler.doctor import DoctorConfig, SystemDoctorProbes

    config = DoctorConfig(
        repo_root=tmp_path / "repo",
        runtime_root=tmp_path / "runtime",
        target="osx-arm64",
    )
    config.repo_root.mkdir(parents=True, exist_ok=True)
    captured: dict[str, Any] = {}

    def runner(argv: Any, cwd: Path) -> subprocess.CompletedProcess[str]:
        """Stand in for the author wrapper by answering the probe from the pool."""

        request_path = Path(argv[-1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        captured["request"] = request
        queue_root = tmp_path / "queue"
        job = _enqueue(queue_root, kind="capability-probe", request=request)
        pool = AuthorPool(queue_root, owner="s")
        claimed = pool.claim(job.job_id)
        now = datetime.now(timezone.utc)
        pool.complete_capability_probe(
            claimed,
            _valid_evidence(
                nonce=request["nonce"],
                observed_at=now.isoformat().replace("+00:00", "Z"),
            ),
            tool_calls=6,
        )
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    os.environ["MENAGERIE_AUTHOR_COMMAND"] = "author-wrapper"
    try:
        probes = SystemDoctorProbes(config, command_runner=runner)
        assert probes.author_tools() == frozenset(REQUIRED_AUTHOR_TOOLS)
    finally:
        os.environ.pop("MENAGERIE_AUTHOR_COMMAND", None)
    assert captured["request"]["format"] == CAPABILITY_PROBE_FORMAT


def test_doctor_rejects_a_receipt_the_pool_never_proved(tmp_path: Path) -> None:
    """A wrapper that exits zero without a proven receipt must not satisfy the check."""

    from menagerie.crawler.doctor import DoctorConfig, SystemDoctorProbes

    config = DoctorConfig(
        repo_root=tmp_path / "repo",
        runtime_root=tmp_path / "runtime",
        target="osx-arm64",
    )
    config.repo_root.mkdir(parents=True, exist_ok=True)

    def runner(argv: Any, cwd: Path) -> subprocess.CompletedProcess[str]:
        """Exit zero and write nothing, the way a stubbed probe would."""

        return subprocess.CompletedProcess(list(argv), 0, stdout="ok", stderr="")

    os.environ["MENAGERIE_AUTHOR_COMMAND"] = "author-wrapper"
    try:
        assert SystemDoctorProbes(config, command_runner=runner).author_tools() == frozenset()
    finally:
        os.environ.pop("MENAGERIE_AUTHOR_COMMAND", None)


# -- operator wrappers -----------------------------------------------------


def test_author_wrapper_reports_a_version_receipt(capsys: pytest.CaptureFixture[str]) -> None:
    """``wrapper_versions`` needs a real ``--version`` on the configured command."""

    assert operator_author.main(["--version"]) == 0
    assert "menagerie-author-operator" in capsys.readouterr().out


def test_author_wrapper_enqueues_and_reports_the_pool_answer(tmp_path: Path) -> None:
    """The wrapper is the subprocess contract; the pool is what actually answers it."""

    queue_root = tmp_path / "queue"
    output = tmp_path / "work" / "result.json"
    request = _write_request(
        tmp_path / "work",
        {
            "envelope_version": "menagerie.crawler.author-envelope.v3",
            "work_id": "work-m1",
            "stable_id": "m1",
            "campaign_id": "c1-mech",
            "required_output_path": str(output),
        },
    )
    answered: dict[str, Any] = {}

    def fake_sleep(_seconds: float) -> None:
        """Answer the job on the first poll, as a live pool would."""

        if answered:
            return
        pool = AuthorPool(queue_root, owner="s")
        job = pool.pending()[0]
        claimed = pool.claim(job.job_id)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"kind": "PROPOSED"}), encoding="utf-8")
        pool.complete(claimed, tool_calls=5)
        answered["done"] = True

    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    code, detail = operator_author.serve_request(
        request,
        queue_root=queue_root,
        monotonic=lambda: next(ticks),
        sleep=fake_sleep,
    )
    assert code == 0, detail
    assert answered


def test_author_wrapper_maps_a_backoff_to_the_quota_exit_code(tmp_path: Path) -> None:
    """A usage limit must pause the scheduler, not burn the model."""

    queue_root = tmp_path / "queue"
    request = _write_request(
        tmp_path / "work",
        {
            "envelope_version": "menagerie.crawler.author-envelope.v3",
            "work_id": "work-m1",
            "stable_id": "m1",
            "required_output_path": str(tmp_path / "work" / "result.json"),
        },
    )

    def fake_sleep(_seconds: float) -> None:
        """Publish a backoff sidecar on the first poll."""

        pool = AuthorPool(queue_root, owner="s")
        for job in pool.pending():
            if not job.backoff_path.exists():
                pool.backoff(job, excerpt="usage limit reached. try again at 18:00.")

    ticks = iter([0.0, 1.0, 2.0, 3.0])
    code, detail = operator_author.serve_request(
        request, queue_root=queue_root, monotonic=lambda: next(ticks), sleep=fake_sleep
    )
    assert code == operator_author.EXIT_QUOTA
    assert "provider pause" in detail


def test_author_wrapper_stall_is_retryable_infrastructure(tmp_path: Path) -> None:
    """A dead pool must look like a stalled queue, never a failed model."""

    request = _write_request(
        tmp_path / "work",
        {
            "envelope_version": "menagerie.crawler.author-envelope.v3",
            "work_id": "work-m1",
            "stable_id": "m1",
            "required_output_path": str(tmp_path / "work" / "result.json"),
        },
    )
    ticks = iter([0.0, 100.0, 200.0])
    code, detail = operator_author.serve_request(
        request,
        queue_root=tmp_path / "queue",
        stall_seconds=10.0,
        monotonic=lambda: next(ticks),
        sleep=lambda _s: None,
    )
    assert code == operator_author.EXIT_RETRYABLE
    assert "never claimed" in detail


def test_author_wrapper_refuses_an_unrecognized_envelope(tmp_path: Path) -> None:
    """Guessing a kind would enqueue a job the pool has no brief for."""

    request = _write_request(tmp_path / "work", {"envelope_version": "something-else"})
    code, detail = operator_author.serve_request(
        request, queue_root=tmp_path / "queue", monotonic=lambda: 0.0, sleep=lambda _s: None
    ) if False else (0, "")
    with pytest.raises(operator_author.OperatorAuthorError, match="neither a capability probe"):
        operator_author.classify_request(json.loads(request.read_text(encoding="utf-8")))
    assert (code, detail) == (0, "")


# -- notifier receipt ------------------------------------------------------


def _fake_transport(tmp_path: Path, *, exit_code: int) -> Path:
    """Write a stand-in delivery transport.

    Parameters
    ----------
    tmp_path, exit_code:
        Destination and the status the transport should exit with.

    Returns
    -------
    Path
        Executable transport path.
    """

    script = tmp_path / f"transport-{exit_code}.sh"
    script.write_text(f'#!/bin/sh\necho "delivered: $1"\nexit {exit_code}\n', encoding="utf-8")
    script.chmod(0o755)
    return script


def test_notifier_writes_a_nonce_receipt_only_after_a_successful_delivery(
    tmp_path: Path,
) -> None:
    """The receipt is evidence of delivery, not evidence the script existed."""

    receipt_path = tmp_path / "receipt.json"
    code = operator_notify.deliver(
        "crawler doctor nonce abc123",
        transport=(str(_fake_transport(tmp_path, exit_code=0)),),
        receipt_path=receipt_path,
        nonce="abc123",
    )
    assert code == 0
    receipt = operator_notify.read_receipt(receipt_path)
    assert receipt is not None
    assert receipt["nonce"] == "abc123"
    assert receipt["transport_exit_code"] == 0
    assert receipt["message_sha256"] == hashlib.sha256(
        b"crawler doctor nonce abc123"
    ).hexdigest()


def test_notifier_failure_writes_no_receipt(tmp_path: Path) -> None:
    """A failed delivery must fail the doctor, which means writing nothing."""

    receipt_path = tmp_path / "receipt.json"
    code = operator_notify.deliver(
        "crawler doctor nonce abc123",
        transport=(str(_fake_transport(tmp_path, exit_code=3)),),
        receipt_path=receipt_path,
        nonce="abc123",
    )
    assert code == 3
    assert not receipt_path.exists()


def test_notifier_without_a_transport_writes_no_receipt(tmp_path: Path) -> None:
    """A missing delivery script is a real finding, not a degraded pass."""

    receipt_path = tmp_path / "receipt.json"
    code = operator_notify.deliver(
        "message",
        transport=(str(tmp_path / "does-not-exist.sh"),),
        receipt_path=receipt_path,
        nonce="abc123",
    )
    assert code != 0
    assert not receipt_path.exists()


def test_doctor_notifier_probe_accepts_the_shim_receipt(tmp_path: Path) -> None:
    """End-to-end: the doctor's own probe must accept what the shim writes."""

    from menagerie.crawler.doctor import DoctorConfig, SystemDoctorProbes

    transport = _fake_transport(tmp_path, exit_code=0)
    config = DoctorConfig(
        repo_root=_REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        target="osx-arm64",
    )
    os.environ["MENAGERIE_NOTIFY_TRANSPORT"] = str(transport)
    try:
        assert SystemDoctorProbes(config).notifier_delivery() is True
    finally:
        os.environ.pop("MENAGERIE_NOTIFY_TRANSPORT", None)


def test_doctor_notifier_probe_fails_when_delivery_fails(tmp_path: Path) -> None:
    """The probe must be able to fail; a check that cannot fail proves nothing."""

    from menagerie.crawler.doctor import DoctorConfig, SystemDoctorProbes

    transport = _fake_transport(tmp_path, exit_code=4)
    config = DoctorConfig(
        repo_root=_REPO_ROOT,
        runtime_root=tmp_path / "runtime",
        target="osx-arm64",
    )
    os.environ["MENAGERIE_NOTIFY_TRANSPORT"] = str(transport)
    try:
        assert SystemDoctorProbes(config).notifier_delivery() is False
    finally:
        os.environ.pop("MENAGERIE_NOTIFY_TRANSPORT", None)


def test_notifier_resolution_wraps_the_discovered_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resolved default must be the receipt shim around the same discovered script."""

    import shutil as shutil_module

    from menagerie.crawler.driver_progress import _resolve_notify_command

    monkeypatch.setattr(shutil_module, "which", lambda _name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    script = tmp_path / "scripts" / "send-to-jmt.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)

    resolved = _resolve_notify_command(None)
    assert resolved is not None
    assert resolved[0] == sys.executable
    assert resolved[1:4] == ("-m", "menagerie.crawler.operator_notify", "--transport")
    assert resolved[4] == str(script)


def test_prompt_fragments_ship_with_the_package() -> None:
    """A brief that cannot be rendered is a pool that cannot dispatch."""

    for name in (
        "stage_source_request.md",
        "stage_author.md",
        "stage_capability_probe.md",
        *(f"campaign_{campaign}.md" for campaign in CAMPAIGN_AUTHOR_MODELS),
    ):
        path = pool_module.PROMPT_ROOT / name
        assert path.is_file(), f"missing pool prompt {name}"
        assert path.read_text(encoding="utf-8").strip()


def test_stable_hash_is_used_for_tool_receipts() -> None:
    """Each per-tool receipt binds the nonce, the challenge, and what the tool returned."""

    receipt = validate_capability_evidence(
        nonce=PROBE_NONCE,
        evidence=_valid_evidence(),
        requested_at=REQUESTED_AT,
        now=REQUESTED_AT + timedelta(seconds=45),
    )
    entry = next(item for item in receipt["receipts"] if item["tool"] == "web_fetch_exa")
    assert entry["receipt"].startswith("sha256:") or len(entry["receipt"]) >= 32
    assert entry["receipt"] != stable_hash({"nonce": PROBE_NONCE})
