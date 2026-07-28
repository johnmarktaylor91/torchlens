"""Author executor: headless round trips, machine effort, typed classification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.author_attempts import (
    latest_attempt,
    list_attempts,
    new_attempt,
    record_checker_findings,
)
from menagerie.crawler.author_dispatch import AuthorEffortGrant
from menagerie.crawler.author_executor import (
    EXIT_BACKOFF,
    EXIT_OK,
    EXIT_PERMANENT,
    EXIT_RETRYABLE,
    RECEIPT_VERSION,
    main,
)
from menagerie.crawler.driver_admission import (
    CommandAuthorLane,
    DriverIntegrationError,
    _verify_executor_receipt,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tests.executor_test_support import (
    RESOLVED_SHA,
    executor_environment,
    read_invocations,
    write_author_envelope,
    write_broker_fixtures,
    write_fake_claude,
    write_source_request,
)


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """One configured executor rig: fake harness, fixtures, log, author root."""

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


def _run_source_round(rig, stable_id: str = "m1") -> tuple[int, Path]:
    root = rig["root"]
    request = write_source_request(root, stable_id)
    return main([str(request)]), root


def _run_author_round(rig, stable_id: str = "m1") -> tuple[int, Path]:
    root = rig["root"]
    request = write_author_envelope(root, stable_id)
    return main([str(request)]), root


def test_source_round_publishes_machine_derived_pack(rig, capsys) -> None:
    """Stage 1 + broker publish a pack whose exact strings are machine-derived."""

    code, root = _run_source_round(rig)
    assert code == EXIT_OK
    published = json.loads((root / "source-targets.json").read_text(encoding="utf-8"))
    row = published["sources"][0]
    assert row["revision"] == RESOLVED_SHA
    assert row["expected_sha256"].startswith("sha256:")
    # The receipt on stdout is attempt-bound and digest-matches the bytes.
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["receipt_version"] == RECEIPT_VERSION
    assert receipt["result_sha256"] == hash_bytes(
        (root / "source-targets.json").read_bytes()
    )
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.status == "sources-published"
    assert receipt["attempt_nonce"] == attempt.nonce


def test_pinned_recipe_flags_are_load_bearing_and_present(rig) -> None:
    """Every flag of the verified web-tools recipe is on the session argv."""

    code, _root = _run_source_round(rig)
    assert code == EXIT_OK
    argv = read_invocations(rig["log"])[0]["argv"]
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--mcp-config" in argv
    assert json.loads(argv[argv.index("--mcp-config") + 1]) == {
        "mcpServers": {"exa": {"type": "http", "url": "https://mcp.exa.ai/mcp"}}
    }
    tools_at = argv.index("--allowedTools")
    tools = argv[tools_at + 1 : argv.index("--output-format")]
    assert "WebSearch" in tools
    assert "mcp__exa__web_search_exa" in tools
    assert "mcp__exa__web_fetch_exa" in tools
    assert "ToolSearch" in tools
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--session-id" in argv


def test_author_round_resumes_the_stage1_session(rig) -> None:
    """Stage 2 runs with ``--resume <stage-1 session>`` -- no cold reread."""

    code, root = _run_source_round(rig)
    assert code == EXIT_OK
    stage1 = latest_attempt(root)
    assert stage1 is not None
    stage1_session = stage1.record["stage1"]["session_id"]

    code, root = _run_author_round(rig)
    assert code == EXIT_OK
    assert (root / "result.json").is_file()
    stage2_call = read_invocations(rig["log"])[1]
    assert stage2_call["stage"] == "stage2"
    argv = stage2_call["argv"]
    assert argv[argv.index("--resume") + 1] == stage1_session
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.status == "completed"
    assert attempt.record["stage2"]["resumed_from"] == stage1_session


def test_effort_records_match_the_harness_json(rig) -> None:
    """Effort is machine-counted from the harness JSON, not self-reported."""

    _run_source_round(rig)
    _run_author_round(rig)
    attempt = latest_attempt(rig["root"])
    assert attempt is not None
    for stage in ("stage1", "stage2"):
        effort = attempt.record[stage]["effort"]
        # These exact values are what the fake harness printed as its JSON.
        assert effort["duration_ms"] == 1234
        assert effort["duration_api_ms"] == 987
        assert effort["num_turns"] == 7
        assert effort["total_cost_usd"] == 0.0123
        assert effort["usage"] == {
            "input_tokens": 111,
            "output_tokens": 222,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 20,
        }
        assert effort["timed_out"] is False
        assert effort["wall_seconds_observed"] > 0


def test_structured_limit_signal_is_the_only_pause_authority(rig) -> None:
    """A structured harness limit is exit 76; free-text noise never is."""

    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "limit")
    code, _root = _run_source_round(rig)
    assert code == EXIT_BACKOFF


def test_rate_limit_stderr_noise_does_not_pause(rig) -> None:
    """GitHub rate-limit chatter on stderr never becomes a provider pause."""

    rig["monkeypatch"].setenv(
        "FAKE_CLAUDE_STDERR_NOISE", "GitHub API rate limit exceeded for 1.2.3.4"
    )
    code, _root = _run_source_round(rig)
    assert code == EXIT_OK


def test_session_crash_is_retryable_not_quota(rig) -> None:
    """A crashed session (with rate-limit words on stderr) exits 75, not 76."""

    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "crash")
    code, root = _run_source_round(rig)
    assert code == EXIT_RETRYABLE
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.record["outcome"]["failure_reason"] == "session-crashed"


def test_negative_discovery_arm_is_typed_permanent(rig) -> None:
    """A typed negative arm records its evidence and declares exit 64."""

    rig["monkeypatch"].setenv(
        "FAKE_CLAUDE_DISCOVERY",
        json.dumps(
            {
                "discovery_version": "menagerie.crawler.author-discovery.v1",
                "arm": "NO_USABLE_SOURCE",
                "search_evidence": {"queries": ["q"], "places": [], "conclusion": "none"},
            }
        ),
    )
    code, root = _run_source_round(rig)
    assert code == EXIT_PERMANENT
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.record["outcome"]["failure_reason"] == "discovery-no-usable-source"


def test_tool_failure_arm_fails_loudly_retryable(rig) -> None:
    """RETRYABLE_TOOL_FAILURE is the fail-loudly arm: exit 75, verbatim error kept."""

    rig["monkeypatch"].setenv(
        "FAKE_CLAUDE_DISCOVERY",
        json.dumps(
            {
                "discovery_version": "menagerie.crawler.author-discovery.v1",
                "arm": "RETRYABLE_TOOL_FAILURE",
                "tool": "mcp__exa__web_search_exa",
                "error": "tool not found",
            }
        ),
    )
    code, root = _run_source_round(rig)
    assert code == EXIT_RETRYABLE
    attempt = latest_attempt(root)
    assert attempt is not None
    outcome = attempt.record["outcome"]
    assert outcome["failure_reason"] == "research-tools-unavailable"
    assert outcome["detail"]["tool"] == "mcp__exa__web_search_exa"


def test_supplement_round_is_granted_exactly_once(rig) -> None:
    """A typed supplement request earns one broker pass and one resume."""

    _run_source_round(rig)
    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "supplement-request")
    code, root = _run_author_round(rig)
    assert code == EXIT_OK
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.status == "completed"
    assert attempt.record["supplement"]["manifest_path"]
    stages = [entry["stage"] for entry in read_invocations(rig["log"])]
    assert stages == ["stage1", "stage2", "supplement"]


def test_resume_failure_falls_back_cold_with_recorded_discovery(rig) -> None:
    """A dead provider session reruns cold with the recorded stage-1 output."""

    _run_source_round(rig)
    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "resume-fail")
    code, root = _run_author_round(rig)
    assert code == EXIT_OK
    attempt = latest_attempt(root)
    assert attempt is not None
    assert attempt.record["stage2"]["cold_start_reason"] == "resume-exit-5"
    calls = read_invocations(rig["log"])
    assert "--resume" in calls[1]["argv"]
    assert "--resume" not in calls[2]["argv"]
    assert "COLD START" in calls[2]["prompt"]


def test_prior_attempt_failure_is_rendered_into_the_next_brief(rig) -> None:
    """The feedback channel: a failed attempt's reason reaches the next brief."""

    rig["monkeypatch"].setenv("FAKE_CLAUDE_MODE", "crash")
    code, root = _run_source_round(rig)
    assert code == EXIT_RETRYABLE
    rig["monkeypatch"].delenv("FAKE_CLAUDE_MODE")
    code, root = _run_source_round(rig)
    assert code == EXIT_OK
    retry_brief = read_invocations(rig["log"])[1]["prompt"]
    assert "WHAT WENT WRONG LAST TIME" in retry_brief
    assert "session-crashed" in retry_brief


def test_checker_findings_reach_the_repair_generation_brief(rig) -> None:
    """Checker-rejected leaves are named, verbatim, in generation 2's brief."""

    _run_source_round(rig)
    code, root = _run_author_round(rig)
    assert code == EXIT_OK
    # The checker rejects the accepted generation; the driver records findings.
    assert record_checker_findings(
        root,
        gate_kind="metadata_batch",
        generation=1,
        required_repairs=[
            "external_metadata.citation.title is unsupported by any excerpt",
            "taxonomy.family names a family the sources never state",
        ],
        root_cause_fingerprint="fp-123",
    )
    # The repair generation re-enters through the lane: source round + author.
    code, root = _run_source_round(rig)
    assert code == EXIT_OK
    repair_brief = read_invocations(rig["log"])[2]["prompt"]
    assert "WHAT WENT WRONG LAST TIME" in repair_brief
    assert "external_metadata.citation.title is unsupported" in repair_brief
    assert "taxonomy.family names a family" in repair_brief


def test_ten_model_rung_headless_with_zero_managing_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ten models complete source + author rounds with no managing session.

    No queue directory, no pool, no operator: the only components are the
    executor subprocess contract, the fake harness, and the hermetic broker.
    """

    fake = write_fake_claude(tmp_path / "bin")
    fixtures = write_broker_fixtures(tmp_path / "fixtures")
    log_dir = tmp_path / "log"
    executor_environment(
        monkeypatch, fake_claude=fake, fixtures=fixtures, log_dir=log_dir
    )
    monkeypatch.delenv("MENAGERIE_AUTHOR_QUEUE", raising=False)
    completed = []
    for index in range(10):
        stable_id = f"model-{index:02d}"
        root = tmp_path / "work" / stable_id / "author"
        assert main([str(write_source_request(root, stable_id))]) == EXIT_OK
        assert main([str(write_author_envelope(root, stable_id))]) == EXIT_OK
        assert (root / "result.json").is_file()
        attempt = latest_attempt(root)
        assert attempt is not None and attempt.status == "completed"
        completed.append(stable_id)
    assert len(completed) == 10
    # Zero managing session: nothing ever created a queue or claimed a lease.
    assert not list((tmp_path / "work").rglob("pending")), "no queue dirs may exist"
    stages = [entry["stage"] for entry in read_invocations(log_dir)]
    assert stages == ["stage1", "stage2"] * 10


def test_capability_probe_publishes_evidence_via_executor(rig, capsys) -> None:
    """The doctor probe runs headless; the executor publishes the evidence."""

    probe_dir = rig["tmp"] / "probe"
    probe_dir.mkdir()
    nonce = "f" * 32
    request = {
        "format": "menagerie.crawler.author-capability-probe.v1",
        "nonce": nonce,
        "deadline_seconds": 30,
        "required_output_path": str(probe_dir / "evidence.json"),
    }
    request_path = probe_dir / "probe-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    assert main([str(request_path)]) == EXIT_OK
    assert (probe_dir / "evidence.json").is_file()
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["kind"] == "capability-probe"
    assert receipt["attempt_nonce"] == nonce


def test_lane_receipt_verification_rejects_tampered_bytes(tmp_path: Path) -> None:
    """The lane refuses a result whose bytes drifted from the receipt digest."""

    output = tmp_path / "result.json"
    output.write_text('{"kind": "PROPOSED"}', encoding="utf-8")
    receipt = {
        "receipt_version": RECEIPT_VERSION,
        "attempt_nonce": "n1",
        "result_sha256": hash_bytes(output.read_bytes()),
    }
    stdout = json.dumps(receipt)
    # Matching bytes pass.
    _verify_executor_receipt(stdout, output, stable_id="m1", required=True)
    # A late writer racing the published result is refused.
    output.write_text('{"kind": "STALE_OVERWRITE"}', encoding="utf-8")
    with pytest.raises(DriverIntegrationError, match="do not match the executor receipt"):
        _verify_executor_receipt(stdout, output, stable_id="m1", required=True)


def test_lane_requires_receipt_when_configured(tmp_path: Path) -> None:
    """`require_receipt` makes a receiptless success a typed integration failure."""

    output = tmp_path / "result.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(DriverIntegrationError, match="printed no publication receipt"):
        _verify_executor_receipt("all done", output, stable_id="m1", required=True)
    # Opportunistic mode tolerates legacy wrappers with no receipt.
    _verify_executor_receipt("all done", output, stable_id="m1", required=False)
    lane = CommandAuthorLane(["true"], effort_grant=AuthorEffortGrant())
    assert lane.require_receipt is False


def test_supersession_quarantines_stale_results_before_new_attempt(
    tmp_path: Path,
) -> None:
    """Sol's laundering repro, ported: a late old completion is quarantined.

    The old attempt's late ``result.json`` can never be stamped with a new
    attempt's nonce: attempts never share paths, and opening attempt N+1
    moves any stray result into quarantine, never into publication.
    """

    root = tmp_path / "author"
    old = new_attempt(root, stable_id="m1", campaign_id="c1-mech", kind="author")
    old.update(status="stage2-running")
    stale = old.paths.directory / "result.json"
    stale.write_text('{"kind": "STALE", "from": "old-attempt"}', encoding="utf-8")

    fresh = new_attempt(root, stable_id="m1", campaign_id="c1-mech", kind="author")
    assert not stale.exists(), "the stale result must leave the publishable path"
    superseded = list_attempts(root)[0]
    assert superseded.status == "superseded"
    assert superseded.record["superseded"]["by_nonce"] == fresh.nonce
    quarantined = superseded.record["quarantine"]
    assert len(quarantined) == 1
    payload = json.loads(Path(quarantined[0]["quarantined_to"]).read_text("utf-8"))
    assert payload["from"] == "old-attempt"
