"""Hard-kill matrix: the executor dies at every stage boundary and resumes.

These tests SIGKILL the real executor subprocess — the weakest state holder in
the author path — at four points: mid-stage-1, between stages (after durable
stage-1 state, before publication), mid-stage-2, and post-timeout with a
still-live provider session. Each must resume correctly on the next
invocation, and a late completion from a superseded attempt must be
quarantined, never published.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from menagerie.crawler.author_attempts import latest_attempt, list_attempts, new_attempt
from menagerie.crawler.author_executor import EXIT_OK, EXIT_RETRYABLE
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tests.executor_test_support import (
    read_invocations,
    subprocess_environment,
    write_author_envelope,
    write_broker_fixtures,
    write_fake_claude,
    write_source_request,
)

pytestmark = pytest.mark.heavy

_POLL_TIMEOUT = 20.0


def _wait_for(predicate, *, timeout: float = _POLL_TIMEOUT, message: str = "") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for: {message}")


class _Rig:
    """A subprocess executor rig with scriptable fake-harness behavior."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.fake = write_fake_claude(tmp_path / "bin")
        self.fixtures = write_broker_fixtures(tmp_path / "fixtures")
        self.log = tmp_path / "log"
        self.marker = tmp_path / "marker"
        self.marker.mkdir()
        self.root = tmp_path / "work" / "m1" / "author"

    def env(self, **extra: str) -> dict:
        return subprocess_environment(
            fake_claude=self.fake,
            fixtures=self.fixtures,
            log_dir=self.log,
            FAKE_CLAUDE_MARKER=str(self.marker),
            **extra,
        )

    def spawn(self, request: Path, **extra: str) -> subprocess.Popen:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "menagerie.crawler.author_executor",
                "--campaign",
                "c1-mech",
                str(request),
            ],
            env=self.env(**extra),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def run(self, request: Path, **extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "menagerie.crawler.author_executor",
                "--campaign",
                "c1-mech",
                str(request),
            ],
            env=self.env(**extra),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def kill_marked_fake(self, stage: str) -> None:
        marker = self.marker / f"{stage}.started"
        if marker.is_file():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


@pytest.fixture()
def rig(tmp_path: Path) -> _Rig:
    return _Rig(tmp_path)


def test_kill_mid_stage1_resumes_with_fresh_attempt_and_feedback(rig: _Rig) -> None:
    """A mid-stage-1 kill leaves a durable record; the retry opens attempt 2.

    The interrupted provider session never completed, so it is provably
    unrecoverable; the retry's brief carries the prior attempt.
    """

    request = write_source_request(rig.root, "m1")
    executor = rig.spawn(request, FAKE_CLAUDE_MODE="sleep-stage1")
    try:
        _wait_for(
            lambda: (rig.marker / "stage1.started").is_file(),
            message="stage 1 session start",
        )
        _wait_for(
            lambda: latest_attempt(rig.root) is not None
            and latest_attempt(rig.root).status == "stage1-running",
            message="durable stage1-running record",
        )
        executor.kill()
        executor.wait(timeout=10)
    finally:
        rig.kill_marked_fake("stage1")

    completed = rig.run(request)
    assert completed.returncode == EXIT_OK, completed.stderr
    attempts = list_attempts(rig.root)
    assert [handle.number for handle in attempts] == [1, 2]
    assert attempts[0].status == "superseded"
    assert attempts[1].status == "sources-published"
    retry_brief = read_invocations(rig.log)[-1]["prompt"]
    assert "WHAT WENT WRONG LAST TIME" in retry_brief


def test_kill_between_stages_replays_broker_without_rerunning_stage1(rig: _Rig) -> None:
    """A kill after durable stage-1 completion resumes at the broker.

    No second stage-1 session runs: the recorded discovery output feeds the
    broker directly — no cold reread of the sources.
    """

    request = write_source_request(rig.root, "m1")
    executor = rig.spawn(request, MENAGERIE_EXECUTOR_PAUSE_AFTER="stage1")
    try:
        _wait_for(
            lambda: latest_attempt(rig.root) is not None
            and latest_attempt(rig.root).status == "stage1-complete",
            message="durable stage1-complete record",
        )
        executor.kill()
        executor.wait(timeout=10)
    finally:
        pass

    completed = rig.run(request)
    assert completed.returncode == EXIT_OK, completed.stderr
    attempt = latest_attempt(rig.root)
    assert attempt is not None
    assert attempt.number == 1, "the same attempt resumes; no new session"
    assert attempt.status == "sources-published"
    stages = [entry["stage"] for entry in read_invocations(rig.log)]
    assert stages == ["stage1"], "stage 1 must not re-run"
    events = attempt.paths.events.read_text(encoding="utf-8")
    assert "resumed-broker-from-record" in events


def test_kill_mid_stage2_resumes_stage1_session_in_fresh_attempt(rig: _Rig) -> None:
    """A mid-stage-2 kill resumes via ``--resume <stage-1 session>``.

    The new attempt inherits the recorded stage-1 session identity — context
    retention without a cold reread — and never reuses the killed attempt's
    directory.
    """

    request = write_source_request(rig.root, "m1")
    assert rig.run(request).returncode == EXIT_OK
    stage1_attempt = latest_attempt(rig.root)
    assert stage1_attempt is not None
    stage1_session = stage1_attempt.record["stage1"]["session_id"]

    author_request = write_author_envelope(rig.root, "m1")
    executor = rig.spawn(author_request, FAKE_CLAUDE_MODE="sleep-stage2")
    try:
        _wait_for(
            lambda: (rig.marker / "stage2.started").is_file(),
            message="stage 2 session start",
        )
        executor.kill()
        executor.wait(timeout=10)
    finally:
        rig.kill_marked_fake("stage2")

    completed = rig.run(author_request)
    assert completed.returncode == EXIT_OK, completed.stderr
    attempts = [
        handle for handle in list_attempts(rig.root) if handle.record["kind"] == "author"
    ]
    assert attempts[-1].status == "completed"
    assert attempts[-1].record["inherited"]["stage1"]["session_id"] == stage1_session
    resume_calls = [
        entry
        for entry in read_invocations(rig.log)
        if entry["stage"] == "stage2" and "--resume" in entry["argv"]
    ]
    argv = resume_calls[-1]["argv"]
    assert argv[argv.index("--resume") + 1] == stage1_session


def test_executor_timeout_supersedes_attempt_and_exits_retryable(rig: _Rig) -> None:
    """The external wall kill supersedes the attempt with a typed exit 75."""

    request = write_source_request(rig.root, "m1")
    assert rig.run(request).returncode == EXIT_OK
    author_request = write_author_envelope(rig.root, "m1")
    completed = rig.run(
        author_request,
        FAKE_CLAUDE_MODE="sleep-stage2",
        FAKE_CLAUDE_SLEEP="30",
        MENAGERIE_AUTHOR_WALL_SECONDS="1",
    )
    assert completed.returncode == EXIT_RETRYABLE
    assert "wall-exceeded" in completed.stderr
    timed_out = latest_attempt(rig.root)
    assert timed_out is not None
    assert timed_out.status == "superseded"
    assert timed_out.record["superseded"]["reason"] == "timeout"


def test_late_completion_from_live_provider_session_is_quarantined(rig: _Rig) -> None:
    """Sol's laundering scenario against the executor, end to end.

    The executor is killed while the provider session is still alive (its own
    process group survives the executor's death). The orphan later writes its
    result into the superseded attempt's directory. The retry publishes the
    NEW attempt's bytes; the stale result is quarantined at the next attempt
    boundary and its bytes never reach the required path.
    """

    request = write_source_request(rig.root, "m1")
    assert rig.run(request).returncode == EXIT_OK
    author_request = write_author_envelope(rig.root, "m1")
    stale_payload = '{"kind": "STALE", "from": "orphaned-session"}'
    executor = rig.spawn(
        author_request,
        FAKE_CLAUDE_MODE="late-write-stage2",
        FAKE_CLAUDE_SLEEP="3",
        FAKE_CLAUDE_LATE_RESULT=stale_payload,
    )
    _wait_for(
        lambda: (rig.marker / "stage2.started").is_file(),
        message="stage 2 session start",
    )
    # Kill ONLY the executor; the provider session lives on in its own group.
    executor.kill()
    executor.wait(timeout=10)

    fresh_payload = '{"kind": "PROPOSED", "from": "fresh-attempt"}'
    completed = rig.run(author_request, FAKE_CLAUDE_RESULT=fresh_payload)
    assert completed.returncode == EXIT_OK, completed.stderr

    # The published result is the fresh attempt's, receipt-bound to its nonce.
    published = (rig.root / "result.json").read_bytes()
    assert json.loads(published)["from"] == "fresh-attempt"
    receipt = json.loads(completed.stdout.strip().splitlines()[-1])
    fresh_attempt = latest_attempt(rig.root)
    assert fresh_attempt is not None
    assert receipt["attempt_nonce"] == fresh_attempt.nonce
    assert receipt["result_sha256"] == hash_bytes(published)

    # Wait for the orphaned session to land its late write in the OLD attempt
    # (stage 2 ran inside the attempt that published the sources, so the
    # superseded attempt is that one, whatever kind opened it).
    orphan_target = [
        handle for handle in list_attempts(rig.root) if handle.status == "superseded"
    ][-1]
    _wait_for(
        lambda: (orphan_target.paths.directory / "result.json").is_file(),
        message="orphaned provider session's late write",
    )
    late_bytes = (orphan_target.paths.directory / "result.json").read_bytes()
    assert json.loads(late_bytes)["from"] == "orphaned-session"
    # The published result is untouched by the late write.
    assert (rig.root / "result.json").read_bytes() == published

    # The next attempt boundary quarantines the late output for good.
    new_attempt(rig.root, stable_id="m1", campaign_id="c1-mech", kind="author")
    assert not (orphan_target.paths.directory / "result.json").exists()
    reloaded = [
        handle
        for handle in list_attempts(rig.root)
        if handle.nonce == orphan_target.nonce
    ][0]
    quarantined = reloaded.record["quarantine"]
    assert any(
        json.loads(Path(entry["quarantined_to"]).read_text("utf-8")).get("from")
        == "orphaned-session"
        for entry in quarantined
    )
