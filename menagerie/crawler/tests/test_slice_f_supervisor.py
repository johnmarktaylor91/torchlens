"""Tests for the W1.6 campaign supervisor and host-global execution flock."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import inspect
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import time
from typing import BinaryIO, Optional, Sequence

import pytest

from menagerie.crawler.campaign_config import CampaignConfig
from menagerie.crawler.cli import (
    EXIT_ERROR,
    EXIT_LOCKED,
    EXIT_OPERATOR_OUTAGE,
    EXIT_PAUSED,
    EXIT_REVIEW_PAUSED,
)
from menagerie.crawler.driver_progress import CommandNotifier
from menagerie.crawler.driver_receipts import SupervisedForwardLane
from menagerie.crawler import worker_supervisor
from menagerie.crawler.supervisor import (
    CRASH_LIMIT,
    CrawlerSupervisor,
    QueueStall,
    SupervisorConfig,
    SupervisorError,
    render_launchd_plist,
    stalled_queue_jobs,
    validate_supervised_campaign,
)

NOW = datetime(2026, 7, 27, 16, 0, tzinfo=timezone.utc)


class FakeNotifier:
    """Record supervisor notifications and optionally fail initial attempts."""

    def __init__(self, *, failures: int = 0) -> None:
        """Configure the number of missing-receipt attempts."""

        self.failures = failures
        self.messages: list[tuple[str, str]] = []

    def notify(self, summary: str, *, idempotency_key: str) -> bool:
        """Record one notification and return its scripted receipt status."""

        self.messages.append((summary, idempotency_key))
        if self.failures:
            self.failures -= 1
            return False
        return True


class FakeProcess:
    """Pollable driver process with a scripted status sequence."""

    def __init__(self, returncode: int, *, running_polls: int = 0) -> None:
        """Store the exit code and number of pre-exit polls."""

        self.returncode = returncode
        self.running_polls = running_polls

    def poll(self) -> Optional[int]:
        """Return ``None`` for the scripted running polls, then the status."""

        if self.running_polls:
            self.running_polls -= 1
            return None
        return self.returncode

    def wait(self) -> int:
        """Return the scripted final status."""

        return self.returncode


class FakeProcessFactory:
    """Create fake processes from an ordered exit-code script."""

    def __init__(self, codes: Sequence[int], *, running_polls: int = 0) -> None:
        """Store process exit codes and optional initial running polls."""

        self.codes = list(codes)
        self.running_polls = running_polls
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self,
        command: Sequence[str],
        cwd: Path,
        stdout: BinaryIO,
        stderr: BinaryIO,
    ) -> FakeProcess:
        """Return the next fake process and record its exact argv."""

        del cwd, stdout, stderr
        self.commands.append(tuple(command))
        return FakeProcess(self.codes.pop(0), running_polls=self.running_polls)


def _config(
    tmp_path: Path,
    *,
    queue_root: Optional[Path] = None,
    wake_episode_id: Optional[str] = None,
) -> SupervisorConfig:
    """Build a test supervisor configuration."""

    campaign_config = tmp_path / "campaign.json"
    campaign_config.write_text("{}\n", encoding="utf-8")
    return SupervisorConfig(
        campaign_id="c1-mech",
        repo_root=tmp_path.resolve(),
        campaign_config_path=campaign_config.resolve(),
        author_queue_root=queue_root.resolve() if queue_root is not None else None,
        runtime_root=(tmp_path / ".crawl-local" / "supervisor" / "c1-mech").resolve(),
        wake_episode_id=wake_episode_id,
        poll_seconds=1.0,
        notification_retry_seconds=1.0,
    )


@pytest.mark.parametrize(
    ("driver_exit", "message_fragment"),
    [
        (EXIT_PAUSED, "usage"),
        (EXIT_REVIEW_PAUSED, "review checkpoint"),
        (EXIT_OPERATOR_OUTAGE, "RETRYABLE INFRASTRUCTURE"),
    ],
)
def test_supervisor_handles_meaningful_stop_codes_without_restart(
    tmp_path: Path,
    driver_exit: int,
    message_fragment: str,
) -> None:
    """Usage, review, and operator stops each alert once and halt cleanly."""

    notifier = FakeNotifier()
    factory = FakeProcessFactory([driver_exit])
    supervisor = CrawlerSupervisor(
        _config(tmp_path),
        notifier,
        process_factory=factory,
        now=lambda: NOW,
        sleep=lambda _seconds: None,
    )

    assert supervisor.run() == 0
    assert len(factory.commands) == 1
    assert message_fragment in notifier.messages[0][0]


def test_supervisor_backs_off_lock_contention_then_restarts(tmp_path: Path) -> None:
    """Exit 3 is ordinary driver ownership contention, not a crash."""

    notifier = FakeNotifier()
    factory = FakeProcessFactory([EXIT_LOCKED, EXIT_PAUSED])
    sleeps: list[float] = []
    supervisor = CrawlerSupervisor(
        _config(tmp_path),
        notifier,
        process_factory=factory,
        now=lambda: NOW,
        sleep=sleeps.append,
    )

    assert supervisor.run() == 0
    assert len(factory.commands) == 2
    assert sleeps == [30.0]


def test_supervisor_crash_circuit_is_persisted_and_alerted(tmp_path: Path) -> None:
    """Five unexpected exits in 30 minutes open the circuit after bounded backoff."""

    notifier = FakeNotifier(failures=1)
    factory = FakeProcessFactory([EXIT_ERROR] * CRASH_LIMIT)
    sleeps: list[float] = []
    supervisor = CrawlerSupervisor(
        _config(tmp_path),
        notifier,
        process_factory=factory,
        now=lambda: NOW,
        sleep=sleeps.append,
    )

    assert supervisor.run() == 0
    assert len(factory.commands) == CRASH_LIMIT
    assert "circuit opened" in notifier.messages[-1][0]
    assert len(notifier.messages) == 2
    state = json.loads(
        (tmp_path / ".crawl-local" / "supervisor" / "c1-mech" / "state.json").read_text()
    )
    assert len(state["crash_timestamps"]) == CRASH_LIMIT


def test_queue_watchdog_reports_stall_as_retryable_infrastructure(tmp_path: Path) -> None:
    """A >45-minute pending job is surfaced without creating a model result."""

    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "claimed").mkdir()
    job = {
        "job_id": "author-job-1",
        "attempt_nonce": "attempt-1",
        "stable_id": "m_stalled",
        "enqueued_at": (NOW - timedelta(minutes=46)).isoformat().replace("+00:00", "Z"),
    }
    (queue / "pending" / "author-job-1.json").write_text(json.dumps(job), encoding="utf-8")
    (queue / "claimed" / "author-job-1.json").write_text("{}\n", encoding="utf-8")

    assert stalled_queue_jobs(queue, now=NOW) == (
        QueueStall("author-job-1", "attempt-1", "m_stalled", 46 * 60.0, True),
    )
    notifier = FakeNotifier()
    supervisor = CrawlerSupervisor(
        _config(tmp_path, queue_root=queue),
        notifier,
        process_factory=FakeProcessFactory([EXIT_OPERATOR_OUTAGE], running_polls=1),
        now=lambda: NOW,
        sleep=lambda _seconds: None,
    )

    assert supervisor.run() == 0
    assert "author queue stalled >45 min" in notifier.messages[0][0]
    assert "RETRYABLE INFRASTRUCTURE" in notifier.messages[0][0]
    assert not (tmp_path / "menagerie" / "crawler" / "records" / "models").exists()


def test_global_execution_flock_serializes_separate_campaign_clones(tmp_path: Path) -> None:
    """Two processes with different campaign roots cannot own the global slot together."""

    lock_path = tmp_path / "host" / "execution.flock"
    holder_ready = tmp_path / "holder-ready"
    release_holder = tmp_path / "release-holder"
    contender_ready = tmp_path / "contender-ready"
    program = (
        "import os,sys,time; from pathlib import Path; "
        "from menagerie.crawler.execution_lock import "
        "acquire_global_execution_flock,release_global_execution_flock; "
        "fd=acquire_global_execution_flock(Path(sys.argv[1])); "
        "Path(sys.argv[2]).write_text('acquired'); "
        "\nwhile not Path(sys.argv[3]).exists(): time.sleep(0.01)\n"
        "release_global_execution_flock(fd)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", program, str(lock_path), str(holder_ready), str(release_holder)]
    )
    contender_release = tmp_path / "release-contender"
    contender: Optional[subprocess.Popen[bytes]] = None
    try:
        deadline = datetime.now().timestamp() + 5
        while not holder_ready.exists() and datetime.now().timestamp() < deadline:
            time.sleep(0.01)
        assert holder_ready.exists()
        contender = subprocess.Popen(
            [
                sys.executable,
                "-c",
                program,
                str(lock_path),
                str(contender_ready),
                str(contender_release),
            ]
        )
        time.sleep(0.2)
        assert not contender_ready.exists()
        release_holder.write_text("release", encoding="utf-8")
        deadline = datetime.now().timestamp() + 5
        while not contender_ready.exists() and datetime.now().timestamp() < deadline:
            time.sleep(0.01)
        assert contender_ready.exists()
    finally:
        release_holder.touch()
        contender_release.touch()
        holder.wait(timeout=5)
        if contender is not None:
            contender.wait(timeout=5)


def test_global_execution_flock_default_is_outside_campaign_runtime(
    tmp_path: Path,
) -> None:
    """The production default cannot accidentally resolve inside a clone."""

    from menagerie.crawler.execution_lock import global_execution_flock_path

    assert not global_execution_flock_path().is_relative_to(tmp_path / ".crawl-local")


def test_production_worker_spawn_inherits_the_global_flock() -> None:
    """The only model-worker spawn path carries the global descriptor into its child."""

    lane_source = inspect.getsource(SupervisedForwardLane.forward)
    spawn_source = inspect.getsource(worker_supervisor.run_isolated_subprocess)

    assert "global_lock_path=self.execution_flock_path" in lane_source
    assert "worker_lease_handle.global_lock_fd" in spawn_source
    assert "pass_fds=inherited_fds" in spawn_source


def test_command_notifier_requires_the_operator_shim_nonce_receipt(tmp_path: Path) -> None:
    """Exit zero without a receipt is failure; the shipped shim makes it success."""

    receipt_root = tmp_path / "receipts"
    silent = CommandNotifier(
        f"{sys.executable} -c pass",
        receipt_root=receipt_root,
    )
    assert silent.notify("silent", idempotency_key="nonce-silent") is False

    shim = CommandNotifier(
        (
            f"{sys.executable} -m menagerie.crawler.operator_notify "
            "--transport /usr/bin/true"
        ),
        receipt_root=receipt_root,
    )
    assert shim.notify("visible", idempotency_key="nonce-visible") is True


def test_launchd_plist_keeps_only_unexpected_supervisor_death_alive(tmp_path: Path) -> None:
    """The rendered agent does not relaunch handled pause/complete exits."""

    payload = plistlib.loads(
        render_launchd_plist(
            campaign_id="c1-mech",
            repo_root=tmp_path,
            campaign_config_path=tmp_path / "campaign.json",
            author_queue_root=tmp_path / "queue",
            python_executable=Path(sys.executable),
        )
    )

    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["ProgramArguments"][0].endswith("tools/crawler_supervisor.sh")
    assert payload["ProgramArguments"][-2:] == ["--author-queue", str(tmp_path / "queue")]


def test_checkpoint_policy_requires_only_c1_gate_and_advance_notifications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1 keeps 1000+900/950/1000 while C2-C4 require a zero checkpoint."""

    from menagerie.crawler.partitioner import CampaignBinding, CAMPAIGN_SPECS
    import menagerie.crawler.supervisor as supervisor_module

    snapshot = object()
    monkeypatch.setattr(supervisor_module, "load_intake_snapshot", lambda _path: snapshot)

    def binding(campaign_id: str) -> CampaignBinding:
        """Return a minimal manifest binding for one frozen campaign."""

        spec = next(item for item in CAMPAIGN_SPECS if item.campaign_id == campaign_id)
        return CampaignBinding(spec, 1, "p", "sha256:x", "i", "intake-x", "sha256:y", 1)

    base = dict(
        repo_root=tmp_path,
        intake_root=tmp_path / "intake",
        target="osx-arm64",
        run_id="run",
        author_queue_root=tmp_path / "queue",
        author_command=(sys.executable,),
        checker_command=(sys.executable,),
        environment_command=(sys.executable,),
        notify_command=(sys.executable,),
        public_mirror=tmp_path / "public",
        private_mirror=tmp_path / "private",
        phase="pytorch",
        only_status=None,
    )
    c1 = CampaignConfig(
        **base,
        review_checkpoint_at=1000,
        progress_milestones=(900, 950, 1000),
    )
    monkeypatch.setattr(
        supervisor_module, "find_campaign_binding", lambda _path, _snapshot: binding("c1-mech")
    )
    validate_supervised_campaign(tmp_path, "c1-mech", c1)
    with pytest.raises(SupervisorError, match="notify"):
        validate_supervised_campaign(
            tmp_path,
            "c1-mech",
            CampaignConfig(
                **base,
                review_checkpoint_at=1000,
                progress_milestones=(900, 950),
            ),
        )

    monkeypatch.setattr(
        supervisor_module, "find_campaign_binding", lambda _path, _snapshot: binding("c2-disco")
    )
    with pytest.raises(SupervisorError, match="must be 0"):
        validate_supervised_campaign(tmp_path, "c2-disco", c1)
