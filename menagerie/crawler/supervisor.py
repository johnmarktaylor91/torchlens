"""Per-campaign crawler keep-alive and launchd integration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import plistlib
import shlex
import subprocess
import sys
import time
from typing import Any, BinaryIO, Callable, Mapping, Optional, Protocol, Sequence

from menagerie.crawler.campaign_config import CampaignConfig, load_campaign_config
from menagerie.crawler.cli import (
    EXIT_ERROR,
    EXIT_LOCKED,
    EXIT_OK,
    EXIT_OPERATOR_OUTAGE,
    EXIT_PAUSED,
    EXIT_REVIEW_PAUSED,
)
from menagerie.crawler.constants import AUTHOR_QUEUE_STALL_SECONDS
from menagerie.crawler.driver_progress import CommandNotifier
from menagerie.crawler.identity import atomic_replace_bytes, canonical_json_bytes, stable_hash
from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.partitioner import DEFAULT_CAMPAIGN_MANIFEST, find_campaign_binding

CRASH_WINDOW_SECONDS = 30 * 60
CRASH_LIMIT = 5
MAX_RESTART_BACKOFF_SECONDS = 15 * 60
LOCK_BACKOFF_SECONDS = 30.0
WATCHDOG_POLL_SECONDS = 30.0
NOTIFICATION_RETRY_SECONDS = 60.0


class SupervisorError(RuntimeError):
    """Raised when the supervisor configuration cannot preserve its contracts."""


class NotificationLane(Protocol):
    """Receipt-bearing operator notification boundary."""

    def notify(self, summary: str, *, idempotency_key: str) -> bool:
        """Deliver one idempotent summary with a nonce receipt."""

        ...


class ManagedProcess(Protocol):
    """Minimal subprocess surface polled by the keep-alive."""

    def poll(self) -> Optional[int]:
        """Return the exit status, or ``None`` while running."""

        ...

    def wait(self) -> int:
        """Wait for process completion and return its status."""

        ...


ProcessFactory = Callable[
    [Sequence[str], Path, BinaryIO, BinaryIO],
    ManagedProcess,
]


@dataclass(frozen=True)
class QueueStall:
    """One author queue job older than the R6 liveness deadline."""

    job_id: str
    attempt_nonce: str
    stable_id: str
    age_seconds: float
    claimed: bool
    marker_path: Optional[Path] = None


@dataclass(frozen=True)
class SupervisorConfig:
    """Frozen per-campaign keep-alive configuration."""

    campaign_id: str
    repo_root: Path
    campaign_config_path: Path
    author_queue_root: Optional[Path]
    runtime_root: Path
    wake_episode_id: Optional[str] = None
    stall_seconds: float = float(AUTHOR_QUEUE_STALL_SECONDS)
    poll_seconds: float = WATCHDOG_POLL_SECONDS
    notification_retry_seconds: float = NOTIFICATION_RETRY_SECONDS

    def __post_init__(self) -> None:
        """Validate absolute paths and positive watchdog intervals."""

        if not self.campaign_id:
            raise ValueError("campaign_id cannot be empty")
        for path in (self.repo_root, self.campaign_config_path, self.runtime_root):
            if not path.is_absolute():
                raise ValueError(f"supervisor path must be absolute: {path}")
        if self.author_queue_root is not None and not self.author_queue_root.is_absolute():
            raise ValueError("author_queue_root must be absolute")
        if min(self.stall_seconds, self.poll_seconds, self.notification_retry_seconds) <= 0:
            raise ValueError("supervisor intervals must be positive")


@dataclass(frozen=True)
class SupervisorState:
    """Durable crash-window and delivered-stall projection."""

    crash_timestamps: tuple[str, ...] = ()
    notified_stalls: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return the closed durable representation.

        Returns
        -------
        dict[str, object]
            JSON-compatible supervisor state.
        """

        return {
            "format": "menagerie.crawler.supervisor-state.v1",
            "crash_timestamps": list(self.crash_timestamps),
            "notified_stalls": list(self.notified_stalls),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SupervisorState":
        """Parse one durable state document.

        Parameters
        ----------
        value:
            Decoded JSON mapping.

        Returns
        -------
        SupervisorState
            Validated state.
        """

        if value.get("format") != "menagerie.crawler.supervisor-state.v1":
            raise SupervisorError("supervisor state format is invalid")
        crashes = value.get("crash_timestamps")
        stalls = value.get("notified_stalls")
        if (
            not isinstance(crashes, list)
            or any(not isinstance(item, str) for item in crashes)
            or not isinstance(stalls, list)
            or any(not isinstance(item, str) for item in stalls)
        ):
            raise SupervisorError("supervisor state fields are invalid")
        return cls(tuple(crashes), tuple(stalls))


class CrawlerSupervisor:
    """Restart one campaign driver and alert on every terminal stop condition."""

    def __init__(
        self,
        config: SupervisorConfig,
        notifier: NotificationLane,
        *,
        process_factory: Optional[ProcessFactory] = None,
        now: Optional[Callable[[], datetime]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        """Bind injected process, clock, sleep, and receipt notifier effects."""

        self.config = config
        self.notifier = notifier
        self._process_factory = process_factory or _start_process
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or time.sleep
        self.config.runtime_root.mkdir(parents=True, exist_ok=True)
        self._state_path = self.config.runtime_root / "state.json"
        self._state = _load_state(self._state_path)

    def run(self) -> int:
        """Supervise until completion, a deliberate pause, or the crash circuit opens.

        Returns
        -------
        int
            Always ``0`` for a handled stop. launchd's unsuccessful-exit keep-alive is
            therefore reserved for an unexpected death of this supervisor itself.
        """

        while True:
            returncode = self._run_driver_once()
            if returncode == EXIT_OK:
                status = _driver_state_status(self.config.repo_root / ".crawl-local")
                if status and (
                    status.startswith("complete")
                    or status.startswith("phase-complete")
                    or status.startswith("terminal-partition-complete")
                    or status.startswith("phase-terminal-partition-complete")
                ):
                    self._notify_until_delivered(
                        "campaign-complete",
                        f"Menagerie crawler {self.config.campaign_id} completed: {status}",
                    )
                return EXIT_OK
            if returncode == EXIT_PAUSED:
                self._notify_until_delivered(
                    "usage-pause",
                    (
                        f"Menagerie crawler {self.config.campaign_id} paused for provider "
                        "usage; the durable reset-time wake episode now owns resume"
                    ),
                )
                return EXIT_OK
            if returncode == EXIT_REVIEW_PAUSED:
                self._notify_until_delivered(
                    "review-pause",
                    (
                        f"Menagerie crawler {self.config.campaign_id} reached its human "
                        "review checkpoint; resume requires --after-review"
                    ),
                )
                return EXIT_OK
            if returncode == EXIT_OPERATOR_OUTAGE:
                self._notify_until_delivered(
                    "operator-outage",
                    (
                        f"Menagerie crawler {self.config.campaign_id} stopped for "
                        "RETRYABLE INFRASTRUCTURE; restore the managing session/operator lane"
                    ),
                )
                return EXIT_OK
            if returncode == EXIT_LOCKED:
                self._sleep(LOCK_BACKOFF_SECONDS)
                continue
            crashes = self._record_crash()
            if len(crashes) >= CRASH_LIMIT:
                self._notify_until_delivered(
                    "crash-circuit-open",
                    (
                        f"Menagerie crawler {self.config.campaign_id} circuit opened after "
                        f"{len(crashes)} unexpected exits in 30 minutes"
                    ),
                )
                return EXIT_OK
            self._sleep(_restart_backoff_seconds(len(crashes)))

    def _run_driver_once(self) -> int:
        """Spawn one driver invocation while polling the author queue watchdog."""

        command = _driver_command(self.config)
        stdout_path = self.config.runtime_root / "driver.stdout.log"
        stderr_path = self.config.runtime_root / "driver.stderr.log"
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = self._process_factory(command, self.config.repo_root, stdout, stderr)
            while process.poll() is None:
                self._check_queue_stalls()
                self._sleep(self.config.poll_seconds)
            return process.wait()

    def _check_queue_stalls(self) -> None:
        """Notify once for each queue job crossing the 45-minute deadline."""

        queue_root = self.config.author_queue_root
        if queue_root is None:
            return
        notified = set(self._state.notified_stalls)
        changed = False
        for stall in stalled_queue_jobs(
            queue_root,
            now=self._now(),
            stall_seconds=self.config.stall_seconds,
        ):
            identity = stable_hash(
                {
                    "campaign_id": self.config.campaign_id,
                    "job_id": stall.job_id,
                    "attempt_nonce": stall.attempt_nonce,
                }
            )
            if identity in notified:
                continue
            delivered = self.notifier.notify(
                (
                    f"Menagerie crawler {self.config.campaign_id} RETRYABLE INFRASTRUCTURE: "
                    f"author queue stalled >45 min for {stall.stable_id} "
                    f"({'claimed' if stall.claimed else 'unclaimed'})"
                ),
                idempotency_key=identity,
            )
            if delivered:
                notified.add(identity)
                changed = True
                if stall.marker_path is not None:
                    stall.marker_path.unlink(missing_ok=True)
        if changed:
            self._state = SupervisorState(
                self._state.crash_timestamps,
                tuple(sorted(notified)),
            )
            _write_state(self._state_path, self._state)

    def _record_crash(self) -> tuple[str, ...]:
        """Append and persist one unexpected driver exit inside the rolling window."""

        now = self._now()
        cutoff = now - timedelta(seconds=CRASH_WINDOW_SECONDS)
        recent = [
            timestamp
            for timestamp in self._state.crash_timestamps
            if _parse_timestamp(timestamp) >= cutoff
        ]
        recent.append(_format_timestamp(now))
        self._state = SupervisorState(tuple(recent), self._state.notified_stalls)
        _write_state(self._state_path, self._state)
        return tuple(recent)

    def _notify_until_delivered(self, event: str, summary: str) -> None:
        """Retry a terminal alert until the notifier produces its nonce receipt."""

        identity = stable_hash(
            {
                "campaign_id": self.config.campaign_id,
                "event": event,
                "wake_episode_id": self.config.wake_episode_id,
            }
        )
        while not self.notifier.notify(summary, idempotency_key=identity):
            self._sleep(self.config.notification_retry_seconds)


def stalled_queue_jobs(
    queue_root: Path,
    *,
    now: datetime,
    stall_seconds: float = float(AUTHOR_QUEUE_STALL_SECONDS),
) -> tuple[QueueStall, ...]:
    """Return pending queue jobs older than the retryable-infrastructure deadline.

    Parameters
    ----------
    queue_root:
        Campaign-local author queue.
    now:
        Current aware UTC instant.
    stall_seconds:
        Queue latency threshold.

    Returns
    -------
    tuple[QueueStall, ...]
        Deterministically ordered stalled jobs. Malformed jobs are left to the lane's
        fail-closed protocol validation and are not guessed here.
    """

    if now.tzinfo is None:
        raise ValueError("queue watchdog requires an aware timestamp")
    stalls: list[QueueStall] = []
    marked_attempts: set[tuple[str, str]] = set()
    for path in sorted((queue_root / "watchdog").glob("*.stall.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("format") != "menagerie.crawler.author-queue-stall.v1":
                continue
            job_id = str(value["job_id"])
            attempt_nonce = str(value["attempt_nonce"])
            stable_id = str(value["stable_id"])
            raw_claimed = value["claimed"]
            if not isinstance(raw_claimed, bool):
                continue
            enqueued_at = _parse_timestamp(str(value["enqueued_at"]))
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
        age = max(stall_seconds, (now - enqueued_at).total_seconds())
        stalls.append(
            QueueStall(job_id, attempt_nonce, stable_id, age, raw_claimed, marker_path=path)
        )
        marked_attempts.add((job_id, attempt_nonce))
    for path in sorted((queue_root / "pending").glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            enqueued_at = _parse_timestamp(str(value["enqueued_at"]))
            job_id = str(value["job_id"])
            attempt_nonce = str(value["attempt_nonce"])
            stable_id = str(value["stable_id"])
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
        if (job_id, attempt_nonce) in marked_attempts:
            continue
        age = max(0.0, (now - enqueued_at).total_seconds())
        if age <= stall_seconds:
            continue
        claim_path = queue_root / "claimed" / f"{job_id}.json"
        stalls.append(
            QueueStall(job_id, attempt_nonce, stable_id, age, claim_path.is_file())
        )
    return tuple(
        sorted(stalls, key=lambda stall: (stall.job_id, stall.attempt_nonce))
    )


def validate_supervised_campaign(
    repo_root: Path,
    campaign_id: str,
    persisted: CampaignConfig,
) -> None:
    """Validate checkpoint policy before launchd is allowed to run a campaign.

    Parameters
    ----------
    repo_root, campaign_id, persisted:
        Repository, requested campaign label, and mode-0600 campaign config.

    Raises
    ------
    SupervisorError
        If a production binding could lose C1's review gate, impose it on C2-C4,
        or reach the checkpoint without 900/950/1000 outbox identities.
    """

    snapshot = load_intake_snapshot(persisted.intake_root)
    binding = find_campaign_binding(repo_root / DEFAULT_CAMPAIGN_MANIFEST, snapshot)
    if binding is None:
        return
    if binding.spec.campaign_id != campaign_id:
        raise SupervisorError(
            f"campaign config belongs to {binding.spec.campaign_id}, not {campaign_id}"
        )
    if persisted.review_checkpoint_at != binding.spec.review_checkpoint_at:
        raise SupervisorError(
            f"{campaign_id} review checkpoint must be {binding.spec.review_checkpoint_at}"
        )
    if campaign_id == "c1-mech" and not {900, 950, 1000}.issubset(
        persisted.progress_milestones
    ):
        raise SupervisorError("c1-mech must notify at 900, 950, and 1000")


def render_launchd_plist(
    *,
    campaign_id: str,
    repo_root: Path,
    campaign_config_path: Path,
    author_queue_root: Optional[Path],
    python_executable: Path,
) -> bytes:
    """Render one concrete per-campaign launchd agent.

    Parameters
    ----------
    campaign_id, repo_root, campaign_config_path, author_queue_root, python_executable:
        Exact launch inputs embedded as argv leaves without shell interpolation.

    Returns
    -------
    bytes
        Valid XML plist with unsuccessful-exit keep-alive for supervisor death only.
    """

    script = repo_root / "tools" / "crawler_supervisor.sh"
    arguments = [
        str(script),
        "--python",
        str(python_executable),
        "run",
        "--campaign-id",
        campaign_id,
        "--repo-root",
        str(repo_root),
        "--campaign-config",
        str(campaign_config_path),
    ]
    if author_queue_root is not None:
        arguments.extend(("--author-queue", str(author_queue_root)))
    payload = {
        "Label": f"org.torchlens.menagerie-crawler.{campaign_id}",
        "ProgramArguments": arguments,
        "WorkingDirectory": str(repo_root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ProcessType": "Background",
        "ThrottleInterval": 30,
        "StandardOutPath": str(repo_root / ".crawl-local" / "supervisor" / "launchd.out.log"),
        "StandardErrorPath": str(repo_root / ".crawl-local" / "supervisor" / "launchd.err.log"),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the supervisor and launchd-rendering CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(prog="python -m menagerie.crawler.supervisor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "render-launchd"):
        command = subparsers.add_parser(name)
        command.add_argument("--campaign-id", required=True)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--campaign-config", type=Path, required=True)
        command.add_argument("--author-queue", type=Path)
    run = subparsers.choices["run"]
    run.add_argument("--wake-episode-id")
    render = subparsers.choices["render-launchd"]
    render.add_argument("--python", type=Path, default=Path(sys.executable))
    render.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the keep-alive or render a concrete launchd plist.

    Parameters
    ----------
    argv:
        Optional CLI arguments.

    Returns
    -------
    int
        Stable supervisor command status.
    """

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    repo_root = args.repo_root.resolve()
    config_path = args.campaign_config.resolve()
    persisted = load_campaign_config(config_path)
    author_queue = (
        args.author_queue.resolve()
        if isinstance(args.author_queue, Path)
        else persisted.author_queue_root
    )
    validate_supervised_campaign(repo_root, args.campaign_id, persisted)
    if args.command == "render-launchd":
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_replace_bytes(
            output,
            render_launchd_plist(
                campaign_id=args.campaign_id,
                repo_root=repo_root,
                campaign_config_path=config_path,
                author_queue_root=author_queue,
                python_executable=args.python.resolve(),
            ),
        )
        return EXIT_OK
    runtime_root = repo_root / ".crawl-local" / "supervisor" / args.campaign_id
    notifier = CommandNotifier(
        (
            None
            if persisted.notify_command is None
            else shlex.join(persisted.notify_command)
        ),
        receipt_root=runtime_root / "notification-receipts",
    )
    supervisor = CrawlerSupervisor(
        SupervisorConfig(
            campaign_id=args.campaign_id,
            repo_root=repo_root,
            campaign_config_path=config_path,
            author_queue_root=author_queue,
            runtime_root=runtime_root,
            wake_episode_id=args.wake_episode_id,
        ),
        notifier,
    )
    return supervisor.run()


def _driver_command(config: SupervisorConfig) -> tuple[str, ...]:
    """Return the clean-environment driver resume argv."""

    command = [
        sys.executable,
        "-m",
        "menagerie.crawler",
        "--repo-root",
        str(config.repo_root),
        "resume",
        "--campaign-config",
        str(config.campaign_config_path),
    ]
    if config.author_queue_root is not None:
        command.extend(("--author-queue", str(config.author_queue_root)))
    if config.wake_episode_id is not None:
        command.extend(("--wake-episode-id", config.wake_episode_id))
    return tuple(command)


def _start_process(
    command: Sequence[str],
    cwd: Path,
    stdout: BinaryIO,
    stderr: BinaryIO,
) -> ManagedProcess:
    """Start one non-shell driver child.

    Parameters
    ----------
    command, cwd, stdout, stderr:
        Exact argv, working directory, and append-only logs.

    Returns
    -------
    ManagedProcess
        Pollable driver subprocess.
    """

    return subprocess.Popen(
        list(command),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        shell=False,
        start_new_session=False,
    )


def _load_state(path: Path) -> SupervisorState:
    """Load durable supervisor state, defaulting only when absent."""

    if not path.is_file():
        return SupervisorState()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisorError(f"supervisor state is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise SupervisorError("supervisor state must be an object")
    return SupervisorState.from_mapping(value)


def _write_state(path: Path, state: SupervisorState) -> None:
    """Atomically persist supervisor crash and alert state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_replace_bytes(path, canonical_json_bytes(state.to_dict()) + b"\n")


def _driver_state_status(runtime_root: Path) -> Optional[str]:
    """Return the disposable driver status when readable."""

    path = runtime_root / "driver-state.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = value.get("status") if isinstance(value, Mapping) else None
    return status if isinstance(status, str) else None


def _restart_backoff_seconds(crash_count: int) -> float:
    """Return the bounded exponential restart delay."""

    return float(min(MAX_RESTART_BACKOFF_SECONDS, 30 * (2 ** max(0, crash_count - 1))))


def _parse_timestamp(value: str) -> datetime:
    """Parse one required aware RFC 3339 timestamp."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    """Format one aware instant as canonical UTC."""

    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover - operator entry point
    try:
        raise SystemExit(main())
    except (OSError, SupervisorError, ValueError) as exc:
        print(f"crawler supervisor error: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_ERROR) from exc
