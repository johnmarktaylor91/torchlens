"""Progress reporting, notification, and durable driver state I/O."""

from __future__ import annotations
import json
import logging
import os
import shlex
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional
from menagerie.crawler.author_dispatch import (
    AuthorBackoffSignal,
)
from menagerie.crawler.checker_dispatch import (
    CheckerBackoffSignal,
)
from menagerie.crawler.checkpoint import (
    FunnelSnapshot,
)
from menagerie.crawler.constants import (
    DEFAULT_NOTIFY_TIMEOUT_SECONDS,
)
from menagerie.crawler.env_lifecycle import (
    DiskRecoveryError,
    EnvironmentProbeError,
    EnvironmentSolveError,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
)
from menagerie.crawler.intake import (
    IntakeItem,
    legacy_requires_fidelity_audit,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.status import (
    funnel_counts,
)
from menagerie.crawler.driver_contracts import (
    DriverIntegrationError,
)

LOGGER = logging.getLogger("menagerie.crawler.driver")


@dataclass(frozen=True)
class _EnvironmentFailureTransition:
    """Terminal stage and reason selected for one lifecycle exception type."""

    exception_type: type[Exception]
    stage: str
    reason_code: str


_ENVIRONMENT_FAILURE_TRANSITIONS: tuple[_EnvironmentFailureTransition, ...] = (
    _EnvironmentFailureTransition(EnvironmentProbeError, "environment", "probe-failed"),
    _EnvironmentFailureTransition(EnvironmentSolveError, "environment", "solve-failed"),
    _EnvironmentFailureTransition(DiskRecoveryError, "resource", "disk-floor"),
    _EnvironmentFailureTransition(Exception, "environment", "build-failed"),
)
if tuple(row.exception_type for row in _ENVIRONMENT_FAILURE_TRANSITIONS) != (
    EnvironmentProbeError,
    EnvironmentSolveError,
    DiskRecoveryError,
    Exception,
):
    raise RuntimeError("environment failure transition table is not exhaustive")


class CommandNotifier:
    """Best-effort argv-only notifier with log-only fallback."""

    def __init__(
        self,
        command: Optional[str],
        *,
        timeout_seconds: float = DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    ) -> None:
        """Resolve an explicit command or the conventional JMT script."""

        if timeout_seconds <= 0:
            raise ValueError("notifier timeout must be positive")
        self._argv = _resolve_notify_command(command)
        self._timeout_seconds = timeout_seconds

    @property
    def command(self) -> Optional[tuple[str, ...]]:
        """Return the resolved notifier command.

        Returns
        -------
        tuple[str, ...] | None
            Resolved argv, or ``None`` for log-only notification.
        """

        return self._argv

    def notify(self, summary: str, *, idempotency_key: str) -> bool:
        """Invoke the notifier once, logging and continuing on any failure."""

        ascii_summary = _ascii_line(summary)
        if self._argv is None:
            LOGGER.warning("crawler notification (log-only): %s", ascii_summary)
            return False
        try:
            completed = subprocess.run(
                [*self._argv, ascii_summary],
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                env={**os.environ, "MENAGERIE_NOTIFICATION_IDEMPOTENCY_KEY": idempotency_key},
            )
        except subprocess.TimeoutExpired:
            LOGGER.warning(
                "crawler notifier timed out after %.1fs: %s",
                self._timeout_seconds,
                ascii_summary,
            )
            return False
        except Exception as exc:  # noqa: BLE001 -- notifications are strictly best-effort
            LOGGER.warning("crawler notifier failed (%s): %s", exc, ascii_summary)
            return False
        if completed.returncode != 0:
            LOGGER.warning(
                "crawler notifier exited %s (%s): %s",
                completed.returncode,
                completed.stderr.strip(),
                ascii_summary,
            )
            return False
        return True


def _funnel_snapshot(current: Mapping[str, Mapping[str, Any]]) -> FunnelSnapshot:
    """Collapse terminal status codes into the four review buckets."""

    counts: Counter[str] = Counter()
    for record in current.values():
        counts[str(record["status"]["kind"])] += 1
    return FunnelSnapshot(
        runs=counts["runs"],
        deferred=counts["deferred"],
        skipped=counts["skipped"],
        failed=counts["failed"],
    )


def _completion_workflows(current: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Return pending workflow gates not expressible by terminal partition shape."""

    workflows: list[str] = []
    for record in current.values():
        flags = set(record.get("flags", [])) | set(
            record.get("intake", {}).get("preserved_legacy_flags", [])
        )
        fidelity = record.get("fidelity", {})
        if legacy_requires_fidelity_audit(flags) and (
            not fidelity.get("gate_id")
            or fidelity.get("verdict")
            not in {"match", "minor-drift", "major-drift", "slop", "cannot-verify"}
        ):
            workflows.append("fidelity-pending")
    return tuple(workflows)


def _review_report(current: Mapping[str, JsonObject], threshold: int) -> JsonObject:
    """Build the blocking checkpoint report required for human review."""

    snapshot = _funnel_snapshot(current)
    fidelity = Counter(
        str(record.get("fidelity", {}).get("verdict") or "not-required")
        for record in current.values()
    )
    accepted = [
        record
        for _stable_id, record in sorted(current.items())
        if record.get("status", {}).get("kind") == "runs"
    ][:5]
    total = max(1, len(current))
    skip_rate = snapshot.skipped / total
    concerns: list[str] = []
    if skip_rate >= 0.20:
        concerns.append(f"skip-rate spike: {skip_rate:.1%}")
    if snapshot.failed / total >= 0.20:
        concerns.append(f"failure-rate spike: {snapshot.failed / total:.1%}")
    return {
        "report_kind": "menagerie-crawler-review-checkpoint-v1",
        "review_checkpoint_at": threshold,
        "models_completed": len(current),
        "funnel_snapshot": snapshot.to_dict(),
        "funnel_counts": dict(funnel_counts(current)),
        "fidelity_verdict_distribution": dict(sorted(fidelity.items())),
        "accepted_sample": accepted,
        "concerning_patterns": concerns,
    }


def _progress_summary(completed: int, milestone: int, snapshot: FunnelSnapshot) -> str:
    """Return one concise ASCII progress notification."""

    return _ascii_line(
        f"Menagerie crawler milestone {milestone}: completed={completed} "
        f"runs={snapshot.runs} deferred={snapshot.deferred} skipped={snapshot.skipped} "
        f"failed={snapshot.failed}"
    )


def _review_summary(completed: int, snapshot: FunnelSnapshot, report_path: Path) -> str:
    """Return one concise ASCII blocking-review notification."""

    return _ascii_line(
        f"Menagerie crawler review checkpoint: completed={completed} runs={snapshot.runs} "
        f"deferred={snapshot.deferred} skipped={snapshot.skipped} failed={snapshot.failed} "
        f"report={report_path}"
    )


def _ascii_line(value: str) -> str:
    """Normalize arbitrary text into one plain-ASCII line."""

    return " ".join(value.encode("ascii", errors="replace").decode("ascii").split())


def _resolve_notify_command(command: Optional[str]) -> Optional[tuple[str, ...]]:
    """Resolve an explicit notifier or the conventional home/PATH script."""

    if command:
        parsed = tuple(shlex.split(command))
        return parsed or None
    found = shutil.which("send-to-jmt.sh")
    if found is not None:
        return (found,)
    for candidate in (
        Path.home() / "scripts" / "send-to-jmt.sh",
        Path.home() / "bin" / "send-to-jmt.sh",
        Path.home() / ".claude" / "scripts" / "send-to-jmt.sh",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return (str(candidate),)
    return None


def _framework_from_intake(item: IntakeItem) -> str:
    """Infer only the closed native-tail markers; otherwise route to PyTorch."""

    text = f"{item.zoo} {item.name}".lower()
    for marker in ("tensorflow", "keras", "jax", "flax", "paddle"):
        if marker in text:
            return marker
    return "pytorch"


def _framework_phase(item: IntakeItem) -> str:
    """Return the configured scheduler phase for one intake row."""

    return "pytorch" if _framework_from_intake(item) == "pytorch" else "native-tail"


def _environment_failure(exc: Exception) -> tuple[str, str]:
    """Map a typed environment-lifecycle exception to its closed failure reason."""

    # Sandbox admission is identified by both type name and defining module. It stays
    # explicit because reducing that precondition to an exception-class row would weaken it.
    if _is_sandbox_unavailable(exc):
        return "policy", "sandbox-unavailable-v1"
    for transition in _ENVIRONMENT_FAILURE_TRANSITIONS:
        if isinstance(exc, transition.exception_type):
            return transition.stage, transition.reason_code
    raise AssertionError("exhaustive environment failure transition table did not match")


def _is_sandbox_unavailable(exc: Exception) -> bool:
    """Recognize the supervisor's typed fail-closed sandbox signal."""

    return (
        type(exc).__name__ == "SandboxUnavailableError"
        and type(exc).__module__ == "menagerie.crawler.policy"
    )


def _future_reset(now: str, signal: CheckerBackoffSignal | AuthorBackoffSignal) -> str:
    """Compute an exact reset timestamp when the provider supplied only a delay."""

    instant = datetime.fromisoformat(now.removesuffix("Z") + "+00:00")
    seconds = signal.retry_after_seconds if signal.retry_after_seconds is not None else 3600
    return (instant + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _load_driver_state(path: Path) -> JsonObject:
    """Load the disposable cursor, treating absence as a fresh campaign."""

    if not path.is_file():
        return {"status": "new"}
    return _read_json(path)


def _write_driver_state(path: Path, state: Mapping[str, Any]) -> None:
    """Atomically persist the disposable scheduler cursor."""

    _write_json_atomic(path, state)


def _read_json(path: Path) -> JsonObject:
    """Read one JSON object or raise a typed integration error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriverIntegrationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DriverIntegrationError(f"expected a JSON object at {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically fsync one deterministic JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _boot_id() -> str:
    """Return the kernel boot identity when available."""

    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"
