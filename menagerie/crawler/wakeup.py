"""Usage-limit pause events and idempotent reset-time one-shot wakeups."""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import AppendResult, JsonObject
from menagerie.crawler.recordio import JsonlLedger


class WakeupBackend(str, Enum):
    """Supported one-shot OS wakeup mechanisms."""

    LAUNCHD = "launchd"
    SYSTEMD_TIMER = "systemd-timer"
    CRON = "cron"


class WakeupError(RuntimeError):
    """Base class for wakeup configuration failures."""


class WakeupConfigurationError(WakeupError):
    """Raised when an exact reset-time wakeup cannot be represented."""


@dataclass(frozen=True)
class OperationalContext:
    """Common immutable fields carried by every operational event."""

    run_id: str
    machine_id: str
    queued_work_counts: Mapping[str, int]
    current_environment: Optional[str]

    def __post_init__(self) -> None:
        """Validate common event context.

        Raises
        ------
        ValueError
            If identity or queue counts are invalid.
        """

        if not self.run_id.strip() or not self.machine_id.strip():
            raise ValueError("run_id and machine_id must be non-empty")
        if any(not key.strip() or value < 0 for key, value in self.queued_work_counts.items()):
            raise ValueError("queued work counts require non-empty keys and non-negative values")


@dataclass(frozen=True)
class WakeupSpec:
    """Persisted one-shot scheduler definition."""

    wakeup_id: str
    backend: WakeupBackend
    reset_at: str
    callback_argv: tuple[str, ...]
    definition_path: Path
    scheduler_definition: str


@dataclass(frozen=True)
class WakeupScheduleResult:
    """Result of idempotently recording and configuring a reset wakeup."""

    spec: WakeupSpec
    created: bool
    pause_event: Optional[AppendResult]
    wakeup_event: Optional[AppendResult]


CommandExists = Callable[[str], bool]
WakeupActivator = Callable[[WakeupSpec], None]


def detect_wakeup_backend(
    *,
    platform_name: Optional[str] = None,
    command_exists: Optional[CommandExists] = None,
) -> WakeupBackend:
    """Feature-detect launchd, systemd timer, or cron.

    Parameters
    ----------
    platform_name:
        Optional test override for ``platform.system()``.
    command_exists:
        Optional command-presence probe.

    Returns
    -------
    WakeupBackend
        launchd on macOS, systemd when available elsewhere, otherwise cron.

    Raises
    ------
    WakeupConfigurationError
        If no supported scheduler exists.
    """

    system = (platform_name or platform.system()).lower()
    exists = command_exists or (lambda name: shutil.which(name) is not None)
    if system == "darwin":
        if not exists("launchctl"):
            raise WakeupConfigurationError("macOS wakeups require launchctl")
        return WakeupBackend.LAUNCHD
    if exists("systemctl"):
        return WakeupBackend.SYSTEMD_TIMER
    if exists("crontab"):
        return WakeupBackend.CRON
    raise WakeupConfigurationError("no launchd, systemd, or cron wakeup mechanism found")


def build_usage_pause_event(
    *,
    provider: str,
    observed_response: str,
    reset_at: str,
    context: OperationalContext,
    created_at: str,
) -> JsonObject:
    """Construct one strict usage-limit pause operational event.

    Parameters
    ----------
    provider, observed_response, reset_at:
        Exact provider limit observation and reset time.
    context:
        Common campaign context.
    created_at:
        Exact event timestamp.

    Returns
    -------
    dict[str, Any]
        Logical event payload ready for ``JsonlLedger.append``.
    """

    if provider not in {"anthropic", "openai"}:
        raise ValueError(f"unsupported usage-limit provider: {provider!r}")
    _parse_utc(reset_at)
    _parse_utc(created_at)
    identity = stable_hash(
        {
            "event_kind": OperationalEventKind.USAGE_PAUSE.value,
            "provider": provider,
            "reset_at": reset_at,
            "run_id": context.run_id,
            "created_at": created_at,
        }
    )
    return _base_event(
        event_id=f"usage-pause-{identity.removeprefix('sha256:')[:24]}",
        created_at=created_at,
        event_kind=OperationalEventKind.USAGE_PAUSE,
        status=OperationalEventStatus.USAGE_PAUSED,
        context=context,
        provider=provider,
        observed_response=observed_response,
        reset_at=reset_at,
        details={"reset_exact": True},
    )


def build_wakeup_event(
    spec: WakeupSpec,
    *,
    provider: str,
    context: OperationalContext,
    created_at: str,
) -> JsonObject:
    """Construct one strict scheduled-wakeup operational event.

    Parameters
    ----------
    spec:
        Installed one-shot scheduler definition.
    provider:
        Provider whose usage window resets.
    context:
        Common campaign context.
    created_at:
        Exact event timestamp.

    Returns
    -------
    dict[str, Any]
        Logical event payload ready for append.
    """

    _parse_utc(created_at)
    return _base_event(
        event_id=f"wakeup-{spec.wakeup_id}",
        created_at=created_at,
        event_kind=OperationalEventKind.WAKEUP,
        status=OperationalEventStatus.WAKEUP_SCHEDULED,
        context=context,
        provider=provider,
        observed_response=None,
        reset_at=spec.reset_at,
        details={
            "backend": spec.backend.value,
            "definition_path": str(spec.definition_path),
            "one_shot": True,
        },
    )


def record_operational_event(ledger: JsonlLedger, event: Mapping[str, object]) -> AppendResult:
    """Append one operational event through the canonical strict ledger.

    Parameters
    ----------
    ledger:
        Operational-event ledger.
    event:
        Complete logical event without sequence/hash fields.

    Returns
    -------
    AppendResult
        Idempotent append result.
    """

    if ledger.schema_version != OPERATIONAL_EVENT_SCHEMA_VERSION:
        raise ValueError("record_operational_event requires an operational-event ledger")
    return ledger.append(event)


def _record_or_replay(ledger: JsonlLedger, event: Mapping[str, object]) -> AppendResult:
    """Return an existing immutable event identity or append it once.

    Parameters
    ----------
    ledger, event:
        Operational ledger and logical event payload.

    Returns
    -------
    AppendResult
        Existing identity for a wakeup replay, otherwise canonical append result.
    """

    event_id = event.get("event_id")
    for existing in ledger.records:
        if existing.get("event_id") == event_id:
            return AppendResult(existing, appended=False)
    return record_operational_event(ledger, event)


class WakeupManager:
    """Persist and activate one scheduler definition per provider reset time."""

    def __init__(
        self,
        wakeup_root: Path,
        ledger: JsonlLedger,
        callback_argv: Sequence[str],
        *,
        backend: Optional[WakeupBackend] = None,
        platform_name: Optional[str] = None,
        command_exists: Optional[CommandExists] = None,
        activator: Optional[WakeupActivator] = None,
    ) -> None:
        """Initialize an idempotent one-shot wakeup registry.

        Parameters
        ----------
        wakeup_root:
            Gitignored runtime directory for scheduler definitions.
        ledger:
            Append-only operational-event ledger.
        callback_argv:
            Exact argv invoked at reset time.
        backend, platform_name, command_exists:
            Optional explicit backend or feature-detection controls.
        activator:
            Optional OS integration callback. Definitions are durably written
            before this callback executes.
        """

        if not callback_argv or any(not argument for argument in callback_argv):
            raise WakeupConfigurationError("callback argv must contain non-empty arguments")
        self.root = wakeup_root
        self.ledger = ledger
        self.callback_argv = tuple(callback_argv)
        self.backend = backend or detect_wakeup_backend(
            platform_name=platform_name, command_exists=command_exists
        )
        self.activator = activator or _activate_wakeup

    def record_pause_and_schedule(
        self,
        *,
        provider: str,
        observed_response: str,
        reset_at: str,
        context: OperationalContext,
        created_at: str,
    ) -> WakeupScheduleResult:
        """Record a pause and create exactly one reset-time wakeup.

        Repeating the same provider/reset/callback identity reads the existing
        verified definition and performs no append or activation.

        Parameters
        ----------
        provider, observed_response, reset_at:
            Exact usage-limit observation.
        context:
            Common event context.
        created_at:
            Exact pause/scheduling timestamp.

        Returns
        -------
        WakeupScheduleResult
            New or replayed scheduler definition and event append results.
        """

        _parse_utc(reset_at)
        identity = stable_hash(
            {
                "provider": provider,
                "reset_at": reset_at,
                "callback_argv": list(self.callback_argv),
            }
        ).removeprefix("sha256:")
        wakeup_id = identity[:32]
        path = self.root / f"{wakeup_id}.{self._suffix()}"
        scheduler_definition = self._render_definition(wakeup_id, reset_at)
        spec = WakeupSpec(
            wakeup_id=wakeup_id,
            backend=self.backend,
            reset_at=reset_at,
            callback_argv=self.callback_argv,
            definition_path=path,
            scheduler_definition=scheduler_definition,
        )
        pause = _record_or_replay(
            self.ledger,
            build_usage_pause_event(
                provider=provider,
                observed_response=observed_response,
                reset_at=reset_at,
                context=context,
                created_at=created_at,
            ),
        )
        created = not path.exists()
        if path.exists():
            if path.read_text(encoding="utf-8") != scheduler_definition:
                raise WakeupConfigurationError(f"conflicting wakeup definition: {path}")
        else:
            self._write_definition(spec)
        active_marker = path.with_suffix(f"{path.suffix}.active")
        if not active_marker.exists():
            self.activator(spec)
            self._write_text_atomic(active_marker, spec.wakeup_id + "\n")
        wakeup = _record_or_replay(
            self.ledger,
            build_wakeup_event(spec, provider=provider, context=context, created_at=created_at),
        )
        return WakeupScheduleResult(spec, created, pause, wakeup)

    def _suffix(self) -> str:
        """Return the scheduler-definition suffix.

        Returns
        -------
        str
            Backend-specific suffix.
        """

        return {
            WakeupBackend.LAUNCHD: "plist",
            WakeupBackend.SYSTEMD_TIMER: "timer",
            WakeupBackend.CRON: "cron",
        }[self.backend]

    def _render_definition(self, wakeup_id: str, reset_at: str) -> str:
        """Render a deterministic backend-specific one-shot definition.

        Parameters
        ----------
        wakeup_id, reset_at:
            Scheduler identity and exact UTC reset time.

        Returns
        -------
        str
            Complete scheduler definition.
        """

        instant = _parse_utc(reset_at)
        argv_json = json.dumps(list(self.callback_argv), separators=(",", ":"))
        if self.backend is WakeupBackend.LAUNCHD:
            instant = instant.astimezone()
            args = "".join(f"<string>{_xml_escape(arg)}</string>" for arg in self.callback_argv)
            return (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                '<plist version="1.0"><dict>'
                f"<key>Label</key><string>org.torchlens.crawler.{wakeup_id}</string>"
                f"<key>ProgramArguments</key><array>{args}</array>"
                "<key>StartCalendarInterval</key><dict>"
                f"<key>Year</key><integer>{instant.year}</integer>"
                f"<key>Month</key><integer>{instant.month}</integer>"
                f"<key>Day</key><integer>{instant.day}</integer>"
                f"<key>Hour</key><integer>{instant.hour}</integer>"
                f"<key>Minute</key><integer>{instant.minute}</integer>"
                "</dict><key>RunAtLoad</key><false/></dict></plist>\n"
            )
        if self.backend is WakeupBackend.SYSTEMD_TIMER:
            calendar = reset_at.removesuffix("Z").replace("T", " ") + " UTC"
            return (
                "[Unit]\nDescription=TorchLens crawler one-shot wakeup\n"
                "[Timer]\n"
                f"OnCalendar={calendar}\nPersistent=true\n"
                f"Unit={wakeup_id}.service\n"
                "[Install]\nWantedBy=timers.target\n"
                f"# callback_argv={argv_json}\n"
            )
        command = " ".join(shlex.quote(argument) for argument in self.callback_argv)
        year_guard = f'[ "$(date -u +\\%Y)" = "{instant.year}" ] && {command}'
        return (
            f"# one-shot {wakeup_id} at {reset_at}\n"
            "CRON_TZ=UTC\n"
            f"{instant.minute} {instant.hour} {instant.day} {instant.month} * {year_guard}\n"
        )

    def _write_definition(self, spec: WakeupSpec) -> None:
        """Atomically persist and fsync a scheduler definition.

        Parameters
        ----------
        spec:
            Complete wakeup definition.
        """

        spec.definition_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = spec.definition_path.with_name(
            f".{spec.definition_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(spec.scheduler_definition)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, spec.definition_path)
        finally:
            temporary.unlink(missing_ok=True)
        if spec.backend is WakeupBackend.SYSTEMD_TIMER:
            service_path = spec.definition_path.with_suffix(".service")
            command = " ".join(shlex.quote(argument) for argument in spec.callback_argv)
            service = (
                "[Unit]\nDescription=TorchLens crawler usage-reset resume\n"
                "[Service]\nType=oneshot\n"
                f"ExecStart={command}\n"
            )
            self._write_text_atomic(service_path, service)

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        """Atomically persist and fsync a small wakeup state file.

        Parameters
        ----------
        path, content:
            Destination and complete text.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _activate_wakeup(spec: WakeupSpec) -> None:
    """Install and start one backend-specific scheduler definition.

    Parameters
    ----------
    spec:
        Durably written scheduler definition.

    Raises
    ------
    WakeupConfigurationError
        If the host scheduler refuses installation.
    """

    if spec.backend is WakeupBackend.LAUNCHD:
        domain = f"gui/{os.getuid()}"
        label = f"org.torchlens.crawler.{spec.wakeup_id}"
        present = subprocess.run(
            ["launchctl", "print", f"{domain}/{label}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if present.returncode == 0:
            return
        _run_checked(["launchctl", "bootstrap", domain, str(spec.definition_path)])
        return
    if spec.backend is WakeupBackend.SYSTEMD_TIMER:
        service_path = spec.definition_path.with_suffix(".service")
        _run_checked(
            [
                "systemctl",
                "--user",
                "link",
                str(service_path),
                str(spec.definition_path),
            ]
        )
        _run_checked(["systemctl", "--user", "daemon-reload"])
        _run_checked(["systemctl", "--user", "start", spec.definition_path.name])
        return
    current = subprocess.run(["crontab", "-l"], check=False, capture_output=True, text=True)
    if current.returncode not in {0, 1}:
        raise WakeupConfigurationError(f"cannot inspect current crontab: {current.stderr.strip()}")
    marker = f"# one-shot {spec.wakeup_id} "
    if marker in current.stdout:
        return
    existing_crontab = current.stdout
    if existing_crontab and not existing_crontab.endswith("\n"):
        existing_crontab += "\n"
    installed = subprocess.run(
        ["crontab", "-"],
        input=existing_crontab + spec.scheduler_definition,
        check=False,
        capture_output=True,
        text=True,
    )
    if installed.returncode != 0:
        raise WakeupConfigurationError(
            f"cannot install one-shot cron wakeup: {installed.stderr.strip()}"
        )


def _run_checked(argv: Sequence[str]) -> None:
    """Run one scheduler argv and raise a typed error on failure.

    Parameters
    ----------
    argv:
        Exact scheduler command.

    Raises
    ------
    WakeupConfigurationError
        If the command exits nonzero.
    """

    completed = subprocess.run(list(argv), check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise WakeupConfigurationError(
            f"scheduler command {list(argv)!r} failed: {completed.stderr.strip()}"
        )


def _base_event(
    *,
    event_id: str,
    created_at: str,
    event_kind: OperationalEventKind,
    status: OperationalEventStatus,
    context: OperationalContext,
    provider: Optional[str],
    observed_response: Optional[str],
    reset_at: Optional[str],
    details: JsonObject,
) -> JsonObject:
    """Build the strict common operational-event envelope.

    Parameters
    ----------
    event_id, created_at, event_kind, status, context:
        Mandatory event identity and campaign fields.
    provider, observed_response, reset_at:
        Optional provider-limit fields.
    details:
        Structured type-specific details.

    Returns
    -------
    dict[str, Any]
        Logical operational event.
    """

    return {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "created_at": created_at,
        "event_kind": event_kind.value,
        "status": status.value,
        "provider": provider,
        "observed_response": observed_response,
        "reset_at": reset_at,
        "queued_work_counts": dict(context.queued_work_counts),
        "current_environment": context.current_environment,
        "run_id": context.run_id,
        "machine_id": context.machine_id,
        "details": details,
    }


def _parse_utc(value: str) -> datetime:
    """Parse an exact RFC 3339 UTC timestamp.

    Parameters
    ----------
    value:
        Timestamp ending in ``Z``.

    Returns
    -------
    datetime
        Aware parsed instant.

    Raises
    ------
    WakeupConfigurationError
        If the timestamp is not exact UTC date-time syntax.
    """

    if not value.endswith("Z"):
        raise WakeupConfigurationError("wakeup timestamps must be RFC 3339 UTC ending in Z")
    try:
        return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise WakeupConfigurationError(f"invalid wakeup timestamp {value!r}") from exc


def _xml_escape(value: str) -> str:
    """Escape an argv leaf for a launchd plist.

    Parameters
    ----------
    value:
        Raw argv string.

    Returns
    -------
    str
        XML-safe value.
    """

    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
