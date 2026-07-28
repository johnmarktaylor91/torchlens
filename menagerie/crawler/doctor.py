"""Strict, injectable go/no-go preflight for crawler campaigns."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import secrets
import shlex
import shutil
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from menagerie.crawler.checkpoint import CRAWLER_BRANCH
from menagerie.crawler.driver_progress import _resolve_notify_command
from menagerie.crawler.execution_lock import global_execution_flock_path
from menagerie.crawler.executable_paths import normalize_executable
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.policy import (
    ExecutionPolicy,
    PolicyViolation,
    build_safe_environment,
    static_source_check,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.wakeup import (
    WakeupConfigurationError,
    detect_wakeup_backend,
    inspect_wakeup_definitions,
)

GIB = 1024**3



#: Freshness bound for the author capability probe. The probe demands three live web
#: tool calls, a >=800 char document capture, a sha256 over it, and timestamp evidence.
#: A measured honest run takes ~3-4 minutes, so the original 120s could only be met by a
#: response that skipped the work. The unfakeable part of this check is the nonce binding,
#: the digest verification, and the three-way agreement on an unpredictable live value --
#: this bound exists only to stop a stale receipt being replayed, which it still does.
#: Default timeout for a doctor probe that shells out to a fast, synchronous tool.
DEFAULT_PROBE_TIMEOUT_SECONDS = 180
AUTHOR_CAPABILITY_PROBE_SECONDS = 900

class DoctorError(RuntimeError):
    """Raised when one or more strict preflight checks fail."""

    def __init__(self, failures: Sequence[str]) -> None:
        """Attach every failed check to one loud exception."""

        self.failures = tuple(failures)
        super().__init__("doctor failed: " + "; ".join(self.failures))


@dataclass(frozen=True)
class DoctorConfig:
    """Target and local policy required by the strict preflight."""

    repo_root: Path
    runtime_root: Path
    target: str
    expected_branch: str = CRAWLER_BRANCH
    minimum_disk_bytes: int = 100 * GIB
    strict: bool = True

    def __post_init__(self) -> None:
        """Reject unknown targets and negative capacity requirements."""

        if self.target not in {"osx-arm64", "linux-x86_64-cuda"}:
            raise ValueError("unsupported crawler target")
        if self.minimum_disk_bytes < 0:
            raise ValueError("minimum disk bytes cannot be negative")


@dataclass(frozen=True)
class DoctorReport:
    """Complete strict preflight result."""

    target: str
    checks: Mapping[str, str]
    passed: bool


class DoctorProbes(Protocol):
    """Injectable host/environment observations used by the doctor."""

    def machine(self) -> tuple[str, str]:
        """Return operating system and architecture."""

        ...

    def branch(self) -> str:
        """Return the current Git branch."""

        ...

    def disk_free_bytes(self) -> int:
        """Return free bytes on the campaign filesystem."""

        ...

    def lock_available(self) -> bool:
        """Return whether the driver kernel lock can be acquired."""

        ...

    def mirrors_reachable(self) -> bool:
        """Return whether configured public/private mirrors are reachable."""

        ...

    def author_tools(self) -> frozenset[str]:
        """Exercise and attest Claude author research tools."""

        ...

    def wrapper_versions(self) -> Mapping[str, str]:
        """Return resolved version receipts for all three operator wrappers."""

        ...

    def codex_ready(self) -> bool:
        """Return whether Codex auth, pinned model, and high effort execute."""

        ...

    def environment_tool_versions(self) -> Mapping[str, str]:
        """Return version receipts for conda, mamba, and conda-lock."""

        ...

    def notifier_delivery(self) -> bool:
        """Deliver a nonce and validate its bound receipt."""

        ...

    def worker_slot_available(self) -> bool:
        """Return whether the global worker-slot lock is available."""

        ...

    def dynamic_disk_reserve_bytes(self) -> int:
        """Return free bytes after accounting for current runtime occupancy."""

        ...

    def secret_findings(self) -> tuple[str, ...]:
        """Return repository secret-scan findings."""

        ...

    def policy_tripwires(self) -> Mapping[str, bool]:
        """Return pass/fail observations for offline, socket, and write auditing."""

        ...

    def wakeup_available(self) -> bool:
        """Return whether a recurring backend and all active projections are healthy."""

        ...

    def torchlens_import_violations(self) -> tuple[str, ...]:
        """Return execution source paths violating the static TorchLens ban."""

        ...


class CommandRunner(Protocol):
    """Argv-only command runner with an optional per-probe timeout.

    Most doctor probes shell out to fast synchronous tools and take the default
    timeout. The author capability probe blocks on a live author session doing real
    web research, so it must be able to ask for a longer window.
    """

    def __call__(
        self,
        argv: Sequence[str],
        cwd: Path,
        timeout: float = ...,
    ) -> subprocess.CompletedProcess[str]:
        ...


class SystemDoctorProbes:
    """Read-only real-host probes used by the CLI doctor command."""

    def __init__(
        self,
        config: DoctorConfig,
        *,
        command_runner: CommandRunner | None = None,
    ) -> None:
        """Bind the target paths and injectable argv-only command runner."""

        self.config = config
        self._run = command_runner or _run_command

    def machine(self) -> tuple[str, str]:
        """Return the real host system and normalized machine architecture."""

        return (platform.system(), platform.machine())

    def branch(self) -> str:
        """Return the exact current Git branch."""

        result = self._run(["git", "branch", "--show-current"], self.config.repo_root)
        return result.stdout.strip() if result.returncode == 0 else ""

    def disk_free_bytes(self) -> int:
        """Return available campaign filesystem capacity."""

        self.config.runtime_root.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self.config.runtime_root).free

    def lock_available(self) -> bool:
        """Probe the same nonblocking kernel lock used by the driver."""

        path = self.config.runtime_root / "locks" / "driver.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        return True

    def mirrors_reachable(self) -> bool:
        """Require all configured mirror roots to exist and be readable."""

        values = [
            os.environ.get("MENAGERIE_PUBLIC_MIRROR"),
            os.environ.get("MENAGERIE_PRIVATE_MIRROR"),
        ]
        if any(value is None for value in values):
            return False
        return all(Path(str(value)).exists() and os.access(str(value), os.R_OK) for value in values)

    def author_tools(self) -> frozenset[str]:
        """Exercise all required tools through a nonce-bound author queue request."""

        raw_command = os.environ.get("MENAGERIE_AUTHOR_COMMAND")
        if raw_command is None:
            return frozenset()
        command = tuple(shlex.split(raw_command))
        if not command:
            return frozenset()
        root = self.config.runtime_root / "doctor" / "author-capability"
        root.mkdir(parents=True, exist_ok=True)
        nonce = secrets.token_hex(24)
        requested_at = datetime.now(timezone.utc)
        receipt_path = root / "receipt.json"
        request = {
            "format": "menagerie.crawler.author-capability-probe.v1",
            "nonce": nonce,
            "requested_at": requested_at.isoformat().replace("+00:00", "Z"),
            "deadline_seconds": AUTHOR_CAPABILITY_PROBE_SECONDS,
            "required_tools": ["WebSearch", "web_search_exa", "web_fetch_exa"],
            "required_output_path": str(receipt_path.resolve()),
        }
        request_path = root / "request.json"
        request_path.write_bytes(canonical_json_bytes(request) + b"\n")
        # This probe blocks on a live author session doing genuine web research, so it
        # cannot share the fast-tool default. Allow the same window the request grants.
        completed = self._run(
            [*command, str(request_path)],
            self.config.repo_root,
            timeout=float(AUTHOR_CAPABILITY_PROBE_SECONDS),
        )
        observed_at = datetime.now(timezone.utc)
        if (
            completed.returncode != 0
            or observed_at > requested_at + timedelta(seconds=AUTHOR_CAPABILITY_PROBE_SECONDS)
            or not receipt_path.is_file()
        ):
            return frozenset()
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return frozenset()
        if not isinstance(receipt, Mapping) or receipt.get("nonce") != nonce:
            return frozenset()
        try:
            completed_at = datetime.fromisoformat(
                str(receipt["completed_at"]).removesuffix("Z") + "+00:00"
            )
        except (KeyError, ValueError):
            return frozenset()
        if not requested_at <= completed_at <= requested_at + timedelta(
            seconds=AUTHOR_CAPABILITY_PROBE_SECONDS
        ):
            return frozenset()
        raw_receipts = receipt.get("receipts")
        if not isinstance(raw_receipts, list):
            return frozenset()
        tools = {
            str(value.get("tool"))
            for value in raw_receipts
            if isinstance(value, Mapping)
            and value.get("nonce") == nonce
            and value.get("exercised") is True
            and isinstance(value.get("receipt"), str)
            and bool(str(value["receipt"]).strip())
        }
        return frozenset(tools)

    def wrapper_versions(self) -> Mapping[str, str]:
        """Resolve and execute ``--version`` for all configured wrappers."""

        versions: dict[str, str] = {}
        for lane in ("author", "checker", "environment"):
            raw_command = os.environ.get(f"MENAGERIE_{lane.upper()}_COMMAND")
            if raw_command is None:
                continue
            command = tuple(shlex.split(raw_command))
            if not command:
                continue
            executable = _resolve_executable(command[0], self.config.repo_root)
            if executable is None:
                continue
            completed = self._run([executable, *command[1:], "--version"], self.config.repo_root)
            combined = f"{completed.stdout}\n{completed.stderr}".strip()
            if completed.returncode == 0 and combined:
                versions[lane] = combined[:500]
        return versions

    def codex_ready(self) -> bool:
        """Exercise Codex auth with the pinned metadata model at high effort."""

        if shutil.which("codex") is None:
            return False
        auth = self._run(["codex", "login", "status"], self.config.repo_root)
        if auth.returncode != 0:
            return False
        probe = self._run(
            [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "-C",
                str(self.config.repo_root),
                "-m",
                "gpt-5.6-terra",
                "-c",
                "model_reasoning_effort=high",
                "--json",
                "Reply with the single word ready.",
            ],
            self.config.repo_root,
        )
        combined = f"{probe.stderr}\n{probe.stdout}".lower()
        return (
            probe.returncode == 0
            and "model metadata for" not in combined
            and "model is not supported" not in combined
        )

    def environment_tool_versions(self) -> Mapping[str, str]:
        """Execute required conda-family version probes."""

        versions: dict[str, str] = {}
        for tool in ("conda", "mamba", "conda-lock"):
            if shutil.which(tool) is None:
                continue
            completed = self._run([tool, "--version"], self.config.repo_root)
            combined = f"{completed.stdout}\n{completed.stderr}".strip()
            if completed.returncode == 0 and combined:
                versions[tool] = combined[:500]
        return versions

    def notifier_delivery(self) -> bool:
        """Send a nonce and require a matching machine-readable receipt."""

        command = _resolve_notify_command(None)
        if command is None:
            return False
        root = self.config.runtime_root / "doctor" / "notifier"
        root.mkdir(parents=True, exist_ok=True)
        nonce = secrets.token_hex(24)
        receipt_path = root / "receipt.json"
        completed = subprocess.run(
            [*command, f"Menagerie crawler doctor nonce {nonce}"],
            cwd=self.config.repo_root,
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "MENAGERIE_NOTIFICATION_IDEMPOTENCY_KEY": nonce,
                "MENAGERIE_NOTIFICATION_RECEIPT_PATH": str(receipt_path),
            },
        )
        if completed.returncode != 0 or not receipt_path.is_file():
            return False
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(receipt, Mapping) and receipt.get("nonce") == nonce

    def worker_slot_available(self) -> bool:
        """Probe the cross-campaign worker execution slot."""

        return _lock_available(global_execution_flock_path())

    def dynamic_disk_reserve_bytes(self) -> int:
        """Subtract current crawler runtime occupancy from filesystem free space."""

        occupied = sum(
            path.stat().st_size for path in self.config.runtime_root.rglob("*") if path.is_file()
        )
        return max(0, self.disk_free_bytes() - occupied)

    def secret_findings(self) -> tuple[str, ...]:
        """Scan first-party crawler files for credential material."""

        private_key_marker = b"-----BEGIN " + b"PRIVATE KEY-----"
        rsa_private_key_marker = b"-----BEGIN RSA " + b"PRIVATE KEY-----"
        markers = (
            private_key_marker,
            rsa_private_key_marker,
            b"OPENAI_API" + b"_KEY=",
            b"ANTHROPIC_API" + b"_KEY=",
            b"AWS_SECRET_ACCESS" + b"_KEY=",
        )
        findings: list[str] = []
        root = self.config.repo_root / "menagerie" / "crawler"
        for path in root.rglob("*"):
            if not path.is_file() or "tests" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                findings.append(str(path))
                continue
            if any(marker in data for marker in markers):
                findings.append(str(path))
        return tuple(findings)

    def policy_tripwires(self) -> Mapping[str, bool]:
        """Exercise offline flags, socket blocking, and outside-write auditing."""

        root = self.config.runtime_root / "doctor-policy"
        safe = build_safe_environment(
            root / "safe", base_environment={"PATH": os.environ.get("PATH", "")}
        )
        offline = all(
            safe.get(key) == value
            for key, value in {
                "MENAGERIE_EXECUTION_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }.items()
        )
        socket_ok = False
        write_ok = False
        scratch = root / "scratch"
        outside = root.parent / "doctor-forbidden-write"
        with ExecutionPolicy(scratch) as observation:
            try:
                socket.create_connection(("example.invalid", 443))
            except PolicyViolation as exc:
                socket_ok = exc.reason_code == "network-attempt"
            try:
                with outside.open("w", encoding="utf-8") as handle:
                    handle.write("forbidden")
            except PolicyViolation as exc:
                write_ok = exc.reason_code == "write-outside-scratch"
        socket_ok = socket_ok and observation.network_attempted
        write_ok = write_ok and observation.write_outside_scratch_attempted
        return {"offline": offline, "socket": socket_ok, "write-audit": write_ok}

    def wakeup_available(self) -> bool:
        """Verify a ruled recurring backend and every active episode projection."""

        try:
            backend = detect_wakeup_backend()
            events = scan_jsonl(
                self.config.repo_root
                / "menagerie"
                / "crawler"
                / "records"
                / "operational"
                / "events.jsonl"
            )
            inspection = inspect_wakeup_definitions(
                self.config.runtime_root / "wakeups", events, backend
            )
        except (OSError, WakeupConfigurationError):
            return False
        return inspection.healthy

    def torchlens_import_violations(self) -> tuple[str, ...]:
        """Statically scan every execution adapter/port Python source."""

        violations: list[str] = []
        crawler_root = self.config.repo_root / "menagerie" / "crawler"
        roots = (crawler_root / "adapters", crawler_root / "ports", crawler_root / "patches")
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.py"):
                try:
                    static_source_check(path)
                except PolicyViolation as exc:
                    if exc.reason_code == "torchlens-import":
                        violations.append(str(path))
        return tuple(violations)


def run_doctor(config: DoctorConfig, probes: DoctorProbes | None = None) -> DoctorReport:
    """Run every strict preflight check and fail with all recorded findings."""

    active = probes or SystemDoctorProbes(config)
    checks: dict[str, str] = {}
    failures: list[str] = []

    system, machine = active.machine()
    expected = {
        "osx-arm64": ("darwin", {"arm64", "aarch64"}),
        "linux-x86_64-cuda": ("linux", {"x86_64", "amd64"}),
    }[config.target]
    target_ok = system.lower() == expected[0] and machine.lower() in expected[1]
    _record(checks, failures, "target", target_ok, f"got {system}/{machine}")

    branch = active.branch()
    branch_ok = branch == config.expected_branch and branch not in {"main", "master"}
    _record(checks, failures, "branch", branch_ok, f"got {branch or '<unknown>'}")

    free = active.disk_free_bytes()
    _record(
        checks,
        failures,
        "disk",
        free >= config.minimum_disk_bytes,
        f"{free} bytes free; need {config.minimum_disk_bytes}",
    )
    _record(checks, failures, "lock", active.lock_available(), "driver lock is busy")
    _record(
        checks,
        failures,
        "worker-slot-lock",
        active.worker_slot_available(),
        "global execution slot is busy",
    )
    _record(checks, failures, "mirrors", active.mirrors_reachable(), "mirror reachability failed")
    wrapper_versions = active.wrapper_versions()
    required_wrappers = {"author", "checker", "environment"}
    _record(
        checks,
        failures,
        "wrappers",
        required_wrappers.issubset(wrapper_versions),
        f"missing version receipts {sorted(required_wrappers - set(wrapper_versions))}",
    )
    environment_versions = active.environment_tool_versions()
    required_environment_tools = {"conda", "mamba", "conda-lock"}
    _record(
        checks,
        failures,
        "environment-tools",
        required_environment_tools.issubset(environment_versions),
        f"missing versions {sorted(required_environment_tools - set(environment_versions))}",
    )
    _record(
        checks,
        failures,
        "codex-auth-model-effort",
        active.codex_ready(),
        "Codex auth or pinned high-effort model probe failed",
    )
    _record(
        checks,
        failures,
        "notifier-delivery",
        active.notifier_delivery(),
        "nonce delivery receipt failed",
    )
    dynamic_reserve = active.dynamic_disk_reserve_bytes()
    _record(
        checks,
        failures,
        "dynamic-disk-reserve",
        dynamic_reserve >= config.minimum_disk_bytes,
        f"{dynamic_reserve} bytes dynamically reserved; need {config.minimum_disk_bytes}",
    )

    tools = active.author_tools()
    required_tools = {"WebSearch", "web_search_exa", "web_fetch_exa"}
    _record(
        checks,
        failures,
        "author-web-tools",
        required_tools.issubset(tools),
        f"missing {sorted(required_tools - set(tools))}",
    )

    secrets = active.secret_findings()
    _record(checks, failures, "secrets", not secrets, f"findings={list(secrets)}")
    tripwires = active.policy_tripwires()
    for name in ("offline", "socket", "write-audit"):
        _record(
            checks,
            failures,
            f"policy:{name}",
            tripwires.get(name) is True,
            "tripwire self-test failed",
        )
    _record(
        checks,
        failures,
        "wakeup",
        active.wakeup_available(),
        "recurring backend unavailable or active wake episode needs repair",
    )
    violations = active.torchlens_import_violations()
    _record(
        checks,
        failures,
        "torchlens-import-ban",
        not violations,
        f"violations={list(violations)}",
    )
    if failures and config.strict:
        raise DoctorError(failures)
    return DoctorReport(config.target, checks, not failures)


def _record(
    checks: dict[str, str], failures: list[str], name: str, passed: bool, detail: str
) -> None:
    """Record one doctor result and retain a loud failure detail."""

    checks[name] = "pass" if passed else f"fail: {detail}"
    if not passed:
        failures.append(f"{name}: {detail}")


def _run_command(
    argv: Sequence[str], cwd: Path, timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run one read-only doctor command without a shell."""

    try:
        return subprocess.run(
            list(argv),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            list(argv),
            124,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or "doctor probe timed out"),
        )


def _resolve_executable(value: str, cwd: Path) -> str | None:
    """Resolve one configured executable without a shell.

    Parameters
    ----------
    value, cwd:
        Executable token and repository-relative base.

    Returns
    -------
    str | None
        Absolute executable path when resolvable, with symlinks left intact so a configured
        virtualenv interpreter is not collapsed into its base installation.
    """

    resolved = normalize_executable(value, cwd=cwd)
    return None if resolved is None else str(resolved)


def _lock_available(path: Path) -> bool:
    """Probe one nonblocking exclusive kernel lock.

    Parameters
    ----------
    path:
        Lock file path.

    Returns
    -------
    bool
        Whether the lock can be acquired now.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    return True
