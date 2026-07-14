"""Strict, injectable go/no-go preflight for crawler campaigns."""

from __future__ import annotations

import fcntl
import os
import platform
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

from menagerie.crawler.checkpoint import CRAWLER_BRANCH
from menagerie.crawler.policy import (
    ExecutionPolicy,
    PolicyViolation,
    build_safe_environment,
    static_source_check,
)
from menagerie.crawler.wakeup import WakeupConfigurationError, detect_wakeup_backend

GIB = 1024**3


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
        """Return available Claude author research tools."""

        ...

    def secret_findings(self) -> tuple[str, ...]:
        """Return repository secret-scan findings."""

        ...

    def policy_tripwires(self) -> Mapping[str, bool]:
        """Return pass/fail observations for offline, socket, and write auditing."""

        ...

    def wakeup_available(self) -> bool:
        """Return whether a ruled one-shot wakeup backend exists."""

        ...

    def torchlens_import_violations(self) -> tuple[str, ...]:
        """Return execution source paths violating the static TorchLens ban."""

        ...


CommandRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


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
        """Inspect Claude Code help/MCP output for WebSearch and both Exa tools."""

        if shutil.which("claude") is None:
            return frozenset()
        help_result = self._run(["claude", "--help"], self.config.repo_root)
        mcp_result = self._run(["claude", "mcp", "list"], self.config.repo_root)
        combined = (
            f"{help_result.stdout}\n{help_result.stderr}\n{mcp_result.stdout}\n{mcp_result.stderr}"
        )
        return frozenset(
            tool
            for tool in ("WebSearch", "web_search_exa", "web_fetch_exa")
            if tool.lower() in combined.lower()
        )

    def secret_findings(self) -> tuple[str, ...]:
        """Scan first-party crawler files for credential material."""

        private_key_marker = b"-----BEGIN " + b"PRIVATE KEY-----"
        rsa_private_key_marker = b"-----BEGIN RSA " + b"PRIVATE KEY-----"
        markers = (
            private_key_marker,
            rsa_private_key_marker,
            b"OPENAI_API_KEY=",
            b"ANTHROPIC_API_KEY=",
            b"AWS_SECRET_ACCESS_KEY=",
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
        """Feature-detect a ruled one-shot scheduler backend."""

        try:
            detect_wakeup_backend()
        except WakeupConfigurationError:
            return False
        return True

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
    _record(checks, failures, "mirrors", active.mirrors_reachable(), "mirror reachability failed")

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
    _record(checks, failures, "wakeup", active.wakeup_available(), "no wakeup mechanism available")
    violations = active.torchlens_import_violations()
    _record(
        checks,
        failures,
        "torchlens-import-ban",
        not violations,
        f"violations={list(violations)}",
    )
    if failures:
        raise DoctorError(failures)
    return DoctorReport(config.target, checks, True)


def _record(
    checks: dict[str, str], failures: list[str], name: str, passed: bool, detail: str
) -> None:
    """Record one doctor result and retain a loud failure detail."""

    checks[name] = "pass" if passed else f"fail: {detail}"
    if not passed:
        failures.append(f"{name}: {detail}")


def _run_command(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one read-only doctor command without a shell."""

    return subprocess.run(list(argv), cwd=cwd, check=False, capture_output=True, text=True)
