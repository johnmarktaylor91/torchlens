"""Parent-owned argv-only subprocess isolation and resource observations."""

from __future__ import annotations

import json
import os
import re
import resource
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
    STDIO_TAIL_MAX_CHARS,
    FailureStage,
)
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.policy import (
    SandboxUnavailableError,
    build_safe_environment,
    detect_os_sandbox,
    generate_macos_sandbox_profile,
    wrap_with_os_sandbox,
)

_DENIED_ERRNOS = ("EACCES", "ENETDOWN", "ENETUNREACH", "EPERM", "EROFS")
_WRITE_OPEN_FLAGS = ("O_APPEND", "O_CREAT", "O_RDWR", "O_TMPFILE", "O_TRUNC", "O_WRONLY")
_WRITE_SYSCALLS = frozenset(
    {
        "chmod",
        "chown",
        "creat",
        "fchmodat",
        "fchownat",
        "link",
        "linkat",
        "mkdir",
        "mkdirat",
        "mknod",
        "mknodat",
        "open",
        "openat",
        "openat2",
        "rename",
        "renameat",
        "renameat2",
        "rmdir",
        "symlink",
        "symlinkat",
        "truncate",
        "unlink",
        "unlinkat",
        "utime",
        "utimensat",
        "utimes",
    }
)
_SYSCALL_PATTERN = re.compile(r"(?:^|\s)([a-z][a-z0-9_]*)\(")
_QUOTED_PATH_PATTERN = re.compile(r'"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class SupervisorObservation:
    """Facts observed exclusively by the parent process.

    Parameters
    ----------
    argv, cwd:
        Exact non-shell invocation.
    exit_code, signal_number:
        Process result observed by the parent.
    wall_seconds, cpu_seconds, peak_rss_bytes:
        Parent-measured resource facts.
    timed_out, rss_exceeded:
        Resource enforcement outcomes.
    stdout/stderr fields:
        Hashes, sizes, bounded tails, and local paths.
    """

    argv: tuple[str, ...]
    cwd: str
    exit_code: Optional[int]
    signal_number: Optional[int]
    wall_seconds: float
    cpu_seconds: float
    peak_rss_bytes: int
    timed_out: bool
    rss_exceeded: bool
    stdout_sha256: str
    stdout_bytes: int
    stdout_tail: str
    stderr_sha256: str
    stderr_bytes: int
    stderr_tail: str
    stdout_path: str
    stderr_path: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible parent observation.

        Returns
        -------
        dict[str, Any]
            Complete supervisor facts.
        """

        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "exit_code": self.exit_code,
            "signal": self.signal_number,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "timed_out": self.timed_out,
            "rss_exceeded": self.rss_exceeded,
            "stdout_sha256": self.stdout_sha256,
            "stdout_bytes": self.stdout_bytes,
            "stdout_tail": self.stdout_tail,
            "stderr_sha256": self.stderr_sha256,
            "stderr_bytes": self.stderr_bytes,
            "stderr_tail": self.stderr_tail,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }


@dataclass(frozen=True)
class SupervisedResult:
    """Parent observation plus an optional verified atomic worker receipt.

    Parameters
    ----------
    observation:
        Facts measured by the supervisor.
    worker_receipt:
        Parsed receipt only when a complete valid atomic file exists.
    receipt_error:
        Parent diagnosis when the receipt is absent or invalid.
    """

    observation: SupervisorObservation
    worker_receipt: Optional[dict[str, Any]]
    receipt_error: Optional[str]


@dataclass(frozen=True)
class SandboxDenialObservation:
    """OS-boundary network and outside-write denials observed by the supervisor.

    Parameters
    ----------
    network_attempted:
        Whether a network syscall reached the offline boundary.
    socket_targets:
        Sanitized syscall targets observed by the broker.
    write_outside_scratch_attempted:
        Whether a write syscall outside the allowed roots was denied.
    write_paths:
        Sanitized denied write paths.
    """

    network_attempted: bool = False
    socket_targets: tuple[str, ...] = ()
    write_outside_scratch_attempted: bool = False
    write_paths: tuple[str, ...] = ()

    @property
    def poisoned(self) -> bool:
        """Return whether the denial invalidates an otherwise successful receipt.

        Returns
        -------
        bool
            True for any observed denied network or outside-write operation.
        """

        return self.network_attempted or self.write_outside_scratch_attempted


def _child_limit(rss_limit_bytes: int) -> None:
    """Apply a fail-closed child address-space limit before exec.

    Parameters
    ----------
    rss_limit_bytes:
        Requested memory cap in bytes.
    """

    if rss_limit_bytes > 0:
        resource.setrlimit(resource.RLIMIT_AS, (rss_limit_bytes, rss_limit_bytes))


def _linux_rss(pid: int) -> int:
    """Read current Linux resident bytes for a child process.

    Parameters
    ----------
    pid:
        Child process ID.

    Returns
    -------
    int
        Resident bytes, or zero when unavailable.
    """

    status_path = Path(f"/proc/{pid}/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _kill_process_group(process: subprocess.Popen[Any]) -> None:
    """Terminate a complete isolated process group.

    Parameters
    ----------
    process:
        Root child process.
    """

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _tail(data: bytes) -> str:
    """Decode a bounded stdout/stderr tail.

    Parameters
    ----------
    data:
        Complete captured stream bytes.

    Returns
    -------
    str
        Replacement-decoded bounded tail.
    """

    return data.decode("utf-8", errors="replace")[-STDIO_TAIL_MAX_CHARS:]


def _rusage_seconds(usage: resource.struct_rusage) -> float:
    """Return combined user and system CPU seconds.

    Parameters
    ----------
    usage:
        Child resource usage snapshot.

    Returns
    -------
    float
        CPU seconds.
    """

    return float(usage.ru_utime + usage.ru_stime)


def _linux_audited_argv(
    sandboxed_argv: Sequence[str], audit_executable: str, audit_path: Path
) -> tuple[str, ...]:
    """Insert a syscall-observation broker inside a Linux sandbox command.

    Parameters
    ----------
    sandboxed_argv:
        Complete bubblewrap or unshare command.
    audit_executable:
        Absolute path to ``strace``.
    audit_path:
        Writable broker output path.

    Returns
    -------
    tuple[str, ...]
        Sandbox command whose child is supervised by the syscall broker.

    Raises
    ------
    SandboxUnavailableError
        If the sandbox wrapper has no child-command separator.
    """

    separators = [index for index, value in enumerate(sandboxed_argv) if value == "--"]
    if not separators:
        raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
    separator = separators[-1]
    broker = (
        audit_executable,
        "-f",
        "-qq",
        "-e",
        "trace=%network,%file",
        "-s",
        "4096",
        "-o",
        str(audit_path),
        "--",
    )
    return (
        *sandboxed_argv[: separator + 1],
        *broker,
        *sandboxed_argv[separator + 1 :],
    )


def _syscall_name(line: str) -> Optional[str]:
    """Return the traced syscall name from one broker line.

    Parameters
    ----------
    line:
        One text line emitted by the syscall broker.

    Returns
    -------
    str | None
        Syscall name, excluding strace process prefixes.
    """

    match = _SYSCALL_PATTERN.search(line)
    return None if match is None else match.group(1)


def _network_target(line: str) -> str:
    """Return a bounded network target description from a traced syscall.

    Parameters
    ----------
    line:
        Network syscall trace line.

    Returns
    -------
    str
        Sanitized address-family and endpoint excerpt.
    """

    family = "AF_INET6" if "AF_INET6" in line else "AF_INET"
    address_match = re.search(r'(?:inet_addr|inet_pton\([^,]+,)\("?([^"),]+)', line)
    port_match = re.search(r"sin6?_port=htons\((\d+)\)", line)
    address = address_match.group(1) if address_match is not None else "unknown"
    port = port_match.group(1) if port_match is not None else "unknown"
    return f"{family}:{address}:{port}"[:500]


def _decoded_trace_paths(line: str) -> tuple[str, ...]:
    """Decode quoted filesystem paths from one syscall trace line.

    Parameters
    ----------
    line:
        File syscall trace line.

    Returns
    -------
    tuple[str, ...]
        Best-effort decoded quoted values in call order.
    """

    values: list[str] = []
    for match in _QUOTED_PATH_PATTERN.finditer(line):
        encoded = match.group(1)
        try:
            values.append(bytes(encoded, "utf-8").decode("unicode_escape"))
        except UnicodeDecodeError:
            values.append(encoded)
    return tuple(values)


def _outside_allowed_roots(path_text: str, cwd: Path, write_roots: Sequence[Path]) -> bool:
    """Return whether a traced write path is outside every allowed root.

    Parameters
    ----------
    path_text:
        Traced absolute or working-directory-relative path.
    cwd:
        Worker working directory.
    write_roots:
        Sole writable roots granted to the OS sandbox.

    Returns
    -------
    bool
        True when the path is not beneath an allowed root.
    """

    path = Path(path_text)
    candidate = (path if path.is_absolute() else cwd / path).resolve()
    roots = tuple(root.resolve() for root in write_roots)
    return not any(candidate == root or root in candidate.parents for root in roots)


def _parse_linux_denial_audit(
    audit_path: Path, cwd: Path, write_roots: Sequence[Path]
) -> SandboxDenialObservation:
    """Parse Linux syscall telemetry into closed worker policy observations.

    Parameters
    ----------
    audit_path:
        Broker output produced for one child process tree.
    cwd:
        Worker working directory used to resolve relative write targets.
    write_roots:
        Sole OS-sandbox writable roots.

    Returns
    -------
    SandboxDenialObservation
        Deduplicated network and outside-write denials.
    """

    try:
        lines = audit_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return SandboxDenialObservation()
    socket_targets: list[str] = []
    write_paths: list[str] = []
    for line in lines:
        syscall = _syscall_name(line)
        if syscall in {"connect", "sendmsg", "sendto"} and (
            "AF_INET" in line or "AF_INET6" in line
        ):
            socket_targets.append(_network_target(line))
            continue
        if syscall not in _WRITE_SYSCALLS or not any(
            f" {errno} " in line for errno in _DENIED_ERRNOS
        ):
            continue
        if syscall in {"open", "openat", "openat2"} and not any(
            flag in line for flag in _WRITE_OPEN_FLAGS
        ):
            continue
        paths = _decoded_trace_paths(line)
        outside_paths = [path for path in paths if _outside_allowed_roots(path, cwd, write_roots)]
        if outside_paths:
            write_paths.extend(outside_paths)
        elif not paths:
            write_paths.append(f"<{syscall}:denied-outside-sandbox>")
    unique_targets = tuple(dict.fromkeys(socket_targets))
    unique_paths = tuple(dict.fromkeys(write_paths))
    return SandboxDenialObservation(
        network_attempted=bool(unique_targets),
        socket_targets=unique_targets,
        write_outside_scratch_attempted=bool(unique_paths),
        write_paths=unique_paths,
    )


def _macos_denial_audit(stderr: bytes) -> SandboxDenialObservation:
    """Parse sandbox-exec denial messages when Seatbelt emits them to stderr.

    Parameters
    ----------
    stderr:
        Complete captured child stderr.

    Returns
    -------
    SandboxDenialObservation
        Denial flags derived only from explicit Seatbelt deny messages.
    """

    network: list[str] = []
    writes: list[str] = []
    for line in stderr.decode("utf-8", errors="replace").splitlines():
        lowered = line.lower()
        if "deny" not in lowered:
            continue
        if "network" in lowered:
            network.append(line[-500:])
        if "file-write" in lowered or "file write" in lowered:
            writes.append(line[-500:])
    return SandboxDenialObservation(
        network_attempted=bool(network),
        socket_targets=tuple(dict.fromkeys(network)),
        write_outside_scratch_attempted=bool(writes),
        write_paths=tuple(dict.fromkeys(writes)),
    )


def _merge_denial_observations(
    *observations: SandboxDenialObservation,
) -> SandboxDenialObservation:
    """Merge OS denial channels without losing any observed target.

    Parameters
    ----------
    *observations:
        Denial observations for the same process tree.

    Returns
    -------
    SandboxDenialObservation
        Union of all flags and targets.
    """

    targets = tuple(
        dict.fromkeys(
            target for observation in observations for target in observation.socket_targets
        )
    )
    paths = tuple(
        dict.fromkeys(path for observation in observations for path in observation.write_paths)
    )
    return SandboxDenialObservation(
        network_attempted=any(observation.network_attempted for observation in observations),
        socket_targets=targets,
        write_outside_scratch_attempted=any(
            observation.write_outside_scratch_attempted for observation in observations
        ),
        write_paths=paths,
    )


def _atomic_rewrite_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Atomically replace one supervisor-poisoned self-hashed receipt.

    Parameters
    ----------
    path:
        Existing worker receipt path.
    receipt:
        Receipt payload including its recomputed self hash.
    """

    temporary = path.with_name(f".{path.name}.{os.getpid()}.supervisor.tmp")
    data = canonical_json_bytes(receipt) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def poison_receipt_for_sandbox_denial(receipt_path: Path, denial: SandboxDenialObservation) -> bool:
    """Poison an otherwise successful worker receipt after an OS denial.

    Parameters
    ----------
    receipt_path:
        Atomic worker receipt to audit and, when necessary, replace.
    denial:
        Parent-observed OS denial telemetry.

    Returns
    -------
    bool
        True only when a valid worker receipt was poisoned.
    """

    if not denial.poisoned or not receipt_path.is_file():
        return False
    try:
        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(loaded, dict) or loaded.get("receipt_version") != (
        "menagerie.crawler.worker-receipt.v1"
    ):
        return False
    payload = {key: value for key, value in loaded.items() if key != "receipt_sha256"}
    policy_value = payload.get("policy_observation")
    if not isinstance(policy_value, Mapping):
        return False
    policy = dict(policy_value)
    policy["network_attempted"] = bool(policy.get("network_attempted")) or (
        denial.network_attempted
    )
    policy["socket_targets"] = list(
        dict.fromkeys([*policy.get("socket_targets", []), *denial.socket_targets])
    )
    policy["write_outside_scratch_attempted"] = (
        bool(policy.get("write_outside_scratch_attempted"))
        or denial.write_outside_scratch_attempted
    )
    policy["write_paths"] = list(
        dict.fromkeys([*policy.get("write_paths", []), *denial.write_paths])
    )
    payload["policy_observation"] = policy
    reason_code = "network-attempt" if denial.network_attempted else "write-outside-scratch"
    error = {
        "reason_code": reason_code,
        "exception_type": ("menagerie.crawler.worker_supervisor.SandboxDenialObservation"),
        "message": "OS sandbox denied a forbidden operation caught by worker code",
        "traceback": None,
    }
    payload["error"] = error
    per_mode_value = payload.get("per_mode")
    if isinstance(per_mode_value, Mapping):
        per_mode: dict[str, Any] = {}
        for mode, mode_value in per_mode_value.items():
            if isinstance(mode_value, Mapping):
                poisoned_mode = dict(mode_value)
                poisoned_mode["error"] = error
                per_mode[str(mode)] = poisoned_mode
            else:
                per_mode[str(mode)] = mode_value
        payload["per_mode"] = per_mode
    record = {**payload, "receipt_sha256": stable_hash(payload)}
    _atomic_rewrite_receipt(receipt_path, record)
    return True


def _poison_receipts_in_roots(
    write_roots: Sequence[Path], denial: SandboxDenialObservation
) -> None:
    """Poison worker receipts found directly in explicit result roots.

    Parameters
    ----------
    write_roots:
        Explicit result directories granted to the child.
    denial:
        Parent-observed OS denial telemetry.
    """

    if not denial.poisoned:
        return
    for root in write_roots:
        try:
            candidates = tuple(root.glob("*.json"))
        except OSError:
            continue
        for candidate in candidates:
            poison_receipt_for_sandbox_denial(candidate, denial)


def run_isolated_subprocess(
    argv: Sequence[str],
    scratch_root: Path,
    *,
    timeout_seconds: float = DEFAULT_FORWARD_TIMEOUT_SECONDS,
    rss_limit_bytes: int = 12 * 1024**3,
    cwd: Optional[Path] = None,
    base_environment: Optional[Mapping[str, str]] = None,
    additional_write_roots: Sequence[Path] = (),
) -> SupervisorObservation:
    """Launch a fresh credential-scrubbed subprocess inside an OS sandbox.

    Parameters
    ----------
    argv:
        Exact executable argument vector.
    scratch_root:
        Fresh writable cache/log root.
    timeout_seconds:
        Parent-enforced wall timeout.
    rss_limit_bytes:
        Parent-observed RSS and child address-space cap.
    cwd:
        Read-only source working directory. Defaults to the current directory.
    base_environment:
        Optional environment filtered through the safe allowlist.
    additional_write_roots:
        Explicit result roots writable in addition to scratch.

    Returns
    -------
    SupervisorObservation
        Parent-only process and resource facts.

    Raises
    ------
    SandboxUnavailableError
        If no complete OS sandbox is working on this host.
    """

    if not argv or any(not isinstance(value, str) or "\x00" in value for value in argv):
        raise ValueError("argv must contain non-empty NUL-free strings")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    scratch_root.mkdir(parents=True, exist_ok=True)
    write_roots = (scratch_root.resolve(), *(path.resolve() for path in additional_write_roots))
    for root in write_roots:
        root.mkdir(parents=True, exist_ok=True)
    stdout_path = scratch_root / "stdout.log"
    stderr_path = scratch_root / "stderr.log"
    safe_environment = build_safe_environment(scratch_root, base_environment=base_environment)
    working_directory = (cwd or Path.cwd()).resolve()
    sandbox = detect_os_sandbox()
    if sandbox is None:
        raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
    profile_path: Optional[Path] = None
    if sandbox.kind == "sandbox-exec":
        profile_path = scratch_root / "worker-sandbox.sb"
        profile_path.write_text(
            generate_macos_sandbox_profile(write_roots),
            encoding="utf-8",
        )
    sandboxed_argv = wrap_with_os_sandbox(
        sandbox,
        argv,
        working_directory,
        write_roots,
        macos_profile_path=profile_path,
    )
    denial_audit_path: Optional[Path] = None
    if sandbox.kind in {"bubblewrap", "unshare"}:
        denial_audit_executable = shutil.which("strace")
        if denial_audit_executable is None:
            raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
        denial_audit_path = scratch_root / "sandbox-denial-audit.log"
        denial_audit_path.unlink(missing_ok=True)
        sandboxed_argv = _linux_audited_argv(
            sandboxed_argv,
            denial_audit_executable,
            denial_audit_path,
        )
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    timed_out = False
    rss_exceeded = False
    peak_rss = 0
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            list(sandboxed_argv),
            cwd=working_directory,
            env=safe_environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
            start_new_session=True,
            close_fds=True,
            preexec_fn=partial(_child_limit, rss_limit_bytes),
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            current_rss = _linux_rss(process.pid)
            peak_rss = max(peak_rss, current_rss)
            if current_rss and rss_limit_bytes > 0 and current_rss > rss_limit_bytes:
                rss_exceeded = True
                _kill_process_group(process)
                break
            if elapsed >= timeout_seconds:
                timed_out = True
                _kill_process_group(process)
                break
            time.sleep(0.01)
        process.wait()
    wall_seconds = time.monotonic() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = max(0.0, _rusage_seconds(usage_after) - _rusage_seconds(usage_before))
    peak_rss = max(peak_rss, int(usage_after.ru_maxrss) * 1024)
    return_code = process.returncode
    signal_number = -return_code if return_code is not None and return_code < 0 else None
    exit_code = return_code if return_code is not None and return_code >= 0 else None
    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes()
    denial = _macos_denial_audit(stderr)
    if denial_audit_path is not None:
        denial = _merge_denial_observations(
            denial,
            _parse_linux_denial_audit(
                denial_audit_path,
                working_directory,
                write_roots,
            ),
        )
    _poison_receipts_in_roots(additional_write_roots, denial)
    return SupervisorObservation(
        argv=sandboxed_argv,
        cwd=str(working_directory),
        exit_code=exit_code,
        signal_number=signal_number,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        peak_rss_bytes=peak_rss,
        timed_out=timed_out,
        rss_exceeded=rss_exceeded,
        stdout_sha256=hash_bytes(stdout),
        stdout_bytes=len(stdout),
        stdout_tail=_tail(stdout),
        stderr_sha256=hash_bytes(stderr),
        stderr_bytes=len(stderr),
        stderr_tail=_tail(stderr),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _sandbox_unavailable_observation(
    argv: Sequence[str], scratch_root: Path, working_directory: Path
) -> SupervisorObservation:
    """Create honest parent facts for a worker refused before launch.

    Parameters
    ----------
    argv:
        Worker command that was not launched.
    scratch_root:
        Supervisor log root.
    working_directory:
        Requested worker working directory.

    Returns
    -------
    SupervisorObservation
        Zero-resource observation with a durable closed failure log.
    """

    scratch_root.mkdir(parents=True, exist_ok=True)
    stdout_path = scratch_root / "stdout.log"
    stderr_path = scratch_root / "stderr.log"
    stdout = b""
    status = f"failed:{FailureStage.SANDBOX_UNAVAILABLE.value}"
    stderr = f"{status}\n".encode("utf-8")
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    return SupervisorObservation(
        argv=tuple(argv),
        cwd=str(working_directory),
        exit_code=None,
        signal_number=None,
        wall_seconds=0.0,
        cpu_seconds=0.0,
        peak_rss_bytes=0,
        timed_out=False,
        rss_exceeded=False,
        stdout_sha256=hash_bytes(stdout),
        stdout_bytes=0,
        stdout_tail="",
        stderr_sha256=hash_bytes(stderr),
        stderr_bytes=len(stderr),
        stderr_tail=_tail(stderr),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _load_receipt(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Load and verify a complete atomic worker receipt.

    Parameters
    ----------
    path:
        Expected final receipt path.

    Returns
    -------
    tuple[dict[str, Any] | None, str | None]
        Verified receipt or a parent-owned error.
    """

    if not path.exists():
        return None, "missing-receipt"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid-receipt:{type(exc).__name__}"
    if not isinstance(value, dict):
        return None, "invalid-receipt:not-an-object"
    claimed_hash = value.get("receipt_sha256")
    payload = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if claimed_hash != stable_hash(payload):
        return None, "invalid-receipt:hash-mismatch"
    return value, None


def supervise_worker(
    request_path: Path,
    receipt_path: Path,
    scratch_root: Path,
    *,
    timeout_seconds: float = DEFAULT_FORWARD_TIMEOUT_SECONDS,
    rss_limit_bytes: int = 12 * 1024**3,
    cwd: Optional[Path] = None,
) -> SupervisedResult:
    """Launch the standard worker and attach only a verified atomic receipt.

    Parameters
    ----------
    request_path, receipt_path:
        Immutable request and expected atomic child result.
    scratch_root:
        Fresh logs/caches root.
    timeout_seconds, rss_limit_bytes:
        Parent resource caps.
    cwd:
        Source working directory.

    Returns
    -------
    SupervisedResult
        Parent observation plus an optional verified child receipt.
    """

    receipt_path.unlink(missing_ok=True)
    argv = (
        sys.executable,
        "-m",
        "menagerie.crawler.worker",
        "--request",
        str(request_path),
        "--receipt",
        str(receipt_path),
    )
    working_directory = (cwd or Path.cwd()).resolve()
    try:
        observation = run_isolated_subprocess(
            argv,
            scratch_root,
            timeout_seconds=timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
            cwd=working_directory,
            additional_write_roots=(receipt_path.parent,),
        )
    except SandboxUnavailableError:
        observation = _sandbox_unavailable_observation(argv, scratch_root, working_directory)
        status = f"failed:{FailureStage.SANDBOX_UNAVAILABLE.value}"
        return SupervisedResult(observation, None, status)
    receipt, receipt_error = _load_receipt(receipt_path)
    if observation.exit_code != 0 or observation.signal_number is not None:
        return SupervisedResult(observation, None, receipt_error or "worker-exit-nonzero")
    return SupervisedResult(observation, receipt, receipt_error)
