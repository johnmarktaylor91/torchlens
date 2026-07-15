"""Parent-owned argv-only subprocess isolation and resource observations."""

from __future__ import annotations

import json
import os
import re
import resource
import signal
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
    STDIO_TAIL_MAX_CHARS,
    FailureStage,
)
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.policy import (
    _PARENT_ALLOWED_READ_PATHS_ENV,
    SandboxUnavailableError,
    _linux_runtime_code_roots,
    _linux_runtime_read_paths,
    _runtime_code_path_allowed,
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
_SPECIAL_READ_ROOTS = (Path("/dev"), Path("/proc"), Path("/sys"))
_SYSTEM_READ_FILES = frozenset(
    {
        Path("/etc/group"),
        Path("/etc/hosts"),
        Path("/etc/ld.so.cache"),
        Path("/etc/locale.alias"),
        Path("/etc/localtime"),
        Path("/etc/nsswitch.conf"),
        Path("/etc/passwd"),
        Path("/etc/resolv.conf"),
        Path("/usr/share/locale/locale.alias"),
    }
)
_TERMINAL_TRACE_PATTERN = re.compile(r"\+\+\+ (?:exited with|killed by) .+ \+\+\+$")
_MACOS_AUDIT_COMPLETION_MARKER = "MENAGERIE_MACOS_SANDBOX_AUDIT_COMPLETE_V1"


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
    """OS-boundary policy denials observed by the supervisor.

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
    checkpoint_or_weight_read_attempted, checkpoint_paths:
        Undeclared model-data reads observed at the kernel boundary.
    telemetry_failure:
        Fail-closed broker-integrity diagnostic, if telemetry was not trustworthy.
    """

    network_attempted: bool = False
    socket_targets: tuple[str, ...] = ()
    write_outside_scratch_attempted: bool = False
    write_paths: tuple[str, ...] = ()
    checkpoint_or_weight_read_attempted: bool = False
    checkpoint_paths: tuple[str, ...] = ()
    telemetry_failure: Optional[str] = None

    @property
    def poisoned(self) -> bool:
        """Return whether the denial invalidates an otherwise successful receipt.

        Returns
        -------
        bool
            True for any observed denied network or outside-write operation.
        """

        return (
            self.network_attempted
            or self.write_outside_scratch_attempted
            or self.checkpoint_or_weight_read_attempted
            or self.telemetry_failure is not None
        )


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
    """Wrap a Linux sandbox command in a parent-owned syscall broker.

    Parameters
    ----------
    sandboxed_argv:
        Complete bubblewrap or unshare command.
    audit_executable:
        Absolute path to ``strace``.
    audit_path:
        Parent-owned broker output path outside every child-writable bind.

    Returns
    -------
    tuple[str, ...]
        Broker command supervising the complete sandboxed process tree.
    """

    return (
        audit_executable,
        "-f",
        "-q",
        "-yy",
        "-e",
        "trace=%network,%file,%process",
        "-s",
        "4096",
        "-o",
        str(audit_path),
        "--",
        *sandboxed_argv,
    )


def _parent_owned_audit_path(
    scratch_root: Path,
    write_roots: Sequence[Path],
    *,
    filename: str = "sandbox-syscalls.log",
) -> tuple[Path, tuple[int, int]]:
    """Create immutable-identity telemetry storage outside child-writable roots.

    Parameters
    ----------
    scratch_root:
        Supervisor scratch directory used to choose a nearby parent-owned sibling.
    write_roots:
        Roots that the OS sandbox exposes writable to the child.
    filename:
        Fixed telemetry filename inside the parent-owned directory.

    Returns
    -------
    tuple[pathlib.Path, tuple[int, int]]
        Audit path and its parent-recorded device/inode identity.

    Raises
    ------
    SandboxUnavailableError
        If no parent-owned location can be established.
    """

    roots = tuple(root.resolve() for root in write_roots)
    base = scratch_root.resolve().parent
    for _attempt in range(8):
        directory = base / f".menagerie-parent-audit-{os.getpid()}-{time.time_ns()}"
        if not any(directory == root or root in directory.parents for root in roots):
            try:
                directory.mkdir(mode=0o700)
                path = directory / filename
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(descriptor)
                os.link(path, directory / f"{filename}.anchor")
                status = path.stat()
                return path, (status.st_dev, status.st_ino)
            except OSError as exc:
                raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value) from exc
        base = base.parent
    raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)


def _request_allowed_read_paths(argv: Sequence[str]) -> tuple[Path, ...]:
    """Derive exact executable/source/input paths from a worker argv and request.

    Parameters
    ----------
    argv:
        Original unsandboxed command vector.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Parent-verified paths allowed as runtime source or standard input.
    """

    allowed: list[Path] = []
    if argv:
        executable = Path(argv[0])
        if executable.exists():
            allowed.append(executable)
    try:
        request_index = argv.index("--request") + 1
        request_path = Path(argv[request_index]).resolve()
    except (ValueError, IndexError):
        return tuple(dict.fromkeys(allowed))
    allowed.append(request_path)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return tuple(dict.fromkeys(allowed))
    if not isinstance(request, Mapping):
        return tuple(dict.fromkeys(allowed))
    standard_asset = _request_standard_asset(request)
    if standard_asset is not None:
        allowed.append(standard_asset)
    recipe = request.get("recipe")
    if isinstance(recipe, Mapping):
        adapter_path = recipe.get("path")
        if isinstance(adapter_path, str) and adapter_path:
            resolved_adapter = Path(adapter_path).resolve()
            allowed.append(resolved_adapter)
            for directory in (resolved_adapter.parent, *resolved_adapter.parents[:4]):
                sidecar = directory / "driver-author-artifact.json"
                if sidecar.is_file():
                    allowed.append(sidecar)
                    break
    input_contract = request.get("input_contract")
    if isinstance(input_contract, Mapping):
        input_code_path = input_contract.get("code_path")
        if isinstance(input_code_path, str) and input_code_path:
            allowed.append(Path(input_code_path))
    return tuple(dict.fromkeys(path.resolve() for path in allowed))


def _request_standard_asset(request: Mapping[str, Any]) -> Optional[Path]:
    """Return the one bundled standard asset selected by a worker request.

    Parameters
    ----------
    request:
        Parsed immutable worker request.

    Returns
    -------
    pathlib.Path | None
        Exact possible standard-input asset, or ``None`` for random-only modalities.
    """

    modality_value = request.get("modality")
    modalities: tuple[str, ...]
    if isinstance(modality_value, str):
        modalities = (modality_value.strip().lower(),)
    elif isinstance(modality_value, list):
        modalities = tuple(str(value).strip().lower() for value in modality_value)
    else:
        modalities = ()
    asset_root = Path(__file__).with_name("assets") / "standard"
    if any(value in {"vision", "image", "computer-vision", "video"} for value in modalities):
        return asset_root / "image.ppm"
    if any(value in {"language", "text", "nlp"} for value in modalities):
        return asset_root / "text.txt"
    if any(value in {"audio", "speech"} for value in modalities):
        return asset_root / "audio.csv"
    if any(value in {"tabular", "recsys"} for value in modalities):
        return asset_root / "tabular.csv"
    return None


def _runtime_read_roots(argv: Sequence[str], cwd: Path) -> tuple[Path, ...]:
    """Return environment/source roots limited to runtime-code reads on macOS.

    Parameters
    ----------
    argv:
        Original worker command vector.
    cwd:
        Read-only source working directory.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Python environment and source roots.
    """

    roots = [cwd.resolve()]
    if argv:
        executable = Path(argv[0]).resolve()
        roots.append(
            executable.parent.parent if executable.parent.name == "bin" else executable.parent
        )
    return tuple(dict.fromkeys(roots))


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


def _telemetry_failure_observation(detail: str) -> SandboxDenialObservation:
    """Return a fail-closed policy observation for invalid broker telemetry.

    Parameters
    ----------
    detail:
        Bounded parent-owned integrity diagnosis.

    Returns
    -------
    SandboxDenialObservation
        Poisoning checkpoint-read observation.
    """

    marker = f"<sandbox-telemetry-invalid:{detail}>"
    return SandboxDenialObservation(
        checkpoint_or_weight_read_attempted=True,
        checkpoint_paths=(marker,),
        telemetry_failure=detail,
    )


def _resolved_trace_path(path_text: str, cwd: Path) -> Path:
    """Resolve one traced path conservatively against the worker directory.

    Parameters
    ----------
    path_text:
        Absolute or tracee-relative path.
    cwd:
        Worker working directory.

    Returns
    -------
    pathlib.Path
        Normalized candidate path.
    """

    path = Path(path_text)
    return (path if path.is_absolute() else cwd / path).resolve()


def _read_path_is_allowed(
    path_text: str,
    cwd: Path,
    write_roots: Sequence[Path],
    allowed_read_paths: Sequence[Path],
    runtime_code_roots: Sequence[Path],
    *,
    directory_only: bool = False,
) -> bool:
    """Return whether a kernel-level read belongs to the closed runtime allowlist.

    Parameters
    ----------
    path_text:
        Path decoded from one successful read-only open.
    cwd:
        Worker working directory.
    write_roots:
        Fresh scratch/result roots whose contents are parent-authorized runtime state.
    allowed_read_paths:
        Exact source/input paths derived from the immutable worker request.
    runtime_code_roots:
        Environment and verified-source roots limited to runtime-code reads.
    directory_only:
        Whether the traced open explicitly required a directory descriptor.

    Returns
    -------
    bool
        True only for declared inputs, source/runtime code, or OS runtime support.
    """

    raw_path = Path(path_text)
    lexical_candidate = (raw_path if raw_path.is_absolute() else cwd / raw_path).absolute()
    candidate = _resolved_trace_path(path_text, cwd)
    roots = tuple(root.resolve() for root in write_roots)
    allowed = tuple(path.resolve() for path in allowed_read_paths)
    runtime_roots = tuple(root.resolve() for root in runtime_code_roots)
    if lexical_candidate in _SYSTEM_READ_FILES:
        return True
    if path_text in {"self", "self/fd", "self/mountinfo"} or re.fullmatch(
        r"\d+/(?:fd|ns)(?:/.*)?", path_text
    ):
        return True
    if any(candidate == root or root in candidate.parents for root in _SPECIAL_READ_ROOTS):
        return True
    if any(candidate == root or root in candidate.parents for root in roots):
        return True
    if any(candidate == path or path in candidate.parents for path in allowed):
        return True
    if directory_only and any(
        candidate == root or root in candidate.parents for root in runtime_roots
    ):
        return True
    if candidate.is_relative_to(Path("/usr/lib/locale")) and (
        candidate.name == "locale-archive" or candidate.name.startswith("LC_")
    ):
        return True
    if candidate.name == "gconv-modules.cache" and "gconv" in candidate.parts:
        return True
    try:
        if candidate.is_dir():
            return True
    except OSError:
        pass
    return _runtime_code_path_allowed(candidate, runtime_code_roots)


def _trace_line_is_well_formed(line: str) -> bool:
    """Return whether one nonempty strace line has a recognized complete form.

    Parameters
    ----------
    line:
        One parent-owned telemetry line.

    Returns
    -------
    bool
        True for syscall, signal, continuation, or terminal records.
    """

    return bool(
        _syscall_name(line)
        or _TERMINAL_TRACE_PATTERN.search(line)
        or "<unfinished ...>" in line
        or "resumed>" in line
        or re.search(r"(?:^|\s)--- SIG[A-Z0-9]+ ", line)
    )


def _parse_linux_denial_audit(
    audit_path: Path,
    cwd: Path,
    write_roots: Sequence[Path],
    *,
    expected_identity: Optional[tuple[int, int]] = None,
    allowed_read_paths: Sequence[Path] = (),
    runtime_code_roots: Sequence[Path] = (),
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
    expected_identity:
        Parent-recorded device/inode pair for replacement detection.
    allowed_read_paths:
        Explicit source/input paths authorized for model execution.
    runtime_code_roots:
        Environment and verified-source roots limited to runtime-code reads.

    Returns
    -------
    SandboxDenialObservation
        Deduplicated network and outside-write denials.
    """

    try:
        status = audit_path.stat()
        if not stat.S_ISREG(status.st_mode):
            return _telemetry_failure_observation("not-regular")
        if expected_identity is not None and (status.st_dev, status.st_ino) != expected_identity:
            return _telemetry_failure_observation("replaced")
        anchor = audit_path.with_name(f"{audit_path.name}.anchor")
        if anchor.exists():
            anchor_status = anchor.stat()
            if (anchor_status.st_dev, anchor_status.st_ino) != (status.st_dev, status.st_ino):
                return _telemetry_failure_observation("replaced")
        content = audit_path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return _telemetry_failure_observation("unparsable-encoding")
    except OSError:
        return _telemetry_failure_observation("missing")
    lines = content.splitlines()
    if not lines:
        return _telemetry_failure_observation("empty")
    if not _TERMINAL_TRACE_PATTERN.search(lines[-1]):
        return _telemetry_failure_observation("truncated")
    if any(not _trace_line_is_well_formed(line) for line in lines if line.strip()):
        return _telemetry_failure_observation("unparsable-record")
    socket_targets: list[str] = []
    write_paths: list[str] = []
    checkpoint_paths: list[str] = []
    for line in lines:
        syscall = _syscall_name(line)
        if syscall in {"connect", "sendmsg", "sendto"} and (
            "AF_INET" in line or "AF_INET6" in line
        ):
            socket_targets.append(_network_target(line))
            continue
        if syscall in {"open", "openat", "openat2"} and not any(
            flag in line for flag in _WRITE_OPEN_FLAGS
        ):
            paths = _decoded_trace_paths(line)
            if paths:
                path_text = paths[0]
                if not _read_path_is_allowed(
                    path_text,
                    cwd,
                    write_roots,
                    allowed_read_paths,
                    runtime_code_roots,
                    directory_only="O_DIRECTORY" in line,
                ):
                    checkpoint_paths.append(path_text)
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
    unique_checkpoints = tuple(dict.fromkeys(checkpoint_paths))
    return SandboxDenialObservation(
        network_attempted=bool(unique_targets),
        socket_targets=unique_targets,
        write_outside_scratch_attempted=bool(unique_paths),
        write_paths=unique_paths,
        checkpoint_or_weight_read_attempted=bool(unique_checkpoints),
        checkpoint_paths=unique_checkpoints,
    )


def _macos_denial_message(line: str) -> Optional[str]:
    """Extract one Seatbelt denial message from an audited macOS record.

    Parameters
    ----------
    line:
        One parent-owned NDJSON or legacy textual audit record.

    Returns
    -------
    str | None
        Denial message, or ``None`` when the record is unrecognized.
    """

    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, Mapping):
            return None
        for key in ("eventMessage", "composedMessage", "message"):
            message = value.get(key)
            if isinstance(message, str) and "deny" in message.lower():
                return message
        return None
    return stripped if "deny" in stripped.lower() else None


def _macos_denial_audit(telemetry: bytes) -> SandboxDenialObservation:
    """Parse completion-marked parent-owned macOS Seatbelt telemetry.

    Parameters
    ----------
    telemetry:
        Complete bytes written by the parent-controlled unified-log collector.

    Returns
    -------
    SandboxDenialObservation
        Parsed denials, with fail-closed poison for missing or malformed completion.
    """

    try:
        lines = telemetry.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        return _telemetry_failure_observation("unparsable-encoding")
    if not lines:
        return _telemetry_failure_observation("empty")
    completed = lines[-1] == _MACOS_AUDIT_COMPLETION_MARKER
    records = lines[:-1] if completed else lines

    network: list[str] = []
    writes: list[str] = []
    checkpoint_paths: list[str] = []
    unparseable = False
    for line in records:
        message = _macos_denial_message(line)
        if message is None:
            unparseable = True
            continue
        lowered = message.lower()
        recognized = False
        if "network" in lowered:
            network.append(message[-500:])
            recognized = True
        if "file-write" in lowered or "file write" in lowered:
            writes.append(message[-500:])
            recognized = True
        if "file-read-data" in lowered or "file read data" in lowered:
            checkpoint_paths.append(message[-500:])
            recognized = True
        if not recognized:
            unparseable = True
    observed = SandboxDenialObservation(
        network_attempted=bool(network),
        socket_targets=tuple(dict.fromkeys(network)),
        write_outside_scratch_attempted=bool(writes),
        write_paths=tuple(dict.fromkeys(writes)),
        checkpoint_or_weight_read_attempted=bool(checkpoint_paths),
        checkpoint_paths=tuple(dict.fromkeys(checkpoint_paths)),
    )
    failures: list[SandboxDenialObservation] = []
    if not completed:
        failures.append(_telemetry_failure_observation("truncated"))
    if unparseable:
        failures.append(_telemetry_failure_observation("unparsable-record"))
    return _merge_denial_observations(observed, *failures)


def _parse_macos_denial_audit(
    audit_path: Path,
    *,
    expected_identity: Optional[tuple[int, int]] = None,
) -> SandboxDenialObservation:
    """Verify and parse one parent-owned macOS Seatbelt audit channel.

    Parameters
    ----------
    audit_path:
        Parent-controlled unified-log output path.
    expected_identity:
        Parent-recorded device/inode pair for replacement detection.

    Returns
    -------
    SandboxDenialObservation
        Parsed denial or fail-closed telemetry-integrity observation.
    """

    try:
        status = audit_path.stat()
        if not stat.S_ISREG(status.st_mode):
            return _telemetry_failure_observation("not-regular")
        if expected_identity is not None and (status.st_dev, status.st_ino) != expected_identity:
            return _telemetry_failure_observation("replaced")
        anchor = audit_path.with_name(f"{audit_path.name}.anchor")
        anchor_status = anchor.stat()
        if (anchor_status.st_dev, anchor_status.st_ino) != (status.st_dev, status.st_ino):
            return _telemetry_failure_observation("replaced")
        telemetry = audit_path.read_bytes()
    except OSError:
        return _telemetry_failure_observation("missing")
    return _macos_denial_audit(telemetry)


@dataclass
class _MacOSAuditChannel:
    """Parent-owned unified-log collector state for one sandboxed process tree."""

    path: Path
    expected_identity: tuple[int, int]
    process: subprocess.Popen[Any]
    handle: BinaryIO


def _start_macos_denial_audit(
    scratch_root: Path, write_roots: Sequence[Path]
) -> _MacOSAuditChannel:
    """Start the parent-controlled macOS Seatbelt denial collector.

    Parameters
    ----------
    scratch_root:
        Supervisor scratch root used to place a non-child-writable sibling channel.
    write_roots:
        Every root writable by the sandboxed child.

    Returns
    -------
    _MacOSAuditChannel
        Live parent-owned collector and immutable channel identity.

    Raises
    ------
    SandboxUnavailableError
        If the unified-log audit API cannot be started.
    """

    log_executable = shutil.which("log")
    if log_executable is None:
        raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
    path, identity = _parent_owned_audit_path(
        scratch_root,
        write_roots,
        filename="macos-seatbelt.ndjson",
    )
    handle = path.open("wb")
    try:
        process = subprocess.Popen(
            (
                log_executable,
                "stream",
                "--style",
                "ndjson",
                "--predicate",
                'eventMessage CONTAINS[c] "deny"',
            ),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.DEVNULL,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )
        time.sleep(0.05)
        if process.poll() is not None:
            raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
    except (OSError, SandboxUnavailableError):
        handle.close()
        raise
    return _MacOSAuditChannel(path, identity, process, handle)


def _finish_macos_denial_audit(channel: _MacOSAuditChannel) -> None:
    """Stop a macOS collector and append the parent completion marker if trustworthy.

    Parameters
    ----------
    channel:
        Live parent-owned collector state.
    """

    completed = False
    try:
        channel.process.terminate()
        return_code = channel.process.wait(timeout=5)
        completed = return_code in {0, -signal.SIGTERM}
    except (OSError, subprocess.TimeoutExpired):
        _kill_process_group(channel.process)
        channel.process.wait()
    finally:
        channel.handle.flush()
        os.fsync(channel.handle.fileno())
        channel.handle.close()
    if not completed:
        return
    try:
        status = channel.path.stat()
        if (status.st_dev, status.st_ino) != channel.expected_identity:
            return
        with channel.path.open("ab") as handle:
            if status.st_size > 0:
                with channel.path.open("rb") as read_handle:
                    read_handle.seek(-1, os.SEEK_END)
                    if read_handle.read(1) != b"\n":
                        handle.write(b"\n")
            handle.write((_MACOS_AUDIT_COMPLETION_MARKER + "\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        return


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
    checkpoint_paths = tuple(
        dict.fromkeys(path for observation in observations for path in observation.checkpoint_paths)
    )
    telemetry_failures = tuple(
        dict.fromkeys(
            observation.telemetry_failure
            for observation in observations
            if observation.telemetry_failure is not None
        )
    )
    return SandboxDenialObservation(
        network_attempted=any(observation.network_attempted for observation in observations),
        socket_targets=targets,
        write_outside_scratch_attempted=any(
            observation.write_outside_scratch_attempted for observation in observations
        ),
        write_paths=paths,
        checkpoint_or_weight_read_attempted=any(
            observation.checkpoint_or_weight_read_attempted for observation in observations
        ),
        checkpoint_paths=checkpoint_paths,
        telemetry_failure=";".join(telemetry_failures) if telemetry_failures else None,
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
    policy["checkpoint_or_weight_read_attempted"] = (
        bool(policy.get("checkpoint_or_weight_read_attempted"))
        or denial.checkpoint_or_weight_read_attempted
    )
    policy["checkpoint_paths"] = list(
        dict.fromkeys([*policy.get("checkpoint_paths", []), *denial.checkpoint_paths])
    )
    payload["policy_observation"] = policy
    if denial.network_attempted:
        reason_code = "network-attempt"
    elif denial.write_outside_scratch_attempted:
        reason_code = "write-outside-scratch"
    else:
        reason_code = "checkpoint-read"
    error = {
        "reason_code": reason_code,
        "exception_type": ("menagerie.crawler.worker_supervisor.SandboxDenialObservation"),
        "message": (
            "parent-owned syscall telemetry observed a forbidden operation or failed integrity"
        ),
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
    allowed_read_paths = _request_allowed_read_paths(argv)
    safe_environment[_PARENT_ALLOWED_READ_PATHS_ENV] = json.dumps(
        [str(path) for path in allowed_read_paths],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    sandbox = detect_os_sandbox()
    if sandbox is None:
        raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
    profile_path: Optional[Path] = None
    linux_runtime_code_roots: tuple[Path, ...] = ()
    linux_runtime_read_paths: tuple[Path, ...] = ()
    if sandbox.kind == "sandbox-exec":
        profile_path = scratch_root / "worker-sandbox.sb"
        profile_path.write_text(
            generate_macos_sandbox_profile(
                write_roots,
                allowed_read_paths=allowed_read_paths,
                runtime_read_roots=_runtime_read_roots(argv, working_directory),
            ),
            encoding="utf-8",
        )
    elif sandbox.kind == "bubblewrap":
        linux_runtime_code_roots = _linux_runtime_code_roots(argv, working_directory)
        linux_runtime_read_paths = _linux_runtime_read_paths(argv)
    sandboxed_argv = wrap_with_os_sandbox(
        sandbox,
        argv,
        working_directory,
        write_roots,
        macos_profile_path=profile_path,
        allowed_read_paths=allowed_read_paths,
    )
    denial_audit_path: Optional[Path] = None
    denial_audit_identity: Optional[tuple[int, int]] = None
    macos_audit_channel: Optional[_MacOSAuditChannel] = None
    if sandbox.kind == "bubblewrap":
        denial_audit_executable = shutil.which("strace")
        if denial_audit_executable is None:
            raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
        denial_audit_path, denial_audit_identity = _parent_owned_audit_path(
            scratch_root, write_roots
        )
        sandboxed_argv = _linux_audited_argv(
            sandboxed_argv,
            denial_audit_executable,
            denial_audit_path,
        )
    elif sandbox.kind == "sandbox-exec":
        macos_audit_channel = _start_macos_denial_audit(scratch_root, write_roots)
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    timed_out = False
    rss_exceeded = False
    peak_rss = 0
    try:
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
    finally:
        if macos_audit_channel is not None:
            _finish_macos_denial_audit(macos_audit_channel)
    wall_seconds = time.monotonic() - started
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = max(0.0, _rusage_seconds(usage_after) - _rusage_seconds(usage_before))
    peak_rss = max(peak_rss, int(usage_after.ru_maxrss) * 1024)
    return_code = process.returncode
    signal_number = -return_code if return_code is not None and return_code < 0 else None
    exit_code = return_code if return_code is not None and return_code >= 0 else None
    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_bytes()
    denial = SandboxDenialObservation()
    if macos_audit_channel is not None:
        denial = _parse_macos_denial_audit(
            macos_audit_channel.path,
            expected_identity=macos_audit_channel.expected_identity,
        )
    elif denial_audit_path is not None:
        denial = _parse_linux_denial_audit(
            denial_audit_path,
            working_directory,
            write_roots,
            expected_identity=denial_audit_identity,
            allowed_read_paths=(*allowed_read_paths, *linux_runtime_read_paths),
            runtime_code_roots=linux_runtime_code_roots,
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
