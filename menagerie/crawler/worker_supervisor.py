"""Parent-owned argv-only subprocess isolation and resource observations."""

from __future__ import annotations

import json
import os
import re
import resource
import secrets
import signal
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
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
    _runtime_package_data_paths,
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
_OPEN_RESULT_PATTERN = re.compile(r"\)\s+=\s+(-?\d+)(?:<[^>]*>)?(?:\s|$)")
_RESUMED_TRACE_PATTERN = re.compile(
    r"^(?P<leader>\s*(?:(?:\[pid\s+)?(?P<pid>\d+)\]?\s+)?)"
    r"<\.\.\. (?P<syscall>[a-z][a-z0-9_]*) resumed>(?P<suffix>.*)$"
)
_TRACE_PID_PATTERN = re.compile(r"^\s*(?:\[pid\s+)?(?P<pid>\d+)\]?\s+")
_NON_FILE_DESCRIPTOR_PREFIXES = ("anon_inode:", "memfd:", "pipe:", "socket:")
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
_MACOS_AUDIT_SENTINEL_PREFIX = "/private/var/empty/.menagerie-seatbelt-audit-"
_PARENT_COMPLETION_CHALLENGE_ENV = "MENAGERIE_PARENT_COMPLETION_CHALLENGE"
_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V1 "


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
    failed_read_probe_paths:
        Undeclared read-only opens that failed before returning a descriptor.
    success_attestation_sha256, attested_receipt_sha256:
        Parent-owned success attestation and the exact child receipt it witnessed.
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
    failed_read_probe_paths: tuple[str, ...] = ()
    success_attestation_sha256: Optional[str] = None
    attested_receipt_sha256: Optional[str] = None

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
            "failed_read_probe_paths": list(self.failed_read_probe_paths),
            "success_attestation_sha256": self.success_attestation_sha256,
            "attested_receipt_sha256": self.attested_receipt_sha256,
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
    success_attestation_sha256:
        Parent-owned proof of a normally completed worker success path.
    """

    observation: SupervisorObservation
    worker_receipt: Optional[dict[str, Any]]
    receipt_error: Optional[str]
    success_attestation_sha256: Optional[str] = None


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
    failed_read_probe_paths:
        Undeclared read-only opens that returned a negative file descriptor. These
        retain diagnostic value but do not poison a receipt because no bytes were read.
    telemetry_failure:
        Fail-closed broker-integrity diagnostic, if telemetry was not trustworthy.
    """

    network_attempted: bool = False
    socket_targets: tuple[str, ...] = ()
    write_outside_scratch_attempted: bool = False
    write_paths: tuple[str, ...] = ()
    checkpoint_or_weight_read_attempted: bool = False
    checkpoint_paths: tuple[str, ...] = ()
    failed_read_probe_paths: tuple[str, ...] = ()
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


def _verified_worker_completion(stdout: bytes, challenge: str) -> Optional[dict[str, str]]:
    """Verify the final normal-completion line against a parent-only challenge.

    Parameters
    ----------
    stdout:
        Exact worker stdout bytes observed by the parent.
    challenge:
        Fresh challenge removed from the worker environment before model code runs.

    Returns
    -------
    dict[str, str] | None
        Bound child receipt digest and proof, or ``None`` when normal completion was
        not witnessed.
    """

    try:
        lines = stdout.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        return None
    if not lines or not lines[-1].startswith(_WORKER_COMPLETION_PREFIX):
        return None
    try:
        value = json.loads(lines[-1][len(_WORKER_COMPLETION_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    receipt_sha256 = value.get("receipt_sha256")
    proof = value.get("proof")
    if not isinstance(receipt_sha256, str) or not isinstance(proof, str):
        return None
    expected = stable_hash(
        {
            "version": "menagerie.crawler.worker-completion.v1",
            "challenge": challenge,
            "receipt_sha256": receipt_sha256,
        }
    )
    if proof != expected:
        return None
    return {"receipt_sha256": receipt_sha256, "completion_line": lines[-1]}


def parent_success_attestation_sha256(completion_line: str, observation: Mapping[str, Any]) -> str:
    """Hash one parent-owned success attestation from exact process facts.

    Parameters
    ----------
    completion_line:
        Challenge-verified final worker completion line.
    observation:
        Exact persisted parent process facts.

    Returns
    -------
    str
        Domain-separated parent attestation digest.
    """

    return stable_hash(
        {
            "version": "menagerie.crawler.parent-success-attestation.v1",
            "completion_line": completion_line,
            "exit_code": observation.get("exit_code"),
            "signal": observation.get("signal"),
            "wall_seconds": observation.get("wall_seconds"),
            "cpu_seconds": observation.get("cpu_seconds"),
            "peak_rss_bytes": observation.get("peak_rss_bytes"),
            "stdout_sha256": observation.get("stdout_sha256"),
            "stderr_sha256": observation.get("stderr_sha256"),
        }
    )


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
        (
            "trace=%network,%file,%process,mmap,read,pread64,readv,preadv,preadv2,"
            "readlinkat,name_to_handle_at,open_by_handle_at"
        ),
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
        if executable.is_file():
            allowed.append(executable)
    try:
        request_index = argv.index("--request") + 1
        request_path = Path(argv[request_index]).resolve()
    except (ValueError, IndexError):
        return tuple(dict.fromkeys(allowed))
    if request_path.is_file():
        allowed.append(request_path)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return tuple(dict.fromkeys(allowed))
    if not isinstance(request, Mapping):
        return tuple(dict.fromkeys(allowed))
    standard_asset = _request_standard_asset(request)
    if standard_asset is not None and standard_asset.is_file():
        allowed.append(standard_asset)
    recipe = request.get("recipe")
    if isinstance(recipe, Mapping):
        adapter_path = recipe.get("path")
        if isinstance(adapter_path, str) and adapter_path:
            resolved_adapter = Path(adapter_path).resolve()
            if resolved_adapter.is_file():
                allowed.append(resolved_adapter)
            for directory in (resolved_adapter.parent, *resolved_adapter.parents[:4]):
                sidecar = directory / "driver-author-artifact.json"
                if sidecar.is_file():
                    allowed.append(sidecar)
                    break
    input_manifest = request.get("input_manifest")
    if isinstance(input_manifest, Mapping):
        model_root_value = input_manifest.get("validated_model_root")
        input_path_value = input_manifest.get("validated_input_code_path")
        if isinstance(model_root_value, str) and isinstance(input_path_value, str):
            model_root = Path(model_root_value)
            input_path = Path(input_path_value)
            if (
                model_root.is_absolute()
                and model_root.is_dir()
                and input_path.is_absolute()
                and input_path.is_file()
                and input_path.resolve().is_relative_to(model_root.resolve())
            ):
                allowed.append(input_path)
    # input_contract is author-authored. Its code_path and scalar leaves are never
    # promoted into a parent read grant; accepted adapter closure bytes already cover
    # every executable model-local helper the worker may load.
    return tuple(
        dict.fromkeys(path.resolve() for path in allowed if path.is_absolute() and path.is_file())
    )


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


def _trace_process_id(line: str) -> str:
    """Return the strace process identifier or a single-process sentinel.

    Parameters
    ----------
    line:
        One syscall telemetry line.

    Returns
    -------
    str
        Stable key used to pair unfinished and resumed syscall records.
    """

    match = _TRACE_PID_PATTERN.match(line)
    return match.group("pid") if match is not None else "<single-process>"


def _complete_trace_records(lines: Sequence[str]) -> Optional[tuple[str, ...]]:
    """Join interleaved strace unfinished/resumed records without losing results.

    Parameters
    ----------
    lines:
        Integrity-checked raw strace lines.

    Returns
    -------
    tuple[str, ...] | None
        Complete records, or ``None`` when continuation telemetry is inconsistent.
    """

    pending: dict[tuple[str, str], str] = {}
    completed: list[str] = []
    for line in lines:
        if "<unfinished ...>" in line:
            syscall = _syscall_name(line)
            if syscall is None:
                return None
            key = (_trace_process_id(line), syscall)
            if key in pending:
                return None
            pending[key] = line.replace("<unfinished ...>", "", 1).rstrip()
            continue
        resumed = _RESUMED_TRACE_PATTERN.match(line)
        if resumed is not None:
            key = (_trace_process_id(line), resumed.group("syscall"))
            prefix = pending.pop(key, None)
            if prefix is None:
                return None
            completed.append(f"{prefix}{resumed.group('suffix')}")
            continue
        completed.append(line)
    return None if pending else tuple(completed)


def _read_only_open_result(line: str) -> Optional[int]:
    """Return the file descriptor result from one complete read-only open.

    Parameters
    ----------
    line:
        Complete ``open``, ``openat``, or ``openat2`` trace record.

    Returns
    -------
    int | None
        Nonnegative descriptor for success, negative result for failure, or ``None``
        when the supposedly complete result is unparsable.
    """

    matches = _OPEN_RESULT_PATTERN.findall(line)
    return int(matches[-1]) if matches else None


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
        if expected_identity is not None:
            anchor_status = anchor.stat()
            if (anchor_status.st_dev, anchor_status.st_ino) != (status.st_dev, status.st_ino):
                return _telemetry_failure_observation("replaced")
        content = audit_path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return _telemetry_failure_observation("unparsable-encoding")
    except OSError:
        return _telemetry_failure_observation("missing")
    raw_lines = content.splitlines()
    if not raw_lines:
        return _telemetry_failure_observation("empty")
    if not _TERMINAL_TRACE_PATTERN.search(raw_lines[-1]):
        return _telemetry_failure_observation("truncated")
    if any(not _trace_line_is_well_formed(line) for line in raw_lines if line.strip()):
        return _telemetry_failure_observation("unparsable-record")
    completed_lines = _complete_trace_records(raw_lines)
    if completed_lines is None:
        return _telemetry_failure_observation("unparsable-continuation")
    socket_targets: list[str] = []
    write_paths: list[str] = []
    checkpoint_paths: list[str] = []
    failed_read_probe_paths: list[str] = []
    for line in completed_lines:
        syscall = _syscall_name(line)
        if syscall in {"connect", "send", "sendmsg", "sendmmsg", "sendto"} and (
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
                allowed = _read_path_is_allowed(
                    path_text,
                    cwd,
                    write_roots,
                    allowed_read_paths,
                    runtime_code_roots,
                    directory_only="O_DIRECTORY" in line,
                )
                if allowed:
                    continue
                result = _read_only_open_result(line)
                if result is None:
                    return _telemetry_failure_observation("unparsable-open-result")
                checkpoint_paths.append(path_text)
                if result < 0:
                    failed_read_probe_paths.append(path_text)
            continue
        if syscall in {"readlinkat", "name_to_handle_at"}:
            paths = _decoded_trace_paths(line)
            if paths and not _read_path_is_allowed(
                paths[0],
                cwd,
                write_roots,
                allowed_read_paths,
                runtime_code_roots,
            ):
                checkpoint_paths.append(paths[0])
            continue
        if syscall == "open_by_handle_at":
            checkpoint_paths.append("<open_by_handle_at:undeclared-file-handle>")
            continue
        if syscall == "mmap":
            descriptor_path = re.search(r"\b\d+<([^>]+)>", line)
            descriptor_path_text = descriptor_path.group(1) if descriptor_path is not None else None
            if (
                descriptor_path_text is not None
                and not descriptor_path_text.startswith(_NON_FILE_DESCRIPTOR_PREFIXES)
                and not _read_path_is_allowed(
                    descriptor_path_text,
                    cwd,
                    write_roots,
                    allowed_read_paths,
                    runtime_code_roots,
                )
            ):
                checkpoint_paths.append(descriptor_path_text)
            continue
        if syscall in {"read", "pread64", "readv", "preadv", "preadv2"}:
            descriptor_path = re.search(r"\b\d+<([^>]+)>", line)
            descriptor_path_text = descriptor_path.group(1) if descriptor_path is not None else None
            if (
                descriptor_path_text is not None
                and not descriptor_path_text.startswith(_NON_FILE_DESCRIPTOR_PREFIXES)
                and not _read_path_is_allowed(
                    descriptor_path_text,
                    cwd,
                    write_roots,
                    allowed_read_paths,
                    runtime_code_roots,
                )
            ):
                checkpoint_paths.append(descriptor_path_text)
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
    unique_failed_probes = tuple(dict.fromkeys(failed_read_probe_paths))
    return SandboxDenialObservation(
        network_attempted=bool(unique_targets),
        socket_targets=unique_targets,
        write_outside_scratch_attempted=bool(unique_paths),
        write_paths=unique_paths,
        checkpoint_or_weight_read_attempted=bool(unique_checkpoints),
        checkpoint_paths=unique_checkpoints,
        failed_read_probe_paths=unique_failed_probes,
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


def _macos_denial_process_ids(line: str) -> frozenset[int]:
    """Extract process and parent identifiers from one macOS log record.

    Parameters
    ----------
    line:
        Unified-log NDJSON or legacy Seatbelt text.

    Returns
    -------
    frozenset[int]
        Process identifiers explicitly associated with the record.
    """

    stripped = line.strip()
    identifiers: set[int] = set()
    if stripped.startswith("{"):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, Mapping):
            for key in ("processID", "parentProcessID", "pid", "ppid"):
                candidate = value.get(key)
                if isinstance(candidate, int):
                    identifiers.add(candidate)
    identifiers.update(int(value) for value in re.findall(r"\((\d+)\)\s+deny", line))
    return frozenset(identifiers)


def _macos_denial_process_path(line: str) -> Optional[Path]:
    """Extract the denied process image path from one unified-log record.

    Parameters
    ----------
    line:
        Unified-log NDJSON record.

    Returns
    -------
    pathlib.Path | None
        Resolved process image when the record carries one.
    """

    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, Mapping):
        return None
    for key in ("processImagePath", "senderImagePath"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.startswith("/"):
            return Path(candidate).resolve()
    return None


def _macos_denial_audit(
    telemetry: bytes,
    *,
    expected_process_ids: Sequence[int] = (),
    expected_process_roots: Sequence[Path] = (),
    ignored_process_ids: Sequence[int] = (),
) -> SandboxDenialObservation:
    """Parse completion-marked parent-owned macOS Seatbelt telemetry.

    Parameters
    ----------
    telemetry:
        Complete bytes written by the parent-controlled unified-log collector.
    expected_process_ids:
        Sandboxed worker process-tree roots. Records outside this scope are noise.
    expected_process_roots:
        Parent-owned executable/source roots that scope a descendant whose denial record
        omitted its ancestry.
    ignored_process_ids:
        Parent-created audit sentinel processes that prove collector delivery only.

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
    expected_ids = set(expected_process_ids)
    ignored_ids = set(ignored_process_ids)
    process_roots = tuple(path.resolve() for path in expected_process_roots)
    unparseable = False
    for line in records:
        record_ids = _macos_denial_process_ids(line)
        if ignored_ids.intersection(record_ids):
            continue
        if expected_ids or process_roots:
            process_path = _macos_denial_process_path(line)
            scoped_by_path = process_path is not None and any(
                process_path == root or root in process_path.parents for root in process_roots
            )
            if not expected_ids.intersection(record_ids) and not scoped_by_path:
                continue
            # Records may carry only the denied descendant PID. Runtime-root scoping
            # admits that first record; retaining its IDs grows the ancestry closure.
            expected_ids.update(record_ids)
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
        # Seatbelt also emits unrelated denials such as mach-lookup. They are not
        # evidence of the network/file policy classes and must not poison this worker.
        del recognized
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
    expected_process_ids: Sequence[int] = (),
    expected_process_roots: Sequence[Path] = (),
    ignored_process_ids: Sequence[int] = (),
) -> SandboxDenialObservation:
    """Verify and parse one parent-owned macOS Seatbelt audit channel.

    Parameters
    ----------
    audit_path:
        Parent-controlled unified-log output path.
    expected_identity:
        Parent-recorded device/inode pair for replacement detection.
    expected_process_ids:
        Sandboxed process-tree root identifiers used to discard machine-wide noise.
    expected_process_roots:
        Parent-owned roots used to recognize descendant process images.
    ignored_process_ids:
        Parent-owned audit sentinel process identifiers.

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
    return _macos_denial_audit(
        telemetry,
        expected_process_ids=expected_process_ids,
        expected_process_roots=expected_process_roots,
        ignored_process_ids=ignored_process_ids,
    )


@dataclass
class _MacOSAuditChannel:
    """Parent-owned unified-log collector state for one sandboxed process tree."""

    path: Path
    expected_identity: tuple[int, int]
    process: subprocess.Popen[Any]
    handle: BinaryIO
    worker_pid: Optional[int] = None
    sandbox_executable: Optional[str] = None
    profile_path: Optional[Path] = None
    worker_executable: Optional[str] = None
    sentinel_process_ids: list[int] = field(default_factory=list)


def _emit_macos_audit_sentinel(channel: _MacOSAuditChannel, phase: str) -> bool:
    """Emit and observe a parent-owned denied read through one Seatbelt collector.

    Parameters
    ----------
    channel:
        Live collector and exact sandbox launch authority.
    phase:
        ``startup`` or ``post-exit`` lifetime boundary.

    Returns
    -------
    bool
        True only when the collector receives the unique denial within 2.5 seconds.
    """

    if (
        channel.sandbox_executable is None
        or channel.profile_path is None
        or channel.worker_executable is None
    ):
        return False
    sentinel_path = f"{_MACOS_AUDIT_SENTINEL_PREFIX}{secrets.token_hex(16)}-{phase}"
    script = "import sys\ntry:\n open(sys.argv[1], 'rb')\nexcept OSError:\n pass"
    try:
        sentinel = subprocess.Popen(
            (
                channel.sandbox_executable,
                "-f",
                str(channel.profile_path),
                channel.worker_executable,
                "-I",
                "-S",
                "-c",
                script,
                sentinel_path,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            start_new_session=True,
            close_fds=True,
        )
        channel.sentinel_process_ids.append(sentinel.pid)
        if sentinel.wait(timeout=5) != 0:
            return False
    except (OSError, subprocess.TimeoutExpired):
        return False
    deadline = time.monotonic() + 2.5
    marker = sentinel_path.encode("utf-8")
    while time.monotonic() < deadline:
        try:
            if marker in channel.path.read_bytes():
                return True
        except OSError:
            return False
        time.sleep(0.05)
    return False


def _start_macos_denial_audit(
    scratch_root: Path,
    write_roots: Sequence[Path],
    *,
    sandbox_executable: str,
    profile_path: Path,
    worker_executable: str,
) -> _MacOSAuditChannel:
    """Start the parent-controlled macOS Seatbelt denial collector.

    Parameters
    ----------
    scratch_root:
        Supervisor scratch root used to place a non-child-writable sibling channel.
    write_roots:
        Every root writable by the sandboxed child.
    sandbox_executable, profile_path, worker_executable:
        Exact launch authority used for parent-owned collector sentinel probes.

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
    process: Optional[subprocess.Popen[Any]] = None
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
        # Start before the worker and leave an explicit overlap beyond the first
        # ~100 ms, where unified-log startup otherwise races very short-lived children.
        time.sleep(0.2)
        if process.poll() is not None:
            raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
        channel = _MacOSAuditChannel(
            path,
            identity,
            process,
            handle,
            sandbox_executable=sandbox_executable,
            profile_path=profile_path,
            worker_executable=worker_executable,
        )
        if not _emit_macos_audit_sentinel(channel, "startup"):
            raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
    except (OSError, SandboxUnavailableError):
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_group(process)
                process.wait()
        handle.close()
        raise
    return channel


def _finish_macos_denial_audit(channel: _MacOSAuditChannel) -> None:
    """Stop a macOS collector and append the parent completion marker if trustworthy.

    Parameters
    ----------
    channel:
        Live parent-owned collector state.
    """

    completed = False
    drained = False
    try:
        # A denial emitted only after the worker exits proves that log delivery spans
        # the child's entire lifetime. Absence is telemetry-incomplete and receives no
        # terminal marker, so parsing poisons closed.
        drained = _emit_macos_audit_sentinel(channel, "post-exit")
        channel.process.terminate()
        return_code = channel.process.wait(timeout=5)
        completed = drained and return_code in {0, -signal.SIGTERM}
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
    failed_read_probe_paths = tuple(
        dict.fromkeys(
            path for observation in observations for path in observation.failed_read_probe_paths
        )
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
        failed_read_probe_paths=failed_read_probe_paths,
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


def _poison_receipt_for_policy_failure(
    receipt_path: Path,
    reason_code: str,
    *,
    denial: Optional[SandboxDenialObservation] = None,
) -> bool:
    """Atomically poison one valid receipt for a parent-observed policy failure.

    Parameters
    ----------
    receipt_path:
        Atomic worker receipt to audit and, when necessary, replace.
    reason_code:
        Closed policy reason selected from observed flags.
    denial:
        Optional OS-boundary details merged into the worker policy observation.

    Returns
    -------
    bool
        True only when a valid worker receipt was poisoned.
    """

    if not receipt_path.is_file():
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
    if loaded.get("receipt_sha256") != stable_hash(payload):
        return False
    policy_value = payload.get("policy_observation")
    if not isinstance(policy_value, Mapping):
        return False
    policy = dict(policy_value)
    if denial is not None:
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

    if not denial.poisoned:
        return False
    if denial.network_attempted:
        reason_code = "network-attempt"
    elif denial.write_outside_scratch_attempted:
        reason_code = "write-outside-scratch"
    else:
        reason_code = "checkpoint-read"
    return _poison_receipt_for_policy_failure(receipt_path, reason_code, denial=denial)


def _caught_policy_reason(receipt: Mapping[str, Any]) -> Optional[str]:
    """Return the dominant dirty policy reason from one verified worker receipt.

    Parameters
    ----------
    receipt:
        Self-hash-verified child receipt.

    Returns
    -------
    str | None
        Closed policy reason, or ``None`` for a clean observation.
    """

    policy = receipt.get("policy_observation")
    if not isinstance(policy, Mapping):
        return None
    reasons = (
        ("network_attempted", "network-attempt"),
        ("checkpoint_or_weight_read_attempted", "checkpoint-read"),
        ("cache_read_attempted", "checkpoint-read"),
        ("write_outside_scratch_attempted", "write-outside-scratch"),
        ("credentials_present", "credentials-exposed"),
        ("torchlens_import_attempted", "torchlens-import"),
    )
    for field_name, reason_code in reasons:
        if policy.get(field_name):
            return reason_code
    return None


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
    worker_completion_challenge: Optional[str] = None,
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
    worker_completion_challenge:
        Fresh parent secret used only for the standard worker's normal-completion
        attestation. Generic isolated subprocesses leave this unset.

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
    if worker_completion_challenge is not None:
        safe_environment[_PARENT_COMPLETION_CHALLENGE_ENV] = worker_completion_challenge
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
    macos_runtime_read_roots: tuple[Path, ...] = ()
    if sandbox.kind == "sandbox-exec":
        macos_runtime_read_roots = _runtime_read_roots(argv, working_directory)
        runtime_package_data_paths = _runtime_package_data_paths(macos_runtime_read_roots)
        profile_path = scratch_root / "worker-sandbox.sb"
        profile_path.write_text(
            generate_macos_sandbox_profile(
                write_roots,
                allowed_read_paths=(*allowed_read_paths, *runtime_package_data_paths),
                runtime_read_roots=macos_runtime_read_roots,
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
    success_attestation_path: Optional[Path] = None
    success_attestation_identity: Optional[tuple[int, int]] = None
    if worker_completion_challenge is not None:
        success_attestation_path, success_attestation_identity = _parent_owned_audit_path(
            scratch_root,
            write_roots,
            filename="worker-success-attestation.json",
        )
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
        if profile_path is None:
            raise SandboxUnavailableError(FailureStage.SANDBOX_UNAVAILABLE.value)
        macos_audit_channel = _start_macos_denial_audit(
            scratch_root,
            write_roots,
            sandbox_executable=sandbox.executable,
            profile_path=profile_path,
            worker_executable=argv[0],
        )
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
            if macos_audit_channel is not None:
                macos_audit_channel.worker_pid = process.pid
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
            expected_process_ids=(
                (macos_audit_channel.worker_pid,)
                if macos_audit_channel.worker_pid is not None
                else ()
            ),
            expected_process_roots=macos_runtime_read_roots,
            ignored_process_ids=macos_audit_channel.sentinel_process_ids,
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
    parent_observation = {
        "exit_code": exit_code,
        "signal": signal_number,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_rss_bytes": peak_rss,
        "stdout_sha256": hash_bytes(stdout),
        "stderr_sha256": hash_bytes(stderr),
    }
    completion = (
        _verified_worker_completion(stdout, worker_completion_challenge)
        if worker_completion_challenge is not None
        and exit_code == 0
        and signal_number is None
        and not timed_out
        and not rss_exceeded
        else None
    )
    success_attestation_sha256: Optional[str] = None
    attested_receipt_sha256: Optional[str] = None
    if (
        completion is not None
        and success_attestation_path is not None
        and success_attestation_identity is not None
    ):
        success_attestation_sha256 = parent_success_attestation_sha256(
            completion["completion_line"], parent_observation
        )
        attestation = {
            "version": "menagerie.crawler.parent-success-attestation.v1",
            "receipt_sha256": completion["receipt_sha256"],
            "completion_line": completion["completion_line"],
            "observation": parent_observation,
            "attestation_sha256": success_attestation_sha256,
        }
        try:
            status = success_attestation_path.stat()
            if (status.st_dev, status.st_ino) == success_attestation_identity:
                with success_attestation_path.open("r+b") as handle:
                    handle.seek(0)
                    handle.truncate()
                    handle.write(canonical_json_bytes(attestation) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                attested_receipt_sha256 = completion["receipt_sha256"]
            else:
                success_attestation_sha256 = None
        except OSError:
            success_attestation_sha256 = None
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
        failed_read_probe_paths=denial.failed_read_probe_paths,
        success_attestation_sha256=success_attestation_sha256,
        attested_receipt_sha256=attested_receipt_sha256,
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
    completion_challenge = secrets.token_hex(32)
    try:
        observation = run_isolated_subprocess(
            argv,
            scratch_root,
            timeout_seconds=timeout_seconds,
            rss_limit_bytes=rss_limit_bytes,
            cwd=working_directory,
            additional_write_roots=(receipt_path.parent,),
            worker_completion_challenge=completion_challenge,
        )
    except SandboxUnavailableError:
        observation = _sandbox_unavailable_observation(argv, scratch_root, working_directory)
        status = f"failed:{FailureStage.SANDBOX_UNAVAILABLE.value}"
        return SupervisedResult(observation, None, status)
    receipt, receipt_error = _load_receipt(receipt_path)
    caught_policy_reason = _caught_policy_reason(receipt) if receipt is not None else None
    if caught_policy_reason is not None:
        _poison_receipt_for_policy_failure(receipt_path, caught_policy_reason)
        receipt, receipt_error = _load_receipt(receipt_path)
    success_attestation = observation.success_attestation_sha256
    if receipt is not None and observation.attested_receipt_sha256 != receipt.get("receipt_sha256"):
        success_attestation = None
    if observation.exit_code != 0 or observation.signal_number is not None:
        return SupervisedResult(
            observation,
            receipt,
            receipt_error or ("worker-exit-nonzero" if receipt is None else None),
            None,
        )
    if receipt is not None and success_attestation is None:
        return SupervisedResult(observation, receipt, "missing-parent-success-attestation", None)
    return SupervisedResult(observation, receipt, receipt_error, success_attestation)
