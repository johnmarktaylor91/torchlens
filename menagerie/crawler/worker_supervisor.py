"""Parent-owned argv-only subprocess isolation and resource observations."""

from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.constants import DEFAULT_FORWARD_TIMEOUT_SECONDS, STDIO_TAIL_MAX_CHARS
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.policy import build_safe_environment


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


def run_isolated_subprocess(
    argv: Sequence[str],
    scratch_root: Path,
    *,
    timeout_seconds: float = DEFAULT_FORWARD_TIMEOUT_SECONDS,
    rss_limit_bytes: int = 12 * 1024**3,
    cwd: Optional[Path] = None,
    base_environment: Optional[Mapping[str, str]] = None,
) -> SupervisorObservation:
    """Launch a fresh credential-scrubbed subprocess without a shell.

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

    Returns
    -------
    SupervisorObservation
        Parent-only process and resource facts.
    """

    if not argv or any(not isinstance(value, str) or "\x00" in value for value in argv):
        raise ValueError("argv must contain non-empty NUL-free strings")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    scratch_root.mkdir(parents=True, exist_ok=True)
    stdout_path = scratch_root / "stdout.log"
    stderr_path = scratch_root / "stderr.log"
    safe_environment = build_safe_environment(scratch_root, base_environment=base_environment)
    working_directory = (cwd or Path.cwd()).resolve()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    timed_out = False
    rss_exceeded = False
    peak_rss = 0
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            list(argv),
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
    return SupervisorObservation(
        argv=tuple(argv),
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
    observation = run_isolated_subprocess(
        argv,
        scratch_root,
        timeout_seconds=timeout_seconds,
        rss_limit_bytes=rss_limit_bytes,
        cwd=cwd,
    )
    receipt, receipt_error = _load_receipt(receipt_path)
    return SupervisedResult(observation, receipt, receipt_error)
