"""Shared subprocess driver for isolated menagerie workers."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class WorkerSubprocessResult:
    """Completed menagerie worker subprocess result."""

    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    peak_rss_mb: int | None = None


def _drain_pipe(stream: Any, sink: list[str]) -> None:
    """Read a worker pipe to EOF into ``sink``.

    Parameters
    ----------
    stream:
        Text pipe returned by ``subprocess.Popen``.
    sink:
        Mutable chunk buffer receiving pipe text.
    """

    try:
        for chunk in iter(stream.readline, ""):
            sink.append(chunk)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _kill_worker_group(proc: subprocess.Popen[str]) -> None:
    """Terminate a worker process group, falling back to the direct child.

    Parameters
    ----------
    proc:
        Worker process created with ``start_new_session=True``.
    """

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    except Exception:
        proc.kill()


def _peak_rss_mb(psutil_proc: Any, current_peak_mb: int | None) -> tuple[Any, int | None]:
    """Sample a worker process tree's resident memory.

    Parameters
    ----------
    psutil_proc:
        ``psutil.Process`` instance for the worker.
    current_peak_mb:
        Highest sampled RSS so far.

    Returns
    -------
    tuple[Any, int | None]
        Possibly disabled psutil process handle and updated peak RSS in MB.
    """

    try:
        rss_bytes = psutil_proc.memory_info().rss
        for child in psutil_proc.children(recursive=True):
            try:
                rss_bytes += child.memory_info().rss
            except Exception:
                continue
        sampled_mb = int(rss_bytes // (1024**2))
        return psutil_proc, max(current_peak_mb or 0, sampled_mb)
    except Exception:
        return None, current_peak_mb


def run_worker_subprocess(
    command: Sequence[str],
    timeout_sec: float,
    env: Mapping[str, str] | None = None,
    poll_interval_sec: float = 0.5,
    sample_peak_rss: bool = False,
) -> WorkerSubprocessResult:
    """Run a menagerie worker in its own process group.

    Parameters
    ----------
    command:
        Worker command.
    timeout_sec:
        Maximum wall time in seconds.
    env:
        Optional child process environment.
    poll_interval_sec:
        Parent polling interval for timeout/RSS checks.
    sample_peak_rss:
        Whether to sample RSS for the worker process tree with ``psutil``.

    Returns
    -------
    WorkerSubprocessResult
        Worker output, return status, timeout flag, and optional peak RSS.
    """

    child_env = dict(env) if env is not None else None
    proc = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env,
        start_new_session=True,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_reader = threading.Thread(
        target=_drain_pipe, args=(proc.stdout, stdout_chunks), daemon=True
    )
    stderr_reader = threading.Thread(
        target=_drain_pipe, args=(proc.stderr, stderr_chunks), daemon=True
    )
    stdout_reader.start()
    stderr_reader.start()
    peak_rss_mb: int | None = None
    psutil_proc = None
    if sample_peak_rss:
        try:
            import psutil

            psutil_proc = psutil.Process(proc.pid)
        except Exception:
            psutil_proc = None

    timeout_start = time.monotonic()
    while proc.poll() is None:
        if psutil_proc is not None:
            psutil_proc, peak_rss_mb = _peak_rss_mb(psutil_proc, peak_rss_mb)
        if time.monotonic() - timeout_start >= timeout_sec:
            _kill_worker_group(proc)
            if hasattr(proc, "communicate"):
                try:
                    proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            else:
                proc.wait()
            stdout_reader.join(timeout=5.0)
            stderr_reader.join(timeout=5.0)
            return WorkerSubprocessResult(
                args=command,
                returncode=proc.returncode if proc.returncode is not None else -signal.SIGKILL,
                stdout="".join(stdout_chunks),
                stderr="".join(stderr_chunks),
                timed_out=True,
                peak_rss_mb=peak_rss_mb,
            )
        time.sleep(poll_interval_sec)

    proc.wait()
    stdout_reader.join()
    stderr_reader.join()
    if peak_rss_mb is None and psutil_proc is not None:
        try:
            peak_rss_mb = int(psutil_proc.memory_info().rss // (1024**2))
        except Exception:
            peak_rss_mb = None
    return WorkerSubprocessResult(
        args=command,
        returncode=proc.returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        timed_out=False,
        peak_rss_mb=peak_rss_mb,
    )
