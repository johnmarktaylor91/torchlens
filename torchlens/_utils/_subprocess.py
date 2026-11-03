"""The ONE bounded-subprocess spawn discipline for every TorchLens child.

Moved from ``visualization/_render_utils`` (R40): the doctor Graphviz probe
and the bundle git-provenance stamp each hand-rolled ``subprocess.run`` and
leaked orphaned grandchildren on their timeout paths, while this runner's
group teardown already solved that class. Living under ``utils`` keeps it
importable without the visualization package's hard ``graphviz`` dependency.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from collections.abc import Mapping
from typing import Any

# Grace period between SIGTERM and SIGKILL when a timed-out render's whole
# process group is torn down. Graphviz exits promptly on SIGTERM; the
# escalation only matters for a wedged engine (or a plugin it forked) that
# ignores the polite signal.
_KILL_GRACE_SECONDS = 0.5

# POSIX process-group support. ``start_new_session`` needs ``os.setsid`` and
# group teardown needs ``os.killpg``/``os.getpgid``; feature-check instead of
# parsing platform strings so exotic POSIX-likes degrade the same way Windows
# does (leader-only kill, matching the historical ``subprocess.run`` cleanup).
_HAS_PROCESS_GROUPS = all(hasattr(os, name) for name in ("setsid", "killpg", "getpgid"))

# Linux parent-death signal (R40): group teardown runs in the PARENT, so a
# hard parent SIGKILL left the session-leading child running with nothing to
# reap it. PR_SET_PDEATHSIG makes the kernel deliver SIGKILL to the child
# when its parent dies. Resolved HERE, in the parent, because the preexec
# hook runs between fork and exec where imports can deadlock.
_PRCTL: Any = None
if _HAS_PROCESS_GROUPS:
    try:
        import ctypes

        _PRCTL = ctypes.CDLL("libc.so.6").prctl
    except Exception:  # pragma: no cover - non-glibc POSIX
        _PRCTL = None

_PR_SET_PDEATHSIG = 1
# Pre-resolved as plain ints (r8 R40, opus b6): the hook runs between fork
# and exec, where object allocation (a ``contextlib.suppress`` instance, an
# enum ``int()`` coercion) can touch interpreter state that is not
# fork-safe under an allocator lock held by another thread at fork time.
_PDEATHSIG_KILL = 9


def _arm_parent_death_signal() -> None:
    """preexec hook: SIGKILL this child when its parent process dies.

    Runs post-fork pre-exec; only CALLS the pre-resolved libc function with
    pre-resolved int arguments (no imports, no allocation). Best-effort: a
    failure leaves the historical behavior (child survives parent death
    until box reboot).
    """

    if _PRCTL is not None:
        try:
            _PRCTL(_PR_SET_PDEATHSIG, _PDEATHSIG_KILL, 0, 0, 0)
        except Exception:
            pass


# Group-reaper watchdog (r8 R40 arm 2, sol probe): PDEATHSIG protects only
# the DIRECT child -- a plain non-detached grandchild (``sh -c 'sleep &'``,
# a forking dot plugin) survived a hard parent SIGKILL because the group
# teardown runs in the (now dead) parent and the leader's own PDEATHSIG
# death orphans the rest of its group. The reaper is a tiny detached
# ``/bin/sh`` in its OWN session that polls the parent pid; when the parent
# vanishes it SIGKILLs the child's whole process group and exits. It also
# exits on its own as soon as the target group is empty, and the parent
# reaps it explicitly on every normal/exception path.
# NOTE: no ``--`` separator before the negative pgid -- dash's builtin
# ``kill`` rejects it ("Illegal number: -") while accepting the bare
# ``kill -KILL -PGID`` spelling, which bash and busybox accept too.
_REAPER_SCRIPT = (
    "while kill -0 {parent_pid} 2>/dev/null; do "
    "kill -0 -{pgid} 2>/dev/null || exit 0; "
    "sleep 1; "
    "done; "
    "kill -KILL -{pgid} 2>/dev/null"
)


def _spawn_group_reaper(pgid: int) -> subprocess.Popen[Any] | None:
    """Start the parent-death group reaper for one spawned child group."""

    if not _HAS_PROCESS_GROUPS or not os.path.exists("/bin/sh"):
        return None
    try:
        return subprocess.Popen(
            ["/bin/sh", "-c", _REAPER_SCRIPT.format(parent_pid=os.getpid(), pgid=pgid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        # Best-effort belt: spawning the watchdog must never fail a render.
        return None


def _reap_group_reaper(reaper: subprocess.Popen[Any] | None) -> None:
    """Stop and reap the watchdog once the parent has finished teardown."""

    if reaper is None:
        return
    with contextlib.suppress(Exception):
        reaper.kill()
    with contextlib.suppress(Exception):
        reaper.wait(timeout=_KILL_GRACE_SECONDS)


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    """Tear down ``proc`` and every descendant sharing its process group.

    ``subprocess.run``'s timeout cleanup kills only the direct child, so a
    forking ``dot`` (plugin loaders, wrapper scripts) leaked grandchildren
    on every render timeout. SIGTERM the whole group, wait a short grace
    period, escalate to SIGKILL, and reap the leader. Where process groups
    are unavailable (Windows), fall back to the historical leader-only kill.
    """

    pgid: int | None = None
    if _HAS_PROCESS_GROUPS:
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None
    if pgid is None:
        proc.kill()
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=_KILL_GRACE_SECONDS)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):  # SIGKILL always lands
        proc.wait(timeout=_KILL_GRACE_SECONDS)


def run_bounded_subprocess(
    cmd: list[str],
    *,
    timeout: float,
    check: bool = True,
    capture_output: bool = True,
    input: bytes | str | None = None,
    cwd: str | os.PathLike[str] | None = None,
    text: bool = False,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[Any]:
    """Run ``cmd`` bounded by ``timeout``, killing its whole process group.

    The ONE spawn seam for every TorchLens subprocess (Graphviz renders and
    code panels, the doctor ``dot`` probe, the bundle git-provenance stamp).
    Mirrors ``subprocess.run`` semantics for the argument subset those paths
    use (``check`` raises ``CalledProcessError`` with captured stderr;
    timeout raises ``TimeoutExpired``), but on timeout or any other
    exception the entire process group is terminated via
    :func:`_terminate_process_group`, not just the direct child, and the
    child is armed to die with its parent (Linux ``PR_SET_PDEATHSIG``).
    Tests monkeypatch this function to simulate Graphviz outcomes. ``env``
    (default: inherit) replaces the child's environment, as in ``Popen``.
    """

    stdin = subprocess.PIPE if input is not None else None
    pipe = subprocess.PIPE if capture_output else None
    proc = subprocess.Popen(
        cmd,
        stdin=stdin,
        stdout=pipe,
        stderr=pipe,
        cwd=cwd,
        text=text,
        env=env,
        start_new_session=_HAS_PROCESS_GROUPS,
        preexec_fn=_arm_parent_death_signal if _PRCTL is not None else None,
    )
    # ``start_new_session`` makes the child its own group leader, so its pid
    # IS the group id the reaper watches.
    reaper = _spawn_group_reaper(proc.pid) if _HAS_PROCESS_GROUPS else None
    try:
        stdout, stderr = proc.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        # Drain pipes and reap after the group kill, mirroring
        # ``subprocess.run``'s own timeout epilogue.
        with contextlib.suppress(subprocess.TimeoutExpired, ValueError, OSError):
            proc.communicate(timeout=_KILL_GRACE_SECONDS)
        raise
    except BaseException:
        _terminate_process_group(proc)
        raise
    finally:
        _reap_group_reaper(reaper)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
