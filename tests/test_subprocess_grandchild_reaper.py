"""grind-r8 cluster 4.0.8: subprocess grandchildren, arm 2 (R40).

``PR_SET_PDEATHSIG`` protects only the DIRECT child: on a hard parent
SIGKILL the session-leading child dies, but a plain non-detached grandchild
(``sh -c 'sleep &'``, a forking dot plugin) survived with nothing left to
kill its group (sol probe; the group teardown runs in the -- now dead --
parent). Every spawn now also starts a detached group-reaper watchdog in its
own session that SIGKILLs the child's whole process group once the parent
pid vanishes, and exits on its own when the group empties first.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from torchlens._utils._subprocess import _HAS_PROCESS_GROUPS, run_bounded_subprocess

pytestmark = pytest.mark.heavy

posix_only = pytest.mark.skipif(
    not (_HAS_PROCESS_GROUPS and os.path.exists("/bin/sh")),
    reason="needs POSIX process groups and /bin/sh",
)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


@posix_only
def test_normal_run_reaps_its_watchdog_and_group() -> None:
    """A completed run leaves neither the child group nor the watchdog."""

    result = run_bounded_subprocess(["/bin/sh", "-c", "echo done"], timeout=30, text=True)
    assert result.stdout.strip() == "done"
    # No lingering sh children of this process (the watchdog was reaped).
    children = subprocess.run(  # noqa: S603 - test introspection
        ["ps", "--ppid", str(os.getpid()), "-o", "comm="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    assert "sh" not in children


@posix_only
def test_grandchild_dies_after_parent_sigkill(tmp_path) -> None:
    """SIGKILLing the spawning process must not orphan grandchildren."""

    pgid_file = tmp_path / "pgid"
    child_script = tmp_path / "spawner.py"
    child_script.write_text(
        textwrap.dedent(
            f"""
            import os, sys, threading, time
            sys.path.insert(0, {str(os.getcwd())!r})
            from torchlens._utils._subprocess import run_bounded_subprocess

            def run() -> None:
                # sh (group leader, PDEATHSIG-armed) + a backgrounded
                # grandchild sleep that PDEATHSIG does NOT cover.
                run_bounded_subprocess(
                    ["/bin/sh", "-c", "sleep 60 & sleep 60"],
                    timeout=120,
                    check=False,
                )

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            # Find the spawned sh: it is our direct child whose pid == pgid.
            deadline = time.time() + 10
            pgid = None
            while time.time() < deadline and pgid is None:
                for pid in os.listdir("/proc"):
                    if not pid.isdigit():
                        continue
                    try:
                        with open(f"/proc/{{pid}}/stat") as handle:
                            fields = handle.read().split()
                    except OSError:
                        continue
                    if fields[3] == str(os.getpid()) and fields[1] == "(sh)":
                        candidate = int(pid)
                        if os.getpgid(candidate) == candidate:
                            pgid = candidate
                            break
                time.sleep(0.05)
            assert pgid is not None
            with open({str(pgid_file)!r}, "w") as handle:
                handle.write(str(pgid))
            time.sleep(600)  # block until SIGKILLed by the test
            """
        )
    )
    spawner = subprocess.Popen(  # noqa: S603 - test child
        [sys.executable, str(child_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not pgid_file.exists():
            assert spawner.poll() is None, "spawner died before arming"
            time.sleep(0.1)
        assert pgid_file.exists(), "spawner never reported the child pgid"
        pgid = int(pgid_file.read_text())
        assert _group_alive(pgid)

        spawner.send_signal(signal.SIGKILL)
        spawner.wait(timeout=10)

        # PDEATHSIG kills the leader immediately; the reaper's 1s poll must
        # sweep the surviving grandchild within a few seconds.
        deadline = time.time() + 8
        while time.time() < deadline and _group_alive(pgid):
            time.sleep(0.2)
        assert not _group_alive(pgid), (
            "a grandchild survived parent SIGKILL: the group reaper never fired"
        )
    finally:
        if spawner.poll() is None:
            spawner.kill()
            spawner.wait(timeout=10)
        with contextlib.suppress(OSError, ValueError):
            os.killpg(int(pgid_file.read_text()), signal.SIGKILL)
