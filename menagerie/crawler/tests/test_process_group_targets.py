"""Prove every wrapper teardown signals only a process group it can prove it owns.

``os.killpg(process.pid, ...)`` uses a PID as a PGID. That is correct only while
the child happens to lead its own group and its PID has not been recycled; when
either stops holding, the call signals whatever group happens to hold that ID.
These tests exercise both directions: a legitimate group is still torn down
completely, and an unverifiable one is refused -- with a control that shows the
refused signal really would have killed a bystander.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Optional

import pytest

from menagerie.crawler import author_executor as author_executor_module
from menagerie.crawler import operator_checker as operator_checker_module
from menagerie.crawler import worker_supervisor as worker_supervisor_module

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]
_SLEEPER = "import time; time.sleep(120)"
# Used where a refusal must NOT leave the caller draining a live child: short
# enough that a regression which reintroduces the unbounded drain fails on a real
# assertion within a minute instead of hanging the suite indefinitely.
_BOUNDED_SLEEPER = "import time; time.sleep(20)"


def _await_exit(pid: int, timeout_seconds: float = 10.0) -> bool:
    """Wait for an unrelated process to disappear.

    Parameters
    ----------
    pid:
        Process to watch. Must not be an unreaped child of this process, whose
        zombie would keep answering signal-zero probes.
    timeout_seconds:
        Bound on the wait.

    Returns
    -------
    bool
        True when the process is gone before the deadline.
    """

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


class _ReapedChildStub:
    """A ``Popen``-shaped child this parent has already reaped.

    Once a parent reaps its child, the kernel is free to hand that PID to anyone,
    so the PID may now name a stranger who leads a real process group. That state
    is unreachable with a live child and is the exact shape of the PID-reuse
    hazard, so it is constructed directly here rather than approximated.

    Parameters
    ----------
    pid:
        PID the parent last knew as its child's -- here, deliberately a PID that
        now belongs to an unrelated group leader.
    """

    def __init__(self, pid: int) -> None:
        """Bind the recycled PID and start with a completed return code."""

        self.pid = pid
        self.returncode: Optional[int] = 0
        self.signals: list[int] = []

    def terminate(self) -> None:
        """Record a single-process SIGTERM instead of delivering one."""

        self.signals.append(signal.SIGTERM)

    def kill(self) -> None:
        """Record a single-process SIGKILL instead of delivering one."""

        self.signals.append(signal.SIGKILL)

    def communicate(self, timeout: Optional[float] = None) -> tuple[str, str]:
        """Return the empty streams a reaped child would have left behind.

        Parameters
        ----------
        timeout:
            Ignored; a reaped child never blocks.

        Returns
        -------
        tuple[str, str]
            Empty stdout and stderr.
        """

        del timeout
        return "", ""


def _spawn_group_leader() -> subprocess.Popen[bytes]:
    """Spawn a long-lived bystander that leads its own process group.

    Returns
    -------
    subprocess.Popen[bytes]
        Running leader whose PGID equals its PID.
    """

    return subprocess.Popen(
        [sys.executable, "-c", _SLEEPER],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_author_session_teardown_refuses_a_recycled_pid_that_names_a_foreign_group(
    request: pytest.FixtureRequest,
) -> None:
    """A PID that now leads a stranger's group must never be signalled."""

    victim = _spawn_group_leader()

    def reap_victim() -> None:
        """Tear the surviving bystander down once the assertions are done."""

        victim.kill()
        victim.wait(timeout=5)

    request.addfinalizer(reap_victim)
    assert os.getpgid(victim.pid) == victim.pid

    stub = _ReapedChildStub(victim.pid)
    group = worker_supervisor_module.capture_process_group(stub)  # type: ignore[arg-type]
    # The stub's PID does lead a group, so isolation alone would have let this
    # through; the reaped-child fact is what refuses it.
    assert group.isolated is True

    stdout, stderr = author_executor_module._kill_session(group)  # noqa: SLF001

    assert (stdout, stderr) == ("", "")
    # Survival is asserted over a real interval, not sampled immediately: a
    # bystander that has just been signalled is still briefly alive, so an
    # instantaneous liveness probe would pass even for the unguarded call. The
    # control test below kills its own bystander in well under this window.
    with pytest.raises(subprocess.TimeoutExpired):
        victim.wait(timeout=2.0)
    # Only the "child" this parent still nominally owns was signalled at all --
    # a no-op on an already-reaped process.
    assert stub.signals == [signal.SIGTERM]


def test_the_refused_signal_would_really_have_killed_the_bystander(
    request: pytest.FixtureRequest,
) -> None:
    """The control for the refusal above: the old call is genuinely lethal.

    Without this, a passing refusal test proves only that nothing happened, which
    is also what a broken test proves.
    """

    victim = _spawn_group_leader()
    request.addfinalizer(lambda: victim.wait(timeout=5))
    stub = _ReapedChildStub(victim.pid)

    # Exactly the pre-guard call: a raw PID passed where a PGID is expected.
    started = time.monotonic()
    os.killpg(stub.pid, signal.SIGTERM)

    # ``wait``, not a signal-zero probe: the victim is this process's own child,
    # so its PID stays answerable as a zombie until it is reaped.
    assert victim.wait(timeout=10) == -signal.SIGTERM
    # Well inside the survival window the refusal test asserts, so that test's
    # two-second silence is a real refusal rather than a slow kill.
    assert time.monotonic() - started < 2.0


def test_author_session_teardown_refuses_a_child_that_does_not_lead_its_own_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child sharing a foreign group is torn down alone, never by group."""

    # No ``start_new_session``: this child joins the test runner's own group, so
    # its PID is provably not a PGID it leads.
    child = subprocess.Popen(
        [sys.executable, "-c", _SLEEPER],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    group = worker_supervisor_module.capture_process_group(child)
    assert group.isolated is False
    assert os.getpgid(child.pid) != child.pid

    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, signum: signalled.append((pgid, signum)))

    author_executor_module._kill_session(group)  # noqa: SLF001

    assert signalled == []
    assert child.wait(timeout=10) == -signal.SIGTERM


def test_author_session_teardown_still_kills_a_verified_group_completely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The permitting direction: a legitimate group dies whole, not just its root."""

    marker = tmp_path / "grandchild.pid"
    program = (
        "import subprocess, sys, time, pathlib\n"
        f"child = subprocess.Popen([sys.executable, '-c', {_SLEEPER!r}])\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n"
        "time.sleep(120)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", program],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not marker.is_file():
        time.sleep(0.02)
    assert marker.is_file(), "grandchild never announced itself"
    grandchild_pid = int(marker.read_text())

    real_killpg = os.killpg
    signalled: list[tuple[int, int]] = []

    def spy(pgid: int, signum: int) -> None:
        """Record then deliver a real group signal.

        Parameters
        ----------
        pgid, signum:
            Targeted process group and signal.
        """

        signalled.append((pgid, signum))
        real_killpg(pgid, signum)

    monkeypatch.setattr(os, "killpg", spy)

    author_executor_module._kill_session(group=worker_supervisor_module.capture_process_group(child))  # noqa: SLF001

    assert signalled == [(child.pid, signal.SIGTERM)]
    assert child.wait(timeout=10) == -signal.SIGTERM
    assert _await_exit(grandchild_pid) is True


def test_checker_timeout_teardown_refuses_a_child_that_does_not_lead_its_own_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The checker's timeout path must also refuse an unverifiable group."""

    real_popen = subprocess.Popen

    def popen_without_a_new_session(*args: Any, **kwargs: Any) -> Any:
        """Spawn without session isolation so the child shares a foreign group.

        Parameters
        ----------
        args, kwargs:
            Forwarded ``subprocess.Popen`` arguments.

        Returns
        -------
        subprocess.Popen
            Child sharing the test runner's process group.
        """

        kwargs["start_new_session"] = False
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen_without_a_new_session)
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, signum: signalled.append((pgid, signum)))

    attempt = operator_checker_module._invoke_codex(  # noqa: SLF001
        (sys.executable, "-c", _BOUNDED_SLEEPER),
        tmp_path / "last-message.json",
        0.5,
    )

    assert signalled == []
    assert attempt.timed_out is True
    # The refusal must not leave the wrapper draining a live child forever: the
    # root child is still unambiguously ours, so it is killed on its own.
    assert attempt.returncode == -signal.SIGKILL


def test_checker_timeout_teardown_still_kills_a_verified_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The permitting direction for the checker's timeout teardown."""

    real_killpg = os.killpg
    signalled: list[tuple[int, int]] = []

    def spy(pgid: int, signum: int) -> None:
        """Record then deliver a real group signal.

        Parameters
        ----------
        pgid, signum:
            Targeted process group and signal.
        """

        signalled.append((pgid, signum))
        real_killpg(pgid, signum)

    monkeypatch.setattr(os, "killpg", spy)

    attempt = operator_checker_module._invoke_codex(  # noqa: SLF001
        (sys.executable, "-c", _SLEEPER),
        tmp_path / "last-message.json",
        0.5,
    )

    assert attempt.timed_out is True
    assert [signum for _pgid, signum in signalled] == [signal.SIGKILL]
    assert attempt.returncode == -signal.SIGKILL


def _killpg_call_sites() -> set[tuple[str, str]]:
    """Locate every production ``os.killpg`` call in the crawler package.

    Returns
    -------
    set[tuple[str, str]]
        ``(file name, enclosing function)`` for each call site.
    """

    sites: set[tuple[str, str]] = set()
    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        enclosing: dict[ast.AST, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for descendant in ast.walk(node):
                    enclosing.setdefault(descendant, node.name)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "killpg"
            ):
                sites.add((path.name, enclosing.get(node, "<module>")))
    return sites


def test_raw_group_signalling_is_confined_to_the_one_verified_routine() -> None:
    """No production module may signal a process group it has not verified.

    A closed inventory rather than a spot check: the two wrapper regressions this
    change fixes were both new ``os.killpg`` call sites written beside the
    hardened one, and only an exhaustive assertion catches the third.
    """

    assert _killpg_call_sites() == {
        ("worker_supervisor.py", "signal_verified_process_group"),
    }
