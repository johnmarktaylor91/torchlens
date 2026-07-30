"""Prove the ambient-signal guard fires, and that it fires only when it should.

A guard exercised only in the direction that stays quiet is not evidence of
anything: silence is also what a wholly disconnected fixture produces. So the
central test here drives a *real* signal through the *real* shutdown handlers in
a subprocess pytest run whose test body passes, and asserts that the run still
fails with the signal named.
"""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import threading

import pytest

from menagerie.crawler.tests.shutdown_signal_guard import (
    declare_self_signal,
    unaccounted_shutdown_signals,
)
from menagerie.crawler.worker_supervisor import (
    _OBSERVED_SHUTDOWN_SIGNAL_LIMIT,
    clear_observed_shutdown_signals,
    observed_shutdown_signals,
    shutdown_signal_handlers,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# A test body that PASSES while a real signal reaches the real handlers. The
# body is deliberately green so the subprocess run can only fail through the
# guard, which is precisely the masquerade being closed: a signal that decides an
# outcome without ever naming itself.
_AMBIENT_SIGNAL_TEST = """
import os
import signal
import threading

from menagerie.crawler.worker_supervisor import shutdown_signal_handlers


def test_body_that_passes_under_an_ambient_signal():
    shutdown_event = threading.Event()
    with shutdown_signal_handlers(shutdown_event):
        os.kill(os.getpid(), signal.{name})
        for _ in range(10000):
            if shutdown_event.is_set():
                break
    assert shutdown_event.is_set()
"""

_DECLARED_SIGNAL_TEST = """
import os
import signal
import threading

from menagerie.crawler.tests.shutdown_signal_guard import declare_self_signal
from menagerie.crawler.worker_supervisor import shutdown_signal_handlers


def test_body_that_declares_its_own_signal():
    shutdown_event = threading.Event()
    with shutdown_signal_handlers(shutdown_event):
        declare_self_signal(signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(10000):
            if shutdown_event.is_set():
                break
    assert shutdown_event.is_set()


def test_body_that_receives_no_signal_at_all():
    assert True
"""

_GUARDED_CONFTEST = """
from menagerie.crawler.tests.shutdown_signal_guard import (  # noqa: F401
    external_shutdown_signal_guard,
)
"""


def _run_guarded_pytest(directory: Path) -> subprocess.CompletedProcess[str]:
    """Run pytest over a throwaway suite that installs only the signal guard.

    Parameters
    ----------
    directory:
        Directory holding the generated conftest and test modules.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed pytest run with merged output captured.
    """

    return subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(directory),
        ),
        cwd=str(_REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGINT"])
def test_guard_fails_a_passing_test_that_an_ambient_signal_reached(
    tmp_path: Path, signal_name: str
) -> None:
    """An undeclared signal must fail the run and name itself and the test."""

    (tmp_path / "conftest.py").write_text(_GUARDED_CONFTEST, encoding="utf-8")
    (tmp_path / "test_ambient_signal.py").write_text(
        _AMBIENT_SIGNAL_TEST.format(name=signal_name), encoding="utf-8"
    )

    completed = _run_guarded_pytest(tmp_path)

    output = completed.stdout + completed.stderr
    # The body itself passed: only the guard can be the source of the failure.
    assert "1 passed" in output, output
    assert completed.returncode != 0, output
    assert f"external {signal_name} received during test" in output, output
    assert "test_ambient_signal.py::test_body_that_passes_under_an_ambient_signal" in output, output


def test_guard_stays_quiet_for_a_declared_signal_and_for_no_signal(tmp_path: Path) -> None:
    """Declared self-signals and signal-free tests must both pass cleanly."""

    (tmp_path / "conftest.py").write_text(_GUARDED_CONFTEST, encoding="utf-8")
    (tmp_path / "test_declared_signal.py").write_text(_DECLARED_SIGNAL_TEST, encoding="utf-8")

    completed = _run_guarded_pytest(tmp_path)

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "2 passed" in output, output
    assert "external" not in output, output


def test_installed_handlers_record_the_delivered_signal_number() -> None:
    """The handler must record which signal arrived, not merely that one did."""

    clear_observed_shutdown_signals()
    shutdown_event = threading.Event()
    try:
        with shutdown_signal_handlers(shutdown_event):
            declare_self_signal(signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGTERM)
            for _ in range(10000):
                if shutdown_event.is_set():
                    break
        assert shutdown_event.is_set()
        assert observed_shutdown_signals() == (signal.SIGTERM,)
    finally:
        clear_observed_shutdown_signals()


def test_signals_delivered_outside_the_handler_window_are_not_recorded() -> None:
    """The ledger describes handler deliveries only, and says so honestly."""

    clear_observed_shutdown_signals()
    shutdown_event = threading.Event()
    with shutdown_signal_handlers(shutdown_event):
        pass
    assert observed_shutdown_signals() == ()


def test_observed_signal_ledger_is_bounded() -> None:
    """A signal storm must not grow the ledger without limit."""

    clear_observed_shutdown_signals()
    shutdown_event = threading.Event()
    try:
        with shutdown_signal_handlers(shutdown_event):
            for _ in range(_OBSERVED_SHUTDOWN_SIGNAL_LIMIT + 10):
                declare_self_signal(signal.SIGTERM)
                os.kill(os.getpid(), signal.SIGTERM)
        assert len(observed_shutdown_signals()) <= _OBSERVED_SHUTDOWN_SIGNAL_LIMIT
    finally:
        clear_observed_shutdown_signals()


def test_unaccounted_signals_are_a_multiset_difference() -> None:
    """One declaration must excuse exactly one delivery, never two."""

    assert unaccounted_shutdown_signals((), ()) == ()
    assert unaccounted_shutdown_signals((signal.SIGTERM,), (signal.SIGTERM,)) == ()
    assert unaccounted_shutdown_signals((signal.SIGTERM,), ()) == (signal.SIGTERM,)
    assert unaccounted_shutdown_signals(
        (signal.SIGTERM, signal.SIGTERM), (signal.SIGTERM,)
    ) == (signal.SIGTERM,)
    assert unaccounted_shutdown_signals((signal.SIGINT,), (signal.SIGTERM,)) == (signal.SIGINT,)


def test_shipped_shutdown_tests_signal_inside_a_forked_child() -> None:
    """The guard's premise: no shipped shutdown test signals the pytest process.

    Every deliberate shutdown test forks before it signals, so its deliveries
    land in the child's ledger. That is what makes an observation in the pytest
    process itself provably external rather than merely presumed to be.
    """

    directory = Path(__file__).resolve().parent
    for name in (
        "test_boundary_shutdown_composition.py",
        "test_release_shutdown_matrix_composition.py",
    ):
        source = (directory / name).read_text(encoding="utf-8")
        assert "os.kill(os.getpid()" in source, name
        assert 'multiprocessing.get_context("fork").Process' in source, name
        assert "declare_self_signal" not in source, name


def test_the_signal_ledger_starts_empty_under_the_autouse_guard() -> None:
    """The guard must hand every test a clean ledger, not a shared accumulator."""

    assert observed_shutdown_signals() == ()


def test_ambient_signal_scope_is_documented_as_handler_windows_only(
    tmp_path: Path,
) -> None:
    """A signal outside any handler window keeps the interpreter's disposition.

    Stated as a test so the guard's boundary is not mistaken for total coverage:
    the guard can only see deliveries a shutdown handler observed, because
    outside that window SIGTERM ends the process and SIGINT raises rather than
    quietly becoming ``interrupted:shutdown``.
    """

    script = textwrap.dedent(
        """
        import os
        import signal

        os.kill(os.getpid(), signal.SIGTERM)
        print("unreachable")
        """
    )
    script_path = tmp_path / "unhandled_sigterm.py"
    script_path.write_text(script, encoding="utf-8")

    completed = subprocess.run(
        (sys.executable, str(script_path)),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == -signal.SIGTERM
    assert "unreachable" not in completed.stdout
