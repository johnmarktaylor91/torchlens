"""Child-process and fork capture-guard behavior for the global toggle state.

Split from ``test_global_state_inventory.py`` (r7 R43-F1 file-size ratchet
paydown): the guard family that proves a non-rank child process is refused
typed (``child_process_capture_unsupported``), spoofed/inherited rank
evidence never readmits a fork child, the interpreter's importer reclaims a
stolen rank stamp, the main process (and its worker threads) is never
refused, and a fork DURING a traced forward inherits no live logging state.
"""

from __future__ import annotations

import os
import sys
import threading

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state

pytestmark = pytest.mark.smoke  # measured <0.5s per test (W051-GATE, AUD-CODE 0.1)


def test_child_process_capture_refusal_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child-process guard raises a typed, actionable refusal.

    It used to raise a bare ``RuntimeError`` whose message began with
    "WARNING:" — unbranchable by callers and mislabelled as a warning while it
    was in fact a hard refusal. Its docstring also implied it refused THREAD
    concurrency, which it never did (that class is covered by atomic admission,
    the non-owner pause no-op, and DataParallel unwrapping).
    """

    import multiprocessing as mp

    from torchlens.utils.display import warn_parallel

    class _FakeChild:
        """Stand in for a non-rank, non-daemonic child process."""

        name = "Process-1"
        daemon = False

    monkeypatch.setattr(mp, "current_process", lambda: _FakeChild())
    # r-b6 R40-3b: the guard no longer trusts the user-assignable process
    # name — child detection keys on ``parent_process()`` (plus the raw-fork
    # PID stamp), so the fake must present a parent to read as a child.
    monkeypatch.setattr(mp, "parent_process", lambda: _FakeChild())

    with pytest.raises(tl.errors.CaptureContextError) as refusal:
        warn_parallel()

    assert refusal.value.fields["code"] == "child_process_capture_unsupported"
    assert refusal.value.fields["process_name"] == "Process-1"
    assert not str(refusal.value).startswith("WARNING:")


def test_child_process_guard_ignores_spoofed_main_process_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child named "MainProcess" is still refused (r-b6 R40-3b).

    ``process.name`` is a user-assignable constructor kwarg
    (``mp.Process(name="MainProcess")``); the historical name-keyed check let
    such a child capture and corrupt the per-interpreter toggle state.
    """

    import multiprocessing as mp

    from torchlens.utils.display import warn_parallel

    class _SpoofedChild:
        """A child process wearing the main process's name."""

        name = "MainProcess"
        daemon = False

    monkeypatch.setattr(mp, "current_process", lambda: _SpoofedChild())
    monkeypatch.setattr(mp, "parent_process", lambda: _SpoofedChild())

    with pytest.raises(tl.errors.CaptureContextError) as refusal:
        warn_parallel()
    assert refusal.value.fields["code"] == "child_process_capture_unsupported"


def test_child_process_guard_refuses_fork_inherited_group_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inherited initialized-group stamp does not read as a rank (r-b6 R40-3b).

    A forked child of a rank inherits ``dist.is_initialized() == True``; the
    rank carve-out must not readmit it. The PID stamp of the process that
    first observed the group is the discriminator.
    """

    import multiprocessing as mp

    import torchlens.utils.display as display_mod
    from torchlens.utils.display import warn_parallel

    class _FakeChild:
        """A non-daemonic child claiming rank via an inherited group flag."""

        name = "Process-2"
        daemon = False

    class _FakeDist:
        """Distributed module stub reporting an initialized group."""

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

    monkeypatch.setattr(mp, "current_process", lambda: _FakeChild())
    monkeypatch.setattr(mp, "parent_process", lambda: _FakeChild())
    fake_dist = _FakeDist()
    monkeypatch.setattr(torch, "distributed", fake_dist)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)  # type: ignore[arg-type]

    # Positive control: a process that stamps its OWN pid reads as a rank, so
    # the refusal below can only come from the inheritance discriminator.
    monkeypatch.setitem(display_mod._DIST_GROUP_OBSERVED_PID, "pid", os.getpid())
    warn_parallel()

    # The "parent rank" observed the group under a different PID; the fork
    # child inherits that stamp and must be refused. A real fork child also
    # inherits the parent's import-PID value (which differs from its own pid),
    # so the simulation mismatches the import stamp too — otherwise the R40
    # importer-reclaim rule would correctly treat this test process (which IS
    # the interpreter's importer) as the rank.
    monkeypatch.setattr(display_mod, "_WARN_PARALLEL_IMPORT_PID", os.getpid() + 1)
    monkeypatch.setitem(display_mod._DIST_GROUP_OBSERVED_PID, "pid", os.getpid() + 1)
    with pytest.raises(tl.errors.CaptureContextError) as refusal:
        warn_parallel()
    assert refusal.value.fields["code"] == "child_process_capture_unsupported"


def test_child_process_guard_import_pid_rank_reclaims_stolen_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fork child stamping FIRST cannot invert the refusal onto the rank (R40).

    A genuine rank that raw-forks BEFORE its first capture let the child stamp
    ITS pid first via ``setdefault``; the child then read as the "rank" and the
    real rank was refused ``child_process_capture_unsupported``. The
    interpreter's original importer (import-PID process) can never be a fork
    child, so it reclaims the stamp unconditionally; a non-importer process
    holding someone else's stamp stays refused.
    """

    import multiprocessing as mp

    import torchlens.utils.display as display_mod
    from torchlens.utils.display import warn_parallel

    class _FakeRank:
        """The real rank process (spawn-style: fresh import, own import PID)."""

        name = "Process-1"
        daemon = False

    class _FakeDist:
        """Distributed module stub reporting an initialized group."""

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

    monkeypatch.setattr(mp, "current_process", lambda: _FakeRank())
    monkeypatch.setattr(mp, "parent_process", lambda: _FakeRank())
    fake_dist = _FakeDist()
    monkeypatch.setattr(torch, "distributed", fake_dist)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)  # type: ignore[arg-type]

    # This pytest process imported torchlens itself, so it plays the genuine
    # spawn rank (import PID == own pid). A fork child stamped first.
    assert os.getpid() == display_mod._WARN_PARALLEL_IMPORT_PID
    stolen_pid = os.getpid() + 12345
    monkeypatch.setitem(display_mod._DIST_GROUP_OBSERVED_PID, "pid", stolen_pid)

    # The import-PID rank reclaims the stamp and is accepted.
    warn_parallel()
    assert display_mod._DIST_GROUP_OBSERVED_PID["pid"] == os.getpid()

    # A NON-importer process holding someone else's stamp stays refused: the
    # reclaim rule is import-PID-only and never readmits a fork child.
    monkeypatch.setattr(display_mod, "_WARN_PARALLEL_IMPORT_PID", os.getpid() + 1)
    monkeypatch.setitem(display_mod._DIST_GROUP_OBSERVED_PID, "pid", stolen_pid)
    with pytest.raises(tl.errors.CaptureContextError) as refusal:
        warn_parallel()
    assert refusal.value.fields["code"] == "child_process_capture_unsupported"


def test_main_process_capture_is_never_refused() -> None:
    """The guard is a no-op in the main process, including from a worker thread."""

    from torchlens.utils.display import warn_parallel

    warn_parallel()
    errors: list[BaseException] = []

    def call_from_thread() -> None:
        """A capture on a worker thread is legitimate and must not be refused."""

        try:
            warn_parallel()
        except BaseException as error:  # pragma: no cover - reported by the test
            errors.append(error)

    worker = threading.Thread(target=call_from_thread)
    worker.start()
    worker.join(timeout=5.0)
    assert errors == [], f"a main-process worker thread was refused: {errors!r}"


@pytest.mark.skipif(
    not hasattr(os, "fork") or not hasattr(os, "register_at_fork"),
    reason="fork hygiene needs os.fork + os.register_at_fork",
)
def test_fork_child_inherits_no_mid_capture_logging_state() -> None:
    """b8-sol R56-8: a fork DURING a traced forward must not keep logging.

    The forking thread's ident is preserved as the child's main thread, so an
    inherited ``_logging_enabled=True`` + ``_active_trace`` passed the
    owner-thread gate and child-side torch ops logged into the child's
    inherited trace copy (child-local corruption; the parent is unaffected
    through COW). The ``os.register_at_fork`` hygiene handler clears both in
    the child; new captures in the child stay governed by the existing
    child-process guards.
    """

    read_fd, write_fd = os.pipe()

    class ForkInsideForward(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            pid = os.fork()
            if pid == 0:  # child: observe inherited capture state, never log
                try:
                    clean = (not _state._logging_enabled) and (_state._active_trace is None)
                    os.write(write_fd, b"1" if clean else b"0")
                finally:
                    os._exit(0)
            os.waitpid(pid, 0)
            return v + 1

    tl.trace(ForkInsideForward(), torch.randn(2))
    os.close(write_fd)
    child_verdict = os.read(read_fd, 1)
    os.close(read_fd)
    assert child_verdict == b"1", (
        "the fork child inherited _logging_enabled/_active_trace mid-capture: "
        "child-side torch ops would log into the inherited trace copy"
    )


def test_main_process_rank_stamps_ownership_so_fork_children_stay_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A MAIN-process rank claims the group stamp on its first capture (R40).

    The main-guard early-return skipped the stamp entirely, so a rank running
    in the interpreter's main process never claimed ownership -- a raw
    ``os.fork()`` child then ``setdefault``ed its OWN pid and was accepted as
    a rank (probe-proven with a real gloo world=1, b6 fable, 4th round).
    """

    import multiprocessing as mp

    import torchlens.utils.display as display_mod
    from torchlens.utils.display import warn_parallel

    class _MainProcess:
        name = "MainProcess"
        daemon = False

    class _FakeDist:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

    real_pid = os.getpid()
    monkeypatch.setattr(mp, "current_process", lambda: _MainProcess())
    monkeypatch.setattr(mp, "parent_process", lambda: None)
    fake_dist = _FakeDist()
    monkeypatch.setattr(torch, "distributed", fake_dist)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)  # type: ignore[arg-type]
    monkeypatch.setattr(display_mod, "_DIST_GROUP_OBSERVED_PID", {})
    monkeypatch.setattr(display_mod, "_WARN_PARALLEL_IMPORT_PID", real_pid)

    # Phase 1: the main-process rank captures first and must STAMP, not just
    # early-return.
    warn_parallel()
    assert display_mod._DIST_GROUP_OBSERVED_PID.get("pid") == real_pid, (
        "the main-guard early-return skipped the rank ownership stamp"
    )

    # Phase 2: a raw fork child (different pid, inherited import-PID and
    # stamp) must be refused instead of setdefault-ing its own pid.
    monkeypatch.setattr(display_mod.os, "getpid", lambda: real_pid + 1)
    try:
        with pytest.raises(tl.errors.CaptureContextError) as refusal:
            warn_parallel()
    finally:
        monkeypatch.undo()
    assert refusal.value.fields["code"] == "child_process_capture_unsupported"
