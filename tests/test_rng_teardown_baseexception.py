"""grind-p3 T11.8: RNG-monitor teardown always drains the restore queue.

``_teardown`` latches ``_torn_down = True`` at entry and its per-restore
guard caught only ``Exception``, so a ``BaseException`` (a Ctrl-C landing in
the registry/profile steps or inside one patch restore) escaped mid-unwind
and permanently stranded the remaining restore queue: the retry no-opped on
the latch and the process-wide patches leaked, with the next window
snapshotting the leaked wrappers as its own originals. Every stage is now
BaseException-isolated, the queue always drains, and the first BaseException
re-raises only after the unwind completes.
"""

from __future__ import annotations

import pytest

from torchlens.utils.rng import host_nondeterminism_monitor


def _fresh_monitor() -> host_nondeterminism_monitor:
    """Build an unentered monitor whose teardown surfaces are all inert."""

    return host_nondeterminism_monitor(model=None)


def test_keyboard_interrupt_in_one_restore_still_drains_the_queue():
    """A BaseException in one restore never strands the later restores."""

    monitor = _fresh_monitor()
    ran: list[str] = []

    def _boom() -> None:
        ran.append("boom")
        raise KeyboardInterrupt

    monitor._restores.extend([lambda: ran.append("first"), _boom, lambda: ran.append("last")])
    with pytest.raises(KeyboardInterrupt):
        monitor._teardown()
    # Restores run in reverse registration order; ALL of them must have run.
    assert ran == ["last", "boom", "first"]
    assert monitor._restores == []
    assert "patch_restore_interrupted" in monitor.result.uncertain_detail


def test_keyboard_interrupt_before_the_queue_still_drains_it():
    """A BaseException in the profile-hook stage never strands the queue."""

    monitor = _fresh_monitor()
    ran: list[str] = []
    monitor._restores.append(lambda: ran.append("restore"))

    def _hooks_boom() -> None:
        raise KeyboardInterrupt

    monitor._restore_profile_hooks = _hooks_boom  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        monitor._teardown()
    assert ran == ["restore"]
    assert monitor._restores == []
    assert "teardown_interrupted" in monitor.result.uncertain_detail


@pytest.mark.smoke
def test_second_teardown_after_interrupt_is_a_safe_no_op():
    """The idempotence latch keeps holding after an interrupted unwind."""

    monitor = _fresh_monitor()
    ran: list[str] = []

    def _boom() -> None:
        raise KeyboardInterrupt

    monitor._restores.extend([_boom, lambda: ran.append("drained")])
    with pytest.raises(KeyboardInterrupt):
        monitor._teardown()
    assert ran == ["drained"]
    monitor._teardown()  # must not raise and must not re-run anything
    assert ran == ["drained"]


def test_plain_exception_semantics_unchanged():
    """Ordinary restore Exceptions stay flagged-and-swallowed as before."""

    monitor = _fresh_monitor()
    ran: list[str] = []

    def _fail() -> None:
        raise RuntimeError("restore failed")

    monitor._restores.extend([lambda: ran.append("ok"), _fail])
    monitor._teardown()  # no raise
    assert ran == ["ok"]
    assert "patch_restore_failed" in monitor.result.uncertain_detail
