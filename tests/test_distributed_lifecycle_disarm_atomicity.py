"""disarm() atomicity: no lifecycle entry point may mutate a RETIRED state.

``next_seq`` and ``resolve_group_identity`` used to snapshot the module-level
``_STATE`` BEFORE taking ``_LOCK``. A concurrent ``disarm()`` (which nulls
``_STATE`` under the lock) could then complete first, after which the stale
snapshot mutated the retired ``_ArmedState`` (ticked its seq counters, seeded
its identity/ledger) and RETURNED SUCCESS -- even though the public API had
already reported the process unarmed.

These tests drive the exact interleave deterministically: a gated lock wrapper
pauses the racing thread at its ``_LOCK`` acquisition (i.e. after any pre-lock
snapshot), lets ``disarm()`` finish on the main thread, then releases it. The
fixed code re-reads ``_STATE`` under the lock and refuses with the unarmed
``RuntimeError``; the retired state stays byte-identical.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
import torch

from torchlens.distributed import (  # noqa: E402
    _lifecycle as lifecycle,
    has_vetted_snapshot,
)
from torchlens.distributed._lifecycle import GroupIdentity  # noqa: E402

# F1 ruling (Lead, 2026-10-01): full collective arming only runs where a
# census-vetted torch build exists (torchlens.distributed.has_vetted_snapshot());
# on an unvetted torch, arm() fails closed with UncapturedCollectiveOpError.
# Both tests in this module arm() as their first step.
pytestmark = [
    pytest.mark.skipif(
        not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
        reason="torch.distributed gloo unavailable",
    ),
    pytest.mark.skipif(
        not has_vetted_snapshot(),
        reason="full collective arming requires a census-vetted torch build "
        "(torchlens.distributed.has_vetted_snapshot() is False here)",
    ),
]


class _GatedLock:
    """Lock wrapper that pauses ONE designated thread at its first acquisition.

    Every other thread (``disarm()`` on the main thread included) passes
    straight through to the real lock, so the wrapper reproduces the race
    window without ever deadlocking the interleave it stages.
    """

    def __init__(self, inner: threading.Lock) -> None:
        self._inner = inner
        self.gate_thread: threading.Thread | None = None
        self.reached_gate = threading.Event()
        self.proceed = threading.Event()
        self._fired = False

    def __enter__(self) -> Any:
        if not self._fired and threading.current_thread() is self.gate_thread:
            self._fired = True
            self.reached_gate.set()
            assert self.proceed.wait(timeout=30), "interleave gate timed out"
        return self._inner.__enter__()

    def __exit__(self, *exc: Any) -> Any:
        return self._inner.__exit__(*exc)


@pytest.fixture()
def gated_lock(monkeypatch):
    """Arm-clean lifecycle with the gated lock installed."""

    lifecycle.disarm()
    gate = _GatedLock(lifecycle._LOCK)
    monkeypatch.setattr(lifecycle, "_LOCK", gate)
    try:
        yield gate
    finally:
        monkeypatch.setattr(lifecycle, "_LOCK", gate._inner)
        lifecycle.disarm()


def _run_gated(gate: _GatedLock, fn: Any) -> tuple[Any, BaseException | None]:
    """Run ``fn`` on the gated thread through the disarm interleave."""

    result: list[Any] = [None]
    error: list[BaseException | None] = [None]

    def target() -> None:
        try:
            result[0] = fn()
        except BaseException as exc:  # noqa: BLE001 - recorded for assertion
            error[0] = exc

    thread = threading.Thread(target=target)
    gate.gate_thread = thread
    thread.start()
    assert gate.reached_gate.wait(timeout=30), "racing thread never reached _LOCK"
    lifecycle.disarm()
    assert not lifecycle.is_armed()
    gate.proceed.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    return result[0], error[0]


def test_next_seq_concurrent_with_disarm_refuses_and_leaves_state_untouched(
    gated_lock: _GatedLock,
) -> None:
    lifecycle.arm()
    retired = lifecycle._STATE
    assert retired is not None
    identity = GroupIdentity(
        membership_digest="d" * 64,
        lifetime_ordinal=0,
        ordinal_source="seeded",
        global_ranks=(0,),
        backend=None,
    )

    result, error = _run_gated(gated_lock, lambda: lifecycle.next_seq(identity, "coll"))

    assert isinstance(error, RuntimeError), (
        f"next_seq returned {result!r} after disarm() reported unarmed; a seq "
        "was issued against a retired armed state"
    )
    assert retired.seq_counters == {}, "retired state's seq counters were mutated"


def test_resolve_group_identity_concurrent_with_disarm_refuses_and_never_seeds(
    gated_lock: _GatedLock, tmp_path
) -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        lifecycle.arm()
        retired = lifecycle._STATE
        assert retired is not None
        ledger_len_before = len(retired.ledger.events)

        result, error = _run_gated(gated_lock, lambda: lifecycle.resolve_group_identity(None))

        assert isinstance(error, RuntimeError), (
            f"resolve_group_identity returned {result!r} after disarm() reported "
            "unarmed; an identity was seeded into a retired armed state"
        )
        assert retired.identities == {}, "retired state's identity map was mutated"
        assert len(retired.ledger.events) == ledger_len_before, (
            "retired state's lifecycle ledger was appended to"
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
