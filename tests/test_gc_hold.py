"""The collector hold around a capture (``torchlens._capture.gc_hold``).

A capture freezes the objects tracked before it and raises the generation-0
threshold, so automatic collections during the capture scan only what the
capture made. These tests pin what the hold must keep:

* the collector's enabled flag, thresholds and freeze count are the same after
  a capture as before it, on every exit path (completed, failed, halted,
  interrupted, nested, repeated), and when the capture started with the
  collector disabled, with user thresholds, or with objects already frozen;
* the collector keeps running during the capture, so a model that makes
  cyclic garbage on every op stays under a stated peak;
* nothing recorded depends on when a collection ran: the same capture with a
  collection forced on every op, with the collector disabled, and under the
  hold records the same labels and values, and a save-budget refusal names
  the same op.
"""

from __future__ import annotations

import contextlib
import gc
import weakref
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from fixtures.capture_outcome_models import ThreeStageModel, halt_on_relu
from torch import nn

import torchlens as tl
from torchlens._capture import gc_hold
from torchlens._save_budget import SaveBudgetExceededError
from torchlens.errors import TraceCaptureError
from torchlens.options import CaptureOptions

State = tuple[bool, tuple[int, ...], int]

#: Collector states seen inside a forward. Module-level on purpose: a capture
#: puts the model's own attributes back afterwards, so a list on the model
#: would come back empty.
_SEEN: list[State] = []


def _collector_state() -> State:
    """Return the enabled flag, thresholds and freeze count."""

    return gc.isenabled(), gc.get_threshold(), gc.get_freeze_count()


def _assert_restored(before: State, after: State) -> None:
    """Enabled flag and thresholds exact; nothing the hold froze is left frozen.

    The freeze count is not compared exactly: frozen objects the capture frees
    leave the permanent generation by reference counting.
    """

    assert after[:2] == before[:2]
    if before[2] == 0:
        assert after[2] < 1000, f"the hold left {after[2]} objects frozen"
    else:
        assert after[2] > before[2] // 2, "the hold unfroze the caller's frozen objects"


@pytest.fixture(autouse=True)
def _restore_collector() -> Iterator[None]:
    """Start each test unfrozen; put the session's collector back afterwards.

    The test session freezes its warm-up objects at collection
    (``tests/_collection_warmup.py``); the hold leaves a process that already
    has frozen objects alone, so each test here starts unfrozen to exercise
    the freeze, and the session's freeze is put back at the end.
    """

    enabled, thresholds, frozen = _collector_state()
    assert gc_hold._STATE.depth == 0
    gc.unfreeze()
    _SEEN.clear()
    try:
        yield
    finally:
        _SEEN.clear()
        gc.unfreeze()
        gc.set_threshold(*thresholds)
        if enabled:
            gc.enable()
        else:
            gc.disable()
        if frozen:
            gc.freeze()
        assert gc_hold._STATE.depth == 0, "a capture left the collector hold taken"


class _Probe(nn.Module):
    """Two linear layers that record the collector state seen inside the forward."""

    def __init__(self, inside: Callable[[], None] | None = None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.inside = inside

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _SEEN.append(_collector_state())
        y = torch.relu(self.fc1(x))
        if self.inside is not None:
            self.inside()
        return self.fc2(y)


def _raise(exc: BaseException) -> Callable[[], None]:
    def inside() -> None:
        raise exc

    return inside


def test_completed_capture_restores_state_and_holds_inside() -> None:
    before = _collector_state()
    tl.trace(_Probe(), torch.randn(2, 4))
    _assert_restored(before, _collector_state())
    enabled, thresholds, frozen = _SEEN[0]
    assert enabled
    assert thresholds == (gc_hold.HOLD_GEN0_THRESHOLD, *before[1][1:])
    assert frozen > 10_000, "the hold froze nothing"


@pytest.mark.parametrize(
    "exc",
    [ValueError("model bug"), KeyboardInterrupt()],
    ids=["exception", "keyboard_interrupt"],
)
def test_failed_and_interrupted_captures_restore_state(exc: BaseException) -> None:
    before = _collector_state()
    # A model's own exception reaches the caller as a TraceCaptureError chained
    # from it; KeyboardInterrupt propagates as itself.
    expected = TraceCaptureError if isinstance(exc, Exception) else type(exc)
    with pytest.raises(expected) as info:
        tl.trace(_Probe(_raise(exc)), torch.randn(2, 4))
    if expected is TraceCaptureError:
        assert isinstance(info.value.__cause__, type(exc))
    _assert_restored(before, _collector_state())
    assert _SEEN and _SEEN[0][2] > 10_000


def test_halted_capture_restores_state() -> None:
    before = _collector_state()
    with torch.no_grad():
        tl.trace(ThreeStageModel(), torch.ones(1, 3), stop_after=halt_on_relu)
    _assert_restored(before, _collector_state())


def test_repeated_captures_restore_state_each_time() -> None:
    before = _collector_state()
    model = _Probe()
    for _ in range(3):
        tl.trace(model, torch.randn(2, 4))
        _assert_restored(before, _collector_state())
    assert len(_SEEN) == 3
    assert all(seen[2] > 10_000 for seen in _SEEN), "a later capture froze nothing"


def test_nested_entry_is_a_no_op_and_the_outer_exit_restores() -> None:
    before = _collector_state()
    inner_seen: list[State] = []

    def nested() -> None:
        outer = _collector_state()
        gc_hold.hold_collector()
        try:
            inner_seen.append(_collector_state())
        finally:
            gc_hold.release_collector()
        assert _collector_state()[:2] == outer[:2], "an inner exit released the outer hold"
        assert _collector_state()[2] > outer[2] // 2
        # A real nested capture: whether it is admitted or refused, it must not
        # release the outer hold.
        with contextlib.suppress(Exception):  # refusal or success are both fine here
            tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        assert _collector_state()[:2] == outer[:2]
        assert _collector_state()[2] > outer[2] // 2

    tl.trace(_Probe(nested), torch.randn(2, 4))
    assert len(inner_seen) == 1
    assert inner_seen[0][:2] == _SEEN[0][:2]
    _assert_restored(before, _collector_state())


def test_capture_started_with_the_collector_disabled_leaves_it_alone() -> None:
    gc.disable()
    before = _collector_state()
    tl.trace(_Probe(), torch.randn(2, 4))
    _assert_restored(before, _collector_state())
    assert _SEEN[0][:2] == before[:2], "the hold changed a disabled collector"
    assert _SEEN[0][2] < 1000, "the hold froze under a disabled collector"


@pytest.mark.parametrize(
    ("user", "inside"),
    [
        ((123, 5, 7), (gc_hold.HOLD_GEN0_THRESHOLD, 5, 7)),
        ((200_000, 4, 3), (200_000, 4, 3)),
        ((0, 10, 10), (0, 10, 10)),
    ],
    ids=["low_raised", "high_kept", "zero_kept"],
)
def test_user_thresholds_are_raised_only_upward_and_restored(
    user: tuple[int, int, int], inside: tuple[int, int, int]
) -> None:
    gc.set_threshold(*user)
    tl.trace(_Probe(), torch.randn(2, 4))
    assert gc.get_threshold() == user
    assert _SEEN[0][1] == inside


def test_objects_frozen_by_the_caller_stay_frozen_and_are_not_refrozen() -> None:
    gc.freeze()
    before = _collector_state()
    tl.trace(_Probe(), torch.randn(2, 4))
    _assert_restored(before, _collector_state())
    # Not refrozen: a hold that froze would also unfreeze, emptying the generation.
    assert _SEEN[0][2] > before[2] // 2
    assert _SEEN[0][1][0] == gc_hold.HOLD_GEN0_THRESHOLD


def test_release_without_hold_is_harmless() -> None:
    before = _collector_state()
    gc_hold.release_collector()
    assert _collector_state() == before


# ---------------------------------------------------------------------------
# Cyclic garbage stays bounded during a capture
# ---------------------------------------------------------------------------

_JUNK_BYTES = 16 * 1024
_CYCLIC_OPS = 6000
#: Stated peak: at most this many garbage cycles (16 KiB each, 40 MiB in all)
#: alive at once during a 6000-op capture. Without collections during the
#: capture all 6000 (94 MiB) would be alive at its end. A capture makes about
#: 50 tracked objects per op, so the hold's threshold runs a collection about
#: every 2000 ops.
_CYCLIC_PEAK = 2500


class _Junk:
    """A self-referencing object holding a buffer: freed only by the collector."""

    __slots__ = ("buf", "me", "__weakref__")

    def __init__(self) -> None:
        self.buf = bytearray(_JUNK_BYTES)
        self.me = self


#: Live and peak garbage-cycle counts (module-level: a capture puts the
#: model's own attributes back afterwards).
_CYCLES = {"live": 0, "peak": 0, "freed_during_forward": 0}


def _cycle_freed() -> None:
    _CYCLES["live"] -= 1


class _CyclicGarbageModel(nn.Module):
    """Makes one garbage cycle per op and records how many were alive at once."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(_CYCLIC_OPS):
            x = x + 1
            weakref.finalize(_Junk(), _cycle_freed)
            _CYCLES["live"] += 1
            _CYCLES["peak"] = max(_CYCLES["peak"], _CYCLES["live"])
        _CYCLES["freed_during_forward"] = _CYCLIC_OPS - _CYCLES["live"]
        return x


def test_cyclic_garbage_made_every_op_stays_under_the_stated_peak() -> None:
    gc.collect()
    _CYCLES.update(live=0, peak=0, freed_during_forward=0)
    tl.trace(_CyclicGarbageModel(), torch.zeros(1), save=None)
    print(
        f"cyclic garbage: peak {_CYCLES['peak']} of {_CYCLIC_OPS}, freed during forward {_CYCLES['freed_during_forward']}"
    )
    assert _CYCLES["freed_during_forward"] > 0, "no collection ran during the capture"
    assert _CYCLES["peak"] <= _CYCLIC_PEAK, f"{_CYCLES['peak']} garbage cycles alive at once"
    gc.collect()
    assert _CYCLES["live"] == 0


# ---------------------------------------------------------------------------
# Nothing recorded depends on when a collection ran
# ---------------------------------------------------------------------------


class _DedupModel(nn.Module):
    """Reuses one source tensor across ops (identity-cache hits) and optionally collects."""

    def __init__(self, collect_every_op: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 6)
        self.collect_every_op = collect_every_op

    def _maybe_collect(self) -> None:
        if self.collect_every_op:
            gc.collect()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        outs = []
        for _ in range(4):
            self._maybe_collect()
            outs.append(torch.relu(h))
            self._maybe_collect()
            outs.append(h.view(-1).view(h.shape))
        return torch.stack(outs).sum(0)


def _recorded(trace: Any) -> list[tuple[Any, ...]]:
    rows = []
    for op in trace.ops:
        out = op.out
        rows.append(
            (
                op.label,
                tuple(out.shape) if isinstance(out, torch.Tensor) else None,
                out.detach().clone() if isinstance(out, torch.Tensor) else None,
            )
        )
    return rows


def _same(a: list[tuple[Any, ...]], b: list[tuple[Any, ...]]) -> bool:
    if [r[:2] for r in a] != [r[:2] for r in b]:
        return False
    return all(
        (x[2] is None and y[2] is None) or torch.equal(x[2], y[2])
        for x, y in zip(a, b, strict=True)
    )


def test_recorded_values_do_not_depend_on_collection_timing() -> None:
    x = torch.randn(3, 6)
    torch.manual_seed(0)
    held = _recorded(tl.trace(_DedupModel(collect_every_op=False), x))
    torch.manual_seed(0)
    forced = _recorded(tl.trace(_DedupModel(collect_every_op=True), x))
    gc.disable()
    torch.manual_seed(0)
    disabled = _recorded(tl.trace(_DedupModel(collect_every_op=False), x))
    assert _same(held, forced)
    assert _same(held, disabled)


def _refusal_label() -> str:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(model, torch.randn(8, 64), capture=CaptureOptions(save_budget=5000))
    return str(excinfo.value.fields["label"])


def test_save_budget_refusal_names_the_same_op_whatever_the_collector_does() -> None:
    held = _refusal_label()
    gc.disable()
    disabled = _refusal_label()
    gc.enable()
    gc.set_threshold(gc_hold.HOLD_GEN0_THRESHOLD * 10)
    rare = _refusal_label()
    assert held == disabled == rare
