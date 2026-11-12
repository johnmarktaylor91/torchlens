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
from torchlens.options import CaptureOptions


def _collector_state() -> tuple[bool, tuple[int, ...], int]:
    """Return the collector state a capture must leave unchanged."""

    return gc.isenabled(), gc.get_threshold(), gc.get_freeze_count()


@pytest.fixture(autouse=True)
def _restore_collector() -> Iterator[None]:
    """Put the collector back the way the test found it, whatever the test did."""

    enabled, thresholds, _frozen = _collector_state()
    assert gc_hold._STATE.depth == 0
    try:
        yield
    finally:
        gc.unfreeze()
        gc.set_threshold(*thresholds)
        if enabled:
            gc.enable()
        else:
            gc.disable()
        assert gc_hold._STATE.depth == 0, "a capture left the collector hold taken"


class _Probe(nn.Module):
    """Two linear layers that record the collector state seen inside the forward."""

    def __init__(self, inside: Callable[[], None] | None = None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.inside = inside
        self.seen: list[tuple[bool, tuple[int, ...], int]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(_collector_state())
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
    model = _Probe()
    tl.trace(model, torch.randn(2, 4))
    assert _collector_state() == before
    enabled, thresholds, frozen = model.seen[0]
    assert enabled
    assert thresholds == (gc_hold.HOLD_GEN0_THRESHOLD, *before[1][1:])
    assert frozen > before[2], "the hold froze nothing"


@pytest.mark.parametrize(
    "exc",
    [ValueError("model bug"), KeyboardInterrupt()],
    ids=["exception", "keyboard_interrupt"],
)
def test_failed_and_interrupted_captures_restore_state(exc: BaseException) -> None:
    before = _collector_state()
    with pytest.raises(type(exc)):
        tl.trace(_Probe(_raise(exc)), torch.randn(2, 4))
    assert _collector_state() == before


def test_halted_capture_restores_state() -> None:
    before = _collector_state()
    with torch.no_grad():
        tl.trace(ThreeStageModel(), torch.ones(1, 3), stop_after=halt_on_relu)
    assert _collector_state() == before


def test_repeated_captures_restore_state_each_time() -> None:
    before = _collector_state()
    model = _Probe()
    for _ in range(3):
        tl.trace(model, torch.randn(2, 4))
        assert _collector_state() == before
    assert len({seen[2] for seen in model.seen}) == 1, "a freeze leaked into the next capture"


def test_nested_entry_is_a_no_op_and_the_outer_exit_restores() -> None:
    before = _collector_state()
    inner_seen: list[tuple[bool, tuple[int, ...], int]] = []

    def nested() -> None:
        outer = _collector_state()
        gc_hold.hold_collector()
        try:
            inner_seen.append(_collector_state())
        finally:
            gc_hold.release_collector()
        assert _collector_state() == outer, "an inner exit released the outer hold"
        # A real nested capture: whether it is admitted or refused, it must not
        # release the outer hold.
        try:
            tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        except Exception:  # noqa: BLE001 - refusal or success are both fine here
            pass
        assert _collector_state() == outer

    model = _Probe(nested)
    tl.trace(model, torch.randn(2, 4))
    assert inner_seen == [model.seen[0]]
    assert _collector_state() == before


def test_capture_started_with_the_collector_disabled_leaves_it_alone() -> None:
    gc.disable()
    before = _collector_state()
    model = _Probe()
    tl.trace(model, torch.randn(2, 4))
    assert _collector_state() == before
    assert model.seen[0] == before, "the hold changed a disabled collector"


@pytest.mark.parametrize(
    ("user", "inside"),
    [
        ((123, 5, 7), (gc_hold.HOLD_GEN0_THRESHOLD, 5, 7)),
        ((50_000, 4, 3), (50_000, 4, 3)),
        ((0, 10, 10), (0, 10, 10)),
    ],
    ids=["low_raised", "high_kept", "zero_kept"],
)
def test_user_thresholds_are_raised_only_upward_and_restored(
    user: tuple[int, int, int], inside: tuple[int, int, int]
) -> None:
    gc.set_threshold(*user)
    model = _Probe()
    tl.trace(model, torch.randn(2, 4))
    assert gc.get_threshold() == user
    assert model.seen[0][1] == inside


def test_objects_frozen_by_the_caller_stay_frozen_and_are_not_refrozen() -> None:
    gc.freeze()
    before = _collector_state()
    model = _Probe()
    tl.trace(model, torch.randn(2, 4))
    assert _collector_state() == before, "the hold unfroze the caller's objects"
    assert model.seen[0][2] == before[2]
    assert model.seen[0][1][0] == gc_hold.HOLD_GEN0_THRESHOLD


def test_release_without_hold_is_harmless() -> None:
    before = _collector_state()
    gc_hold.release_collector()
    assert _collector_state() == before


# ---------------------------------------------------------------------------
# Cyclic garbage stays bounded during a capture
# ---------------------------------------------------------------------------

_JUNK_BYTES = 64 * 1024
_CYCLIC_OPS = 3000
#: Stated peak: at most this many garbage cycles (64 KiB each, 64 MiB in all)
#: alive at once during a 3000-op capture. Without collections during the
#: capture all 3000 (188 MiB) would be alive at its end.
_CYCLIC_PEAK = 1000


class _Junk:
    """A self-referencing object holding a buffer: freed only by the collector."""

    __slots__ = ("buf", "me", "__weakref__")

    def __init__(self) -> None:
        self.buf = bytearray(_JUNK_BYTES)
        self.me = self


class _CyclicGarbageModel(nn.Module):
    """Makes one garbage cycle per op and records how many were alive at once."""

    def __init__(self) -> None:
        super().__init__()
        self.live = 0
        self.peak = 0
        self.freed_during_forward = 0

    def _freed(self) -> None:
        self.live -= 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(_CYCLIC_OPS):
            x = x + 1
            weakref.finalize(_Junk(), self._freed)
            self.live += 1
            self.peak = max(self.peak, self.live)
        self.freed_during_forward = _CYCLIC_OPS - self.live
        return x


def test_cyclic_garbage_made_every_op_stays_under_the_stated_peak() -> None:
    gc.collect()
    model = _CyclicGarbageModel()
    tl.trace(model, torch.zeros(1), save=None)
    assert model.freed_during_forward > 0, "no collection ran during the capture"
    assert model.peak <= _CYCLIC_PEAK, f"{model.peak} garbage cycles alive at once"
    gc.collect()
    assert model.live == 0


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
    return all((x[2] is None and y[2] is None) or torch.equal(x[2], y[2]) for x, y in zip(a, b))


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
