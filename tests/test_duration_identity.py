"""Each op owns its own ``func_duration``; the save does not depend on timing coincidences.

Op durations are wall-clock measurements, so two ops get bit-identical values only by
chance. When equal values were collapsed onto one shared ``Duration`` object, the
canonical pickler wrote the second as a memo back-reference, and ``metadata.pkl`` (and
the exact save-path byte counters) moved with the clock. These tests drive the capture
clock so every measured op gets the same duration, then check that each op still holds
its own object and that the saved bytes match a capture whose durations all differ.
"""

from __future__ import annotations

import pickletools
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import wrappers
from torchlens.options import CaptureOptions


class _TinyNet(nn.Module):
    """A few distinct timed ops on a small input."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = torch.sigmoid(h) * 2.0
        return self.fc2(h).sum(dim=-1)


def _equal_clock() -> Callable[[], float]:
    """Return a clock that advances one fixed step per read: every op times the same."""

    state = {"t": 1024.0}

    def clock() -> float:
        state["t"] += 2.0**-10  # exact in binary, so every difference is bit-identical
        return state["t"]

    return clock


def _distinct_clock() -> Callable[[], float]:
    """Return a clock whose successive steps grow, so every op times differently."""

    state = {"n": 0}

    def clock() -> float:
        state["n"] += 1
        return 1024.0 + (state["n"] ** 2) * 2.0**-20

    return clock


def _capture(monkeypatch: pytest.MonkeyPatch, clock: Callable[[], float]) -> Any:
    """Capture ``_TinyNet`` with the op-timing clock replaced by ``clock``."""

    torch.manual_seed(0)
    model = _TinyNet()
    x = torch.randn(2, 4)
    with monkeypatch.context() as patch:
        patch.setattr(wrappers, "time", SimpleNamespace(time=clock))
        return tl.trace(model, x, capture=CaptureOptions(seed=0, name="durid"))


def _measured_durations(trace: Any) -> list[Any]:
    """Return the ``func_duration`` objects of the ops that executed a function."""

    return [
        op.func_duration
        for op in trace.ops
        if op.func_duration is not None and float(op.func_duration) != 0.0
    ]


def _float_writes(metadata: bytes) -> int:
    """Count floats written into the pickle; a memo back-reference to a Duration writes none."""

    return sum(1 for op, _arg, _pos in pickletools.genops(metadata) if op.name == "BINFLOAT")


def _saved_metadata(trace: Any, target: Path) -> bytes:
    """Save ``trace`` to ``target`` and return its ``metadata.pkl`` bytes."""

    trace.save(str(target))
    return (target / "metadata.pkl").read_bytes()


def test_equal_op_durations_stay_distinct_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ops timed to bit-identical durations keep one Duration object each."""

    trace = _capture(monkeypatch, _equal_clock())
    durations = _measured_durations(trace)
    assert len(durations) >= 3, "the model must time several ops"
    assert len({float(d).hex() for d in durations}) == 1, "the clock must equalise every op"
    assert len({id(d) for d in durations}) == len(durations), (
        "ops with equal measured durations share one Duration object"
    )


def test_save_bytes_do_not_depend_on_op_timings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Equal and all-different op timings save the same number of bytes and Durations."""

    equal = _capture(monkeypatch, _equal_clock())
    distinct = _capture(monkeypatch, _distinct_clock())
    n_measured = len(_measured_durations(equal))
    assert len({float(d).hex() for d in _measured_durations(distinct)}) == len(
        _measured_durations(distinct)
    ), "the distinct clock must give every op its own duration"

    equal_bytes = _saved_metadata(equal, tmp_path / "equal.tlspec")
    distinct_bytes = _saved_metadata(distinct, tmp_path / "distinct.tlspec")

    equal_floats = _float_writes(equal_bytes)
    assert equal_floats == _float_writes(distinct_bytes), (
        "an op's Duration was saved as a back-reference to another op's"
    )
    assert equal_floats >= n_measured
    assert len(equal_bytes) == len(distinct_bytes)
