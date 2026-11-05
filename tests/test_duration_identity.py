"""Each op owns its own ``func_duration``; the save does not depend on timing coincidences.

Op durations are wall-clock measurements, so two ops get bit-identical values only by
chance. When equal values were collapsed onto one shared ``Duration`` object, the
pre-3 canonical pickler wrote the second as a memo back-reference, so the saved bytes
moved with the clock. These tests drive the capture clock so every measured op gets the
same duration, then check that each op still holds its own object, that every op's
``func_duration`` cell in the saved op table carries its own value, that save and
reload keep each value, and that the artifact matches, byte for byte in size, a capture
whose durations all differ.

Saved format 3 writes no pickle at any level, so the artifact is read through its own
files: ``manifest.json`` names every file and the ``records/op.json`` table carries the
per-op durations. The pre-3 check that counted pickle float opcodes does not apply to a
pickle-free format and is gone. The pickle wrote every float at a fixed width; format 3
writes floats by exact repr, so the byte comparison uses the save-path counters' size, which
holds each wall-clock timing float (``TIMING_FLOAT_PATHS``) at a fixed width and counts every
other byte as written: a shared or back-referenced Duration cell still changes it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from benchmarks import save_path_counters as spc
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
        op.cost.duration
        for op in trace.ops
        if op.cost.duration is not None and float(op.cost.duration) != 0.0
    ]


_TYPED_VALUE_KEY = "$tl"


def _save(trace: Any, target: Path) -> dict[str, Any]:
    """Save ``trace`` to ``target`` and return its parsed ``manifest.json``."""

    trace.save(str(target))
    assert not list(target.rglob("*.pkl")), "saved format 3 writes no pickle"
    manifest: dict[str, Any] = json.loads((target / "manifest.json").read_text())
    return manifest


def _op_table(target: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the columnar op table the manifest names."""

    table: dict[str, Any] = json.loads((target / manifest["records"]["op"]["path"]).read_text())
    return table


def _floats_in(cell: Any) -> list[float]:
    """Return every float written inside one table cell, depth first."""

    if isinstance(cell, float):
        return [cell]
    if isinstance(cell, dict):
        return [f for item in cell.values() for f in _floats_in(item)]
    if isinstance(cell, list):
        return [f for item in cell for f in _floats_in(item)]
    return []


def _has_ref(cell: Any) -> bool:
    """Return whether a cell holds a record reference instead of its own value."""

    if isinstance(cell, dict):
        return cell.get(_TYPED_VALUE_KEY) == "ref" or any(_has_ref(v) for v in cell.values())
    if isinstance(cell, list):
        return any(_has_ref(item) for item in cell)
    return False


def _saved_durations(trace: Any, table: dict[str, Any]) -> list[Any]:
    """Return the saved ``func_duration`` cells of the measured ops, in trace order."""

    cells = dict(zip(table["columns"]["label"], table["columns"]["func_duration"], strict=True))
    measured = {id(d) for d in _measured_durations(trace)}
    return [cells[op.label] for op in trace.ops if id(op.cost.duration) in measured]


def _timing_normalised_size(target: Path) -> int:
    """Return the artifact's size with each timing float at a fixed width, every other byte raw."""

    sizes = spc.bundle_bytes(target)
    found = sizes["timing_floats_by_file"]
    assert set(found) == set(spc.TIMING_FLOAT_PATHS), found
    assert all(count >= 1 for count in found.values()), found
    size: int = sizes["bytes_written"]
    return size


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

    sizes = []
    for trace, target in (
        (equal, tmp_path / "equal.tlspec"),
        (distinct, tmp_path / "distinct.tlspec"),
    ):
        manifest = _save(trace, target)
        cells = _saved_durations(trace, _op_table(target, manifest))
        assert len(cells) == n_measured
        assert not any(_has_ref(cell) for cell in cells), (
            "an op's Duration was saved as a reference to another record"
        )
        assert [_floats_in(cell) for cell in cells] == [
            [float(d)] for d in _measured_durations(trace)
        ], "every measured op's saved cell carries its own duration value"
        sizes.append(_timing_normalised_size(target))

    assert sizes[0] == sizes[1], "the saved bytes depend on which ops share a timing"


def test_reload_keeps_each_op_duration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every measured op keeps its own Duration value and object through save and reload."""

    for clock, name in ((_equal_clock(), "equal"), (_distinct_clock(), "distinct")):
        trace = _capture(monkeypatch, clock)
        target = tmp_path / f"{name}.tlspec"
        trace.save(str(target))
        loaded = tl.load(str(target))
        by_label = {op.label: op.cost.duration for op in loaded.ops}
        expected = {op.label: op.cost.duration for op in trace.ops}
        assert by_label.keys() == expected.keys()
        for label, duration in expected.items():
            assert by_label[label] == duration, f"{label} lost its duration on reload"
        reloaded = _measured_durations(loaded)
        assert len(reloaded) == len(_measured_durations(trace))
        assert len({id(d) for d in reloaded}) == len(reloaded), (
            f"{name}: reloaded ops with equal durations share one Duration object"
        )
