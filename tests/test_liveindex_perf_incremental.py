"""Incremental ``LiveIndex`` edges for re-seen labels match a full rebuild.

A re-seen ``append`` and every ``replace`` move only the replaced label's
child edges. These tests drive seeded random event streams through that path
and through a reference index that runs ``rebuild_edges`` after every event,
compare the whole state after each step, and count the full rebuilds and
child-list operations a re-seen append costs.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

from benchmarks.live_index_append_counters import counting, synthetic_stream
from torchlens.fastlog import Recorder
from torchlens.ir.live_index import LiveIndex


@dataclass(frozen=True)
class _Edge:
    parent_label_raw: str


@dataclass(frozen=True)
class _Event:
    label_raw: str
    parents: tuple[_Edge, ...]


def _event(rng: random.Random, label: str, pool: list[str]) -> _Event:
    """Draw parents (duplicates, outside labels and none at all included)."""

    choices = [*pool, "input_0", "input_1"]
    picks = [rng.choice(choices) for _ in range(rng.randint(0, 4))]
    if picks and rng.random() < 0.2:
        picks.append(picks[0])
    return _Event(label, tuple(_Edge(parent) for parent in picks))


def _state(index: LiveIndex) -> dict[str, Any]:
    return {
        "labels": list(index.labels),
        "keys": list(index.by_raw_label),
        "events": [id(event) for event in index.by_raw_label.values()],
        "children": {key: list(value) for key, value in index.children_by_parent.items()},
    }


def _rebuilt(index: LiveIndex) -> dict[str, Any]:
    reference = index.copy()
    reference.rebuild_edges()
    return _state(reference)


def _step(rng: random.Random, index: LiveIndex, reference: LiveIndex, known: list[str]) -> None:
    """Apply one random operation to both indexes; the reference then rebuilds."""

    roll = rng.random()
    if roll < 0.35 or not known:
        label = f"op_{len(known)}"
        known.append(label)
        event = _event(rng, label, known[:-1])
        index.append(event)
        reference.append(event)
    elif roll < 0.75:
        label = rng.choice(known)
        same = rng.random() < 0.5
        parents = index.by_raw_label[label].parents if same else _event(rng, label, known).parents
        event = _Event(label, parents)
        index.append(event)
        reference.append(event)
    elif roll < 0.97:
        event = _event(rng, rng.choice(known), known)
        index.replace(event)
        reference.replace(event)
    else:
        event = _event(rng, f"unindexed_{len(known)}", known)
        index.replace(event)
        reference.replace(event)
    reference.rebuild_edges()


@pytest.mark.parametrize("seed", range(8))
def test_reseen_streams_match_a_full_rebuild_after_every_event(seed: int) -> None:
    rng = random.Random(seed)
    index, reference = LiveIndex(), LiveIndex()
    known: list[str] = []
    for _ in range(400):
        _step(rng, index, reference, known)
        state = _state(index)
        assert state == _state(reference)
        assert sorted(state["children"]) == sorted(_state(reference)["children"])
        assert all(state["children"].values()), "a rebuild never leaves an empty child list"
        for label in rng.sample(known, min(3, len(known))):
            assert index.children(label) == reference.children(label)
            assert index.parents(label) == reference.parents(label)


def test_labels_outside_the_label_list_gain_no_edges() -> None:
    """A journal cleared to its structural lookup keeps no edges, as a rebuild does."""

    index = LiveIndex()
    index.append(_Event("a", ()))
    index.append(_Event("b", (_Edge("a"),)))
    index.clear()
    index.by_raw_label = {"a": _Event("a", ()), "b": _Event("b", (_Edge("a"),))}
    index.append(_Event("b", (_Edge("x"),)))
    index.replace(_Event("a", (_Edge("y"),)))
    assert _state(index) == _rebuilt(index)
    assert dict(index.children_by_parent) == {}


def test_inserted_child_takes_its_label_list_position() -> None:
    index = LiveIndex()
    for label in ("p", "a", "b", "c", "d"):
        index.append(_Event(label, (_Edge("p"),) if label in "bd" else ()))
    index.append(_Event("c", (_Edge("p"),)))
    index.replace(_Event("a", (_Edge("p"),)))
    assert index.children("p") == ("a", "b", "c", "d")
    index.append(_Event("b", ()))
    index.replace(_Event("d", (_Edge("q"),)))
    assert index.children("p") == ("a", "c")
    assert index.children_by_parent["q"] == ["d"]
    assert _state(index) == _rebuilt(index)


@pytest.mark.parametrize("rewire_every", [0, 10])
def test_reseen_appends_do_no_full_rebuild(rewire_every: int) -> None:
    """Re-feeding N labels costs no rebuild and at most old plus new parents per append."""

    passes = synthetic_stream(100, rewire_every=rewire_every)
    index = LiveIndex()
    with counting() as tally:
        for events in passes:
            for event in events:
                index.append(event)
    counts = tally.counts
    assert counts["reseen_append"] == 100
    assert counts["rebuild_edges"] == 0
    assert counts["reseen_append_rebuilds"] == 0
    assert counts["reseen_append_over_bound"] == 0
    if not rewire_every:
        assert counts["reseen_append_list_ops"] == 0
    assert _state(index) == _rebuilt(index)


class _Chain(nn.Module):
    def __init__(self, fail_on: int | None = None) -> None:
        super().__init__()
        self.fail_on = fail_on
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        for step in range(4):
            if self.calls == self.fail_on and step == 2:
                raise RuntimeError("pass fails halfway")
            x = torch.relu(torch.add(x, step))
        return x


def _save_ops(ctx: Any) -> bool:
    return ctx.kind == "op"


@pytest.mark.parametrize("fail_on", [None, 3])
def test_recorder_folds_repeat_labels_without_rebuilding(fail_on: int | None) -> None:
    """Multi-pass folds and the failed-partial fold keep a rebuild-equal index."""

    with (
        counting() as tally,
        Recorder(_Chain(fail_on), save=_save_ops, on_forward_error="return_partial") as recorder,
    ):
        for _ in range(3):
            recorder(torch.ones(2))
        index = recorder._capture_events.live_index
    assert tally.counts["reseen_append"] > 0
    assert tally.counts["reseen_append_rebuilds"] == 0
    assert tally.counts["reseen_append_over_bound"] == 0
    assert _state(index) == _rebuilt(index)
    assert isinstance(index.children_by_parent, defaultdict)
