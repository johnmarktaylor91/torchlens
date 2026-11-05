"""A sweep of short replays builds no progress bar; a long replay still gets one.

The README's patching sweep runs ``italy.fork().do(head, patch_from(france))`` once
per attention head. Each replay walks a cone of more than ten ops, so when the
replay bar was built up front a notebook drew one tqdm widget per ``.do()`` call.
The replay bar now waits until the replay has run for a while: a quick replay
builds no bar object at all (no terminal bar, no notebook widget), and a replay
that outlasts the wait starts its bar where it stands and finishes it.
"""

from __future__ import annotations

import importlib.util
import time
from collections.abc import Generator, Iterator
from typing import Any, ClassVar

import pytest
import torch
import tqdm
import tqdm.notebook
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl
from torchlens._utils import display

_N_LAYER, _N_HEAD = 2, 2


class _RecordingBar:
    """Stand-in for a tqdm bar that records construction, updates and close."""

    built: ClassVar[list[_RecordingBar]] = []

    def __init__(self, iterable: Any = None, **kwargs: Any) -> None:
        self.iterable = iterable
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.n = kwargs.get("initial", 0)
        self.closed = False
        _RecordingBar.built.append(self)

    def __iter__(self) -> Iterator[Any]:
        for item in self.iterable:
            yield item
            self.n += 1

    def update(self, count: int = 1) -> None:
        self.n += count

    def close(self) -> None:
        self.closed = True


class _FastClock:
    """The ``time`` module with a monotonic clock that jumps ten seconds per read."""

    def __init__(self) -> None:
        self.now = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    def monotonic(self) -> float:
        self.now += 10.0
        return self.now


@pytest.fixture
def notebook_bars(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingBar]:
    """Pretend to run in a notebook kernel with ipywidgets, recording every bar built."""

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(display, "in_notebook", lambda: True)
    monkeypatch.setattr(
        display.importlib.util,
        "find_spec",
        lambda name, *args: object() if name == "ipywidgets" else real_find_spec(name, *args),
    )
    monkeypatch.setattr(tqdm.notebook, "tqdm", _RecordingBar)
    monkeypatch.setattr(tqdm, "tqdm", _RecordingBar)
    monkeypatch.setattr(_RecordingBar, "built", [])
    return _RecordingBar.built


@pytest.fixture(scope="module")
def pair() -> Iterator[tuple[tl.Trace, tl.Trace]]:
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=_N_LAYER,
        n_head=_N_HEAD,
        n_embd=8,
        n_positions=16,
        vocab_size=32,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(config).eval()
    france = tl.trace(model, torch.tensor([[1, 5, 7, 3, 9]]))
    try:
        italy = tl.trace(model, torch.tensor([[1, 5, 7, 4, 9]]))
        try:
            yield france, italy
        finally:
            italy.cleanup()
    finally:
        france.cleanup()


def _head(block: int, head: int) -> Any:
    return tl.module(f"transformer.h.{block}.attn") & tl.head(head, "v")


def test_a_sweep_of_short_replays_in_a_notebook_builds_no_bar(
    pair: tuple[tl.Trace, tl.Trace], notebook_bars: list[_RecordingBar]
) -> None:
    """Every head patched three times over: no notebook widget, no text bar."""

    france, italy = pair
    for _ in range(3):
        for block in range(_N_LAYER):
            for head in range(_N_HEAD):
                patched = italy.fork().do(_head(block, head), tl.patch_from(france))
                assert patched["lm_head"].out.shape[-1] == 32
    assert notebook_bars == []


def test_a_replay_that_outlasts_the_wait_finishes_under_one_bar(
    pair: tuple[tl.Trace, tl.Trace],
    notebook_bars: list[_RecordingBar],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same replay, with the clock past the wait, runs to the end under one bar."""

    monkeypatch.setattr(display, "time", _FastClock())
    france, italy = pair
    italy.fork().do(_head(0, 0), tl.patch_from(france))
    assert len(notebook_bars) == 1
    bar = notebook_bars[0]
    assert bar.desc == "torchlens.push"
    assert bar.total is not None and bar.total > 10
    assert bar.n == bar.total
    assert bar.closed


def test_a_short_delayed_loop_yields_everything_and_builds_nothing(
    notebook_bars: list[_RecordingBar],
) -> None:
    """A loop that ends before its wait gives back every item and builds no bar."""

    items = display.delayed_progress_bar(range(50), total=50, desc="x", delay=60.0)
    assert list(items) == list(range(50))
    assert notebook_bars == []


def test_a_delayed_loop_past_its_wait_starts_the_bar_where_it_stands(
    notebook_bars: list[_RecordingBar], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bar starts at the items already done and is closed once the loop ends."""

    monkeypatch.setattr(display, "time", _FastClock())
    items = display.delayed_progress_bar(range(50), total=50, desc="x", delay=5.0)
    assert list(items) == list(range(50))
    assert len(notebook_bars) == 1
    bar = notebook_bars[0]
    assert bar.iterable is None
    assert (bar.total, bar.n, bar.closed) == (50, 50, True)


def test_a_delayed_loop_closes_its_bar_when_the_caller_stops_early(
    notebook_bars: list[_RecordingBar], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay that raises mid-cone still closes the bar it started."""

    monkeypatch.setattr(display, "time", _FastClock())
    items = display.delayed_progress_bar(range(50), total=50, desc="x", delay=5.0)
    assert isinstance(items, Generator)
    assert [next(items) for _ in range(5)] == [0, 1, 2, 3, 4]
    items.close()
    assert len(notebook_bars) == 1
    assert notebook_bars[0].closed


def test_an_undelayed_bar_is_still_built_up_front(notebook_bars: list[_RecordingBar]) -> None:
    """Extraction keeps its bar: no delay builds the bar before the first item."""

    items = display.progress_bar(range(50), total=50, desc="torchlens.extract")
    assert len(notebook_bars) == 1
    assert list(items) == list(range(50))
