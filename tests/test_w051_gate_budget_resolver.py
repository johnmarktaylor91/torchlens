"""AUD-CODE 3.9 regression: the budget resolver never serves the full graph.

A 40-block Sequential produced a two-point schedule (t=0.0 -> 282 visible,
t=1.0 -> 4); both miss the 100-220 band and the resolver served the FULL
graph as "nearest" (|282-220| = 62 < |100-4| = 96). The dial must genuinely
compact: while the schedule offers any compacting stop, the t=0.0 full graph
is never the served band-missed point.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.trace import Trace
from torchlens.visualization import lenses


@pytest.fixture
def tiny_log() -> Any:
    log = tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))
    yield log
    log.cleanup()


def _step(t: float, visible: int) -> Any:
    return SimpleNamespace(t=t, visible_count=visible, collapsed_addresses=frozenset())


def _install_schedule(monkeypatch: Any, *counts: int) -> None:
    ts = [0.0] if len(counts) == 1 else [i / (len(counts) - 1) for i in range(len(counts))]
    schedule = SimpleNamespace(steps=tuple(_step(t, c) for t, c in zip(ts, counts, strict=True)))
    monkeypatch.setattr(Trace, "collapse_schedule", lambda self, context=None: schedule)


def _resolve_with_warnings(log: Any) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        budget = lenses.resolve_budget(log)
    codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
    return budget, codes


def test_two_point_schedule_serves_the_compacting_stop(tiny_log: Any, monkeypatch: Any) -> None:
    """The audit's exact shape: 282 -> 4 serves 4, never the 282 wall."""

    _install_schedule(monkeypatch, 282, 4)
    budget, codes = _resolve_with_warnings(tiny_log)
    assert budget.visible_count == 4
    assert budget.draw_kwargs == {"collapse": 1.0}
    assert not budget.in_band
    assert "lens_budget_band_missed" in codes
    assert any("band missed" in line for line in budget.disclosure)


def test_nearest_compacting_stop_wins_among_compacting_stops(
    tiny_log: Any, monkeypatch: Any
) -> None:
    """282 -> 230 -> 4: the nearest COMPACTING point (230) is served, not 282."""

    _install_schedule(monkeypatch, 282, 230, 4)
    budget, codes = _resolve_with_warnings(tiny_log)
    assert budget.visible_count == 230
    assert "lens_budget_band_missed" in codes


def test_in_band_stop_is_still_preferred(tiny_log: Any, monkeypatch: Any) -> None:
    """An in-band point is served silently and coarsest-first, as before."""

    _install_schedule(monkeypatch, 282, 160, 4)
    budget, codes = _resolve_with_warnings(tiny_log)
    assert budget.visible_count == 160
    assert budget.in_band
    assert "lens_budget_band_missed" not in codes


def test_schedule_without_compacting_stop_serves_the_full_graph_disclosed(
    tiny_log: Any, monkeypatch: Any
) -> None:
    """Only when NO compacting stop exists is the full graph served, coded."""

    _install_schedule(monkeypatch, 282)
    budget, codes = _resolve_with_warnings(tiny_log)
    assert budget.visible_count == 282
    assert not budget.in_band
    assert "lens_budget_band_missed" in codes
