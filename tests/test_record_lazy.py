"""Deferred record contexts under a lean ``tl.module`` read.

A lean recording (``lean=True``, provisional spelling) whose save selector is a
plain ``tl.module`` union defers every record context and builds one only when
a selected module exits. Each kept record must equal the default recording's
record field for field, including the whole chain of recent events its context
carries (only the wall-clock ``time_since_pass_start`` differs between runs).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import Any

import pytest
import torch
from _record_lean_facts import SCENARIOS, scenario_input
from torch import nn

import torchlens as tl
from torchlens._capture._lazy_contexts import LazyContextHistory
from torchlens._ir.predicate import RecordContext
from torchlens.recording.types import Recording


class _Shared(nn.Module):
    """Calls one block twice, routes through an identity and returns a tuple."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.block = nn.Sequential(OrderedDict(fc=nn.Linear(6, 6), act=nn.ReLU()))
        self.skip = nn.Identity()
        self.head = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block(self.block(x))
        return self.head(self.skip(h)), h.sum(dim=1)


def _shared_input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 6)


def _cases() -> list[tuple[str, Any, Any, Any]]:
    cases = [
        (name, build, lambda b=inp: scenario_input(b), tl.module(site))
        for name, build, inp, site in SCENARIOS
    ]
    cases += [
        ("union", SCENARIOS[0][1], lambda: scenario_input(SCENARIOS[0][2]), None),
        ("pass_qualified", _Shared, _shared_input, tl.module("block:2")),
        ("both_passes", _Shared, _shared_input, tl.module("block")),
        ("identity", _Shared, _shared_input, tl.module("skip")),
        ("nested_leaf", _Shared, _shared_input, tl.module("block.act")),
    ]
    return cases


_CASES = {name: (build, inp, save) for name, build, inp, save in _cases()}


def _save(name: str) -> Any:
    save = _CASES[name][2]
    return tl.module("block1") | tl.module("head") if save is None else save


def _record(name: str, *, lean: bool) -> Recording:
    build, inp, _ = _CASES[name]
    with torch.no_grad():
        return tl.record(build(), inp(), save=_save(name), lean=lean)


def _normalized(ctx: RecordContext, memo: dict[int, RecordContext]) -> RecordContext:
    """Return ``ctx`` with every wall-clock field zeroed, recursively."""

    key = id(ctx)
    if key not in memo:
        memo[key] = replace(
            ctx,
            time_since_pass_start=0.0,
            recent_events=tuple(_normalized(e, memo) for e in ctx.recent_events),
            recent_ops=tuple(_normalized(e, memo) for e in ctx.recent_ops),
        )
    return memo[key]


def _assert_records_equal(default: Recording, lean: Recording) -> None:
    assert len(lean.records) == len(default.records) > 0
    memo: dict[int, RecordContext] = {}
    for d, ln in zip(default.records, lean.records, strict=True):
        assert _normalized(ln.ctx, memo) == _normalized(d.ctx, memo)
        assert ln.spec == d.spec
        assert torch.equal(ln.ram_payload, d.ram_payload)


@pytest.fixture
def deferred_count(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the contexts the capture deferred."""

    count = [0]
    original = LazyContextHistory.defer

    def counting(self: LazyContextHistory, *args: Any, **kwargs: Any) -> None:
        count[0] += 1
        original(self, *args, **kwargs)

    monkeypatch.setattr(LazyContextHistory, "defer", counting)
    return count


@pytest.mark.parametrize("name", list(_CASES))
def test_lean_module_read_defers_contexts_and_equals_the_default(
    name: str, deferred_count: list[int]
) -> None:
    default = _record(name, lean=False)
    assert deferred_count[0] == 0
    lean = _record(name, lean=True)
    assert deferred_count[0] > 0
    _assert_records_equal(default, lean)
    assert torch.equal(torch.as_tensor(default.num_ops), torch.as_tensor(lean.num_ops))


def test_recorder_passes_defer_and_equal_the_default(deferred_count: list[int]) -> None:
    """A second pass reuses the raw labels of the first; its records still match."""

    def run(lean: bool) -> Recording:
        with (
            torch.no_grad(),
            tl.recording.Recorder(
                _Shared(), save=tl.module("block:1") | tl.module("head"), lean=lean
            ) as recorder,
        ):
            recorder(_shared_input())
            recorder(_shared_input() * 2)
        return recorder.recording

    default = run(False)
    assert deferred_count[0] == 0
    lean = run(True)
    assert deferred_count[0] > 0
    assert sorted(lean.by_pass) == [1, 2]
    _assert_records_equal(default, lean)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"save": tl.func("relu")},
        {"save": tl.module("block2") | tl.func("relu")},
        {"save": tl.module("block2"), "history_size": 0},
        {"save": tl.module("block2"), "include_source_events": True},
        {"save": tl.module("block2"), "stop_after": lambda ctx: False},
    ],
    ids=["op_selector", "mixed_union", "no_history", "source_events", "halt"],
)
def test_ineligible_lean_reads_build_contexts_eagerly(
    kwargs: dict[str, Any], deferred_count: list[int]
) -> None:
    name = "small_block2"
    build, inp, _ = _CASES[name]
    with torch.no_grad():
        default = tl.record(build(), inp(), **kwargs)
        lean = tl.record(build(), inp(), lean=True, **kwargs)
    assert deferred_count[0] == 0
    _assert_records_equal(default, lean)
