"""Deferred record contexts under a lean ``tl.module`` read.

A lean recording (``lean=True``, provisional spelling) whose save selector is a
plain ``tl.module`` union defers every record context and builds one only when
a selected module exits. Each kept record must equal the default recording's
record field for field, including the whole chain of recent events its context
carries (only the wall-clock ``time_since_pass_start`` differs between runs).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, replace
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


def _record(name: str, model: nn.Module, *, lean: bool) -> Recording:
    # One model instance for both recordings: a module stack frame carries the
    # module's id, so two instances never give equal contexts.
    with torch.no_grad():
        return tl.record(model, _CASES[name][1](), save=_save(name), lean=lean)


_Flat = tuple[RecordContext, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]


def _key(ctx: RecordContext) -> tuple[int, int]:
    return (ctx.pass_number, ctx.event_index)


def _chain(root: RecordContext) -> dict[tuple[int, int], _Flat]:
    """Return every context reachable from ``root`` keyed by pass and event index.

    Each entry is the context with its wall-clock time zeroed and its two history
    windows replaced by the keys of their members, so comparing two chains costs one
    comparison per context. Comparing the nested contexts directly would revisit each
    shared context once per path to it, which grows exponentially with the pass length.
    """

    chain: dict[tuple[int, int], _Flat] = {}
    stack = [root]
    seen: set[int] = set()
    while stack:
        ctx = stack.pop()
        if id(ctx) in seen:
            continue
        seen.add(id(ctx))
        flat = (
            replace(ctx, time_since_pass_start=0.0, recent_events=(), recent_ops=()),
            tuple(_key(e) for e in ctx.recent_events),
            tuple(_key(e) for e in ctx.recent_ops),
        )
        assert chain.setdefault(_key(ctx), flat) == flat
        stack.extend(ctx.recent_events)
        stack.extend(ctx.recent_ops)
    return chain


def _differing_fields(lean: _Flat, default: _Flat) -> dict[str, tuple[Any, Any]]:
    """Name the fields where two flattened contexts differ."""

    diff = {
        f.name: (getattr(lean[0], f.name), getattr(default[0], f.name))
        for f in fields(default[0])
        if getattr(lean[0], f.name) != getattr(default[0], f.name)
    }
    if lean[1:] != default[1:]:
        diff["windows"] = (lean[1:], default[1:])
    return diff


def _assert_records_equal(default: Recording, lean: Recording) -> None:
    assert len(lean.records) == len(default.records) > 0
    for d, ln in zip(default.records, lean.records, strict=True):
        assert _key(ln.ctx) == _key(d.ctx)
        lean_chain, default_chain = _chain(ln.ctx), _chain(d.ctx)
        assert lean_chain.keys() == default_chain.keys()
        for key, flat in default_chain.items():
            assert lean_chain[key] == flat, (key, _differing_fields(lean_chain[key], flat))
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
    model = _CASES[name][0]()
    default = _record(name, model, lean=False)
    assert deferred_count[0] == 0
    lean = _record(name, model, lean=True)
    assert deferred_count[0] > 0
    _assert_records_equal(default, lean)
    assert torch.equal(torch.as_tensor(default.num_ops), torch.as_tensor(lean.num_ops))


def test_recorder_passes_defer_and_equal_the_default(deferred_count: list[int]) -> None:
    """A second pass reuses the raw labels of the first; its records still match."""

    model = _Shared()

    def run(lean: bool) -> Recording:
        with (
            torch.no_grad(),
            tl.recording.Recorder(
                model, save=tl.module("block:1") | tl.module("head"), lean=lean
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
        {"save": tl.module("block2"), "recording": tl.options.RecordingOptions(history_size=0)},
        {
            "save": tl.module("block2"),
            "recording": tl.options.RecordingOptions(include_source_events=True),
        },
        {"save": tl.module("block2"), "stop_after": lambda ctx: False},
    ],
    ids=["op_selector", "mixed_union", "no_history", "source_events", "halt"],
)
def test_ineligible_lean_reads_build_contexts_eagerly(
    kwargs: dict[str, Any], deferred_count: list[int]
) -> None:
    name = "small_block2"
    build, inp, _ = _CASES[name]
    model = build()
    with torch.no_grad():
        default = tl.record(model, inp(), **kwargs)
        lean = tl.record(model, inp(), lean=True, **kwargs)
    assert deferred_count[0] == 0
    _assert_records_equal(default, lean)
