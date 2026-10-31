"""Plain-label site resolution through one label index per replay.

The replay engine resolves every hook's site selector against the trace,
twice per replay (origins and hook targeting). Inside
``resolver._label_index_scope`` a plain ``tl.label`` selector is answered from
a label index built once per trace instead of evaluating the selector on
every op. These tests pin that the answer, the refusals and the warnings are
the ones the evaluator gives, that other selector kinds still evaluate, and
that a push with label hooks no longer evaluates selectors per op per hook.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._ir import selector_eval
from torchlens.intervention import resolver


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        for _ in range(2):
            h = torch.relu(self.fc2(h))
        return self.fc3(h)


@pytest.fixture(scope="module")
def trace():
    torch.manual_seed(0)
    log = tl.trace(
        _Net().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    try:
        yield log
    finally:
        log.cleanup()


def _spellings(log: tl.Trace) -> list[str]:
    """Every label spelling the label evaluator reads, plus a missing one."""

    spellings: dict[str, None] = {}
    for op in log._layer_list:
        spellings.update(dict.fromkeys(selector_eval._string_labels(op, "site")))
        spellings.update(dict.fromkeys(str(key) for key in (op.lookup_keys or ())))
    return [*spellings, "no_such_label_1_1"]


def _outcome(log: tl.Trace, spelling: str, max_fanout: int) -> tuple[Any, ...]:
    """Resolved site identities, or the refusal, plus the warning texts."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            table = resolver.resolve_sites(log, tl.label(spelling), max_fanout=max_fanout)
            result: Any = tuple(id(site) for site in table)
        except Exception as exc:  # noqa: BLE001 - the refusal itself is compared
            result = (type(exc).__name__, str(exc))
    return result, tuple(str(w.message) for w in caught)


def test_label_resolution_inside_the_scope_equals_the_evaluator(trace) -> None:
    spellings = _spellings(trace)
    assert len(spellings) > len(trace._layer_list)
    for max_fanout in (1, len(trace._layer_list)):
        want = [_outcome(trace, spelling, max_fanout) for spelling in spellings]
        with resolver._label_index_scope():
            got = [_outcome(trace, spelling, max_fanout) for spelling in spellings]
        assert got == want
    # The bare label of the twice-run layer matches both passes: the refusal
    # (or the fan-out) is the evaluator's, inside the scope too.
    assert any(isinstance(result[0], str) for result, _warnings in want)


def test_the_index_answers_label_queries_without_evaluating_selectors(
    trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real = resolver.evaluate

    def counting(*args: Any, **kwargs: Any) -> bool:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(resolver, "evaluate", counting)
    labels = [op.label for op in trace._layer_list]
    outside = [resolver.resolve_sites(trace, tl.label(label)).first() for label in labels]
    assert len(calls) == len(labels) * len(trace._layer_list)
    calls.clear()
    with resolver._label_index_scope():
        inside = [resolver.resolve_sites(trace, tl.label(label)).first() for label in labels]
        assert calls == []
        # Other selector kinds still go through the evaluator, unchanged.
        relus = resolver.resolve_sites(trace, tl.func("relu"), max_fanout=8)
        assert len(calls) == len(trace._layer_list)
    assert [id(op) for op in inside] == [id(op) for op in outside]
    assert [op.func_name for op in relus] == ["relu"] * 3


def test_each_trace_in_one_scope_gets_its_own_index(trace) -> None:
    torch.manual_seed(1)
    other = tl.trace(
        _Net().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    try:
        label = trace._layer_list[3].label
        with resolver._label_index_scope():
            mine = resolver.resolve_sites(trace, tl.label(label)).first()
            theirs = resolver.resolve_sites(other, tl.label(label)).first()
        assert mine is trace._layer_list[3]
        assert theirs is other._layer_list[3]
    finally:
        other.cleanup()


def test_the_index_skips_positional_keys_and_steps_aside_for_unknown_ones() -> None:
    class _Site:
        layer_label = "a_1_1"
        label = "a_1_1:1"
        lookup_keys: tuple[Any, ...] = ("a_1_1", 7, -3)

    class _Key:
        def __eq__(self, other: object) -> bool:
            return other == "a_1_1"

        __hash__ = object.__hash__

    site = _Site()
    index = selector_eval._label_site_index([site])
    assert index == {"a_1_1": (site,), "a_1_1:1": (site,)}
    odd = _Site()
    odd.lookup_keys = ("a_1_1", _Key())
    assert selector_eval._label_site_index([odd]) is None


def test_a_push_with_label_hooks_evaluates_no_selector_per_hook(
    trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    real = resolver.evaluate

    def counting(*args: Any, **kwargs: Any) -> bool:
        calls.append(1)
        return real(*args, **kwargs)

    hooked = [op.label for op in trace._layer_list if op.func_name in ("relu", "linear")]
    assert len(hooked) >= 5
    fork = trace.fork()
    monkeypatch.setattr(resolver, "evaluate", counting)
    pushed = fork.push(
        replay=tl.options.ReplayOptions(hooks={tl.label(label): tl.scale(1.0) for label in hooked})
    )
    # Before the index, origins and hook targeting each evaluated every hook's
    # selector on every op: 2 x hooks x ops calls.
    assert len(calls) < len(trace._layer_list)
    final = [op for op in pushed._layer_list if op.is_output][-1]
    assert torch.equal(final.out, [op for op in trace._layer_list if op.is_output][-1].out)
