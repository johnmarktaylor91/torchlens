"""The hooks and streamed-storage oracles stay armed: each still fails on the defect it guards.

The hooks oracle checks every armed pass the cell ran (counted at the capture
door) for exactly one fire per selected op; the streamed-storage oracle exempts
only captures that really passed a predicate keyword. These tests feed each
check the defect it exists to catch.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tests.behaviour._fill_io import _check_streamed_payloads, _predicate_capture
from tests.behaviour._hook_passes import HookPass, hook_pass_problems
from tests.behaviour.features import feature
from tests.behaviour.harness import OracleFailure


def _armed(*states: Any) -> HookPass:
    return HookPass(True, states)


def test_one_fire_per_selected_op_in_every_armed_pass_is_clean() -> None:
    first, second = {}, {}
    passes = [_armed(first, first), HookPass(False, ()), _armed(second, second)]
    assert hook_pass_problems(passes, 2) == []


def test_a_hook_firing_twice_in_one_pass_fails() -> None:
    first, second = {}, {}
    passes = [_armed(first, first, first, first), _armed(second, second)]
    assert hook_pass_problems(passes, 2) == ["armed pass 0 fired 4 times for 2 selected ops"]


def test_an_armed_pass_with_zero_fires_fails() -> None:
    first = {}
    passes = [_armed(first, first), _armed()]
    assert hook_pass_problems(passes, 2) == ["armed pass 1 fired 0 times for 2 selected ops"]


def test_one_pass_split_over_two_run_states_fails() -> None:
    assert hook_pass_problems([_armed({}, {})], 2) == [
        "armed pass 0 fired under more than one ctx.run_state"
    ]


def test_fires_in_unarmed_passes_or_outside_every_pass_fail() -> None:
    first = {}
    passes = [_armed(first), HookPass(False, ({},))]
    assert hook_pass_problems(passes, 1, fires_outside=2) == [
        "1 fires in passes that did not arm the hook",
        "2 fires outside every capture pass",
    ]


def test_no_armed_pass_fails() -> None:
    assert hook_pass_problems([HookPass(False, ())], 0) == ["no armed pass was counted"]


class _Cell(SimpleNamespace):
    def expect(self, condition: bool, detail: str) -> None:
        if not condition:
            raise OracleFailure(detail)

    def changes(self, oracle: str) -> bool:
        return False


def _streamed_cell(trace_kwargs: dict[str, Any]) -> _Cell:
    return _Cell(
        features=(feature("storage"), feature("hooks")),
        trace_kwargs=trace_kwargs,
        case=SimpleNamespace(has=lambda tag: False),
    )


def _trace_keeping_payloads() -> Any:
    op = SimpleNamespace(label="relu_1_2", out_ref="blob-0", out=object(), is_output=False)
    return SimpleNamespace(saved_ops=[op], ops=[op], outcome=None)


def test_hooks_cell_without_intervene_keeping_streamed_payloads_still_fails() -> None:
    cell = _streamed_cell({"storage": object()})
    assert not _predicate_capture(cell)
    with pytest.raises(OracleFailure, match="streamed payloads kept in memory"):
        _check_streamed_payloads(cell, _trace_keeping_payloads(), {})


def test_hooks_cell_that_passed_intervene_is_a_predicate_capture() -> None:
    assert _predicate_capture(_streamed_cell({"storage": object(), "intervene": object()}))
