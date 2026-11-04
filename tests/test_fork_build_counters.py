"""Deterministic counters for fork build cost over a fork sweep.

A patching sweep forks one parent trace once per site. Each fork appends a
``fork`` row to the parent's ``state_history``, and every later fork used to
deep-copy that growing history and count its fork rows one Python call at a
time, so the N-th fork of a sweep cost more than the first. These tests pin
that a fork's Python calls and ``copy.deepcopy`` calls do not depend on how
many forks the parent already has, and that the fork's history is still its
own copy, row for row.
"""

from __future__ import annotations

import copy
import gc
import sys
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl

_DEEPCOPY_CODE = copy.deepcopy.__code__


def _model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=8, n_positions=16, vocab_size=32, attn_implementation="eager"
    )
    return GPT2LMHeadModel(config).eval()


@pytest.fixture(scope="module")
def parent() -> Iterator[tl.Trace]:
    trace = tl.trace(_model(), torch.tensor([[1, 5, 7, 4, 9]]))
    try:
        yield trace
    finally:
        trace.cleanup()


def _fork_counts(trace: tl.Trace) -> tuple[int, int]:
    """Profiler events (calls and C calls) and ``copy.deepcopy`` calls of one fork.

    Garbage from earlier forks is collected first and automatic collection is
    off while counting, so no finalizer of another object is counted.
    """

    events = 0
    deepcopies = 0

    def count(frame: Any, event: str, _arg: Any) -> None:
        nonlocal events, deepcopies
        if event == "call" or event == "c_call":
            events += 1
            if event == "call" and frame.f_code is _DEEPCOPY_CODE:
                deepcopies += 1

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    sys.setprofile(count)
    try:
        member = trace.fork()
    finally:
        sys.setprofile(None)
        if was_enabled:
            gc.enable()
    del member
    return events, deepcopies


def test_fork_cost_does_not_grow_with_earlier_forks(parent: tl.Trace) -> None:
    """The 100th fork of a parent costs exactly the calls and deep copies of the 2nd."""

    parent.fork()  # warm-up: lazy imports and first-touch caches
    early = _fork_counts(parent)
    for _ in range(97):
        parent.fork()
    gc.collect()
    late = _fork_counts(parent)
    forks = sum(1 for row in parent.state_history if row.get("op") == "fork")
    assert forks == 100
    assert late == early, {"fork 2": early, "fork 100": late}


def test_fork_history_is_an_isolated_copy(parent: tl.Trace) -> None:
    """The fork's history equals the parent's at fork time and shares no row or list with it."""

    before = [dict(row) for row in parent.state_history]
    fork = parent.fork()
    assert fork.state_history == before
    assert fork.state_history is not parent.state_history
    shared = {id(row) for row in parent.state_history} & {id(row) for row in fork.state_history}
    assert not shared
    fork.state_history[0]["probe"] = "fork-only"
    fork.state_history.append({"op": "probe"})
    assert "probe" not in parent.state_history[0]
    assert parent.state_history[: len(before)] == before
    sibling = parent.fork()
    assert "probe" not in sibling.state_history[0]
    assert {"op": "probe"} not in sibling.state_history
    assert fork.name != sibling.name


def test_sealed_capture_fields_are_shared_and_tables_stay_isolated(parent: tl.Trace) -> None:
    """Capture-sealed fields are shared by reference; fork-writable tables stay the fork's own."""

    fork = parent.fork()
    for name in ("_capture_parent_edge_truth", "_pre_forward_rng_states"):
        assert name in parent.__dict__, name
        assert fork.__dict__[name] is parent.__dict__[name], name
    label = parent._layer_list[-1].layer_label
    assert fork._layer_dict_all_keys is not parent._layer_dict_all_keys
    assert fork._layer_dict_all_keys[label] is fork._layer_list[-1]
    assert parent._layer_dict_all_keys[label] is parent._layer_list[-1]
    for name in ("_layer_num_to_lookup_keys_dict", "_graph_op_equivalence_classes"):
        fork_table, parent_table = getattr(fork, name), getattr(parent, name)
        assert fork_table == parent_table and fork_table is not parent_table, name
        key = next(iter(parent_table))
        assert fork_table[key] == parent_table[key] and fork_table[key] is not parent_table[key]
