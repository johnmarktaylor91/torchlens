"""The capture-core counter tool and its bar file.

``benchmarks/capture_core_cost.py`` measures what one ``tl.trace`` costs in
deterministic counts, and ``benchmarks/perf_baselines/capture_core_bars.json``
holds the bars later changes are judged by. These tests pin the tool's
contract: two measurements of one commit agree, the bar arithmetic and the
checker behave as documented, the committed bar file is the plan's fractions
applied to its own baseline, and the two observers that read product state
(store cell writes, held tensor bytes) see what they claim to see.
"""

from __future__ import annotations

import json
import weakref

import pytest
import torch
from torch import nn
from torch.utils.weak import WeakIdKeyDictionary

import torchlens as tl
from benchmarks import capture_core_bars as bars_mod, capture_core_cost as cost
from benchmarks.capture_core_counters import (
    StoreWriteObserver,
    TensorBytesObserver,
    caller_owned_keys,
    count_calls,
    count_phases,
    held_tensor_bytes,
    retained_after,
)
from torchlens._utils import _tensor_label as labels


def _synthetic(scale: float = 1.0) -> dict:
    """A measurement document with every counter family, sized by ``scale``."""

    models = {}
    for name in (*bars_mod.FULL_MODELS, *bars_mod.SMALL_MODELS):
        record = {
            "rows": 10,
            "c1": {
                "total": int(1000 * scale),
                "phases": {"wrapper": int(400 * scale), "post": int(600 * scale)},
            },
            "c4": {"Op": 10, "Layer": 2},
            "c8": {"fast_lane_share": 0.0},
        }
        if name in bars_mod.FULL_MODELS:
            record["c2"] = {
                "default": {"blocks": 500, "objects": 200},
                "save_none": {"blocks": 400, "objects": 150},
            }
            record["c3"] = {
                "cells": {"capture": 300, "ingest": 100, "passes": 50},
                "calls": {"cell_set.passes": 40},
            }
            record["c7"] = {"to_pandas": 900, "repr": 90, "save": 500}
            record["c9"] = {
                mode: {"peak_bytes": 64, "argument_bytes_max": 0, "saved_output_bytes_max": 64}
                for mode in ("default", "save_none")
            }
        models[name] = record
    derived = {"c5.calls_per_module_call": 100.0, "c6.minimal_trace_calls": 2000.0}
    return {"schema": 1, "environment": {}, "models": models, "derived": derived}


def test_flatten_names_every_gated_counter() -> None:
    flat = bars_mod.flatten(_synthetic())
    assert flat["c1.chain1000.total_per_op"] == 100.0
    assert flat["c1.chain1000.phase.wrapper"] == 40.0
    assert flat["c3.chain1000.emit_ingest_cells_per_op"] == 40.0
    assert flat["c7.chain1000.trace_plus_to_pandas_per_op"] == 190.0
    assert flat["c9.resnet18.save_none.argument_bytes_max"] == 0
    assert flat["c5.calls_per_module_call"] == 100.0
    assert "c2.one_op.default.blocks_per_op" not in flat


def test_bars_are_fractions_of_the_baseline() -> None:
    bars = bars_mod.build_bars(_synthetic(), "abc123")
    wrapper = next(e for e in bars["lanes"]["W1"] if e["counter"].endswith("phase.wrapper"))
    assert wrapper["fraction"] == pytest.approx(90 / 361, abs=1e-6)
    assert wrapper["bar"] == pytest.approx(40.0 * 90 / 361)
    assert bars["base_commit"] == "abc123"
    assert all(name.split(".")[0] in {"c2", "c7", "c9"} for name in bars["every_lane"])
    assert bars["landed"] == {}


def test_check_passes_at_the_baseline_and_names_what_rose() -> None:
    bars = bars_mod.build_bars(_synthetic(), "abc123")
    assert bars_mod.check(bars_mod.flatten(_synthetic()), bars) == []
    worse = _synthetic()
    worse["models"]["varied200"]["c2"]["default"]["blocks"] = 520
    failures = bars_mod.check(bars_mod.flatten(worse), bars)
    assert len(failures) == 1
    assert "c2.varied200.default.blocks_per_op" in failures[0]


def test_a_zero_baseline_may_not_rise_at_all() -> None:
    bars = bars_mod.build_bars(_synthetic(), "abc123")
    worse = _synthetic()
    worse["models"]["chain1000"]["c9"]["save_none"]["argument_bytes_max"] = 8
    failures = bars_mod.check(bars_mod.flatten(worse), bars)
    assert any("c9.chain1000.save_none.argument_bytes_max" in line for line in failures)


def test_lane_bars_apply_only_when_the_lane_is_named() -> None:
    bars = bars_mod.build_bars(_synthetic(), "abc123")
    flat = bars_mod.flatten(_synthetic())
    assert bars_mod.check(flat, bars) == []
    failures = bars_mod.check(flat, bars, lane="W1")
    assert len(failures) == 1 and "phase.wrapper" in failures[0]
    flat["c1.chain1000.phase.wrapper"] = 9.0
    assert bars_mod.check(flat, bars, lane="W1") == []
    assert bars_mod.check(flat, bars, lane="no-such-lane") != []


def test_a_minimum_bar_fails_below_and_passes_at_the_bar() -> None:
    bars = bars_mod.build_bars(_synthetic(), "abc123")
    flat = bars_mod.flatten(_synthetic())
    assert len(bars_mod.check(flat, bars, lane="W2")) == 2
    flat["c8.resnet18.fast_lane_share"] = 0.95
    flat["c8.gpt2_tiny4.fast_lane_share"] = 0.99
    assert bars_mod.check(flat, bars, lane="W2") == []


def test_tighten_lands_a_lane_at_measured_plus_headroom() -> None:
    bars = bars_mod.build_bars(_synthetic(), "abc123")
    flat = bars_mod.flatten(_synthetic())
    flat["c1.chain1000.phase.wrapper"] = 8.0
    landed = bars_mod.tighten(bars, "W1", flat)
    assert landed["landed"] == {"c1.chain1000.phase.wrapper": pytest.approx(8.4)}
    assert bars["landed"] == {}
    flat["c1.chain1000.phase.wrapper"] = 8.5
    assert any("landed bar" in line for line in bars_mod.check(flat, landed))


def test_compare_accepts_half_a_percent_and_reports_more() -> None:
    first = {"a": 1000.0, "b": 0.0, "c": 5.0}
    assert bars_mod.compare(first, {"a": 1004.0, "b": 0.0, "c": 5.0}) == []
    differences = bars_mod.compare(first, {"a": 1010.0, "b": 0.0})
    assert differences == ["a: 1000 against 1010", "c: measured in only one run"]


def test_committed_bar_file_is_the_plan_applied_to_its_baseline() -> None:
    bars = json.loads(cost.BARS_PATH.read_text(encoding="utf-8"))
    baseline = bars["baseline"]
    assert bars["schema"] == bars_mod.SCHEMA
    assert len(bars["base_commit"]) == 40
    expected = {(bar.lane, bar.counter): bar for bar in bars_mod.lane_bars()}
    stored = {(lane, e["counter"]): e for lane, entries in bars["lanes"].items() for e in entries}
    assert set(stored) == set(expected)
    for key, entry in stored.items():
        bar = expected[key]
        assert entry["kind"] == bar.kind
        if bar.kind == "fraction":
            assert entry["counter"] in baseline, entry["counter"]
            assert entry["fraction"] == pytest.approx(bar.value, abs=1e-6)
            assert entry["bar"] == pytest.approx(baseline[entry["counter"]] * bar.value)
        else:
            assert entry["bar"] == bar.value
    assert bars["every_lane"]
    assert set(bars["every_lane"]) <= set(baseline)
    for model in bars_mod.FULL_MODELS:
        assert f"c1.{model}.total_per_op" in baseline
    # A landed bar sits below the baseline by design, so the baseline is held
    # to the standing rules only; each landed bar names a lane counter.
    assert bars_mod.check(baseline, {**bars, "landed": {}}) == []
    lane_counters = {entry["counter"] for entries in bars["lanes"].values() for entry in entries}
    assert bars["landed_headroom"] == bars_mod.LANDED_HEADROOM
    for name, limit in bars["landed"].items():
        assert name in lane_counters and name in baseline, name
        assert limit > 0, name


def test_two_measurements_of_one_commit_agree() -> None:
    first = cost.measure(("one_op",), wall=False, isolate=False)
    second = cost.measure(("one_op",), wall=False, isolate=False)
    assert "error" not in first["models"]["one_op"], first["models"]["one_op"]
    flat = bars_mod.flatten(first)
    assert bars_mod.compare(flat, bars_mod.flatten(second)) == []
    assert flat["c1.one_op.total_per_op"] > 0
    assert flat["c6.minimal_trace_calls"] == first["models"]["one_op"]["c1"]["total"]
    c8 = first["models"]["one_op"]["c8"]
    assert c8["commit_ops"] >= 1 and c8["wrapper_calls"] >= 1
    assert c8["fast_lane_calls"] == 1 and 0.0 < c8["fast_lane_share"] <= 1.0
    # A trace keeps no Op view once built (B5: C4 is 0 for Op and Layer).
    assert first["models"]["one_op"]["c4"].get("Op", 0) == 0


class _Referent:
    """A weakly referenceable object for the C2 weak-reference tests."""


def _ignore(_ref: object) -> None:
    """Weak-reference callback; a reference with a callback is never shared."""


def test_c2_counts_a_kept_weak_reference_only_while_its_referent_lives() -> None:
    alive = _Referent()
    _, empty = retained_after(lambda: [])
    _, live = retained_after(lambda: [weakref.ref(alive)])
    _, dead = retained_after(lambda: [weakref.ref(_Referent())])
    assert live["objects"] - empty["objects"] == 1
    assert dead["objects"] == empty["objects"]
    assert dead["dead_weakrefs"] - empty["dead_weakrefs"] == 1


def test_c2_rises_when_a_trace_keeps_a_live_weak_reference() -> None:
    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU()).eval()
    x = torch.randn(2, 3)
    tl.trace(model, x)
    plain, without = retained_after(lambda: (tl.trace(model, x), None))
    del plain
    # Without a callback CPython hands back the model's existing weak reference.
    planted, with_ref = retained_after(lambda: (tl.trace(model, x), weakref.ref(model, _ignore)))
    del planted
    assert with_ref["objects"] - without["objects"] == 1


def _small_traced_model() -> tuple[nn.Module, torch.Tensor]:
    """A small model and input, traced once so later traces are warm."""

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU()).eval()
    x = torch.randn(2, 3)
    tl.trace(model, x)
    return model, x


def test_c2_counts_a_weak_id_key_dictionary_the_trace_keeps() -> None:
    model, x = _small_traced_model()
    _, alone = retained_after(lambda: None)
    _, one = retained_after(WeakIdKeyDictionary)
    plain, without = retained_after(lambda: (tl.trace(model, x), None))
    del plain
    planted, with_dict = retained_after(lambda: (tl.trace(model, x), WeakIdKeyDictionary()))
    del planted
    own_objects = one["objects"] - alone["objects"]
    assert own_objects > 1
    assert with_dict["objects"] - without["objects"] == own_objects


def _trace_leaving_buckets(model: nn.Module, x: torch.Tensor, extra: int) -> object:
    """Trace, then add ``extra`` empty alias buckets to the session it retired."""

    trace = tl.trace(model, x)
    retired = labels._RETIRED_LABEL_SESSION
    assert retired is not None
    for key in range(extra):
        # Negative keys: no storage address collides with them.
        retired.by_storage_ptr[-1 - key] = WeakIdKeyDictionary()
    return trace


def test_c2_leaves_out_the_label_session_a_trace_leaves_behind() -> None:
    # The alias index of the last trace's label session is released by the
    # next trace's start and its size follows the allocator; neither the
    # session left before the window nor the one the measured trace leaves
    # moves the count.
    model, x = _small_traced_model()
    counts = []
    for before, after in ((0, 0), (3, 0), (0, 2)):
        _trace_leaving_buckets(model, x, before)
        trace, held = retained_after(lambda after=after: _trace_leaving_buckets(model, x, after))
        del trace
        counts.append(held["objects"])
    assert counts[1] == counts[0] and counts[2] == counts[0], counts


def test_phase_counter_attributes_calls_and_leaves_no_profile_hook() -> None:
    import sys

    model, x = cost.Chain(4), torch.zeros(4)
    tl.trace(model, x)
    before = sys.getprofile()
    trace, phases, tags = count_phases(lambda: tl.trace(model, x))
    assert sys.getprofile() is before
    assert tags["commit_ops"] >= 4
    assert phases["wrapper"] > 0 and phases["post"] > 0 and phases["entry"] > 0
    _, total = count_calls(lambda: len(trace))
    assert total >= 1


def test_store_write_observer_sees_capture_ingest_and_pass_writes() -> None:
    model, x = cost.Chain(4), torch.zeros(4)
    tl.trace(model, x)
    observer = StoreWriteObserver()
    trace, _phases, _tags = count_phases(lambda: tl.trace(model, x), *observer.hooks())
    report = observer.report()
    assert report["capture_rows"] >= 4
    assert report["store_rows_at_freeze"] >= report["capture_rows"]
    assert report["cells"]["capture"] > 0
    assert report["cells"]["ingest"] > 0
    assert report["cells"]["passes"] > 0
    assert len(trace) >= 4


def test_held_tensor_bytes_classifies_by_the_outermost_name() -> None:
    saved, argument, shared, loose = (torch.zeros(n) for n in (2, 4, 8, 16))
    own_input, input_copy, forward_copy = torch.zeros(32), torch.zeros(64), torch.zeros(128)
    owner = nn.Linear(3, 3)
    root = {
        "out": saved,
        "saved_args": [argument, {"out": shared.view(-1)}],
        "transformed_out": shared,
        "layout": (loose, owner.weight),
        "module": owner,
        "kwargs": {"x": own_input[:8]},
        "_output_attribution_input_tensors": (own_input, input_copy),
        "workspace": {"root_forward_args": ((forward_copy,), {"args": forward_copy[:4]})},
    }
    owned = caller_owned_keys(owner, {"x": [own_input]})
    totals, holders, truncated = held_tensor_bytes(root, *owned)
    assert not truncated
    assert totals["saved_output"] == (2 + 8) * 4
    assert totals["argument"] == 4 * 4
    assert totals["other"] == (16 + 64) * 4
    assert totals["forward_input"] == 128 * 4
    assert totals["caller_owned"] == (9 + 32) * 4
    assert holders["argument"] == {"saved_args": 16}
    assert holders["saved_output"] == {"out": 8, "transformed_out": 32}


def test_tensor_bytes_observer_samples_every_op_boundary_of_a_short_forward() -> None:
    model, x = cost.Chain(4), torch.zeros(4)
    tl.trace(model, x)
    observer = TensorBytesObserver(model, x)
    count_phases(lambda: tl.trace(model, x), *observer.hooks())
    report = observer.report()
    assert report["boundaries"] >= 4
    assert report["sampled"] == report["boundaries"] + 1
    assert not report["walk_truncated"]
    assert report["peak_bytes"] >= report["saved_output_bytes_max"]
    assert report["peak_bytes"] > 0
