"""Tests for the call-time floor instrument (``benchmarks/capture_core_floor.py``).

The section counter must partition the forward's calls, find the emit's
sections on a plain model, leave no profile hook behind and count the same
twice. The flattener and the marker check are tested on synthetic documents.
"""

from __future__ import annotations

import sys

import pytest
import torch

import torchlens as tl
from benchmarks import capture_core_cost as cost, capture_core_floor as floor


def _counted(model: torch.nn.Module, x: torch.Tensor) -> dict:
    tl.trace(model, x)
    tl.trace(model, x)
    return floor.measure_sections(model, x)


def test_sections_partition_a_plain_forward_and_leave_no_hook() -> None:
    before = sys.getprofile()
    record = _counted(cost.Chain(8), torch.zeros(4))
    assert sys.getprofile() is before
    assert record["outputs"] == 8
    calls = record["calls"]
    assert set(calls) == {*floor.SECTIONS, *floor.OVERLAYS}
    for section in ("arg_scan", "call_site", "autograd", "output_facts", "payload"):
        assert calls[section] > 0, section
        assert record["hits"][section] >= 8, section
    assert calls["wrapper_outside_emit"] > 0 and calls["emit_rest"] > 0
    assert calls["label_read_stamp"] > 0
    # Chain has no submodules, so module handling is a cost per trace and the
    # argument scan a cost per output.
    longer = _counted(cost.Chain(32), torch.zeros(4))["calls"]
    assert longer["module_handling"] - calls["module_handling"] < 24 * 4
    # per output: at least 5x the per-trace module bound above; not a cost bar
    assert longer["arg_scan"] - calls["arg_scan"] > 24 * 20


def test_module_calls_are_attributed_to_module_handling() -> None:
    x = torch.randn(2, 4)
    modules = _counted(cost.ModRelu().eval(), x)
    functions = _counted(cost.FuncRelu(), x)
    assert modules["outputs"] == functions["outputs"] == cost.RELU_CALLS
    per_output = cost.RELU_CALLS
    assert modules["calls"]["module_handling"] > 10 * functions["calls"]["module_handling"]
    assert functions["calls"]["module_handling"] < 10 * per_output
    assert functions["hits"]["call_site"] == per_output
    assert modules["hits"]["call_site"] >= per_output
    assert modules["calls"]["model_code"] > 0


def test_two_counts_of_one_model_agree() -> None:
    """In one process the counts agree to the bars' tolerance.

    Separate processes with address randomization off agree exactly; the job
    runs ``--counters`` twice and ``--compare`` for that.
    """

    from benchmarks import capture_core_bars as bars_mod

    model, x = cost.LoopCell().eval(), torch.randn(2, 16)
    first = floor.flatten({"models": {"m": {"floor": _counted(model, x)}}})
    second = floor.flatten({"models": {"m": {"floor": floor.measure_sections(model, x)}}})
    assert bars_mod.compare(first, second) == []


def _document(calls: dict[str, int], hits: dict[str, int], outputs: int = 10) -> dict:
    return {"models": {"chain1000": {"floor": {"outputs": outputs, "calls": calls, "hits": hits}}}}


def test_flatten_divides_by_outputs_and_sums_the_partition() -> None:
    calls = dict.fromkeys((*floor.SECTIONS, *floor.OVERLAYS), 10)
    hits = dict.fromkeys(floor.SECTION_MARKERS, 10)
    flat = floor.flatten(_document(calls, hits))
    assert flat["floor.chain1000.call_site_per_output"] == 1.0
    assert flat["floor.chain1000.forward_total_per_output"] == len(floor.SECTIONS)
    assert flat["floor.chain1000.hits.payload_per_output"] == 1.0
    assert flat["floor.chain1000.outputs"] == 10
    assert floor.flatten({"models": {"one_op": {"rows": 1}}}) == {}


def test_check_markers_names_a_section_whose_markers_stopped_firing() -> None:
    calls = dict.fromkeys((*floor.SECTIONS, *floor.OVERLAYS), 10)
    hits = dict.fromkeys(floor.SECTION_MARKERS, 10)
    baseline = floor.flatten(_document(calls, hits))
    assert floor.check_markers(baseline, baseline) == []
    moved = floor.flatten(_document(calls, {**hits, "call_site": 0}))
    problems = floor.check_markers(moved, baseline)
    assert len(problems) == 1 and "call_site" in problems[0]


@pytest.mark.parametrize("section", sorted(floor.SECTION_MARKERS))
def test_every_marker_names_a_function_of_the_package(section: str) -> None:
    from pathlib import Path

    source = "\n".join(
        path.read_text(encoding="utf-8") for path in Path(tl.__file__).parent.rglob("*.py")
    )
    missing = [name for name in floor.SECTION_MARKERS[section] if f"def {name}(" not in source]
    assert not missing, missing
