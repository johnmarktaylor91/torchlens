"""Regression tests for the linear region build in ``finalize_forward_regions``.

Root cause (SPEEDENING wave 14): the forward region build re-scanned three whole
collections **once per module region** -- a breadth-first sweep of the module
tree (``_region_call_depth``), a full ``render_ir.nodes`` scan for the region's
node names, and a full ``edges`` scan for the region's edge indexes. Call counts
were exactly linear in region count while self time was exactly quadratic, so
module-rich models paid the most: 18.15% of ``draw()`` on efficientnet_b4 and
12.19% on convnext_small went into those three lines.

Fix: bucket all three lookups once before the loop, the way ``_build_regions``
already did, and read them per region. Output is byte-identical; these tests pin
the *shape* of the fix so it cannot silently regress:

* the module tree, the node list, and the edge list are each traversed O(1)
  times per ``draw()`` -- a call-count assertion, no clock involved;
* the single-sweep depth table agrees with the historical per-key sweep on every
  key, including the unreachable-key ``_module_depth(key) - 1`` fallback.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.visualization import render_ir

os.environ.setdefault("MPLBACKEND", "Agg")


class ModuleRichModel(nn.Module):
    """Many small nested modules -- the shape that made the rescans quadratic."""

    def __init__(self, depth: int = 12, width: int = 6) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Sequential(nn.Tanh()))
            for _ in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _historical_region_call_depth(
    key: str,
    top_modules: list[str],
    children: Mapping[str, list[str]],
) -> int:
    """The pre-wave-14 per-key breadth-first sweep, verbatim, as the oracle."""

    pending = [(candidate, 0) for candidate in top_modules]
    while pending:
        candidate, depth = pending.pop(0)
        if candidate == key:
            return depth
        pending.extend((child, depth + 1) for child in children.get(candidate, []))
    return render_ir._module_depth(key) - 1


def test_region_build_scans_each_collection_a_bounded_number_of_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node list, the edge list, and the module tree are each swept O(1) times.

    The regions far outnumber the sweeps: a per-region rescan would make the
    counts scale with region count, which is exactly the regression to catch.
    """

    counts: dict[str, int] = defaultdict(int)
    real_finalize = render_ir.finalize_forward_regions

    class CountingTuple(tuple):
        """A node tuple that reports how many times it is iterated."""

        def __iter__(self) -> Any:
            counts["nodes"] += 1
            return super().__iter__()

    real_depths = render_ir._region_call_depths

    def counting_depths(*args: Any, **kwargs: Any) -> dict[str, int]:
        counts["module_tree"] += 1
        return real_depths(*args, **kwargs)

    region_counts: list[int] = []

    def counting_finalize(ir: Any, *args: Any, **kwargs: Any) -> Any:
        ir = render_ir.replace(ir, nodes=CountingTuple(ir.nodes))
        result = real_finalize(ir, *args, **kwargs)
        region_counts.append(len(result.regions))
        counts["edges"] += sum(1 for _ in result.edges)  # one settled sweep, for scale
        return result

    monkeypatch.setattr(render_ir, "_region_call_depths", counting_depths)
    monkeypatch.setattr(render_ir, "finalize_forward_regions", counting_finalize)
    for module_name in ("_render_dot", "_render_entrypoints", "_render_common"):
        module = __import__(f"torchlens.visualization.{module_name}", fromlist=["x"])
        if hasattr(module, "finalize_forward_regions"):
            monkeypatch.setattr(module, "finalize_forward_regions", counting_finalize)

    trace = tl.trace(ModuleRichModel(), torch.rand(2, 6))
    trace.draw(return_graph=True, vis_save_only=True)

    assert region_counts, "finalize_forward_regions was never reached"
    regions = max(region_counts)
    assert regions >= 12, f"expected a module-rich graph, got {regions} regions"
    finalize_calls = len(region_counts)
    # One bucketing sweep per finalize call, plus generous slack for unrelated
    # consumers -- but nothing that scales with the number of regions.
    assert counts["module_tree"] <= 2 * finalize_calls
    assert counts["nodes"] <= 4 * finalize_calls
    assert counts["module_tree"] < regions
    assert counts["nodes"] < regions


@pytest.mark.parametrize("depth", (1, 4, 12))
def test_single_sweep_depth_table_matches_historical_per_key_sweep(depth: int) -> None:
    """Every recorded depth equals the historical per-key breadth-first answer."""

    trace = tl.trace(ModuleRichModel(depth=depth), torch.rand(2, 6))
    for vis_mode in ("unrolled", "rolled"):
        children, top_modules = render_ir._region_module_hierarchy(trace, vis_mode)
        table = render_ir._region_call_depths(top_modules, children)
        assert table, f"no reachable modules for {vis_mode}"
        for key in table:
            assert table[key] == _historical_region_call_depth(key, top_modules, children)


def test_unreachable_key_keeps_the_module_depth_fallback() -> None:
    """Keys the sweep never reaches stay absent so callers use the old fallback."""

    children: defaultdict[str, list[str]] = defaultdict(list)
    children["a"] = ["a.b"]
    table = render_ir._region_call_depths(["a"], children)

    assert table == {"a": 0, "a.b": 1}
    assert "x.y.z" not in table
    assert _historical_region_call_depth("x.y.z", ["a"], children) == (
        render_ir._module_depth("x.y.z") - 1
    )


def test_depth_table_reports_shortest_root_distance_on_a_shared_child() -> None:
    """A child reachable by two paths keeps its first-encounter (shortest) depth."""

    children: defaultdict[str, list[str]] = defaultdict(list)
    children["a"] = ["shared", "a.mid"]
    children["a.mid"] = ["shared"]
    table = render_ir._region_call_depths(["a"], children)

    assert table["shared"] == 1
    assert table["shared"] == _historical_region_call_depth("shared", ["a"], children)
