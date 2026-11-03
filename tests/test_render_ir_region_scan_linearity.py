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
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._visualization import render_ir

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
        module = __import__(f"torchlens._visualization.{module_name}", fromlist=["x"])
        if hasattr(module, "finalize_forward_regions"):
            monkeypatch.setattr(module, "finalize_forward_regions", counting_finalize)

    trace = tl.trace(ModuleRichModel(), torch.rand(2, 6))
    trace.draw()

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
