"""Vizmech wave-3 items 21-22: theme x direction x depth x format x engine gate.

Pairwise coverage per PR (memo s7 axes; the full cross-product is a scheduled
venue). Every render is audited THROUGH THE ENGINE THAT RENDERED it (D4/D8:
the same cluster measured 356 pt under dot and 1260 pt under ``neato -n``),
and every failure message carries the dot version, theme, and engine (plank
8). Item 21: the rank engine is a first-class axis value, so every asserted
geometry class has its ``neato -n`` twin -- the axis that was empty in every
lab's corpus.

Re-baseline law (item 22): these assertions bind to zero-violation targets
directly; any future re-baseline happens only at zero open root causes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._geometry_audit import (
    audit_layout,
    compute_usability_envelope,
    graphviz_version,
    parse_layout_json,
    run_layout_json,
)

pytestmark = pytest.mark.smoke

_AXES: dict[str, tuple] = {
    "theme": ("torchlens", "paper", "dark", "colorblind", "high_contrast"),
    "direction": ("bottomup", "topdown", "leftright"),
    "depth": (1, 1000),
    "fmt": ("svg", "png"),
    "engine": ("dot", "rank"),
}


def _pairwise_cover(axes: dict[str, tuple]) -> list[dict[str, object]]:
    """Deterministic greedy all-pairs cover over the axis dict.

    Not minimal, but stable across runs (no randomness) and complete: every
    value pair of every axis pair appears in at least one row.
    """

    names = list(axes)
    wanted: set[tuple[str, object, str, object]] = set()
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for va in axes[a]:
                for vb in axes[b]:
                    wanted.add((a, va, b, vb))

    def pairs_of(row: dict[str, object]) -> set[tuple[str, object, str, object]]:
        out = set()
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                out.add((a, row[a], b, row[b]))
        return out

    from itertools import product

    all_rows = [
        dict(zip(names, values, strict=True)) for values in product(*(axes[name] for name in names))
    ]
    cover: list[dict[str, object]] = []
    while wanted:
        best_row = max(all_rows, key=lambda row: (len(pairs_of(row) & wanted),))
        gained = pairs_of(best_row) & wanted
        if not gained:
            break
        cover.append(best_row)
        wanted -= gained
    return cover


_COVER = _pairwise_cover(_AXES)


class _Nested(nn.Module):
    """Nested fixture: clusters, a reused activation, and a two-input join."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.inner(x)
        return torch.sigmoid(hidden) + hidden


@pytest.fixture(scope="module")
def nested_trace():
    """One capture shared by every axis combo (draw is re-entrant)."""

    trace = tl.trace(_Nested(), torch.randn(1, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _context(combo: dict[str, object]) -> str:
    """Failure-message context: dot version + theme + engine + the combo."""

    return (
        f"[{graphviz_version()}] theme={combo['theme']} engine={combo['engine']} "
        f"direction={combo['direction']} depth={combo['depth']} fmt={combo['fmt']}"
    )


@pytest.mark.parametrize(
    "combo",
    _COVER,
    ids=lambda c: f"{c['theme']}-{c['direction']}-d{c['depth']}-{c['fmt']}-{c['engine']}",
)
def test_axis_combo_renders_clean(
    combo: dict[str, object], nested_trace: tl.Trace, tmp_path: Path
) -> None:
    """One pairwise-cover row: render, audit through the rendering engine."""

    dot = nested_trace.draw(
        vis_save_only=True,
        vis_fileformat=str(combo["fmt"]),
        vis_outpath=str(tmp_path / "axis"),
        vis_theme=str(combo["theme"]),
        direction=combo["direction"],  # type: ignore[arg-type]
        vis_call_depth=int(combo["depth"]),  # type: ignore[arg-type]
        vis_node_placement=str(combo["engine"]) if combo["engine"] == "rank" else "dot",
        show_legend=True,
    )
    engine = "rank" if combo["engine"] == "rank" else "dot"
    parsed = parse_layout_json(run_layout_json(dot, engine), engine=engine, dot_source=dot)
    result = audit_layout(parsed)
    assert result.hard_violation_count == 0, f"{_context(combo)}\n{result.describe('axis combo')}"
    result.require_minimums(node=3, legend_text=3)
    envelope = compute_usability_envelope(parsed)
    assert envelope.arrowhead_fused_pairs == 0, _context(combo)
    # Engine discriminator (D8): pinned positions exactly on the rank path.
    if engine == "rank":
        assert parsed.pin_count > 0, _context(combo)
    else:
        assert parsed.pin_count == 0, _context(combo)


def test_pairwise_cover_is_complete() -> None:
    """The greedy cover really covers every axis-value pair."""

    names = list(_AXES)
    covered = set()
    for row in _COVER:
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                covered.add((a, row[a], b, row[b]))
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for va in _AXES[a]:
                for vb in _AXES[b]:
                    assert (a, va, b, vb) in covered, f"pair missing: {a}={va}, {b}={vb}"


def test_rank_entry_cluster_width_vs_member_extent(tmp_path: Path) -> None:
    """Item 16/21: the rank-path cluster-width assertion, via neato -n.

    The defect measured one-node clusters at a median 91% of graph width
    (ratio >> 10 vs member extent); the healthy fixture stays under 5x with
    the item-16 fix in place (measured ~2.2x)."""

    trace = tl.trace(_Nested(), torch.randn(1, 4))
    dot = trace.draw(
        vis_save_only=True,
        vis_fileformat="svg",
        vis_outpath=str(tmp_path / "rank_width"),
        vis_node_placement="rank",
    )
    parsed = parse_layout_json(run_layout_json(dot, "rank"), engine="rank", dot_source=dot)
    envelope = compute_usability_envelope(parsed)
    assert envelope.cluster_width_ratios, "rank entry lost its clusters"
    assert envelope.cluster_width_ratio_max < 5.0, envelope.cluster_width_ratios
