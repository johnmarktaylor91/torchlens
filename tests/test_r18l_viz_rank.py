"""Regression tests for r18l viz-plots + rank layout + graph caption findings.

Covers:
- H10  caption ``[:-2]`` left ``</FONT`` unterminated -> GraphvizRenderError on
       any ``_has_direct_writes=True`` trace.
- M4   rank ``_compute_topological_layout`` sibling order depended on set/hash
       iteration -> nondeterministic across PYTHONHASHSEED.
- M5   ``feature_map_evolution`` leaked the internal ``mds_evolution``/``MDS``
       error vocabulary on a recurrent layer.
- M6   ``channel_grid`` silently used only batch item 0 (undocumented).
- M7   rank engine silently dropped ``show_legend``.
- M8   ``render_lineplot`` produced a blank plot when no finite (x, y) PAIR existed.
- M9   ``_normalize_finite`` returned a uniform array on reversed vmin/vmax while
       its docstring claimed it raises.
- M10  rank engine silently ignored ``dpi`` and ``vis_graph_overrides``.
- F12  ``render_lineplot`` drew out-of-range points over the chart furniture.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.viz.node_plots import (
    _lineplot_point,
    _normalize_finite,
    render_heatmap,
    render_lineplot,
)


# ---------------------------------------------------------------------------
# H10 — caption terminates the FONT tag; draw() after set() must not crash
# ---------------------------------------------------------------------------
def test_h10_direct_writes_caption_renders(tmp_path):
    trace = tl.trace(nn.Linear(4, 2), torch.randn(1, 4))
    trace._has_direct_writes = True
    out = str(tmp_path / "h10")
    # Must not raise GraphvizRenderError.
    src = trace.draw(vis_save_only=True, vis_fileformat="svg", vis_outpath=out)
    assert isinstance(src, str)
    # The FONT element must be properly closed and the direct-writes line present.
    assert "</FONT>" in src
    assert "</FONTDirect" not in src
    assert "Direct writes detected" in src


def test_h10_caption_no_direct_writes_still_closes_font(tmp_path):
    trace = tl.trace(nn.Linear(4, 2), torch.randn(1, 4))
    out = str(tmp_path / "h10b")
    src = trace.draw(vis_save_only=True, vis_fileformat="svg", vis_outpath=out)
    assert "</FONT>" in src
    assert "Direct writes detected" not in src


# ---------------------------------------------------------------------------
# M9 — _normalize_finite raises on reversed bounds (per its documented contract)
# ---------------------------------------------------------------------------
def test_m9_reversed_bounds_raise():
    arr = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="vmax must be greater than or equal to vmin"):
        _normalize_finite(arr, 3.0, 0.0)
    with pytest.raises(ValueError, match="vmax must be greater than or equal to vmin"):
        render_heatmap(arr, vmin=3.0, vmax=0.0)


def test_m9_equal_bounds_still_uniform():
    arr = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    out = _normalize_finite(arr, 2.0, 2.0)
    assert np.all(out == 0.0)


def test_m9_ordered_bounds_normalize():
    arr = np.asarray([[0.0, 1.0], [2.0, 4.0]])
    out = _normalize_finite(arr, 0.0, 4.0)
    assert out.max() == pytest.approx(1.0)
    assert out.min() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# M8 — render_lineplot requires at least one finite (x, y) PAIR
# ---------------------------------------------------------------------------
def test_m8_no_finite_pair_raises():
    xs = np.asarray([np.nan, 0.0])
    ys = np.asarray([1.0, np.nan])  # finite y and finite x at disjoint indices
    with pytest.raises(ValueError, match=r"finite \(x, y\) point"):
        render_lineplot(ys, x_values=xs)


def test_m8_one_finite_pair_ok():
    xs = np.asarray([0.0, np.nan])
    ys = np.asarray([1.0, 5.0])  # one drawable pair at index 0
    img = render_lineplot(ys, x_values=xs)
    assert img.size[0] > 0 and img.size[1] > 0


# ---------------------------------------------------------------------------
# F12 — out-of-range lineplot points are clamped into the plot rectangle
# ---------------------------------------------------------------------------
def test_f12_out_of_range_point_clamped():
    pt = _lineplot_point(
        0.0,
        100.0,  # far above y-axis high of 1.0
        low_x=0.0,
        high_x=2.0,
        low_y=0.0,
        high_y=1.0,
        plot_left=10,
        plot_right=100,
        plot_top=5,
        plot_bottom=80,
    )
    assert pt is not None
    x, y = pt
    assert 10.0 <= x <= 100.0
    assert 5.0 <= y <= 80.0


def test_f12_in_range_point_unchanged():
    pt = _lineplot_point(
        1.0,
        0.5,
        low_x=0.0,
        high_x=2.0,
        low_y=0.0,
        high_y=1.0,
        plot_left=10,
        plot_right=100,
        plot_top=5,
        plot_bottom=80,
    )
    assert pt is not None
    x, y = pt
    assert x == pytest.approx(55.0)
    assert y == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# M6 — channel_grid documents that it renders only the first batch element
# ---------------------------------------------------------------------------
def test_m6_channel_grid_uses_first_batch_element():
    base = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]], [[[4.0, 5.0], [6.0, 7.0]]]])
    changed_tail = base.clone()
    changed_tail[1] = torch.tensor([[[400.0, -500.0], [600.0, -700.0]]])  # batch 1 differs
    cg = tl.viz.channel_grid(n=1, max_size=40)
    # Documented: batch element 0 is the only one rendered, so changing batch 1
    # must not change the output.
    assert cg(base).tobytes() == cg(changed_tail).tobytes()
    # ... but changing batch 0 must change the output (proves it uses batch 0).
    changed_head = base.clone()
    changed_head[0] = torch.tensor([[[9.0, -9.0], [3.0, -3.0]]])
    assert cg(base).tobytes() != cg(changed_head).tobytes()


# ---------------------------------------------------------------------------
# M5 — feature_map_evolution does not leak the mds_evolution/MDS error vocabulary
# ---------------------------------------------------------------------------
def test_m5_feature_map_evolution_recurrent_error_vocabulary():
    class RecurConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 1, bias=False)

        def forward(self, x):
            for _ in range(3):
                x = self.conv(x)
            return x

    trace = tl.trace(RecurConv(), torch.ones(2, 2, 4, 4))
    with pytest.raises(ValueError) as excinfo:
        tl.viz.feature_map_evolution(trace)
    message = str(excinfo.value)
    assert "mds_evolution" not in message
    assert "MDS" not in message
    assert "feature_map_evolution" in message
    # The pass-selection guidance must survive the recast.
    assert "select a pass" in message


# ---------------------------------------------------------------------------
# M4 — rank _compute_topological_layout order is PYTHONHASHSEED-independent
# ---------------------------------------------------------------------------
_M4_SNIPPET = (
    "from torchlens.visualization._rank_layout_internal.layout import "
    "_compute_topological_layout\n"
    "labels=['alpha','beta','gamma','delta','epsilon','zeta']\n"
    "nd={n:{'node_label':n,'attrs':{}} for n in labels}\n"
    "sz={n:(20.,10.) for n in labels}\n"
    "pos,_,_=_compute_topological_layout(nd,[],sz,{},{})\n"
    "order=[n for n,_ in sorted(pos.items(),key=lambda kv:kv[1][0])]\n"
    "print(','.join(order))\n"
)


def _rank_order_for_seed(seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    # Point the subprocess at this worktree so it imports the patched module.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.check_output([sys.executable, "-c", _M4_SNIPPET], env=env, text=True)
    return out.strip()


@pytest.mark.heavy
def test_m4_rank_order_deterministic_across_hashseed():
    orders = {seed: _rank_order_for_seed(seed) for seed in ("0", "1", "2", "3", "7")}
    assert len(set(orders.values())) == 1, orders


def test_m4_within_rank_sort_is_total_order():
    # Multi-root graph, no edges: every node is its own rank-0 root; positions
    # must follow the (module, node_label) total order deterministically.
    from torchlens.visualization._rank_layout_internal.layout import (
        _compute_topological_layout,
    )

    labels = ["zeta", "alpha", "mu", "beta"]
    nd = {n: {"node_label": n, "attrs": {}} for n in labels}
    sz = dict.fromkeys(labels, (20.0, 10.0))
    pos, _, _ = _compute_topological_layout(nd, [], sz, {}, {})
    order = [n for n, _ in sorted(pos.items(), key=lambda kv: kv[1][0])]
    assert order == sorted(labels)


# ---------------------------------------------------------------------------
# M7 — rank engine honors show_legend (parity with the dot engine)
# ---------------------------------------------------------------------------
def _rank_model():
    return nn.Sequential(nn.Linear(2, 2), nn.ReLU())


def test_m7_rank_engine_emits_legend(tmp_path):
    trace = tl.trace(_rank_model(), torch.ones(1, 2))
    src = trace.draw(
        vis_outpath=str(tmp_path / "m7rank"),
        vis_fileformat="svg",
        vis_save_only=True,
        vis_node_placement="rank",
        show_legend=True,
    )
    # One-table legend (vizmech 13): a single pinned plaintext node.
    assert "tl_legend" in src
    assert "TorchLens legend" in src
    svg = (tmp_path / "m7rank.svg").read_text()
    assert "TorchLens legend" in svg
    assert ">input<" in svg and ">output<" in svg


def test_m7_rank_engine_no_legend_by_default(tmp_path):
    trace = tl.trace(_rank_model(), torch.ones(1, 2))
    src = trace.draw(
        vis_outpath=str(tmp_path / "m7rank_off"),
        vis_fileformat="svg",
        vis_save_only=True,
        vis_node_placement="rank",
        show_legend=False,
    )
    assert "tl_legend" not in src


# ---------------------------------------------------------------------------
# M10 — rank engine honors dpi and vis_graph_overrides
# ---------------------------------------------------------------------------
def test_m10_rank_engine_applies_dpi_and_overrides(tmp_path):
    trace = tl.trace(_rank_model(), torch.ones(1, 2))
    src = trace.draw(
        vis_outpath=str(tmp_path / "m10rank"),
        vis_fileformat="png",
        vis_save_only=True,
        vis_node_placement="rank",
        dpi=123,
        vis_graph_overrides={"bgcolor": "lightyellow"},
    )
    # dpi applies on the raster target (png)...
    assert "dpi=123" in src
    assert "bgcolor=" in src and "lightyellow" in src
    # ...and is DROPPED on a vector target (svg): raster-only, vizmech D23.
    src_svg = trace.draw(
        vis_outpath=str(tmp_path / "m10rank_svg"),
        vis_fileformat="svg",
        vis_save_only=True,
        vis_node_placement="rank",
        dpi=123,
    )
    assert "dpi=123" not in src_svg


# ---------------------------------------------------------------------------
# Rank engine creates its output directory (parity with the dot engine)
# ---------------------------------------------------------------------------
def test_rank_engine_creates_missing_output_directory(tmp_path):
    """The rank layout path must create a non-existent nested container_path.

    The dot-layout path (_render_utils.render_dot_to_file) always creates its
    output directory first; the rank-layout path (chosen automatically for
    large graphs -- e.g. detection models with many nodes) wrote straight to
    ``f"{vis_outpath}.dot"`` with no such check, raising ``FileNotFoundError``
    whenever ``container_path``'s directory did not already exist (weekly CI,
    2026-10-02: keypointrcnn/fasterrcnn detection model tests).
    """

    outdir = tmp_path / "new" / "nested" / "dir"
    assert not outdir.exists()
    trace = tl.trace(_rank_model(), torch.ones(1, 2))
    trace.draw(
        vis_outpath=str(outdir / "graph"),
        vis_fileformat="svg",
        vis_save_only=True,
        vis_node_placement="rank",
    )
    # The intermediate .dot source is removed after a successful render
    # (render_rank_layout: `if render_succeeded and os.path.exists(source_path):
    # os.remove(source_path)`), so only the final rendered artifact persists;
    # the directory-creation fix is proven by this file existing at all.
    assert (outdir / "graph.svg").exists()
