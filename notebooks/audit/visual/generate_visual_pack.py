"""Visual audit pack generator for TorchLens (v2).

Run as:
    python notebooks/audit/visual/generate_visual_pack.py

Produces:
    notebooks/audit/visual/visual_audit.pdf   -- the stapled audit montage
    notebooks/audit/visual/coverage_matrix.md -- auto-generated coverage matrix
    notebooks/audit/visual/_pages/            -- per-page intermediates (gitignored)

Design: single render-then-staple script.  Every demo is ONE composed PDF page
(header + caption + graph panels embedded at native resolution), organized into
titled sections with a table-of-contents page.  The goal is a scrollable
montage of the complete TorchLens visual language: after scrolling it, there
should be zero surprises about how any TorchLens render can look.

Coverage is enforced mechanically: every page declares `covers=` axis tags,
and the coverage matrix diffs those tags against the AXES enumeration derived
from the renderer/option source.  Uncovered axes are listed as defects.
"""

from __future__ import annotations

# matplotlib.use("Agg") and the sys.path shim must precede zoo/torchlens
# imports, so module-level imports legitimately follow code (E402).
# ruff: noqa: E402

import gc
import pathlib
import shutil
import sys
import traceback

_VISUAL_DIR = pathlib.Path(__file__).resolve().parent
_AUDIT_DIR = _VISUAL_DIR.parent  # notebooks/audit/
sys.path.insert(0, str(_AUDIT_DIR))
sys.path.insert(0, str(_VISUAL_DIR))

import matplotlib

matplotlib.use("Agg")

import torch

import torchlens as tl
from _models import ZOO
from _pagekit import Page, Panel, Section, compose_page, compose_text_page
from _visual_models import VZOO

MODELS: dict = {**ZOO, **VZOO}

_PAGES_DIR = _VISUAL_DIR / "_pages"
_OUT_PDF = _VISUAL_DIR / "visual_audit.pdf"
_MATRIX_MD = _VISUAL_DIR / "coverage_matrix.md"

torch.manual_seed(1234)

# ---------------------------------------------------------------------------
# Trace cache
# ---------------------------------------------------------------------------

_TRACE_CACHE: dict = {}


def get_trace(model_key: str, variant: str = "plain", builder=None):
    """Return a cached trace for (model_key, variant)."""
    ck = (model_key, variant)
    if ck in _TRACE_CACHE:
        return _TRACE_CACHE[ck]
    if builder is not None:
        trace = builder()
    else:
        m, x = MODELS[model_key]()
        xs = x if isinstance(x, tuple) else (x,)
        if variant == "plain":
            trace = tl.trace(m, *xs)
        elif variant == "backward":
            trace = tl.trace(m, *xs, backward_ready=True)
            out = trace[-1].out
            trace.log_backward(out)
        else:
            raise ValueError(f"Unknown trace variant {variant!r} without builder")
    _TRACE_CACHE[ck] = trace
    return trace


def _intervened_trace(model_key: str):
    def _build():
        m, x = MODELS[model_key]()
        return tl.trace(m, x, intervene=tl.when(tl.func("relu"), tl.zero_ablate()))

    return _build


def _transform_trace(model_key: str):
    def _build():
        m, x = MODELS[model_key]()
        return tl.trace(m, x, transform=lambda inp: (inp - inp.mean()) / (inp.std() + 1e-6))

    return _build


def _raw_input_trace(model_key: str):
    def _build():
        m, x = MODELS[model_key]()
        return tl.trace(m, x, save_raw_input=True, save_raw_output=True)

    return _build


# ---------------------------------------------------------------------------
# Custom hooks used by demo pages
# ---------------------------------------------------------------------------


def _spec_highlight_relu(layer_log, spec):
    """node_spec_fn demo: paint every relu gold with a bold border."""
    if getattr(layer_log, "func_name", "") == "relu":
        spec.fillcolor = "#FFD700"
        spec.penwidth = 2.5
        spec.color = "#8a6d00"
        spec.lines = [*spec.lines, "<-- node_spec_fn hit"]
    return spec


def _collapsed_spec_tag(module_log, spec):
    """collapsed_node_spec_fn demo: tag every collapsed box."""
    spec.fillcolor = "#D0E8FF"
    spec.lines = [*spec.lines, "<-- collapsed_node_spec_fn hit"]
    return spec


def _collapse_res_blocks(module_log):
    """collapse_fn demo: collapse every SmallResBlock instance."""
    return getattr(module_log, "module_type", "") == "SmallResBlock"


def _skip_relus(layer_log):
    """skip_fn demo: keep everything EXCEPT relu ops."""
    return getattr(layer_log, "func_name", "") != "relu"


def _custom_overlay_kwargs(trace) -> dict:
    """Custom node_overlay mapping: score = op position in the graph."""
    labels = [layer.layer_label for layer in trace.layer_list]
    return {"node_overlay": {lab: float(i) for i, lab in enumerate(labels)}}


def _plan_subtitle(t: float):
    def _fn(trace) -> str:
        try:
            plan = trace.collapse_plan(mode=t)
            total = getattr(plan, "total", None)
            if total is None:
                total = repr(plan)
            return f"collapse={t} -- {total} visible nodes"
        except Exception as exc:  # diagnostic subtitle must never kill the page
            return f"collapse={t} (plan unavailable: {type(exc).__name__})"

    return _fn


def _collapse_diag_text() -> str:
    """Text body for the collapse diagnostics page (resnet18)."""
    trace = get_trace("resnet18")
    lines: list[str] = []
    lines.append("Trace.collapse_plan(mode=...) -- renderer-faithful plan per mode (resnet18):")
    lines.append("")
    for mode in ("auto", "max", 0.5):
        try:
            plan = trace.collapse_plan(mode=mode)
            lines.append(f"  collapse_plan(mode={mode!r}):")
            lines.append(f"    {plan!r}")
        except Exception as exc:
            lines.append(f"  collapse_plan(mode={mode!r}) FAILED: {exc}")
    lines.append("")
    lines.append("Trace.collapse_schedule() -- ordered monotone float schedule:")
    lines.append("")
    lines.append("      t      target  visible  #collapsed-module-addresses")
    try:
        schedule = trace.collapse_schedule()
        steps = list(schedule.steps)
        shown = steps if len(steps) <= 28 else steps[:14] + steps[-14:]
        skipped = len(steps) - len(shown)
        prev_t = None
        for i, step in enumerate(shown):
            if skipped and i == 14:
                lines.append(f"      ... ({skipped} intermediate steps elided) ...")
            t = getattr(step, "t", None)
            marker = " " if t != prev_t else "="
            prev_t = t
            lines.append(
                f"    {marker} {t:<7.3f}{getattr(step, 'target_count', '?'):>7}"
                f"{getattr(step, 'visible_count', '?'):>9}"
                f"{len(getattr(step, 'collapsed_addresses', ())):>10}"
            )
        lines.append("")
        lines.append(f"  Total steps: {len(steps)}.  Monotone contract: visible count never")
        lines.append("  increases with t; collapsed-address sets are nested supersets.")
    except Exception as exc:
        lines.append(f"    collapse_schedule() FAILED: {exc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coverage axes (enumerated from source: _literals.py, _trace_viz.py draw(),
# themes.py THEME_PRESETS, modes.py NODE_MODES, rendering.py node/edge kinds)
# ---------------------------------------------------------------------------

AXES: dict[str, str] = {
    # --- draw() kwargs x values ---
    "vis_mode:unrolled": "vis_mode='unrolled' (per-pass op nodes)",
    "vis_mode:rolled": "vis_mode='rolled' (aggregate layer nodes, (xN) labels)",
    "collapse:none": "collapse='none' (full graph)",
    "collapse:auto": "collapse='auto' (readability-targeted engine)",
    "collapse:max": "collapse='max' (aggressive condensation + segments)",
    "collapse:float": "collapse=t float slider in [0,1], monotone schedule",
    "fold_runs:none": "fold_runs=None default policy (off for collapse='none')",
    "fold_runs:true": "fold_runs=True (fold every legal run)",
    "fold_runs:false": "fold_runs=False (folding disabled)",
    "fold_runs:standalone": "fold_runs=True with collapse='none' (fold-only mode)",
    "direction:bottomup": "direction='bottomup' (default)",
    "direction:topdown": "direction='topdown'",
    "direction:leftright": "direction='leftright'",
    "node_mode:default": "node_mode='default'",
    "node_mode:profiling": "node_mode='profiling' (time/memory rows)",
    "node_mode:vision": "node_mode='vision' (conv-oriented rows)",
    "node_mode:attention": "node_mode='attention' (attention-oriented rows)",
    "theme:torchlens": "vis_theme='torchlens' (default)",
    "theme:paper": "vis_theme='paper'",
    "theme:dark": "vis_theme='dark'",
    "theme:colorblind": "vis_theme='colorblind'",
    "theme:high_contrast": "vis_theme='high_contrast'",
    "show_legend": "show_legend=True legend box",
    "buffers:never": "show_buffer_layers='never'",
    "buffers:meaningful": "show_buffer_layers='meaningful' (default)",
    "buffers:always": "show_buffer_layers='always'",
    "containers:labels": "show_containers='labels'",
    "containers:cluster": "show_containers='cluster'",
    "containers:collapsed": "show_containers='collapsed'",
    "containers:auto": "show_containers='auto'",
    "container_max_inline": "container_max_inline threshold",
    "module_focus": "module= submodule focus with boundary stubs",
    "vis_call_depth": "vis_call_depth module-box nesting cutoff",
    "skip_fn": "skip_fn layer-hiding predicate (edges chain through)",
    "collapse_fn": "collapse_fn custom module-collapse predicate",
    "node_spec_fn": "node_spec_fn per-node customization callback",
    "collapsed_node_spec_fn": "collapsed_node_spec_fn collapsed-box callback",
    "overlay:flops": "node_overlay='flops' builtin overlay",
    "overlay:time": "node_overlay='time' builtin overlay",
    "overlay:magnitude": "node_overlay='magnitude' builtin overlay",
    "overlay:nan": "node_overlay='nan' non-finite highlighter",
    "overlay:custom": "node_overlay={label: score} custom mapping",
    "node_label_fields": "node_label_fields row picker",
    "code_panel:true": "code_panel=True source panel",
    "code_panel:init_forward": "code_panel='init+forward' source panel",
    "font_size": "font_size= override",
    "dpi": "dpi= render density override",
    "for_paper": "for_paper=True publication styling",
    "order_siblings:on": "order_siblings=True (default) sibling ordering post-pass",
    "order_siblings:off": "order_siblings=False raw dot order",
    "placement:dot": "vis_node_placement='dot' Graphviz engine",
    "placement:rank": "vis_node_placement='rank' internal rank layout",
    "overrides:graph": "vis_graph_overrides dict",
    "overrides:edge": "vis_edge_overrides dict",
    "overrides:module": "vis_module_overrides dict",
    "intervention:node_mark": "vis_intervention_mode='node_mark'",
    "intervention:as_node": "vis_intervention_mode='as_node' hook nodes",
    "cone:on": "vis_show_cone=True downstream cone styling",
    "cone:off": "vis_show_cone=False",
    "input_transform_summary": "show_input_transform_summary=True input xlabel",
    "raw_io_render": "save_raw_input/save_raw_output thumbnails on boundary nodes",
    # --- draw_backward / draw_combined ---
    "draw_backward:rolled": "draw_backward default rolled grad_fn graph",
    "draw_backward:unrolled": "draw_backward vis_mode='unrolled'",
    "draw_backward:bwd": "draw_backward bwd= pass selection",
    "draw_combined": "draw_combined forward+backward graph",
    "combined:intervening_cluster": "draw_combined intervening_cluster placement",
    # --- node kinds ---
    "node:raw_op": "plain operation node",
    "node:param_op": "parameter-bearing op node styling",
    "node:boundary_io": "input/output boundary nodes",
    "node:multi_io": "multiple input / multiple output boundary nodes",
    "node:buffer": "buffer node (+ hidden-buffer double border)",
    "node:module_box": "nested module cluster box",
    "node:atomic_module": "atomic module box",
    "node:collapsed_module": "collapsed module representative box",
    "node:ellipsis": "run-fold ellipsis node ('+N more Class')",
    "node:segment": "segment box (max-mode condensed range)",
    "node:container": "container nodes (dict/tuple outputs)",
    "node:boundary_stub": "module-focus boundary stubs",
    "node:intervention": "intervention site/hook rendering",
    "node:grad_fn": "backward grad_fn nodes",
    # --- edge kinds ---
    "edge:data": "plain data edge",
    "edge:multiplicity": "parallel edges for repeated args",
    "edge:back_edge": "rolled-mode recurrence back-edge",
    "edge:self_loop": "rolled-mode self-loop",
    "edge:cross_module": "edge crossing module-box boundary",
    "edge:container_tie": "dashed container grouping tie",
    "edge:backward": "backward gradient edges",
    "edge:fwd_bwd_tie": "dashed forward-to-backward correspondence edges",
    # --- label grammar ---
    "label:xN": "'(xN)' true recurrence multiplier (same params)",
    "label:plusN": "'+N more Class' ellipsis (distinct same-type instances)",
    "label:remainder": "collapsed-box 'N layers total' remainder (incl. buffers)",
    "label:segment_range": "segment box address-range label",
    "label:arm": "conditional IF/ELIF/ELSE arm labels",
    # --- control flow / structure ---
    "control_flow:branch": "tensor-driven if/else taken-arm rendering",
    "topology:parallel": "parallel branch / merge topology",
    "topology:transformer": "transformer / attention block topology",
    "scale:real_model": "real torchvision-scale architecture",
    "degenerate": "degenerate graphs (single op, no modules, scalar out)",
    "regression:large_bbox": "large-graph bbox non-blank regression",
    "artifact:apparent_cycle": "interleaved run-fold apparent-cycle artifact (known)",
    "diag:collapse_plan": "Trace.collapse_plan() diagnostic",
    "diag:collapse_schedule": "Trace.collapse_schedule() diagnostic",
}

# Axes that are deliberately NOT given a page, with the honest reason.
NA_AXES: dict[str, str] = {
    "vis_mode:none": "vis_mode='none' renders nothing by definition.",
    "vis_renderer:dagua": "Experimental opt-in backend (torchlens.experimental.dagua); "
    "not part of the stable visual language.",
    "placement:auto": "'auto' just selects dot or rank by cost estimate; both concrete "
    "engines are shown on the placement page.",
    "code_panel:forward_class": "code_panel='forward' and 'class' use the same panel "
    "machinery as True / 'init+forward' (shown); only the excerpt differs.",
    "overlay:others": "node_overlay 'bytes'/'grad-norm'/'intervention'/'bundle_delta' "
    "use the same border-intensity machinery as the overlays shown.",
    "combined:cluster_rest": "intervening_cluster 'outside'/'downstream' mirror the two "
    "placements shown ('upstream'/'own') with different cluster targets.",
    "overrides:grad_edge": "vis_grad_edge_overrides styles gradient edges via the same "
    "override dict machinery as vis_edge_overrides (shown).",
    "plumbing": "vis_outpath / vis_fileformat / vis_save_only / return_graph / "
    "vis_graph_overrides-free aliases (vis_opt, view, depth, renderer, layout, "
    "node_style, vis_node_mode, vis_buffers, vis_direction) have no visual identity "
    "of their own.",
    "show:dispatcher": "Trace.show() dispatches to draw()/repr; no separate visuals.",
}

# ---------------------------------------------------------------------------
# SECTION / PAGE DEFINITIONS
# ---------------------------------------------------------------------------

SECTIONS: list[Section] = [
    # =====================================================================
    Section(
        "A",
        "Node & Edge Vocabulary",
        "The basic visual alphabet: what op nodes, parameter ops, module boxes, "
        "buffers, boundary nodes, and edges look like before any advanced option "
        "is turned on.",
        [
            Page(
                label="a1_baseline",
                title="Baseline render: the anatomy of a TorchLens graph",
                caption=(
                    "Default draw() of a 2-layer MLP. Vocabulary to internalize: rounded boxes are single "
                    "operations; each label shows the op name with its layer index, the output shape, and (for "
                    "parameter ops like linear) the parameter shapes. Parameter-bearing ops are shaded; "
                    "parameter-free ops (relu) are white. The oval-ish nodes at the ends are the input and "
                    "output boundary nodes. Rectangles around groups of ops are module boxes labeled with the "
                    "module address and class.\n"
                    "CHECK: fonts render cleanly, arrowheads point along data flow (bottom-up by default), "
                    "nothing overlaps."
                ),
                panels=[Panel("tiny_mlp -- draw() with all defaults", "tiny_mlp")],
                covers=[
                    "vis_mode:unrolled",
                    "node:raw_op",
                    "node:param_op",
                    "node:boundary_io",
                    "node:module_box",
                    "edge:data",
                    "buffers:meaningful",
                ],
            ),
            Page(
                label="a2_kitchen_sink",
                title="Kitchen sink: nesting, buffers, functional ops, randomness",
                caption=(
                    "DemoModel exercises several vocabulary items at once: a cos op on the input, an addition "
                    "that reads a registered buffer (buffer node styled distinctly, edge into the add), a "
                    "two-level nested module (inner_module containing loop_module), and an op fed by a fresh "
                    "random tensor.\n"
                    "CHECK: the buffer node is visually distinct from data ops; module boxes nest without "
                    "clipping; edges crossing module-box boundaries stay attached to their nodes."
                ),
                panels=[
                    Panel(
                        "demo_model -- show_buffer_layers='always'",
                        "demo_model",
                        kwargs={"show_buffer_layers": "always"},
                    )
                ],
                covers=["node:atomic_module", "node:buffer", "edge:cross_module"],
            ),
            Page(
                label="a3_multi_io",
                title="Multiple inputs and multiple outputs",
                caption=(
                    "A model with TWO tensor inputs (different sizes) and TWO tensor outputs returned as a "
                    "tuple. Each input gets its own boundary node; each output gets its own terminal node.\n"
                    "CHECK: both inputs present and labeled distinctly; both outputs present; the shared "
                    "hidden branches (ya+yb and ya-yb) fan out correctly from the two encoders."
                ),
                panels=[Panel("multi_in_multi_out -- 2 inputs, 2 outputs", "multi_in_multi_out")],
                covers=["node:multi_io", "topology:parallel"],
            ),
            Page(
                label="a4_buffers",
                title="Buffer visibility: never / meaningful / always",
                caption=(
                    "BatchNorm keeps running_mean / running_var / num_batches_tracked buffers. "
                    "show_buffer_layers controls their rendering: 'never' hides all buffer nodes, "
                    "'meaningful' (the default) shows buffers that carry semantic content while hiding "
                    "bookkeeping counters, 'always' shows every buffer.\n"
                    "CHECK: in 'never'/'meaningful', ops whose buffers are hidden get a DOUBLE BORDER "
                    "(peripheries=2) as an honest 'something is hidden here' marker; in 'always' the "
                    "num_batches_tracked counter also appears."
                ),
                panels=[
                    Panel(
                        "show_buffer_layers='never'",
                        "batch_norm",
                        kwargs={"show_buffer_layers": "never"},
                    ),
                    Panel(
                        "show_buffer_layers='meaningful' (default)",
                        "batch_norm",
                        kwargs={"show_buffer_layers": "meaningful"},
                    ),
                    Panel(
                        "show_buffer_layers='always'",
                        "batch_norm",
                        kwargs={"show_buffer_layers": "always"},
                    ),
                ],
                covers=["buffers:never", "buffers:meaningful", "buffers:always", "node:buffer"],
            ),
            Page(
                label="a5_edge_multiplicity",
                title="Edge multiplicity: the same tensor used twice",
                caption=(
                    "LEFT: add(x, x) -- the same input feeds both arguments, so TWO parallel edges run from "
                    "the input to the add node. RIGHT: cat([x, x]) -- sequence-argument multiplicity, again "
                    "two edges.\n"
                    "CHECK: both edges are individually visible (not overdrawn into one), with correct "
                    "argument labeling."
                ),
                panels=[
                    Panel("add_twice -- add(x, x)", "add_twice"),
                    Panel("cat_twice -- cat([x, x])", "cat_twice"),
                ],
                covers=["edge:multiplicity"],
            ),
            Page(
                label="a6_legend",
                title="The built-in legend",
                caption=(
                    "show_legend=True appends a legend box decoding the node color/shape vocabulary.\n"
                    "CHECK: legend entries match the actual styling used in the graph above them; legend "
                    "does not overlap the graph."
                ),
                panels=[
                    Panel("tiny_mlp -- show_legend=True", "tiny_mlp", kwargs={"show_legend": True})
                ],
                covers=["show_legend"],
            ),
        ],
    ),
    # =====================================================================
    Section(
        "B",
        "Layout, Direction & Placement",
        "How the same graph is arranged on the page: flow direction, sibling "
        "ordering, the large-graph regression case, and the two layout engines.",
        [
            Page(
                label="b1_direction",
                title="direction: bottomup / topdown / leftright",
                caption=(
                    "The same MLP laid out in all three flow directions. 'bottomup' is the default "
                    "(inputs at the bottom, like a stack trace); 'topdown' and 'leftright' re-rank the "
                    "same topology.\n"
                    "CHECK: arrowheads always point from producer to consumer; labels stay horizontal and "
                    "legible in leftright mode."
                ),
                panels=[
                    Panel(
                        "direction='bottomup' (default)",
                        "tiny_mlp",
                        kwargs={"direction": "bottomup"},
                    ),
                    Panel("direction='topdown'", "tiny_mlp", kwargs={"direction": "topdown"}),
                    Panel("direction='leftright'", "tiny_mlp", kwargs={"direction": "leftright"}),
                ],
                covers=["direction:bottomup", "direction:topdown", "direction:leftright"],
            ),
            Page(
                label="b2_order_siblings",
                title="order_siblings: execution order vs raw dot order",
                caption=(
                    "Four parallel projection branches executed in a fixed order. With order_siblings=True "
                    "(the default) a post-pass re-orders true parallel siblings left-to-right by execution "
                    "order; with False you get whatever order Graphviz dot picks.\n"
                    "CHECK: in the LEFT panel the branches appear in execution order (proj_0..proj_3); the "
                    "RIGHT panel may show them shuffled -- that difference is exactly what the option buys."
                ),
                panels=[
                    Panel(
                        "order_siblings=True (default)",
                        "parallel_fanout",
                        kwargs={"order_siblings": True},
                    ),
                    Panel(
                        "order_siblings=False", "parallel_fanout", kwargs={"order_siblings": False}
                    ),
                ],
                covers=["order_siblings:on", "order_siblings:off", "topology:parallel"],
            ),
            Page(
                label="b3_large_chain",
                title="REGRESSION: deep-chain bounding box",
                caption=(
                    "24-layer deep chain rendered topdown. This is a bbox/clipping regression page: the "
                    "render must be NON-BLANK and the full chain visible.\n"
                    "CHECK: all 24 repeats present, nothing clipped at page edges, edge routing stays "
                    "vertical without drift."
                ),
                panels=[
                    Panel(
                        "large_chain -- 24 layers, direction='topdown'",
                        "large_chain",
                        kwargs={"direction": "topdown"},
                    )
                ],
                covers=["regression:large_bbox"],
            ),
            Page(
                label="b4_placement",
                title="vis_node_placement: 'dot' vs 'rank'",
                caption=(
                    "Two layout engines render the same deep chain. 'dot' is Graphviz; 'rank' is the "
                    "internal pure-Python rank layout used automatically when the estimated dot cost "
                    "exceeds ~20k units (very large graphs with long-range edges). 'auto' (default) picks "
                    "between them -- both concrete engines are shown here.\n"
                    "CHECK: rank output is expected to look boxier/more mechanical than dot -- that is the "
                    "trade-off, not a bug. Node content should be identical between the two."
                ),
                panels=[
                    Panel(
                        "vis_node_placement='dot'",
                        "large_chain",
                        kwargs={"vis_node_placement": "dot"},
                    ),
                    Panel(
                        "vis_node_placement='rank'",
                        "large_chain",
                        kwargs={"vis_node_placement": "rank"},
                    ),
                ],
                covers=["placement:dot", "placement:rank"],
            ),
        ],
    ),
    # =====================================================================
    Section(
        "C",
        "Containers, Module Focus, Depth & Custom Hooks",
        "Structured (dict/tuple) outputs, zooming into one module, limiting module "
        "nesting depth, and every user-callback hook that can reshape the graph.",
        [
            Page(
                label="c1_containers",
                title="show_containers: labels / cluster / collapsed / auto",
                caption=(
                    "Models returning dict or tuple outputs. show_containers controls how container "
                    "structure is drawn: 'labels' annotates leaf tensors with their key/index, 'cluster' "
                    "wraps members in a dashed cluster box, 'collapsed' merges the container into a single "
                    "node, 'auto' picks per container size (see container_max_inline on the next page).\n"
                    "CHECK: dict keys ('a'/'b', 'left'/'right') and tuple indices are correct; dashed "
                    "grouping ties connect producers to mid-graph containers without implying data flow."
                ),
                panels=[
                    Panel(
                        "dict output -- show_containers='labels'",
                        "dict_output",
                        kwargs={"show_containers": "labels"},
                    ),
                    Panel(
                        "mid-graph container -- show_containers='cluster'",
                        "mid_graph_container",
                        kwargs={"show_containers": "cluster"},
                    ),
                    Panel(
                        "tuple output -- show_containers='collapsed'",
                        "tuple_output",
                        kwargs={"show_containers": "collapsed"},
                    ),
                    Panel(
                        "dict output -- show_containers='auto'",
                        "dict_output",
                        kwargs={"show_containers": "auto"},
                    ),
                ],
                covers=[
                    "containers:labels",
                    "containers:cluster",
                    "containers:collapsed",
                    "containers:auto",
                    "node:container",
                    "edge:container_tie",
                ],
                ncols=2,
            ),
            Page(
                label="c2_container_max_inline",
                title="container_max_inline: when 'auto' stops inlining",
                caption=(
                    "With show_containers='auto', containers with at most container_max_inline leaves are "
                    "shown inline/labeled; larger ones are collapsed. LEFT: default threshold (12) keeps the "
                    "4-tuple inline. RIGHT: threshold 2 forces the same 4-tuple to collapse.\n"
                    "CHECK: the collapsed form labels the container with its size instead of showing leaves."
                ),
                panels=[
                    Panel(
                        "container_max_inline=12 (default)",
                        "tuple_output",
                        kwargs={"show_containers": "auto", "container_max_inline": 12},
                    ),
                    Panel(
                        "container_max_inline=2",
                        "tuple_output",
                        kwargs={"show_containers": "auto", "container_max_inline": 2},
                    ),
                ],
                covers=["container_max_inline"],
            ),
            Page(
                label="c3_module_focus",
                title="module=: focus on one submodule",
                caption=(
                    "draw(module='inner_module') renders ONLY the ops inside that submodule, with synthetic "
                    "boundary stubs standing in for the rest of the graph.\n"
                    "CHECK: only inner_module ops appear; entry/exit stubs mark where tensors come from and "
                    "go to; nothing else from the parent graph leaks in."
                ),
                panels=[
                    Panel(
                        "demo_model -- module='inner_module'",
                        "demo_model",
                        kwargs={"module": "inner_module"},
                    )
                ],
                covers=["module_focus", "node:boundary_stub"],
            ),
            Page(
                label="c4_call_depth",
                title="vis_call_depth: limiting module-box nesting",
                caption=(
                    "vis_call_depth caps how deep module boxes nest. LEFT: unlimited (default 1000) shows "
                    "inner_module containing loop_module. RIGHT: depth 1 draws only the outermost module "
                    "boxes; deeper structure is flattened (op nodes remain, boxes disappear).\n"
                    "CHECK: at depth 1 no box appears INSIDE another box."
                ),
                panels=[
                    Panel("vis_call_depth=1000 (default)", "demo_model"),
                    Panel("vis_call_depth=1", "demo_model", kwargs={"vis_call_depth": 1}),
                ],
                covers=["vis_call_depth"],
            ),
            Page(
                label="c5_skip_fn",
                title="skip_fn: hiding layers while chaining edges through",
                caption=(
                    "skip_fn is a keep-predicate: layers for which it returns False are hidden and their "
                    "in/out edges are chained through. RIGHT panel hides every relu.\n"
                    "CHECK: relu nodes are gone on the right; linear ops connect DIRECTLY to each other "
                    "(the data path is preserved, just abbreviated). Input/output nodes can never be skipped."
                ),
                panels=[
                    Panel("default (all ops)", "tiny_mlp"),
                    Panel("skip_fn hides relu ops", "tiny_mlp", kwargs={"skip_fn": _skip_relus}),
                ],
                covers=["skip_fn"],
            ),
            Page(
                label="c6_collapse_fn",
                title="collapse_fn: custom module collapse + remainder labels",
                caption=(
                    "collapse_fn is a module predicate: subtrees for which it returns True render as a single "
                    "collapsed representative box. Here every SmallResBlock in an 8-block stack is collapsed.\n"
                    "CHECK: each collapsed box states its class and an honest 'N layers total' remainder "
                    "count -- the count includes buffer leaves (each block hides conv+bn+relu+add AND the "
                    "batch-norm buffer reads). No block internals leak out."
                ),
                panels=[
                    Panel(
                        "block_stack -- collapse_fn=every SmallResBlock",
                        "block_stack",
                        kwargs={"collapse_fn": _collapse_res_blocks},
                    )
                ],
                covers=["collapse_fn", "node:collapsed_module", "label:remainder"],
            ),
            Page(
                label="c7_node_spec_fn",
                title="node_spec_fn / collapsed_node_spec_fn: per-node customization",
                caption=(
                    "node_spec_fn receives (layer_log, default_spec) and may mutate/return the NodeSpec: here "
                    "every relu is painted gold with an extra label row. collapsed_node_spec_fn does the same "
                    "for collapsed module boxes (RIGHT, applied to the collapsed block stack).\n"
                    "CHECK: only the targeted nodes change; added label rows render inside the node."
                ),
                panels=[
                    Panel(
                        "node_spec_fn highlights relu ops",
                        "tiny_mlp",
                        kwargs={"node_spec_fn": _spec_highlight_relu},
                    ),
                    Panel(
                        "collapsed_node_spec_fn tags collapsed boxes",
                        "block_stack",
                        kwargs={
                            "collapse_fn": _collapse_res_blocks,
                            "collapsed_node_spec_fn": _collapsed_spec_tag,
                        },
                    ),
                ],
                covers=["node_spec_fn", "collapsed_node_spec_fn"],
            ),
            Page(
                label="c8_overrides",
                title="Raw Graphviz override dicts: graph / edge / module",
                caption=(
                    "Power-user escape hatches passing attributes straight to Graphviz: "
                    "vis_graph_overrides (whole-graph attrs; here a tinted background), vis_edge_overrides "
                    "(every edge; here thick red), vis_module_overrides (module boxes; here dashed blue). "
                    "vis_grad_edge_overrides styles gradient edges through the same machinery.\n"
                    "CHECK: overrides apply globally and verbatim -- these are raw attrs, no validation."
                ),
                panels=[
                    Panel(
                        "vis_graph_overrides={'bgcolor': '#FFFBE6'}",
                        "tiny_mlp",
                        kwargs={"vis_graph_overrides": {"bgcolor": "#FFFBE6"}},
                    ),
                    Panel(
                        "vis_edge_overrides={'color': 'red', 'penwidth': '2'}",
                        "tiny_mlp",
                        kwargs={"vis_edge_overrides": {"color": "red", "penwidth": "2"}},
                    ),
                    Panel(
                        "vis_module_overrides={'color': 'blue', 'style': 'dashed'}",
                        "tiny_mlp",
                        kwargs={"vis_module_overrides": {"color": "blue", "style": "dashed"}},
                    ),
                ],
                covers=["overrides:graph", "overrides:edge", "overrides:module"],
            ),
        ],
    ),
    # =====================================================================
    Section(
        "D",
        "Loop Rolling & Recurrence",
        "TorchLens's loop-rolling machinery: vis_mode='rolled' merges repeated "
        "applications of the SAME parameters into one set of nodes with '(xN)' "
        "multipliers and recurrence back-edges. Unrolled mode shows every pass. "
        "This section walks the machinery from a single reused op to multi-state "
        "recurrent cells, varying pass counts, and loops containing branches.",
        [
            Page(
                label="d1_rolled_concept",
                title="The concept: unrolled vs rolled on a reused op",
                caption=(
                    "One ReLU module called 4 times in a loop. LEFT (unrolled): four separate op nodes, one "
                    "per pass, labels carrying the pass index. RIGHT (rolled): ONE node labeled '(x4)' with a "
                    "self-loop edge representing the iteration.\n"
                    "GRAMMAR: '(xN)' always means TRUE recurrence -- the same parameters/op applied N times. "
                    "CHECK: the rolled self-loop is visible and the multiplier matches the unrolled count."
                ),
                panels=[
                    Panel(
                        "vis_mode='unrolled' -- 4 passes",
                        "reused_relu_loop",
                        kwargs={"vis_mode": "unrolled"},
                    ),
                    Panel(
                        "vis_mode='rolled' -- one node, (x4)",
                        "reused_relu_loop",
                        kwargs={"vis_mode": "rolled"},
                    ),
                ],
                covers=["vis_mode:rolled", "label:xN", "edge:self_loop"],
            ),
            Page(
                label="d2_rnn_cell",
                title="RNNCell over a sequence: hidden-state back-edge",
                caption=(
                    "nn.RNNCell applied to 4 timesteps in a Python loop; the hidden state h feeds back into "
                    "the next call. LEFT: unrolled -- the 4 cell applications stack up, each consuming the "
                    "previous h. RIGHT: rolled -- one cell body with a BACK-EDGE closing the recurrence "
                    "loop.\n"
                    "CHECK: the back-edge is distinguishable from forward data edges, its midpoint/routing "
                    "does not collide with node labels, and the '(x4)' multiplier is present."
                ),
                panels=[
                    Panel(
                        "unrolled -- 4 timesteps explicit",
                        "rnn_cell_seq",
                        kwargs={"vis_mode": "unrolled"},
                    ),
                    Panel(
                        "rolled -- recurrence back-edge",
                        "rnn_cell_seq",
                        kwargs={"vis_mode": "rolled"},
                    ),
                ],
                covers=["edge:back_edge"],
            ),
            Page(
                label="d3_lstm_cell",
                title="LSTMCell: TWO recurrent states, two back-edges",
                caption=(
                    "nn.LSTMCell carries both h and c between timesteps, so rolled mode must close TWO "
                    "recurrence loops.\n"
                    "CHECK: both back-edges present and separately routed; tuple unpacking of (h, c) does "
                    "not produce stray nodes; unrolled and rolled agree on op content."
                ),
                panels=[
                    Panel("unrolled", "lstm_cell_seq", kwargs={"vis_mode": "unrolled"}),
                    Panel(
                        "rolled -- h and c back-edges",
                        "lstm_cell_seq",
                        kwargs={"vis_mode": "rolled"},
                    ),
                ],
                covers=["edge:back_edge", "label:xN"],
            ),
            Page(
                label="d4_gru_and_fused",
                title="GRUCell loop vs fused nn.LSTM: what rolling can and cannot see",
                caption=(
                    "LEFT: nn.GRUCell in a Python loop rolls like the other cells. RIGHT: fused nn.LSTM -- "
                    "the entire sequence is ONE fused kernel call, so the graph shows a single lstm op and "
                    "there is NOTHING to roll. This is the documented limitation: fused kernels do not "
                    "expose per-timestep internals.\n"
                    "CHECK: left rolls to a compact loop; right is one op node with weight/bias params "
                    "attached -- if you expected per-step structure from nn.LSTM, this page is why you "
                    "don't get it."
                ),
                panels=[
                    Panel("gru_cell_seq -- rolled", "gru_cell_seq", kwargs={"vis_mode": "rolled"}),
                    Panel("fused nn.LSTM -- one kernel call", "fused_lstm"),
                ],
                covers=["label:xN"],
            ),
            Page(
                label="d5_pass_counts",
                title="Pass-count sweep: (x2) / (x5) / (x8)",
                caption=(
                    "The SAME weight-tied refinement block run for 2, 5, and 8 iterations, all rolled. The "
                    "graph structure is identical; only the '(xN)' multiplier changes.\n"
                    "CHECK: multipliers read exactly (x2), (x5), (x8); node content is otherwise identical "
                    "across the three panels."
                ),
                panels=[
                    Panel("2 iterations", "weight_tied_loop_2", kwargs={"vis_mode": "rolled"}),
                    Panel("5 iterations", "weight_tied_loop_5", kwargs={"vis_mode": "rolled"}),
                    Panel("8 iterations", "weight_tied_loop_8", kwargs={"vis_mode": "rolled"}),
                ],
                covers=["label:xN"],
            ),
            Page(
                label="d6_branching_loop",
                title="Loop with internal branching",
                caption=(
                    "A weight-tied loop whose body takes a tensor-driven if/else each iteration -- different "
                    "iterations can take different arms. Unrolled shows the per-pass truth (which arm each "
                    "iteration actually took); rolled must reconcile the arms into one body.\n"
                    "CHECK: unrolled arms match per-iteration reality; rolled output remains readable and "
                    "does not silently drop an arm that was taken in some pass."
                ),
                panels=[
                    Panel(
                        "unrolled -- per-pass arms",
                        "branching_loop",
                        kwargs={"vis_mode": "unrolled"},
                    ),
                    Panel("rolled", "branching_loop", kwargs={"vis_mode": "rolled"}),
                ],
                covers=["control_flow:branch", "label:xN"],
            ),
            Page(
                label="d7_back_edge_midpoint",
                title="REGRESSION: back-edge midpoint placement",
                caption=(
                    "Single RNN-style cell with hidden-state recurrence, rolled. Historical regression view: "
                    "back-edge label/midpoint placement used to collide with nodes.\n"
                    "CHECK: back-edge arcs cleanly, its midpoint label (if any) sits clear of node boxes, "
                    "arrow direction closes the loop from output back to input side."
                ),
                panels=[
                    Panel("rnn_cell_loop -- rolled", "rnn_cell_loop", kwargs={"vis_mode": "rolled"})
                ],
                covers=["edge:back_edge"],
            ),
        ],
    ),
    # =====================================================================
    Section(
        "E",
        "Collapse & Run Folding (v2 machinery)",
        "The smart-collapse engine: string modes none/auto/max, the continuous "
        "float slider with its monotone schedule, run folding ('+N more' "
        "ellipsis), segment boxes, remainder labels, known artifacts, and the "
        "collapse_plan/collapse_schedule diagnostics.",
        [
            Page(
                label="e1_collapse_modes",
                title="collapse: 'none' vs 'auto' vs 'max' on ResNet-18",
                caption=(
                    "The same ResNet-18 trace at the three string collapse levels. 'none' is the full graph "
                    "-- deliberately unreadable at page scale (that unreadability is the problem collapse "
                    "solves). 'auto' targets a readable overview: stages collapse into representative boxes. "
                    "'max' condenses further.\n"
                    "CHECK: 'auto' is scannable in one glance (stem, 4 stages, head); collapsed boxes carry "
                    "honest layer counts; 'max' stays honest about what it hides rather than pretending "
                    "structure is smaller than it is."
                ),
                panels=[
                    Panel(
                        "collapse='none' (full graph, expect unreadable)",
                        "resnet18",
                        kwargs={"collapse": "none"},
                    ),
                    Panel("collapse='auto'", "resnet18", kwargs={"collapse": "auto"}),
                    Panel("collapse='max'", "resnet18", kwargs={"collapse": "max"}),
                ],
                covers=[
                    "collapse:none",
                    "collapse:auto",
                    "collapse:max",
                    "scale:real_model",
                    "node:collapsed_module",
                ],
            ),
            Page(
                label="e2_float_filmstrip",
                title="Float collapse slider: t = 0.0 to 1.0 filmstrip",
                caption=(
                    "collapse accepts a float t in [0, 1] selecting a deterministic MONOTONE schedule: "
                    "t=0.0 is byte-identical to 'none', t=1.0 to 'max', and increasing t never uncollapses "
                    "anything (collapsed sets are nested). Subtitles show the visible node count reported by "
                    "Trace.collapse_plan(mode=t).\n"
                    "CHECK: visible counts never increase left-to-right; each frame looks like a strictly "
                    "further-condensed version of the previous one, never a re-arrangement."
                ),
                panels=[
                    Panel(
                        "t=0.0",
                        "resnet18",
                        kwargs={"collapse": 0.0},
                        subtitle_fn=_plan_subtitle(0.0),
                    ),
                    Panel(
                        "t=0.25",
                        "resnet18",
                        kwargs={"collapse": 0.25},
                        subtitle_fn=_plan_subtitle(0.25),
                    ),
                    Panel(
                        "t=0.5",
                        "resnet18",
                        kwargs={"collapse": 0.5},
                        subtitle_fn=_plan_subtitle(0.5),
                    ),
                    Panel(
                        "t=0.75",
                        "resnet18",
                        kwargs={"collapse": 0.75},
                        subtitle_fn=_plan_subtitle(0.75),
                    ),
                    Panel(
                        "t=1.0",
                        "resnet18",
                        kwargs={"collapse": 1.0},
                        subtitle_fn=_plan_subtitle(1.0),
                    ),
                ],
                covers=["collapse:float"],
                ncols=5,
            ),
            Page(
                label="e3_fold_runs",
                title="fold_runs: None / True / False on MobileNetV2",
                caption=(
                    "Run folding elides REPEATED runs of same-class blocks into a representative plus an "
                    "ellipsis node. fold_runs=None is the default policy (folding active under "
                    "collapse='auto'/'max'); True folds every legal run; False disables folding entirely.\n"
                    "CHECK: folded panels show one representative block plus a '+N more ...' ellipsis; the "
                    "False panel shows every block explicitly; block counts add up between the versions."
                ),
                panels=[
                    Panel(
                        "collapse='auto', fold_runs=None (default policy)",
                        "mobilenet_v2",
                        kwargs={"collapse": "auto", "fold_runs": None},
                    ),
                    Panel(
                        "collapse='auto', fold_runs=True",
                        "mobilenet_v2",
                        kwargs={"collapse": "auto", "fold_runs": True},
                    ),
                    Panel(
                        "collapse='auto', fold_runs=False",
                        "mobilenet_v2",
                        kwargs={"collapse": "auto", "fold_runs": False},
                    ),
                ],
                covers=["fold_runs:none", "fold_runs:true", "fold_runs:false", "scale:real_model"],
            ),
            Page(
                label="e4_standalone_fold",
                title="Standalone folding: fold_runs=True with collapse='none'",
                caption=(
                    "fold_runs=True works WITHOUT collapse: the full graph is kept except that boring "
                    "repeated stacks fold away ('full graph minus boring stacks' mode). LEFT: plain "
                    "collapse='none'. RIGHT: same graph with fold_runs=True.\n"
                    "CHECK: on the right, the run of identical residual blocks folds to representative + "
                    "ellipsis while everything else (stem, head) stays at full detail."
                ),
                panels=[
                    Panel(
                        "collapse='none', fold_runs=None",
                        "block_stack",
                        kwargs={"collapse": "none"},
                    ),
                    Panel(
                        "collapse='none', fold_runs=True",
                        "block_stack",
                        kwargs={"collapse": "none", "fold_runs": True},
                    ),
                ],
                covers=["fold_runs:standalone"],
            ),
            Page(
                label="e5_ellipsis_grammar",
                title="The '+N more' ellipsis, up close",
                caption=(
                    "An 8-block stack of SAME-CLASS but DIFFERENT-PARAMETER residual blocks under "
                    "collapse='auto'. The run folds to ONE representative box (showing the stats of a single "
                    "instance) plus an ellipsis node carrying the multiplicity.\n"
                    "GRAMMAR (memorize this): '+N more Class' = N distinct same-type instances with their "
                    "own parameters (an ellipsis of siblings). '(xN)' = TRUE recurrence, the same parameters "
                    "applied N times (Section D). These are different claims and must never be conflated.\n"
                    "CHECK: representative box shows single-instance stats; the ellipsis states the correct "
                    "remaining count and class name."
                ),
                panels=[
                    Panel(
                        "block_stack -- collapse='auto'", "block_stack", kwargs={"collapse": "auto"}
                    ),
                ],
                covers=["node:ellipsis", "label:plusN"],
            ),
            Page(
                label="e6_segments",
                title="Segment boxes: max-mode condensed ranges",
                caption=(
                    "collapse='max' on ResNet-50 may condense legal intervals of consecutive siblings into "
                    "SEGMENT boxes -- synthetic dashed ranges labeled with the address range and op/param "
                    "counts. A segment asserts ONLY 'these consecutive siblings live here'; it never claims "
                    "to be a real module and never carries a single class name for mixed content.\n"
                    "CHECK: segment boxes are dashed (adjacency-only), labels give an honest range + count, "
                    "and no segment is styled like a normal module box."
                ),
                panels=[
                    Panel("resnet50 -- collapse='max'", "resnet50", kwargs={"collapse": "max"})
                ],
                covers=["node:segment", "label:segment_range", "scale:real_model"],
            ),
            Page(
                label="e7_remainder_labels",
                title="Remainder labels: 'N layers total' includes buffer leaves",
                caption=(
                    "block_stack at collapse='max'. Collapsed boxes must account for EVERYTHING they hide: "
                    "the 'N layers total' remainder counts ops AND buffer leaves (each block's batch-norm "
                    "reads running stats).\n"
                    "CHECK: counts are consistent with the uncollapsed block (conv, bn, relu, add, plus "
                    "buffer reads); no box under-reports what it swallowed."
                ),
                panels=[
                    Panel(
                        "block_stack -- collapse='max'", "block_stack", kwargs={"collapse": "max"}
                    )
                ],
                covers=["label:remainder"],
            ),
            Page(
                label="e8_interleaved_artifact",
                title="KNOWN ARTIFACT: interleaved run-folds can look like a cycle",
                caption=(
                    "An A/B/A/B/A/B alternating stack with fold_runs=True. Folding the A-run and the B-run "
                    "separately leaves edges that appear to run A -> B -> A: an APPARENT CYCLE. The "
                    "underlying graph is acyclic -- this is a rendering artifact of folding two interleaved "
                    "runs, currently accepted and documented (known nit N2).\n"
                    "CHECK: recognize this shape so you don't misread it as real recurrence. Compare with "
                    "the unfolded panel: the true structure is a straight alternating chain. Note the "
                    "distinction from Section D: real recurrence would say '(xN)'."
                ),
                panels=[
                    Panel("fold_runs=None -- true structure", "interleaved_stack"),
                    Panel(
                        "fold_runs=True -- apparent cycle artifact",
                        "interleaved_stack",
                        kwargs={"fold_runs": True},
                    ),
                ],
                covers=["artifact:apparent_cycle"],
            ),
            Page(
                label="e9_collapse_diagnostics",
                title="Diagnostics: Trace.collapse_plan() and Trace.collapse_schedule()",
                caption="",
                text_fn=_collapse_diag_text,
                covers=["diag:collapse_plan", "diag:collapse_schedule"],
            ),
        ],
    ),
    # =====================================================================
    Section(
        "F",
        "Node Content: Modes, Overlays, Labels, Code Panel, Typography",
        "Everything that changes what is written INSIDE nodes or layered on top "
        "of them, plus source-code panels and typography knobs.",
        [
            Page(
                label="f1_node_modes",
                title="node_mode: 'default' vs 'profiling'",
                caption=(
                    "node_mode picks the label preset. 'profiling' adds per-op timing and activation-memory "
                    "rows to every node.\n"
                    "CHECK: profiling rows are present on every op, formatted compactly (duration units, "
                    "human-readable byte sizes)."
                ),
                panels=[
                    Panel("node_mode='default'", "tiny_mlp"),
                    Panel("node_mode='profiling'", "tiny_mlp", kwargs={"node_mode": "profiling"}),
                ],
                covers=["node_mode:default", "node_mode:profiling"],
            ),
            Page(
                label="f2_domain_modes",
                title="node_mode: 'vision' and 'attention'",
                caption=(
                    "Domain presets: 'vision' emphasizes conv-relevant fields (kernel/channels/spatial "
                    "shapes) -- shown on a conv net; 'attention' emphasizes attention-relevant fields -- "
                    "shown on a transformer encoder.\n"
                    "CHECK: the extra rows make sense for the domain and do not bloat unrelated ops."
                ),
                panels=[
                    Panel(
                        "mini_inception -- node_mode='vision'",
                        "mini_inception",
                        kwargs={"node_mode": "vision"},
                    ),
                    Panel(
                        "mini_transformer -- node_mode='attention'",
                        "mini_transformer",
                        kwargs={"node_mode": "attention"},
                    ),
                ],
                covers=["node_mode:vision", "node_mode:attention", "topology:transformer"],
            ),
            Page(
                label="f3_overlays",
                title="node_overlay: builtin metrics and custom scores",
                caption=(
                    "node_overlay tints node borders by a per-node score. Builtins read trace metadata "
                    "('flops', 'time', 'magnitude', ...); a {layer_label: score} mapping supplies arbitrary "
                    "external scores (here: op position in the graph).\n"
                    "CHECK: border intensity varies across nodes and tracks the metric (heaviest linear "
                    "should stand out under 'flops'); custom mapping ramps monotonically from input to "
                    "output."
                ),
                panels=[
                    Panel("node_overlay='flops'", "tiny_mlp", kwargs={"node_overlay": "flops"}),
                    Panel("node_overlay='time'", "tiny_mlp", kwargs={"node_overlay": "time"}),
                    Panel(
                        "node_overlay='magnitude'", "tiny_mlp", kwargs={"node_overlay": "magnitude"}
                    ),
                    Panel(
                        "custom mapping {label: position}",
                        "tiny_mlp",
                        kwargs_fn=_custom_overlay_kwargs,
                    ),
                ],
                covers=["overlay:flops", "overlay:time", "overlay:magnitude", "overlay:custom"],
                ncols=2,
            ),
            Page(
                label="f4_nan_overlay",
                title="node_overlay='nan': finding the first non-finite op",
                caption=(
                    "A model that goes NaN in the middle (sqrt of a negative shift). The 'nan' overlay "
                    "highlights ops whose outputs contain non-finite values -- the debugging money shot: "
                    "the clean prefix stays unmarked, the sqrt and everything downstream lights up.\n"
                    "CHECK: the boundary between clean and NaN ops is exactly at the sqrt."
                ),
                panels=[
                    Panel(
                        "nan_midway -- node_overlay='nan'",
                        "nan_midway",
                        kwargs={"node_overlay": "nan"},
                    )
                ],
                covers=["overlay:nan"],
            ),
            Page(
                label="f5_label_fields",
                title="node_label_fields: choosing the label rows",
                caption=(
                    "node_label_fields replaces the default label rows with an explicit list. Supported "
                    "fields: label/name, type/op, shape, memory/bytes, module, params, pass, flops, time.\n"
                    "CHECK: rows appear in the requested order and nothing else."
                ),
                panels=[
                    Panel(
                        "node_label_fields=['name', 'shape']",
                        "tiny_mlp",
                        kwargs={"node_label_fields": ["name", "shape"]},
                    ),
                    Panel(
                        "node_label_fields=['type', 'params', 'time']",
                        "tiny_mlp",
                        kwargs={"node_label_fields": ["type", "params", "time"]},
                    ),
                ],
                covers=["node_label_fields"],
            ),
            Page(
                label="f6_code_panel",
                title="code_panel: source code alongside the graph",
                caption=(
                    "code_panel renders the model source captured at trace time in a side panel. True shows "
                    "the forward method; 'init+forward' includes the constructor. ('forward' and 'class' "
                    "variants use the same machinery with different excerpts.)\n"
                    "CHECK: code is readable, aligned with the graph, and matches the actual model source."
                ),
                panels=[
                    Panel("code_panel=True", "demo_model", kwargs={"code_panel": True}),
                    Panel(
                        "code_panel='init+forward'",
                        "demo_model",
                        kwargs={"code_panel": "init+forward"},
                    ),
                ],
                covers=["code_panel:true", "code_panel:init_forward"],
            ),
            Page(
                label="f7_typography",
                title="Typography: font_size, dpi, for_paper",
                caption=(
                    "font_size scales label text; dpi scales raster density (crisper zoom, larger file); "
                    "for_paper=True switches to publication styling (paper theme defaults, tuned linework).\n"
                    "CHECK: font_size=16 is visibly larger; dpi=200 panel is denser/crisper at equal layout; "
                    "for_paper looks print-ready (neutral background, Helvetica)."
                ),
                panels=[
                    Panel("font_size=16", "tiny_mlp", kwargs={"font_size": 16}),
                    Panel("dpi=200", "tiny_mlp", kwargs={"dpi": 200}),
                    Panel("for_paper=True", "tiny_mlp", kwargs={"for_paper": True}),
                ],
                covers=["font_size", "dpi", "for_paper"],
            ),
            Page(
                label="f8_raw_io",
                title="Raw input/output rendering on boundary nodes",
                caption=(
                    "Tracing with save_raw_input=True / save_raw_output=True embeds thumbnails of image-like "
                    "tensors directly in the input/output boundary nodes (batch items up to the batch_render "
                    "limit).\n"
                    "CHECK: thumbnails render inside the boundary nodes without distorting the layout; batch "
                    "items are individually visible."
                ),
                panels=[
                    Panel(
                        "small_conv -- save_raw_input + save_raw_output",
                        "small_conv",
                        trace_variant="raw_io",
                        trace_builder=_raw_input_trace("small_conv"),
                    )
                ],
                covers=["raw_io_render"],
            ),
            Page(
                label="f9_input_transform",
                title="show_input_transform_summary: preprocessing provenance",
                caption=(
                    "When a trace records an input preprocessor (trace(..., transform=fn)), "
                    "show_input_transform_summary=True annotates the input node with an external label "
                    "stating the preprocessing description and its verification status.\n"
                    "CHECK: the input node carries a 'preprocess' xlabel with a verified/UNVERIFIED status "
                    "line; without a recorded preprocessor this option renders nothing extra."
                ),
                panels=[
                    Panel(
                        "traced with transform=normalize -- summary on",
                        "tiny_mlp",
                        trace_variant="transform",
                        trace_builder=_transform_trace("tiny_mlp"),
                        kwargs={"show_input_transform_summary": True},
                    )
                ],
                covers=["input_transform_summary"],
            ),
        ],
    ),
    # =====================================================================
    Section(
        "G",
        "Themes",
        "The five built-in visual themes applied to the same graph.",
        [
            Page(
                label="g1_themes",
                title="vis_theme: torchlens / paper / dark / colorblind / high_contrast",
                caption=(
                    "The same MLP under every built-in theme. 'torchlens' is the default; 'paper' is "
                    "publication-neutral; 'dark' inverts for slides; 'colorblind' uses an accessible "
                    "palette; 'high_contrast' maximizes edge/border weight.\n"
                    "CHECK: text stays readable in every theme (especially on the dark background); node "
                    "type distinctions survive each palette; edges remain visible."
                ),
                panels=[
                    Panel("torchlens (default)", "tiny_mlp", kwargs={"vis_theme": "torchlens"}),
                    Panel("paper", "tiny_mlp", kwargs={"vis_theme": "paper"}),
                    Panel("dark", "tiny_mlp", kwargs={"vis_theme": "dark"}),
                    Panel("colorblind", "tiny_mlp", kwargs={"vis_theme": "colorblind"}),
                    Panel("high_contrast", "tiny_mlp", kwargs={"vis_theme": "high_contrast"}),
                ],
                covers=[
                    "theme:torchlens",
                    "theme:paper",
                    "theme:dark",
                    "theme:colorblind",
                    "theme:high_contrast",
                ],
                ncols=3,
            ),
        ],
    ),
    # =====================================================================
    Section(
        "H",
        "Backward & Combined Graphs",
        "Rendering the captured autograd graph: draw_backward's grad_fn nodes, "
        "and draw_combined's forward+backward composite with correspondence ties.",
        [
            Page(
                label="h1_backward",
                title="draw_backward: the grad_fn graph",
                caption=(
                    "After log_backward(), draw_backward renders the captured autograd graph: grad_fn nodes "
                    "(AddmmBackward, ReluBackward, ...) connected by gradient-flow edges. Default vis_mode "
                    "is 'rolled'; 'unrolled' shows per-call grad_fn instances.\n"
                    "CHECK: grad_fn names match the forward ops they differentiate; edge direction follows "
                    "gradient flow; no forward nodes leak into this graph."
                ),
                panels=[
                    Panel(
                        "draw_backward (rolled default)",
                        "linear_relu",
                        "draw_backward",
                        trace_variant="backward",
                    ),
                    Panel(
                        "draw_backward vis_mode='unrolled'",
                        "linear_relu",
                        "draw_backward",
                        kwargs={"vis_mode": "unrolled"},
                        trace_variant="backward",
                    ),
                ],
                covers=[
                    "draw_backward:rolled",
                    "draw_backward:unrolled",
                    "node:grad_fn",
                    "edge:backward",
                ],
            ),
            Page(
                label="h2_backward_opts",
                title="draw_backward options: direction and pass selection",
                caption=(
                    "draw_backward accepts its own direction (default 'topdown') and a bwd= one-based "
                    "backward-pass filter (with several recorded backward passes, bwd=2 shows only the "
                    "second).\n"
                    "CHECK: leftright backward flows horizontally; bwd=1 on a single-backward trace equals "
                    "the default render."
                ),
                panels=[
                    Panel(
                        "vis_direction='leftright'",
                        "linear_relu",
                        "draw_backward",
                        kwargs={"vis_direction": "leftright"},
                        trace_variant="backward",
                    ),
                    Panel(
                        "bwd=1 (select first backward pass)",
                        "linear_relu",
                        "draw_backward",
                        kwargs={"bwd": 1},
                        trace_variant="backward",
                    ),
                ],
                covers=["draw_backward:bwd"],
            ),
            Page(
                label="h3_combined",
                title="draw_combined: forward and backward in one graph",
                caption=(
                    "draw_combined renders forward ops and backward grad_fns side by side (default "
                    "leftright), with DASHED correspondence ties linking each forward op to the grad_fn "
                    "that differentiates it.\n"
                    "CHECK: the forward and backward halves are visually separable; dashed ties pair the "
                    "right nodes; gradient edges run opposite to data edges."
                ),
                panels=[
                    Panel(
                        "linear_relu -- draw_combined",
                        "linear_relu",
                        "draw_combined",
                        trace_variant="backward",
                    ),
                    Panel(
                        "scalar_out -- draw_combined",
                        "scalar_out",
                        "draw_combined",
                        trace_variant="backward",
                    ),
                ],
                covers=["draw_combined", "edge:fwd_bwd_tie", "edge:backward"],
            ),
            Page(
                label="h4_combined_cluster",
                title="draw_combined: intervening_cluster placement",
                caption=(
                    "intervening_cluster controls which cluster absorbs nodes that sit between the forward "
                    "and backward halves: 'upstream' (default) attaches them to the forward side, 'own' "
                    "gives them their own cluster ('outside'/'downstream' mirror these with different "
                    "targets).\n"
                    "CHECK: the in-between nodes visibly move between the two panels."
                ),
                panels=[
                    Panel(
                        "intervening_cluster='upstream' (default)",
                        "linear_relu",
                        "draw_combined",
                        kwargs={"intervening_cluster": "upstream"},
                        trace_variant="backward",
                    ),
                    Panel(
                        "intervening_cluster='own'",
                        "linear_relu",
                        "draw_combined",
                        kwargs={"intervening_cluster": "own"},
                        trace_variant="backward",
                    ),
                ],
                covers=["combined:intervening_cluster"],
            ),
        ],
    ),
    # =====================================================================
    Section(
        "I",
        "Control Flow & Interventions",
        "Tensor-driven branching (taken arms only, with arm labels) and how "
        "intervention sites, hooks, and affected cones are marked.",
        [
            Page(
                label="i1_conditionals",
                title="if/else and elif ladders: taken-arm rendering",
                caption=(
                    "TorchLens records the arm that actually EXECUTED. LEFT: simple if/else -- the taken arm "
                    "(relu) appears with an IF arm label; the untaken arm (sigmoid) is absent. RIGHT: an "
                    "elif ladder -- the matching ELIF arm renders with its label.\n"
                    "CHECK: arm labels (IF/ELIF/ELSE) are present and attached to the right ops; nothing "
                    "from untaken arms appears."
                ),
                panels=[
                    Panel("simple_if_else -- taken arm only", "simple_if_else"),
                    Panel("elif_ladder", "elif_ladder"),
                ],
                covers=["control_flow:branch", "label:arm"],
            ),
            Page(
                label="i2_branch_cnn",
                title="Conditional dual-head CNN",
                caption=(
                    "A conv net that routes through one of two linear heads depending on the input mean "
                    "(here: positive input, so up_head).\n"
                    "CHECK: exactly one head present (up_head); the conditional annotation marks the branch "
                    "point; down_head absent."
                ),
                panels=[Panel("tiny_branch_cnn", "tiny_branch_cnn")],
                covers=["control_flow:branch"],
            ),
            Page(
                label="i3_intervention_modes",
                title="vis_intervention_mode: 'node_mark' vs 'as_node'",
                caption=(
                    "A trace captured with an intervention (zero-ablating every relu). 'node_mark' (default) "
                    "marks the intervened op node itself (magenta site styling); 'as_node' inserts explicit "
                    "hook nodes showing where the intervention fires in the data path.\n"
                    "CHECK: LEFT -- relu nodes carry the intervention site color; RIGHT -- separate hook "
                    "nodes appear inline, styled distinctly from data ops."
                ),
                panels=[
                    Panel(
                        "vis_intervention_mode='node_mark' (default)",
                        "tiny_mlp",
                        kwargs={"vis_intervention_mode": "node_mark"},
                        trace_variant="intervened",
                        trace_builder=_intervened_trace("tiny_mlp"),
                    ),
                    Panel(
                        "vis_intervention_mode='as_node'",
                        "tiny_mlp",
                        kwargs={"vis_intervention_mode": "as_node"},
                        trace_variant="intervened",
                        trace_builder=_intervened_trace("tiny_mlp"),
                    ),
                ],
                covers=["intervention:node_mark", "intervention:as_node", "node:intervention"],
            ),
            Page(
                label="i4_cone",
                title="vis_show_cone: the downstream affected region",
                caption=(
                    "The intervention CONE is every op downstream of an intervention site -- the region "
                    "whose values the intervention could have changed. vis_show_cone=True (default) tints "
                    "it; False renders sites only.\n"
                    "CHECK: LEFT -- everything downstream of the first intervened relu is tinted (lighter "
                    "magenta), upstream ops are untouched; RIGHT -- cone styling gone, site marks remain."
                ),
                panels=[
                    Panel(
                        "vis_show_cone=True (default)",
                        "tiny_mlp",
                        kwargs={"vis_show_cone": True},
                        trace_variant="intervened",
                        trace_builder=_intervened_trace("tiny_mlp"),
                    ),
                    Panel(
                        "vis_show_cone=False",
                        "tiny_mlp",
                        kwargs={"vis_show_cone": False},
                        trace_variant="intervened",
                        trace_builder=_intervened_trace("tiny_mlp"),
                    ),
                ],
                covers=["cone:on", "cone:off"],
            ),
        ],
    ),
    # =====================================================================
    Section(
        "J",
        "Real Architectures at Page Scale",
        "Full-page renders of realistic architectures: what a user actually sees "
        "when they point TorchLens at a real model.",
        [
            Page(
                label="j1_resnet_overview",
                title="ResNet-18 at collapse='auto': the intended overview experience",
                caption=(
                    "The flagship 'point it at a real model' render: ResNet-18 with the default-recommended "
                    "collapse='auto'. This page is the product screenshot -- judge it as one.\n"
                    "CHECK: reads top-to-bottom as stem -> 4 stages -> pool -> fc; stage boxes are uniform "
                    "in visual weight (no mixed granularity); labels honest about hidden content."
                ),
                panels=[
                    Panel("resnet18 -- collapse='auto'", "resnet18", kwargs={"collapse": "auto"})
                ],
                covers=["scale:real_model"],
            ),
            Page(
                label="j2_transformer",
                title="Transformer encoder, fully unrolled",
                caption=(
                    "Two TransformerEncoder layers at full detail: the attention pattern (q/k/v projections, "
                    "matmul-softmax-matmul diamond, residual adds, layernorms, feedforward) repeated twice.\n"
                    "CHECK: the attention diamond is recognizable; residual skip edges route around the "
                    "blocks cleanly; the two layers render identically."
                ),
                panels=[Panel("mini_transformer -- unrolled", "mini_transformer")],
                covers=["topology:transformer"],
            ),
            Page(
                label="j3_inception",
                title="Inception-style branching: parallel paths and concat merges",
                caption=(
                    "Two inception blocks: each splits into four parallel branches (1x1, 3x3, 5x5, pool) "
                    "that merge in a concat.\n"
                    "CHECK: branch fan-out and concat fan-in are visually clean; branches keep left-to-right "
                    "execution order (order_siblings); no edge crossings that could be avoided."
                ),
                panels=[Panel("mini_inception -- unrolled", "mini_inception")],
                covers=["topology:parallel"],
            ),
        ],
    ),
    # =====================================================================
    Section(
        "K",
        "Degenerate & Edge Cases",
        "The smallest and strangest graphs TorchLens must render sanely.",
        [
            Page(
                label="k1_degenerate",
                title="Degenerate graphs: single op, no modules, scalar out, no params",
                caption=(
                    "Four boundary-condition models: (1) forward is exactly one op; (2) functional ops only, "
                    "no submodules -- so no module boxes at all; (3) output is a 0-dim scalar; (4) a chain "
                    "with no parameters anywhere.\n"
                    "CHECK: each renders without error or visual junk; scalar output labeled sanely "
                    "(no empty shape garbage); absence of module boxes/params does not break styling."
                ),
                panels=[
                    Panel("single_op -- x + 1", "single_op"),
                    Panel("no_submodules -- functional only", "no_submodules"),
                    Panel("scalar_out -- 0-dim output", "scalar_out"),
                    Panel("paramless_deep -- no params", "paramless_deep"),
                ],
                covers=["degenerate"],
                ncols=2,
            ),
        ],
    ),
]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_panel(page: Page, panel: Panel, idx: int) -> tuple[str, pathlib.Path | None, str]:
    """Render one panel; returns (subtitle, png_path_or_None, error)."""
    subtitle = panel.subtitle
    stem = _PAGES_DIR / f"{page.label}_p{idx}"
    try:
        trace = get_trace(panel.model_key, panel.trace_variant, panel.trace_builder)
        kwargs = dict(panel.kwargs)
        if panel.kwargs_fn is not None:
            kwargs.update(panel.kwargs_fn(trace))
        if panel.subtitle_fn is not None:
            subtitle = panel.subtitle_fn(trace)
        fn = getattr(trace, panel.method)
        fn(vis_outpath=str(stem), vis_fileformat="png", vis_save_only=True, **kwargs)
        png = pathlib.Path(str(stem) + ".png")
        if not png.exists():
            raise FileNotFoundError(f"expected {png}")
        return subtitle, png, ""
    except Exception as exc:
        print(f"    PANEL FAILED [{page.label}#{idx}]: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return subtitle, None, f"{type(exc).__name__}: {exc}"


def _page_plan() -> list[tuple[str, object]]:
    """Return the ordered page plan: ('toc'|'section'|'page', payload)."""
    plan: list[tuple[str, object]] = [("toc", None)]
    for section in SECTIONS:
        plan.append(("section", section))
        for page in section.pages:
            plan.append(("page", (section, page)))
    return plan


def _toc_body(plan: list[tuple[str, object]]) -> str:
    lines = [
        "A scrollable montage of the complete TorchLens visual language.",
        "Every page = one demo: header, caption (what is shown / what to check), graph panels.",
        "",
    ]
    for num, (kind, payload) in enumerate(plan, 1):
        if kind == "section":
            section = payload
            lines.append(f"  p{num:>3}  SECTION {section.letter}: {section.title}")
        elif kind == "page":
            _sec, page = payload
            lines.append(f"  p{num:>3}      {page.title}")
    lines += [
        "",
        "Visual-grammar cheat sheet:",
        "  '(xN)'          true recurrence: the SAME parameters applied N times (rolled mode)",
        "  '+N more Class' ellipsis: N further DISTINCT same-class instances (run folding)",
        "  'N layers total' collapsed-box remainder: everything hidden inside, incl. buffers",
        "  dashed segment  adjacency-only range: consecutive siblings, NOT a real module",
        "  double border   this op has hidden buffer dependencies (peripheries=2)",
    ]
    return "\n".join(lines)


def main() -> None:
    if _PAGES_DIR.exists():
        shutil.rmtree(_PAGES_DIR)
    _PAGES_DIR.mkdir(parents=True)

    plan = _page_plan()
    total = len(plan)
    print("TorchLens Visual Audit Pack Generator (v2)")
    print(f"Output: {_OUT_PDF}")
    print(f"Planned pages: {total}")
    print()

    page_pdfs: list[pathlib.Path] = []
    errors: list[str] = []

    for num, (kind, payload) in enumerate(plan, 1):
        if kind == "toc":
            out = _PAGES_DIR / "000_toc.pdf"
            compose_text_page(
                out,
                "TorchLens Visual Audit Pack",
                f"page {num}/{total}",
                "Table of Contents",
                _toc_body(plan),
                banner=True,
            )
            page_pdfs.append(out)
            print(f"[{num:03d}/{total}] TOC")
            continue

        if kind == "section":
            section = payload
            out = _PAGES_DIR / f"{num:03d}_section_{section.letter}.pdf"
            body = "\n".join(
                [section.blurb, "", "Pages in this section:"]
                + [f"  - {p.title}" for p in section.pages]
            )
            compose_text_page(
                out,
                f"Section {section.letter}",
                f"page {num}/{total}",
                f"{section.letter}. {section.title}",
                body,
                body_fontsize=10.5,
                banner=True,
            )
            page_pdfs.append(out)
            print(f"[{num:03d}/{total}] SECTION {section.letter}: {section.title}")
            continue

        section, page = payload
        print(f"[{num:03d}/{total}] {page.label} ...", flush=True)
        out = _PAGES_DIR / f"{num:03d}_{page.label}.pdf"
        header_left = f"Section {section.letter}: {section.title}"
        header_right = f"page {num}/{total}  |  {page.label}"

        try:
            if page.text_fn is not None:
                compose_text_page(out, header_left, header_right, page.title, page.text_fn())
            else:
                items = [_render_panel(page, panel, i) for i, panel in enumerate(page.panels, 1)]
                n_failed = sum(1 for _s, p, _e in items if p is None)
                if n_failed:
                    errors.append(f"p{num} {page.label}: {n_failed}/{len(items)} panels failed")
                compose_page(
                    out,
                    header_left,
                    header_right,
                    page.title,
                    page.caption,
                    [(s, p, e) for s, p, e in items],
                    ncols=page.ncols,
                )
            page_pdfs.append(out)
        except Exception as exc:
            errors.append(f"p{num} {page.label}: PAGE COMPOSE FAILED: {exc}")
            print(f"    PAGE FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

        gc.collect()

    # --- staple ---
    print()
    print(f"Stapling {len(page_pdfs)} pages -> {_OUT_PDF} ...")
    from pypdf import PdfWriter

    writer = PdfWriter()
    for p in page_pdfs:
        writer.append(str(p))
    with open(_OUT_PDF, "wb") as fh:
        writer.write(fh)
    print(f"Done. {_OUT_PDF} ({_OUT_PDF.stat().st_size // 1024} KB, {len(writer.pages)} pages)")

    # --- coverage matrix ---
    _write_coverage_matrix(plan)

    print()
    if errors:
        print(f"GAPS ({len(errors)}):")
        for e in errors:
            print(f"  * {e}")
    else:
        print("All pages rendered successfully -- no GAPs.")


def _write_coverage_matrix(plan: list[tuple[str, object]]) -> None:
    """Emit coverage_matrix.md derived from page `covers` tags vs AXES."""
    coverage: dict[str, list[int]] = {tag: [] for tag in AXES}
    unknown_tags: list[str] = []
    page_rows: list[str] = []

    for num, (kind, payload) in enumerate(plan, 1):
        if kind != "page":
            continue
        section, page = payload
        for tag in page.covers:
            if tag in coverage:
                coverage[tag].append(num)
            else:
                unknown_tags.append(f"{page.label}: {tag}")
        models = sorted({p.model_key for p in page.panels}) if page.panels else ["(text page)"]
        page_rows.append(
            f"| {num} | {section.letter} | `{page.label}` | {page.title} | "
            f"{', '.join(f'`{m}`' for m in models)} | {len(page.panels) or '-'} |"
        )

    uncovered = [tag for tag, pages in coverage.items() if not pages]

    lines: list[str] = []
    lines.append("# Visual Audit Pack -- Coverage Matrix")
    lines.append("")
    lines.append("AUTO-GENERATED by `generate_visual_pack.py` -- do not hand-edit.")
    lines.append("Coverage axes are enumerated from the renderer/option source")
    lines.append("(`torchlens/_literals.py`, `Trace.draw()` in `_trace_viz.py`,")
    lines.append("`visualization/themes.py`, `visualization/modes.py`, node/edge kinds in")
    lines.append("`visualization/rendering.py`). Each axis must be demonstrated by at least")
    lines.append("one page or carry an explicit N/A rationale; anything else is a defect.")
    lines.append("")
    lines.append("## Axis coverage")
    lines.append("")
    lines.append("| Axis | Meaning | Covered by page(s) |")
    lines.append("|------|---------|--------------------|")
    for tag, desc in AXES.items():
        pages = coverage[tag]
        cell = ", ".join(f"p{n}" for n in pages) if pages else "**UNCOVERED -- DEFECT**"
        lines.append(f"| `{tag}` | {desc} | {cell} |")
    lines.append("")
    lines.append("## Deliberately not paged (with rationale)")
    lines.append("")
    lines.append("| Axis | Rationale |")
    lines.append("|------|-----------|")
    for tag, reason in NA_AXES.items():
        lines.append(f"| `{tag}` | {reason} |")
    lines.append("")
    lines.append("## Page inventory")
    lines.append("")
    lines.append("| Page | Section | Label | Title | Models | Panels |")
    lines.append("|------|---------|-------|-------|--------|--------|")
    lines.extend(page_rows)
    lines.append("")
    if uncovered:
        lines.append("## DEFECTS: uncovered axes")
        lines.append("")
        for tag in uncovered:
            lines.append(f"- `{tag}`")
        lines.append("")
    if unknown_tags:
        lines.append("## WARNINGS: pages declaring unknown axis tags")
        lines.append("")
        for t in unknown_tags:
            lines.append(f"- {t}")
        lines.append("")
    lines.append("## How to extend")
    lines.append("")
    lines.append("See `CLAUDE.md` in this directory: add a `Page` to `SECTIONS` in")
    lines.append("`generate_visual_pack.py` with honest `covers=` tags (add new axes to `AXES`")
    lines.append("when a new visual feature ships), re-run the script, and re-run the")
    lines.append("completeness + fresh-eyes critics.")
    lines.append("")

    _MATRIX_MD.write_text("\n".join(lines))
    print(f"Coverage matrix: {_MATRIX_MD}")
    if uncovered:
        print(f"  WARNING: {len(uncovered)} uncovered axes (listed as defects in matrix)")
    if unknown_tags:
        print(f"  WARNING: {len(unknown_tags)} unknown tags declared by pages")


if __name__ == "__main__":
    main()
