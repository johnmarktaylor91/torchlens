"""Visual audit pack generator for TorchLens.

Run as:
    python notebooks/audit/visual/generate_visual_pack.py

Produces:
    notebooks/audit/visual/visual_audit.pdf  -- the stapled audit pack
    notebooks/audit/visual/_pages/           -- per-page intermediate PDFs (gitignored)
"""

from __future__ import annotations

# This script needs matplotlib.use("Agg") and a sys.path shim before importing the
# zoo / torchlens, so module-level imports legitimately follow code (E402).
# ruff: noqa: E402

import pathlib
import shutil
import sys
import traceback

# ---------------------------------------------------------------------------
# Path setup: let 'from _models import ZOO' work from this subdir
# ---------------------------------------------------------------------------
_AUDIT_DIR = pathlib.Path(__file__).resolve().parent.parent  # notebooks/audit/
sys.path.insert(0, str(_AUDIT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

import torchlens as tl
from _models import ZOO

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
_VISUAL_DIR = pathlib.Path(__file__).resolve().parent
_PAGES_DIR = _VISUAL_DIR / "_pages"
_OUT_PDF = _VISUAL_DIR / "visual_audit.pdf"

# ---------------------------------------------------------------------------
# PAGE LIST
# Each entry:
#   label        - short filesystem-safe name (NN_label)
#   model_key    - key into ZOO
#   draw_method  - 'draw' | 'draw_backward' | 'draw_combined'
#   draw_kwargs  - dict passed to the draw call (vis_outpath injected at runtime)
#   caption      - one-line "what to nit-check"
#   extra_setup  - optional callable(trace) -> trace  for backward setup etc.
# ---------------------------------------------------------------------------


def _setup_backward(trace: tl.Trace) -> tl.Trace:
    """Run log_backward so draw_backward / draw_combined work."""
    out = trace[-1].out
    trace.log_backward(out)
    return trace


def _trace_with_backward(model_key: str) -> tl.Trace:
    """Helper: trace with backward_ready=True, run backward."""
    m, x = ZOO[model_key]()
    trace = tl.trace(m, x, backward_ready=True)
    out = trace[-1].out
    trace.log_backward(out)
    return trace


# Format: (label, model_key, draw_method, draw_kwargs, caption, extra_setup_fn_or_None)
PAGE_LIST: list[tuple[str, str, str, dict, str, object]] = [
    # 01 -- default unrolled, tiny_mlp (baseline sanity)
    (
        "01_default_unrolled",
        "tiny_mlp",
        "draw",
        {"vis_mode": "unrolled"},
        "Baseline: default unrolled layout, node shapes, edge arrows, label fonts",
        None,
    ),
    # 02 -- large-graph bbox regression (non-blank PDF is the check)
    (
        "02_large_graph_bbox",
        "large_chain",
        "draw",
        {"vis_mode": "unrolled", "direction": "topdown"},
        "REGRESSION: PDF must be NON-BLANK; check bbox/clipping on deep 24-layer chain",
        None,
    ),
    # 03 -- edge multiplicity: add_twice (both edges from same input to add)
    (
        "03_edge_multiplicity_add",
        "add_twice",
        "draw",
        {"vis_mode": "unrolled"},
        "Edge multiplicity: two parallel edges from input to add node -- both visible?",
        None,
    ),
    # 04 -- edge multiplicity: cat_twice (sequence-arg multiplicity)
    (
        "04_edge_multiplicity_cat",
        "cat_twice",
        "draw",
        {"vis_mode": "unrolled"},
        "Edge multiplicity: two sequence-arg edges to cat -- both present, correct labels?",
        None,
    ),
    # 05 -- rolled vis_mode + self-loop label merge (reused_relu_loop)
    (
        "05_rolled_self_loop",
        "reused_relu_loop",
        "draw",
        {"vis_mode": "rolled"},
        "Rolled mode: single ReLU called 4x -- self-loop present, label merge correct?",
        None,
    ),
    # 06 -- back-edge midpoints (rnn_cell_loop, rolled)
    (
        "06_back_edge_midpoints",
        "rnn_cell_loop",
        "draw",
        {"vis_mode": "rolled"},
        "Rolled back-edges: hidden-state recurrence -- back-edge midpoint placement correct?",
        None,
    ),
    # 07 -- sibling ordering (parallel_fanout, order_siblings=True)
    (
        "07_sibling_ordering",
        "parallel_fanout",
        "draw",
        {"vis_mode": "unrolled", "order_siblings": True},
        "Sibling ordering: 4 parallel proj branches -- left-to-right order preserved?",
        None,
    ),
    # 08 -- container mode: labels (dict_output)
    (
        "08_container_labels",
        "dict_output",
        "draw",
        {"vis_mode": "unrolled", "show_containers": "labels"},
        "Container show_containers='labels': dict output leaf labels shown correctly?",
        None,
    ),
    # 09 -- container mode: cluster (mid_graph_container)
    (
        "09_container_cluster",
        "mid_graph_container",
        "draw",
        {"vis_mode": "unrolled", "show_containers": "cluster"},
        "Container show_containers='cluster': mid-graph dict submodule cluster box OK?",
        None,
    ),
    # 10 -- container mode: collapsed (tuple_output)
    (
        "10_container_collapsed",
        "tuple_output",
        "draw",
        {"vis_mode": "unrolled", "show_containers": "collapsed"},
        "Container show_containers='collapsed': 4-element tuple collapsed to single node?",
        None,
    ),
    # 11 -- atomic-module rectangles (demo_model default) + buffer edges always
    (
        "11_buffers_always",
        "batch_norm",
        "draw",
        {"vis_mode": "unrolled", "show_buffer_layers": "always"},
        "Buffer edges show_buffer_layers='always': running_mean/var nodes present and styled?",
        None,
    ),
    # 12 -- buffer edges: demo_model (register_buffer + show_buffer_layers)
    (
        "12_demo_buffer_module",
        "demo_model",
        "draw",
        {"vis_mode": "unrolled", "show_buffer_layers": "always"},
        "DemoModel buffers: example_buffer edge visible? Module rectangles correct?",
        None,
    ),
    # 13 -- conditional arm labels: simple_if_else
    (
        "13_conditional_arm",
        "simple_if_else",
        "draw",
        {"vis_mode": "unrolled"},
        "Conditional: if/else arm -- taken-arm op present, not-taken ops absent?",
        None,
    ),
    # 14 -- conditional arm: tiny_branch_cnn (conv + dual head conditional)
    (
        "14_conditional_cnn_branch",
        "tiny_branch_cnn",
        "draw",
        {"vis_mode": "unrolled"},
        "Branch CNN: conv + conditional dual-head -- correct head shown, other absent?",
        None,
    ),
    # 15 -- backward graph (linear_relu)
    (
        "15_backward_graph",
        "linear_relu",
        "draw_backward",
        {"vis_mode": "unrolled"},
        "Backward graph: GradFn nodes, backward edges, direction OK?",
        "needs_backward",
    ),
    # 16 -- combined fwd+bwd graph
    (
        "16_combined_fwdbwd",
        "linear_relu",
        "draw_combined",
        {"vis_mode": "unrolled"},
        "Combined fwd+bwd: both halves present, cluster boundary clear?",
        "needs_backward",
    ),
    # 17 -- code_panel composition (demo_model)
    (
        "17_code_panel",
        "demo_model",
        "draw",
        {"vis_mode": "unrolled", "code_panel": True},
        "code_panel=True: source code panel alongside graph -- alignment, readability?",
        None,
    ),
    # 18 -- node_mode profiling (shows timing/memory in nodes)
    (
        "18_node_mode_profiling",
        "tiny_mlp",
        "draw",
        {"vis_mode": "unrolled", "node_mode": "profiling"},
        "node_mode='profiling': timing/memory fields in node labels -- present, formatted?",
        None,
    ),
    # 19 -- node_mode attention
    (
        "19_node_mode_attention",
        "tiny_mlp",
        "draw",
        {"vis_mode": "unrolled", "node_mode": "attention"},
        "node_mode='attention': attention-specific label fields -- no crash, reasonable output?",
        None,
    ),
    # 20 -- node_overlay='flops' (overlay values on nodes)
    (
        "20_node_overlay_flops",
        "tiny_mlp",
        "draw",
        {"vis_mode": "unrolled", "node_overlay": "flops"},
        "node_overlay='flops': FLOPs values overlaid on nodes -- present, sized correctly?",
        None,
    ),
    # 21 -- vis_theme='dark' + show_legend=True
    (
        "21_theme_dark_legend",
        "tiny_mlp",
        "draw",
        {"vis_mode": "unrolled", "vis_theme": "dark", "show_legend": True},
        "Theme 'dark' + legend: dark bg colors, legend box present, text readable?",
        None,
    ),
    # 22 -- vis_theme='colorblind' (accessibility theme)
    (
        "22_theme_colorblind",
        "tiny_mlp",
        "draw",
        {"vis_mode": "unrolled", "vis_theme": "colorblind"},
        "Theme 'colorblind': accessible palette applied correctly across node types?",
        None,
    ),
    # 23 -- module focus (demo_model, module='inner_module')
    (
        "23_module_focus",
        "demo_model",
        "draw",
        {"vis_mode": "unrolled", "module": "inner_module"},
        "Module focus: only inner_module ops shown? Input/output stubs present?",
        None,
    ),
    # 24 -- direction leftright (tiny_mlp)
    (
        "24_direction_leftright",
        "tiny_mlp",
        "draw",
        {"vis_mode": "unrolled", "direction": "leftright"},
        "direction='leftright': graph flows left-to-right, labels legible?",
        None,
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_caption_pdf(
    page_num: int,
    model_key: str,
    draw_method: str,
    draw_kwargs: dict,
    caption: str,
    out_path: pathlib.Path,
    failed: bool = False,
    error_msg: str = "",
) -> None:
    """Render a text-only caption card to a PDF."""
    fig, ax = plt.subplots(figsize=(8.5, 3))
    ax.axis("off")

    status = "FAILED" if failed else ""
    color = "red" if failed else "black"

    lines = [
        f"Page {page_num}  |  model: {model_key}  |  method: {draw_method}  {status}",
        f"draw kwargs: {draw_kwargs}",
        "",
        f"What to nit-check: {caption}",
    ]
    if failed and error_msg:
        lines += ["", f"ERROR: {error_msg[:200]}"]

    text = "\n".join(lines)
    ax.text(
        0.02,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        color=color,
        wrap=True,
    )

    # Thin border for visual separation
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("#cccccc")

    fig.tight_layout(pad=0.3)
    fig.savefig(str(out_path), format="pdf", bbox_inches="tight")
    plt.close(fig)


def _render_graph_page(
    trace: tl.Trace, draw_method: str, draw_kwargs: dict, out_stem: pathlib.Path
) -> pathlib.Path:
    """Call trace.<draw_method>(...) and return the path of the produced PDF."""
    fn = getattr(trace, draw_method)
    fn(
        vis_outpath=str(out_stem),
        vis_fileformat="pdf",
        vis_save_only=True,
        **draw_kwargs,
    )
    # draw() appends .pdf
    pdf_path = pathlib.Path(str(out_stem) + ".pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected {pdf_path} but it was not produced")
    return pdf_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Clean and recreate _pages/
    if _PAGES_DIR.exists():
        shutil.rmtree(_PAGES_DIR)
    _PAGES_DIR.mkdir(parents=True)

    print("TorchLens Visual Audit Pack Generator")
    print(f"Output: {_OUT_PDF}")
    print(f"Pages:  {_PAGES_DIR}")
    print(f"Total planned pages: {len(PAGE_LIST)}")
    print()

    page_pdfs: list[pathlib.Path] = []
    errors: list[str] = []

    for idx, (label, model_key, draw_method, draw_kwargs, caption, extra_setup) in enumerate(
        PAGE_LIST, 1
    ):
        print(f"[{idx:02d}/{len(PAGE_LIST)}] {label} ...", end=" ", flush=True)

        caption_path = _PAGES_DIR / f"{label}_caption.pdf"
        graph_stem = _PAGES_DIR / f"{label}_graph"

        # --- Try to render the graph ---
        graph_pdf: pathlib.Path | None = None
        error_msg = ""
        try:
            if extra_setup == "needs_backward":
                # Special: trace with backward
                m, x = ZOO[model_key]()
                trace = tl.trace(m, x, backward_ready=True)
                out_tensor = trace[-1].out
                trace.log_backward(out_tensor)
            else:
                m, x = ZOO[model_key]()
                trace = tl.trace(m, x)
                if callable(extra_setup):
                    trace = extra_setup(trace)

            graph_pdf = _render_graph_page(trace, draw_method, draw_kwargs, graph_stem)
            size_kb = graph_pdf.stat().st_size // 1024
            print(f"OK ({size_kb} KB)")

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
            print(f"FAILED -- {error_msg}")
            print(f"  Traceback:\n{tb}")
            errors.append(f"Page {idx:02d} ({label}): {error_msg}")

        # --- Always emit a caption card ---
        failed = graph_pdf is None
        try:
            _make_caption_pdf(
                idx,
                model_key,
                draw_method,
                draw_kwargs,
                caption,
                caption_path,
                failed=failed,
                error_msg=error_msg,
            )
        except Exception as cap_exc:
            print(f"  WARNING: caption card also failed: {cap_exc}")

        # Collect pages: caption first, then graph (if available)
        if caption_path.exists():
            page_pdfs.append(caption_path)
        if graph_pdf is not None and graph_pdf.exists():
            page_pdfs.append(graph_pdf)

    # --- Staple ---
    print()
    print(f"Stapling {len(page_pdfs)} PDF pages -> {_OUT_PDF} ...")

    try:
        from pypdf import PdfWriter

        writer = PdfWriter()
        for p in page_pdfs:
            writer.append(str(p))
        with open(_OUT_PDF, "wb") as fh:
            writer.write(fh)
        size_kb = _OUT_PDF.stat().st_size // 1024
        print(f"Done. {_OUT_PDF} ({size_kb} KB, {len(writer.pages)} pages)")
    except ImportError:
        # Fallback: pdfunite
        print("pypdf not available; falling back to pdfunite ...")
        import subprocess

        args = ["pdfunite"] + [str(p) for p in page_pdfs] + [str(_OUT_PDF)]
        subprocess.run(args, check=True)
        size_kb = _OUT_PDF.stat().st_size // 1024
        print(f"Done via pdfunite. {_OUT_PDF} ({size_kb} KB)")

    # --- Summary ---
    print()
    if errors:
        print(f"GAPS ({len(errors)} pages failed to render):")
        for e in errors:
            print(f"  * {e}")
    else:
        print("All pages rendered successfully -- no GAPs.")


if __name__ == "__main__":
    main()
