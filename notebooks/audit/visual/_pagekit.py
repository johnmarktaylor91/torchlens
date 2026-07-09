"""Page-composition helpers for the TorchLens visual audit pack.

Each audit "page" is a single PDF page composed with matplotlib:
a header strip (section / page number / title), a caption block explaining
what is demonstrated and what the eye should check, and one or more graph
panels rendered by Graphviz to PNG and embedded at native resolution
(the matplotlib PDF backend keeps the full-resolution raster, so zooming
into the stapled PDF preserves all detail).

Text-only pages (table of contents, section headers, diagnostics dumps)
are composed the same way without image panels.
"""

from __future__ import annotations

import pathlib
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # graphviz output of big graphs is legitimately huge

PAGE_W_IN = 13.0  # fixed page width (inches)
MAX_PANEL_H_IN = 42.0  # cap per-panel display height (very deep graphs)
HEADER_COLOR = "#1a3a5c"
CAPTION_BG = "#f4f6f8"


# ---------------------------------------------------------------------------
# Page / panel specs
# ---------------------------------------------------------------------------


@dataclass
class Panel:
    """One rendered graph image on a page."""

    subtitle: str
    model_key: str
    method: str = "draw"  # 'draw' | 'draw_backward' | 'draw_combined'
    kwargs: dict = field(default_factory=dict)
    trace_variant: str = "plain"  # 'plain' | 'backward' | custom key
    trace_builder: Optional[Callable[[], Any]] = None  # overrides default trace
    kwargs_fn: Optional[Callable[[Any], dict]] = None  # trace -> extra kwargs
    subtitle_fn: Optional[Callable[[Any], str]] = None  # trace -> subtitle


@dataclass
class Page:
    """One PDF page in the pack."""

    label: str  # filesystem-safe id
    title: str
    caption: str  # what is demonstrated + what the eye should check
    panels: list[Panel] = field(default_factory=list)
    covers: list[str] = field(default_factory=list)  # coverage-axis tags
    ncols: int = 0  # 0 = auto (1->1, 2->2, else 3)
    text_fn: Optional[Callable[[], str]] = None  # text-only page body
    notes: str = ""  # extra coverage-matrix notes


@dataclass
class Section:
    letter: str
    title: str
    blurb: str
    pages: list[Page]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _wrap(text: str, width: int = 150) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            out_lines.append("")
            continue
        out_lines.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(out_lines)


def compose_page(
    out_path: pathlib.Path,
    header_left: str,
    header_right: str,
    title: str,
    caption: str,
    panel_items: list[tuple[str, Optional[pathlib.Path], str]],
    ncols: int = 0,
) -> None:
    """Compose one page PDF.

    Parameters
    ----------
    panel_items:
        List of ``(subtitle, png_path_or_None, error_msg)``.  A missing image
        renders as a red error placeholder so failures stay visible in the pack.
    ncols:
        Panel grid columns; 0 selects 1/2/3 automatically.
    """

    n = len(panel_items)
    if ncols <= 0:
        ncols = 1 if n <= 1 else (2 if n == 2 else min(3, n))
    ncols = min(ncols, max(n, 1))

    # --- measure images ---
    images: list[Optional[Image.Image]] = []
    for _sub, path, _err in panel_items:
        img = None
        if path is not None and path.exists():
            img = Image.open(path)
            img.load()
        images.append(img)

    gutter_in = 0.25
    side_margin_in = 0.35
    content_w_in = PAGE_W_IN - 2 * side_margin_in
    panel_w_in = (content_w_in - (ncols - 1) * gutter_in) / ncols

    # Row packing: an image whose natural size (~100 dpi) is much wider than a
    # grid cell gets its own FULL-WIDTH row -- squeezing wide graphs into a
    # grid cell renders them illegible, which defeats the page.
    def _is_wide(img: Optional[Image.Image]) -> bool:
        return ncols > 1 and img is not None and (img.width / 100.0) > panel_w_in * 1.55

    rows: list[list[int]] = []
    current: list[int] = []
    for i in range(n):
        if _is_wide(images[i]):
            if current:
                rows.append(current)
                current = []
            rows.append([i])
        else:
            current.append(i)
            if len(current) == ncols:
                rows.append(current)
                current = []
    if current:
        rows.append(current)

    def _row_panel_w(row: list[int]) -> float:
        if len(row) == 1 and _is_wide(images[row[0]]):
            return content_w_in
        return panel_w_in

    # display size rule: never blow a small render up past its native size at
    # ~100 dpi (blurry); cap very deep graphs at MAX_PANEL_H_IN.
    def _disp_size(img: Image.Image, width_budget: float) -> tuple[float, float]:
        disp_w = min(width_budget, img.width / 100.0)
        disp_h = disp_w * img.height / img.width
        if disp_h > MAX_PANEL_H_IN:
            disp_h = MAX_PANEL_H_IN
            disp_w = disp_h * img.width / img.height
        return disp_w, disp_h

    # per-row display heights
    subtitle_h_in = 0.28
    row_heights: list[float] = []
    for row in rows:
        budget = _row_panel_w(row)
        h = 1.2  # min height (error placeholder)
        for i in row:
            if images[i] is not None:
                h = max(h, _disp_size(images[i], budget)[1])
        row_heights.append(h + subtitle_h_in)
    nrows = len(rows)

    caption_lines = _wrap(caption).count("\n") + 1
    header_h_in = 0.42
    title_h_in = 0.36
    caption_h_in = 0.20 * caption_lines + 0.25
    panels_h_in = sum(row_heights) + 0.15 * max(nrows - 1, 0)
    page_h_in = header_h_in + title_h_in + caption_h_in + panels_h_in + 0.5

    fig = plt.figure(figsize=(PAGE_W_IN, page_h_in))

    def y_frac(y_in: float) -> float:
        return 1.0 - y_in / page_h_in

    # --- header strip ---
    fig.patches.append(
        plt.Rectangle(
            (0, y_frac(header_h_in)),
            1,
            header_h_in / page_h_in,
            transform=fig.transFigure,
            facecolor=HEADER_COLOR,
            edgecolor="none",
        )
    )
    fig.text(
        side_margin_in / PAGE_W_IN,
        y_frac(header_h_in / 2),
        header_left,
        va="center",
        ha="left",
        fontsize=11,
        color="white",
        fontweight="bold",
    )
    fig.text(
        1 - side_margin_in / PAGE_W_IN,
        y_frac(header_h_in / 2),
        header_right,
        va="center",
        ha="right",
        fontsize=10,
        color="#cfe0f0",
    )

    # --- title ---
    y_cursor = header_h_in + 0.10
    fig.text(
        side_margin_in / PAGE_W_IN,
        y_frac(y_cursor + 0.12),
        title,
        va="center",
        ha="left",
        fontsize=14,
        fontweight="bold",
        color="#111111",
    )
    y_cursor += title_h_in

    # --- caption block ---
    fig.patches.append(
        plt.Rectangle(
            (side_margin_in / PAGE_W_IN * 0.5, y_frac(y_cursor + caption_h_in - 0.05)),
            1 - side_margin_in / PAGE_W_IN,
            (caption_h_in - 0.10) / page_h_in,
            transform=fig.transFigure,
            facecolor=CAPTION_BG,
            edgecolor="#d5dbe2",
            linewidth=0.6,
        )
    )
    fig.text(
        side_margin_in / PAGE_W_IN,
        y_frac(y_cursor + 0.08),
        _wrap(caption),
        va="top",
        ha="left",
        fontsize=9.5,
        color="#222222",
        linespacing=1.35,
    )
    y_cursor += caption_h_in

    # --- panels ---
    for r, row in enumerate(rows):
        row_h = row_heights[r]
        row_budget = _row_panel_w(row)
        for c, i in enumerate(row):
            sub, _path, err = panel_items[i]
            img = images[i]
            x0_in = side_margin_in + c * (panel_w_in + gutter_in)
            # subtitle -- clipped to the panel width so neighbors never overlap
            sub_fontsize = 10 if ncols <= 3 else 8.5
            max_chars = max(int(row_budget * (14 if ncols <= 3 else 17)), 12)
            sub_text = sub if len(sub) <= max_chars else sub[: max_chars - 3] + "..."
            fig.text(
                x0_in / PAGE_W_IN,
                y_frac(y_cursor + 0.14),
                sub_text,
                va="center",
                ha="left",
                fontsize=sub_fontsize,
                fontstyle="italic",
                color="#333333",
            )
            img_top_in = y_cursor + subtitle_h_in
            if img is not None:
                disp_w, disp_h = _disp_size(img, row_budget)
                ax = fig.add_axes(
                    (
                        x0_in / PAGE_W_IN,
                        y_frac(img_top_in + disp_h),
                        disp_w / PAGE_W_IN,
                        disp_h / page_h_in,
                    )
                )
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#c9ced4")
                    spine.set_linewidth(0.6)
            else:
                ax = fig.add_axes(
                    (
                        x0_in / PAGE_W_IN,
                        y_frac(img_top_in + 1.0),
                        panel_w_in / PAGE_W_IN,
                        1.0 / page_h_in,
                    )
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor("#fff0f0")
                ax.text(
                    0.5,
                    0.5,
                    f"RENDER FAILED\n{err[:300]}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="red",
                    wrap=True,
                )
        y_cursor += row_h + 0.15

    fig.savefig(str(out_path), format="pdf")
    plt.close(fig)
    for img in images:
        if img is not None:
            img.close()


def compose_text_page(
    out_path: pathlib.Path,
    header_left: str,
    header_right: str,
    title: str,
    body: str,
    body_fontsize: float = 9.0,
    title_fontsize: float = 15,
    banner: bool = False,
) -> None:
    """Compose a text-only page (TOC, section header, diagnostics)."""

    # Wrap long prose lines so section blurbs never clip at the page edge.
    body = "\n".join(
        wrapped
        for line in body.splitlines()
        for wrapped in (textwrap.wrap(line, width=112) or [""])
    )

    n_lines = body.count("\n") + 1
    header_h_in = 0.42
    body_h_in = 0.165 * n_lines * (body_fontsize / 9.0) + 0.6
    page_h_in = max(header_h_in + 0.9 + body_h_in, 4.0)
    if banner:
        page_h_in = max(page_h_in, 6.0)

    fig = plt.figure(figsize=(PAGE_W_IN, page_h_in))

    def y_frac(y_in: float) -> float:
        return 1.0 - y_in / page_h_in

    fig.patches.append(
        plt.Rectangle(
            (0, y_frac(header_h_in)),
            1,
            header_h_in / page_h_in,
            transform=fig.transFigure,
            facecolor=HEADER_COLOR,
            edgecolor="none",
        )
    )
    fig.text(
        0.027,
        y_frac(header_h_in / 2),
        header_left,
        va="center",
        ha="left",
        fontsize=11,
        color="white",
        fontweight="bold",
    )
    fig.text(
        0.973,
        y_frac(header_h_in / 2),
        header_right,
        va="center",
        ha="right",
        fontsize=10,
        color="#cfe0f0",
    )

    title_y = header_h_in + 0.35
    fig.text(
        0.027,
        y_frac(title_y),
        title,
        va="center",
        ha="left",
        fontsize=title_fontsize + (6 if banner else 0),
        fontweight="bold",
        color=HEADER_COLOR if banner else "#111111",
    )
    fig.text(
        0.027,
        y_frac(title_y + 0.45),
        body,
        va="top",
        ha="left",
        fontsize=body_fontsize,
        fontfamily="monospace",
        color="#222222",
        linespacing=1.4,
    )
    fig.savefig(str(out_path), format="pdf")
    plt.close(fig)
