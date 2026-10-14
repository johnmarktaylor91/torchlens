"""The native static array grid (B3, F16): budgeted, masked, motif-coded.

One HTML emitter over the ported truncation protocol
(:mod:`._truncation`), the six-state motif vocabulary and diverging map
(:mod:`._motifs`), and the axis badges (:mod:`._axis_labels`). Static facet
conventions from the treescope memo (section 4, tier 2):

- hierarchical gaps: OUTER facet gaps render 2x the inner gap;
- ``name:size`` axis badges on every grid;
- truncation bands are VISIBLY distinct dotted cells riding the validity
  mask -- never a data zero painted neutral;
- zero-element tensors get an explicit EMPTY rendering;
- bounded by construction: the truncation budgets are enforced BEFORE any
  conversion, so a grid never transfers more than the visible cells.

Every spelling is DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

import numpy as np

from ._axis_labels import AxisLabel, axis_labels
from ._motifs import (
    Motif,
    SignedBounds,
    classify_value,
    resolve_motif_table,
    resolve_signed_bounds,
    value_color,
)
from ._truncation import (
    CELL_BUDGET_DEFAULT,
    EDGE_ITEMS_DEFAULT,
    PER_AXIS_BUDGET_DEFAULT,
    infer_balanced_truncation,
    truncate_tensor_with_mask,
)

__all__ = ["GridBudgets", "GridRender", "array_grid_html"]


@dataclass(frozen=True)
class GridRender:
    """One rendered grid plus its honesty facts.

    Attributes
    ----------
    html:
        Self-contained fragment (scoped classes, inline colors only).
    cells_shown / cells_total:
        Visible cell count vs the payload's true element count -- the
        transfer-bound assertion surface (< 2 percent of numel on the big
        fixtures rides these numbers).
    truncated:
        Whether any axis was truncated.
    disclosure:
        Bounds/trim/fallback line from :class:`SignedBounds`, plus the
        truncation disclosure when applicable.
    """

    html: str
    cells_shown: int
    cells_total: int
    truncated: bool
    disclosure: str


_GRID_CSS_EMITTED_MARKER = "tl-grid"


def _cell_html(
    value: Any,
    state: str,
    bounds: SignedBounds,
    motifs: dict[str, Motif],
    index: str | None,
) -> str:
    """Render one cell: color claim for finite values, motif otherwise.

    Every real cell carries its source index and value as static
    attributes (``title`` for the native tooltip, ``data-tl-index`` and
    ``data-tl-value`` for the optional hover readout), so the card reads
    completely with scripts blocked. Truncation-band cells carry no index.
    """

    hover = ""
    if index is not None:
        shown = f"{float(value):.6g}" if state == "finite" else motifs[state].title
        hover = (
            f' data-tl-index="{escape(index, quote=True)}"'
            f' data-tl-value="{escape(shown, quote=True)}"'
        )
    if state == "finite":
        color = value_color(float(value), bounds)
        title = f"{index} = {float(value):.6g}" if index is not None else f"{float(value):.6g}"
        return (
            f'<td class="tl-grid-cell" style="background:{color}" '
            f'title="{escape(title, quote=True)}"{hover}></td>'
        )
    motif = motifs[state]
    title = motif.title if index is None else f"{index} = {motif.title}"
    return (
        f'<td class="tl-grid-cell {escape(motif.css_class, quote=True)}" '
        f'title="{escape(title, quote=True)}"{hover}>{escape(motif.glyph)}</td>'
    )


def _source_index(
    dest: tuple[int, ...], shape: tuple[int, ...], edges: tuple[int | None, ...]
) -> str:
    """Map a rendered cell position back to its index in the source tensor.

    A truncated axis renders ``edge`` head cells, one band cell, then
    ``edge`` tail cells; tail cell ``d`` reads source ``size - (2*edge+1-d)``.
    """

    parts = []
    for position, size, edge in zip(dest, shape, edges, strict=True):
        if edge is None or position < edge:
            parts.append(position)
        else:
            parts.append(size - (2 * edge + 1 - position))
    return "[" + ", ".join(str(part) for part in parts) + "]"


def _badge_row(labels: tuple[AxisLabel, ...]) -> str:
    """Render the axis badges plus the positional-fallback disclosure."""

    badges = " ".join(escape(label.badge) for label in labels)
    positional = all(label.provenance == "positional" for label in labels) and bool(labels)
    suffix = " (positional axes)" if positional else ""
    return f'<div class="tl-grid-axes">{badges}{escape(suffix)}</div>'


@dataclass(frozen=True)
class GridBudgets:
    """Explicit truncation budgets for one grid render (memo 3.5).

    The defaults are the ArrayAutovisualizer numbers; every render passes
    them explicitly to the ported protocol through this record.
    """

    cells: int = CELL_BUDGET_DEFAULT
    per_axis: int = PER_AXIS_BUDGET_DEFAULT
    edge_items: int = EDGE_ITEMS_DEFAULT


def array_grid_html(
    tensor: Any,
    *,
    roles: tuple[str | None, ...] | None = None,
    budgets: GridBudgets | None = None,
    stats: Any | None = None,
    theme: str = "torchlens",
) -> GridRender:
    """Render one tensor as a bounded static HTML grid.

    Parameters
    ----------
    tensor:
        ``torch.Tensor`` payload (any device; only visible cells convert).
    roles:
        Optional proven per-axis role names (see
        :func:`torchlens.notebook._axis_labels.axis_labels`).
    budgets:
        Truncation budgets, passed EXPLICITLY to the ported protocol
        (defaults to the ArrayAutovisualizer numbers; memo 3.5).
    stats:
        Optional ``TensorStats`` record supplying exact bounds/moments; when
        absent, bounds come from the VISIBLE cells only, disclosed.
    theme:
        Themes preset consumed by the motif/anchor seams.

    Returns
    -------
    GridRender
        Fragment plus honesty facts; never raises on exotic payloads --
        callers wrap in the card never-raise boundary.
    """

    budgets = budgets or GridBudgets()
    shape = tuple(int(dim) for dim in tensor.shape)
    total = 1
    for dim in shape:
        total *= dim
    if total == 0:
        html = (
            '<div class="tl-grid tl-grid-empty">EMPTY tensor '
            f"shape ({escape(', '.join(str(d) for d in shape))})</div>"
        )
        return GridRender(html, 0, 0, False, "empty tensor")

    working = tensor
    if working.ndim == 0:
        working = working.reshape(1)
        shape = (1,)
    edges = infer_balanced_truncation(shape, budgets.cells, budgets.per_axis, budgets.edge_items)
    values, mask = truncate_tensor_with_mask(working, edges)
    truncated = any(edge is not None for edge in edges)

    bounds = _resolve_bounds(values, mask, stats)
    motifs = {motif.state: motif for motif in resolve_motif_table(theme)}
    labels = axis_labels(shape, roles)

    # Collapse to 2D for rendering: the last axis is columns, every leading
    # axis folds into row groups. Hierarchical facet gaps: a row starting a
    # more-major group gets a proportionally larger top gap (outer 2x inner).
    grid = values.reshape(-1, values.shape[-1]) if values.ndim > 1 else values.reshape(1, -1)
    grid_mask = mask.reshape(grid.shape)
    group_sizes = _group_sizes(values.shape)

    row_shape = tuple(values.shape[:-1]) if values.ndim > 1 else ()
    source_shape = tuple(int(dim) for dim in working.shape)
    rows_html: list[str] = []
    for row_index in range(grid.shape[0]):
        gap_level = _gap_level(row_index, group_sizes)
        style = f' style="border-top:{2 * gap_level}px solid transparent"' if gap_level else ""
        row_dest = (
            tuple(int(i) for i in np.unravel_index(row_index, row_shape)) if row_shape else ()
        )
        cells = "".join(
            _cell_html(
                grid[row_index, col],
                _state(grid, grid_mask, row_index, col, bounds),
                bounds,
                motifs,
                _source_index((*row_dest, col), source_shape, edges)
                if bool(grid_mask[row_index, col])
                else None,
            )
            for col in range(grid.shape[1])
        )
        rows_html.append(f"<tr{style}>{cells}</tr>")

    disclosure = bounds.disclosure
    if truncated:
        disclosure += f"; truncated to {values.size} of {total} cells"
    html = (
        f'<div class="tl-grid">{_badge_row(labels)}'
        f'<table class="tl-grid-table">{"".join(rows_html)}</table>'
        '<div class="tl-grid-hover" aria-live="polite"></div>'
        f'<div class="tl-grid-disclosure">{escape(disclosure)}</div></div>'
    )
    return GridRender(html, int(values.size), total, truncated, disclosure)


def _state(grid: Any, grid_mask: Any, row: int, col: int, bounds: SignedBounds) -> str:
    """Classify one rendered cell."""

    try:
        value = float(grid[row, col])
    except (TypeError, ValueError):
        return "unknown"
    return classify_value(value, bool(grid_mask[row, col]), bounds)


def _resolve_bounds(values: Any, mask: Any, stats: Any | None) -> SignedBounds:
    """Bounds from exact stats when supplied, else from visible cells."""

    if stats is not None:
        return resolve_signed_bounds(
            getattr(stats, "finite_min", None),
            getattr(stats, "finite_max", None),
            getattr(stats, "mean", None),
            getattr(stats, "sd", None),
        )
    import numpy as np

    visible = values[mask]
    finite = visible[np.isfinite(visible.astype(float))] if visible.size else visible
    if finite.size == 0:
        return resolve_signed_bounds(None, None, None, None)
    bounds = resolve_signed_bounds(
        float(finite.min()), float(finite.max()), float(finite.mean()), float(finite.std())
    )
    return SignedBounds(
        bounds.vmin,
        bounds.vmax,
        bounds.mode,
        bounds.trimmed,
        bounds.disclosure + " (bounds from visible cells)",
    )


def _group_sizes(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-stride of each facet-group boundary, innermost first.

    For shape ``(A, B, C, cols)`` the rendered rows number ``A*B*C``: every
    ``C`` rows a B-group starts (inner gap), every ``B*C`` rows an A-group
    starts (outer gap). Returns ``(C, B*C)`` -- running products of the
    row axes excluding the outermost, whose boundary never falls inside
    the rendered range.
    """

    row_axes = shape[:-1] if len(shape) > 1 else ()
    sizes: list[int] = []
    running = 1
    for dim in reversed(row_axes[1:]):
        running *= dim
        sizes.append(running)
    return tuple(sizes)


def _gap_level(row_index: int, group_sizes: tuple[int, ...]) -> int:
    """Facet-gap level of one row: outer group boundaries get larger gaps.

    Level 1 is the innermost boundary; each more-major boundary adds one,
    so the emitted gap (``2 * level`` px) keeps outer gaps 2x inner ones.
    """

    if row_index == 0:
        return 0
    level = 0
    for depth, size in enumerate(group_sizes, start=1):
        if row_index % size == 0:
            level = depth
    return level
