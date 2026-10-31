"""Causal receipt grids: attachment validation + visual grammar (D11-D14).

The mech-interp kit EXECUTES counterfactuals; tviz owns attachment
validation, the visual grammar, and the artifact. A receipt refuses at
construction (see :class:`torchlens.transformer.CausalReceipt`) on zero fires, a
moved negative control, or an unmoved positive control -- and the GRID
figure additionally refuses without the measured joint effect, because a
grid of large per-head numbers invites "remove them all and get the sum"
and the truth can have the opposite sign (memo D13: measured on gpt2 layer
0, one head +13.27 alone, all twelve together -0.68).

Grammar (D16): effects are printed numbers on a diverging-colored effect
matrix of their OWN -- effects never recolor attention cells; annotated
attention figures keep the pattern channel for the pattern and put the
effect on headers/borders (see ``render_attention``'s ``annotation=``).

All spellings are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ._errors import refuse
from ._mpl import DIVERGING_CMAP, colorbar, figure, require_matplotlib, save_figure
from ._records import Artifact, CausalReceipt, GqaInfo
from ._wording import (
    JOINT_EFFECT_WORDING,
    KV_GROUP_RECEIPT_WORDING,
    KV_JOINT_EFFECT_WORDING,
    MEASURED_N_OF_M,
    OUTPUT_EQUIVALENT_WORDING,
    RECEIPT_CELL_NO_BASELINE_WORDING,
    RECEIPT_CELL_WORDING,
)

__all__ = ["receipt_from_patch_grid", "render_receipt_grid"]


def receipt_from_patch_grid(  # noqa: PLR0913 -- every mandatory D12 evidence field is explicit
    grid: Any,
    *,
    layer: str,
    negative_control: float,
    positive_control: float,
    joint_effect: float | None = None,
    intervention: str | None = None,
    metric: str = "metric",
    disclosure: str = "direct",
    kv_groups: GqaInfo | None = None,
) -> CausalReceipt:
    """Build a validated receipt from a mech-interp patch grid row.

    Parameters
    ----------
    grid:
        A :class:`torchlens.mechinterp.PatchGrid` with a ``head`` axis. Each
        effect is the row's patched metric minus the grid's
        ``receipts["corrupted_metric"]``: the grid patches clean values into
        corrupted reruns, so the unpatched corrupted run is the baseline and
        a cell that changes nothing reads exactly 0. A grid without that
        receipt refuses. The grid's fire ledger supplies the fire proof (the
        row's own count when the ledger shows one fire per cell).
    layer:
        The layer coordinate to extract (module address as recorded in the
        grid's ``layer`` coordinates).
    negative_control:
        Measured self-patch metric delta (mandatory, must be ~0).
    positive_control:
        Measured large-perturbation metric delta (mandatory, must move --
        the only check that catches the silent no-fire class).
    joint_effect:
        Measured joint effect of the same intervention on all the row's
        heads together, as a delta from the same corrupted baseline;
        required later by the grid FIGURE (D13).
    intervention:
        Intervention wording; the default says what a patch grid does
        ("clean activation of one head patched into the corrupted run", or
        "... one kv group ..." with ``kv_groups``).
    metric:
        Metric wording printed on the figure.
    disclosure:
        ``"direct"`` or the output-equivalent wording for lowered engines.
    kv_groups:
        Pass the model's grouping for a grid over a key or value site
        (``sites="k"`` / ``"v"``): its head axis counts shared KV groups,
        not query heads, and the receipt then says which query heads share
        each group. Under grouped-query attention a per-query-head effect
        is not measurable at those sites. A query-head interceptor grid
        (``mechinterp.patch_query_heads_grid``) already counts query heads
        and refuses ``kv_groups``.

    A grid whose receipts carry ``effect_site`` (the query-head
    interceptor's disclosure) sets the receipt's ``effect_source`` to it, so
    every rendering prints which op the effect was measured at.

    Returns
    -------
    CausalReceipt
        The validated per-head (or per-KV-group) receipt for ``layer``.
    """

    if "head" not in grid.axes:
        refuse(
            code="tv_receipt_invalid",
            message=f"The patch grid has axes {grid.axes}; a head receipt needs a 'head' axis.",
            remedy="run mechinterp.patch_heads_grid (axes ('layer', 'head'))",
            axes=list(grid.axes),
        )
    layer_labels = list(grid.coordinates["layer"])
    if layer not in layer_labels:
        refuse(
            code="tv_receipt_invalid",
            message=f"Layer {layer!r} is not in the grid coordinates.",
            remedy=f"pick one of {layer_labels[:6]}",
            layers=layer_labels,
        )
    receipts = dict(getattr(grid, "receipts", {}) or {})
    baseline = _corrupted_baseline(receipts)
    row = grid.values[layer_labels.index(layer)]
    heads = tuple(int(head) for head in grid.coordinates["head"])
    effects = tuple(float(value) - baseline for value in row.reshape(-1))
    effect_site = receipts.get("effect_site")
    if effect_site is not None and kv_groups is not None:
        refuse(
            code="tv_receipt_invalid",
            message="This grid came from the query-head interceptor: its head axis already "
            "counts query heads, so it is not a kv-group grid.",
            remedy="drop kv_groups=; the receipt is an ordinary query-head receipt",
            effect_site=str(effect_site),
        )
    receipt = CausalReceipt(
        layer=layer,
        heads=heads,
        effects=effects,
        metric=str(receipts.get("metric_provenance", metric)),
        intervention=intervention or _default_intervention(kv_groups),
        engine=str(receipts.get("engine", "rerun")),
        disclosure=disclosure,
        fires=_row_fires(receipts, n_row_cells=len(effects)),
        negative_control=negative_control,
        positive_control=positive_control,
        joint_effect=joint_effect,
        kv_groups=kv_groups,
        baseline_metric=baseline,
        baseline_run="unpatched corrupted run",
    )
    return receipt if effect_site is None else replace(receipt, effect_source=str(effect_site))


def _corrupted_baseline(receipts: dict[str, Any]) -> float:
    """Return the grid's corrupted-run metric; refuse a grid that lacks it."""

    baseline = receipts.get("corrupted_metric")
    if baseline is None:
        refuse(
            code="tv_receipt_invalid",
            message="The patch grid records no corrupted-run baseline metric, so its cells "
            "are patched metric values, not effects; a receipt plots effects only.",
            remedy="build the grid with mechinterp.grid, patch_heads_grid or "
            "patch_query_heads_grid (they record receipts['corrupted_metric']), or set "
            "receipts['corrupted_metric'] to the unpatched corrupted run's metric",
            receipts=sorted(receipts),
        )
    return float(baseline)


def _row_fires(receipts: dict[str, Any], *, n_row_cells: int) -> int:
    """Return the fires behind one grid row, not the whole grid's count.

    The grid driver raises on a cell that did not fire, so a ledger whose
    total equals the cell count proves exactly one fire per cell and the
    row's count is its cell count. Any other ledger keeps the grid total.
    """

    fires = int(receipts.get("fires", 0))
    return n_row_cells if fires and fires == receipts.get("n_cells") else fires


def _default_intervention(kv_groups: GqaInfo | None) -> str:
    """Return the patch-grid intervention wording for one cell's unit."""

    unit = "one head" if kv_groups is None else "one kv group"
    return f"clean activation of {unit} patched into the corrupted run"


def _grid_matrix(receipts: list[CausalReceipt]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Assemble the [layer, head] effect matrix; NaN marks unmeasured."""

    heads = receipts[0].heads
    for receipt in receipts[1:]:
        if receipt.kv_groups != receipts[0].kv_groups:
            refuse(
                code="tv_receipt_invalid",
                message="Receipts mix query-head and kv-group cells (or different "
                "groupings); one grid shows one unit.",
                remedy="render query-head and kv-group receipts as separate grids",
            )
        if receipt.heads != heads:
            refuse(
                code="tv_receipt_invalid",
                message="Receipts disagree on the head axis; attachment is by "
                "coordinates, never array position.",
                remedy="build every row over the same original head indices",
            )
    matrix = np.full((len(receipts), len(heads)), np.nan)
    for row_index, receipt in enumerate(receipts):
        for col, effect in enumerate(receipt.effects):
            if effect is not None:
                matrix[row_index, col] = effect
    return matrix, heads


def _joint_line(receipts: list[CausalReceipt], joint_effect: float | None) -> str:
    """Return the mandatory joint-measurement line (D13); refuse without it."""

    if len(receipts) == 1 and joint_effect is None:
        joint_effect = receipts[0].joint_effect
    if joint_effect is None:
        refuse(
            code="tv_receipt_invalid",
            message="A receipt grid must print the measured JOINT effect of the "
            "displayed set (D13): single-head effects need not be additive (measured on "
            "gpt2 layer 0: one head +13.27 alone, all twelve together -0.68), so the "
            "figure refuses rather than inviting the false sum inference.",
            remedy="measure one extra forward applying the intervention to all displayed "
            "heads together "
            "and pass joint_effect= (or carry it on the single receipt)",
        )
    n_heads = sum(receipt.measured_n for receipt in receipts)
    cell_sum = sum(receipt.cell_sum for receipt in receipts)
    wording = JOINT_EFFECT_WORDING if receipts[0].kv_groups is None else KV_JOINT_EFFECT_WORDING
    return wording.format(n=n_heads, joint=joint_effect, cell_sum=cell_sum)


def _kv_group_lines(receipt: CausalReceipt) -> list[str]:
    """Return the kv-group disclosure: what a cell is, then who shares it."""

    gqa = receipt.kv_groups
    if gqa is None:
        return []
    lines = [
        KV_GROUP_RECEIPT_WORDING.format(
            per_group=gqa.n_query_heads // gqa.n_kv_heads,
            n_query=gqa.n_query_heads,
            n_groups=gqa.n_kv_heads,
        )
    ]
    sharing = receipt.sharing_lines()
    lines.extend("; ".join(sharing[start : start + 2]) for start in range(0, len(sharing), 2))
    return lines


def _cell_line(receipts: list[CausalReceipt]) -> str:
    """Return what one cell shows; refuse rows measured from different baselines."""

    baselines = {(receipt.baseline_metric, receipt.baseline_run) for receipt in receipts}
    if len(baselines) > 1:
        refuse(
            code="tv_receipt_invalid",
            message="The receipts were measured from different baseline runs "
            f"({sorted(map(str, baselines))}); their cells are not on one scale.",
            remedy="render receipts from one grid (one clean/corrupted pair) together",
        )
    ((baseline, run),) = baselines
    metric = receipts[0].metric
    if baseline is None:
        return RECEIPT_CELL_NO_BASELINE_WORDING.format(metric=metric)
    return RECEIPT_CELL_WORDING.format(metric=metric, run=run, baseline=baseline)


def _receipt_disclosures(receipts: list[CausalReceipt], joint_effect: float | None) -> list[str]:
    """Assemble the grid figure's full honesty footer."""

    first = receipts[0]
    measured = sum(receipt.measured_n for receipt in receipts)
    total = sum(len(receipt.heads) for receipt in receipts)
    lines = [
        f"intervention: {first.intervention}; metric: {first.metric}",
        _cell_line(receipts),
        f"engine: {first.engine} ({first.disclosure})",
        _joint_line(receipts, joint_effect),
        MEASURED_N_OF_M.format(n=measured, m=total) + " (unmeasured cells are blank, never zero)",
        f"controls: negative {first.negative_control:+.2e} "
        f"(tol {first.negative_control_tol:.1e}), positive {first.positive_control:+.3g}; "
        f"fires: {sum(receipt.fires for receipt in receipts)}",
        f"pattern source: {first.pattern_source}; effect source: {first.effect_source}",
        *_kv_group_lines(first),
    ]
    if first.disclosure == OUTPUT_EQUIVALENT_WORDING:
        lines.append(
            "lowered engine: the effect is measured at the real post-projection output "
            "site, never written to the reconstructed facet"
        )
    return lines


def _column_labels(receipt: CausalReceipt) -> list[str]:
    """Return x labels: ``h3`` for query heads, ``kv 1`` over ``q0-3`` for groups."""

    if receipt.kv_groups is None:
        return [f"h{head}" for head in receipt.heads]
    labels = []
    for group in receipt.heads:
        members = receipt.kv_groups.query_heads(group)
        labels.append(f"kv {group + 1}\nq{members[0]}-{members[-1]}")
    return labels


#: Smallest text size on the figure, in points: 8 pt at the 200 dpi export
#: reads as about 11 CSS px in a 760 px pane, the legibility floor.
_TEXT_PT = 8

#: Cells darker than this fraction of the colour scale print white text.
_DARK_CELL = 0.6

#: Height of one matrix row, in inches.
_ROW_INCHES = 0.5

#: Vertical gaps, in inches: page edge, between bands, under the title.
_EDGE_INCHES = 0.06
_GAP_INCHES = 0.12


def _wrap_footer(lines: list[str], width_inches: float) -> str:
    """Wrap each disclosure line to ``width_inches`` (continuations indented)."""

    chars = max(40, int(width_inches * 72 / (0.55 * _TEXT_PT)))
    return "\n".join(textwrap.fill(line, width=chars, subsequent_indent="    ") for line in lines)


def _cell_format(scale: float) -> str:
    """Return the cell format: two significant digits of the largest effect.

    A small-effect grid gets more decimals, so a cell that moved never
    prints as ``+0.00``; ``+0.00`` (two decimals) stays for scales of 0.1
    and up.
    """

    decimals = min(6, max(2, 1 - math.floor(math.log10(scale))))
    return f"{{:+.{decimals}f}}"


def _fit_footer(fig: Any, footer: Any, lines: list[str], *, left: float, renderer: Any) -> None:
    """Place the footer at ``left`` and wrap it until it ends inside the figure."""

    footer.set_x(left)
    width = fig.get_figwidth() * (1.0 - left)
    for _ in range(4):
        footer.set_text(_wrap_footer(lines, width))
        overflow = footer.get_window_extent(renderer).x1 - fig.bbox.x1
        if overflow <= 0:
            return
        width -= overflow / fig.dpi + 0.1


def _height_inches(extent: Any, dpi: float) -> float:
    """Return a drawn extent's height in inches."""

    return float(extent.height) / dpi


def _reserve_footer(  # noqa: PLR0913 -- the artists the layout measures
    fig: Any, ax: Any, bar: Any, title: Any, footer: Any, *, lines: list[str], n_rows: int
) -> None:
    """Size the figure so title, matrix, tick labels and footer never overlap.

    The footer starts at the layer labels' left edge and wraps to the width
    from there to the figure's right edge. The bands are then stacked in
    inches from the measured extents of the drawn text (footer lines, x tick
    labels, title, colorbar label), so a long footer reserves its own height
    instead of drawing over the matrix. The matrix is at least as tall as
    the colorbar label, which is centred on it.
    """

    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    left = min(0.01, ax.get_tightbbox(renderer).x0 / fig.bbox.width)
    _fit_footer(fig, footer, lines, left=left, renderer=renderer)
    footer_h = _height_inches(footer.get_window_extent(renderer), dpi)
    title_h = _height_inches(title.get_window_extent(renderer), dpi)
    tick_h = (ax.bbox.y0 - ax.get_tightbbox(renderer).y0) / dpi
    label_h = _height_inches(bar.ax.yaxis.label.get_window_extent(renderer), dpi)
    matrix_h = max(_ROW_INCHES * n_rows, label_h + _GAP_INCHES)
    bottom = _EDGE_INCHES + footer_h + _GAP_INCHES + tick_h
    top = _EDGE_INCHES + title_h + _GAP_INCHES
    height = bottom + matrix_h + top
    fig.set_size_inches(fig.get_figwidth(), height)
    footer.set_y(_EDGE_INCHES / height)
    title.set_y(1.0 - _EDGE_INCHES / height)
    fig.subplots_adjust(bottom=bottom / height, top=1.0 - top / height)


def render_receipt_grid(
    receipts: list[CausalReceipt] | CausalReceipt,
    path: Path | str,
    *,
    joint_effect: float | None = None,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render the measured-intervention receipt grid (THE differentiator).

    Layers on the y axis (one row per receipt), heads on the x axis; each
    measured cell prints its effect (the change in the metric from the
    unpatched baseline run, whose value the footer prints) on a
    zero-centered diverging scale; unmeasured cells are blank, never zero.
    The joint-measurement line, the controls, the fire count, and the
    engine disclosure are printed ON the figure, in a footer whose height is
    reserved below the matrix so it never covers a cell or a label.

    Parameters
    ----------
    receipts:
        One validated receipt per layer row (or a single receipt).
    path:
        Output path (PNG/SVG/PDF by suffix).
    joint_effect:
        Measured joint effect of the whole displayed set; REQUIRED when
        more than one row is displayed (a single receipt may carry its
        own).
    svg_fonttype:
        ``'path'`` (default) or ``'none'``.

    Returns
    -------
    Artifact
        The written file plus every rendered disclosure line.
    """

    rows = [receipts] if isinstance(receipts, CausalReceipt) else list(receipts)
    if not rows:
        refuse(
            code="tv_receipt_invalid",
            message="No receipts to render.",
            remedy="pass at least one validated CausalReceipt",
        )
    matrix, heads = _grid_matrix(rows)
    disclosures = _receipt_disclosures(rows, joint_effect)
    matplotlib = require_matplotlib()
    scale = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    scale = scale or 1.0
    fig = figure(width=0.62 * len(heads) + 2.8, height=_ROW_INCHES * len(rows) + 2.0)
    ax = fig.add_subplot(111)
    cmap = matplotlib.colormaps[DIVERGING_CMAP].copy()
    cmap.set_bad("#f5f5f5")
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        vmin=-scale,
        vmax=scale,
        rasterized=False,
    )
    ax.invert_yaxis()
    cell_format = _cell_format(scale)
    for row_index in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row_index, col]
            if np.isfinite(value):
                ax.text(
                    col + 0.5,
                    row_index + 0.5,
                    cell_format.format(value),
                    ha="center",
                    va="center",
                    fontsize=_TEXT_PT,
                    color="white" if abs(value) > _DARK_CELL * scale else "black",
                )
    ax.set_xticks([col + 0.5 for col in range(len(heads))])
    ax.set_xticklabels(_column_labels(rows[0]), fontsize=_TEXT_PT)
    ax.set_yticks([row_index + 0.5 for row_index in range(len(rows))])
    ax.set_yticklabels([receipt.layer for receipt in rows], fontsize=_TEXT_PT)
    ax.tick_params(length=0)
    ax.set_xlabel("query head" if rows[0].kv_groups is None else "kv group", fontsize=_TEXT_PT)
    bar = colorbar(fig, mesh, ax, label=f"change in\n{rows[0].metric}")
    bar.ax.yaxis.label.set_fontsize(_TEXT_PT)
    bar.ax.tick_params(labelsize=_TEXT_PT)
    unit_title = "head" if rows[0].kv_groups is None else "kv-group"
    title = fig.suptitle(f"measured {unit_title} effects on {rows[0].metric}", fontsize=10)
    footer = fig.text(0.01, 0.0, "", fontsize=_TEXT_PT, va="bottom")
    _reserve_footer(fig, ax, bar, title, footer, lines=disclosures, n_rows=len(rows))
    written = save_figure(fig, path, svg_fonttype=svg_fonttype)
    return Artifact(
        paths=(written,),
        format=Path(path).suffix.lstrip(".").lower(),
        title="receipt grid",
        disclosure_lines=tuple(disclosures),
        provenance=f"engine: {rows[0].engine} ({rows[0].disclosure})",
        fingerprint="",
        svg_fonttype=svg_fonttype,
        pages_total=1,
    )
