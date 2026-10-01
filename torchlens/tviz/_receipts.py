"""Causal receipt grids: attachment validation + visual grammar (D11-D14).

The mech-interp kit EXECUTES counterfactuals; tviz owns attachment
validation, the visual grammar, and the artifact. A receipt refuses at
construction (see :class:`torchlens.tviz.CausalReceipt`) on zero fires, a
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

from pathlib import Path
from typing import Any

import numpy as np

from ._errors import refuse
from ._mpl import DIVERGING_CMAP, colorbar, figure, require_matplotlib, save_figure
from ._records import Artifact, CausalReceipt
from ._wording import JOINT_EFFECT_WORDING, MEASURED_N_OF_M, OUTPUT_EQUIVALENT_WORDING

__all__ = ["receipt_from_patch_grid", "render_receipt_grid"]


def receipt_from_patch_grid(  # noqa: PLR0913 -- every mandatory D12 evidence field is explicit
    grid: Any,
    *,
    layer: str,
    negative_control: float,
    positive_control: float,
    joint_effect: float | None = None,
    intervention: str = "head contribution zeroed",
    metric: str = "metric",
    disclosure: str = "direct",
) -> CausalReceipt:
    """Build a validated receipt from a mech-interp patch grid row.

    Parameters
    ----------
    grid:
        A :class:`torchlens.mechinterp.PatchGrid` with a ``head`` axis; the
        row named by ``layer`` supplies the per-head effects. The grid's
        fire ledger supplies the fire proof.
    layer:
        The layer coordinate to extract (module address as recorded in the
        grid's ``layer`` coordinates).
    negative_control:
        Measured self-patch metric delta (mandatory, must be ~0).
    positive_control:
        Measured large-perturbation metric delta (mandatory, must move --
        the only check that catches the silent no-fire class).
    joint_effect:
        Measured joint effect of ablating all the row's heads together;
        required later by the grid FIGURE (D13).
    intervention:
        Intervention wording ("head contribution zeroed", never "attention
        module removed").
    metric:
        Metric wording printed on the figure.
    disclosure:
        ``"direct"`` or the output-equivalent wording for lowered engines.

    Returns
    -------
    CausalReceipt
        The validated per-head receipt for ``layer``.
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
    row = grid.values[layer_labels.index(layer)]
    heads = tuple(int(head) for head in grid.coordinates["head"])
    effects = tuple(float(value) for value in row.reshape(-1))
    receipts = dict(getattr(grid, "receipts", {}) or {})
    return CausalReceipt(
        layer=layer,
        heads=heads,
        effects=effects,
        metric=str(receipts.get("metric_provenance", metric)),
        intervention=intervention,
        engine=str(receipts.get("engine", "rerun")),
        disclosure=disclosure,
        fires=int(receipts.get("fires", 0)),
        negative_control=negative_control,
        positive_control=positive_control,
        joint_effect=joint_effect,
    )


def _grid_matrix(receipts: list[CausalReceipt]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Assemble the [layer, head] effect matrix; NaN marks unmeasured."""

    heads = receipts[0].heads
    for receipt in receipts[1:]:
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
            "displayed set (D13): single-head effects are not additive (measured on "
            "gpt2 layer 0: one head +13.27 alone, all twelve together -0.68), so the "
            "figure refuses rather than inviting the false sum inference.",
            remedy="measure one extra forward ablating all displayed heads together "
            "and pass joint_effect= (or carry it on the single receipt)",
        )
    n_heads = sum(receipt.measured_n for receipt in receipts)
    cell_sum = sum(receipt.cell_sum for receipt in receipts)
    return JOINT_EFFECT_WORDING.format(n=n_heads, joint=joint_effect, cell_sum=cell_sum)


def _receipt_disclosures(receipts: list[CausalReceipt], joint_effect: float | None) -> list[str]:
    """Assemble the grid figure's full honesty footer."""

    first = receipts[0]
    measured = sum(receipt.measured_n for receipt in receipts)
    total = sum(len(receipt.heads) for receipt in receipts)
    lines = [
        f"intervention: {first.intervention}; metric: {first.metric}",
        f"engine: {first.engine} ({first.disclosure})",
        _joint_line(receipts, joint_effect),
        MEASURED_N_OF_M.format(n=measured, m=total) + " (unmeasured cells are blank, never zero)",
        f"controls: negative {first.negative_control:+.2e} "
        f"(tol {first.negative_control_tol:.1e}), positive {first.positive_control:+.3g}; "
        f"fires: {sum(receipt.fires for receipt in receipts)}",
        f"pattern source: {first.pattern_source}; effect source: {first.effect_source}",
    ]
    if first.disclosure == OUTPUT_EQUIVALENT_WORDING:
        lines.append(
            "lowered engine: the effect is measured at the real post-projection output "
            "site, never written to the reconstructed facet"
        )
    return lines


def render_receipt_grid(
    receipts: list[CausalReceipt] | CausalReceipt,
    path: Path | str,
    *,
    joint_effect: float | None = None,
    svg_fonttype: str = "path",
) -> Artifact:
    """Render the measured-ablation receipt grid (THE differentiator).

    Layers on the y axis (one row per receipt), heads on the x axis; each
    measured cell prints its effect on a zero-centered diverging scale;
    unmeasured cells are blank, never zero. The joint-measurement line, the
    controls, the fire count, and the engine disclosure are printed ON the
    figure.

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
    fig = figure(
        width=0.62 * len(heads) + 2.8,
        height=0.5 * len(rows) + 0.24 * len(disclosures) + 1.7,
    )
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
    for row_index in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row_index, col]
            if np.isfinite(value):
                ax.text(
                    col + 0.5,
                    row_index + 0.5,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
    ax.set_xticks([col + 0.5 for col in range(len(heads))])
    ax.set_xticklabels([f"h{head}" for head in heads], fontsize=7)
    ax.set_yticks([row_index + 0.5 for row_index in range(len(rows))])
    ax.set_yticklabels([receipt.layer for receipt in rows], fontsize=6)
    ax.tick_params(length=0)
    colorbar(fig, mesh, ax, label=f"measured effect on {rows[0].metric}")
    fig.suptitle("measured head-ablation effects", fontsize=9)
    fig.text(0.01, 0.01, "\n".join(disclosures), fontsize=6, va="bottom")
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
