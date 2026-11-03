"""Draw the Gallery figures from each entry's saved result files.

Each entry script saves what its figure needs in its ``--out`` folder: the stamped
``<entry>.json`` summary and, where arrays or derived tables are involved,
``<entry>_figure.npz`` or ``<entry>_figure.json``. The ``draw_*`` functions here read
those files and write ``<name>.png`` (1,520 pixels wide, the documentation display width)
and ``<name>.svg`` beside them, so a figure is redrawn without rerunning the model
(``python examples/gallery/<entry>.py --out DIR --figure-only``). Only numpy and
matplotlib are imported.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

WIDTH = 7.6  # inches; 1,520 pixels at DPI
DPI = 200
INK = "#1f1f1e"
MUTED = "#555650"
RULE = "#e6e5e0"
BLUE = "#0072b2"
VERMILION = "#d55e00"
GREEN = "#009e73"
PINK = "#cc79a7"
ORANGE = "#e69f00"
SKY = "#56b4e9"
BLACK = "#000000"
GREY = "#8a8a86"
# Text sizes for the pane target (7.6 in at 200 dpi, shown at half size): digits in tick
# labels read at about two thirds of the em height, so ticks sit above the 11 CSS px floor.
SMALL = 11.5
TITLE = 13.5
STYLE = {
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "grid.color": RULE,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}
TOKEN_CATEGORIES = (
    "first subject token",
    "middle subject tokens",
    "last subject token",
    "first subsequent token",
    "further tokens",
    "last token",
)


def _load(path: Path) -> dict[str, Any]:
    """Read one JSON result file."""
    return json.loads(Path(path).read_text("utf-8"))


def _num(value: float, spec: str) -> str:
    """Format a number with a true minus sign."""
    return format(value, spec).replace("-", "−")


def _first_int(text: str, pattern: str) -> int | None:
    """Return the first integer captured by ``pattern`` in ``text``, or None."""
    match = re.search(pattern, str(text))
    return int(match.group(1)) if match else None


def _save(fig: Any, out: Path, stem: str) -> list[Path]:
    """Write ``stem.png`` and ``stem.svg`` to ``out`` and close the figure."""
    import matplotlib.pyplot as plt

    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(w_pad=0.02, h_pad=0.02)
    paths = []
    for suffix in ("png", "svg"):
        path = Path(out) / f"{stem}.{suffix}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _overlay(mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    """Return an RGBA image that paints ``mask`` in ``color`` and leaves the rest clear."""
    from matplotlib.colors import to_rgb

    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[..., :3] = to_rgb(color)
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    return rgba


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(row0, row1, col0, col1)`` of a mask's bounding box (end-exclusive)."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def draw_causal_tracing(out: Path) -> list[Path]:
    """Draw the AIE heatmap and the wrong-site controls from ``causal_tracing.json``.

    Parameters
    ----------
    out:
        The entry's output folder.

    Returns
    -------
    list[Path]
        The PNG and SVG written.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out = Path(out)
    summary = _load(out / "causal_tracing.json")
    grid = summary["measured"]["aie_grid_single_layer"]
    rows = [c for c in TOKEN_CATEGORIES if c in grid] + sorted(set(grid) - set(TOKEN_CATEGORIES))
    matrix = np.array([grid[c] for c in rows], dtype=float)
    peak = summary["oracles"]["paper_rule"]["measured"]
    wrong_layer = summary["controls"]["wrong_layer"]["measured"]
    wrong_fact = summary["controls"]["wrong_fact"]["measured"]
    column = np.array([[peak["peak_aie"]], [wrong_layer["aie"]], [wrong_fact["aie"]]])
    limit = float(max(np.abs(matrix).max(), np.abs(column).max()))
    layer, category = int(peak["peak_layer"]), str(peak["peak_category"])
    facts = _first_int(summary.get("pinned", {}).get("facts", ""), r"first (\d+)")
    draws = _first_int(summary.get("pinned", {}).get("noise", ""), r"(\d+) samples")
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(WIDTH, 4.3), layout="constrained", width_ratios=(12, 1.4)
        )
        image = left.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        left.set(
            xticks=range(matrix.shape[1]),
            yticks=range(len(rows)),
            yticklabels=rows,
            xlabel="Layer whose MLP output is restored",
            ylabel="Token position restored",
        )
        left.tick_params(length=0)
        left.tick_params(axis="x", labelsize=SMALL)
        setup = "GPT-2 small"
        if facts and draws:
            setup += f"; {facts} facts, {draws} noise draws each"
        ate = summary["measured"].get("ate")
        if ate is not None:
            setup += f"\ncorruption removes {ate:.3f} of the answer probability"
        left.set_title(setup, loc="left", fontsize=SMALL, color=MUTED)
        row = rows.index(category)
        left.add_patch(
            Rectangle((layer - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6)
        )
        left.text(
            layer + 0.65,
            row,
            f"peak {_num(peak['peak_aie'], '+.4f')}",
            va="center",
            ha="left",
            fontsize=SMALL,
            color=INK,
        )
        right.imshow(column, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        names = (
            f"clean value\nfrom layer {layer}",
            f"value from\nlayer {int(wrong_layer['wrong_layer'])}",
            "value from\nanother fact",
        )
        right.set(
            xticks=[],
            yticks=range(3),
            yticklabels=[
                f"{name}\n{_num(value, '+.4f')}" for name, value in zip(names, column[:, 0])
            ],
        )
        right.tick_params(length=0)
        fig.colorbar(
            image,
            ax=[left, right],
            label="Average indirect effect (answer probability recovered)",
            shrink=0.92,
            pad=0.02,
        )
        fig.suptitle(
            f"Restoring one MLP output recovers the most at layer {layer}, {category};\n"
            "restoring a wrong value there (other layer, other fact) recovers nothing",
            fontsize=TITLE,
        )
        return _save(fig, out, "causal_tracing")


def draw_cnn_fields(out: Path) -> list[Path]:
    """Draw the input field and the projective field from the saved ``cnn_fields`` files.

    Parameters
    ----------
    out:
        The entry's output folder, holding ``cnn_fields.json`` and ``cnn_fields_figure.npz``.

    Returns
    -------
    list[Path]
        The PNG and SVG written.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out = Path(out)
    summary = _load(out / "cnn_fields.json")
    arrays = np.load(out / "cnn_fields_figure.npz")
    image, spatial, empirical = arrays["image"], arrays["spatial"], arrays["empirical"]
    projected, changed, patch = arrays["projected"], arrays["changed"], arrays["patch"]
    oracles = summary["oracles"]
    field_iou = float(oracles["gradient_support"]["measured"])
    projective_iou = float(oracles["projective_support"]["measured"])
    layer = str(summary["pinned"].get("layer", "layer"))
    side = int(image.shape[0])
    height, width = projected.shape
    r0, r1, c0, c1 = _bbox(spatial)
    p0, p1, q0, q1 = _bbox(patch)
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 5.4), layout="constrained")
        left.imshow(image, interpolation="nearest")
        left.imshow(_overlay(empirical, BLUE, 0.35), interpolation="nearest")
        left.add_patch(
            Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, edgecolor=VERMILION, lw=2)
        )
        left.add_patch(Rectangle((q0 - 0.5, p0 - 0.5), q1 - q0, p1 - p0, color=GREEN))
        left.text(
            q1 + 0.03 * side,
            p0 - 0.015 * side,
            f"{q1 - q0} x {p1 - p0} patch",
            color=GREEN,
            fontsize=SMALL,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )
        left.set(xticks=(0, side // 2, side - 1), yticks=(0, side // 2, side - 1))
        left.set_title(f"Input ({side} x {side} noise)\nfield of one {layer} unit", loc="left")
        left.set_xlabel("Pixel")
        left.set_ylabel("Pixel")
        right.imshow(_overlay(projected, BLUE, 0.35), interpolation="nearest")
        ys, xs = np.nonzero(changed)
        right.scatter(xs, ys, s=16, color=VERMILION, zorder=3)
        right.set_xticks(np.arange(-0.5, width, 1), minor=True)
        right.set_yticks(np.arange(-0.5, height, 1), minor=True)
        right.grid(which="minor", color=RULE, linewidth=0.6)
        right.tick_params(which="minor", length=0)
        right.set(
            xticks=range(0, width, 4),
            yticks=range(0, height, 4),
            xlim=(-0.5, width - 0.5),
            ylim=(height - 0.5, -0.5),
        )
        right.set_title(
            f"{layer} output ({height} x {width} units)\nreach of the green patch", loc="left"
        )
        right.set_xlabel("Feature-map unit")
        right.set_ylabel("Feature-map unit")
        fig.supxlabel(
            f"Outline: the geometric box [{c0}, {c1}); blue: gradient support (left) and "
            "projective field (right);\ndots: units changed by zeroing the green patch",
            fontsize=SMALL,
            color=MUTED,
        )
        fig.suptitle(
            "The geometric fields match the measured influence exactly:\n"
            f"IoU {field_iou:.2f} at the input, {projective_iou:.2f} at the output",
            fontsize=TITLE,
        )
        return _save(fig, out, "cnn_fields")


def _site_label(key: str) -> str:
    """Turn a result key such as ``residual_3`` into a short human label."""
    match = re.fullmatch(r"residual_(\d+)", key)
    return f"add {match.group(1)}" if match else key


def draw_neuro_rdms(out: Path) -> list[Path]:
    """Draw the per-stage RDMs and the aligned-versus-shuffled correlation curves.

    Parameters
    ----------
    out:
        The entry's output folder, holding ``neuro_rdms.json`` and ``neuro_rdms_figure.npz``.

    Returns
    -------
    list[Path]
        The PNG and SVG written.
    """
    import matplotlib.pyplot as plt

    out = Path(out)
    summary = _load(out / "neuro_rdms.json")
    arrays = np.load(out / "neuro_rdms_figure.npz")
    rdms, stages = arrays["rdms"], [str(s) for s in arrays["stages"]]
    correlation_min = float(arrays["correlation_min"])
    shuffled_max = float(arrays["shuffled_max"])
    aligned = summary["oracles"]["per_layer_rdm"]["measured"]
    shuffled = summary["controls"]["shuffled_order"]["measured"]
    keys = list(aligned)
    x = np.arange(len(keys))
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(WIDTH, 5.8), layout="constrained")
        spec = fig.add_gridspec(2, len(rdms), height_ratios=(0.75, 1.25))
        panels = [fig.add_subplot(spec[0, i]) for i in range(len(rdms))]
        for panel, stage, rdm in zip(panels, stages, rdms, strict=True):
            image = panel.imshow(rdm, cmap="cividis", vmin=0, vmax=2, interpolation="nearest")
            panel.set_title(stage)
            panel.set(xticks=(0, rdm.shape[0] - 1), yticks=(0, rdm.shape[0] - 1))
            panel.tick_params(length=0, labelsize=13.5)  # single digits read small
        panels[0].set_ylabel("Stimulus")
        fig.colorbar(image, ax=panels, label="Correlation distance (1 − r)", shrink=0.9, pad=0.02)
        curve = fig.add_subplot(spec[1, :])
        curve.plot(x, [aligned[k] for k in keys], "o-", color=INK, lw=2, ms=5)
        curve.plot(x, [shuffled[k] for k in keys], "s--", color=BLUE, lw=2, ms=5)
        curve.text(x[-1] + 0.3, aligned[keys[-1]], "aligned stimulus order", va="center")
        curve.text(x[-1] + 0.3, shuffled[keys[-1]], "one side shuffled", va="center", color=BLUE)
        for level, label in ((correlation_min, "pass bar"), (shuffled_max, "control bar")):
            curve.plot(
                [-0.6, x[-1] + 0.2], [level, level], color=GREY, linestyle=":", lw=1, zorder=0
            )
            curve.text(
                -0.4, level - 0.03, f"{label} {level:g}", va="top", fontsize=SMALL, color=GREY
            )
        curve.axhline(0, color=RULE, lw=0.7, zorder=0)
        curve.set(
            xticks=x,
            xticklabels=[_site_label(k) for k in keys],
            xlim=(-0.6, len(keys) + 2.6),
            ylim=(-0.25, 1.12),
            ylabel="Pearson r with the plain-hooks RDM",
            xlabel="Site: the selected stages, then each residual addition in execution order",
        )
        curve.tick_params(axis="x", labelsize=SMALL, rotation=30)
        curve.grid(axis="y")
        fig.suptitle(
            f"All {len(keys)} sites match the plain-hooks RDMs (r ≥ {min(aligned.values()):.3f});\n"
            f"shuffling one side's stimulus order drops r to {_num(max(shuffled.values()), '.2f')} or less",
            fontsize=TITLE,
        )
        return _save(fig, out, "neuro_rdms")


def draw_ioi_circuit(out: Path) -> list[Path]:
    """Draw the head ROC curve and the strongest head-to-head edges.

    Parameters
    ----------
    out:
        The entry's output folder, holding ``ioi_circuit_figure.json``.

    Returns
    -------
    list[Path]
        The PNGs and SVGs written (``ioi_roc`` then ``ioi_circuit``).
    """
    out = Path(out)
    data = _load(out / "ioi_circuit_figure.json")
    return _draw_ioi_roc(data, out) + _draw_ioi_edges(data, out)


def _draw_ioi_roc(data: dict[str, Any], out: Path) -> list[Path]:
    """Draw the ROC curve of head ranking against Wang membership."""
    import matplotlib.pyplot as plt

    positives, heads = int(data["positives"]), int(data["heads"])
    roc = np.array(data["roc"], dtype=float)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH, 4.9), layout="constrained")
        ax.plot([0, 1], [0, 1], color=GREY, linestyle=":", lw=1, zorder=1)
        ax.text(0.72, 0.68, "chance", color=GREY, rotation=38, rotation_mode="anchor")
        ax.plot(roc[:, 0], roc[:, 1], color=BLUE, lw=2.2, zorder=2)
        ax.scatter([data["fpr_at_k"]], [data["recall_at_k"]], s=48, color=VERMILION, zorder=3)
        ax.annotate(
            f"top {data['k']} heads: {data['recall_at_k']:.0%} of Wang's {positives}",
            (data["fpr_at_k"], data["recall_at_k"]),
            xytext=(data["fpr_at_k"] + 0.06, data["recall_at_k"] - 0.1),
            color=VERMILION,
            arrowprops={"arrowstyle": "-", "color": VERMILION, "lw": 0.8},
        )
        ax.set(
            xlim=(-0.005, 1.005),
            ylim=(0, 1.03),
            xlabel="False positive rate (share of the other heads ranked above)",
            ylabel="True positive rate (share of Wang's heads recovered)",
        )
        ax.grid(axis="y")
        setup = "GPT-2 small, one run"
        if "rows" in data and "steps" in data:
            setup = f"GPT-2 small, {data['rows']:,} IOI rows, {data['steps']} integration steps, one run"
        ax.set_title(setup, loc="left", color=MUTED)
        fig.suptitle(
            f"Summed outgoing attribution separates Wang's {positives} IOI heads\n"
            f"from the other {heads - positives} heads: ROC AUC {data['auc']:.3f}",
            fontsize=TITLE,
        )
        return _save(fig, out, "ioi_roc")


def _draw_ioi_edges(data: dict[str, Any], out: Path) -> list[Path]:
    """Draw the strongest directed head-to-head edges with Wang's classes coloured."""
    import matplotlib.pyplot as plt

    groups: dict[str, list[list[int]]] = data["groups"]
    edges: list[dict[str, Any]] = data["edges"]
    layers, per_layer = int(data["layers"]), int(data["heads_per_layer"])
    colors = (BLUE, VERMILION, GREEN, PINK, ORANGE, SKY, BLACK)
    class_of = {tuple(pair): label for label, pairs in groups.items() for pair in pairs}
    ends = [class_of.get(tuple(edge["target"]), "other") for edge in edges]
    top_class = max(set(ends), key=lambda label: (ends.count(label), label))
    scale = max((abs(edge["score"]) for edge in edges), default=1.0)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH + 0.2, 5.8), layout="constrained")
        for edge in edges:
            ax.annotate(
                "",
                tuple(edge["target"]),
                tuple(edge["source"]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": GREY,
                    "alpha": 0.7,
                    "lw": 0.5 + 2.2 * abs(edge["score"]) / scale,
                },
                zorder=1,
            )
        ax.scatter(
            [layer for layer in range(layers) for _ in range(per_layer)],
            list(range(per_layer)) * layers,
            s=16,
            color="#dddddd",
            zorder=2,
        )
        for (label, pairs), color in zip(groups.items(), colors, strict=False):
            ax.scatter(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                s=46,
                color=color,
                label=label,
                zorder=3,
            )
        ax.set(
            xticks=range(layers),
            yticks=range(per_layer),
            xlabel="Layer (zero-based)",
            ylabel="Head index (zero-based)",
            xlim=(-0.5, layers - 0.5),
            ylim=(-0.5, per_layer - 0.5),
        )
        ax.tick_params(length=0, labelsize=13.5)  # single digits read small
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=4,
            fontsize=SMALL,
            title="Wang's head classes (grey: other heads); arrow width: edge score size",
            title_fontsize=SMALL,
        )
        fig.suptitle(
            f"The {len(edges)} strongest head-to-head edges: {ends.count(top_class)} of them end at "
            f"{top_class} heads",
            fontsize=TITLE,
        )
        return _save(fig, out, "ioi_circuit")
