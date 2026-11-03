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
            1, 2, figsize=(WIDTH, 4.9), layout="constrained", width_ratios=(12, 1.4)
        )
        image = left.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        left.set(
            xticks=range(matrix.shape[1]),
            xticklabels=[str(i) if i % 2 == 0 else "" for i in range(matrix.shape[1])],
            yticks=range(len(rows)),
            yticklabels=rows,
            xlabel="Layer whose MLP output is restored",
            ylabel="Token position restored",
        )
        left.tick_params(length=0)
        left.tick_params(axis="x", labelsize=SMALL)
        setup = "GPT-2 small"
        if facts and draws:
            setup += f"; {facts} facts, {draws} noise draws on the subject tokens each"
        ate = summary["measured"].get("ate")
        if ate is not None:
            setup += f"\ncorruption removes {ate:.3f} of the answer probability on average"
        setup += "\nrestoring patches one clean-run MLP output into the corrupted run"
        fig.supxlabel(setup, fontsize=SMALL, color=MUTED)
        row = rows.index(category)
        left.add_patch(
            Rectangle((layer - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6)
        )
        low_row, low_layer = np.unravel_index(int(np.argmin(matrix)), matrix.shape)
        low = float(matrix[low_row, low_layer])
        left.add_patch(
            Rectangle(
                (low_layer - 0.5, low_row - 0.5),
                1,
                1,
                fill=False,
                edgecolor=INK,
                linewidth=1.2,
                linestyle=(0, (2, 2)),
            )
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
        right.set_xlabel(f"patched at\nlayer {layer},\n{category}", fontsize=SMALL, color=MUTED)
        bar = fig.colorbar(
            image,
            ax=[left, right],
            label="Average indirect effect\n(answer probability restored minus corrupted)",
            shrink=0.92,
            pad=0.02,
            ticks=[-limit, -limit / 2, 0, limit / 2, limit],
        )
        bar.ax.set_yticklabels([_num(t, "+.4f") if t else "0" for t in bar.get_ticks()])
        fig.suptitle(
            f"Restoring one MLP output recovers the most at layer {layer}, {category};\n"
            "a wrong value there (other layer, other fact) recovers almost nothing;\n"
            f"the dashed cell (layer {low_layer}, {rows[low_row]}) is the most negative: "
            f"{_num(low, '+.4f')}",
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
    pinned = summary["pinned"]
    layer = str(pinned.get("layer", "layer"))
    side = int(image.shape[0])
    height, width = projected.shape
    r0, r1, c0, c1 = _bbox(spatial)
    p0, p1, q0, q1 = _bbox(patch)
    f0, f1, g0, g1 = _bbox(projected)
    # The noise input is shown as faint grey so the measured blue fill and the patch stay visible.
    faint = 0.78 + 0.22 * image.mean(axis=-1)
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 5.6), layout="constrained")
        left.imshow(faint, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        left.imshow(_overlay(empirical, BLUE, 0.45), interpolation="nearest")
        left.add_patch(
            Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, edgecolor=VERMILION, lw=2)
        )
        left.add_patch(Rectangle((q0 - 0.5, p0 - 0.5), q1 - q0, p1 - p0, color=GREEN))
        left.text(
            (q0 + q1 - 1) / 2,
            p0 - 0.03 * side,
            f"{q1 - q0} x {p1 - p0} patch zeroed",
            color=INK,
            fontsize=SMALL,
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": GREEN, "alpha": 0.9, "pad": 1.5},
        )
        left.set(xticks=(0, side // 2, side - 1), yticks=(0, side // 2, side - 1))
        left.tick_params(labelsize=13.5)  # single digits read small
        right.tick_params(which="major", labelsize=13.5)
        left.set_title(
            f"Input ({side} x {side} noise, shown faint)\nfield of one {layer} unit "
            f"(channel {pinned.get('channel', 0)}, centre)",
            loc="left",
        )
        left.set_xlabel("Pixel")
        left.set_ylabel("Pixel")
        right.imshow(_overlay(changed, BLUE, 0.45), interpolation="nearest")
        right.add_patch(
            Rectangle((g0 - 0.5, f0 - 0.5), g1 - g0, f1 - f0, fill=False, edgecolor=VERMILION, lw=2)
        )
        right.text(
            (width - 1) / 2,
            0,
            "outside: unchanged",
            ha="center",
            va="center",
            fontsize=SMALL,
            color=INK,
        )
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
            f"Vermilion outline: the geometric prediction (pixels {c0} to {c1 - 1} on both axes; "
            f"units {g0} to {g1 - 1});\nblue: the measurement (pixels with nonzero gradient; "
            "units changed by zeroing the green patch)",
            fontsize=SMALL,
            color=MUTED,
        )
        fig.suptitle(
            "The geometric fields match the measured influence exactly:\n"
            f"IoU (overlap) {field_iou:.2f} at the input, {projective_iou:.2f} at the output",
            fontsize=TITLE,
        )
        return _save(fig, out, "cnn_fields")


def _site_label(key: str) -> str:
    """Turn a result key such as ``residual_3`` into a short human label."""
    match = re.fullmatch(r"residual_(\d+)", key)
    return f"add {match.group(1)}" if match else key


def draw_neuro_rdms(out: Path) -> list[Path]:
    """Draw the per-stage RDMs and the agreement curves from the saved ``neuro_rdms`` files.

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
    keys = [k for k in stages if k in aligned] + sorted(
        (k for k in aligned if k not in stages), key=lambda k: (len(k), k)
    )
    x = np.arange(len(keys))
    arch_key = str(summary.get("pinned", {}).get("arch", "the network"))
    arch = {"resnet18": "ResNet-18", "vit_b_16": "ViT-B/16"}.get(arch_key, arch_key)
    n_stimuli = int(rdms[0].shape[0])
    top = float(np.ceil(rdms.max() * 10) / 10)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(WIDTH, 6.0), layout="constrained")
        # The square RDM panels fix the top row's height; the curve row is a multiple of it.
        spec = fig.add_gridspec(
            2, len(rdms) + 1, height_ratios=(1.0, 2.6), width_ratios=[1.0] * len(rdms) + [0.07]
        )
        panels = [fig.add_subplot(spec[0, i]) for i in range(len(rdms))]
        for panel, stage, rdm in zip(panels, stages, rdms, strict=True):
            image = panel.imshow(rdm, cmap="cividis", vmin=0, vmax=top, interpolation="nearest")
            panel.set_title(stage)
            panel.set(xticks=[], yticks=[])
        panels[0].set_ylabel("TorchLens\nRDM")
        bar = fig.colorbar(
            image,
            cax=fig.add_subplot(spec[0, -1]),
            label="Dissimilarity\n(1 − r)",
            ticks=np.linspace(0, top, 4),
        )
        bar.ax.tick_params(labelsize=13.5)  # single digits read small
        curve = fig.add_subplot(spec[1, :])
        curve.set_title(
            f"{arch}; {n_stimuli} Fashion-MNIST images, so each RDM (representational "
            f"dissimilarity matrix)\nis {n_stimuli} x {n_stimuli}; the {len(rdms)} stage RDMs "
            "above are TorchLens's",
            loc="left",
            fontsize=SMALL,
            color=MUTED,
        )
        curve.plot(x, [aligned[k] for k in keys], "o-", color=INK, lw=2, ms=5)
        curve.plot(x, [shuffled[k] for k in keys], "s--", color=BLUE, lw=2, ms=5)
        curve.text(
            x.mean(),
            min(aligned.values()) - 0.05,
            f"aligned order: pass bar {correlation_min:g}",
            ha="center",
            va="top",
        )
        curve.text(
            x.mean(),
            min(shuffled.values()) - 0.06,
            "one side shuffled",
            ha="center",
            va="top",
            color=BLUE,
        )
        curve.plot(
            [-0.6, x[-1] + 0.4],
            [shuffled_max, shuffled_max],
            color=MUTED,
            linestyle="--",
            lw=1.4,
            zorder=0,
        )
        curve.text(
            -0.4,
            shuffled_max - 0.03,
            f"control bar {shuffled_max:g}: shuffled r must stay below it",
            va="top",
            fontsize=SMALL,
            color=MUTED,
        )
        curve.axhline(0, color=RULE, lw=0.7, zorder=0)
        labels = [_site_label(k) for k in keys]
        curve.set(
            xticks=x,
            xticklabels=[label if i % 2 == 0 else f"\n{label}" for i, label in enumerate(labels)],
            xlim=(-0.6, len(keys) - 0.4),
            ylim=(-0.4, 1.08),
            yticks=(-0.25, 0, 0.25, 0.5, 0.75, 1.0),
            ylabel="Pearson r between the TorchLens RDM\nand the plain-hooks RDM",
            xlabel="Site: the selected stages, then each residual addition in execution order",
        )
        curve.tick_params(axis="x", labelsize=SMALL)
        curve.tick_params(axis="y", labelsize=13.5)
        curve.grid(axis="y")
        fig.suptitle(
            f"All {len(keys)} sites match the plain-hooks RDMs (r rounds to "
            f"{min(aligned.values()):.3f}; bar {correlation_min:g});\n"
            "shuffling one side's stimulus order drops r to "
            f"{_num(max(shuffled.values()), '.3f')} at most",
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
        ax.plot(roc[:, 0], roc[:, 1], color=BLUE, lw=2.2, zorder=2)
        ax.scatter([data["fpr_at_k"]], [data["recall_at_k"]], s=48, color=VERMILION, zorder=3)
        ax.annotate(
            f"cutoff at the top {data['k']} heads:\n{data['recall_at_k']:.0%} of Wang's {positives}",
            (data["fpr_at_k"], data["recall_at_k"]),
            xytext=(data["fpr_at_k"] + 0.02, data["recall_at_k"] - 0.15),
            color=INK,
            va="top",
            arrowprops={"arrowstyle": "-", "color": VERMILION, "lw": 0.8},
        )
        ax.set(
            xlim=(-0.005, 1.005),
            ylim=(0, 1.03),
            xlabel=f"False positive rate (share of the other {heads - positives} heads above the cutoff)",
            ylabel=f"True positive rate\n(share of Wang's {positives} heads above the cutoff)",
        )
        ax.grid(axis="y")
        setup = "GPT-2 small, one run"
        if "rows" in data and "steps" in data:
            setup = f"GPT-2 small, {data['rows']:,} IOI rows, {data['steps']} integration steps, one run"
        setup += ";\ndotted diagonal: chance"
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
    from matplotlib.patheffects import withStroke

    groups: dict[str, list[list[int]]] = data["groups"]
    edges: list[dict[str, Any]] = data["edges"]
    layers, per_layer = int(data["layers"]), int(data["heads_per_layer"])
    colors = (BLUE, VERMILION, GREEN, PINK, ORANGE, SKY, BLACK)
    class_of = {tuple(pair): label for label, pairs in groups.items() for pair in pairs}
    ends = [class_of.get(tuple(edge["target"]), "other") for edge in edges]
    top_class = max(set(ends), key=lambda label: (ends.count(label), label))
    joined = sum(
        1 for e in edges if tuple(e["source"]) in class_of and tuple(e["target"]) in class_of
    )
    sizes = [abs(float(edge["score"])) for edge in edges]
    low, high = min(sizes), max(sizes)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(WIDTH + 0.2, 5.8), layout="constrained")
        for edge, start, end, rad in _edge_geometry(edges):
            share = (abs(float(edge["score"])) - low) / (high - low) if high > low else 1.0
            width = 0.9 + 2.0 * share
            ax.annotate(
                "",
                end,
                start,
                arrowprops={
                    "arrowstyle": "-|>,head_length=0.5,head_width=0.25",
                    "color": MUTED,
                    "lw": width,
                    "shrinkA": 0,
                    "shrinkB": 0,
                    "connectionstyle": f"arc3,rad={rad:.2f}",
                    "path_effects": [withStroke(linewidth=width + 2.4, foreground="white")],
                },
                zorder=4,
            )
        ax.scatter(
            [layer for layer in range(layers) for _ in range(per_layer)],
            list(range(per_layer)) * layers,
            s=16,
            color="#e2e2de",
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
        fig.legend(
            loc="outside lower center",
            ncol=4,
            fontsize=SMALL,
            columnspacing=2.4,
            title=f"Wang's head classes (grey: other heads); arrows: the {len(edges)} largest "
            f"absolute edge scores,\nwidth by |score| ({low:.2f} thinnest to {high:.2f} thickest)",
            title_fontsize=SMALL,
        )
        count = "All" if joined == len(edges) else f"{joined} of the"
        fig.suptitle(
            f"{count} {len(edges)} strongest head-to-head edges join Wang's head classes;\n"
            f"{ends.count(top_class)} of them end at {top_class} heads",
            fontsize=TITLE,
        )
        return _save(fig, out, "ioi_circuit")


def _edge_geometry(
    edges: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], tuple[float, float], tuple[float, float], float]]:
    """Return ``(edge, start, end, curvature)`` with the ends spread around each head.

    Arrows into one head leave and land at distinct points on a small circle around the
    dot, and their curvatures differ, so the arrowheads stay apart and a reader can count
    them. Arrows are drawn above the dots with a white halo, so a line passing a head it
    does not touch stays visibly continuous.
    """
    import math

    fan_in: dict[tuple[int, int], int] = {}
    fan_out: dict[tuple[int, int], int] = {}
    for edge in edges:
        fan_in[tuple(edge["target"])] = fan_in.get(tuple(edge["target"]), 0) + 1
        fan_out[tuple(edge["source"])] = fan_out.get(tuple(edge["source"]), 0) + 1
    seen_in: dict[tuple[int, int], int] = {}
    seen_out: dict[tuple[int, int], int] = {}
    geometry = []
    for edge in edges:
        source, target = tuple(edge["source"]), tuple(edge["target"])
        rank_in = seen_in.get(target, 0)
        rank_out = seen_out.get(source, 0)
        seen_in[target], seen_out[source] = rank_in + 1, rank_out + 1
        offset_in = 0.6 * (rank_in - (fan_in[target] - 1) / 2)
        offset_out = 0.6 * (rank_out - (fan_out[source] - 1) / 2)
        theta = math.atan2(source[1] - target[1], source[0] - target[0])
        end = (
            target[0] + 0.32 * math.cos(theta + offset_in),
            target[1] + 0.32 * math.sin(theta + offset_in),
        )
        start = (
            source[0] - 0.26 * math.cos(theta - offset_out),
            source[1] - 0.26 * math.sin(theta - offset_out),
        )
        rad = 0.18 * (rank_in - (fan_in[target] - 1) / 2) + 0.12 * (
            rank_out - (fan_out[source] - 1) / 2
        )
        geometry.append((edge, start, end, rad))
    return geometry
