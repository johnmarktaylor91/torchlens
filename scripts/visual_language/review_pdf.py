"""Write the review mode's ``changes.pdf``: one scroll from what changed to the sentinel strip.

Pages, in order: a cover naming both commits and the counts; one page per changed, added or
removed slide with every panel as base and head side by side; one page per sentinel slide,
base and head side by side; one page listing the unchanged slides. Each panel is drawn from
the DOT its build wrote, by Graphviz (``dot -Tpng``) in the folder the build ran in, so
relative image paths resolve as they did there. Matplotlib, part of the ``viz`` extra,
lays out the pages.
"""

from __future__ import annotations

import io
import subprocess
import textwrap
from pathlib import Path
from typing import Any

PAGE = (11.0, 8.5)
ROWS_PER_PAGE = 3
DPI = 72


def panel_png(dot: Path, cwd: Path) -> bytes | None:
    """The panel drawn as PNG, or None when Graphviz cannot draw it."""

    done = subprocess.run(
        ["dot", "-Tpng", f"-Gdpi={DPI}", str(dot)], cwd=cwd, capture_output=True, check=False
    )
    return done.stdout if done.returncode == 0 and done.stdout else None


def _show(ax: Any, png: bytes | None, words: str) -> None:
    import matplotlib.image as mpimg

    ax.set_axis_off()
    if png is None:
        ax.text(0.5, 0.5, words, ha="center", va="center", fontsize=13, color="#555555")
        return
    ax.imshow(mpimg.imread(io.BytesIO(png), format="png"))


def _text_page(pdf: Any, title: str, lines: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=PAGE)
    fig.text(0.04, 0.95, title, fontsize=18, weight="bold", va="top")
    wrapped = [w for line in lines for w in (textwrap.wrap(line, 96) or [""])]
    fig.text(0.04, 0.89, "\n".join(wrapped[:40]), fontsize=12, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def _pair_pages(
    pdf: Any,
    title: str,
    slide_id: str,
    builds: dict[str, Any],
    cwds: dict[str, Path],
    shas: dict[str, str],
) -> None:
    import matplotlib.pyplot as plt

    renders = {side: builds[side][1].get(slide_id) for side in ("base", "head")}
    panels = sorted({p for r in renders.values() if r for p in r.panels})
    if not panels:
        lines = [
            f"{side}: {r.text if r and r.text else 'no picture'}" for side, r in renders.items()
        ]
        _text_page(pdf, f"{title}: {slide_id}", lines)
        return
    for start in range(0, len(panels), ROWS_PER_PAGE):
        chunk = panels[start : start + ROWS_PER_PAGE]
        fig, axes = plt.subplots(len(chunk), 2, figsize=PAGE, squeeze=False)
        fig.suptitle(f"{title}: {slide_id}", fontsize=16, weight="bold")
        for row, panel in enumerate(chunk):
            for col, side in enumerate(("base", "head")):
                render = renders[side]
                entry = render.panels.get(panel) if render else None
                png = panel_png(entry[1], cwds[side]) if entry else None
                missing = f"not in {side}" if render is None else f"panel {panel} not drawn"
                _show(axes[row][col], png, missing)
                axes[row][col].set_title(f"{side} {shas[side][:10]}  panel {panel}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)


def write_changes_pdf(
    path: Path,
    result: dict[str, Any],
    teach: list[Any],
    strips: dict[str, Any],
    strip_trees: dict[str, Path],
    trees: dict[str, Path],
) -> None:
    """Write the whole review as one PDF at ``path``."""

    import matplotlib

    matplotlib.use("Agg")
    # Embedded TrueType text, not Type 3 glyphs, so PDF viewers and checkers read it.
    matplotlib.rcParams["pdf.fonttype"] = 42
    from matplotlib.backends.backend_pdf import PdfPages

    shas = {side: result[side]["sha"] for side in ("base", "head")}
    builds = {"base": teach[0], "head": teach[1]}
    with PdfPages(path) as pdf:
        cover = [
            f"base {result['base']['ref']} = {shas['base']}",
            f"head {result['head']['ref']} = {shas['head']}",
            "",
            f"teaching slides: {len(result['changed'])} changed, {len(result['added'])} added, "
            f"{len(result['removed'])} removed, {len(result['unchanged'])} unchanged",
            f"sentinel strip: {len(result['sentinel']['changed'])} changed of "
            f"{sum(len(result['sentinel'][k]) for k in ('changed', 'added', 'removed', 'unchanged'))}",
            f"changed slides that show measured time (may be a new measurement): "
            f"{', '.join(result['measured']) or 'none'}",
            "slides a build did not finish cleanly: "
            + (
                ", ".join(f"{k} ({v['base']} / {v['head']})" for k, v in result["failed"].items())
                or "none"
            ),
            "sentinel rows a build did not finish cleanly: "
            + (
                ", ".join(
                    f"{k} ({v['base']} / {v['head']})"
                    for k, v in result["sentinel"]["failed"].items()
                )
                or "none"
            ),
            "",
            "Ignored as different by construction (never a change):",
            *(f"  {name}: {why}" for name, why in result["normalization"].items()),
            "",
            f"build exit codes: base deck {result['base']['deck_exit']}, strip "
            f"{result['base']['strip_exit']}; head deck {result['head']['deck_exit']}, strip "
            f"{result['head']['strip_exit']}",
        ]
        _text_page(pdf, "What changed", cover)
        for kind in ("changed", "added", "removed"):
            for slide_id in result[kind]:
                _pair_pages(pdf, f"Slide {kind}", slide_id, builds, trees, shas)
        strip_order = strips["head"][0] or strips["base"][0]
        changed = set(result["sentinel"]["changed"])
        for slide_id in strip_order:
            mark = "changed" if slide_id in changed else "unchanged"
            _pair_pages(pdf, f"Sentinel strip ({mark})", slide_id, strips, strip_trees, shas)
        _text_page(pdf, "Unchanged slides", [", ".join(result["unchanged"]) or "none"])
