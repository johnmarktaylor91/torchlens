"""The fixed top-left block of a forward picture: the header, then the key.

The header names the model, counts what is drawn and gives the parameter
total. The key sits directly under it, inside the same graph label, so the
two always share the top-left corner: a top crop of a big model keeps both,
and Graphviz can never float the key to another corner of the page (it was a
separate legend node before, placed wherever the layout had room).

The key draws each mark with its real shape and fill: ovals for single
function calls (input and output are function calls too), boxes for leaf
modules, cylinders for buffers; grey fill means weights on any shape, light
for trainable and dark for frozen. Graphviz HTML labels can only draw
rectangles, so the swatches are small drawn PNG images in a shared,
content-addressed folder under the system temp directory, referenced by
absolute path so the DOT source renders from any working directory. Rows
appear only for marks the picture contains; the key is opt-in
(``show_legend=True``).

Spellings here are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import atexit
import functools
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._label_format import format_memory
from ._render_utils import html_escape

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ._legend import LegendSection
    from .node_universe import NodeUniverse
    from .request import ResolvedRenderRequest
    from .themes import VisualizationTheme

__all__ = [
    "KEY_TABLE_ID",
    "KeySwatch",
    "caption_from_label",
    "drawn_module_marks",
    "drawn_count_line",
    "graph_caption_body",
    "header_block_label",
    "key_swatches",
    "params_caption_detail",
]

#: HTML table ID of the key row (also the SVG group id); tests and audits key on it.
KEY_TABLE_ID = "tl_key"
#: Swatch size in points, and the supersampling factor of the drawn PNG.
_SWATCH_WIDTH = 22
_SWATCH_HEIGHT = 12
_SWATCH_SCALE = 4
#: Bumped when the drawing changes, so cached swatches from older code are not reused.
_SWATCH_VERSION = 1
#: Cell spacing inside the key table, in points.
_KEY_SPACING = 3
#: Swatches per key row before the row wraps.
_SWATCHES_PER_ROW = 3
#: Stroke of the key's module frame, in points (module frames run 2 to 5 pt by depth).
_MODULE_FRAME_LINE = 3.0
#: Node unit kinds the header counts: each draws one node with a tensor size.
_COUNTED_UNIT_KINDS = frozenset({"raw_op", "boundary", "module_box"})


@dataclass(frozen=True)
class KeySwatch:
    """One key entry: a mark drawn with its real shape and fill, plus a word.

    Attributes
    ----------
    text:
        Plain words for the mark (escaped at build time).
    shape:
        ``"oval"``, ``"box"`` or ``"cylinder"``; ``"frame"`` for a module box
        and ``"arrow"`` for an arrow with the downstream highlighter under it.
    fill:
        Fill colour as the picture paints it under the active skin (for
        ``"arrow"``, the highlighter colour with its opacity, ``#RRGGBBAA``).
    border:
        Outline colour as the picture paints it.
    line:
        Outline width in points.
    """

    text: str
    shape: str
    fill: str
    border: str
    line: float = 1.0


def params_caption_detail(trace: Trace) -> str:
    """Return the header's parameter line, for example ``12 params (48 B, all trainable)``."""

    if trace.num_params == 0:
        return "0 params"
    memory = trace.total_param_memory
    if trace.num_params_frozen == 0:
        return f"{trace.num_params} params ({memory}, all trainable)"
    if trace.num_params_trainable == 0:
        return f"{trace.num_params} params ({memory}, all frozen)"
    return (
        f"{trace.num_params} params "
        f"({memory}, {trace.num_params_trainable}/{trace.num_params} trainable)"
    )


def _unit_memory(trace: Trace, unit: Any) -> int | None:
    """Return the byte size the unit's node label shows, or ``None`` when unknown."""

    emission = unit.emission
    node = emission.node if emission.node is not None else next(iter(unit.source_nodes), None)
    if unit.kind == "module_box" and emission.call is not None:
        # A collapsed module box shows its module's output tensor
        # (``_render_nodes`` module box label); fall back to the node as it does.
        try:
            module_call = trace.module_calls[emission.call]
            node = trace.ops[module_call.output_ops[-1]]
        except (KeyError, IndexError, AttributeError, TypeError):
            pass
    value = getattr(node, "activation_memory", None)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def drawn_count_line(trace: Trace, universe: NodeUniverse, vis_mode: str) -> str:
    """Return the header's count line, for example ``4 ops (48 B)``.

    Every drawn node counts, input and output included, and the size sums
    the sizes their labels show, so the line matches the picture. The unit
    is ``layers`` in the rolled view and ``ops`` in the unrolled view.
    Repeat-fold ellipses, segment capsules and container summaries carry no
    tensor size of their own and are not counted.
    """

    count = 0
    total = 0
    for unit in universe.units:
        if unit.kind not in _COUNTED_UNIT_KINDS:
            continue
        count += 1
        total += _unit_memory(trace, unit) or 0
    if vis_mode == "rolled":
        noun = "layer" if count == 1 else "layers"
    else:
        noun = "op" if count == 1 else "ops"
    return f"{count} {noun} ({format_memory(total)})"


def graph_caption_body(trace: Trace, count_line: str, request: ResolvedRenderRequest) -> str:
    """Assemble the header body: model, counts, edits, honesty banner, and disclosures.

    Returns
    -------
    str
        HTML-like caption body without the outer FONT wrapper.
    """

    from .._capture_honesty import honesty_banner_lines
    from ._render_utils import strip_path_directories

    caption_body = (
        f"<B>{html_escape(trace.model_class_name)}</B>"
        f"<br align='left'/>{html_escape(count_line)}"
        f"<br align='left'/>{params_caption_detail(trace)}<br align='left'/>"
    )
    edit_line = _header_edit_line(trace, request)
    if edit_line is not None:
        caption_body += f"{edit_line}<br align='left'/>"
    if getattr(trace, "_has_direct_writes", False):
        caption_body += "Direct writes detected - recipe propagation will overlay<br align='left'/>"
    # Capture-honesty banner (WT1 A-V row 23): a poisoned/diverged sparse run,
    # an unverified capture, or an episode capture must not RENDER clean --
    # the graph is a publishable artifact, so the caption carries the facts.
    # The picture keeps a cited source location's file name and line only.
    for banner_line in honesty_banner_lines(trace):
        banner_text = html_escape(strip_path_directories(banner_line))
        caption_body += f"<B>{banner_text}</B><br align='left'/>"
    encoding_state = getattr(request, "encoding", None)
    if encoding_state is not None and getattr(encoding_state, "stack_spec", None) is not None:
        # Stacking disclosure is part of the contract (memo 4.1): the
        # rendered output captions which annotation produced the columns.
        caption_body += (
            f"stacked by: {html_escape(encoding_state.stack_spec.display_name)}<br align='left'/>"
        )
    # No silent floor (collapse memo D4): a floor-fallback plan renders with a
    # VISIBLE notice, never as if the optimizer had chosen it.
    collapse_result = getattr(request.collapse_fn, "_torchlens_v2_result", None)
    if collapse_result is not None and getattr(collapse_result, "planner", "") == "floor_fallback":
        floor_note = (
            f"collapse fell back to the floor plan ({collapse_result.visible_count} nodes visible)"
        )
        if getattr(collapse_result, "k_cap_exhausted", False):
            floor_note += f" -- root owns {collapse_result.root_own_units} units, above K_CAP"
        caption_body += f"<B>{html_escape(floor_note)}</B><br align='left'/>"
    if getattr(request.collapse_fn, "_torchlens_overview_reason", None) is not None:
        caption_body += (
            "<B>collapse overview: top-level modules only (trace past the smart-collapse "
            "ceiling); see the expansion manifest</B><br align='left'/>"
        )
    reuse_note = module_reuse_note(trace, request)
    if reuse_note is not None:
        caption_body += f"{html_escape(reuse_note)}<br align='left'/>"
    return caption_body


def _header_edit_line(trace: Trace, request: ResolvedRenderRequest) -> str | None:
    """Return the header's edit line (``1 edit: Δ zero_ablate on linear_1_1``), if any."""

    from ._edit_marks import edit_ink, header_edit_line
    from .node_spec import intervention_sites_for_log
    from .themes import resolve_theme

    sites = intervention_sites_for_log(trace)
    if not sites:
        return None
    theme = resolve_theme(request.theme, for_paper=request.for_paper)
    color = str((request.graph_overrides or {}).get("intervention_site_color", edit_ink(theme)))
    return header_edit_line(trace, sites, color, vis_mode=request.vis_mode)


def module_reuse_note(trace: Trace, request: ResolvedRenderRequest) -> str | None:
    """Disclose module reuse on an unrolled overview, never switching modes (collapse 5).

    The collapse panel ratified "disclose, don't switch" for weight-shared
    models: the default ``vis_mode`` stays, and an overview render (a smart
    collapse mode) says how many modules ran more than once and that rolled
    mode draws the loop. Nested modules of a reused module are not counted
    again.

    Returns
    -------
    str | None
        The note, or ``None`` when the render is rolled, not an overview, or
        no module ran more than once.
    """

    overview = request.collapse in ("auto", "max") or (
        isinstance(request.collapse, float) and request.collapse > 0.0
    )
    if request.vis_mode != "unrolled" or not overview:
        return None
    reused = {
        str(module.address): int(module.num_calls)
        for module in trace.modules
        if str(getattr(module, "address", "")) not in ("", "self")
        and int(getattr(module, "num_calls", 1) or 1) > 1
    }
    outermost = {
        address: calls
        for address, calls in reused.items()
        if not any(address.startswith(f"{other}.") for other in reused)
    }
    if not outermost:
        return None
    count = len(outermost)
    calls = sum(outermost.values())
    noun = "module" if count == 1 else "modules"
    return (
        f"this model reuses {count} {noun} across {calls} calls; vis_mode='rolled' shows the loop"
    )


def drawn_module_marks(node_attrs: Iterable[Mapping[str, Any]], has_regions: bool) -> bool:
    """Return whether the picture draws a module: a module box node or a nested-module box."""

    if has_regions:
        return True
    for attrs in node_attrs:
        if "invis" in str(attrs.get("style", "")):
            continue
        if str(attrs.get("shape", "")).lower() in {"box", "box3d"}:
            return True
    return False


def key_swatches(
    theme: VisualizationTheme,
    drawn: frozenset[str],
    *,
    has_module: bool,
    mutated_parameter: bool = False,
) -> tuple[KeySwatch, ...]:
    """Return the key entries for the marks this picture draws, in display order.

    Both weight greys are listed whenever any weighted node is drawn, so a
    reader can tell trainable from frozen weights. Plain operations get no
    entry (they are the page colour). A drawn module gets a module frame
    swatch; an edited node adds ``edited`` (the ringed oval) and, with the
    downstream highlighter drawn, ``changed by the edit`` (an arrow over the
    translucent purple bar).
    """

    from ._render_common import (
        BOOL_NODE_COLOR,
        BUFFER_NODE_COLOR,
        DEFAULT_BG_COLOR,
        FROZEN_PARAMS_BG_COLOR,
        INPUT_COLOR,
        OUTPUT_COLOR,
        TRAINABLE_PARAMS_BG_COLOR,
    )
    from .node_spec import NodeSpec
    from .themes import apply_theme_to_spec

    has_weights = "parameterized" in drawn
    entries = (
        ("input", "oval", INPUT_COLOR, "input" in drawn),
        ("output", "oval", OUTPUT_COLOR, "output" in drawn),
        ("trainable weights", "box", TRAINABLE_PARAMS_BG_COLOR, has_weights),
        ("frozen weights", "box", FROZEN_PARAMS_BG_COLOR, has_weights),
        ("module", "frame", DEFAULT_BG_COLOR, has_module),
        ("buffer", "cylinder", BUFFER_NODE_COLOR, "buffer" in drawn),
        ("mutated parameter", "cylinder", TRAINABLE_PARAMS_BG_COLOR, mutated_parameter),
        ("true/false test", "oval", BOOL_NODE_COLOR, "boolean" in drawn),
    )
    swatches = []
    for text, shape, fill, shown in entries:
        if not shown:
            continue
        if shape == "frame":
            swatches.append(_module_frame_swatch(theme))
            continue
        themed = apply_theme_to_spec(NodeSpec([text], shape=shape, fillcolor=fill), theme)
        swatches.append(
            KeySwatch(
                text=text,
                shape=shape,
                fill=str(themed.fillcolor),
                border=str(themed.color or theme.default_border),
            )
        )
    swatches.extend(_edit_swatches(theme, drawn))
    return tuple(swatches)


def _module_frame_swatch(theme: VisualizationTheme) -> KeySwatch:
    """Return the module entry: a real module frame, not a plain op's thin box.

    The frame has the module box's page fill and ink and a thick stroke, so
    the key matches the expanded modules in the picture (Lead, 29 Oct 2026).
    """

    from .themes import _theme_cluster_attrs

    cluster = _theme_cluster_attrs(theme)
    return KeySwatch(
        text="module",
        shape="frame",
        fill=cluster.get("fillcolor", str(theme.graph.get("bgcolor", "white"))),
        border=cluster.get("color", "black"),
        line=_MODULE_FRAME_LINE,
    )


def _edit_swatches(theme: VisualizationTheme, drawn: frozenset[str]) -> list[KeySwatch]:
    """Return the edit entries: a ringed oval ``edited`` and the highlighted arrow."""

    from ._edit_highlighter import highlighter_rgba
    from ._edit_marks import EDIT_RING_PENWIDTH, edit_ink

    ink = edit_ink(theme)
    swatches = []
    if "edited" in drawn:
        swatches.append(
            KeySwatch(
                text="edited",
                shape="oval",
                fill=theme.default_fill,
                border=ink,
                line=EDIT_RING_PENWIDTH,
            )
        )
    if "changed by the edit" in drawn:
        swatches.append(
            KeySwatch(
                text="changed by the edit",
                shape="arrow",
                fill=highlighter_rgba(theme),
                border=str(theme.edge.get("color", theme.default_border)),
            )
        )
    return swatches


def _swatch_dir() -> Path:
    """Return the directory the key's swatch images live in.

    The images are content-addressed (shape and colours name the file), so one
    shared directory under the system temp folder serves every draw and the
    DOT source keeps rendering from any working directory after the trace is
    gone. A directory another user owns falls back to a private one.
    """

    shared = Path(tempfile.gettempdir()) / "torchlens-key"
    try:
        shared.mkdir(exist_ok=True)
        if os.access(shared, os.W_OK):
            return shared
    except OSError:
        pass
    return _private_swatch_dir()


@functools.lru_cache(maxsize=1)
def _private_swatch_dir() -> Path:
    """Return a per-process swatch directory, removed at exit."""

    path = Path(tempfile.mkdtemp(prefix="torchlens-key-"))
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


def _swatch_image(swatch: KeySwatch) -> str:
    """Draw one swatch as a PNG (once per shape and colours) and return its absolute path."""

    from PIL import Image, ImageColor, ImageDraw

    def colour(name: str) -> tuple[int, ...]:
        """Return an RGBA tuple for a Graphviz colour name or hex value."""
        value = ImageColor.getcolor(name, "RGBA")
        return value if isinstance(value, tuple) else (value, value, value, 255)

    parts = [swatch.shape, swatch.fill, swatch.border]
    if swatch.line != 1.0:
        parts.append(f"w{swatch.line:g}")
    stem = "-".join(part.strip("#").lower() for part in parts)
    image_dir = _swatch_dir()
    path = image_dir / f"key-{stem}-v{_SWATCH_VERSION}.png"
    if path.is_file():
        return str(path)
    width = _SWATCH_WIDTH * _SWATCH_SCALE
    height = _SWATCH_HEIGHT * _SWATCH_SCALE
    line = max(1, round(swatch.line * _SWATCH_SCALE))
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    fill, border = colour(swatch.fill), colour(swatch.border)
    left, top = line // 2, line // 2
    right, bottom = width - 1 - line // 2, height - 1 - line // 2
    if swatch.shape == "arrow":
        _draw_arrow_swatch(draw, width, height, fill, border)
    elif swatch.shape == "oval":
        draw.ellipse((left, top, right, bottom), fill=fill, outline=border, width=line)
    elif swatch.shape == "cylinder":
        cap = (bottom - top) // 3
        draw.rectangle((left, top + cap // 2, right, bottom - cap // 2), fill=fill)
        draw.ellipse((left, bottom - cap, right, bottom), fill=fill, outline=border, width=line)
        draw.line((left, top + cap // 2, left, bottom - cap // 2), fill=border, width=line)
        draw.line((right, top + cap // 2, right, bottom - cap // 2), fill=border, width=line)
        draw.ellipse((left, top, right, top + cap), fill=fill, outline=border, width=line)
    else:
        draw.rectangle((left, top, right, bottom), fill=fill, outline=border, width=line)
    # Write then rename, so a concurrent draw never reads a half-written file.
    partial = image_dir / f".{path.name}.{os.getpid()}.tmp"
    image.save(partial, format="PNG")
    os.replace(partial, path)
    return str(path)


def _draw_arrow_swatch(
    draw: Any, width: int, height: int, under: tuple[int, ...], ink: tuple[int, ...]
) -> None:
    """Draw a short arrow (1 pt line, small head) over a wide translucent bar."""

    middle = height // 2
    bar = 8 * _SWATCH_SCALE // 2
    radius = bar
    draw.rounded_rectangle(
        (radius // 2, middle - bar, width - 1 - radius // 2, middle + bar),
        radius=radius,
        fill=under,
    )
    head = 3 * _SWATCH_SCALE
    tip = width - 2 * _SWATCH_SCALE
    draw.line((3 * _SWATCH_SCALE, middle, tip - head, middle), fill=ink, width=_SWATCH_SCALE)
    draw.polygon(
        ((tip, middle), (tip - head, middle - head // 2), (tip - head, middle + head // 2)),
        fill=ink,
    )


@functools.lru_cache(maxsize=256)
def _text_width(text: str, size: float, bold: bool = False) -> float:
    """Return ``text``'s width in points in the bundled face Graphviz draws with."""

    from PIL import ImageFont

    from ._fonts import _FONT_DIR

    name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
    try:
        font = ImageFont.truetype(str(_FONT_DIR / name), size=max(1, round(size * 10)))
        return float(font.getlength(text)) / 10.0
    except OSError:
        # A stripped install without the bundled face: a generous average width.
        return 0.62 * size * len(text)


def _key_rows_html(swatches: tuple[KeySwatch, ...], theme: VisualizationTheme) -> str:
    """Return the key as one fixed-size HTML table of swatch-and-word pairs.

    Graphviz stretches a nested table to fill its cell, which spread the
    swatches away from their words whenever the header or a section below was
    wider than the key. The table is therefore sized to its content (the words
    measured in the bundled face, with a small margin) and left-aligned.
    """

    size = theme.typography.secondary_size
    font_color = theme.default_font
    bold = "bold" in theme.typography.family.lower()
    rows = [
        swatches[start : start + _SWATCHES_PER_ROW]
        for start in range(0, len(swatches), _SWATCHES_PER_ROW)
    ]
    text_columns = [0.0] * _SWATCHES_PER_ROW
    for row in rows:
        for index, swatch in enumerate(row):
            text_columns[index] = max(
                text_columns[index], _text_width(swatch.text, size, bold) * 1.06 + 2.0
            )
    used = [width for width in text_columns if width > 0.0]
    width = sum(used) + _SWATCH_WIDTH * len(used) + _KEY_SPACING * (2 * len(used) + 1)
    row_height = max(float(_SWATCH_HEIGHT), size * 1.25)
    height = row_height * len(rows) + _KEY_SPACING * (len(rows) + 1)
    html_rows = []
    for row in rows:
        cells = []
        for swatch in row:
            source = html_escape(_swatch_image(swatch))
            cells.append(
                f'<TD FIXEDSIZE="TRUE" WIDTH="{_SWATCH_WIDTH}" HEIGHT="{_SWATCH_HEIGHT}">'
                f'<IMG SRC="{source}" SCALE="TRUE"/></TD>'
                f'<TD ALIGN="LEFT"><FONT POINT-SIZE="{theme.typography.secondary_pt}" '
                f'COLOR="{font_color}">{html_escape(swatch.text)}</FONT></TD>'
            )
        html_rows.append(f"<TR>{''.join(cells)}</TR>")
    return (
        f'<TR><TD ALIGN="LEFT"><TABLE ID="{KEY_TABLE_ID}" BORDER="0" CELLBORDER="0" '
        f'CELLSPACING="{_KEY_SPACING}" CELLPADDING="0" FIXEDSIZE="TRUE" ALIGN="LEFT" '
        f'WIDTH="{round(width)}" HEIGHT="{round(height)}">{"".join(html_rows)}</TABLE></TD></TR>'
    )


def header_block_label(
    graph_caption: str,
    swatches: tuple[KeySwatch, ...],
    sections: tuple[LegendSection, ...],
    theme: VisualizationTheme,
) -> str:
    """Return the graph label: the header, then the key, as one top-left block.

    Parameters
    ----------
    graph_caption:
        The header label (``<<FONT ...>...</FONT>>``) built before layout.
    swatches:
        Key entries for the drawn marks; empty draws no swatch row.
    sections:
        Further key sections (the channel disclosure), drawn under the swatches.
    theme:
        Resolved skin (text colour, size, border).

    Returns
    -------
    str
        The header label unchanged when there is no key, else the block.
    """

    if not swatches and not sections:
        return graph_caption
    header = graph_caption[1:-1]
    rows = f'<TR><TD ALIGN="LEFT" BALIGN="LEFT">{header}</TD></TR>'
    if swatches:
        rows += _key_rows_html(swatches, theme)
    if sections:
        from ._legend import build_legend_table_label

        rows += f'<TR><TD ALIGN="LEFT">{build_legend_table_label(sections, theme)[1:-1]}</TD></TR>'
    return '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">' + rows + "</TABLE>>"


def caption_from_label(label: str, theme: VisualizationTheme) -> str:
    """Return a graph label as a header caption the key can sit under.

    An HTML label (``<...>``) is used as is; a plain-text label (a
    ``vis_graph_overrides`` label or a lens caption) is escaped, its ``\\n``
    breaks become left-aligned line breaks, and it takes the page's text ink.
    """

    if label.startswith("<") and label.endswith(">"):
        return label
    lines = label.split("\\n")
    body = "".join(f"{html_escape(line)}<br align='left'/>" for line in lines)
    return f"<<FONT COLOR='{theme.default_font}'>{body}</FONT>>"
