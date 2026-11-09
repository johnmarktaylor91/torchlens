"""The fixed top-left block of a forward picture: the header, then the key.

The header names the model, counts what is drawn and gives the parameter
total. The key sits directly under it, inside the same graph label, so the
two always share the top-left corner: a top crop of a big model keeps both,
and Graphviz can never float the key to another corner of the page (it was a
separate legend node before, placed wherever the layout had room).

The key draws each mark with its real shape and fill: ovals for single
function calls (input and output are function calls too), boxes for leaf
modules, cylinders for buffers; the weight fills mark weights on any shape,
light for trainable and dark for frozen. Graphviz HTML labels can only draw
rectangles, so the swatches are small drawn PNG images in a shared,
content-addressed folder under the system temp directory, referenced by
absolute path so the DOT source renders from any working directory. Rows
appear only for marks the picture contains; the key is opt-in
(``show_legend=True``, which also adds the reading key, the notes explaining
the picture's notation), except that a ``module=`` focused draw shows the key
by default (``show_legend=False`` turns it off).

Spellings here are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import atexit
import functools
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._capture_honesty import honesty_banner_lines
from .._utils._record_group_table import record_member
from .._vocab.node_spec import (
    EDITED_HEADER_OMITTED_BANNERS,
    LENS_BANNER_OMIT_KEY,
    LENS_CAPTION_KEY,
)
from ._edit_change import CHANGED_INK, CHANGED_PENWIDTH
from ._edit_highlighter import highlighter_rgba
from ._edit_marks import EDIT_RING_PENWIDTH, edit_ink, header_edit_line
from ._endpoint_rings import endpoint_ring_swatch
from ._focus_scope import focus_address, scope_caption_head
from ._frame_penwidth import MODULE_PENWIDTH
from ._label_format import format_memory, header_count, param_count, params_noun
from ._module_reuse import module_reuse_note
from ._render_common import (
    BOOL_NODE_COLOR,
    BUFFER_NODE_COLOR,
    DEFAULT_BG_COLOR,
    FROZEN_PARAMS_BG_COLOR,
    INPUT_COLOR,
    OUTPUT_COLOR,
    TRAINABLE_PARAMS_BG_COLOR,
)
from ._render_utils import html_escape, strip_path_directories
from ._text_metrics import text_width as _text_width
from .node_spec import NodeSpec, intervention_sites_for_log
from .themes import _hex_rgb, _theme_cluster_attrs, apply_theme_to_spec, resolve_theme

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from ._legend import LegendSection
    from .node_universe import NodeUniverse
    from .request import ResolvedRenderRequest
    from .themes import VisualizationTheme

__all__ = [
    "KEY_TABLE_ID",
    "KeySwatch",
    "caption_from_label",
    "drawn_module_marks",
    "render_module_marks",
    "graph_caption_body",
    "network_count_line",
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
#: Stroke of the key's module frame, in points: the same stroke as every module frame.
_MODULE_FRAME_LINE = MODULE_PENWIDTH


@dataclass(frozen=True)
class KeySwatch:
    """One key entry: a mark drawn with its real shape and fill, plus a word.

    Attributes
    ----------
    text:
        Plain words for the mark (escaped at build time).
    shape:
        ``"oval"``, ``"box"``, ``"box3d"``, ``"double"`` (a double-bordered box)
        or ``"cylinder"``; ``"frame"`` for a module box and ``"arrow"`` for an
        arrow with the downstream highlighter under it.
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
    """Return the header's parameter line, for example ``11.7M params (44.6 MB, all trainable)``.

    Counts from 1,000 up print compact (``124M params``); the exact count stays
    on ``trace.num_params``. The memory is decimal (KB, MB; divisor 1000). A
    nonzero count whose byte total is not a positive number has no recorded
    size, so the line leaves the size out (``170 params (all trainable)``)
    rather than print ``0 B``.
    """

    total = int(trace.num_params)
    if total == 0:
        return "0 params"
    raw_memory = trace._cost_total_param_memory
    known = isinstance(raw_memory, int | float) and not isinstance(raw_memory, bool)
    memory = f"{format_memory(raw_memory)}, " if known and raw_memory > 0 else ""
    count = param_count(total)
    noun = params_noun(total)
    if trace.num_params_frozen == 0:
        return f"{count} {noun} ({memory}all trainable)"
    if trace.num_params_trainable == 0:
        return f"{count} {noun} ({memory}all frozen)"
    trainable = param_count(int(trace.num_params_trainable))
    return f"{count} {noun} ({memory}{trainable}/{count} trainable)"


def _all_ops(trace: Trace) -> list[Any]:
    """Return every recorded op plus the input and output tensors, each once.

    Buffer sources are left out: like parameters, they are model state, not ops.
    """

    ops = [op for op in trace.ops.values() if not record_member(op, "is_buffer", False)]
    seen = {id(op) for op in ops}
    for label in (*trace.input_layers, *trace.output_layers):
        op = trace.ops[label]
        if id(op) not in seen:
            seen.add(id(op))
            ops.append(op)
    return ops


def _in_module(op: Any, address: str) -> bool:
    """Return whether ``op`` ran inside the module at ``address`` (any call)."""

    return any(str(call).split(":", 1)[0] == address for call in getattr(op, "modules", None) or ())


def network_ops(trace: Trace, focus: str | None) -> list[Any]:
    """Return the ops the header counts: the whole network's, or the focused module's."""

    ops = _all_ops(trace)
    if focus is None:
        return ops
    return [op for op in ops if _in_module(op, focus)]


def _memory(ops: Iterable[Any]) -> int | None:
    """Return the summed activation size of ``ops`` in bytes, or ``None`` if any is unknown.

    A partial sum would print as the network's size while leaving ops out, so
    one op without a recorded size makes the whole total unknown.
    """

    total = 0
    for op in ops:
        value = record_member(op, "activation_memory", None)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        total += int(value)
    return total


def network_count_line(trace: Trace, focus: str | None, vis_mode: str) -> str:
    """Return the header's count line, for example ``372 ops (325.9 MB)``.

    Parameters
    ----------
    trace:
        The trace being drawn.
    focus:
        The ``module=`` focus address (``None`` for a whole-model draw).
    vis_mode:
        ``"rolled"`` adds the layer count (``8 layers (11 ops, 149 B)``).

    Notes
    -----
    When any counted op has no recorded size the size is left out
    (``5 ops``; rolled ``8 layers (11 ops)``) rather than print a partial sum.
    """

    ops = network_ops(trace, focus)
    memory = _memory(ops)
    op_noun = "op" if len(ops) == 1 else "ops"
    ops_text = f"{header_count(len(ops))} {op_noun}"
    if vis_mode != "rolled":
        return ops_text if memory is None else f"{ops_text} ({format_memory(memory)})"
    layers = len({getattr(op, "layer_label", id(op)) for op in ops})
    layer_noun = "layer" if layers == 1 else "layers"
    inner = ops_text if memory is None else f"{ops_text}, {format_memory(memory)}"
    return f"{header_count(layers)} {layer_noun} ({inner})"


def graph_caption_html(
    trace: Trace, universe: NodeUniverse, request: ResolvedRenderRequest, theme: VisualizationTheme
) -> str:
    """Return the complete graph caption: the header body in the theme's ink."""

    del universe  # the count line reports the traced network, never the drawn boxes
    focus = focus_address(getattr(request, "module", None))
    count_line = network_count_line(trace, focus, request.vis_mode)
    body = graph_caption_body(trace, count_line, request)
    return f"<<FONT COLOR='{theme.default_font}'>{body}</FONT>>"


def graph_caption_body(trace: Trace, count_line: str, request: ResolvedRenderRequest) -> str:
    """Assemble the header body: model, counts, edits, honesty banner, and disclosures.

    Returns
    -------
    str
        HTML-like caption body without the outer FONT wrapper.
    """

    caption_body = _caption_head(trace, count_line, getattr(request, "module", None))
    edit_line = _header_edit_line(trace, request)
    if edit_line is not None:
        caption_body += f"{edit_line}<br align='left'/>"
    if getattr(trace, "_has_direct_writes", False):
        caption_body += (
            "Direct writes detected - replay and rerun propagation will overlay them"
            "<br align='left'/>"
        )
    caption_body += _honesty_banner_rows(trace, request, edited=edit_line is not None)
    encoding_state = getattr(request, "encoding", None)
    if encoding_state is not None and getattr(encoding_state, "stack_spec", None) is not None:
        # Stacking disclosure is part of the contract: the
        # rendered output captions which annotation produced the columns.
        caption_body += (
            f"stacked by: {html_escape(encoding_state.stack_spec.display_name)}<br align='left'/>"
        )
    # No silent floor (collapse): a floor-fallback plan renders with a
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
    return caption_body + _lens_caption_rows(request)


def _lens_caption_rows(request: ResolvedRenderRequest) -> str:
    """Return a lens's disclosure lines, in the key's text size, for under the header.

    A lens draw passes its caption in ``graph_overrides`` under
    :data:`LENS_CAPTION_KEY`, so the model header stays and the caption adds
    to it instead of replacing it.
    """

    caption = (getattr(request, "graph_overrides", None) or {}).get(LENS_CAPTION_KEY)
    if not caption:
        return ""
    size = resolve_theme(request.theme).typography.secondary_size
    return "".join(
        f"<FONT POINT-SIZE=\"{size:g}\">{html_escape(line)}</FONT><br align='left'/>"
        for line in str(caption).split("\n")
        if line
    )


def _caption_head(trace: Trace, count_line: str, module: Any) -> str:
    """Return the header's first lines: model (or focus scope), drawn count, parameters."""

    return scope_caption_head(trace, module, count_line, params_caption_detail(trace))


def _honesty_banner_rows(trace: Trace, request: ResolvedRenderRequest, *, edited: bool) -> str:
    """Return the capture-honesty banner rows as caption HTML.

    A poisoned/diverged sparse run, an unverified capture, or an episode
    capture must not RENDER clean: the graph is a publishable
    artifact, so the caption carries the facts. The picture keeps a cited
    source location's file name and line only. A view may move named lines
    to its own record (:data:`LENS_BANNER_OMIT_KEY`; the interventions census);
    an edited trace's edit line drops :data:`EDITED_HEADER_OMITTED_BANNERS`.
    """

    omitted = tuple((request.graph_overrides or {}).get(LENS_BANNER_OMIT_KEY, ()))
    if edited:
        omitted += EDITED_HEADER_OMITTED_BANNERS
    rows = ""
    for banner_line in honesty_banner_lines(trace):
        if omitted and banner_line.startswith(omitted):
            continue
        banner_text = html_escape(strip_path_directories(banner_line))
        if banner_line.startswith("capture advisory:"):
            # An advisory discloses a gap in source provenance, not a failed
            # capture: it stays in the header, in the key's text size, unbolded.
            size = resolve_theme(request.theme).typography.secondary_size
            rows += f"<FONT POINT-SIZE=\"{size:g}\">{banner_text}</FONT><br align='left'/>"
            continue
        rows += f"<B>{banner_text}</B><br align='left'/>"
    return rows


def _header_edit_line(trace: Trace, request: ResolvedRenderRequest) -> str | None:
    """Return the header's edit line (``1 edit: Δ zero_ablate on linear_1_1``), if any."""

    sites = intervention_sites_for_log(trace)
    if not sites:
        return None
    theme = resolve_theme(request.theme)
    color = str((request.graph_overrides or {}).get("intervention_site_color", edit_ink(theme)))
    return header_edit_line(trace, sites, color, vis_mode=request.vis_mode)


def drawn_module_marks(
    node_attrs: Iterable[Mapping[str, Any]], has_regions: bool
) -> frozenset[str]:
    """Return the module marks the picture draws, each with its own key entry.

    ``frame`` is an expanded module's frame (a drawn region); ``box`` a module
    drawn as one solid box (an atomic module, ``@address`` label); ``box3d`` a
    collapsed module; ``double`` the double outline of an op whose module
    holds buffers that are not drawn. Dashed boxes (segments, container
    summaries) are not modules. The set is empty, so falsy, when no module
    mark is drawn.
    """

    marks: set[str] = {"frame"} if has_regions else set()
    for attrs in node_attrs:
        style = str(attrs.get("style", "")).lower()
        if "invis" in style:
            continue
        shape = str(attrs.get("shape", "")).lower()
        if shape == "box3d":
            marks.add("box3d")
        elif shape == "box" and "dashed" not in style:
            marks.add("box")
        if str(attrs.get("peripheries", "")) == "2":
            marks.add("double")
    return frozenset(marks)


def render_module_marks(node_attrs: Iterable[Mapping[str, Any]], render_ir: Any) -> frozenset[str]:
    """Return :func:`drawn_module_marks` for a finished forward IR.

    A module region is a frame only when its cluster was emitted: atomic
    modules drawn as one box and empty module subtrees keep an IR region
    but draw no frame.
    """

    frames = {
        f"cluster_{region.key.replace(':', '_pass')}"
        for region in render_ir.regions
        if region.kind == "module"
    }
    pending = list(render_ir.dot_statements)
    while pending:
        statement = pending.pop()
        if statement.kind == "subgraph":
            if dict(statement.attrs).get("name") in frames:
                return drawn_module_marks(node_attrs, True)
            pending.extend(statement.children)
    return drawn_module_marks(node_attrs, False)


def key_swatches(
    theme: VisualizationTheme,
    drawn: frozenset[str],
    *,
    has_module: bool,
    mutated_parameter: bool = False,
    module_marks: frozenset[str] = frozenset(),
    endpoint_rings: bool = False,
) -> tuple[KeySwatch, ...]:
    """Return the key entries for the marks this picture draws, in display order.

    Both weight greys are listed whenever any weighted node is drawn, so a
    reader can tell trainable from frozen weights. Plain operations get no
    entry, nor does an oval mark the skin fills like one. Each module mark in ``module_marks`` (``drawn_module_marks``) gets
    an entry drawn like it; ``has_module=True`` alone means a module frame.
    An edited node adds ``edited`` (the ringed oval) and, with the downstream
    highlighter drawn, ``changed by the edit``. ``endpoint_rings=True`` (a
    colour scale owns the fills) draws the input and output entries as the
    rings the picture draws, not as filled ovals.
    """

    has_weights = "parameterized" in drawn
    marks = module_marks | ({"frame"} if has_module else set())
    entries = (
        ("input", "oval", INPUT_COLOR, "input" in drawn),
        ("output", "oval", OUTPUT_COLOR, "output" in drawn),
        ("trainable weights", "box", TRAINABLE_PARAMS_BG_COLOR, has_weights),
        ("frozen weights", "box", FROZEN_PARAMS_BG_COLOR, has_weights),
        ("module", "frame", DEFAULT_BG_COLOR, "frame" in marks),
        ("module run as one op", "box", DEFAULT_BG_COLOR, "box" in marks),
        ("collapsed module", "box3d", DEFAULT_BG_COLOR, "box3d" in marks),
        ("hidden buffers", "double", DEFAULT_BG_COLOR, "double" in marks),
        ("buffer", "cylinder", BUFFER_NODE_COLOR, "buffer" in drawn),
        ("mutated parameter", "cylinder", TRAINABLE_PARAMS_BG_COLOR, mutated_parameter),
        ("true/false test", "oval", BOOL_NODE_COLOR, "boolean" in drawn),
    )
    swatches = [
        swatch
        for swatch in (_entry_swatch(theme, *entry[:3]) for entry in entries if entry[3])
        if swatch is not None
    ]
    if endpoint_rings:
        swatches = [endpoint_ring_swatch(theme, swatch) for swatch in swatches]
    swatches.extend(_edit_swatches(theme, drawn))
    return tuple(swatches)


def _entry_swatch(theme: VisualizationTheme, text: str, shape: str, fill: str) -> KeySwatch | None:
    """Return one key entry with the skin's fill and border for its mark.

    ``None`` for an oval the skin fills like a plain op: a swatch identical
    to every function call teaches nothing.
    """

    if shape == "frame":
        return _module_frame_swatch(theme)
    themed = apply_theme_to_spec(NodeSpec([text], shape=shape, fillcolor=fill), theme)
    if shape == "oval" and _hex_rgb(str(themed.fillcolor)) == _hex_rgb(theme.default_fill):
        return None
    return KeySwatch(
        text=text,
        shape=shape,
        fill=str(themed.fillcolor),
        border=str(themed.color or theme.default_border),
    )


def _module_frame_swatch(theme: VisualizationTheme) -> KeySwatch:
    """Return the module entry: a real module frame, not a plain op's thin box.

    The frame has the module box's page fill, ink and stroke, so
    the key matches the expanded modules in the picture (Lead, 29 Oct 2026).
    """

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
    if "changed value" in drawn:
        swatches.append(
            KeySwatch(
                text="changed by the edit",
                shape="oval",
                fill=theme.default_fill,
                border=CHANGED_INK,
                line=CHANGED_PENWIDTH,
            )
        )
    if "changed by the edit" in drawn:
        # Beside the value-change outline, the cone's arrows say "downstream".
        cone_text = "downstream of the edit" if "changed value" in drawn else "changed by the edit"
        swatches.append(
            KeySwatch(
                text=cone_text,
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
    elif swatch.shape == "box3d":
        depth = 3 * _SWATCH_SCALE
        draw.rectangle(
            (left + depth, top, right, bottom - depth), fill=fill, outline=border, width=line
        )
        draw.rectangle(
            (left, top + depth, right - depth, bottom), fill=fill, outline=border, width=line
        )
    elif swatch.shape == "double":
        gap = 2 * _SWATCH_SCALE
        draw.rectangle((left, top, right, bottom), fill=fill, outline=border, width=line)
        draw.rectangle(
            (left + gap, top + gap, right - gap, bottom - gap), outline=border, width=line
        )
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
    text_columns = [0] * _SWATCHES_PER_ROW
    for row in rows:
        for index, swatch in enumerate(row):
            text_columns[index] = max(
                text_columns[index], math.ceil(_text_width(swatch.text, size, bold) * 1.06 + 2.0)
            )
    used = [width for width in text_columns if width > 0]
    width = sum(used) + _SWATCH_WIDTH * len(used) + _KEY_SPACING * (2 * len(used) + 1)
    row_height = max(float(_SWATCH_HEIGHT), size * 1.25)
    height = row_height * len(rows) + _KEY_SPACING * (len(rows) + 1)
    html_rows = []
    for row in rows:
        cells = []
        for index, swatch in enumerate(row):
            source = html_escape(_swatch_image(swatch))
            # Each word cell holds its column's measured width: Graphviz's own
            # (hinted, narrower) measure spread the table's slack evenly, and a
            # browser drew "module run as one op" into the next swatch.
            cells.append(
                f'<TD FIXEDSIZE="TRUE" WIDTH="{_SWATCH_WIDTH}" HEIGHT="{_SWATCH_HEIGHT}">'
                f'<IMG SRC="{source}" SCALE="TRUE"/></TD>'
                f'<TD ALIGN="LEFT" WIDTH="{text_columns[index]}">'
                f'<FONT POINT-SIZE="{theme.typography.secondary_pt}" '
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
    notes: tuple[str, ...] = (),
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
    notes:
        Plain-text reading rows (a focused draw's notation key), drawn under
        the swatches in the key's text size.

    Returns
    -------
    str
        The header label unchanged when there is no key, else the block.
    """

    if not swatches and not sections and not notes:
        return graph_caption
    header = graph_caption[1:-1]
    rows = f'<TR><TD ALIGN="LEFT" BALIGN="LEFT">{header}</TD></TR>'
    if swatches:
        rows += _key_rows_html(swatches, theme)
    if notes:
        body = "".join(f"{html_escape(note)}<br align='left'/>" for note in notes)
        size = theme.typography.secondary_size
        rows += (
            f'<TR><TD ALIGN="LEFT" BALIGN="LEFT"><FONT POINT-SIZE="{size:g}" '
            f"COLOR='{theme.default_font}'>{body}</FONT></TD></TR>"
        )
    if sections:
        from ._legend import build_legend_table_label

        rows += f'<TR><TD ALIGN="LEFT">{build_legend_table_label(sections, theme, fit=True)[1:-1]}</TD></TR>'
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
