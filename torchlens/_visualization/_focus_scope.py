"""The scope of a ``module=`` focused draw: header scope line, reading key, ancestor frames.

A focused draw shows one module's operations. The modules that enclose it
(its ancestors) contain nothing else that is drawn, so their frames would
only nest the picture in empty margins. They are not drawn: the header's
scope line names them instead (``GPT2LMHeadModel › transformer › h.5 ›
mlp``), and the focused module's own frame is the outermost one, with frame
widths measured from it.

Inside the picture, module addresses leave off the path the scope line
already printed (the focused module's enclosing path): under the scope
``GPT2LMHeadModel › transformer › h.5 › mlp`` the frame reads ``@mlp`` and
its children ``@mlp.c_fc`` (``relative_address``; the label rewriting lives
in ``_focus_relabel``).

Spellings here are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member
from ._frame_penwidth import compute_module_penwidth

if TYPE_CHECKING:
    import graphviz

    from .._data_classes.module import Module
    from .._data_classes.trace import Trace
    from .render_ir import RenderIR, RenderIRRegion

__all__ = [
    "drop_ancestor_regions",
    "flatten_ancestor_frames",
    "focus_address",
    "focus_ancestor_keys",
    "focus_parent_prefix",
    "focus_relative_depth",
    "module_at",
    "reading_notes",
    "relative_address",
    "scope_caption_head",
]

#: An op name's counters: ``tanh_6_270`` is the 6th ``tanh`` call and op 270
#: of the whole forward pass; ``:2`` marks a later pass of a repeated op. A
#: DOT node name spells the pass ``pass2``; the drawn label keeps the colon.
_OP_NAME = re.compile(r"^(?P<kind>.+?)_(?P<nth>\d+)_(?P<op>\d+)(?:(?::|pass)(?P<pass>\d+))?$")
#: HTML tags inside a node label, and a ``name=value`` call argument in its text.
_TAG = re.compile(r"<[^>]*>")
_ARGUMENT = re.compile(r"(?:^|\s)[A-Za-z_]\w*=\S")


def _escape(value: str) -> str:
    """Escape the three Graphviz HTML-label specials (``&``, ``<``, ``>``)."""

    return html.escape(value, quote=False)


def focus_address(module: Any) -> str | None:
    """Return the focused module's address, or ``None`` for a whole-model draw."""

    if module is None:
        return None
    address = module if isinstance(module, str) else record_member(module, "address", None)
    if not isinstance(address, str) or address in {"", "self"}:
        return None
    return address


def module_at(trace: Trace, address: str) -> Module | None:
    """Return the module record at ``address``, or ``None`` when the trace has none there."""

    if address not in trace.modules:  # the accessor resolves, it has no .get
        return None
    return trace.modules[address]


def focus_relative_depth(trace: Trace, module: Any, depth: int) -> int:
    """Return the module nesting depth to draw, counted from the focused module.

    ``depth=`` is the maximum module nesting depth shown. A whole-model draw
    counts it from the model (``depth=1`` draws the top-level modules as one
    box each); a focused draw counts it from the focused module, so
    ``depth=1`` draws the module's own child modules as one box each and
    larger values open them, as the same value does for a whole-model draw of
    a model that started at the focused module. The focused module's own
    nesting level is read from the called-module stack of its first recorded
    op (uncalled containers such as a ``ModuleList`` add no level).
    """

    focus = focus_address(module)
    if focus is None:
        return depth
    for op in trace.ops:
        stack = [str(call).split(":", 1)[0] for call in getattr(op, "modules", None) or ()]
        if focus in stack:
            return depth + stack.index(focus) + 1
    return depth


def focus_parent_prefix(module: Any) -> str | None:
    """Return the address the scope line prints above the focused module, or ``None``.

    Addresses drawn inside a focused picture leave this prefix off. A module
    directly under the model (``pre_x``) has none. List indices stay with
    their list: the prefix of ``transformer.h.5`` is ``transformer``.
    """

    focus = focus_address(module)
    if focus is None:
        return None
    return ".".join(_breadcrumb(focus)[:-1]) or None


def relative_address(address: str, prefix: str | None) -> str:
    """Return ``address`` without the focused draw's scope prefix.

    An address under the prefix drops it (``transformer.h.5.mlp.c_fc`` reads
    ``mlp.c_fc`` under ``transformer.h.5``); the prefix module itself reads as
    its last scope step (``h.5``); any other address is returned whole.
    """

    if prefix is None:
        return address
    if address == prefix:
        return _breadcrumb(prefix)[-1]
    if address.startswith(f"{prefix}."):
        return address[len(prefix) + 1 :]
    return address


def _is_ancestor(key: str, focus: str) -> bool:
    """Return whether a module region key names a strict ancestor of ``focus``."""

    address = key.split(":", 1)[0]
    return address != "self" and focus.startswith(f"{address}.")


def focus_ancestor_keys(keys: Iterable[str], module: Any) -> frozenset[str]:
    """Return the module region keys a focused draw does not frame.

    Parameters
    ----------
    keys:
        Module region keys (``address`` or ``address:pass``).
    module:
        The draw's ``module=`` focus (address string or Module), or ``None``.

    Returns
    -------
    frozenset[str]
        The keys whose module encloses the focus; empty for a whole-model draw.
    """

    focus = focus_address(module)
    if focus is None:
        return frozenset()
    return frozenset(key for key in keys if _is_ancestor(key, focus))


def flatten_ancestor_frames(
    graph: graphviz.Digraph,
    roots: Sequence[str],
    payloads: Mapping[str, Any],
    children: Mapping[str, Sequence[str]],
    flatten: frozenset[str],
    emit: Any,
) -> tuple[list[str], int]:
    """Emit flattened frames' contents at top level and return the new frame roots.

    Each flattened frame is replaced by its child frames, so the walk that
    nests module frames starts below it. The frame's own nodes, edges, rank
    groups and container frames are emitted straight into ``graph`` through
    ``emit`` (the frame emitter's own content writer), never dropped.

    Returns
    -------
    tuple[list[str], int]
        The frame roots to walk, and the number of rank groups emitted here.
    """

    if not flatten:
        return list(roots), 0
    new_roots: list[str] = []
    rank_groups = 0
    pending = list(roots)
    while pending:
        key = pending.pop(0)
        payload_key = key if key in payloads else key.split(":", 1)[0]
        if payload_key not in flatten and key not in flatten:
            new_roots.append(key)
            continue
        rank_groups += emit(graph, payloads.get(payload_key, {}))
        pending[0:0] = list(children.get(key, ()))
    return new_roots, rank_groups


def drop_ancestor_regions(render_ir: RenderIR, flatten: frozenset[str]) -> RenderIR:
    """Remove flattened ancestor frames from the IR and re-measure frame widths.

    Nodes and edges the ancestors owned move to the top level; each remaining
    frame's parent skips the dropped ones, and border widths are recomputed so
    the outermost drawn frame takes the widest border.
    """

    if not flatten:
        return render_ir
    kept = {region.key: region for region in render_ir.regions if region.key not in flatten}
    nodes = tuple(
        replace(node, region_path=tuple(key for key in node.region_path if key not in flatten))
        for node in render_ir.nodes
    )
    edges = tuple(
        replace(edge, owner_cluster=None) if edge.owner_cluster in flatten else edge
        for edge in render_ir.edges
    )
    parents = {key: _kept_parent(region, kept, render_ir.regions) for key, region in kept.items()}
    depths = {key: _depth(key, parents) for key in kept if kept[key].kind == "module"}
    max_depth = max(depths.values(), default=0)
    regions = tuple(
        _restyled(region, parents[region.key], depths.get(region.key), max_depth)
        for region in render_ir.regions
        if region.key in kept
    )
    return replace(render_ir, nodes=nodes, edges=edges, regions=regions)


def _kept_parent(
    region: RenderIRRegion,
    kept: Mapping[str, RenderIRRegion],
    regions: Sequence[RenderIRRegion],
) -> str | None:
    """Return the nearest ancestor of ``region`` that is still drawn."""

    by_key = {candidate.key: candidate for candidate in regions}
    parent = region.parent_key
    while parent is not None and parent not in kept:
        parent = by_key[parent].parent_key if parent in by_key else None
    return parent


def _depth(key: str, parents: Mapping[str, str | None]) -> int:
    """Return a drawn frame's 0-based depth among drawn frames."""

    depth = 0
    parent = parents.get(key)
    while parent is not None:
        depth += 1
        parent = parents.get(parent)
    return depth


def _restyled(
    region: RenderIRRegion, parent: str | None, depth: int | None, max_depth: int
) -> RenderIRRegion:
    """Return ``region`` with its kept parent and, for a module, its re-measured border."""

    if depth is None:
        return replace(region, parent_key=parent)
    style = tuple(
        (name, str(compute_module_penwidth(depth, max_depth)) if name == "penwidth" else value)
        for name, value in region.style
    )
    return replace(region, parent_key=parent, style=style)


def _breadcrumb(address: str) -> list[str]:
    """Split a module address into scope steps, keeping list indices on their list.

    ``transformer.h.5.mlp`` reads ``transformer``, ``h.5``, ``mlp``.
    """

    steps: list[str] = []
    for part in address.split("."):
        if part.isdigit() and steps:
            steps[-1] = f"{steps[-1]}.{part}"
        else:
            steps.append(part)
    return steps


#: Joins the scope line's steps: U+203A between non-breaking spaces.
_SCOPE_SEPARATOR = "&#160;&#8250;&#160;"


def scope_caption_head(trace: Trace, module: Any, count_line: str, params_line: str) -> str:
    """Return the header head: a focused draw's scope and drawn view, else the whole model's.

    A focused draw's first line is the scope: the model class, then each
    enclosing module down to the focused one with its class
    (``GPT2LMHeadModel › transformer › h.5 › mlp (GPT2MLP)``); the frames of
    the enclosing modules are not drawn. Its second line counts the focused
    module's ops whatever the drawing collapses (``8 ops (480 KB)``), with no label: the count line of a
    whole-model draw carries none either. Steps are joined by a single
    right-pointing angle quotation mark (U+203A, in the bundled font) between
    non-breaking spaces, so a long scope never wraps at a separator. The whole
    scope line is one bold text run: Graphviz writes every change of weight
    as a separate SVG ``<text>`` placed by its own font metrics, so a line
    that mixed bold and plain steps drifted at each boundary in browsers
    whose font differs (the gap before the last step read wider than the
    others). The whole model's parameter line is printed on a whole-model
    draw only.
    """

    focus = focus_address(module)
    class_name = _escape(trace._source_class_name)
    if focus is None:
        return (
            f"<B>{class_name}</B><br align='left'/>{_escape(count_line)}"
            f"<br align='left'/>{params_line}<br align='left'/>"
        )
    focus_module = trace.modules[focus]
    steps = [_escape(step) for step in _breadcrumb(focus)]
    steps[-1] = f"{steps[-1]}&#160;({_escape(focus_module.class_name)})"
    scope = _SCOPE_SEPARATOR.join([class_name, *steps])
    return f"<B>{scope}</B><br align='left'/>{_escape(count_line)}<br align='left'/>"


def _drawn_op_name(match: re.Match[str], text: str) -> str:
    """Return the op name as the picture spells it: ``add_1_1``, or ``add_1_1:2`` for a pass."""

    base = f"{match['kind']}_{match['nth']}_{match['op']}"
    drawn = re.search(rf"(?<![\w.]){re.escape(base)}(?::\d+)?(?!\w)", text)
    return drawn.group(0) if drawn else base


def _ordinal(number: int) -> str:
    """Return ``1st``, ``2nd``, ``3rd``, ``4th`` ... for ``number``."""

    if 10 <= number % 100 <= 20:
        return f"{number}th"
    return f"{number}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th') }"


def reading_notes(node_attrs: Iterable[Mapping[str, Any]], module: Any) -> tuple[str, ...]:
    """Return the text rows of a focused draw's reading key (``show_legend=True``).

    Each row explains one notation the picture uses, taking a real example
    from the drawn nodes where it can: the ``@`` address prefix, an op
    name's two counters, the output shape and size line, and call arguments.
    A whole-model draw gets no rows.
    """

    if focus_address(module) is None:
        return ()
    texts = [_TAG.sub(" ", str(attrs.get("label", ""))) for attrs in node_attrs]
    names = [str(attrs.get("name", "")) for attrs in node_attrs]
    prefix = focus_parent_prefix(module)
    under = f" under {prefix}" if prefix else ""
    notes = [f"@name: a module address{under} (a frame title or a module drawn as one box)"]
    example = next(
        (
            (m, t)
            for m, t in zip(map(_OP_NAME.match, names), texts, strict=True)
            if m and not m["kind"].startswith("_")
        ),
        None,
    )
    if example is not None:
        match, text = example
        notes.append(
            f"{_drawn_op_name(match, text)}: the {_ordinal(int(match['nth']))} {match['kind']}"
            f" call, op {match['op']} of the whole forward pass"
        )
    notes.append("(shape), size: the op's output tensor and its memory")
    if any(name.startswith("__module_focus_") for name in names):
        notes.append(
            "from module, to module: where an input comes from or an output goes,"
            " then that op's name"
        )
    if any("params:" in text for text in texts):
        notes.append("params: the weights the op used (filled as the key shows)")
    if any(_ARGUMENT.search(text) for text in texts):
        notes.append("name=value: an argument the op was called with")
    return tuple(notes)
