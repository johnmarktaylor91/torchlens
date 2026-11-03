"""The scope of a ``module=`` focused draw: header scope line, reading key, ancestor frames.

A focused draw shows one module's operations. The modules that enclose it
(its ancestors) contain nothing else that is drawn, so their frames would
only nest the picture in empty margins. They are not drawn: the header's
scope line names them instead (``GPT2LMHeadModel > transformer > h.5 >
mlp``), and the focused module's own frame is the outermost one, with frame
widths measured from it.

Spellings here are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ._render_utils import compute_module_penwidth, html_escape

if TYPE_CHECKING:
    import graphviz

    from .._data_classes.trace import Trace
    from .render_ir import RenderIR, RenderIRRegion

__all__ = [
    "drop_ancestor_regions",
    "strip_focus_ancestors",
    "flatten_ancestor_frames",
    "focus_address",
    "focus_ancestor_keys",
    "reading_notes",
    "scope_caption_head",
]

#: An op name's counters: ``tanh_6_266`` is the 6th ``tanh`` call and op 266
#: of the whole forward pass; ``:2`` marks a later pass of a repeated op.
_OP_NAME = re.compile(r"^(?P<kind>.+?)_(?P<nth>\d+)_(?P<op>\d+)(?::\d+)?$")
#: HTML tags inside a node label, and a ``name=value`` call argument in its text.
_TAG = re.compile(r"<[^>]*>")
_ARGUMENT = re.compile(r"(?:^|\s)[A-Za-z_]\w*=\S")


def focus_address(module: Any) -> str | None:
    """Return the focused module's address, or ``None`` for a whole-model draw."""

    if module is None:
        return None
    address = module if isinstance(module, str) else getattr(module, "address", None)
    if not isinstance(address, str) or address in {"", "self"}:
        return None
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


def strip_focus_ancestors(
    render_ir: RenderIR, payload_keys: Iterable[str], module: Any
) -> tuple[RenderIR, frozenset[str]]:
    """Drop a focused draw's ancestor frames from the IR; return it and the dropped keys.

    The keys come from the module-cluster payloads and the IR regions both,
    so a frame known to either is dropped; a whole-model draw changes nothing.
    """

    keys = [*payload_keys, *(region.key for region in render_ir.regions)]
    flatten = focus_ancestor_keys(keys, module)
    return drop_ancestor_regions(render_ir, flatten), flatten


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


def scope_caption_head(trace: Trace, module: Any, count_line: str, params_line: str) -> str:
    """Return the header head of a focused draw: scope, drawn view, whole model.

    The first line is the scope: the model class, then each enclosing module
    down to the focused one with its class (``GPT2LMHeadModel > transformer >
    h.5 > mlp (GPT2MLP)``); the frames of the enclosing modules are not drawn.
    The count line covers the drawn view, the parameter line the whole model,
    and each says so.
    """

    focus = focus_address(module)
    class_name = html_escape(trace._source_class_name)
    if focus is None:
        return (
            f"<B>{class_name}</B><br align='left'/>{html_escape(count_line)}"
            f"<br align='left'/>{params_line}<br align='left'/>"
        )
    focus_module = trace.modules[focus]
    steps = [html_escape(step) for step in _breadcrumb(focus)]
    steps[-1] = f"<B>{steps[-1]}</B> ({html_escape(focus_module.class_name)})"
    scope = " &gt; ".join([f"<B>{class_name}</B>", *steps])
    return (
        f"{scope}<br align='left'/>"
        f"drawn: {html_escape(count_line)}<br align='left'/>"
        f"whole model: {params_line}<br align='left'/>"
    )


def _ordinal(number: int) -> str:
    """Return ``1st``, ``2nd``, ``3rd``, ``4th`` ... for ``number``."""

    if 10 <= number % 100 <= 20:
        return f"{number}th"
    return f"{number}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th') }"


def reading_notes(node_attrs: Iterable[Mapping[str, Any]], module: Any) -> tuple[str, ...]:
    """Return the text rows of a focused draw's reading key.

    Each row explains one notation the picture uses, taking a real example
    from the drawn nodes where it can: the ``@`` address prefix, an op
    name's two counters, the output shape and size line, and call arguments.
    A whole-model draw gets no rows.
    """

    if focus_address(module) is None:
        return ()
    texts = [_TAG.sub(" ", str(attrs.get("label", ""))) for attrs in node_attrs]
    names = [str(attrs.get("name", "")) for attrs in node_attrs]
    notes = ["@name: the module at that address (a frame title, or a module drawn as one box)"]
    example = next(
        (m for m in map(_OP_NAME.match, names) if m and not m["kind"].startswith("_")), None
    )
    if example is not None:
        notes.append(
            f"{example.group(0)}: the {_ordinal(int(example['nth']))} {example['kind']} call,"
            f" op {example['op']} of the whole forward pass"
        )
    notes.append("(shape), size: the op's output tensor and its memory")
    if any("params:" in text for text in texts):
        notes.append("params: the weights the op used (filled as the key shows)")
    if any(_ARGUMENT.search(text) for text in texts):
        notes.append("name=value: an argument the op was called with")
    return tuple(notes)
