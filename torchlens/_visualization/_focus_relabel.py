"""Relabelling a ``module=`` focused draw for its scope.

The header's scope line prints the focused module's enclosing path, so the
picture leaves it off: frame titles and ``@address`` rows are relative to it
(``@mlp``, ``@mlp.c_fc`` under ``GPT2LMHeadModel > transformer > h.5 > mlp``).
An op's shape row is drawn where it tells something new: it is left off when
every drawn input of the op (a drawn op, or the stub standing for an outside
one) has the same shape and size. ``strip_focus_ancestors`` applies both and
drops the ancestor frames, in the render IR and in the recorded statements
the emitted DOT reads.

Spellings here are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ._focus_scope import (
    drop_ancestor_regions,
    focus_address,
    focus_ancestor_keys,
    focus_parent_prefix,
    module_at,
    relative_address,
)
from ._label_format import format_memory, format_shape
from ._render_utils import make_module_cluster_label
from .node_spec import render_lines_to_html

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .render_ir import RenderIR, RenderIRDotStatement, RenderIRNode, RenderIRRegion

__all__ = [
    "relabel_focus_nodes",
    "relabel_focus_regions",
    "shape_size_row",
    "strip_focus_ancestors",
]


def relabel_focus_nodes(
    render_ir: RenderIR,
    trace: Trace,
    module: Any,
    node_calls: list[RenderIRDotStatement] | None = None,
    module_clusters: Mapping[str, Any] | None = None,
) -> RenderIR:
    """Return the IR with a focused draw's node labels rewritten for its scope.

    ``@address`` rows drop the scope prefix, and a drawn op's shape row is left
    off when every drawn input of the op (a drawn op, or the stub standing for
    an outside one) has the same shape and size. A whole-model draw, and a node
    whose label a spec function replaced, are returned unchanged. Labels already
    copied into ``node_calls`` and the ``module_clusters`` payloads are
    rewritten in place, so the emitted DOT agrees with the IR.
    """

    if focus_address(module) is None:
        return render_ir
    prefix = focus_parent_prefix(module)
    drawn = frozenset(node.name for node in render_ir.nodes)
    relabels: dict[str, str] = {}
    nodes = tuple(
        _relabelled_node(node, trace, prefix, drawn, relabels) for node in render_ir.nodes
    )
    if relabels:
        _rewrite_copied_labels(relabels, node_calls, module_clusters)
    return replace(render_ir, nodes=nodes)


def _rewrite_copied_labels(
    relabels: Mapping[str, str],
    node_calls: list[RenderIRDotStatement] | None,
    module_clusters: Mapping[str, Any] | None,
) -> None:
    """Apply ``relabels`` (old label to new) to the recorded statements and payload nodes."""

    if node_calls is not None:
        node_calls[:] = [_relabelled_call(call, relabels) for call in node_calls]
    for payload in (module_clusters or {}).values():
        for args in payload.get("nodes", ()):
            if args.get("label") in relabels:
                args["label"] = relabels[args["label"]]


def _relabelled_node(
    node: RenderIRNode,
    trace: Trace,
    prefix: str | None,
    drawn: frozenset[str],
    relabels: dict[str, str],
) -> RenderIRNode:
    """Return ``node`` with its scope-relative rows, recording old label to new in ``relabels``."""

    spec = node.node_spec
    if spec is None:
        return node
    lines = [_relative_row(row, prefix) for row in spec.lines]
    if node.kind == "raw_op":
        lines = _without_repeated_shape(lines, trace, node.source_label, drawn)
    if lines == list(spec.lines):
        return node
    old_label = render_lines_to_html(list(spec.lines))
    new_label = render_lines_to_html(lines)
    relabels[old_label] = new_label
    swap = {old_label: new_label}
    return replace(
        node,
        node_spec=replace(spec, lines=lines),
        node_calls=tuple(_relabelled_call(call, swap) for call in node.node_calls),
        owned_node_args=tuple(
            (owner, {**args, "label": new_label} if args.get("label") == old_label else args)
            for owner, args in node.owned_node_args
        ),
    )


def _relabelled_call(call: RenderIRDotStatement, swap: Mapping[str, str]) -> RenderIRDotStatement:
    """Return a recorded node statement with its label swapped per ``swap``, else ``call``."""

    if call.kind != "node" or dict(call.attrs).get("label") not in swap:
        return call
    attrs = tuple((name, swap[value] if name == "label" else value) for name, value in call.attrs)
    return replace(call, attrs=attrs)


def _relative_row(row: Any, prefix: str | None) -> Any:
    """Return an ``@address`` row relative to the scope prefix; other rows unchanged."""

    if prefix is None or not isinstance(row, str) or not row.startswith("@"):
        return row
    address, _, calls = row[1:].partition(":")
    return f"@{relative_address(address, prefix)}{':' if calls else ''}{calls}"


def shape_size_row(op: Any) -> str:
    """The ``(shape), size`` row of an op's output tensor, as its node label shows it."""

    return f"{format_shape(op.shape)}, {format_memory(op.activation_memory)}"


def _without_repeated_shape(
    lines: list[Any], trace: Trace, label: str | None, drawn: frozenset[str]
) -> list[Any]:
    """Drop the op's shape row when every drawn input of the op shares it."""

    op = _drawn_op(trace, label)
    if op is None:
        return lines
    row = shape_size_row(op)
    parents = [str(parent) for parent in getattr(op, "parents", ()) or ()]
    if row not in lines or not parents:
        return lines
    for parent in parents:
        parent_op = _drawn_op(trace, parent)
        if parent_op is None or shape_size_row(parent_op) != row or not _is_drawn(parent, drawn):
            return lines
    return [line for line in lines if line != row]


def _drawn_op(trace: Trace, label: str | None) -> Any | None:
    """Return the op behind a drawn node's source label, or ``None``."""

    if not label:
        return None
    try:
        op = trace[label]
    except (KeyError, ValueError, TypeError):
        return None
    return op if hasattr(op, "shape") and hasattr(op, "activation_memory") else None


def _is_drawn(label: str, drawn: frozenset[str]) -> bool:
    """Whether ``label`` is a drawn node, or a focus stub stands for it."""

    names = {label, label.split(":", 1)[0]}
    names |= {f"__module_focus_input_{name.replace(':', 'pass')}" for name in list(names)}
    return bool(names & drawn)


def relabel_focus_regions(render_ir: RenderIR, trace: Trace, module: Any) -> RenderIR:
    """Return the IR with a focused draw's frame titles relative to its scope.

    A frame title built the standard way (``@address`` and the module class)
    is rebuilt from the relative address; a title a module override replaced
    is kept.
    """

    prefix = focus_parent_prefix(module)
    if prefix is None:
        return render_ir
    regions = tuple(_relabelled_region(region, trace, prefix) for region in render_ir.regions)
    return replace(render_ir, regions=regions)


def _relabelled_region(region: RenderIRRegion, trace: Trace, prefix: str) -> RenderIRRegion:
    address = region.key.split(":", 1)[0]
    if region.kind != "module" or not region.label.startswith(address):
        return region
    module = module_at(trace, address)
    if module is None:
        return region
    label = relative_address(address, prefix) + region.label[len(address) :]
    standard = make_module_cluster_label(region.label, module.class_name)
    style = tuple(
        (name, make_module_cluster_label(label, module.class_name))
        if name == "label" and value == standard
        else (name, value)
        for name, value in region.style
    )
    return replace(region, label=label, style=style)


def strip_focus_ancestors(
    trace: Trace,
    render_ir: RenderIR,
    node_calls: list[RenderIRDotStatement],
    module_clusters: Mapping[str, Any],
    module: Any,
) -> tuple[RenderIR, frozenset[str]]:
    """Scope a focused draw's IR: relabel its nodes and frames, drop its ancestor frames.

    Node labels are rewritten in the IR, in the recorded top-level node
    statements (``node_calls``) and in the module-cluster payloads
    (``module_clusters``), the three places the emitted DOT reads them from.
    The ancestor keys come from the payloads and the IR regions both, so a
    frame known to either is dropped. Returns the IR and the dropped keys; a
    whole-model draw changes nothing.
    """

    if focus_address(module) is None:
        return render_ir, frozenset()
    render_ir = relabel_focus_nodes(render_ir, trace, module, node_calls, module_clusters)
    keys = [*module_clusters, *(region.key for region in render_ir.regions)]
    flatten = focus_ancestor_keys(keys, module)
    render_ir = drop_ancestor_regions(render_ir, flatten)
    return relabel_focus_regions(render_ir, trace, module), flatten
