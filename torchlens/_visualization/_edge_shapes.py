"""Optional producer-shape annotations on resolved forward edges."""

from __future__ import annotations

from dataclasses import replace
from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .render_ir import RenderIR, RenderIRDotStatement


def _producer_ops(trace: Trace, label: str, rolled: bool) -> tuple[Any, ...]:
    """Return the ops a rendered producer label stands for.

    A bare label, or any label in a rolled render, names the whole layer, which
    ``trace[...]`` refuses for a multi-pass layer; ``trace.layers[...]`` holds every pass.
    """
    if label in trace.layers and (rolled or ":" not in label):
        return tuple(trace.layers[label].ops)
    entry = trace[label]
    return tuple(entry.ops) if rolled and hasattr(entry, "ops") else (entry,)


def _shape_text(trace: Trace, labels: tuple[str, ...], rolled: bool) -> str:
    """Read each represented producer's own output slot, never the call container."""
    shapes = []
    for label in labels:
        for op in _producer_ops(trace, label, rolled):
            shape = op.shape
            if shape is not None:
                text = "[" + ", ".join(str(dim) for dim in shape) + "]"
                if text not in shapes:
                    shapes.append(text)
    return " / ".join(shapes)


def _label_attrs(attrs: tuple[tuple[str, Any], ...], shape: str) -> tuple[tuple[str, Any], ...]:
    """Compose a shape with existing argument, branch or multiplicity labels."""
    values = dict(attrs)
    previous = values.get("label", "")
    if previous.startswith("<") and previous.endswith(">"):
        # A nested table preserves the existing HTML label's complete structure.
        values["label"] = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            f"<TR><TD>{previous[1:-1]}</TD></TR><TR><TD>{escape(shape)}</TD></TR></TABLE>>"
        )
    else:
        values["label"] = f"{previous}\n{shape}" if previous else shape
    return tuple(values.items())


def _shape_statements(
    statements: tuple[RenderIRDotStatement, ...], shapes: dict[tuple[str, str], str]
) -> tuple[RenderIRDotStatement, ...]:
    """Decorate emitted forward statements, including edges inside module clusters."""
    result = []
    for statement in statements:
        values = dict(statement.attrs)
        tail = statement.args[0] if statement.args else values.get("tail_name", "")
        head = statement.args[1] if len(statement.args) > 1 else values.get("head_name", "")
        shape = shapes.get((tail, head)) if statement.kind == "edge" else None
        attrs = _label_attrs(statement.attrs, shape) if shape else statement.attrs
        result.append(
            replace(statement, attrs=attrs, children=_shape_statements(statement.children, shapes))
        )
    return tuple(result)


def _with_edge_shapes(trace: Trace, ir: RenderIR) -> RenderIR:
    """Resolve the opt-in channel once in IR, keeping default statements identical."""
    if not ir.context.show_edge_shapes:
        return ir
    shapes: dict[tuple[str, str], str] = {}
    edges = []
    for edge in ir.edges:
        if edge.occurrence_key and edge.occurrence_key[0] in {"grad", "correspondence"}:
            edges.append(edge)
            continue
        shape = _shape_text(trace, edge.source_originals, ir.context.vis_mode == "rolled")
        key = (edge.tail_name or edge.source_unit, edge.head_name or edge.target_unit)
        # Projection can merge several producers into the same visible edge.
        if shape and shape not in shapes.get(key, "").split(" / "):
            shapes[key] = " / ".join(filter(None, (shapes.get(key), shape)))
        edges.append(replace(edge, attrs=_label_attrs(edge.attrs, shape)) if shape else edge)
    return replace(
        ir, edges=tuple(edges), dot_statements=_shape_statements(ir.dot_statements, shapes)
    )
