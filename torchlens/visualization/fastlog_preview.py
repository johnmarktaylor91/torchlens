"""Preview fastlog predicate decisions on a full Trace graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from ..capture.predicates import _normalize_capture_decision
from ..capture.projections import _build_record_context
from ..fastlog.exceptions import RecordContextFieldError
from ..fastlog.types import CaptureDecision, ModuleStackFrame, RecordContext
from .node_spec import NodeSpec


class Decision(Enum):
    """Preview rendering decision for one layer."""

    KEPT = "kept"
    REJECTED = "rejected"
    UNREACHABLE = "unreachable"
    EXCEPTION = "exception"


@dataclass(frozen=True, slots=True)
class PreviewNode:
    """Cached preview state for one rendered layer."""

    ctx: RecordContext
    decision: Decision
    tooltip: str | None = None


Predicate = Callable[[RecordContext], CaptureDecision]


def _module_stack_from_layer(op_log: Any) -> tuple[ModuleStackFrame, ...]:
    """Build a synthetic module stack from a ``Op``.

    Parameters
    ----------
    op_log:
        Layer pass object from a fully postprocessed ``Trace``.

    Returns
    -------
    tuple[ModuleStackFrame, ...]
        Synthetic stack frames suitable for preview predicates.
    """

    frames: list[ModuleStackFrame] = []
    addresses = tuple(getattr(op_log, "modules", ()) or ())
    for index, address in enumerate(addresses, start=1):
        module_type = ""
        source_trace = getattr(op_log, "source_trace", None)
        if source_trace is not None:
            modules = getattr(source_trace, "_module_logs", {})
            try:
                module_log = modules[address]
            except (KeyError, TypeError):
                module_log = None
            module_type = str(getattr(module_log, "module_type", ""))
        frames.append(
            ModuleStackFrame(
                address=str(address),
                module_type=module_type,
                module_id=0,
                pass_index=index,
            )
        )
    return tuple(frames)


def _kind_from_layer(op_log: Any) -> str:
    """Return the fastlog event kind represented by a layer pass."""

    if bool(getattr(op_log, "is_input", False)):
        return "input"
    if bool(getattr(op_log, "is_buffer", False)):
        return "buffer"
    return "op"


def _context_from_layer(
    op_log: Any,
    *,
    history: tuple[RecordContext, ...],
    event_index: int,
    op_counts: dict[str, int],
) -> RecordContext:
    """Build a preview ``RecordContext`` for a layer pass."""

    module_stack = _module_stack_from_layer(op_log)
    module_frame = module_stack[-1] if module_stack else None
    raw_label = getattr(op_log, "_label_raw", None)
    layer_type = getattr(op_log, "layer_type", None)
    if isinstance(layer_type, str):
        op_counts[layer_type] = op_counts.get(layer_type, 0) + 1
    return _build_record_context(
        kind=_kind_from_layer(op_log),  # type: ignore[arg-type]
        op_log_or_op_data={
            "label": getattr(op_log, "layer_label", raw_label),
            "raw_label": raw_label,
            "_label_raw": raw_label,
            "raw_index": getattr(op_log, "raw_index", None),
            "layer_type": layer_type,
            "type_index": getattr(op_log, "type_index", None),
            "func_name": getattr(op_log, "func_name", None),
            "parent_labels": tuple(getattr(op_log, "parents", ()) or ()),
            "shape": getattr(op_log, "shape", None),
            "dtype": getattr(op_log, "dtype", None),
            "output_index": getattr(op_log, "multi_output_index", None),
            "is_bottom_level_func": getattr(op_log, "is_bottom_level_func", None),
            "input_output_address": getattr(op_log, "io_role", None),
            "address": module_frame.address if module_frame else None,
            "module_type": module_frame.module_type if module_frame else None,
            "module_pass_index": module_frame.pass_index if module_frame else None,
        },
        module_stack=module_stack,
        history=history,
        op_counts=op_counts,
        pass_index=max(int(getattr(op_log, "call_index", 1)) - 1, 0),
        event_index=event_index,
        step_index=int(getattr(op_log, "step_index", event_index)),
        time_since_pass_start=0.0,
        include_source_events=True,
    )


def _select_predicate(
    predicate: Predicate | None,
    keep_op: Predicate | None,
    keep_module: Predicate | None,
) -> Predicate | None:
    """Resolve the predicate callable for previewed layer events."""

    if predicate is not None:
        return predicate
    if keep_op is not None:
        return keep_op
    return keep_module


def _evaluate_preview_node(
    op_log: Any,
    ctx: RecordContext,
    predicate: Predicate | None,
) -> PreviewNode:
    """Evaluate one preview predicate and capture display metadata."""

    if not bool(getattr(op_log, "has_input_ancestor", True)):
        return PreviewNode(ctx=ctx, decision=Decision.UNREACHABLE)
    if predicate is None:
        return PreviewNode(ctx=ctx, decision=Decision.REJECTED)
    try:
        result = predicate(ctx)
        spec = _normalize_capture_decision(result, ctx, False)
    except RecordContextFieldError as exc:
        return PreviewNode(ctx=ctx, decision=Decision.EXCEPTION, tooltip=str(exc))
    except Exception as exc:  # noqa: BLE001
        return PreviewNode(ctx=ctx, decision=Decision.EXCEPTION, tooltip=repr(exc))
    if spec.save_out or spec.save_metadata:
        return PreviewNode(ctx=ctx, decision=Decision.KEPT)
    return PreviewNode(ctx=ctx, decision=Decision.REJECTED)


def _build_preview_nodes(trace: Any, predicate: Predicate | None) -> dict[str, PreviewNode]:
    """Evaluate preview decisions for all layer ops in a model log."""

    history: list[RecordContext] = []
    op_counts: dict[str, int] = {}
    preview_nodes: dict[str, PreviewNode] = {}
    for event_index, op_log in enumerate(trace.layer_list, start=1):
        ctx = _context_from_layer(
            op_log,
            history=tuple(history),
            event_index=event_index,
            op_counts=op_counts,
        )
        preview_node = _evaluate_preview_node(op_log, ctx, predicate)
        preview_nodes[getattr(op_log, "layer_label_no_pass", ctx.label)] = preview_node
        preview_nodes[getattr(op_log, "layer_label", ctx.label)] = preview_node
        history.append(ctx)
    return preview_nodes


def _append_predicate_input_lines(lines: list[str], ctx: RecordContext) -> None:
    """Append predicate input details to a node label."""

    lines.append(f"op_type: {ctx.layer_type}")
    lines.append(f"raw_label: {ctx.raw_label}")
    lines.append(f"module_path: {ctx.address}")


def _append_module_event_lines(lines: list[str], layer_log: Any) -> None:
    """Append module entry/exit details when available."""

    entered = tuple(getattr(layer_log, "modules_entered", ()) or ())
    exited = tuple(getattr(layer_log, "output_of_modules", ()) or ())
    if entered:
        lines.append("module_enter: " + ", ".join(str(item) for item in entered))
    if exited:
        lines.append("module_exit: " + ", ".join(str(item) for item in exited))


def _make_node_spec_fn(
    preview_nodes: dict[str, PreviewNode],
    *,
    color_kept: str,
    color_rejected: str,
    color_unreachable: str,
    color_predicate_error: str,
    show_predicate_inputs: bool,
    show_module_events: bool,
) -> Callable[[Any, NodeSpec], NodeSpec | None]:
    """Build the runtime ``node_spec_fn`` used by ``draw``."""

    colors = {
        Decision.KEPT: color_kept,
        Decision.REJECTED: color_rejected,
        Decision.UNREACHABLE: color_unreachable,
        Decision.EXCEPTION: color_predicate_error,
    }

    def node_spec_fn(layer_log: Any, default_spec: NodeSpec) -> NodeSpec:
        """Paint one node from cached preview state."""

        preview_node = preview_nodes.get(getattr(layer_log, "layer_label", ""))
        lines = list(default_spec.lines)
        if preview_node is None:
            return default_spec
        lines.append(f"fastlog: {preview_node.decision.value}")
        if show_predicate_inputs:
            _append_predicate_input_lines(lines, preview_node.ctx)
        if show_module_events:
            _append_module_event_lines(lines, layer_log)
        return default_spec.replace(
            lines=lines,
            fillcolor=colors[preview_node.decision],
            tooltip=preview_node.tooltip or default_spec.tooltip,
        )

    return node_spec_fn


def preview_fastlog(
    trace: Any,
    predicate: Predicate | None = None,
    keep_op: Predicate | None = None,
    keep_module: Predicate | None = None,
    color_kept: str = "#98FB98",
    color_rejected: str = "#E6E6E6",
    color_unreachable: str = "#F7D460",
    color_predicate_error: str = "#FF7AB6",
    show_predicate_inputs: bool = True,
    show_module_events: bool = True,
    **render_kwargs: Any,
) -> str:
    """Render a full ``Trace`` colored by fastlog predicate decisions.

    Parameters
    ----------
    trace:
        Fully logged model graph to preview.
    predicate, keep_op, keep_module:
        Predicate callables that receive synthesized ``RecordContext`` objects.
    color_kept, color_rejected, color_unreachable, color_predicate_error:
        Fill colors for preview decisions.
    show_predicate_inputs:
        Whether to append selected predicate fields to node labels.
    show_module_events:
        Whether to append module entry/exit fields already present on layer logs.
    **render_kwargs:
        Forwarded to ``Trace.draw``.

    Returns
    -------
    str
        Graphviz DOT source.
    """

    # TODO(fastlog-dagua): add native dagua support after the v1 Graphviz preview ships.
    if render_kwargs.get("vis_renderer") == "dagua":
        raise NotImplementedError(
            "fastlog preview currently supports Graphviz only; dagua support is planned."
        )
    resolved_predicate = _select_predicate(predicate, keep_op, keep_module)
    preview_nodes = _build_preview_nodes(trace, resolved_predicate)
    node_spec_fn = _make_node_spec_fn(
        preview_nodes,
        color_kept=color_kept,
        color_rejected=color_rejected,
        color_unreachable=color_unreachable,
        color_predicate_error=color_predicate_error,
        show_predicate_inputs=show_predicate_inputs,
        show_module_events=show_module_events,
    )
    return cast(str, trace.draw(node_spec_fn=node_spec_fn, **render_kwargs))
