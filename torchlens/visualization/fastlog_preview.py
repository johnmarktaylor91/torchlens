"""Preview fastlog predicate decisions on a full ModelLog graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..fastlog._predicate import _normalize_capture_decision
from ..fastlog._record_context import _build_record_context
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


def _module_stack_from_layer(layer_pass_log: Any) -> tuple[ModuleStackFrame, ...]:
    """Build a synthetic module stack from a ``LayerPassLog``.

    Parameters
    ----------
    layer_pass_log:
        Layer pass object from a fully postprocessed ``ModelLog``.

    Returns
    -------
    tuple[ModuleStackFrame, ...]
        Synthetic stack frames suitable for preview predicates.
    """

    frames: list[ModuleStackFrame] = []
    module_addresses = tuple(getattr(layer_pass_log, "containing_modules", ()) or ())
    for index, module_address in enumerate(module_addresses, start=1):
        module_type = ""
        source_model_log = getattr(layer_pass_log, "source_model_log", None)
        if source_model_log is not None:
            modules = getattr(source_model_log, "_module_logs", {})
            try:
                module_log = modules[module_address]
            except (KeyError, TypeError):
                module_log = None
            module_type = str(getattr(module_log, "module_type", ""))
        frames.append(
            ModuleStackFrame(
                module_address=str(module_address),
                module_type=module_type,
                module_id=0,
                pass_index=index,
            )
        )
    return tuple(frames)


def _kind_from_layer(layer_pass_log: Any) -> str:
    """Return the fastlog event kind represented by a layer pass."""

    if bool(getattr(layer_pass_log, "is_input_layer", False)):
        return "input"
    if bool(getattr(layer_pass_log, "is_buffer_layer", False)):
        return "buffer"
    return "op"


def _context_from_layer(
    layer_pass_log: Any,
    *,
    history: tuple[RecordContext, ...],
    event_index: int,
    op_counts: dict[str, int],
) -> RecordContext:
    """Build a preview ``RecordContext`` for a layer pass."""

    module_stack = _module_stack_from_layer(layer_pass_log)
    module_frame = module_stack[-1] if module_stack else None
    raw_label = getattr(layer_pass_log, "tensor_label_raw", None)
    layer_type = getattr(layer_pass_log, "layer_type", None)
    if isinstance(layer_type, str):
        op_counts[layer_type] = op_counts.get(layer_type, 0) + 1
    return _build_record_context(
        kind=_kind_from_layer(layer_pass_log),  # type: ignore[arg-type]
        layer_pass_log_or_op_data={
            "label": getattr(layer_pass_log, "layer_label", raw_label),
            "raw_label": raw_label,
            "tensor_label_raw": raw_label,
            "creation_order": getattr(layer_pass_log, "creation_order", None),
            "layer_type": layer_type,
            "layer_type_num": getattr(layer_pass_log, "layer_type_num", None),
            "func_name": getattr(layer_pass_log, "func_name", None),
            "parent_labels": tuple(getattr(layer_pass_log, "parent_layers", ()) or ()),
            "tensor_shape": getattr(layer_pass_log, "tensor_shape", None),
            "tensor_dtype": getattr(layer_pass_log, "tensor_dtype", None),
            "output_index": getattr(layer_pass_log, "iterable_output_index", None),
            "is_bottom_level_func": getattr(layer_pass_log, "is_bottom_level_func", None),
            "input_output_address": getattr(layer_pass_log, "io_role", None),
            "module_address": module_frame.module_address if module_frame else None,
            "module_type": module_frame.module_type if module_frame else None,
            "module_pass_index": module_frame.pass_index if module_frame else None,
        },
        module_stack=module_stack,
        history=history,
        op_counts=op_counts,
        pass_index=max(int(getattr(layer_pass_log, "pass_num", 1)) - 1, 0),
        event_index=event_index,
        op_index=int(getattr(layer_pass_log, "operation_num", event_index)),
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
    layer_pass_log: Any,
    ctx: RecordContext,
    predicate: Predicate | None,
) -> PreviewNode:
    """Evaluate one preview predicate and capture display metadata."""

    if not bool(getattr(layer_pass_log, "has_input_ancestor", True)):
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
    if spec.save_activation or spec.save_metadata:
        return PreviewNode(ctx=ctx, decision=Decision.KEPT)
    return PreviewNode(ctx=ctx, decision=Decision.REJECTED)


def _build_preview_nodes(model_log: Any, predicate: Predicate | None) -> dict[str, PreviewNode]:
    """Evaluate preview decisions for all layer passes in a model log."""

    history: list[RecordContext] = []
    op_counts: dict[str, int] = {}
    preview_nodes: dict[str, PreviewNode] = {}
    for event_index, layer_pass_log in enumerate(model_log.layer_list, start=1):
        ctx = _context_from_layer(
            layer_pass_log,
            history=tuple(history),
            event_index=event_index,
            op_counts=op_counts,
        )
        preview_node = _evaluate_preview_node(layer_pass_log, ctx, predicate)
        preview_nodes[getattr(layer_pass_log, "layer_label_no_pass", ctx.label)] = preview_node
        preview_nodes[getattr(layer_pass_log, "layer_label", ctx.label)] = preview_node
        history.append(ctx)
    return preview_nodes


def _append_predicate_input_lines(lines: list[str], ctx: RecordContext) -> None:
    """Append predicate input details to a node label."""

    lines.append(f"op_type: {ctx.layer_type}")
    lines.append(f"raw_label: {ctx.raw_label}")
    lines.append(f"module_path: {ctx.module_address}")


def _append_module_event_lines(lines: list[str], layer_log: Any) -> None:
    """Append module entry/exit details when available."""

    entered = tuple(getattr(layer_log, "modules_entered", ()) or ())
    exited = tuple(getattr(layer_log, "modules_exited", ()) or ())
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
    """Build the runtime ``node_spec_fn`` used by ``render_graph``."""

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
    model_log: Any,
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
    """Render a full ``ModelLog`` colored by fastlog predicate decisions.

    Parameters
    ----------
    model_log:
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
        Forwarded to ``ModelLog.render_graph``.

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
    preview_nodes = _build_preview_nodes(model_log, resolved_predicate)
    node_spec_fn = _make_node_spec_fn(
        preview_nodes,
        color_kept=color_kept,
        color_rejected=color_rejected,
        color_unreachable=color_unreachable,
        color_predicate_error=color_predicate_error,
        show_predicate_inputs=show_predicate_inputs,
        show_module_events=show_module_events,
    )
    return model_log.render_graph(node_spec_fn=node_spec_fn, **render_kwargs)
