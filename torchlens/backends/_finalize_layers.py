"""Layer attachment helpers for single-pass trace finalization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._data_classes.layer import Layer
from .._data_classes.trace import Trace


def _attach_op_to_layer(
    trace: Trace,
    op_log: Any,
    enrich_layer: Callable[[Any, Any], None] | None,
) -> None:
    """Attach one finalized op to its layer log, creating the layer on first sight."""

    layer_log = trace.layer_logs.get(op_log.layer_label)
    layer_created = layer_log is None
    if layer_log is None:
        layer_log = Layer(op_log)
    layer_log.ops[op_log.pass_index] = op_log
    layer_log.call_labels.append(op_log.label)
    if op_log.num_passes != 1:
        layer_log.num_passes = op_log.num_passes
    if layer_created:
        if enrich_layer is not None:
            enrich_layer(layer_log, op_log)
        trace.layer_logs[op_log.layer_label] = layer_log


def _count_trace_ops(trace: Trace) -> int:
    """Count the ops that are neither inputs, outputs nor buffers."""

    return sum(
        1
        for op_log in trace.layer_list
        if not (op_log.is_input or op_log.is_output or op_log.is_buffer)
    )
