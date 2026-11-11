"""Per-op join cells shared by step-0 ingest (``_materialize._apply_join_cells``).

Two groups of join cells read the same inputs for every op: the parameter
cells, which are constant empties for an op that uses no parameter, and the
display address plus promoted buffer source, which both read the output
payload's ``TensorMeta``. Computing each group in one place reads the meta once
and skips the parameter resolution for parameter-free ops; the values are the
ones the inline join wrote.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import prod
from typing import Any

from .._utils._tensor_label import get_tensor_meta

__all__: tuple[str, ...] = ()


def param_join_cells(param_logs_registry: Any, event: Any) -> dict[str, object]:
    """Return the parameter join cells for one record.

    Parameters
    ----------
    param_logs_registry
        The trace's ``param_logs`` registry (declared input).
    event
        Journal op record (read through the strict protocol).

    Returns
    -------
    dict[str, object]
        ``parent_params`` through ``param_memory``, each container fresh for this op.
    """

    params = tuple(event.params)
    if not params:
        return {
            "parent_params": [],
            "_param_barcodes": [],
            "parent_param_ops": {},
            "_param_logs": [],
            "param_shapes": [],
            "num_params": 0,
            "num_params_trainable": 0,
            "num_params_frozen": 0,
            "param_memory": 0,
        }
    param_logs: list[Any] = []
    for param in params:
        address = getattr(param, "address", None)
        if address is not None and address in param_logs_registry:
            param_logs.append(param_logs_registry[address])
    resolved_param_addresses = {log.address for log in param_logs}
    resolved_params = tuple(param for param in params if param.address in resolved_param_addresses)
    parent_params = [
        parent_param
        for param, parent_param in zip(params, event.parent_params, strict=False)
        if param.address in resolved_param_addresses
    ]
    param_shapes = [param.shape for param in resolved_params]
    return {
        "parent_params": parent_params,
        "_param_barcodes": [param.barcode for param in resolved_params],
        "parent_param_ops": {param.barcode: event.pass_index for param in resolved_params},
        "_param_logs": param_logs,
        "param_shapes": param_shapes,
        "num_params": sum(prod(shape) for shape in param_shapes if shape is not None),
        "num_params_trainable": sum(log.num_params for log in param_logs if log.is_trainable),
        "num_params_frozen": sum(log.num_params for log in param_logs if not log.is_trainable),
        "param_memory": sum(int(log.param_memory) for log in param_logs),
    }


def address_and_buffer_source(
    event: Any,
    payload: object,
    preferred: str | None,
    op_event_labels: Mapping[str, Any] | set[str],
) -> tuple[str | None, str | None]:
    """Return the display address and the promoted buffer source for one record.

    Parameters
    ----------
    event
        Journal op record.
    payload
        The record's output tensor payload.
    preferred
        The registered buffer address, else the capture-recorded one; when
        truthy it is the display address.
    op_event_labels
        Raw labels present in the materialized event stream.

    Returns
    -------
    tuple[str | None, str | None]
        ``preferred`` when truthy, else the payload meta's address, the record
        context's input or output address, or the innermost module's address;
        and the meta's buffer source when it names a materialized op.
    """

    meta = get_tensor_meta(payload)
    buffer_source = None
    if meta is not None and meta.buffer_source in op_event_labels:
        buffer_source = meta.buffer_source
    if preferred:
        return preferred, buffer_source
    if meta is not None and meta.address is not None:
        return meta.address, buffer_source
    record_context = getattr(event, "record_context", None)
    input_output_address = getattr(record_context, "input_output_address", None)
    if isinstance(input_output_address, str):
        return input_output_address, buffer_source
    module_stack = event.module_stack
    if module_stack:
        return module_stack[-1].address, buffer_source
    return None, buffer_source
