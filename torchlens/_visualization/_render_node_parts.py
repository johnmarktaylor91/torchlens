"""Label parts for Graphviz nodes: integer ranges, buffer addresses, module-box outputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from ._render_common import GraphNode, RollingAnnotation


def _compact_int_ranges(values: Sequence[int]) -> str:
    """Return sorted integers in compact range notation.

    Parameters
    ----------
    values:
        Integer values to format.

    Returns
    -------
    str
        Comma-separated values and ranges, for example ``"1,2-4"``.
    """

    if not values:
        return ""
    sorted_values = sorted(set(values))
    ranges: list[str] = []
    start = sorted_values[0]
    previous = sorted_values[0]
    for value in sorted_values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _module_output_summary(trace: Trace, address_w_pass: str, node: GraphNode) -> tuple[Any, Any]:
    """Return the output shape and memory a collapsed module box shows.

    The box reads the module call's last output op, falling back to the node
    that triggered the collapse when the call or its output is not recorded.

    Parameters
    ----------
    trace:
        Trace owning the module calls and ops.
    address_w_pass:
        Module address with its call index.
    node:
        Node triggering the collapse.

    Returns
    -------
    tuple[Any, Any]
        Output shape (``()`` when unknown) and activation memory (``"0 B"`` when unknown).
    """

    try:
        module_call = trace.module_calls[address_w_pass]
        module_output_layer: GraphNode = trace.ops[module_call.output_ops[-1]]
    except (KeyError, IndexError):
        module_output_layer = node
    module_output_shape = getattr(module_output_layer, "shape", None)
    if module_output_shape is None:
        module_output_shape = getattr(module_output_layer, "out_shape", None)
    module_output_shape = module_output_shape or ()
    module_output_fsize = record_member(module_output_layer, "activation_memory", None)
    if module_output_fsize is None:
        module_output_fsize = "0 B"
    return module_output_shape, module_output_fsize


def _buffer_node_address(
    trace: Trace, source_node: Any, annotation: RollingAnnotation | None
) -> str:
    """Return the ``address`` or ``address:version`` text a buffer node shows.

    Parameters
    ----------
    trace:
        Trace whose buffer call counts decide whether a version is shown.
    source_node:
        Unwrapped buffer node.
    annotation:
        Rolled annotation of the node, or ``None``.

    Returns
    -------
    str
        Buffer address, suffixed by its version range or pass when ambiguous.
    """

    if annotation is not None and annotation.buffer_versions:
        return f"{source_node.buffer_address}:{_compact_int_ranges(annotation.buffer_versions)}"
    if trace._buffer_num_calls[source_node.buffer_address] == 1:
        return source_node.buffer_address
    return f"{source_node.buffer_address}:{record_member(source_node, 'buffer_pass')}"
