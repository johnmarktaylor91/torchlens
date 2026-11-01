"""Container-snapshot registration at module call entry and exit."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..._ir.container_registry import ModuleSite, Phase, Role, walk_container

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace


def _register_module_input_container_snapshots(
    trace: Trace,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    module_call_label: str,
) -> None:
    """Register tensor-bearing module input containers at call entry.

    Parameters
    ----------
    trace:
        Active trace.
    args:
        Positional module inputs.
    kwargs:
        Keyword module inputs.
    module_call_label:
        Stable module-call label for this invocation.
    """

    if not getattr(trace, "_capture_container_structure", False):
        return
    registry = trace._wrapper_runtime_ws.container_registry
    event_index = trace._raw_graph_ws.layer_counter
    for index, arg in enumerate(args):
        result = walk_container(arg, role=Role.CALL_INPUT, capability="full_spec")
        if result is None:
            continue
        registry.register_snapshot(
            arg,
            site=ModuleSite(module_call_label=module_call_label, position=("arg", index)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=event_index,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
    for key, value in kwargs.items():
        result = walk_container(value, role=Role.CALL_INPUT, capability="full_spec")
        if result is None:
            continue
        registry.register_snapshot(
            value,
            site=ModuleSite(module_call_label=module_call_label, position=("kwarg", key)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=event_index,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )


def _register_module_output_container_snapshot(
    trace: Trace,
    output: Any,
    *,
    module_call_label: str,
) -> None:
    """Register a tensor-bearing module output container at call exit.

    Parameters
    ----------
    trace:
        Active trace.
    output:
        Raw module output object.
    module_call_label:
        Stable module-call label for this invocation.
    """

    if not getattr(trace, "_capture_container_structure", False):
        return
    result = walk_container(output, role=Role.CALL_OUTPUT, capability="full_spec")
    if result is None:
        return
    trace._wrapper_runtime_ws.container_registry.register_snapshot(
        output,
        site=ModuleSite(module_call_label=module_call_label, position="return"),
        role=Role.CALL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=trace._raw_graph_ws.layer_counter,
        spec=result.spec,
        leaf_occurrences=result.leaf_occurrences,
        reconstructable=result.reconstructable,
    )
