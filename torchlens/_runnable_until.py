"""Stop-frontier and dependency-closure arithmetic for ``run(until=...)``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._runnable import RunnableCallDescriptor


def _until_stop_raw_index(resolved: list[Any]) -> int:
    """The stop frontier: the max recorded raw index across resolved sites."""

    from ._errors import InvalidArgumentError

    stop_raw_index = -1
    for layer in resolved:
        for op in getattr(layer, "ops", None) or (layer,):
            raw_index = getattr(op, "raw_index", None)
            if raw_index is not None:
                stop_raw_index = max(stop_raw_index, int(raw_index))
    if stop_raw_index < 0:
        raise InvalidArgumentError(
            "until= selection resolved to sites without recorded operation "
            "indexes; the stop frontier is undecidable",
            code="run_until_form_invalid",
            remedy="pass sites that resolve to captured operations",
            argument="until",
        )
    return stop_raw_index


def _until_execution_partition(
    trace: Any, stop_raw_index: int
) -> tuple[list[str], list[str], str | None]:
    """Partition the source layers into executed/skipped at the stop frontier."""

    executed: list[str] = []
    skipped: list[str] = []
    stopped_at: str | None = None
    for layer in trace.layer_list:
        raw_index = getattr(layer, "raw_index", None)
        raw_label = layer._layer_label_raw
        if layer.layer_type == "output" or raw_index is None or int(raw_index) > stop_raw_index:
            skipped.append(raw_label)
            continue
        executed.append(raw_label)
        if int(raw_index) == stop_raw_index:
            stopped_at = layer.layer_label
    return executed, skipped, stopped_at


def _until_dependency_closure(
    *,
    prefix: tuple[RunnableCallDescriptor, ...],
    by_id: dict[str, RunnableCallDescriptor],
    op_label_to_call: dict[str, str],
    seed_call_ids: list[str],
    state_slot_ids: set[str],
) -> set[str]:
    """Widened dependency closure: C1 tensor deps, C5 control-witness deps,
    then the C3 declared-state widening over shared state slots."""

    closure: set[str] = set()
    frontier = list(seed_call_ids)
    while frontier:
        call_id = frontier.pop()
        if call_id in closure or call_id not in by_id:
            continue
        closure.add(call_id)
        call = by_id[call_id]
        frontier.extend(call.parent_call_ids)
        for edge in call.control_dependencies:
            parent_call_id = op_label_to_call.get(edge.parent_op_label)
            if parent_call_id is not None:
                frontier.append(parent_call_id)
    closure_state_slots = {
        argument.slot_id
        for call_id in closure
        for argument in by_id[call_id].tensor_arguments
        if argument.slot_id in state_slot_ids
    }
    if closure_state_slots:
        for call in prefix:
            if call.call_id in closure:
                continue
            if any(argument.slot_id in closure_state_slots for argument in call.tensor_arguments):
                closure.add(call.call_id)
    return closure
