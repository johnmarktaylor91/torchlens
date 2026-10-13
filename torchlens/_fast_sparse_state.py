"""Cached-state safety analysis for the guarded fast sparse engine.

The fast sparse session (``_fast_run._FastSparseSession``) caches staged
parameter and buffer state across iterations. That cache is safe only when no
recorded call can write it: an in-place call whose target aliases staged state,
or a functional normalization call that may update running statistics. This
module finds those calls from the frozen sparse descriptor alone.
"""

from __future__ import annotations

from ._runnable_execution import _VIEW_OP_QUALNAMES, _decode_literal, _mutation_target_slot_id
from .runnable import RunnableCallDescriptor, SparseRunDescriptor, is_mode_sensitive_qualname


def _state_storage_closure(
    descriptor: SparseRunDescriptor,
    state_slot_ids: set[str],
) -> set[str]:
    """Return state slots plus every slot reachable through alias-producing calls.

    Parameters
    ----------
    descriptor:
        Frozen sparse replay descriptor.
    state_slot_ids:
        Parameter and buffer slot identifiers that seed sparse execution.

    Returns
    -------
    set[str]
        State-derived slots whose storage may alias cached staged state.
    """

    registry_qualnames = {
        entry.registry_id: entry.key.qualname for entry in descriptor.callable_registry
    }
    closure = set(state_slot_ids)
    changed = True
    while changed:
        changed = False
        for slot in descriptor.tensor_slots:
            if slot.version_of in closure and slot.slot_id not in closure:
                closure.add(slot.slot_id)
                changed = True
        for call in descriptor.calls:
            if registry_qualnames.get(call.registry_id) not in _VIEW_OP_QUALNAMES:
                continue
            if not any(argument.slot_id in closure for argument in call.tensor_arguments):
                continue
            for output_slot_id in call.output_slot_ids:
                if output_slot_id not in closure:
                    closure.add(output_slot_id)
                    changed = True
    return closure


def _literal_bool_argument(call: RunnableCallDescriptor, name: str) -> bool | None:
    """Return a recorded boolean call argument by keyword or positional name.

    Parameters
    ----------
    call:
        Frozen sparse call recipe.
    name:
        Callable parameter name to resolve.

    Returns
    -------
    bool | None
        Recorded boolean value, or ``None`` when absent or non-boolean.
    """

    try:
        positional_index = call.argument_names.index(name)
    except ValueError:
        positional_index = None
    for literal in call.literal_arguments:
        path = tuple(literal.argument_path)
        is_keyword = len(path) == 2 and path == ("kwargs", name)
        is_positional = (
            positional_index is not None and len(path) == 2 and path == ("args", positional_index)
        )
        if not (is_keyword or is_positional):
            continue
        value = _decode_literal(literal.value)
        return value if isinstance(value, bool) else None
    return None


def _normalization_call_may_mutate_state(
    call: RunnableCallDescriptor,
    qualname: str | None,
    state_slot_ids: set[str],
) -> bool:
    """Return whether a functional normalization call may update running state.

    Parameters
    ----------
    call:
        Frozen sparse call recipe.
    qualname:
        Resolved callable qualname from the registry.
    state_slot_ids:
        Parameter and buffer slot identifiers that seed sparse execution.

    Returns
    -------
    bool
        ``True`` when cached running statistics could be updated as a hidden
        side effect. Unknown mode flags fail closed.
    """

    if not is_mode_sensitive_qualname(qualname):
        return False
    if not any(argument.slot_id in state_slot_ids for argument in call.tensor_arguments):
        return False
    tail = (qualname or "").rsplit(".", 1)[-1].removesuffix("_")
    mode_argument = "use_input_stats" if tail.endswith("instance_norm") else "training"
    return _literal_bool_argument(call, mode_argument) is not False


def _state_mutating_call_ids(
    descriptor: SparseRunDescriptor,
    state_slot_ids: set[str],
) -> tuple[str, ...]:
    """Return calls that make cached sparse state unsafe across iterations.

    Parameters
    ----------
    descriptor:
        Frozen sparse replay descriptor.
    state_slot_ids:
        Parameter and buffer slot identifiers that seed sparse execution.

    Returns
    -------
    tuple[str, ...]
        Ordered call identifiers that explicitly or implicitly mutate state.
    """

    registry_qualnames = {
        entry.registry_id: entry.key.qualname for entry in descriptor.callable_registry
    }
    state_storage_ids = _state_storage_closure(descriptor, state_slot_ids)
    return tuple(
        call.call_id
        for call in descriptor.calls
        if (call.is_inplace and _mutation_target_slot_id(call) in state_storage_ids)
        or _normalization_call_may_mutate_state(
            call,
            registry_qualnames.get(call.registry_id),
            state_storage_ids,
        )
    )
