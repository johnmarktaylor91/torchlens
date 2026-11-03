"""Module-output intervention dispatch for Paddle forward post-hooks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ... import _state
from ..._ir.op_record import amend_module_exit_intervention
from ..._ir.selector_eval import (
    module_address_matches,
    module_union_addresses,
    selector_contains_kind,
)
from ..._selection.selectors import FuncSelector
from ...intervention.spec import when
from .._protocols.paddle import PADDLE_REFUSAL_OPTION
from ..registry import BackendUnsupportedError


def _validate_module_selector(selector: Any) -> None:
    """Refuse mixed module trees before any forward executes."""

    if selector_contains_kind(selector, "module") and module_union_addresses(selector) is None:
        raise BackendUnsupportedError(
            "paddle backend module interventions require tl.module(...) or a union "
            "of module selectors; mixed or negated module selectors are unsupported.",
            code=PADDLE_REFUSAL_OPTION,
            option="intervene",
        )


def _apply_module_exit(trace: Any, frame: Any, output: Any) -> Any:
    """Apply one selected module call to all its output tensor leaves."""

    runtime = getattr(trace, "_paddle_intervention_runtime", None)
    predicate = getattr(runtime, "intervene", None)
    addresses = module_union_addresses(getattr(predicate, "selector", None))
    if not addresses or not any(
        module_address_matches((frame.address, frame.call_index), address) for address in addresses
    ):
        return output
    backend = runtime._backend
    by_id = {}
    replacements = {}
    for path, tensor in backend._iter_tensors_with_paths(output):
        if id(tensor) not in by_id:
            by_id[id(tensor)] = _emit_module_leaf(trace, runtime, tensor)
        replacements[path] = by_id[id(tensor)]
    return _replace_leaves(output, replacements)


def _replace_leaves(output: Any, replacements: dict[Any, Any], path: tuple[Any, ...] = ()) -> Any:
    """Rebuild the supported output containers with edited leaves."""

    if path in replacements:
        return replacements[path]
    if isinstance(output, tuple):
        return tuple(
            _replace_leaves(item, replacements, (*path, i)) for i, item in enumerate(output)
        )
    if isinstance(output, list):
        return [_replace_leaves(item, replacements, (*path, i)) for i, item in enumerate(output)]
    if isinstance(output, dict):
        return {
            key: _replace_leaves(item, replacements, (*path, key)) for key, item in output.items()
        }
    return output


def _copy_leaf(tensor: Any) -> Any:
    """Copy the raw producer value without changing its label or existing consumers."""

    return tensor.clone()


def _emit_module_leaf(trace: Any, runtime: Any, tensor: Any) -> Any:
    """Record a declared replacement through Paddle's existing replay machinery."""

    backend = runtime._backend
    label = backend.tensor_store.get_label(tensor)
    predicate = runtime.intervene
    runtime.intervene = when(FuncSelector("interventionreplacement"), predicate.decision)
    try:
        with _state.pause_logging():
            raw = _copy_leaf(tensor)
            replacement = backend.emit_paddle_operation(
                trace,
                "interventionreplacement",
                _copy_leaf,
                (tensor,),
                {},
                raw,
                module_stack=tuple(trace._paddle_module_stack),
            )
    finally:
        runtime.intervene = predicate
    events = trace.capture_events
    boundary = events.op_event_by_label_raw[backend.tensor_store.get_label(replacement)]
    event = events.op_event_by_label_raw.get(label)
    if event is not None:
        fires = tuple(
            replace(
                fire,
                site_label=label,
                fire_record=replace(
                    fire.fire_record, target_label=label, func_call_id=event.func_call_id
                ),
            )
            for fire in boundary.fire_results
        )
        events.append_amendment(
            amend_module_exit_intervention(
                event.seq,
                label,
                intervention_fired=True,
                is_edited=boundary.is_edited,
                fire_results=(*event.fire_results, *fires),
            )
        )
    return replacement
