"""Module-output intervention dispatch, separate from the MLX op door."""

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
from ..registry import BackendUnsupportedError
from ._tap_observers import _replace_mlx_array_leaves


def _validate_module_selector(selector: Any) -> None:
    """Refuse module trees whose meaning needs both dispatch doors."""

    if selector_contains_kind(selector, "module") and module_union_addresses(selector) is None:
        raise BackendUnsupportedError(
            "MLX backend module interventions require tl.module(...) or a union "
            "of module selectors; mixed or negated module selectors are unsupported.",
            code="interventions_unsupported",
        )


def _apply_module_exit(trace: Any, backend: Any, frame: Any, output: Any) -> Any:
    """Replace each distinct output leaf once at a selected module exit."""

    plan = getattr(trace, "_mlx_intervention_plan", None)
    if frame is None or plan is None:
        return output
    addresses = module_union_addresses(plan.selector)
    if not addresses or not any(
        module_address_matches((frame.address, frame.call_index), address) for address in addresses
    ):
        return output
    replacements = {}
    for tensor in backend._iter_arrays(output):
        if id(tensor) not in replacements:
            replacements[id(tensor)] = _emit_module_leaf(trace, backend, plan, tensor)
    return _replace_mlx_array_leaves(backend, output, replacements)


def _module_capture_op(trace: Any, frame: Any, op_name: str | None) -> str | None:
    """Expose selected descendants instead of capturing their parent as one opaque op."""

    plan = getattr(trace, "_mlx_intervention_plan", None)
    if frame is None or plan is None:
        return op_name
    addresses = module_union_addresses(plan.selector) or ()
    prefix = frame.address + "." if frame.address not in ("", "self") else ""
    if any(
        address.rsplit(":", 1)[0].startswith(prefix) and address.rsplit(":", 1)[0] != frame.address
        for address in addresses
    ):
        return None
    return op_name


def _copy_leaf(tensor: Any) -> Any:
    """Produce a distinct identity value so aliasing keeps the raw producer intact."""

    return tensor.astype(tensor.dtype)


def _emit_module_leaf(trace: Any, backend: Any, plan: Any, tensor: Any) -> Any:
    """Use the existing op intervention/replay path for the boundary replacement."""

    label = backend.tensor_store.get_label(tensor)
    # A separate identity boundary preserves the original producer's replay
    # payload, including consumers that ran before this module returned it.
    trace._mlx_intervention_plan = replace(plan, selector=FuncSelector("interventionreplacement"))
    trace._mlx_capture_depth += 1
    try:
        with _state.pause_logging():
            raw = _copy_leaf(tensor)
            replacement = backend.emit_mlx_operation(
                trace,
                "interventionreplacement",
                _copy_leaf,
                (tensor,),
                {},
                raw,
                module_stack=tuple(trace._mlx_module_stack),
            )
    finally:
        trace._mlx_capture_depth -= 1
        trace._mlx_intervention_plan = plan
    events = trace.capture_events
    boundary = events.op_event_by_label_raw[backend.tensor_store.get_label(replacement)]
    event = events.op_event_by_label_raw.get(label)
    if event is not None:
        events.append_amendment(
            amend_module_exit_intervention(
                event.seq,
                label,
                intervention_fired=True,
                is_edited=True,
                fire_results=(
                    *event.fire_results,
                    *(
                        replace(
                            fire,
                            site_label=label,
                            fire_record={
                                "backend": "mlx",
                                "plan_id": fire.plan_id,
                                "site_label": label,
                                "level": "module",
                                "selector": repr(plan.selector),
                                "replaced": fire.replaced,
                            },
                        )
                        for fire in boundary.fire_results
                    ),
                ),
            )
        )
    return replacement
