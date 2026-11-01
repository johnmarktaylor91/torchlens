"""Private helpers the Paddle backend's capture path calls (file cap split).

Pure extractions from :mod:`.backend`: each function moved here unchanged so
that module and its long methods stay within the size caps.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..._data_classes.derived_grad import IntermediateDerivedGradRecord
from ..._ir.capture_events import CaptureEvents
from ..._ir.events import InterventionAppliedEvent, OpEvent
from ..._ir.refs import DtypeRef
from .._finalize import stable_callable_name as _callable_identity
from .._protocols.paddle import PADDLE_REFUSAL_OPTION
from ..registry import BackendUnsupportedError

if TYPE_CHECKING:
    from .backend import GradOptions


def _intermediate_grad_record(
    op: Any, grad: Any, grad_options: GradOptions
) -> IntermediateDerivedGradRecord:
    """Build one exact intermediate derived-gradient record for ``op``."""

    return IntermediateDerivedGradRecord(
        op_label=op.label,
        layer_label=op.layer_label,
        aval=(
            f"Tensor(shape={tuple(getattr(grad, 'shape', ()))}, "
            f"dtype={getattr(grad, 'dtype', None)})"
        ),
        dtype_ref=DtypeRef(backend="paddle", name=str(getattr(grad, "dtype", ""))),
        grad=grad,
        provenance={
            "backend": "paddle",
            "kind": "intermediate_derived_gradient",
            "mechanism": "paddle.grad_tap_replay",
            "loss_id": _callable_identity(grad_options.loss_fn),
            "save_predicate_id": "trace.saved_ops",
            "status": "exact",
            "match_key": "func_call_id+parents+module_stack",
            "max_intermediate_grads": grad_options.max_intermediate_grads,
        },
    )


def _append_intervention_events(
    events: CaptureEvents, emitted: list[OpEvent], func_call_id: int
) -> None:
    """Record a live-fire replacement event for every replaced emitted op."""

    for event in emitted:
        if event.intervention_replaced:
            events.append_intervention(
                InterventionAppliedEvent(
                    label_raw=event.label_raw,
                    kind="replaced",
                    origin="live_fire",
                    timestamp=time.time(),
                    run_token=events.run_nonce,
                    target_seq=event.seq,
                    target_func_call_id=func_call_id,
                )
            )


def _paddle_facet_snapshot(facet_recipes: Any) -> Any:
    """Validate ``trace(facets=...)`` recipes and snapshot the facet registry."""

    if callable(facet_recipes):
        facet_recipes = (facet_recipes,)
    try:
        recipe_tuple = tuple(facet_recipes)
    except TypeError as exc:
        raise BackendUnsupportedError(
            "paddle backend trace(facets=...) requires an iterable of recipe callables.",
            code=PADDLE_REFUSAL_OPTION,
            option="facets",
        ) from exc
    if not all(callable(recipe) for recipe in recipe_tuple):
        raise BackendUnsupportedError(
            "paddle backend trace(facets=...) requires recipe callables.",
            code=PADDLE_REFUSAL_OPTION,
            option="facets",
        )
    from ..._semantic import facets as facets_mod

    return facets_mod.snapshot(recipe_tuple)
