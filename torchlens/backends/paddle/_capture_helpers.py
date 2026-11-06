"""Private helpers the Paddle backend's capture path calls (file cap split).

Pure extractions from :mod:`.backend`: each function moved here unchanged so
that module and its long methods stay within the size caps, plus
:func:`paddle_layer_type`, which spells an op's layer type from its wrap name.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..._data_classes.derived_grad import IntermediateDerivedGradRecord
from ..._ir.capture_events import CaptureEvents
from ..._ir.events import InterventionAppliedEvent, OpEvent
from ..._ir.refs import DtypeRef
from .._finalize import stable_callable_name as _callable_identity

if TYPE_CHECKING:
    from .backend import GradOptions
    from .model_prep import PaddleModuleTree

# Reflected dunders whose operands commute fold to their forward operation, as
# torch's ``_normalize_func_name`` folds them (``x.__radd__(y)`` is ``add``).
_COMMUTATIVE_REFLECTED_DUNDERS = {
    "__radd__": "add",
    "__rmul__": "mul",
    "__rand__": "and",
    "__ror__": "or",
    "__rxor__": "xor",
}


def paddle_layer_type(op_name: str) -> str:
    """Return the layer type that spells a wrapped Paddle op's labels.

    The wrap inventory keys an op by its owner (``functional.relu``,
    ``tensor.reshape``, ``tensor.__add__``) and that key stays the op's
    ``func_name``. The layer type drops the owner namespace, as every other
    backend's does (``relu_1_2``, not ``functional.relu_1_2``), and an operator
    dunder takes its bare operation name the way torch spells it
    (``tensor.__add__`` and ``tensor.__radd__`` give ``add``,
    ``tensor.__rsub__`` gives ``rsub``).

    Parameters
    ----------
    op_name
        Wrapped operation name from the wrap inventory.

    Returns
    -------
    str
        Layer type for label reservation.
    """

    base = op_name.rsplit(".", 1)[-1]
    if len(base) > 4 and base.startswith("__") and base.endswith("__"):
        return _COMMUTATIVE_REFLECTED_DUNDERS.get(base, base.strip("_"))
    return base


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
        if event.is_edited:
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

    from .._facet_recipes import snapshot_facet_recipes

    return snapshot_facet_recipes(facet_recipes, backend_name="paddle")


def _alias_to_primary(tree: PaddleModuleTree) -> dict[str, str]:
    """Return module alias to primary address mapping."""

    aliases: dict[str, str] = {}
    for primary, metadata in tree.metadata.items():
        for alias in metadata.get("all_addresses", [primary]):
            aliases[str(alias)] = primary
    return aliases
