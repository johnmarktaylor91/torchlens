"""What a loaded-sparse run retains, and the run-prep and bind bounds that follow from it.

Loaded-sparse replay snapshots ``Op.out`` only for ops the capture saved, plus every
model output. Those retained clones are summed into the retention floor. Any other op
output lives only as long as replay needs it, so a pure view among them
(``expand``/``broadcast_to``/``as_strided``) allocates nothing and is admitted past
the memory budget, bounded instead by a fixed logical extent.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from .errors import RunCapabilityUnavailableError
from .runnable import (
    RunnableErrorCode,
    SparseRunDescriptor,
    TensorSlotDescriptor,
    TensorSlotRole,
)

_VIEW_LOGICAL_ELEMENT_CEILING = 1 << 32
"""Largest logical extent a pure-view slot the run does not copy may have past the budget.

Such a view allocates nothing, but its logical size still bounds WORK: every op that
reads it touches each logical element (a ``sum`` over a 4e11-element view of 4 floats
takes minutes). Charging the logical bytes against the memory budget used to bound
that work too, so a view admitted past the budget is held to this fixed extent
instead: 2**32 elements, a few seconds for one pass on a CPU core. A view over both
the budget and this ceiling is refused, as every such view was before view-aware
charging.
"""


def _retained_output_slot_ids(trace: Any, descriptor: SparseRunDescriptor) -> frozenset[str]:
    """Return the op-output slots whose value the run snapshots onto the result trace.

    Loaded-sparse replay keeps an ``Op.out`` snapshot only for ops the capture saved
    (``has_saved_activation``), as the capture did, plus every model-``output`` slot
    (the reconstruct clone). Every other op output lives only as long as replay needs it.
    """

    layer_dict = getattr(trace, "layer_dict_all_keys", {}) or {}
    retained = {
        slot.slot_id for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.OUTPUT
    }
    for call in descriptor.calls:
        # strict=False: a length mismatch is refused typed at bind; an unpaired slot
        # here is simply not retained.
        for slot_id, op_label in zip(call.output_slot_ids, call.op_labels, strict=False):
            if bool(getattr(layer_dict.get(op_label), "has_saved_activation", False)):
                retained.add(slot_id)
    return frozenset(retained)


def _unretained_view_lookup(
    trace: Any, descriptor: SparseRunDescriptor, retained_slot_ids: frozenset[str]
) -> Callable[[TensorSlotDescriptor], bool] | None:
    """Return the lookup proving a slot is an unretained pure view, if callables resolve.

    The lookup builds its projector and projects a slot's producing call only when
    asked, and at most once per slot, so a run within budget pays nothing for it. A
    retained slot is never a candidate: its snapshot materializes the full logical size.
    """

    callables = trace._runnable.callables_by_call_id
    if not isinstance(callables, Mapping):
        return None
    projected: dict[str, bool] = {}
    projector: list[Callable[[TensorSlotDescriptor], bool]] = []

    def lookup(slot: TensorSlotDescriptor) -> bool:
        """Project one slot at most once, building the projector on first use."""

        if slot.slot_id in retained_slot_ids:
            return False
        if slot.slot_id not in projected:
            if not projector:
                from ._runnable_execution import _pure_view_projector

                projector.append(_pure_view_projector(descriptor, callables))
            projected[slot.slot_id] = projector[0](slot)
        return projected[slot.slot_id]

    return lookup


def _refuse_view_past_logical_ceiling(
    slot: TensorSlotDescriptor, device: torch.device, logical_bytes: int, available: int
) -> None:
    """Refuse at run-prep a pure-view slot whose logical extent exceeds the ceiling."""

    logical_elements = math.prod(slot.shape)
    if logical_elements <= _VIEW_LOGICAL_ELEMENT_CEILING:
        return
    raise RunCapabilityUnavailableError(
        f"Recorded view slot {slot.slot_id!r} for device {str(device)!r} spans "
        f"{logical_elements} logical elements ({logical_bytes} bytes if materialized, "
        f"{available} bytes available). Its storage is small, but every op that reads it "
        f"touches each element, and the view ceiling is {_VIEW_LOGICAL_ELEMENT_CEILING} "
        "elements; the descriptor may be tampered or the view is too large for this host.",
        code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        detection_stage="op_allocation_preflight",
        device=str(device),
        required_bytes=logical_bytes,
        available_bytes=available,
        logical_elements=logical_elements,
        logical_element_ceiling=_VIEW_LOGICAL_ELEMENT_CEILING,
    )


def _preflight_retention_floor(
    descriptor: SparseRunDescriptor,
    retained_slot_ids: frozenset[str] | None = None,
) -> None:
    """Refuse a run whose GUARANTEED retained clone bytes provably exceed the budget (r59).

    Loaded-sparse replay retains a materialized clone of every SAVED taken-path op output
    for the life of the returned trace (one ``Op.out`` clone per saved
    ``(call, output_slot_id)`` at bind, plus one reconstruct clone per model-``output``
    slot). The SUM of those
    recorded op-output bytes is therefore a guaranteed LOWER BOUND on replay memory --
    a floor. A descriptor whose per-device floor exceeds the never-under-estimating
    live budget could never complete on this host: every honest per-op allocation may
    pass the per-slot bound (``_runnable_state._preflight_run_allocation``) and every clone may pass
    its per-clone byte guard, yet their cumulative retention guarantees a mid-replay
    allocator death (r58 free_1/free_2 accumulation seam: e.g. 40 honest 4 GB output
    slots = 160 GB floor > a 105 GB host budget). This refuses that typed at
    ``run_retention_preflight`` before any call executes.

    The floor UNDER-counts true retention (it ignores live ``slot_values`` entries, raw
    ``call_outputs``, attestation/witness snapshots) and the budget never under-estimates,
    so ``floor > budget`` proves a guaranteed OOM -- a refusal can NEVER hit a completable
    run (zero false refusals). A legit deep model whose cumulative retention is within
    budget (a deep-adds chain of small outputs) is untouched.

    ``retained_slot_ids`` names the op-output slots replay snapshots (saved ops plus
    model outputs, :func:`_retained_output_slot_ids`); when given, only those are charged.
    Without it every op output is charged, the pre-selection floor.
    """

    from ._runnable_state import _allocation_budget_bytes, _slot_device, _torch_dtype

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    floor: dict[str, int] = defaultdict(int)
    devices: dict[str, torch.device] = {}

    def _charge(slot: TensorSlotDescriptor) -> None:
        """Add one slot's guaranteed-retained clone bytes to its device's floor."""

        device = _slot_device(slot)
        key = str(device)
        devices[key] = device
        floor[key] += math.prod(slot.shape) * _torch_dtype(slot.dtype).itemsize

    # One retained ``Op.out`` clone per saved taken-path op label (bind): alias/version
    # slots are NOT in ``output_slot_ids`` so each producing op label charges at most
    # once, mirroring ``_bind_call_outputs``'s ``output_slot_ids`` zip.
    for call in descriptor.calls:
        for slot_id in call.output_slot_ids:
            slot = slots.get(slot_id)
            if slot is not None and (retained_slot_ids is None or slot_id in retained_slot_ids):
                _charge(slot)
    # One additional retained reconstruct clone per model-output slot.
    for slot in descriptor.tensor_slots:
        if slot.role is TensorSlotRole.OUTPUT:
            _charge(slot)

    for key, requested in sorted(floor.items()):
        available = _allocation_budget_bytes(devices[key])
        if requested > available:
            raise RunCapabilityUnavailableError(
                f"Loaded-sparse replay retains at least {requested} bytes of op-output "
                f"clones on device {key!r} for the run's lifetime, but only {available} "
                "bytes are available on this host. The cumulative retained state exceeds "
                "the budget and the run is guaranteed to fail mid-replay; the descriptor "
                "may be tampered or the host is too small.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="run_retention_preflight",
                device=key,
                required_bytes=requested,
                available_bytes=available,
            )


def _check_unretained_extent(
    value: torch.Tensor,
    *,
    call_id: str | None,
    slot_id: str | None,
    affected_op_labels: Sequence[str] = (),
) -> None:
    """Refuse at bind an op output the run will not copy whose logical extent is unbounded.

    A produced output the run does not snapshot costs no copy, but every later op
    reading it touches each logical element. Run-prep admits such a slot past the
    memory budget only up to ``_VIEW_LOGICAL_ELEMENT_CEILING`` elements of its RECORDED
    shape; this holds the REALIZED value to the same bound before any consumer runs, so
    a literal-only tamper (recorded slot honest, size literal inflated) cannot hand a
    4e11-element view to the next op, even under ``return_diverged``.
    """

    numel = int(value.numel())
    if numel <= _VIEW_LOGICAL_ELEMENT_CEILING:
        return
    from ._runnable_state import _allocation_budget_bytes

    logical_bytes = numel * int(value.element_size())
    available = _allocation_budget_bytes(value.device)
    if logical_bytes <= available:
        return
    raise RunCapabilityUnavailableError(
        f"Sparse call {call_id!r} produced output slot {slot_id!r} spanning {numel} "
        f"logical elements ({logical_bytes} bytes if materialized, {available} bytes "
        f"available). Every op that reads it touches each element, and the view ceiling "
        f"is {_VIEW_LOGICAL_ELEMENT_CEILING} elements; the descriptor may be tampered.",
        code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        detection_stage="op_allocation_execution",
        call_id=call_id,
        slot_id=slot_id,
        device=str(value.device),
        required_bytes=logical_bytes,
        available_bytes=available,
        logical_elements=numel,
        logical_element_ceiling=_VIEW_LOGICAL_ELEMENT_CEILING,
        affected_op_labels=tuple(affected_op_labels),
    )
