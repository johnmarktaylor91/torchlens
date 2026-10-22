"""What a loaded-sparse run retains, and the run-prep and bind bounds that follow from it.

Loaded-sparse replay snapshots ``Op.out`` only for ops the capture saved, plus every
model output; every op output also stays alive in the run's slot values until the run
ends. The retention floor sums them. A pure view the run does not copy
(``expand``/``broadcast_to``/``as_strided``) owns no storage, so it is admitted past the
memory budget, bounded instead by a fixed logical extent, but only on CPU and only when
its producer is a view op and every op that reads it, directly or through a version
alias, is known not to copy it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, TypeAlias

import torch

from .errors import RunCapabilityUnavailableError
from .runnable import (
    LiteralTorchSymbol,
    RunnableErrorCode,
    SparseRunDescriptor,
    TensorSlotDescriptor,
    TensorSlotRole,
)

_VIEW_LOGICAL_ELEMENT_CEILING = 1 << 32
"""Largest logical extent the admitted pure-view slots of one run may have, summed.

Such a view allocates nothing, but its logical size still bounds WORK: every op that
reads it touches each logical element (a ``sum`` over a 4e11-element view of 4 floats
takes minutes). Charging the logical bytes against the memory budget used to bound
that work too, so a view admitted past the budget is held to this fixed extent
instead: 2**32 elements in total, a few seconds for one pass on a CPU core. A view that
takes the total past the ceiling is refused, as every such view was before view-aware
charging.
"""

#: Ops whose output is a view of a stride-0 operand, measured by peak RSS on a 128 MiB
#: expanded tensor on torch 2.1.2 and 2.13.0 (function and method forms share one ATen
#: kernel). Only such an op may PRODUCE an admitted view: ``reshape``/``flatten``/
#: ``contiguous`` alias on the contiguous fakes of the run-prep proof but copy a real
#: strided operand, so they are absent.
_VIEW_OPS: tuple[tuple[str, str], ...] = (
    ("Tensor", "__getitem__"),
    ("Tensor", "select"),
    ("torch", "select"),
    ("Tensor", "narrow"),
    ("torch", "narrow"),
    ("Tensor", "permute"),
    ("torch", "permute"),
    ("Tensor", "transpose"),
    ("torch", "transpose"),
    ("Tensor", "t"),
    ("torch", "t"),
    ("Tensor", "unsqueeze"),
    ("torch", "unsqueeze"),
    ("Tensor", "squeeze"),
    ("torch", "squeeze"),
    ("Tensor", "expand"),
    ("torch", "broadcast_to"),
    ("Tensor", "broadcast_to"),
    ("Tensor", "diagonal"),
    ("torch", "diagonal"),
    ("Tensor", "unbind"),
    ("torch", "unbind"),
    ("Tensor", "split"),
    ("torch", "split"),
    ("Tensor", "chunk"),
    ("torch", "chunk"),
    ("Tensor", "movedim"),
    ("torch", "movedim"),
    ("Tensor", "detach"),
    ("torch", "detach"),
    ("Tensor", "view"),
)

_LiteralPosition: TypeAlias = tuple[str, int | str]
"""A literal argument's top-level position: ``("args", index)`` or ``("kwargs", name)``."""

_REDUCTION_FORMS: frozenset[_LiteralPosition] = frozenset(
    {("args", 1), ("args", 2), ("kwargs", "dim"), ("kwargs", "keepdim")}
)
_VARIANCE_FORMS: frozenset[_LiteralPosition] = frozenset(
    {("args", 1), ("kwargs", "dim"), ("kwargs", "correction"), ("kwargs", "keepdim")}
)
_SCALAR_OTHER_FORMS: frozenset[_LiteralPosition] = frozenset({("args", 1)})
_ADD_FORMS: frozenset[_LiteralPosition] = frozenset({("args", 1), ("kwargs", "alpha")})
_NO_LITERAL_FORMS: frozenset[_LiteralPosition] = frozenset()

#: Ops that read a stride-0 operand without copying it, measured the same way on CPU:
#: reductions, and elementwise ops whose own output slot is charged normally. FakeTensor
#: cannot see a kernel's internal temporaries, so a view read by any op outside this list
#: and ``_VIEW_OPS`` keeps its logical charge; ``median``, ``max``, ``argmax``, ``any``,
#: ``mm``, ``matmul``, ``unique`` and ``sort`` were measured copying the full operand.
#: ``norm`` is absent: a string or matrix order runs an SVD on a copy. Each op maps to the
#: literal argument positions it was measured with (``dim``/``keepdim``, ``correction``, a
#: scalar ``other``, ``alpha``); a call passing a literal anywhere else keeps the charge,
#: since an argument can turn a streaming read into a copy (``sum``/``mean``/``prod`` with
#: ``dtype=`` were measured casting the whole operand first). A listed op also qualifies
#: only with a fresh output (no ``out=``, not in place): an ``out=`` tensor is resized to
#: the full broadcast size, which the projection sees as an alias.
#: Closed list: add an op or a form only with a measurement on both torch versions.
_STREAMING_READER_FORMS: dict[tuple[str, str], frozenset[_LiteralPosition]] = {
    ("Tensor", "sum"): _REDUCTION_FORMS,
    ("torch", "sum"): _REDUCTION_FORMS,
    ("Tensor", "mean"): _REDUCTION_FORMS,
    ("torch", "mean"): _REDUCTION_FORMS,
    ("Tensor", "amax"): _REDUCTION_FORMS,
    ("torch", "amax"): _REDUCTION_FORMS,
    ("Tensor", "amin"): _REDUCTION_FORMS,
    ("torch", "amin"): _REDUCTION_FORMS,
    ("Tensor", "prod"): _REDUCTION_FORMS,
    ("torch", "prod"): _REDUCTION_FORMS,
    ("Tensor", "var"): _VARIANCE_FORMS,
    ("torch", "var"): _VARIANCE_FORMS,
    ("Tensor", "std"): _VARIANCE_FORMS,
    ("torch", "std"): _VARIANCE_FORMS,
    ("Tensor", "__add__"): _SCALAR_OTHER_FORMS,
    ("Tensor", "__radd__"): _SCALAR_OTHER_FORMS,
    ("Tensor", "add"): _ADD_FORMS,
    ("torch", "add"): _ADD_FORMS,
    ("Tensor", "__mul__"): _SCALAR_OTHER_FORMS,
    ("Tensor", "__rmul__"): _SCALAR_OTHER_FORMS,
    ("Tensor", "mul"): _SCALAR_OTHER_FORMS,
    ("torch", "mul"): _SCALAR_OTHER_FORMS,
    ("Tensor", "exp"): _NO_LITERAL_FORMS,
    ("torch", "exp"): _NO_LITERAL_FORMS,
    ("Tensor", "relu"): _NO_LITERAL_FORMS,
    ("torch", "relu"): _NO_LITERAL_FORMS,
}


def _resolved_ops(entries: tuple[tuple[str, str], ...]) -> frozenset[Any]:
    """Resolve ``(namespace, name)`` entries to the unwrapped callables the resolver binds."""

    return frozenset(_resolved_forms(dict.fromkeys(entries, _NO_LITERAL_FORMS)))


def _resolved_forms(
    entries: Mapping[tuple[str, str], frozenset[_LiteralPosition]],
) -> dict[Any, frozenset[_LiteralPosition]]:
    """Resolve ``(namespace, name)`` keys to unwrapped callables, keeping their forms."""

    from ._io.runnable_load import _unwrap_decorated
    from .utils._torch_symbols import torch_attr

    resolved: dict[Any, frozenset[_LiteralPosition]] = {}
    for (namespace, name), forms in entries.items():
        found = torch_attr(name) if namespace == "torch" else getattr(torch.Tensor, name, None)
        if found is not None:
            resolved[_unwrap_decorated(found)] = forms
    return resolved


def _retained_output_slot_ids(trace: Any, descriptor: SparseRunDescriptor) -> frozenset[str]:
    """Return the op-output slots whose value the run snapshots onto the result trace.

    Loaded-sparse replay keeps an ``Op.out`` snapshot only for ops the capture saved
    (``has_saved_activation``), as the capture did, plus every model-``output`` slot
    (the reconstruct clone). Every other op output lives only as long as replay needs it.
    """

    from ._runnable_verification import _op_for_label

    retained = {
        slot.slot_id for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.OUTPUT
    }
    for call in descriptor.calls:
        # strict=False: a length mismatch is refused typed at bind; an unpaired slot
        # here is simply not retained.
        for slot_id, op_label in zip(call.output_slot_ids, call.op_labels, strict=False):
            if bool(getattr(_op_for_label(trace, op_label), "has_saved_activation", False)):
                retained.add(slot_id)
    return frozenset(retained)


def _unretained_view_lookup(
    trace: Any, descriptor: SparseRunDescriptor, retained_slot_ids: frozenset[str]
) -> Callable[[TensorSlotDescriptor], bool] | None:
    """Return the lookup proving a slot is an unretained pure view, if callables resolve.

    A slot qualifies only if it lives on CPU (the reader lists were measured there), the
    run does not retain it, its producer is in ``_VIEW_OPS``, every call reading it or a
    version alias of it is in ``_VIEW_OPS`` or ``_STREAMING_READER_FORMS``, and the
    projection proves the producer's output aliases an operand. The lookup
    builds its projector only when asked, and decides each slot at most once, so a run
    within budget pays nothing for it.
    """

    callables = trace._runnable.callables_by_call_id
    if not isinstance(callables, Mapping):
        return None
    decided: dict[str, bool] = {}
    state: dict[str, Any] = {}

    def lookup(slot: TensorSlotDescriptor) -> bool:
        """Decide one slot at most once, building the projector on first use."""

        if slot.device_type != "cpu" or slot.slot_id in retained_slot_ids:
            return False
        if slot.slot_id not in decided:
            if not state:
                state.update(_view_admission_state(descriptor, callables))
            decided[slot.slot_id] = _admits_view(slot, callables, state)
        return decided[slot.slot_id]

    return lookup


def _view_admission_state(
    descriptor: SparseRunDescriptor, callables: Mapping[str, Callable[..., Any]]
) -> dict[str, Any]:
    """Build the producer/reader maps, the op sets and the projector for view admission."""

    from ._runnable_execution import _pure_view_projector

    producers = {slot_id: call for call in descriptor.calls for slot_id in call.output_slot_ids}
    direct_readers: dict[str, list[Any]] = defaultdict(list)
    for call in descriptor.calls:
        for argument in call.tensor_arguments:
            direct_readers[argument.slot_id].append(call)
    readers = _readers_through_version_aliases(descriptor, direct_readers)
    view_ops = _resolved_ops(_VIEW_OPS)
    return {
        "slots": {slot.slot_id: slot for slot in descriptor.tensor_slots},
        "producers": producers,
        "readers": readers,
        "view_ops": view_ops,
        "streaming_ops": _resolved_forms(_STREAMING_READER_FORMS),
        "projector": _pure_view_projector(descriptor, callables),
    }


def _readers_through_version_aliases(
    descriptor: SparseRunDescriptor, direct_readers: Mapping[str, list[Any]]
) -> dict[str, list[Any]]:
    """Map each slot to the calls reading it directly or through a version alias of it.

    Bind writes a produced value into each DIRECT version-alias slot of it (``version_of``
    and ``producer_slot_id`` both naming the producer), so a call reading such an alias
    reads the producer's value, whatever shape the alias records. Bind never writes an
    alias of an alias (the loader refuses one), so one level is exact, and the map stays
    linear in the descriptor.
    """

    readers: dict[str, list[Any]] = defaultdict(list)
    for slot_id, calls in direct_readers.items():
        readers[slot_id].extend(calls)
    for slot in descriptor.tensor_slots:
        if slot.version_of is not None and slot.version_of == slot.producer_slot_id:
            readers[slot.version_of].extend(direct_readers.get(slot.slot_id, ()))
    return readers


def _admits_view(
    slot: TensorSlotDescriptor,
    callables: Mapping[str, Callable[..., Any]],
    state: Mapping[str, Any],
) -> bool:
    """Whether ``slot`` is a view-op output, read only by non-copying ops, proven to alias.

    A reader qualifies if it is a streaming op, or a view op whose every output the
    projection proves is a view too (basic indexing qualifies, advanced indexing does not).
    """

    producer = state["producers"].get(slot.slot_id)
    if (
        producer is None
        or _writes_caller_tensor(producer)
        or not _is_listed(callables.get(producer.call_id), state["view_ops"])
    ):
        return False
    return bool(state["projector"](slot)) and all(
        _reader_keeps_view(call, callables, state) for call in state["readers"][slot.slot_id]
    )


def _reader_keeps_view(
    call: Any, callables: Mapping[str, Callable[..., Any]], state: Mapping[str, Any]
) -> bool:
    """Whether one reader of an admitted view is known not to copy it."""

    if _writes_caller_tensor(call) or _names_dtype(call):
        return False
    func = callables.get(call.call_id)
    if _is_listed(func, state["streaming_ops"]):
        measured = state["streaming_ops"][func]
        return all(
            tuple(argument.argument_path[:2]) in measured for argument in call.literal_arguments
        )
    if not _is_listed(func, state["view_ops"]):
        return False
    slots = state["slots"]
    return all(
        slot_id in slots and bool(state["projector"](slots[slot_id]))
        for slot_id in call.output_slot_ids
    )


def _writes_caller_tensor(call: Any) -> bool:
    """Whether a call writes into a tensor the caller passed (in place, or ``out=``).

    Such a call's output takes the caller tensor's storage, resized to whatever the op
    produces, so the projection sees an alias where the real call may allocate in full.
    """

    return bool(call.is_inplace) or any(
        tuple(argument.argument_path[:2]) == ("kwargs", "out") for argument in call.tensor_arguments
    )


def _names_dtype(call: Any) -> bool:
    """Whether a call passes a dtype (``dtype=``, or any torch-symbol literal).

    A reduction given a dtype casts its whole operand first, a full-size copy of a view.
    """

    return any(
        tuple(argument.argument_path[:2]) == ("kwargs", "dtype")
        or isinstance(argument.value, LiteralTorchSymbol)
        for argument in call.literal_arguments
    )


def _is_listed(func: Any, ops: Collection[Any]) -> bool:
    """Whether a resolved callable is one of ``ops`` (an unhashable callable never is)."""

    try:
        return func is not None and func in ops
    except TypeError:
        return False


def _refuse_view_past_logical_ceiling(
    slot: TensorSlotDescriptor,
    device: torch.device,
    logical_bytes: int,
    available: int,
    prior_elements: int = 0,
) -> int:
    """Refuse at run-prep a pure view that takes the admitted views past the ceiling.

    Returns the admitted views' logical elements including ``slot``.
    """

    logical_elements = prior_elements + math.prod(slot.shape)
    if logical_elements <= _VIEW_LOGICAL_ELEMENT_CEILING:
        return logical_elements
    raise RunCapabilityUnavailableError(
        f"Recorded view slot {slot.slot_id!r} for device {str(device)!r} takes the run's "
        f"admitted views to {logical_elements} logical elements ({logical_bytes} bytes if "
        f"this one were materialized, {available} bytes available). Views own little "
        "storage, but every op that reads one touches each element, and the view ceiling "
        f"is {_VIEW_LOGICAL_ELEMENT_CEILING} elements per run; the descriptor may be "
        "tampered or the views are too large for this host.",
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
    admitted_view_slot_ids: frozenset[str] = frozenset(),
) -> None:
    """Refuse a run whose GUARANTEED retained bytes provably exceed the budget (r59).

    Loaded-sparse replay keeps every taken-path op output alive in its slot values until
    the run ends (plus an ``Op.out`` clone for each saved op), and one reconstruct clone
    per model-``output`` slot for the life of the returned trace. Charging each op
    output once, plus each model output once, is therefore a guaranteed LOWER BOUND on
    replay memory -- a floor. A descriptor whose per-device floor exceeds the never-under-estimating
    live budget could never complete on this host: every honest per-op allocation may
    pass the per-slot bound (``_runnable_state._preflight_run_allocation``) and every clone may pass
    its per-clone byte guard, yet their cumulative retention guarantees a mid-replay
    allocator death (r58 free_1/free_2 accumulation seam: e.g. 40 honest 4 GB output
    slots = 160 GB floor > a 105 GB host budget). This refuses that typed at
    ``run_retention_preflight`` before any call executes.

    The floor UNDER-counts true retention (it leaves out the ``Op.out`` clone of each
    saved op, raw ``call_outputs`` and attestation/witness snapshots) and the budget never
    under-estimates, so ``floor > budget`` proves a guaranteed OOM -- a refusal can NEVER
    hit a completable run (zero false refusals). A legit deep model whose cumulative retention is within
    budget (a deep-adds chain of small outputs) is untouched.

    ``admitted_view_slot_ids`` names the views run-prep admitted past the budget; they
    own no storage, so they are the only op outputs the floor does not charge.
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

    # One live value per taken-path op output: alias/version slots are NOT in
    # ``output_slot_ids`` so each producing op label charges once.
    for call in descriptor.calls:
        for slot_id in call.output_slot_ids:
            slot = slots.get(slot_id)
            if slot is not None and slot_id not in admitted_view_slot_ids:
                _charge(slot)
    # One additional retained reconstruct clone per model-output slot.
    for slot in descriptor.tensor_slots:
        if slot.role is TensorSlotRole.OUTPUT:
            _charge(slot)

    for key, requested in sorted(floor.items()):
        available = _allocation_budget_bytes(devices[key])
        if requested > available:
            raise RunCapabilityUnavailableError(
                f"Loaded-sparse replay retains at least {requested} bytes of op outputs "
                f"on device {key!r} for the run's lifetime, but only {available} "
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
    admitted_view: bool = False,
) -> None:
    """Refuse at bind an op output the run will not copy whose logical extent is unbounded.

    A produced output the run does not snapshot costs no copy, but every later op
    reading it touches each logical element. Run-prep admits such a slot past the
    memory budget only up to ``_VIEW_LOGICAL_ELEMENT_CEILING`` elements of its RECORDED
    shape; this holds the REALIZED value to the same bound before any consumer runs, so
    a literal-only tamper (recorded slot honest, size literal inflated) cannot hand a
    4e11-element view to the next op, even under ``return_diverged``. A view run-prep
    admitted past the budget (``admitted_view``) must also really be on CPU, where its
    readers were measured not to copy it: the recorded device label is the file's claim.
    """

    numel = int(value.numel())
    off_cpu = admitted_view and value.device.type != "cpu"
    if numel <= _VIEW_LOGICAL_ELEMENT_CEILING and not off_cpu:
        return
    from ._runnable_state import _allocation_budget_bytes

    logical_bytes = numel * int(value.element_size())
    available = _allocation_budget_bytes(value.device)
    if logical_bytes <= available and not off_cpu:
        return
    reason = (
        "It was admitted past the budget as a CPU view, but it is not on CPU"
        if off_cpu
        else "Every op that reads it touches each element"
    )
    raise RunCapabilityUnavailableError(
        f"Sparse call {call_id!r} produced output slot {slot_id!r} on {str(value.device)!r} "
        f"spanning {numel} logical elements ({logical_bytes} bytes if materialized, "
        f"{available} bytes available). {reason}, and the view ceiling is "
        f"{_VIEW_LOGICAL_ELEMENT_CEILING} elements; the descriptor may be tampered.",
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


def _fail_open_unless_admitted(
    call: Any, ceiling: Any | None, cause: BaseException | None = None
) -> None:
    """Let an unprojected call fail open, unless it produces an admitted view.

    Run-prep admitted such a call's output past the budget on its view proof alone, so
    the run-time projection is its only pre-execution check; when that cannot run, the
    call is refused at ``op_allocation_preflight`` instead of executed unbounded.
    """

    if ceiling is None or not ceiling.produces_admitted_view(call):
        return
    raise RunCapabilityUnavailableError(
        f"Sparse call {call.call_id!r} produces a view admitted past the memory budget, "
        "but its allocation could not be projected before execution; an unprojected "
        "call cannot be admitted under the fail-closed budget contract. The descriptor "
        "may be tampered.",
        code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        detection_stage="op_allocation_preflight",
        call_id=call.call_id,
        affected_op_labels=tuple(call.op_labels),
    ) from cause
