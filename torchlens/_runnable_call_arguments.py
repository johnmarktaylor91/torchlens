"""Sparse-call argument decoding and binding."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from typing import TYPE_CHECKING, Any, cast

import torch

from ._runnable_state import (
    _OUTPUT_COUNT_FLOOR,
    RunResourceCeiling,
    _allocation_budget_bytes,
    _slot_device,
    _torch_dtype,
)
from .errors import (
    RunCapabilityUnavailableError,
    RunPreconditionError,
    RuntimeSignatureDriftError,
)
from .runnable import (
    CallableRegistryEntry,
    RunnableCallDescriptor,
    RunnableErrorCode,
    SparseRunDescriptor,
    TensorSlotDescriptor,
    TensorSlotRole,
)

if TYPE_CHECKING:
    from ._runnable_execution import (
        _MAX_DECODE_NESTING_DEPTH,
        _count_bounded_fake_tensor_mode_class,
        _decode_literal,
        _is_allocator_death,
        _mutation_target_slot_id,
        _op_for_slot,
        _ProjectionCountExceeded,
        _UnreadableProjectedOutput,
        _value_at_path,
    )

__all__ = (
    "_has_numeric_literal",
    "_has_tensor_operand",
    "_projection_required_by_arguments",
    "_tree_to_fake",
    "_fake_tensor_storage_id",
    "_input_storage_ids",
    "_new_allocation_bytes",
    "_preflight_call_allocation",
    "_pure_view_projector",
    "_projects_as_pure_view",
    "_execute_sparse_call",
    "_populate_source_slots",
    "_write_argument",
    "_write_path",
    "_resolve_setter_output",
)


def _has_numeric_literal(value: Any, _depth: int = 0) -> bool:
    """Return whether a decoded argument tree carries a size-relevant numeric literal.

    One of the two size sources ``_projection_required_by_arguments`` recognizes
    (the other is a tensor operand -- ``_has_tensor_operand``, r61). "Carries a
    size-relevant literal" means the decoded tree holds any non-bool ``int`` or
    any *finite* ``float`` -- the float branch covers multiplier parameters such as
    ``F.interpolate(scale_factor=<float>)`` that an integer-only gate missed
    (r56 free_1). ``bool`` is excluded (never a size argument), and non-finite
    float sentinels (``inf``/``nan``) are not size authorities and fall through to
    the existing literal validation. Bounded recursion mirrors the literal-nesting
    ceiling. Whether such a call is *actually* size-driving is then decided
    structurally by the projection, never by an op-name family list.
    """

    if _depth > _MAX_DECODE_NESTING_DEPTH:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, tuple)):
        return any(_has_numeric_literal(item, _depth + 1) for item in value)
    if isinstance(value, Mapping):
        return any(_has_numeric_literal(item, _depth + 1) for item in value.values())
    return False


def _has_tensor_operand(value: Any, _depth: int = 0) -> bool:
    """True if a decoded argument tree contains a torch.Tensor (bounded, short-circuit).

    isinstance-only -- NO property read (numel/shape) ever fires, so a subclass can
    never route user code through the predicate. Mirrors ``_has_numeric_literal``'s
    recursion over list/tuple/Mapping. Every shape amplifier (``outer``/``kron``/
    broadcast-``mul``/``cartesian_prod``/``tensordot(dims=0)``/``einsum``/``diag`` --
    and the zero-numel family ``mm``/``matmul``/``einsum`` on ``[N,0] @ [0,N]``,
    where numel is a PRODUCT so a 0 dim hides arbitrarily large sibling dims)
    carries a tensor operand, so no amplifier can be pre-filtered out: the r60
    free_1 has-literal premise hole is closed structurally, not by an op list.
    Over-depth returns True (fail-closed: project).
    """

    if _depth > _MAX_DECODE_NESTING_DEPTH:
        return True
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)):
        return any(_has_tensor_operand(item, _depth + 1) for item in value)
    if isinstance(value, Mapping):
        return any(_has_tensor_operand(item, _depth + 1) for item in value.values())
    return False


def _projection_required_by_arguments(args: Sequence[Any], kwargs: Mapping[str, Any]) -> bool:
    """Whether this replay call carries any size source (tensor operand or numeric literal).

    The ONLY sound projection skip is "no size source at all" (r61): a call with
    neither a tensor operand nor a numeric literal gives no fake/meta kernel
    anything to size an output tree from, so it is provably allocation-trivial.
    Any richer skip heuristic (has-literal-only, numel/arity thresholds) gaps on a
    shape-amplifier family and re-opens the class.
    """

    arg_list = list(args)
    kwarg_dict = dict(kwargs)
    return (
        _has_numeric_literal(arg_list)
        or _has_numeric_literal(kwarg_dict)
        or _has_tensor_operand(arg_list)
        or _has_tensor_operand(kwarg_dict)
    )


def _tree_to_fake(mode: Any, value: Any) -> Any:
    """Replace every live tensor in an argument tree with its fake stand-in."""

    if isinstance(value, torch.Tensor):
        try:
            return mode.from_tensor(value)
        except Exception:
            return value
    if isinstance(value, tuple):
        return tuple(_tree_to_fake(mode, item) for item in value)
    if isinstance(value, list):
        return [_tree_to_fake(mode, item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_to_fake(mode, item) for key, item in value.items()}
    return value


def _fake_tensor_storage_id(value: torch.Tensor) -> int | None:
    """Return a fake tensor's storage identity (``untyped_storage()._cdata``), or None.

    The storage ``_cdata`` pointer identifies the underlying allocation. Two
    tensors that share it alias the same storage (a view / in-place / input-return
    output aliases its input). ``FakeTensorMode`` exposes readable fake storages;
    an unreadable one returns ``None`` and is treated per-caller (input side: not
    aliasable; output side: charged fail-closed).
    """

    try:
        return int(value.untyped_storage()._cdata)
    except Exception:
        return None


def _input_storage_ids(value: Any) -> frozenset[int]:
    """Collect the fake-input storage identities across a fake argument tree.

    Only readable storage ids participate: an input whose storage cannot be read
    is simply absent from the set, so it can never make an output *look* like a
    view (an unreadable output is charged regardless). Bounded to the same
    container kinds the fake conversion produces.
    """

    ids: set[int] = set()
    stack: list[Any] = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, torch.Tensor):
            sid = _fake_tensor_storage_id(node)
            if sid is not None:
                ids.add(sid)
        elif isinstance(node, (tuple, list)):
            stack.extend(node)
        elif isinstance(node, Mapping):
            stack.extend(node.values())
    return frozenset(ids)


# NOTE: ``_UnreadableProjectedOutput`` (like ``_ProjectionCountExceeded``) is
# defined in ``_runnable_execution`` and resolves through the rebound globals.


def _new_allocation_bytes(
    projected: Any, input_storage_ids: AbstractSet[int]
) -> dict[torch.device, int]:
    """Sum the NEWLY-allocated projected output bytes per device across an output tree.

    An output tensor whose fake ``untyped_storage()._cdata`` aliases a fake-INPUT
    storage allocates nothing -- it is a pure view, an input-returning op, or an
    in-place op -- so it contributes ZERO new bytes and is skipped. This structural
    storage-aliasing test replaces the r55 hardcoded view-name exclusion
    (``expand``/``broadcast_to``/``as_strided``/``unfold``) with no op/view list,
    and correctly charges only the NEW tensors of a mixed-output op (e.g.
    ``split`` -> views of the input are skipped; a fresh concat is charged). An
    output whose storage identity cannot be read is CHARGED (fail-closed: an
    unreadable materializer must never masquerade as a view). ``_base`` is
    deliberately NOT used: in-place / input-returning outputs have ``_base is
    None`` yet still alias an input storage.
    """

    totals: dict[torch.device, int] = {}
    stack: list[Any] = [projected]
    while stack:
        node = stack.pop()
        if isinstance(node, torch.Tensor):
            sid = _fake_tensor_storage_id(node)
            if sid is not None and sid in input_storage_ids:
                continue  # aliases a fake input storage -> allocates nothing
            try:
                nbytes = int(node.numel()) * int(node.element_size())
                device = node.device
            except Exception as size_error:
                # grind-r5 b7 R22 (sol HIGH): an unreadable projected size
                # must never charge ZERO -- that converts "unknown
                # allocation" into an empty byte charge and lets the real
                # call allocate before any backstop. Refuse typed upstream.
                raise _UnreadableProjectedOutput(
                    f"projected output size unreadable: {type(size_error).__name__}: {size_error}"
                ) from size_error
            totals[device] = totals.get(device, 0) + nbytes
        elif isinstance(node, (tuple, list)):
            stack.extend(node)
        elif isinstance(node, Mapping):
            stack.extend(node.values())
    return totals


def _preflight_call_allocation(
    entry: CallableRegistryEntry | None,
    func: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    call: RunnableCallDescriptor,
    ceiling: RunResourceCeiling | None = None,
) -> None:
    """Refuse a size-source-carrying call by projected NEW bytes AND realized output COUNT.

    Runs the resolved callable under a count-instrumented
    ``FakeTensorMode(allow_non_fake_inputs=True)`` (ZERO allocation) for EVERY taken-path
    call carrying a size source: a non-bool numeric literal (``_has_numeric_literal``) OR
    a tensor operand (``_has_tensor_operand``, r61 -- the has-literal-only gate's premise,
    "no literal implies bounded by live shapes", is FALSE for shape amplifiers such as
    ``outer``/``kron``/broadcast-``mul``/``cartesian_prod``/``tensordot(dims=0)``/
    ``einsum``/``diag`` and the zero-numel ``mm``/``matmul``/``einsum`` on ``[N,0]@[0,N]``).
    The r55 op-name allowlist is deleted, so "size-relevance" is decided structurally by
    the projection, not guessed; the ONLY skip is a call with NO size source at all
    (``_projection_required_by_arguments``). Two structural bounds fire before the real
    call:

    * output COUNT (r59 free_1): the count-bounded mode caps realized fake outputs at
      ``max(recorded * 8, 4096)`` DURING fanout construction and raises
      ``_ProjectionCountExceeded`` -- caught here BEFORE the generic fail-open and
      converted to a typed ``op_output_count_preflight`` refusal, so a
      ``tensor_split(x, N)`` count bomb can neither self-DoS the projection nor slip
      through as a 0-byte fanout.
    * NEW bytes: each projected output-device's NEWLY-allocated total (pure views /
      input-returning / in-place outputs contribute zero -- ``_new_allocation_bytes``)
      is compared against the never-under-estimating live budget and refused typed at
      ``op_allocation_preflight``.

    If the projection RAISES a NON-allocation error -- a data-dependent op with no
    fake/meta impl (``nonzero``/``unique`` -> ``DynamicOutputShapeException``, or ``fold``
    with capture-invalid ``output_size``) -- it FAILS OPEN to the run-prep bound, so a
    legitimate such op is never over-refused (the r51 over-catch anti-pattern). But an
    ALLOCATION-classed projection failure (``std::bad_alloc`` in the C++ prelude at
    int-limit N) fails CLOSED typed (r59 section 2.4): the real op's identical prelude
    would die too, so failing open just runs the death twice. An unavailable
    ``FakeTensorMode`` fails open identically. ``entry`` is retained for call-site
    stability but no longer gates the projection.
    """

    del entry  # r57: no op-name gate; r61: no has-literal gate either -- any size source projects.
    if not _projection_required_by_arguments(args, kwargs):
        return  # provably trivial: no tensor operand and no numeric literal -> no size source
    mode_cls = _count_bounded_fake_tensor_mode_class()
    if mode_cls is None:
        return  # feature-detect fail-open -> run-prep recorded-output bound
    count_ceiling = (
        ceiling.per_call_output_count_ceiling(call)
        if ceiling is not None
        else max(len(getattr(call, "output_slot_ids", ()) or ()) * 8, _OUTPUT_COUNT_FLOOR)
    )
    try:
        with mode_cls(allow_non_fake_inputs=True, ceiling=count_ceiling) as mode:
            fake_args = _tree_to_fake(mode, list(args))
            fake_kwargs = _tree_to_fake(mode, dict(kwargs))
            input_storage_ids = _input_storage_ids(fake_args) | _input_storage_ids(fake_kwargs)
            mode._tl_set_baseline()
            projected = func(*fake_args, **fake_kwargs)
    except _ProjectionCountExceeded as exc:
        # Count breach: a faithful replay never realizes more than the ceiling. Typed
        # refusal, BEFORE the generic fail-open and before the full fake/real tree exists.
        raise RunCapabilityUnavailableError(
            f"Sparse call {call.call_id!r} projects more than {count_ceiling} output "
            f"tensors ({exc}); the recorded output-count literal would drive an "
            "unbounded fanout allocation. The descriptor may be tampered.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="op_output_count_preflight",
            call_id=call.call_id,
            affected_op_labels=call.op_labels,
            output_count_ceiling=count_ceiling,
        ) from exc
    except Exception as exc:
        # r59 section 2.4: an ALLOCATION-classed projection failure (std::bad_alloc in
        # the O(N) C++ prelude at int-limit N) fails CLOSED -- the real op's identical
        # prelude dies too. Every OTHER failure (data-dependent nonzero/unique, fold
        # output_size inconsistency) keeps today's FAIL OPEN so a legitimate op runs.
        if _is_allocator_death(exc):
            raise RunCapabilityUnavailableError(
                f"Sparse call {call.call_id!r} could not be projected without an "
                "allocation failure; the recorded literals drive an out-of-budget "
                "allocation in the operator prelude. The descriptor may be tampered.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="op_allocation_preflight",
                call_id=call.call_id,
                affected_op_labels=call.op_labels,
            ) from exc
        return
    try:
        new_bytes_by_device = _new_allocation_bytes(projected, input_storage_ids)
    except _UnreadableProjectedOutput as exc:
        raise RunCapabilityUnavailableError(
            f"Sparse call {call.call_id!r} produced a projected output whose "
            f"allocation size could not be read ({exc}); an unknown allocation "
            "cannot be admitted under the fail-closed budget contract. The "
            "descriptor may be tampered.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="op_allocation_preflight",
            call_id=call.call_id,
            affected_op_labels=call.op_labels,
        ) from exc
    for device, requested in new_bytes_by_device.items():
        available = _allocation_budget_bytes(device)
        if requested > available:
            raise RunCapabilityUnavailableError(
                f"Sparse call {call.call_id!r} projects a {requested}-byte allocation "
                f"on device {str(device)!r}, but only {available} bytes are available "
                "on this host. The recorded literal arguments would drive an "
                "out-of-budget allocation; the descriptor may be tampered.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="op_allocation_preflight",
                call_id=call.call_id,
                device=str(device),
                required_bytes=requested,
                available_bytes=available,
                affected_op_labels=call.op_labels,
            )


def _pure_view_projector(
    descriptor: SparseRunDescriptor,
    callables: Mapping[str, Callable[..., Any]],
) -> Callable[[TensorSlotDescriptor], bool]:
    """Build the run-prep proof that an op-output slot is a pure view of its operands.

    The run-prep per-slot bound charges every recorded op-output slot its logical size,
    which is what replay allocates for a fresh output. A pure view (``expand``/
    ``broadcast_to``/``as_strided``) allocates nothing when its op runs. The returned
    lookup projects the slot's producing call under the count-bounded ``FakeTensorMode``
    (zero allocation) on contiguous fake operands built from the recorded slot shapes,
    and returns True only if the projected output aliases a fake input storage and
    matches the recorded shape. Any other outcome (fresh output, unresolvable call,
    projection failure) returns False, so the caller keeps the logical charge. At run
    time the per-call projection re-checks the real call before it executes.
    """

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    producers = {slot_id: call for call in descriptor.calls for slot_id in call.output_slot_ids}

    def project(slot: TensorSlotDescriptor) -> bool:
        """Return whether one slot's producing call projects as a pure view."""

        call = producers.get(slot.slot_id)
        func = callables.get(call.call_id) if call is not None else None
        if call is None or func is None:
            return False
        try:
            return _projects_as_pure_view(call, func, slots, slot)
        except Exception:  # any projection failure keeps the logical charge
            return False

    return project


def _projects_as_pure_view(
    call: RunnableCallDescriptor,
    func: Callable[..., Any],
    slots: Mapping[str, TensorSlotDescriptor],
    slot: TensorSlotDescriptor,
) -> bool:
    """Project one call on fake operands; return whether ``slot`` aliases an operand."""

    mode_cls = _count_bounded_fake_tensor_mode_class()
    if mode_cls is None:
        return False
    args: list[Any] = [None] * call.num_positional_args
    kwargs: dict[str, Any] = {}
    for literal_argument in call.literal_arguments:
        _write_argument(
            args, kwargs, literal_argument.argument_path, _decode_literal(literal_argument.value)
        )
    count_ceiling = max(len(call.output_slot_ids) * 8, _OUTPUT_COUNT_FLOOR)
    with mode_cls(allow_non_fake_inputs=True, ceiling=count_ceiling) as mode:
        for tensor_argument in call.tensor_arguments:
            operand = slots[tensor_argument.slot_id]
            fake = torch.empty(
                operand.shape, dtype=_torch_dtype(operand.dtype), device=_slot_device(operand)
            )
            _write_argument(args, kwargs, tensor_argument.argument_path, fake)
        input_storage_ids = _input_storage_ids(args) | _input_storage_ids(kwargs)
        mode._tl_set_baseline()
        value = _value_at_path(func(*args, **kwargs), slot.output_path or ())
        if not isinstance(value, torch.Tensor):
            return False
        storage_id = _fake_tensor_storage_id(value)
        shape = tuple(int(size) for size in value.shape)
        return storage_id is not None and storage_id in input_storage_ids and shape == slot.shape


def _execute_sparse_call(
    call: RunnableCallDescriptor,
    func: Callable[..., Any],
    slot_values: Mapping[str, torch.Tensor],
    *,
    registry_entry: CallableRegistryEntry | None = None,
    ceiling: RunResourceCeiling | None = None,
) -> Any:
    """Construct and execute one sparse call from literal and tensor leaves."""

    args: list[Any] = [None] * call.num_positional_args
    kwargs: dict[str, Any] = {}
    for literal_argument in call.literal_arguments:
        _write_argument(
            args,
            kwargs,
            literal_argument.argument_path,
            _decode_literal(literal_argument.value),
        )
    for tensor_argument in call.tensor_arguments:
        try:
            value = slot_values[tensor_argument.slot_id]
        except KeyError as exc:
            raise RunPreconditionError(
                f"Sparse call {call.call_id!r} references unavailable slot "
                f"{tensor_argument.slot_id!r}.",
                code=RunnableErrorCode.MISSING_TENSOR_SLOT.value,
                call_id=call.call_id,
                slot_id=tensor_argument.slot_id,
            ) from exc
        _write_argument(args, kwargs, tensor_argument.argument_path, value)
    # r55 C3: op-agnostic allocation preflight BEFORE the real dispatch. A typed
    # refusal here replaces the raw allocator OOM-kill a hostile literal would
    # otherwise cause; it never masks a genuine signature drift (the drift path
    # below still runs the real call for every non-refused request).
    _preflight_call_allocation(registry_entry, func, args, kwargs, call, ceiling)
    try:
        return func(*args, **kwargs)
    except RunCapabilityUnavailableError:
        # A typed capability refusal from a nested guard (e.g. a re-materialization
        # clone inside a resolved callable) is already correct -- never re-cloak it as
        # signature drift.
        raise
    except Exception as exc:
        # r59 section 2.4: an allocation-classed death from the REAL call is a capability
        # fact, not signature drift -- re-type it typed at ``op_allocation_execution`` so a
        # gate-4-passing run whose live+transient peak still overruns the host reports an
        # allocation refusal, never a misleading ``runtime_signature_drift``.
        if _is_allocator_death(exc):
            raise RunCapabilityUnavailableError(
                f"Sparse call {call.call_id!r} failed with an allocation error during "
                "execution; its recorded shapes exhaust this host's memory. The "
                "descriptor may be tampered or the host is too small.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="op_allocation_execution",
                call_id=call.call_id,
                affected_op_labels=call.op_labels,
            ) from exc
        raise RuntimeSignatureDriftError(
            f"Resolved callable rejected sparse recipe for {call.call_id!r}: {exc}",
            code=RunnableErrorCode.RUNTIME_SIGNATURE_DRIFT.value,
            call_id=call.call_id,
            affected_op_labels=call.op_labels,
        ) from exc


def _populate_source_slots(
    fork: Any,
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
    *,
    ceiling: RunResourceCeiling,
    witness_slot_ids: frozenset[str] = frozenset(),
    witness_source_snapshots: dict[str, torch.Tensor] | None = None,
) -> None:
    """Populate input and buffer source Ops on the transactional run fork.

    A source slot (model input / buffer) that is itself a tensor->host escape
    witness site is snapshotted here, at population -- the mutation-consistent point
    matching the save-side digest -- so a later in-place op cannot restale the
    staleness comparison.
    """

    for slot in descriptor.tensor_slots:
        if slot.role not in {TensorSlotRole.MODEL_INPUT, TensorSlotRole.BUFFER}:
            continue
        value = slot_values.get(slot.slot_id)
        op = _op_for_slot(fork, slot.slot_id)
        if value is not None and op is not None:
            op._internal_set(
                "out",
                ceiling.guarded_clone(
                    value,
                    call_id=None,
                    slot_id=slot.slot_id,
                    affected_op_labels=(),
                ),
            )
        if (
            value is not None
            and witness_source_snapshots is not None
            and slot.slot_id in witness_slot_ids
        ):
            witness_source_snapshots[slot.slot_id] = ceiling.guarded_clone(
                value,
                call_id=None,
                slot_id=slot.slot_id,
                affected_op_labels=(),
            )


def _write_argument(
    args: list[Any], kwargs: dict[str, Any], path: tuple[str | int, ...], value: Any
) -> None:
    """Write one reconstructed value at an args/kwargs argument path."""

    if len(path) < 2 or path[0] not in {"args", "kwargs"}:
        raise RunPreconditionError(
            f"Invalid sparse argument path {path!r}.",
            code=RunnableErrorCode.CALL_STRUCTURE_MISMATCH.value,
        )
    root: Any = args if path[0] == "args" else kwargs
    _write_path(root, path[1:], value)


def _write_path(root: Any, path: tuple[str | int, ...], value: Any) -> None:
    """Write a value into a dynamically reconstructed list/dict tree."""

    current = root
    for index, component in enumerate(path):
        last = index == len(path) - 1
        if last:
            current[component] = value
            return
        next_component = path[index + 1]
        if isinstance(current, list):
            child = current[cast(int, component)]
            if child is None:
                child = [] if isinstance(next_component, int) else {}
                current[cast(int, component)] = child
        else:
            child = current.get(component)
            if child is None:
                child = [] if isinstance(next_component, int) else {}
                current[component] = child
        current = child


def _resolve_setter_output(
    call: RunnableCallDescriptor,
    output: Any,
    slot_values: Mapping[str, torch.Tensor],
) -> Any:
    """Alias the mutation target for a setter-style in-place call that returns None.

    Ordinary in-place operators (``add_``/``mul_``/``copy_``/``out=``) return the
    tensor they mutated, so the recorded output slot is bound from the Python
    return value. Setter-style mutators such as ``Tensor.__setitem__`` mutate
    their target in place but return ``None``. Their recorded output slot is a
    version of the mutation target, so bind it from that already-mutated tensor
    rather than treating the ``None`` return as a structural mismatch (which would
    otherwise raise a false PathDivergenceError on the original input). Only a
    genuinely in-place call whose runtime return is ``None`` is remapped; every
    other call keeps its real return so honest structure/aliasing checks stand.
    """

    if output is not None or not call.is_inplace:
        return output
    target_slot_id = _mutation_target_slot_id(call)
    if target_slot_id is None:
        return output
    target = slot_values.get(target_slot_id)
    if not isinstance(target, torch.Tensor):
        return output
    return target
