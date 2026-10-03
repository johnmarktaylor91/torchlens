"""State contracts and captured execution contexts."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import torch

from ._io._torch_symbols import torch_attr
from ._runnable_state import (
    _OUTPUT_COUNT_FLOOR,
    state_metadata_full_violations,
)
from .errors import (
    RunPreconditionError,
)
from .runnable import (
    ContractCheck,
    RunnableCallDescriptor,
    RunnableErrorCode,
    SparseRunDescriptor,
    StateSlotRole,
    TensorSlotDescriptor,
    TensorSlotRole,
)
from .utils._torch_compat import tensor_version_or_none

if TYPE_CHECKING:
    from ._runnable_execution import (
        _ALLOCATOR_SIGNATURES,
        _contract_check,
        _ProjectionCountExceeded,
    )

__all__ = (
    "_state_contract_checks",
    "_allowed_state_roles",
    "_pre_call_contract_checks",
    "_context_unavailable_error",
    "_ambient_execution_context_restored",
    "_call_execution_context_entered",
    "_is_allocator_death",
    "_fake_tensor_mode_class",
    "_count_fake_tensor_leaves",
    "_count_bounded_fake_tensor_mode_class",
)


def _state_contract_checks(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> tuple[ContractCheck, ...]:
    """Recheck state tensor and alias contracts inside the DAG transaction."""

    checks: list[ContractCheck] = []
    aliases: dict[str, list[TensorSlotDescriptor]] = {}
    for slot in descriptor.tensor_slots:
        if slot.role not in {TensorSlotRole.PARAMETER, TensorSlotRole.BUFFER}:
            continue
        binding = slot.state_binding
        value = slot_values.get(slot.slot_id)
        present = isinstance(value, torch.Tensor)
        checks.append(
            _contract_check(
                f"state_slot:{slot.slot_id}",
                present,
                RunnableErrorCode.MISSING_TENSOR_SLOT,
                f"State slot {slot.slot_id!r} was not bound for execution.",
                details=(("slot_id", slot.slot_id),),
            )
        )
        if not present:
            continue
        assert value is not None
        checks.append(
            _contract_check(
                f"state_shape:{slot.slot_id}",
                tuple(value.shape) == slot.shape,
                RunnableErrorCode.STATE_SHAPE_MISMATCH,
                f"State slot {slot.slot_id!r} has a runtime shape mismatch.",
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_shape", repr(slot.shape)),
                    ("actual_shape", repr(tuple(value.shape))),
                ),
            )
        )
        # r37 R5 in-transaction tripwire: staging is the sole placement authority,
        # so a device mismatch HERE is a broken staging/state hook -- a typed state
        # failure before any callable, never a mid-call runtime_signature_drift.
        device_ok = value.device.type == slot.device_type and (
            slot.device_index is None
            or value.device.index is None
            or value.device.index == slot.device_index
        )
        checks.append(
            _contract_check(
                f"state_device:{slot.slot_id}",
                device_ok,
                RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
                f"State slot {slot.slot_id!r} was not staged to its recorded device.",
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_device", f"{slot.device_type}:{slot.device_index}"),
                    ("actual_device", str(value.device)),
                ),
            )
        )
        checks.append(
            _contract_check(
                f"state_dtype:{slot.slot_id}",
                str(value.dtype) == slot.dtype,
                RunnableErrorCode.STATE_DTYPE_MISMATCH,
                f"State slot {slot.slot_id!r} has a runtime dtype mismatch.",
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_dtype", slot.dtype),
                    ("actual_dtype", str(value.dtype)),
                ),
            )
        )
        # r63 C1 in-transaction tripwire: the canonical staging clone is the sole state
        # placement authority, so every staged runtime state tensor must still exhibit the
        # full canonical metadata signature (admitted exact class, strided dense layout,
        # unnamed, default stride, zero offset, no lazy conj/neg) before any recorded
        # callable observes it -- a mismatch here is a broken/bypassed staging path or a
        # post-bind mutation of staged state, refused typed rather than replayed.
        metadata_violations = state_metadata_full_violations(value)
        checks.append(
            _contract_check(
                f"state_metadata:{slot.slot_id}",
                not metadata_violations,
                RunnableErrorCode.STATE_METADATA_MISMATCH,
                f"State slot {slot.slot_id!r} no longer carries the canonical staged "
                f"metadata signature (violations: {metadata_violations!r}).",
                details=(
                    ("slot_id", slot.slot_id),
                    ("violations", repr(metadata_violations)),
                ),
            )
        )
        if binding is not None and binding.alias_group is not None:
            aliases.setdefault(binding.alias_group, []).append(slot)
        if binding is not None:
            module_path, separator, leaf_name = binding.state_dict_name.rpartition(".")
            canonical_module = module_path if separator else "self"
            allowed_roles = _allowed_state_roles(leaf_name, slot.role)
            checks.append(
                _contract_check(
                    f"state_name_role:{slot.slot_id}",
                    bool(binding.state_dict_name)
                    and binding.module_path == canonical_module
                    and binding.semantic_role in allowed_roles,
                    RunnableErrorCode.STATE_ROLE_MISMATCH,
                    f"State slot {slot.slot_id!r} has an inconsistent name/role contract.",
                    details=(
                        ("slot_id", slot.slot_id),
                        ("state_dict_name", binding.state_dict_name),
                        ("recorded_module_path", binding.module_path),
                        ("canonical_module_path", canonical_module),
                        ("semantic_role", binding.semantic_role.value),
                    ),
                )
            )
    for alias_group, members in sorted(aliases.items()):
        values = [slot_values[slot.slot_id] for slot in members if slot.slot_id in slot_values]
        checks.append(
            _contract_check(
                f"state_alias:{alias_group}",
                bool(values) and all(value is values[0] for value in values[1:]),
                RunnableErrorCode.STATE_ALIAS_CONFLICT,
                f"State alias group {alias_group!r} did not retain one shared tensor.",
                details=(
                    ("alias_group", alias_group),
                    ("slot_ids", repr(tuple(slot.slot_id for slot in members))),
                ),
            )
        )
    return tuple(checks)


def _allowed_state_roles(
    leaf_name: str,
    slot_role: TensorSlotRole,
) -> frozenset[StateSlotRole]:
    """Return canonical semantic roles for a state-dict leaf name."""

    if leaf_name == "weight":
        return frozenset({StateSlotRole.WEIGHT, StateSlotRole.NORM_SCALE})
    if leaf_name == "bias":
        return frozenset({StateSlotRole.BIAS, StateSlotRole.NORM_OFFSET})
    if leaf_name == "running_mean":
        return frozenset({StateSlotRole.RUNNING_MEAN})
    if leaf_name == "running_var":
        return frozenset({StateSlotRole.RUNNING_VAR})
    if leaf_name in {"num_batches_tracked", "counter"}:
        return frozenset({StateSlotRole.COUNTER})
    if slot_role is TensorSlotRole.BUFFER:
        return frozenset({StateSlotRole.GENERIC_BUFFER})
    return frozenset({StateSlotRole.WEIGHT})


def _pre_call_contract_checks(
    descriptor: SparseRunDescriptor,
    call: RunnableCallDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> tuple[tuple[ContractCheck, ...], dict[str, int]]:
    """Validate callable dispatch/arity metadata and snapshot input versions."""

    registry_entry = next(
        (entry for entry in descriptor.callable_registry if entry.registry_id == call.registry_id),
        None,
    )
    valid_dispatch = (
        call.dispatch_kind in {"function", "method", "dunder", "namespace_alias"}
        and registry_entry is not None
        and registry_entry.key.dispatch_kind == call.dispatch_kind
    )
    referenced_paths = [argument.argument_path for argument in call.tensor_arguments] + [
        argument.argument_path for argument in call.literal_arguments
    ]
    positional_indices = {
        cast(int, path[1])
        for path in referenced_paths
        if len(path) >= 2 and path[0] == "args" and isinstance(path[1], int)
    }
    keyword_names = {
        cast(str, path[1])
        for path in referenced_paths
        if len(path) >= 2 and path[0] == "kwargs" and isinstance(path[1], str)
    }
    # r53 free_1: allocation-free dense pigeonhole. A set of nonnegative ints
    # equals ``set(range(n))`` iff it has exactly ``n`` members spanning
    # ``[0, n-1]`` -- checked WITHOUT materializing ``set(range(n))`` from the
    # persisted integer, so an in-memory descriptor bypassing parse anchoring
    # can no longer scale this tripwire into an allocation bomb. (A negative
    # ``n`` now fails the check outright; previously ``set(range(-n))`` was
    # empty and vacuously matched an empty leaf set.)
    n_positional = call.num_positional_args
    positional_dense = len(positional_indices) == n_positional and (
        n_positional == 0
        or (min(positional_indices) == 0 and max(positional_indices) == n_positional - 1)
    )
    arity_ok = positional_dense and len(keyword_names) == call.num_keyword_args
    checks = (
        _contract_check(
            f"call_dispatch:{call.call_id}",
            valid_dispatch,
            RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
            f"Call {call.call_id!r} has an unsupported dispatch contract.",
            affected_op_labels=call.op_labels,
            details=(("dispatch_kind", call.dispatch_kind),),
        ),
        _contract_check(
            f"call_arity:{call.call_id}",
            arity_ok,
            RunnableErrorCode.CALL_ARITY_MISMATCH,
            f"Call {call.call_id!r} argument leaves do not satisfy its recorded arity.",
            affected_op_labels=call.op_labels,
            details=(
                ("expected_positional", str(call.num_positional_args)),
                ("actual_positional_sites", repr(sorted(positional_indices))),
                ("expected_keyword", str(call.num_keyword_args)),
                ("actual_keyword_names", repr(sorted(keyword_names))),
            ),
        ),
    )
    # r37 hon1_4: inference tensors carry NO version counter (reading ``_version``
    # raises), so a slot whose version is unavailable records NO baseline. The
    # mutation tripwire then enforces only its version-independent legs for that
    # slot (alias identity for in-place calls); value fidelity remains guarded by
    # the output comparison and numeric attestation layers.
    versions: dict[str, int] = {}
    for argument in call.tensor_arguments:
        value = slot_values.get(argument.slot_id)
        if value is None:
            continue
        version = tensor_version_or_none(value)
        if version is not None:
            versions[argument.slot_id] = version
    return checks, versions


def _context_unavailable_error(field: str, detail: str) -> RunPreconditionError:
    """Build the typed refusal for an un-enterable/un-restorable execution context."""

    return RunPreconditionError(
        f"Recorded execution context {field!r} cannot be entered or restored on "
        f"this runtime: {detail}",
        code=RunnableErrorCode.EXECUTION_CONTEXT_UNAVAILABLE.value,
        context_field=field,
    )


@contextmanager
def _ambient_execution_context_restored(ambient: Any) -> Any:
    """Transactionally restore the recorded capture-scoped ambient context (decision E).

    The caller's ambient state is snapshotted, the recorded values are applied
    (``None`` producer-absent fields are left as-is -- there is nothing recorded
    to restore), and the caller's state is re-applied in ``finally`` on every
    exit: success, divergence, callable exception, and numeric-attestation
    rollback. A recorded value this runtime cannot apply rolls back any partial
    application and raises the typed ``execution_context_unavailable`` refusal
    -- never a silent ambient passthrough.
    """

    from .utils._torch_compat import (
        apply_ambient_execution_context,
        read_fp32_precision_policy,
        snapshot_ambient_execution_context,
        write_fp32_precision_policy,
    )

    recorded = {
        "default_dtype": ambient.default_dtype,
        "default_device": ambient.default_device,
        "float32_matmul_precision": ambient.float32_matmul_precision,
        "deterministic_algorithms": ambient.deterministic_algorithms,
        "deterministic_algorithms_warn_only": ambient.deterministic_algorithms_warn_only,
        "cuda_matmul_allow_tf32": ambient.cuda_matmul_allow_tf32,
        "cudnn_allow_tf32": ambient.cudnn_allow_tf32,
        "cudnn_deterministic": ambient.cudnn_deterministic,
        "cudnn_benchmark": ambient.cudnn_benchmark,
        "cudnn_enabled": ambient.cudnn_enabled,
        "flash_sdp_enabled": ambient.flash_sdp_enabled,
        "mem_efficient_sdp_enabled": ambient.mem_efficient_sdp_enabled,
        "math_sdp_enabled": ambient.math_sdp_enabled,
        # r53 hon_2: deterministic uninit-memory fill restores transactionally
        # (setter-based) so a deterministic+fill capture NaN-fills identically.
        "fill_uninitialized_memory": ambient.fill_uninitialized_memory,
    }
    saved = snapshot_ambient_execution_context()
    # The legacy fp32 views in ``saved`` cannot express a per-backend
    # ``fp32_precision`` policy (torch >= 2.9), so the caller's exact policy is
    # kept separately and written back after every ``saved`` re-apply.
    saved_fp32_policy = read_fp32_precision_policy()

    def _restore_caller_ambient() -> None:
        """Re-apply the caller's ambient context, exact fp32 policy included."""

        apply_ambient_execution_context(saved)
        write_fp32_precision_policy(saved_fp32_policy)

    # r37 R4 (corr2-3/corr2-2): the recorded DEFAULT DEVICE is entered as a SCOPED
    # ``with torch.device(recorded)`` mode nested above the caller's existing mode
    # stack -- never via ``torch.set_default_device`` (which mutates process-global
    # mode bookkeeping and, measured, leaks/clobbers DeviceContext modes on every
    # policy: implicit callers gained a mode, nested callers were corrupted). The
    # context-manager exit IS the restoration mechanism -- correct by construction on
    # success, divergence, callable exception, and attestation rollback -- so no
    # restore logic exists for the device at all. A mode-stack length postcondition
    # (feature-probed introspection) is a belt-and-suspenders tripwire.
    device_scope: Any = nullcontext()
    if ambient.default_device is not None:
        try:
            device_scope = torch.device(str(ambient.default_device))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise _context_unavailable_error("default_device", str(exc)) from exc
    from .utils._torch_compat import get_current_function_mode_stack

    stack_before = get_current_function_mode_stack()
    depth_before = len(list(stack_before)) if stack_before is not None else None
    try:
        apply_ambient_execution_context(recorded)
    except RuntimeError as exc:
        try:
            _restore_caller_ambient()
        except RuntimeError:  # pragma: no cover - saved values came from this runtime
            pass
        raise _context_unavailable_error("ambient_context", str(exc)) from exc
    try:
        # r53 hon_1: the recorded GLOBAL autograd/inference mode is entered as
        # SCOPED contexts around the whole sparse run (outside every per-call
        # context), so a Python branch on ``torch.is_grad_enabled()`` /
        # ``is_inference_mode_enabled()`` inside replayed code observes the
        # SAME global a fresh instance under the recorded ambient would --
        # never the caller's ambient mode. Both nest legally inside a caller's
        # ``no_grad()``/``inference_mode()`` and restore the caller's exact
        # thread-local mode on every exit path by construction (the same
        # correct-by-construction posture as the scoped default device above).
        # A Python READ of either flag is deliberately never ceilinged:
        # library code reads them constantly, and record+restore makes the
        # branch deterministic, so a witness would be a mass over-trigger for
        # zero honesty gain.
        with (
            device_scope,
            torch.set_grad_enabled(bool(ambient.grad_enabled)),
            torch.inference_mode(bool(ambient.inference_mode)),
        ):
            yield
    finally:
        # Fence the restore itself: this is 12 sequential process-global torch mutations,
        # and a raise here used to skip the mode-stack tripwire below and leave the
        # caller's globals BLENDED between the recorded and saved contexts with no
        # diagnostic at all. The restore failure is re-raised after the tripwire runs, so
        # neither signal is swallowed.
        restore_error: BaseException | None = None
        try:
            _restore_caller_ambient()
        except BaseException as error:  # noqa: BLE001 - re-raised below, never swallowed
            restore_error = error
        if depth_before is not None and sys.exc_info()[0] is None:
            stack_after = get_current_function_mode_stack()
            depth_after = len(list(stack_after)) if stack_after is not None else None
            if depth_after is not None and depth_after != depth_before:
                raise RuntimeError(
                    "Internal invariant violation: the run transaction changed the "
                    f"caller's TorchFunctionMode stack depth ({depth_before} -> "
                    f"{depth_after}); scoped device-context restoration must be "
                    "exact on every exit path."
                )
        if restore_error is not None:
            raise restore_error


@contextmanager
def _call_execution_context_entered(context: Any) -> Any:
    """Enter the REQUIRED recorded per-call execution context tightly (corr2_8).

    Autocast: an ``enabled=True`` record enters autocast with the recorded
    dtype; an explicit ``enabled=False`` record actively enters a DISABLED
    autocast context when the runtime currently has that device class enabled
    (so a caller's ambient autocast cannot contaminate a disabled capture) and
    is vacuously satisfied when the device class is absent/disabled. Grad and
    inference modes are entered explicitly. Context entry never touches RNG;
    the caller's context is restored in reverse order on every exit. An
    un-enterable recorded context is a typed refusal, never a raw torch error.
    """

    from .utils._torch_compat import autocast_is_enabled

    stack: list[Any] = []
    try:
        for entry in context.autocast:
            if entry.enabled:
                dtype_name = str(entry.dtype or "").removeprefix("torch.")
                # r45 secC_1 (defense-in-depth): route through the single ``torch_attr`` helper
                # so this decode site cannot fire ``torch.__getattr__`` (lazy import /
                # deprecated ``replacement()``) even if a context is ever built off the parser
                # path. The subsequent ``isinstance(..., torch.dtype)`` gate is unchanged.
                dtype = torch_attr(dtype_name)
                if not isinstance(dtype, torch.dtype):
                    raise _context_unavailable_error(
                        f"autocast:{entry.device_type}",
                        f"recorded autocast dtype {entry.dtype!r} is unavailable",
                    )
                try:
                    ctx = torch.amp.autocast(entry.device_type, enabled=True, dtype=dtype)
                    ctx.__enter__()
                except (RuntimeError, ValueError, TypeError) as exc:
                    raise _context_unavailable_error(
                        f"autocast:{entry.device_type}", str(exc)
                    ) from exc
                stack.append(ctx)
                continue
            # Explicit disabled record: enter a disabled context only when the
            # runtime reports that device class currently autocast-enabled; a
            # runtime without the device class is vacuously disabled already.
            try:
                currently_enabled = bool(autocast_is_enabled(entry.device_type))
            except (RuntimeError, TypeError):
                currently_enabled = False
            if currently_enabled:
                try:
                    ctx = torch.amp.autocast(entry.device_type, enabled=False)
                    ctx.__enter__()
                except (RuntimeError, ValueError, TypeError) as exc:
                    raise _context_unavailable_error(
                        f"autocast:{entry.device_type}", str(exc)
                    ) from exc
                stack.append(ctx)
        try:
            inference_ctx = torch.inference_mode(bool(context.inference_mode))
            inference_ctx.__enter__()
            stack.append(inference_ctx)
            grad_ctx = torch.enable_grad() if context.grad_enabled else torch.no_grad()
            grad_ctx.__enter__()
            stack.append(grad_ctx)
        except RuntimeError as exc:
            raise _context_unavailable_error("grad_mode", str(exc)) from exc
        yield
    finally:
        # Per-context fence. One raising ``__exit__`` used to skip every OUTER context,
        # stranding the CALLER's thread inside ``no_grad`` / ``inference_mode`` / an
        # autocast it never asked for for the rest of the process. Every context gets its
        # chance to exit; the first failure is re-raised once the unwind is complete.
        first_error: BaseException | None = None
        for ctx in reversed(stack):
            try:
                ctx.__exit__(None, None, None)
            except BaseException as error:  # noqa: PERF203 - per-item fence is the point
                if first_error is None:
                    first_error = error
        stack.clear()
        if first_error is not None:
            raise first_error


def _is_allocator_death(exc: BaseException) -> bool:
    """Return whether an exception is an allocation failure (fail-closed, not fail-open)."""

    if isinstance(exc, MemoryError):
        return True
    oom = getattr(torch, "OutOfMemoryError", None)
    if isinstance(oom, type) and isinstance(exc, oom):
        return True
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if isinstance(cuda_oom, type) and isinstance(exc, cuda_oom):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc)
        return any(sig in message for sig in _ALLOCATOR_SIGNATURES)
    return False


@lru_cache(maxsize=1)
def _fake_tensor_mode_class() -> type[Any] | None:
    """Return torch's ``FakeTensorMode`` class if importable, else ``None``.

    Routed through ``_torch_compat`` (``HAS_FAKE_TENSOR_MODE``, r-b4 R26-2):
    the projection is a runtime allocation preflight, not a capture-hot probe,
    so an absent ``FakeTensorMode`` still fails OPEN to the run-prep
    recorded-output bound rather than refusing a legitimate run -- but the
    degradation now flips a named flag visible in doctor/compat instead of
    silently vanishing.
    """

    from .utils._torch_compat import get_fake_tensor_mode_class

    return get_fake_tensor_mode_class()


def _count_fake_tensor_leaves(tree: Any) -> int:
    """Count tensor leaves in a projection output tree (bounded container kinds)."""

    total = 0
    stack: list[Any] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, torch.Tensor):
            total += 1
        elif isinstance(node, (tuple, list)):
            stack.extend(node)
        elif isinstance(node, Mapping):
            stack.extend(node.values())
    return total


@lru_cache(maxsize=1)
def _count_bounded_fake_tensor_mode_class() -> type[Any] | None:
    """Return a ``FakeTensorMode`` subclass that caps realized fake-output arity (r59).

    The subclass counts fake tensor leaves produced by EACH ``__torch_dispatch__`` and
    raises ``_ProjectionCountExceeded`` the instant the running total passes the per-call
    ceiling -- DURING fanout construction, before the whole fake output tree materializes.
    This kills the ``tensor_split(x, N)`` self-DoS (r58 free_1): projecting the real op to
    "see" its size would itself build N fake objects; counting during construction aborts
    at ``ceiling + 1`` fakes regardless of N. Data-dependent ops (``nonzero``/``unique``)
    still raise their own ``DynamicOutputShapeException`` through ``super()`` untouched, so
    the fail-open path is preserved. ``None`` when ``FakeTensorMode`` is unavailable.
    """

    base = _fake_tensor_mode_class()
    if base is None:
        return None

    class _CountBoundedFakeTensorMode(base):  # type: ignore[valid-type,misc]
        """``FakeTensorMode`` that aborts once a projection realizes too many fakes.

        The count is per mode instance and cumulative across dispatches, minus
        the baseline set by :meth:`_tl_set_baseline`; crossing ``ceiling``
        raises ``_ProjectionCountExceeded`` mid-construction rather than after
        the output tree exists. Data-dependent shape failures still propagate
        from ``super()`` unchanged.
        """

        def __init__(self, *args: Any, ceiling: int = _OUTPUT_COUNT_FLOOR, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._tl_ceiling = ceiling
            self._tl_fake_count = 0
            self._tl_baseline = 0

        def _tl_set_baseline(self) -> None:
            """Zero the ceiling budget at the current count, after input conversion.

            Only projection OUTPUT tensors may consume the budget. Call this
            once, after converting the real inputs and before dispatching the
            projected op (``from_tensor`` is not contractually dispatch-silent
            across torch versions, so the discount cannot be assumed to be nil).
            """

            self._tl_baseline = self._tl_fake_count

        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            out = super().__torch_dispatch__(func, types, args, kwargs)
            self._tl_fake_count += _count_fake_tensor_leaves(out)
            if self._tl_fake_count - self._tl_baseline > self._tl_ceiling:
                raise _ProjectionCountExceeded(
                    f"projection realized {self._tl_fake_count - self._tl_baseline} fake "
                    f"outputs, exceeding ceiling {self._tl_ceiling}"
                )
            return out

    return _CountBoundedFakeTensorMode
