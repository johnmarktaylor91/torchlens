"""Explicit guarded fast paths for repeated static-model execution."""

from __future__ import annotations

import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from . import _state
from ._call_fingerprint import fingerprinting, install_module_token_hooks
from ._fast_live_plans import (
    _FastOutputPlan,
    _pair_output_leaves,
    build_function_plans,
    build_module_plans,
    recorded_module_call_counts,
)
from ._fast_live_steer import (
    SteerPlan,
    clear_unrefreshed_shape_metadata,
    fast_live_input_admission,
    hooked_run_diagnostic,
    install_steer_hooks,
    refusal_code,
    session_is_active,
)
from ._fast_live_until import (
    FastUntilReached,
    function_prefix_count,
    rearm_cleared_sites,
    resolve_fast_until,
    truncate_after,
)
from ._fast_sparse_state import _state_mutating_call_ids
from ._runnable_execution import (
    _HOST_RNG_SOURCE_KIND,
    _INPUT_CHECK_UNAVAILABLE,
    _ambient_execution_context_restored,
    _call_execution_context_entered,
    _call_witness_checks,
    _container_spec_reconstruction_lossy,
    _contract_check,
    _control_witness_source_slot_ids,
    _declared_nondeterministic_sources,
    _decode_literal,
    _descriptor_has_seeded_rng,
    _finalize_provider_run,
    _first_failed_live_input_check,
    _host_rng_unreproduced,
    _input_alias_topology_checks,
    _input_derived_layout_stale,
    _input_literal_contract_checks,
    _input_metadata_contract_checks,
    _input_nontensor_tree_contract_checks,
    _input_site_value,
    _input_tree_contract_checks,
    _live_runtime_input_leaves,
    _mode_sensitive_op_unwitnessed,
    _model_input_arity_positions,
    _nondeterministic_value_sources,
    _out_argument_slot_id,
    _output_container_spec,
    _output_not_reproduced,
    _path_faithfulness,
    _post_execution_contract_checks,
    _raise_failed_contract_as_divergence,
    _raise_first_divergence,
    _require_loaded_sparse_provider,
    _runtime_mirror_clone,
    _seed_run_generators,
    _seeded_fork_devices,
    _split_mixed_inputs,
    _tensor_derived_scalar_stale,
    _tensor_derived_scalar_witness_slot_ids,
    _tensor_leaf_paths,
    _top_level_input_site_contract_checks,
    _unbound_state_escape_stale,
    _uninit_taint_reaches,
    _value_at_path,
    _write_argument,
    run_loaded_sparse_trace,
)
from ._runnable_state import PreparedRunnableState, RunResourceCeiling, prepare_runnable_state
from .data_classes._trace_accessors import boundary_label_rows
from .errors import PathDivergenceError, RunCapabilityUnavailableError, RuntimeSignatureDriftError
from .ir.container import ContainerSpec, rebuild_container_from_spec
from .runnable import (
    ContractCheck,
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessReport,
    ReadinessStatus,
    RunnableCallDescriptor,
    RunnableErrorCode,
    RunProvider,
    RunResult,
    SparseRunDescriptor,
    StateSource,
    TensorSlotDescriptor,
    TensorSlotRole,
    mark_trace_path_status,
)
from .utils._torch_compat import tensor_has_named_dims
from .utils.rng import restore_host_rng, set_random_seed, snapshot_host_rng


@dataclass(frozen=True, slots=True)
class _CompiledSparseCall:
    """One sparse call with decoded literals and pre-indexed tensor bindings."""

    descriptor: RunnableCallDescriptor
    func: Callable[..., Any]
    base_args: tuple[Any, ...]
    base_kwargs: Mapping[str, Any]
    top_level_tensor_args: tuple[tuple[str, int | str, str], ...]
    nested_tensor_args: tuple[tuple[tuple[str | int, ...], str], ...]
    requires_context: bool

    def execute(self, slot_values: Mapping[str, torch.Tensor]) -> Any:
        """Bind current tensor slots and invoke the resolved callable.

        Parameters
        ----------
        slot_values:
            Runtime tensor slot store.

        Returns
        -------
        Any
            Raw callable output.
        """

        args = list(self.base_args)
        kwargs = dict(self.base_kwargs)
        for root, key, slot_id in self.top_level_tensor_args:
            try:
                value = slot_values[slot_id]
            except KeyError as exc:
                raise RuntimeSignatureDriftError(
                    f"Fast sparse call {self.descriptor.call_id!r} references unavailable "
                    f"slot {slot_id!r}.",
                    code=RunnableErrorCode.MISSING_TENSOR_SLOT.value,
                    call_id=self.descriptor.call_id,
                    slot_id=slot_id,
                ) from exc
            if root == "args":
                args[cast(int, key)] = value
            else:
                kwargs[cast(str, key)] = value
        for path, slot_id in self.nested_tensor_args:
            try:
                value = slot_values[slot_id]
            except KeyError as exc:
                raise RuntimeSignatureDriftError(
                    f"Fast sparse call {self.descriptor.call_id!r} references unavailable "
                    f"slot {slot_id!r}.",
                    code=RunnableErrorCode.MISSING_TENSOR_SLOT.value,
                    call_id=self.descriptor.call_id,
                    slot_id=slot_id,
                ) from exc
            _write_argument(args, kwargs, path, value)
        try:
            if not self.requires_context:
                return self.func(*args, **kwargs)
            with _call_execution_context_entered(self.descriptor.execution_context):
                return self.func(*args, **kwargs)
        except RunCapabilityUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeSignatureDriftError(
                f"Resolved callable rejected compiled fast recipe for "
                f"{self.descriptor.call_id!r}: {exc}",
                code=RunnableErrorCode.RUNTIME_SIGNATURE_DRIFT.value,
                call_id=self.descriptor.call_id,
                affected_op_labels=self.descriptor.op_labels,
            ) from exc


def _compile_sparse_call(
    call: RunnableCallDescriptor,
    func: Callable[..., Any],
    *,
    ambient_grad_enabled: bool,
    ambient_inference_mode: bool,
) -> _CompiledSparseCall:
    """Compile a sparse call's invariant literal tree and tensor assignments.

    Parameters
    ----------
    call:
        Frozen call recipe.
    func:
        Resolved callable.
    ambient_grad_enabled:
        Grad mode already restored around the compiled loop.
    ambient_inference_mode:
        Inference mode already restored around the compiled loop.

    Returns
    -------
    _CompiledSparseCall
        Reusable per-call binder.
    """

    args: list[Any] = [None] * call.num_positional_args
    kwargs: dict[str, Any] = {}
    for literal in call.literal_arguments:
        _write_argument(args, kwargs, literal.argument_path, _decode_literal(literal.value))
    top_level: list[tuple[str, int | str, str]] = []
    nested: list[tuple[tuple[str | int, ...], str]] = []
    for argument in call.tensor_arguments:
        path = tuple(argument.argument_path)
        if len(path) == 2 and path[0] in {"args", "kwargs"}:
            top_level.append((cast(str, path[0]), path[1], argument.slot_id))
        else:
            nested.append((path, argument.slot_id))
    return _CompiledSparseCall(
        descriptor=call,
        func=func,
        base_args=tuple(args),
        base_kwargs=kwargs,
        top_level_tensor_args=tuple(top_level),
        nested_tensor_args=tuple(nested),
        requires_context=(
            call.execution_context.grad_enabled != ambient_grad_enabled
            or call.execution_context.inference_mode != ambient_inference_mode
            or any(entry.enabled for entry in call.execution_context.autocast)
        ),
    )


def _input_layout_ok(value: torch.Tensor) -> bool:
    """Return whether an input is in the fast sparse executor's admitted domain."""

    return (
        type(value) in {torch.Tensor, torch.nn.Parameter}
        and value.layout == torch.strided
        and not value.is_nested
        and not bool(getattr(value, "is_meta", False))
        and not bool(getattr(value, "is_quantized", False))
        and not tensor_has_named_dims(value)
    )


def _fast_input_check(
    slot: TensorSlotDescriptor,
    value: Any,
) -> tuple[ContractCheck, ...]:
    """Build cheap shape/dtype/device/layout checks for one runtime input leaf.

    Parameters
    ----------
    slot:
        Persisted input slot contract.
    value:
        Runtime value resolved at that slot's path.

    Returns
    -------
    tuple[ContractCheck, ...]
        Ordered fast static-guard checks.
    """

    is_tensor = isinstance(value, torch.Tensor)
    shape = tuple(value.shape) if is_tensor else None
    dtype = str(value.dtype) if is_tensor else None
    device_ok = bool(
        is_tensor
        and value.device.type == slot.device_type
        and (
            slot.device_index is None
            or value.device.index is None
            or value.device.index == slot.device_index
        )
    )
    return (
        _contract_check(
            f"fast_input_type:{slot.slot_id}",
            is_tensor,
            RunnableErrorCode.INPUT_TREE_MISMATCH,
            f"Runtime input at {slot.slot_id!r} is not a tensor.",
        ),
        _contract_check(
            f"fast_input_shape:{slot.slot_id}",
            shape == slot.shape,
            RunnableErrorCode.INPUT_SHAPE_MISMATCH,
            f"Runtime input shape {shape} does not match {slot.shape}.",
        ),
        _contract_check(
            f"fast_input_dtype:{slot.slot_id}",
            dtype == slot.dtype,
            RunnableErrorCode.INPUT_DTYPE_MISMATCH,
            f"Runtime input dtype {dtype} does not match {slot.dtype}.",
        ),
        _contract_check(
            f"fast_input_device:{slot.slot_id}",
            device_ok,
            RunnableErrorCode.INPUT_TREE_MISMATCH,
            f"Runtime input device does not match {slot.device_type}:{slot.device_index}.",
        ),
        _contract_check(
            f"fast_input_layout:{slot.slot_id}",
            bool(is_tensor and _input_layout_ok(value)),
            RunnableErrorCode.INPUT_TREE_MISMATCH,
            "Fast sparse execution accepts only plain dense strided tensor inputs.",
        ),
    )


def _module_call_budget(
    trace: Any, module_plans: Sequence[_FastOutputPlan]
) -> tuple[dict[str, frozenset[int]], dict[str, int]]:
    """Return the planned calls per module address and each address's recorded call count.

    Parameters
    ----------
    trace:
        Source trace whose ``module_calls`` the counts come from.
    module_plans:
        Module output plans the fast session collects.

    Returns
    -------
    tuple[dict[str, frozenset[int]], dict[str, int]]
        The 1-based call numbers planned per address, and the number of calls
        the capture recorded per planned address.
    """

    planned: dict[str, frozenset[int]] = {}
    for plan in module_plans:
        planned[plan.address_or_name] = planned.get(plan.address_or_name, frozenset()) | {
            int(plan.call_index or 1)
        }
    return planned, recorded_module_call_counts(trace, frozenset(planned))


def _admitted_collection_labels(trace: Any, plans: Sequence[_FastOutputPlan]) -> set[str]:
    """Return the op labels the fast live session refreshes, refusing uncollectable saves.

    Parameters
    ----------
    trace:
        Source trace whose saved ops the plans must cover.
    plans:
        Module and function output plans the session collects.

    Returns
    -------
    set[str]
        Labels the plans collect plus the model's boundary ops.

    Raises
    ------
    RunCapabilityUnavailableError
        When a saved functional op is covered by no plan.
    """

    boundary_layer_labels = set(getattr(trace, "input_layers", ())) | set(
        getattr(trace, "output_layers", ())
    )
    supported_labels = {label for plan in plans for label in plan.save_labels} | {
        op.label for op in trace.layer_list if op.layer_label in boundary_layer_labels
    }
    unsupported_function_labels = tuple(
        op.label
        for op in trace.layer_list
        if bool(getattr(op, "has_saved_activation", False))
        and str(getattr(op, "func_name", "none")) not in {"none", "identity"}
        and op.label not in supported_labels
    )
    if unsupported_function_labels:
        raise RunCapabilityUnavailableError(
            "fast=True collects functional operations only when they were explicitly "
            "requested by the capture save predicate. Recapture with save=tl.func(...) "
            f"for: {', '.join(unsupported_function_labels[:8])}.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="fast_live_function_plan",
        )
    return supported_labels


def _drop_uncollected_payloads(trace: Any, supported_labels: set[str]) -> None:
    """Clear saved payloads the fast live session will not refresh.

    Parameters
    ----------
    trace:
        Source trace whose ops are updated in place.
    supported_labels:
        Op labels the session's module and function plans collect, plus the
        model's boundary ops.
    """

    for op in trace.layer_list:
        if bool(getattr(op, "has_saved_activation", False)) and op.label not in supported_labels:
            op._internal_set("out", None)
            op._internal_set("transformed_out", None)
            op._internal_set("has_saved_activation", False)


def _remove_fast_live_hooks(handles: list[Any]) -> None:
    """Remove and discard module-hook handles without retaining their session.

    Every handle gets its own removal attempt: ``weakref.finalize`` pops its
    registry entry BEFORE invoking the callback, so this is the one chance to
    remove these hooks -- a single raising ``remove()`` must never strand the
    remaining handles on the user's modules forever. The first failure
    re-raises only after every handle was attempted and the list cleared.
    """

    first_error: BaseException | None = None
    for handle in handles:
        try:
            handle.remove()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    handles.clear()
    if first_error is not None:
        raise first_error


class _FastSparseSession:
    """Verify-once loaded sparse executor with staged state and compiled binders."""

    def __init__(
        self,
        source: Any,
        target: Any,
        descriptor: SparseRunDescriptor,
        readiness: ReadinessReport,
        prepared_state: PreparedRunnableState,
        compiled_calls: tuple[_CompiledSparseCall, ...],
        *,
        seed: int | None,
    ) -> None:
        """Initialize one trusted sparse loop session."""

        self.source = source
        self.target = target
        self.descriptor = descriptor
        self.readiness = readiness
        self.prepared_state = prepared_state
        self.compiled_calls = compiled_calls
        self.seed = seed
        self.positions = _model_input_arity_positions(descriptor)
        self.input_slots = tuple(
            slot for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.MODEL_INPUT
        )
        self.slots_by_id = {slot.slot_id: slot for slot in descriptor.tensor_slots}
        self.state_slot_ids = frozenset(
            slot.slot_id
            for slot in descriptor.tensor_slots
            if slot.role in {TensorSlotRole.PARAMETER, TensorSlotRole.BUFFER}
        )
        # Key the refresh gate by EVERY lookup spelling of a saved entry:
        # descriptor calls carry pass-qualified labels ('linear_1_1:1') while
        # ``layer_label`` is the bare final label, so a bare-label-only set
        # made the in-loop ``save_activation`` refresh dead code -- every fast
        # iteration returned iteration-1 payloads on a verified trace.
        self.saved_labels = frozenset(
            key
            for key, entry in source.layer_dict_all_keys.items()
            if bool(getattr(entry, "has_saved_activation", False))
        )
        aliases: dict[str, list[str]] = {}
        for slot in descriptor.tensor_slots:
            if slot.version_of is not None and slot.version_of == slot.producer_slot_id:
                aliases.setdefault(slot.version_of, []).append(slot.slot_id)
        self.version_alias_ids = {key: tuple(value) for key, value in aliases.items()}
        self.disabled_autocast_devices = frozenset(
            entry.device_type
            for call in descriptor.calls
            for entry in call.execution_context.autocast
            if not entry.enabled
        )
        self.reseed_torch = _descriptor_has_seeded_rng(descriptor)
        self.reseed_host = descriptor.rng_profile.host_rng_consumed
        self.seeded_devices = _seeded_fork_devices(descriptor, seed)
        # Frozen-descriptor faithfulness ceilings, computed once per session; the
        # per-INPUT dynamic ceilings (escape staleness, derived layout, unbound
        # state, alias topology) are re-derived every iteration in ``run`` and
        # settled through ``_path_faithfulness`` -- never a hardcoded verdict.
        self.escape_witness_slot_ids = _tensor_derived_scalar_witness_slot_ids(descriptor)
        self.mode_sensitive_op_unwitnessed = _mode_sensitive_op_unwitnessed(descriptor)
        value_source_taint = _nondeterministic_value_sources(descriptor)
        self.nondeterministic_control_source = _uninit_taint_reaches(
            value_source_taint, _control_witness_source_slot_ids(descriptor)
        )
        self.declared_nondeterministic_sources = _declared_nondeterministic_sources(
            descriptor, value_source_taint
        )

    @classmethod
    def build(
        cls,
        source: Any,
        target: Any,
        *,
        seed: int | None,
    ) -> _FastSparseSession:
        """Build a trusted session after a normal run returned verified evidence."""

        descriptor, readiness, callables = _require_loaded_sparse_provider(source)
        state_ids = {
            slot.slot_id
            for slot in descriptor.tensor_slots
            if slot.role in {TensorSlotRole.PARAMETER, TensorSlotRole.BUFFER}
        }
        mutating_state_calls = _state_mutating_call_ids(descriptor, state_ids)
        if mutating_state_calls:
            raise RunCapabilityUnavailableError(
                "fast=True cannot cache state for a sparse recipe with calls that may update "
                "or mutate declared state "
                f"({', '.join(mutating_state_calls)}). Use ordinary run().",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="fast_state_static_guard",
            )
        prepared = prepare_runnable_state(source, seed=seed)
        compiled = tuple(
            _compile_sparse_call(
                call,
                callables[call.call_id],
                ambient_grad_enabled=descriptor.ambient_context.grad_enabled,
                ambient_inference_mode=descriptor.ambient_context.inference_mode,
            )
            for call in descriptor.calls
        )
        return cls(
            source,
            target,
            descriptor,
            readiness,
            prepared,
            compiled,
            seed=seed,
        )

    def _bind_inputs(
        self, inputs: Any, ceiling: RunResourceCeiling
    ) -> tuple[dict[str, torch.Tensor], tuple[ContractCheck, ...], bool]:
        """Validate raw runtime inputs and bind independent defensive mirrors.

        Returns the bound mirrors, the ordered contract checks, and the alias
        engine's ``unresolved`` ceiling flag -- ``True`` when the three-valued
        alias engine could prove neither overlap nor disjointness for a
        same-storage input pair. The caller MUST thread that flag into
        ``_path_faithfulness`` (r35 decision D: unknown is never VERIFIED).
        """

        values: dict[str, torch.Tensor] = {}
        checks: list[ContractCheck] = []
        checks.extend(
            _top_level_input_site_contract_checks(self.descriptor, inputs, self.positions)
        )
        raw_values: dict[str, torch.Tensor] = {}
        for slot in self.input_slots:
            binding = slot.input_binding
            try:
                if binding is None:
                    raise KeyError(slot.slot_id)
                root = _input_site_value(inputs, binding.model_site_position, self.positions)
                value = _value_at_path(root, binding.container_path)
            except (KeyError, IndexError, TypeError, AttributeError):
                value = None
            checks.extend(_fast_input_check(slot, value))
            if isinstance(value, torch.Tensor):
                raw_values[slot.slot_id] = value
        checks.extend(_input_tree_contract_checks(self.descriptor, inputs))
        checks.extend(_input_literal_contract_checks(self.descriptor, inputs, self.positions))
        checks.extend(_input_metadata_contract_checks(self.descriptor, inputs, self.positions))
        checks.extend(
            _input_nontensor_tree_contract_checks(self.descriptor, inputs, self.positions)
        )
        alias_checks, alias_unresolved = _input_alias_topology_checks(
            self.descriptor, self.input_slots, raw_values
        )
        checks.extend(alias_checks)
        if all(check.passed for check in checks):
            for slot in self.input_slots:
                raw = raw_values.get(slot.slot_id)
                if isinstance(raw, torch.Tensor):
                    values[slot.slot_id] = _runtime_mirror_clone(raw, ceiling, slot)
        return values, tuple(checks), alias_unresolved

    def _bind_outputs(
        self,
        compiled: _CompiledSparseCall,
        output: Any,
        slot_values: dict[str, torch.Tensor],
        *,
        ceiling: RunResourceCeiling,
        witness_source_snapshots: dict[str, torch.Tensor],
    ) -> tuple[ContractCheck, ...]:
        """Bind produced tensors and enforce the per-call static guard."""

        call = compiled.descriptor
        if len(call.output_slot_ids) != len(call.op_labels):
            # The slow path pins this arity inside its structure check; the
            # fast zip below is shortest-wins, so an under-counted op_labels
            # tuple silently skipped every check on the surplus slots.
            return (
                _contract_check(
                    f"fast_output_structure:{call.call_id}",
                    False,
                    RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
                    f"Fast call {call.call_id!r} op labels disagree with its output slots.",
                    affected_op_labels=call.op_labels,
                    details=(
                        ("output_slot_ids", repr(call.output_slot_ids)),
                        ("op_labels", repr(call.op_labels)),
                    ),
                ),
            )
        expected_paths = tuple(
            self.slots_by_id[slot_id].output_path or () for slot_id in call.output_slot_ids
        )
        actual_paths: tuple[tuple[str | int, ...], ...]
        if len(expected_paths) == 1 and not expected_paths[0] and isinstance(output, torch.Tensor):
            actual_paths = ((),)
        else:
            actual_paths = _tensor_leaf_paths(output)
        if tuple(actual_paths) != expected_paths:
            return (
                _contract_check(
                    f"fast_output_structure:{call.call_id}",
                    False,
                    RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
                    f"Fast call {call.call_id!r} output paths changed from the verified run.",
                    affected_op_labels=call.op_labels,
                    details=(("expected", repr(expected_paths)), ("actual", repr(actual_paths))),
                ),
            )
        # grind-r5 b7 R23-A: op_labels is an independent persisted field, so a
        # descriptor with fewer labels would truncate this zip and skip the
        # shape/dtype/device guards for the trailing slots while reporting
        # clean. The slow-path twin enforces exactly this parity inside its
        # output_structure contract check; mirror it here.
        if len(call.output_slot_ids) != len(call.op_labels):
            return (
                _contract_check(
                    f"fast_output_slot_label_parity:{call.call_id}",
                    False,
                    RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
                    f"Fast call {call.call_id!r} records {len(call.output_slot_ids)} output "
                    f"slots but {len(call.op_labels)} op labels; the descriptor may be "
                    "tampered.",
                    affected_op_labels=call.op_labels,
                ),
            )
        out_slot = _out_argument_slot_id(call) if call.is_inplace else None
        for slot_id, op_label, path in zip(
            call.output_slot_ids, call.op_labels, expected_paths, strict=True
        ):
            value = (
                output
                if not path and isinstance(output, torch.Tensor)
                else _value_at_path(output, path)
            )
            slot = self.slots_by_id[slot_id]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != slot.shape:
                return (
                    _contract_check(
                        f"fast_output_shape:{slot_id}",
                        False,
                        RunnableErrorCode.OUTPUT_SHAPE_MISMATCH,
                        f"Fast output {slot_id!r} shape changed from {slot.shape}.",
                        affected_op_labels=(op_label,),
                    ),
                )
            if str(value.dtype) != slot.dtype:
                return (
                    _contract_check(
                        f"fast_output_dtype:{slot_id}",
                        False,
                        RunnableErrorCode.OUTPUT_DTYPE_MISMATCH,
                        f"Fast output {slot_id!r} dtype changed from {slot.dtype}.",
                        affected_op_labels=(op_label,),
                    ),
                )
            if value.device.type != slot.device_type or (
                slot.device_index is not None and value.device.index != slot.device_index
            ):
                return (
                    _contract_check(
                        f"fast_output_device:{slot_id}",
                        False,
                        RunnableErrorCode.OUTPUT_DTYPE_MISMATCH,
                        f"Fast output {slot_id!r} device changed from "
                        f"{slot.device_type}:{slot.device_index}.",
                        affected_op_labels=(op_label,),
                    ),
                )
            slot_values[slot_id] = value
            produced_slot_ids = {slot_id}
            if out_slot is not None:
                slot_values[out_slot] = value
                produced_slot_ids.add(out_slot)
            for alias_id in self.version_alias_ids.get(slot_id, ()):
                slot_values[alias_id] = value
                produced_slot_ids.add(alias_id)
            # Snapshot every escape-witness source slot at its production point so a
            # later in-place mutation of the live tensor cannot restale the digest
            # comparison -- the run-digest then matches the pre-mutation save-digest
            # (same rule as the ordinary transaction's ``_bind_call_outputs``).
            for produced_slot_id in produced_slot_ids & self.escape_witness_slot_ids:
                witness_source_snapshots[produced_slot_id] = ceiling.guarded_clone(
                    value,
                    call_id=call.call_id,
                    slot_id=produced_slot_id,
                    affected_op_labels=call.op_labels,
                )
            if op_label in self.saved_labels:
                op = self.target.layer_dict_all_keys.get(op_label)
                if op is not None:
                    op.save_activation(
                        value,
                        (),
                        {},
                        False,
                        activation_transform=getattr(self.target, "activation_transform", None),
                    )
        if call.control_obligations:
            return _call_witness_checks(self.descriptor, call, slot_values)
        return ()

    def _reconstruct_output(
        self,
        slot_values: dict[str, torch.Tensor],
        call_outputs: Mapping[str, Any],
        *,
        ceiling: RunResourceCeiling,
    ) -> Any:
        """Rebuild the model output without cloning every intermediate into a fork."""

        output_slots = tuple(
            slot for slot in self.descriptor.tensor_slots if slot.role is TensorSlotRole.OUTPUT
        )
        values: list[tuple[tuple[str | int, ...], torch.Tensor]] = []
        for slot in output_slots:
            source_id = slot.producer_slot_id or slot.version_of
            if source_id is None or source_id not in slot_values:
                raise RuntimeSignatureDriftError(
                    f"Fast output slot {slot.slot_id!r} has no produced source.",
                    code=RunnableErrorCode.SLOT_PRODUCTION_MISMATCH.value,
                    slot_id=slot.slot_id,
                )
            value = slot_values[source_id]
            slot_values[slot.slot_id] = value
            op = next(
                (
                    self.target.layer_dict_all_keys.get(label)
                    for label in self.target.output_layers
                    if self.target.layer_dict_all_keys.get(label) is not None
                    and getattr(self.target.layer_dict_all_keys[label], "container_path", ())
                    == (slot.output_path or ())
                ),
                None,
            )
            if op is not None:
                # Store a guarded CLONE, never the returned object itself: the
                # ordinary provider clones at this seam, and an aliased slot
                # lets caller in-place mutation of ``RunResult.output``
                # silently rewrite the trace's "verified" output payload.
                op._internal_set(
                    "out",
                    ceiling.guarded_clone(
                        value,
                        call_id=None,
                        slot_id=slot.slot_id,
                        affected_op_labels=(),
                    ),
                )
            values.append((slot.output_path or (), value))
        spec = next(
            (
                getattr(self.target.layer_dict_all_keys[label], "container_spec", None)
                for label in self.target.output_layers
                if isinstance(
                    getattr(self.target.layer_dict_all_keys[label], "container_spec", None),
                    ContainerSpec,
                )
            ),
            None,
        )
        if isinstance(spec, ContainerSpec):
            last_output = call_outputs.get(self.descriptor.calls[-1].call_id)
            if spec.type_module == "torch.return_types" and last_output is not None:
                return last_output
            return rebuild_container_from_spec(spec, [value for _, value in values])
        if len(values) == 1 and not values[0][0]:
            return values[0][1]
        if values:
            raise RuntimeSignatureDriftError(
                "Fast sparse output has multiple leaves without a container contract.",
                code=RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT.value,
            )
        return None

    def run(self, inputs: Any, *, seed: int | None) -> RunResult:
        """Execute one guarded trusted iteration using cached state and binders."""

        if seed != self.seed:
            raise RunCapabilityUnavailableError(
                "A fast sparse session pins the seed used by its verify-once run; "
                "start a new loaded Trace to change seeds.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="fast_seed_guard",
            )
        ceiling = RunResourceCeiling(self.descriptor)
        input_values, input_checks, input_alias_unresolved = self._bind_inputs(inputs, ceiling)
        _raise_first_divergence(input_checks, DivergencePolicy.RAISE, fork=None)
        slot_values = dict(self.prepared_state.slot_values)
        slot_values.update(input_values)
        # Input/state escape-witness sources are snapshotted at bind (their
        # production point); produced sources are snapshotted in ``_bind_outputs``.
        witness_source_snapshots: dict[str, torch.Tensor] = {}
        for witness_slot_id in self.escape_witness_slot_ids:
            bound = slot_values.get(witness_slot_id)
            if isinstance(bound, torch.Tensor):
                witness_source_snapshots[witness_slot_id] = ceiling.guarded_clone(
                    bound,
                    call_id=None,
                    slot_id=witness_slot_id,
                    affected_op_labels=(),
                )
        call_outputs: dict[str, Any] = {}
        checks: list[ContractCheck] = list(input_checks)
        from .utils._torch_compat import autocast_is_enabled

        enabled_autocast = tuple(
            device for device in self.disabled_autocast_devices if bool(autocast_is_enabled(device))
        )
        if enabled_autocast:
            raise RunCapabilityUnavailableError(
                "fast=True requires the caller's autocast state to match the verified disabled "
                f"context for {enabled_autocast!r}; use ordinary run() for context conversion.",
                code=RunnableErrorCode.EXECUTION_CONTEXT_UNAVAILABLE.value,
                detection_stage="fast_execution_context_guard",
            )
        reseed = seed is not None and (self.reseed_torch or self.reseed_host)
        rng_context = (
            torch.random.fork_rng(devices=self.seeded_devices) if reseed else nullcontext()
        )
        host_rng_saved = snapshot_host_rng() if seed is not None and self.reseed_host else None
        try:
            with (
                _ambient_execution_context_restored(self.descriptor.ambient_context),
                rng_context,
                _state.pause_logging(),
            ):
                if reseed:
                    _seed_run_generators(
                        cast(int, seed), self.seeded_devices, reseed_host=self.reseed_host
                    )
                for compiled in self.compiled_calls:
                    output = compiled.execute(slot_values)
                    call_outputs[compiled.descriptor.call_id] = output
                    call_checks = self._bind_outputs(
                        compiled,
                        output,
                        slot_values,
                        ceiling=ceiling,
                        witness_source_snapshots=witness_source_snapshots,
                    )
                    checks.extend(call_checks)
                    failed = next((check for check in call_checks if not check.passed), None)
                    if failed is not None:
                        # Rollback is impossible on the REUSED fast target:
                        # earlier calls in THIS iteration already refreshed
                        # their saved activations, so a mid-loop divergence
                        # leaves mixed-iteration payloads behind. Poison the
                        # target monotonically before raising (mirror of the
                        # fast-LIVE twin's _poison_and_raise) so downstream
                        # faithful consumers refuse it.
                        mark_trace_path_status(
                            self.target, PathFaithfulness.DIVERGED, failed.diagnostic
                        )
                        _raise_failed_contract_as_divergence(failed, fork=None)
        finally:
            if host_rng_saved is not None:
                restore_host_rng(host_rng_saved)
        output = self._reconstruct_output(slot_values, call_outputs, ceiling=ceiling)
        # The ordinary transaction's per-run contract-check set, verbatim: the
        # input-structure (r67 C2), container, and conditional-arm-entry
        # witness families have their ONLY runtime consumer here, so skipping
        # it left an exact-class-swapped container input replaying the
        # recorded path with a wrong value stamped verified.
        post_checks = _post_execution_contract_checks(
            self.descriptor,
            inputs=inputs,
            output=output,
            slot_values=slot_values,
            fork=self.target,
        )
        checks.extend(post_checks)
        failed_post = next((check for check in post_checks if not check.passed), None)
        if failed_post is not None:
            # This iteration's saved activations are already refreshed on the
            # reused target; poison before raising, as in the mid-loop arm.
            mark_trace_path_status(self.target, PathFaithfulness.DIVERGED, failed_post.diagnostic)
            _raise_failed_contract_as_divergence(failed_post, fork=None)
        checks.append(
            _contract_check(
                "fast_static_guard",
                True,
                RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
                "Fast static guard passed.",
            )
        )
        # Settle through the ONE faithfulness derivation every provider uses. The
        # verify-once gate proved only the FIRST input; each iteration re-derives
        # the per-input dynamic ceilings (tensor->host escape staleness, unbound
        # state escape, input-derived layout, alias-topology unknown) exactly like
        # the ordinary transaction, so a changed input that restales a baked
        # literal or layout predicate settles UNVERIFIABLE, never a false VERIFIED.
        output_container_spec = _output_container_spec(self.target)
        provisional_verdict, provisional_mismatch = _path_faithfulness(
            self.descriptor,
            checks,
            host_rng_unreproduced=_host_rng_unreproduced(self.descriptor, seed),
            tensor_derived_scalar_stale=_tensor_derived_scalar_stale(
                self.descriptor, slot_values, witness_source_snapshots
            ),
            unbound_state_escape_stale=_unbound_state_escape_stale(self.descriptor, slot_values),
            container_reconstruction_lossy=_container_spec_reconstruction_lossy(
                output_container_spec
            ),
            output_not_reproduced=_output_not_reproduced(self.descriptor, output_container_spec),
            mode_sensitive_op_unwitnessed=self.mode_sensitive_op_unwitnessed,
            input_alias_unresolved=input_alias_unresolved,
            nondeterministic_control_source=self.nondeterministic_control_source,
            input_derived_layout_stale=_input_derived_layout_stale(self.descriptor, inputs),
        )
        return _finalize_provider_run(
            fork=self.target,
            output=output,
            readiness=self.readiness,
            state_source=self.prepared_state.state_source,
            initializer_policy_version=self.prepared_state.initializer_policy_version,
            seed=self.prepared_state.seed,
            random_filled_slot_ids=self.prepared_state.random_filled_slot_ids,
            contract_checks=tuple(checks),
            provisional_path_faithfulness=provisional_verdict,
            provisional_mismatch=provisional_mismatch,
            numeric_attestation=NumericAttestationStatus.NOT_APPLICABLE,
            divergence_policy=DivergencePolicy.RAISE,
            nondeterministic_sources=self.declared_nondeterministic_sources,
        )


class _FastLiveSession:
    """Native-forward collector guarded by selected-site and boundary call plans."""

    def __init__(self, trace: Any, model: nn.Module) -> None:
        """Build and install the targeted live-model collection plan."""

        self.trace = trace
        self.model_ref = weakref.ref(model)
        self.owner_thread_id: int | None = None
        self.active = False
        self.module_index = 0
        self.function_index = 0
        self.last_function_output: tuple[str, int] | None = None
        self.failure: ContractCheck | None = None
        self.module_plans = build_module_plans(trace)
        self.function_plans = build_function_plans(
            trace,
            frozenset(label for plan in self.module_plans for label in plan.save_labels),
        )
        self.planned_module_calls, self.expected_module_calls = _module_call_budget(
            trace, self.module_plans
        )
        self.module_call_counts: dict[str, int] = {}
        self.stop_at: tuple[str, int] | None = None
        self.cleared_by_until: tuple[str, ...] = ()
        supported_labels = _admitted_collection_labels(
            trace, (*self.module_plans, *self.function_plans)
        )
        modules = dict(model.named_modules())
        plan_addresses = tuple(dict.fromkeys(plan.address_or_name for plan in self.module_plans))
        # Every typed refusal must fire BEFORE the unsupported-activation wipe
        # below: a refused session must never destroy the user's saved payloads.
        for address in plan_addresses:
            if modules.get("" if address == "self" else address) is None:
                raise RunCapabilityUnavailableError(
                    f"Captured module address {address!r} is absent from the live model.",
                    code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                    detection_stage="fast_live_module_plan",
                )
        # Steered reruns: the staged spec lowers to module-boundary hooks that
        # must run BEFORE the collection hooks below (torch fires forward hooks
        # in registration order), so the collected site value is the
        # post-intervention value, exactly as capture saves it. The plan
        # refuses typed (fast_rerun_target_unsupported) before any hook or wipe.
        self.steer_plan = SteerPlan(trace, model)
        _drop_uncollected_payloads(trace, supported_labels)
        self.function_names = frozenset(plan.address_or_name for plan in self.function_plans)
        # The ordered call fingerprint sealed at capture time is the structure
        # guard that admits a different-size input; ``None`` on a trace that
        # never captured keeps the exact-size guard.
        self.fingerprint_reference = getattr(trace, "_raw_call_fingerprint", None)
        self.shape_varied = False
        self.shape_metadata_cleared = False
        self.allow_size_change = True
        self.poison_on_divergence = True
        # The explicit fast door reports a run verdict, so a hooked run settles
        # unverifiable there (see ``_finalize_run``).
        self.reports_hooked_ceiling = True
        self.output_transform: Any = None
        self.last_fingerprint: tuple[int, int] | None = None
        self.refreshed_labels: frozenset[str] = frozenset(supported_labels)
        self.handles: list[Any] = []
        self._hook_finalizer = weakref.finalize(self, _remove_fast_live_hooks, self.handles)
        session_ref = weakref.ref(self)
        try:
            self.handles.extend(install_steer_hooks(self.steer_plan, session_ref))
            self.handles.extend(install_module_token_hooks(model, session_is_active(session_ref)))
            for address in plan_addresses:
                module = modules["" if address == "self" else address]

                def hook(
                    _module: nn.Module,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    module_address: str = address,
                    ref: weakref.ReferenceType[_FastLiveSession] = session_ref,
                ) -> None:
                    """Forward one module boundary to the active session."""

                    session = ref()
                    if session is not None:
                        session.capture_module(module_address, output)

                self.handles.append(module.register_forward_hook(hook))
        except BaseException:
            self._hook_finalizer()
            raise

    def close(self) -> None:
        """Remove persistent module hooks owned by this session."""

        self._hook_finalizer()

    def _admits_size_change(self, expected_shape: tuple[int, ...], value: torch.Tensor) -> bool:
        """Return whether a same-rank size difference is admitted on this run.

        Size changes are admitted only when the capture sealed a call
        fingerprint (the structure guard that replaces size equality) and the
        rank is unchanged; dtype is checked by the caller as before.
        """

        return (
            self.allow_size_change
            and self.fingerprint_reference is not None
            and value.ndim == len(expected_shape)
        )

    def _poison_and_raise(self, failed: ContractCheck) -> None:
        """Poison the half-refreshed user Trace, then raise the typed divergence.

        Boundary payloads are overwritten in place as the native forward passes
        each site, so a divergence detected at site N leaves sites 1..N-1 holding
        new-input activations while later sites keep capture-time ones. The
        mixed-activation Trace must never pass downstream faithful consumers
        (validation, export, faithful comparison, chaining), so it is
        monotonically poisoned before the raise -- the documented "always raises
        on divergence" posture plus an honest mark on the user-owned object.
        """

        if self.poison_on_divergence:
            mark_trace_path_status(self.trace, PathFaithfulness.DIVERGED, failed.diagnostic)
        _raise_failed_contract_as_divergence(failed, fork=None)

    def wants_function(self, func_name: str) -> bool:
        """Return whether the active scoped collector needs this function type."""

        return (
            self.active
            and threading.get_ident() == self.owner_thread_id
            and func_name in self.function_names
        )

    def _capture_plan_output(self, plan: _FastOutputPlan, output: Any) -> None:
        """Check one runtime output tree and save only selected captured labels."""

        values, reason = _pair_output_leaves(output, plan.paths, len(plan.op_labels))
        if values is None:
            self.failure = _contract_check(
                f"fast_live_output_count:{plan.address_or_name}",
                False,
                RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
                f"Fast live site {plan.address_or_name!r} output structure changed from the "
                f"capture: {reason}.",
                affected_op_labels=plan.op_labels,
            )
            return
        for label, expected_shape, expected_dtype, value in zip(
            plan.op_labels, plan.shapes, plan.dtypes, values
        ):
            if not isinstance(value, torch.Tensor):
                self.failure = _contract_check(
                    f"fast_live_output_type:{label}",
                    False,
                    RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
                    f"Fast live site {label!r} no longer produces a tensor.",
                    affected_op_labels=(label,),
                )
                return
            if expected_shape is not None and tuple(value.shape) != expected_shape:
                if not self._admits_size_change(expected_shape, value):
                    self.failure = _contract_check(
                        f"fast_live_output_shape:{label}",
                        False,
                        RunnableErrorCode.OUTPUT_SHAPE_MISMATCH,
                        f"Fast live site {label!r} shape changed from {expected_shape} to "
                        f"{tuple(value.shape)}.",
                        affected_op_labels=(label,),
                    )
                    return
                self.shape_varied = True
            if expected_dtype is not None and str(value.dtype) != expected_dtype:
                self.failure = _contract_check(
                    f"fast_live_output_dtype:{label}",
                    False,
                    RunnableErrorCode.OUTPUT_DTYPE_MISMATCH,
                    f"Fast live site {label!r} dtype changed from {expected_dtype} to "
                    f"{value.dtype}.",
                    affected_op_labels=(label,),
                )
                return
            if label in plan.save_labels:
                op = self.trace.layer_dict_all_keys[label]
                op.save_activation(
                    value,
                    (),
                    {},
                    False,
                    activation_transform=getattr(self.trace, "activation_transform", None),
                )

    def capture_module(self, address: str, output: Any) -> None:
        """Collect and guard one atomic-module forward-hook event."""

        if not self.active or threading.get_ident() != self.owner_thread_id or self.failure:
            return
        call_index = self.module_call_counts.get(address, 0) + 1
        self.module_call_counts[address] = call_index
        if call_index > self.expected_module_calls.get(address, call_index):
            self.failure = _contract_check(
                "fast_live_module_extra_call",
                False,
                RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
                f"Live model called module {address!r} {call_index} times; the capture "
                f"recorded {self.expected_module_calls[address]}.",
            )
            return
        if call_index not in self.planned_module_calls.get(address, frozenset()):
            # An unsaved call of a module whose other calls are saved: nothing
            # to collect; the per-address call count and the call fingerprint
            # still guard the structure.
            return
        if self.module_index >= len(self.module_plans):
            self.failure = _contract_check(
                "fast_live_module_extra",
                False,
                RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
                f"Live model executed unexpected atomic module {address!r}.",
            )
            return
        plan = self.module_plans[self.module_index]
        if plan.address_or_name != address or int(plan.call_index or 1) != call_index:
            self.failure = _contract_check(
                "fast_live_module_path",
                False,
                RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
                "Live module path changed: expected "
                f"{plan.address_or_name!r} call {plan.call_index or 1}, got {address!r} "
                f"call {call_index}.",
                affected_op_labels=plan.op_labels,
            )
            return
        self.module_index += 1
        self._capture_plan_output(plan, output)
        if self.stop_at == ("module", self.module_index - 1) and self.failure is None:
            raise FastUntilReached

    def capture_function(self, func_name: str, output: Any) -> None:
        """Collect and guard one explicitly requested functional-op event."""

        if not self.wants_function(func_name) or self.failure:
            return
        output_identity = (func_name, id(output))
        if output_identity == self.last_function_output:
            return
        self.last_function_output = output_identity
        if self.function_index >= len(self.function_plans):
            self.failure = _contract_check(
                "fast_live_function_extra",
                False,
                RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
                f"Live model executed an extra requested functional op {func_name!r}.",
            )
            return
        plan = self.function_plans[self.function_index]
        if plan.address_or_name != func_name:
            self.failure = _contract_check(
                "fast_live_function_path",
                False,
                RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
                f"Live functional path changed: expected {plan.address_or_name!r}, "
                f"got {func_name!r}.",
                affected_op_labels=plan.op_labels,
            )
            return
        self.function_index += 1
        self._capture_plan_output(plan, output)
        if self.stop_at == ("function", self.function_index - 1) and self.failure is None:
            raise FastUntilReached

    @contextmanager
    def activated(self) -> Any:
        """Activate this collector for one owner-thread native forward."""

        prior = _state._active_fast_run_collector
        if prior is not None:
            raise RuntimeError("Fast live runs cannot be nested or concurrent.")
        self.owner_thread_id = threading.get_ident()
        self.module_index = 0
        self.module_call_counts = {}
        self.function_index = 0
        self.last_function_output = None
        self.failure = None
        self.shape_varied = False
        self.steer_plan.reset()
        self.active = True
        _state._active_fast_run_collector = self
        try:
            yield
        finally:
            _state._active_fast_run_collector = prior
            self.active = False
            self.owner_thread_id = None

    def _admit_inputs(self, inputs: Any) -> tuple[Any, Mapping[str, Any], bool]:
        """Split the runtime inputs and admit them against the captured input boundary.

        Returns
        -------
        tuple[Any, Mapping[str, Any], bool]
            Positional inputs, keyword inputs, and whether a sealed call fingerprint
            re-admitted a changed input size.
        """

        input_args = inputs
        input_kwargs: Mapping[str, Any] = {}
        if (
            isinstance(inputs, Mapping)
            and {"args", "kwargs"}.issubset(inputs)
            and set(inputs).issubset({"args", "kwargs"})
        ):
            args, kwargs = _split_mixed_inputs(inputs)
            input_args = list(args)
            input_kwargs = dict(kwargs)
        failed_input = _first_failed_live_input_check(self.trace, input_args, input_kwargs)
        input_size_changed = False
        if (
            isinstance(failed_input, ContractCheck)
            and failed_input.name.startswith("input_shape:")
            and self.allow_size_change
            and self.fingerprint_reference is not None
        ):
            # Size equality is not the guard when the capture sealed a call
            # fingerprint: re-admit on rank, dtype, device and arity, and let
            # the fingerprint settle the structure after the forward.
            failed_input, input_size_changed = fast_live_input_admission(
                self.trace, input_args, input_kwargs, allow_shape_change=True
            )
        if failed_input is _INPUT_CHECK_UNAVAILABLE:
            # This consultation has ADMISSION power (it runs BEFORE the
            # forward), so a broken guard must refuse, never read as
            # inputs-match and run the forward unguarded (R22-2).
            raise RunCapabilityUnavailableError(
                "The fast live run's input-contract guard could not classify the "
                "runtime inputs against the captured input boundary; refusing the "
                "guarded fast path rather than running unguarded.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                provider=RunProvider.LIVE,
            )
        if failed_input is not None:
            _raise_failed_contract_as_divergence(failed_input, fork=None)
        return input_args, input_kwargs, input_size_changed

    def run(self, inputs: Any, *, seed: int | None, until: Any = None) -> RunResult:
        """Execute one native forward and enforce the cached static graph guard.

        ``until`` names a saved module the forward stops after: the
        prefix keeps every per-call guard, the cut is disclosed as a
        ``RunTruncation``, and the call fingerprint (a whole-forward guard)
        cannot be compared, so the verdict is ``unverifiable``.
        """

        self.run_started_at = time.monotonic()
        model = self.model_ref()
        if model is None:
            raise RunCapabilityUnavailableError(
                "The fast live session's source model is no longer available.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                provider=RunProvider.LIVE,
            )
        input_args, input_kwargs, input_size_changed = self._admit_inputs(inputs)
        rearm_cleared_sites(self.trace, self.cleared_by_until)
        self.cleared_by_until = ()
        self.stop_at = (
            None
            if until is None
            else resolve_fast_until(self.trace, self.module_plans, self.function_plans, until)
        )
        self.steer_plan.refuse_changed_helpers()
        if seed is not None:
            set_random_seed(seed)
        truncation = None
        with self.activated(), self.steer_plan.context(), fingerprinting() as fingerprint:
            try:
                if isinstance(input_args, list):
                    output = model(*input_args, **dict(input_kwargs))
                else:
                    output = model(input_args, **dict(input_kwargs))
            except FastUntilReached:
                output = None
                truncation = until
        self.shape_varied = self.shape_varied or input_size_changed
        self.last_fingerprint = None if truncation is not None else fingerprint.value
        if self.failure is not None:
            self._poison_and_raise(self.failure)
        if truncation is not None:
            truncation = self._settle_truncated_prefix(until, input_args, input_kwargs)
        else:
            self._check_full_path()
            self._refresh_boundary_payloads(input_args, input_kwargs, output)
        if self.shape_varied and not self.shape_metadata_cleared:
            # Honesty rule: a native forward refreshes only the saved sites and
            # the boundary ops. Every other op's shape and size metadata would
            # otherwise read capture-time numbers as if they were this run's,
            # so they take the not-available spelling (None) once and for all.
            clear_unrefreshed_shape_metadata(self.trace, self.refreshed_labels)
            self.shape_metadata_cleared = True
        # A cut forward leaves downstream staged hooks unfired by design.
        unfired = self.steer_plan.warn_unfired() if truncation is None else ()
        self._record_fast_run(seed=seed, unfired=unfired, until=until)
        return self._finalize_run(output, seed=seed, truncation=truncation)

    def _check_full_path(self) -> None:
        """Enforce the whole-forward guards: call fingerprint and every planned call."""

        if self.fingerprint_reference is not None and self.last_fingerprint != tuple(
            self.fingerprint_reference
        ):
            # The ordered call fingerprint is the structure guard: the same
            # module-entry and torch-call sequence as the capture, independent
            # of tensor sizes. A mismatch is a different taken path (another
            # branch arm, a size-dependent loop count), refused typed.
            failed = _contract_check(
                "fast_live_call_fingerprint",
                False,
                RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
                "Live forward executed a different op structure than the capture: "
                f"call fingerprint {self.last_fingerprint} against recorded "
                f"{tuple(self.fingerprint_reference)}.",
            )
            self._poison_and_raise(failed)
        if self.module_index != len(self.module_plans) or any(
            self.module_call_counts.get(address, 0) != count
            for address, count in self.expected_module_calls.items()
        ):
            failed = _contract_check(
                "fast_live_module_missing",
                False,
                RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
                f"Live module path ended after {self.module_index} of "
                f"{len(self.module_plans)} captured atomic calls.",
            )
            self._poison_and_raise(failed)
        if self.function_index != len(self.function_plans):
            failed = _contract_check(
                "fast_live_function_missing",
                False,
                RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
                f"Live functional path ended after {self.function_index} of "
                f"{len(self.function_plans)} requested calls.",
            )
            self._poison_and_raise(failed)

    def _settle_truncated_prefix(
        self, until: Any, input_args: Any, input_kwargs: Mapping[str, Any]
    ) -> Any:
        """Check the executed prefix of a cut forward, then clear what did not run."""

        kind, index = cast(tuple[str, int], self.stop_at)
        stop_plan = (self.module_plans if kind == "module" else self.function_plans)[index]
        expected_functions = function_prefix_count(self.trace, stop_plan, self.function_plans)
        if self.function_index != expected_functions:
            self._poison_and_raise(
                _contract_check(
                    "fast_until_prefix",
                    False,
                    RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
                    f"Live forward reached until={until!r} after {self.function_index} of the "
                    f"{expected_functions} functional calls the capture ran before it.",
                )
            )
        self._refresh_boundary_payloads(input_args, input_kwargs, None, inputs_only=True)
        truncation, self.cleared_by_until = truncate_after(self.trace, stop_plan, until)
        return truncation

    def _finalize_run(self, output: Any, *, seed: int | None, truncation: Any) -> RunResult:
        """Settle one fast run's report through the shared provider finalizer."""

        readiness = ReadinessReport(
            status=ReadinessStatus.READY,
            provider=RunProvider.LIVE,
            backend=str(getattr(self.trace, "backend", "torch")),
            capability="guarded_static_native_forward",
            resolver_records=(),
            state_sources_available=(StateSource.LIVE_MODEL_STATE,),
            witness_completeness=None,
            diagnostics=(),
        )
        guard_check = _contract_check(
            "fast_static_guard",
            True,
            RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
            "Fast static guard passed.",
        )
        # Derive the verdict instead of asserting it: fast-live returns the model's
        # real native output, but the refreshed trace payloads carry the same honesty
        # obligation as the ordinary live provider, which ceilings a lossy output
        # container at UNVERIFIABLE. fast=True must never improve the verdict the
        # ordinary provider would settle on the same trace.
        lossy = _container_spec_reconstruction_lossy(_output_container_spec(self.trace))
        # A run that applied staged user edits is an edited forward: unverifiable and
        # not pristine until a hook-aware check exists, never VERIFIED.
        hooked = hooked_run_diagnostic(self.trace) if self.reports_hooked_ceiling else None
        provisional = (
            PathFaithfulness.UNVERIFIABLE
            if lossy or truncation is not None or hooked is not None
            else PathFaithfulness.VERIFIED
        )
        # Same declaration contract as the ordinary live provider (deephunt
        # F2): a host-RNG capture's fast report must not read as a
        # deterministic-looking empty tuple next to VERIFIED.
        runnable_seam = getattr(self.trace, "_runnable", None)
        nondeterministic_sources: tuple[str, ...] = (
            (_HOST_RNG_SOURCE_KIND,)
            if runnable_seam is not None and bool(runnable_seam.host_rng_consumed)
            else ()
        )
        return _finalize_provider_run(
            fork=self.trace,
            output=output,
            readiness=readiness,
            state_source=StateSource.LIVE_MODEL_STATE,
            initializer_policy_version=None,
            seed=seed,
            random_filled_slot_ids=(),
            contract_checks=(guard_check,),
            provisional_path_faithfulness=provisional,
            provisional_mismatch=hooked,
            numeric_attestation=NumericAttestationStatus.NOT_PRESENT,
            divergence_policy=DivergencePolicy.RAISE,
            nondeterministic_sources=nondeterministic_sources,
            # The fast-live "fork" IS the user's live Trace: an inherited
            # divergence must raise without evicting it from the registry.
            unregister_fork_on_divergence=False,
            truncation=truncation,
        )

    def _record_fast_run(
        self, *, seed: int | None, unfired: tuple[str, ...], until: Any = None
    ) -> None:
        """Stamp ``last_run``, the operation ledger and the state for one fast run."""

        from ._trace_state import TraceState

        trace = self.trace
        hook_count = len(self.steer_plan.hook_plan)
        record = {
            "op": "rerun",
            "engine": "guarded_fast",
            "started_at": self.run_started_at,
            "strict": True,
            "append": False,
            "hook_count": hook_count,
            "hook_fire_count": self.steer_plan.fire_count,
            "unfired_hook_count": len(unfired),
            "divergence_count": 0,
            "fast_refresh": True,
            "shape_varied": self.shape_varied,
            "call_fingerprint": self.last_fingerprint,
        }
        trace.last_run = {
            "engine": "guarded_fast",
            "timestamp": time.monotonic(),
            "started_at": self.run_started_at,
            "duration_s": time.monotonic() - self.run_started_at,
            "spec_revision": trace._spec_revision,
            "strict": True,
            "append": False,
            "seed": seed,
            "hooks": hook_count,
            "hooks_fired": self.steer_plan.fire_count,
            "hooks_unfired": len(unfired),
            "divergence_count": 0,
            "fast_refresh": True,
            "fast_refused": None,
            "shape_varied": self.shape_varied,
            "call_fingerprint": self.last_fingerprint,
            "refreshed_labels": tuple(sorted(self.refreshed_labels)),
            "until": until,
        }
        trace._record_operation(**record)
        trace.state = TraceState.RERUN_PROPAGATED

    def _match_native_output_leaves(self, output: Any, output_labels: tuple[str, ...]) -> list[Any]:
        """Pair each captured output op with its leaf in the native output, by container path.

        Output ops record their path in the model output (``container_path``);
        an opaque root (every path empty) keeps the positional pairing. A
        different leaf count, a missing path or an extra path is a structural
        divergence and refuses exactly as before.
        """

        captured_paths = [
            tuple(getattr(self.trace.layer_dict_all_keys[label], "container_path", None) or ())
            for label in output_labels
        ]
        values, reason = _pair_output_leaves(output, captured_paths, len(output_labels))
        if values is None:
            failed = _contract_check(
                "fast_live_model_output_structure",
                False,
                RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
                f"Native model output tensor structure changed from the captured boundary: {reason}.",
                affected_op_labels=output_labels,
            )
            self._poison_and_raise(failed)
        assert values is not None
        return values

    def _refresh_boundary_payloads(
        self,
        input_args: Any,
        input_kwargs: Mapping[str, Any],
        output: Any,
        *,
        inputs_only: bool = False,
    ) -> None:
        """Refresh saved model-input/output payloads from the native call boundaries."""

        input_leaves = _live_runtime_input_leaves(input_args, input_kwargs)
        if input_leaves is not None:
            input_labels = tuple(getattr(self.trace, "input_layers", ()))
            if len(input_leaves) != len(input_labels):
                # Guard-and-poison exactly like the output branch below:
                # a truncating zip here silently kept STALE capture-time
                # activations on the surplus input ops (R22-2 layer 2).
                failed = _contract_check(
                    "fast_live_model_input_structure",
                    False,
                    RunnableErrorCode.INPUT_TREE_MISMATCH,
                    f"Native model input tree carries {len(input_leaves)} tensor "
                    f"leaves; the captured input boundary recorded {len(input_labels)}.",
                    affected_op_labels=input_labels,
                )
                self._poison_and_raise(failed)
            # A folded input Layer is listed once per pass under one bare
            # label, which ``layer_dict_all_keys`` maps to the LAST pass; the
            # per-listing rule refreshes each pass with its own input.
            input_ops = boundary_label_rows(input_labels, self.trace.__getitem__)
            for op, value in zip(input_ops, input_leaves, strict=True):
                if op is not None and bool(getattr(op, "has_saved_activation", False)):
                    op.save_activation(value, (), {}, False)
        if inputs_only:
            return
        output_labels = tuple(getattr(self.trace, "output_layers", ()))
        output_values = self._match_native_output_leaves(output, output_labels)
        for label, value in zip(output_labels, output_values, strict=True):
            op = self.trace.layer_dict_all_keys[label]
            expected_shape = tuple(op.shape) if op.shape is not None else None
            expected_dtype = str(op.dtype) if op.dtype is not None else None
            if not isinstance(value, torch.Tensor):
                failed = _contract_check(
                    f"fast_live_model_output_type:{label}",
                    False,
                    RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
                    f"Native model output {label!r} is no longer a tensor.",
                    affected_op_labels=(label,),
                )
                self._poison_and_raise(failed)
            if expected_shape is not None and tuple(value.shape) != expected_shape:
                if not self._admits_size_change(expected_shape, value):
                    failed = _contract_check(
                        f"fast_live_model_output_shape:{label}",
                        False,
                        RunnableErrorCode.OUTPUT_SHAPE_MISMATCH,
                        f"Native model output {label!r} shape changed from {expected_shape} "
                        f"to {tuple(value.shape)}.",
                        affected_op_labels=(label,),
                    )
                    self._poison_and_raise(failed)
                self.shape_varied = True
            if expected_dtype is not None and str(value.dtype) != expected_dtype:
                failed = _contract_check(
                    f"fast_live_model_output_dtype:{label}",
                    False,
                    RunnableErrorCode.OUTPUT_DTYPE_MISMATCH,
                    f"Native model output {label!r} dtype changed from {expected_dtype} to "
                    f"{value.dtype}.",
                    affected_op_labels=(label,),
                )
                self._poison_and_raise(failed)
            if isinstance(value, torch.Tensor) and bool(getattr(op, "has_saved_activation", False)):
                op.save_activation(value, (), {}, False)
        # Same rule as capture (capture/trace.py): the transformed raw output
        # is stored when an output transform exists; the save scope is
        # otherwise untouched, so a sparse save stays sparse on a rerun.
        output_transform = self.output_transform
        self.trace.raw_output = output_transform(output) if output_transform is not None else None


def run_fast_loaded_trace(trace: Any, inputs: Any, *, seed: int | None) -> RunResult:
    """Run or initialize the explicit verify-once loaded sparse fast path.

    Parameters
    ----------
    trace:
        Loaded sparse source Trace.
    inputs:
        Runtime input tree.
    seed:
        Seed pinned for the lifetime of the fast session.

    Returns
    -------
    RunResult
        First ordinary verified result or a later compiled fast result.
    """

    session = trace.__dict__.get("_fast_run_session")
    if isinstance(session, _FastSparseSession):
        return session.run(inputs, seed=seed)
    result = run_loaded_sparse_trace(
        trace,
        inputs,
        seed=seed,
        on_divergence=DivergencePolicy.RAISE,
    )
    if result.report.path_faithfulness is not PathFaithfulness.VERIFIED:
        raise RunCapabilityUnavailableError(
            "fast=True requires a fully verified first sparse run; this artifact settled "
            f"{result.report.path_faithfulness.value}.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            detection_stage="fast_verify_once",
        )
    trace.__dict__["_fast_run_session"] = _FastSparseSession.build(
        trace,
        result.trace,
        seed=seed,
    )
    return result


def run_fast_live_trace(
    trace: Any, inputs: Any, *, seed: int | None, until: Any = None
) -> RunResult:
    """Run the explicit native-forward targeted-collection live fast path."""

    source_ref = getattr(trace, "_source_model_ref", None)
    model = source_ref() if source_ref is not None else None
    if not isinstance(model, nn.Module):
        raise RunCapabilityUnavailableError(
            "The live Trace no longer retains its source model: the trace "
            "holds it only weakly, so live-run availability depends on the "
            "caller keeping a strong reference (an inline-constructed model "
            "is collected at the first gc pass after capture). Keep the "
            "model alive, or save/load a runnable artifact instead.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            provider=RunProvider.LIVE,
        )
    session = trace.__dict__.get("_fast_run_session")
    if (
        not isinstance(session, _FastLiveSession)
        or session.model_ref() is not model
        or not session.steer_plan.follows(trace)
    ):
        if hasattr(session, "close"):
            session.close()
        session = _FastLiveSession(trace, model)
        trace.__dict__["_fast_run_session"] = session
    return session.run(inputs, seed=seed, until=until)


#: Typed refusals that send the legacy rerun door back to the capture engine.
_FAST_RERUN_REFUSALS: tuple[type[BaseException], ...] = (
    RunCapabilityUnavailableError,
    PathDivergenceError,
    RuntimeSignatureDriftError,
)


def try_guarded_fast_rerun(
    trace: Any, model: nn.Module, inputs: Any, *, output_transform: Any = None
) -> tuple[bool, str | None]:
    """Run the legacy intervened rerun through the guarded fast engine when eligible.

    Parameters
    ----------
    trace:
        Live trace carrying the staged spec; refreshed in place on success.
    model:
        The model the caller passed (or the trace's retained live model).
    inputs:
        Transformed forward input.
    output_transform:
        The resolved rerun output transform, applied to the native output for
        ``trace.raw_output`` exactly as the capture engine does.

    Returns
    -------
    tuple[bool, str | None]
        ``(True, None)`` when the fast engine ran; ``(False, code)`` with the
        typed refusal code when the caller must fall back to the capture
        engine. A refused session is closed so the fallback's rebuilt trace
        starts a fresh plan next time; the trace is never poisoned here
        because the fallback replaces its state.
    """

    # The legacy door leaves no TorchLens hook on the user's model between
    # calls (the rerun hook-staging contract), so the session is built for the
    # run and closed after it; the explicit fast=True door keeps its cached
    # session as before.
    close_fast_run_session(trace)
    session: _FastLiveSession | None = None
    try:
        session = _FastLiveSession(trace, model)
        session.poison_on_divergence = False
        # The legacy door returns the refreshed Trace, not a run verdict; the
        # hooked-run ceiling belongs to the explicit run(inputs=, fast=True) door.
        session.reports_hooked_ceiling = False
        session.output_transform = output_transform
        session.run(inputs, seed=None)
    except _FAST_RERUN_REFUSALS as exc:
        return False, refusal_code(exc)
    finally:
        if session is not None:
            session.close()
    return True, None


def close_fast_run_session(trace: Any) -> None:
    """Close and discard a Trace's internal fast-run session, if present.

    Close BEFORE discarding: a raising ``close()`` leaves the session attached
    (and therefore retryable) instead of popping it into an unreachable state
    with its hooks still installed.
    """

    session = trace.__dict__.get("_fast_run_session")
    if hasattr(session, "close"):
        session.close()
    trace.__dict__.pop("_fast_run_session", None)
