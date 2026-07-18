"""Sparse runnable descriptor production from a cooked :class:`Trace` projection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256
import json
import math
import platform
from typing import Any, cast

import numpy as np
import torch

from .. import __version__ as TORCHLENS_VERSION
from .._runnable_state import runnable_tensor_byte_digest
from ..utils._callable_safety import _STORAGE_UNSAFE_NAMES
from ..data_classes._state_adapter import state_items
from ..errors import RunnablePreflightError
from ..intervention.types import (
    CapturedArgTemplate,
    FunctionRegistryKey,
    LiteralTensor,
    LiteralValue,
    ParentRef,
    Unsupported,
)
from ..ir.container import DataclassField, DictKey, HFKey, NamedField, TupleIndex
from ..ir.container_registry import ModelSite, Role
from ..runnable import (
    RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION,
    RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
    RUNNABLE_CALL_RECIPE_VERSION,
    RUNNABLE_INITIALIZER_POLICY_VERSION,
    RUNNABLE_TLSPEC_SCHEMA_VERSION,
    ActivationPayloadLayerDescriptor,
    ActivationPayloadMember,
    AmbientExecutionContext,
    AutocastDeviceContext,
    CallExecutionContext,
    CallableRegistryEntry,
    InputAttestationFingerprint,
    ControlWitness,
    ControlWitnessKind,
    InputSlotBinding,
    LiteralArgumentRef,
    LiteralAtom,
    LiteralAtomKind,
    LiteralMapping,
    LiteralMappingEntry,
    LiteralSequence,
    LiteralSequenceKind,
    LiteralSlice,
    LiteralTorchSymbol,
    LiteralTupleKey,
    NonTensorLiteral,
    PayloadLayerDescriptor,
    PayloadLayersDescriptor,
    ProducerPreflight,
    RunnableCallDescriptor,
    RunnableCompatibility,
    RunnableDiagnostic,
    RunnableErrorCode,
    RunnableRngProfile,
    SparseRunDescriptor,
    SlotByteDigest,
    StateByteDigest,
    StateSlotBinding,
    StateSlotRole,
    TensorArgumentRef,
    TensorSlotDescriptor,
    TensorSlotRole,
    TensorUseSite,
    WitnessCompleteness,
)


@dataclass(slots=True)
class _SlotDraft:
    """Mutable producer-only accumulator for a frozen tensor-slot descriptor."""

    slot_id: str
    role: TensorSlotRole
    shape: tuple[int, ...]
    dtype: str
    device_type: str
    device_index: int | None
    mutable: bool = False
    version_of: str | None = None
    producer_slot_id: str | None = None
    output_path: tuple[str | int, ...] | None = None
    input_binding: InputSlotBinding | None = None
    state_binding: StateSlotBinding | None = None
    use_sites: list[TensorUseSite] | None = None

    def freeze(self) -> TensorSlotDescriptor:
        """Freeze this draft into the Stage-0 descriptor type.

        Returns
        -------
        TensorSlotDescriptor
            Immutable value-free slot descriptor.
        """

        return TensorSlotDescriptor(
            slot_id=self.slot_id,
            role=self.role,
            use_sites=tuple(self.use_sites or ()),
            shape=self.shape,
            dtype=self.dtype,
            rank=len(self.shape),
            device_type=self.device_type,
            device_index=self.device_index,
            mutable=self.mutable,
            version_of=self.version_of,
            producer_slot_id=self.producer_slot_id,
            output_path=self.output_path,
            input_binding=self.input_binding,
            state_binding=self.state_binding,
        )


class _UnsupportedLiteralError(ValueError):
    """Internal signal for a value outside the frozen literal grammar."""


def _call_execution_context(representative: Any) -> CallExecutionContext | None:
    """Build the required v2 per-call execution context from captured op state.

    Reads the op's ``func_autocast_state`` (recorded at the call's execution
    point), including the reserved ``__execution__`` grad/inference entry. A
    missing or shapeless record returns ``None`` so the producer fails closed
    with a typed diagnostic -- context is REQUIRED and EXPLICIT in v2, never
    defaulted.
    """

    state = getattr(representative, "func_autocast_state", None)
    if not isinstance(state, Mapping):
        return None
    execution = state.get("__execution__")
    if not isinstance(execution, Mapping) or "grad_enabled" not in execution:
        return None
    autocast_entries: list[AutocastDeviceContext] = []
    for device_type in sorted(key for key in state if not str(key).startswith("__")):
        entry = state[device_type]
        if not isinstance(entry, Mapping) or "enabled" not in entry:
            return None
        dtype = entry.get("dtype")
        autocast_entries.append(
            AutocastDeviceContext(
                device_type=str(device_type),
                enabled=bool(entry["enabled"]),
                dtype=None if dtype is None else str(dtype),
            )
        )
    return CallExecutionContext(
        autocast=tuple(autocast_entries),
        grad_enabled=bool(execution["grad_enabled"]),
        inference_mode=bool(execution.get("inference_mode", False)),
    )


# Torch ops documented CUDA-nondeterministic in their FORWARD kernels (the
# transpose-conv cuDNN atomicAdd scatter family plus the documented index/scatter
# accumulation set). Used ONLY for the fail-safe positive attestation-ineligibility
# marking (H_B_RESOLUTION R1): a capture running one of these on a CUDA device
# WITHOUT ``use_deterministic_algorithms(True)`` cannot promise byte-reproducible
# activations, so its descriptor is marked ineligible at capture (``not_applicable``
# at run, never a false ATTESTED and never a spurious NumericAttestationError).
_CUDA_NONDETERMINISTIC_QUALNAME_TAILS: frozenset[str] = frozenset(
    {
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "scatter_add",
        "scatter_add_",
        "scatter_reduce",
        "scatter_reduce_",
        "index_add",
        "index_add_",
        "index_copy",
        "index_copy_",
        "index_put",
        "index_put_",
        "put_",
        "bincount",
        "histc",
        "grid_sample",
        "grid_sampler_2d",
        "grid_sampler_3d",
        "embedding_bag",
        "median",
        "kthvalue",
        "ctc_loss",
    }
)


def _descriptor_has_cuda_nondeterministic_call(
    calls: Sequence[RunnableCallDescriptor],
    registry_entries: Sequence[CallableRegistryEntry],
    slot_drafts: Mapping[str, _SlotDraft],
) -> bool:
    """Return whether a documented CUDA-nondeterministic op runs on a CUDA device."""

    qualname_by_registry = {
        entry.registry_id: str(getattr(entry.key, "qualname", "") or "")
        for entry in registry_entries
    }
    for call in calls:
        tail = qualname_by_registry.get(call.registry_id, "").rsplit(".", 1)[-1]
        if tail not in _CUDA_NONDETERMINISTIC_QUALNAME_TAILS:
            continue
        involved_slot_ids = [argument.slot_id for argument in call.tensor_arguments]
        involved_slot_ids.extend(call.output_slot_ids)
        for slot_id in involved_slot_ids:
            draft = slot_drafts.get(slot_id)
            if draft is not None and draft.device_type == "cuda":
                return True
    return False


def _ambient_execution_context(
    trace: Any,
    calls: Sequence[RunnableCallDescriptor],
    registry_entries: Sequence[CallableRegistryEntry],
    slot_drafts: Mapping[str, _SlotDraft],
) -> AmbientExecutionContext | None:
    """Build the required v2 capture-scoped ambient context, or ``None`` if absent.

    ``attestation_ineligible_context`` is the POSITIVE capture-time marking for a
    nondeterministic execution context: ``cudnn.benchmark=True`` (autotuner
    kernel selection) or a documented CUDA-nondeterministic op running without
    ``use_deterministic_algorithms(True)`` (H_B_RESOLUTION R1). Fail-safe: the
    mark only widens ``not_applicable``, never a positive claim.
    """

    snapshot = getattr(trace, "_runnable_capture_ambient", None)
    if not isinstance(snapshot, Mapping) or "default_dtype" not in snapshot:
        return None

    def _optional_bool(name: str) -> bool | None:
        value = snapshot.get(name)
        return None if value is None else bool(value)

    def _optional_str(name: str) -> str | None:
        value = snapshot.get(name)
        return None if value is None else str(value)

    cudnn_benchmark = _optional_bool("cudnn_benchmark")
    deterministic = _optional_bool("deterministic_algorithms")
    ineligible = bool(cudnn_benchmark) or (
        deterministic is not True
        and _descriptor_has_cuda_nondeterministic_call(calls, registry_entries, slot_drafts)
    )
    return AmbientExecutionContext(
        default_dtype=str(snapshot["default_dtype"]),
        default_device=str(snapshot.get("default_device", "cpu")),
        float32_matmul_precision=_optional_str("float32_matmul_precision"),
        deterministic_algorithms=deterministic,
        deterministic_algorithms_warn_only=_optional_bool("deterministic_algorithms_warn_only"),
        cuda_matmul_allow_tf32=_optional_bool("cuda_matmul_allow_tf32"),
        cudnn_allow_tf32=_optional_bool("cudnn_allow_tf32"),
        cudnn_deterministic=_optional_bool("cudnn_deterministic"),
        cudnn_benchmark=cudnn_benchmark,
        cudnn_enabled=_optional_bool("cudnn_enabled"),
        flash_sdp_enabled=_optional_bool("flash_sdp_enabled"),
        mem_efficient_sdp_enabled=_optional_bool("mem_efficient_sdp_enabled"),
        math_sdp_enabled=_optional_bool("math_sdp_enabled"),
        attestation_ineligible_context=ineligible,
    )


def build_sparse_run_descriptor(trace: Any) -> SparseRunDescriptor:
    """Project a cooked Trace into the frozen sparse runnable descriptor.

    Parameters
    ----------
    trace:
        Fully postprocessed Trace produced from a sealed capture core.

    Returns
    -------
    SparseRunDescriptor
        Descriptor with a complete producer preflight report. A failed report
        is returned for diagnostics and must not be written as runnable.
    """

    _normalize_trace_numpy_scalar_metadata(trace)
    diagnostics: list[RunnableDiagnostic] = []
    backend = str(getattr(trace, "backend", "torch"))
    if backend != "torch":
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY,
                f"Sparse runnable rung 1 supports only the torch backend, not {backend!r}.",
                detection_stage="producer_backend",
            )
        )

    ops = list(getattr(trace, "layer_list", ()))
    op_by_alias = _op_alias_index(trace, ops)
    slot_drafts, slot_for_op = _build_op_slot_drafts(trace, ops, diagnostics)
    _build_child_version_slot_drafts(trace, ops, slot_drafts, slot_for_op)
    state_slots = _build_parameter_slot_drafts(trace)
    slot_drafts.update(state_slots)
    _add_persistent_buffer_slot_drafts(trace, slot_drafts)

    registry_entries: list[CallableRegistryEntry] = []
    registry_id_by_key: dict[FunctionRegistryKey, str] = {}
    calls: list[RunnableCallDescriptor] = []
    producer_call_by_slot: dict[str, str] = {}
    saw_unmodelled_host_write = False
    grouped_ops = _group_computational_ops(ops)
    for call_number, call_ops in grouped_ops:
        representative = call_ops[0]
        call_id = f"call:{call_number}"
        func_id = getattr(representative, "func_id", None)
        if not isinstance(func_id, FunctionRegistryKey):
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_CALLABLE_REF,
                    "Computational call has no cooked FunctionRegistryKey; capture with "
                    "intervention_ready=True before saving level='runnable'.",
                    affected_ops=tuple(str(op.label) for op in call_ops),
                    detection_stage="producer_callable",
                )
            )
            continue
        registry_id = registry_id_by_key.get(func_id)
        if registry_id is None:
            registry_id = f"callable:{len(registry_entries) + 1}"
            registry_id_by_key[func_id] = registry_id
            registry_entries.append(CallableRegistryEntry(registry_id=registry_id, key=func_id))

        # r14-C3: a storage-rebinding / storage-reallocating op (``set_`` / ``resize_`` family)
        # cannot be faithfully represented in the sparse DAG -- the resolver already denies it at
        # LOAD as a non-forward callable, which left ``run()`` crashing with a ReattachError. Refuse
        # it at SAVE with a typed diagnostic so the model fails closed here (RunnablePreflightError)
        # rather than crashing at run time, and is never a false VERIFIED.
        storage_unsafe_name = str(getattr(func_id, "qualname", "") or "").rsplit(".", 1)[-1]
        if (
            storage_unsafe_name in _STORAGE_UNSAFE_NAMES
            or str(getattr(representative, "func_name", "")) in _STORAGE_UNSAFE_NAMES
        ):
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.UNTRUSTED_CUSTOM_IMPORT,
                    f"Op {storage_unsafe_name or str(getattr(representative, 'func_name', ''))!r} "
                    "rebinds or reallocates tensor storage (the set_/resize_ family) and is not a "
                    "faithfully representable pure-forward op, so it cannot be saved as a runnable "
                    "call; save is refused here (fail closed at save time, never a crash at run "
                    "time or a false VERIFIED).",
                    registry_id=registry_id,
                    affected_ops=tuple(str(op.label) for op in call_ops),
                    detection_stage="producer_storage_unsafe_op",
                )
            )
            continue

        template = getattr(representative, "args_template", None)
        if not isinstance(template, CapturedArgTemplate):
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
                    "Computational call has no cooked argument template.",
                    registry_id=registry_id,
                    affected_ops=tuple(str(op.label) for op in call_ops),
                    detection_stage="producer_call_recipe",
                )
            )
            continue

        execution_context = _call_execution_context(representative)
        if execution_context is None:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.EXECUTION_CONTEXT_UNAVAILABLE,
                    "Computational call has no captured execution context "
                    "(autocast + grad/inference mode); v2 runnable descriptors "
                    "require an explicit per-call context record.",
                    registry_id=registry_id,
                    affected_ops=tuple(str(op.label) for op in call_ops),
                    detection_stage="producer_execution_context",
                )
            )
            continue

        tensor_args, literal_args, has_unmodelled_host_write = _build_call_arguments(
            representative,
            template,
            call_id=call_id,
            registry_id=registry_id,
            op_by_alias=op_by_alias,
            slot_for_op=slot_for_op,
            slot_drafts=slot_drafts,
            diagnostics=diagnostics,
        )
        saw_unmodelled_host_write = saw_unmodelled_host_write or has_unmodelled_host_write
        output_slot_ids = tuple(slot_for_op[id(op)] for op in call_ops)
        for output_slot_id in output_slot_ids:
            producer_call_by_slot[output_slot_id] = call_id
            for version_slot_id, draft in slot_drafts.items():
                if draft.version_of == output_slot_id:
                    producer_call_by_slot[version_slot_id] = call_id
        parent_call_ids = tuple(
            dict.fromkeys(
                producer_call_by_slot[argument.slot_id]
                for argument in tensor_args
                if argument.slot_id in producer_call_by_slot
            )
        )
        calls.append(
            RunnableCallDescriptor(
                call_id=call_id,
                op_labels=tuple(str(op.label) for op in call_ops),
                registry_id=registry_id,
                dispatch_kind=func_id.dispatch_kind,
                argument_names=tuple(
                    str(name) for name in getattr(representative, "arg_names", ())
                ),
                num_positional_args=int(getattr(representative, "num_pos_args", 0)),
                num_keyword_args=int(getattr(representative, "num_kwargs", 0)),
                tensor_arguments=tuple(tensor_args),
                literal_arguments=tuple(literal_args),
                output_slot_ids=output_slot_ids,
                parent_call_ids=parent_call_ids,
                is_inplace=bool(getattr(representative, "is_inplace", False)),
                runtime_fingerprint=_runtime_fingerprint(
                    representative, func_id, call_ops, execution_context
                ),
                execution_context=execution_context,
            )
        )

    _mark_inplace_versions(calls, slot_drafts)
    witnesses, completeness = _build_control_witnesses(trace, ops, diagnostics)
    literal_witnesses, saw_opaque_leaf = _input_literal_witnesses(trace, start_order=len(witnesses))
    witnesses.extend(literal_witnesses)
    witnesses.extend(_input_metadata_witnesses(trace, start_order=len(witnesses)))
    witnesses.extend(_module_training_mode_witnesses(trace, start_order=len(witnesses)))
    escape_witnesses, escape_incomplete = _escape_witnesses(
        trace, ops, calls, slot_drafts, start_order=len(witnesses)
    )
    witnesses.extend(escape_witnesses)
    if saw_opaque_leaf and completeness is WitnessCompleteness.COMPLETE:
        # An opaque non-tensor input leaf cannot be re-verified, so its control
        # dependency is unobserved: downgrade to keep the run honest
        # (UNVERIFIABLE + NOT_APPLICABLE), never a false VERIFIED/ATTESTED.
        completeness = WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
    if escape_incomplete and completeness is WitnessCompleteness.COMPLETE:
        # A tensor->host escape (of ANY source class or mechanism) whose source cannot
        # be witnessed leaves the escape unobservable at run time: keep the run honestly
        # UNVERIFIABLE. The unified pass folds the former tensor-op and unbound-state
        # incomplete signals into this single fail-closed downgrade.
        completeness = WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
    if (
        _has_forward_value_override_intervention(trace)
        and completeness is WitnessCompleteness.COMPLETE
    ):
        # A forward-modifying intervention (e.g. ``zero_ablate``/``replace_with``)
        # substituted the captured value of an op INSIDE the forward pass. The sparse
        # DAG records only the ORIGINAL op recipe, so a replay recomputes the
        # un-intervened value: the captured output/activations reflect the intervened
        # forward, but the recorded ops do not encode the override. TorchLens cannot
        # cheaply prove at save time whether the override happens to be byte-identical
        # to the natural output (an op-representable no-op like ``scale(1.0)``) without
        # re-executing, so it fails closed here. The single completeness downgrade
        # drives BOTH honesty layers together -- ``_path_faithfulness`` reports
        # UNVERIFIABLE and ``_numeric_attestation_check`` reports NOT_APPLICABLE (never
        # a false VERIFIED, and never a contradicting NumericAttestationError). An
        # observe-only or backward/grad intervention leaves the forward output
        # reproducible byte-for-byte and is NOT flagged here, so it still VERIFIES.
        completeness = WitnessCompleteness.INCOMPLETE_OPAQUE_SIDE_EFFECT
    if _has_input_metadata_view_read(trace) and completeness is WitnessCompleteness.COMPLETE:
        # A metadata predicate (``is_contiguous`` / ``stride`` / ``storage_offset`` / autograd
        # flag) was read on a DERIVED VIEW of a model input (``x.t().is_contiguous()``): the
        # view is an orphan-pruned intermediate the sparse replay never re-derives, so its
        # layout metadata cannot be re-verified against the runtime input. A same-shape layout
        # twin flips such a branch on a fresh model while the replay silently follows the
        # captured arm, so keep the run honestly UNVERIFIABLE + NOT_APPLICABLE rather than a
        # false VERIFIED. A model that never reads metadata on an input-derived view records
        # nothing and stays VERIFIED (no over-trigger).
        completeness = WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
    if _has_pruned_rng_control_flow(trace) and completeness is WitnessCompleteness.COMPLETE:
        # A torch-RNG draw steered pure-Python control flow, so its predicate chain
        # is input-disconnected and was orphaned out of the visible graph. The
        # recorded taken branch is nondeterministic (a fresh seeded forward may take
        # the other arm) yet unwitnessed, so the sparse replay cannot reproduce or
        # even observe the decision: keep the run honestly UNVERIFIABLE +
        # NOT_APPLICABLE, never a false VERIFIED + ATTESTED. A genuinely-dead RNG
        # draw (result influences nothing) is never recorded, so a deterministic
        # model stays VERIFIED.
        completeness = WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
    if _has_pruned_alias_mutation(trace) and completeness is WitnessCompleteness.COMPLETE:
        # An in-place op mutated an UNLABELLED alias (``y.data.add_(5.0)``): the write targets
        # storage the sparse DAG cannot model, so the op was orphan-pruned and the mutation is
        # lost. A replay recomputes the PRE-mutation value (wrong output) yet nothing else
        # witnesses the drop, so keep the run honestly UNVERIFIABLE + NOT_APPLICABLE rather than
        # a false VERIFIED with the mutation gone. An in-place op on a LABELLED alias is graph-
        # connected (replayed) and is never recorded, so a normal model stays VERIFIED.
        completeness = WitnessCompleteness.INCOMPLETE_OPAQUE_SIDE_EFFECT
    if saw_unmodelled_host_write and completeness is WitnessCompleteness.COMPLETE:
        # A surviving in-place op with a removed receiver came from a host alias such as
        # ``buffer.data.add_(1.0)``. The sparse recipe can keep running by reconnecting the receiver
        # to the cooked parent, but the original host write bypassed the ordinary labelled tensor
        # path, so the descriptor cannot honestly prove full path fidelity.
        completeness = WitnessCompleteness.INCOMPLETE_OPAQUE_SIDE_EFFECT
    diagnostics.extend(_preflight_output_contracts(trace, ops))
    ambient_context = _ambient_execution_context(trace, calls, registry_entries, slot_drafts)
    if ambient_context is None:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.EXECUTION_CONTEXT_UNAVAILABLE,
                "Capture recorded no ambient execution context; v2 runnable "
                "descriptors require the explicit capture-scoped backend context "
                "record (re-capture with this TorchLens version).",
                detection_stage="producer_execution_context",
            )
        )
        # Failed-preflight placeholder only: this descriptor can never be written
        # as runnable, and the placeholder is marked attestation-ineligible.
        ambient_context = AmbientExecutionContext(
            default_dtype=str(torch.get_default_dtype()),
            default_device="cpu",
            float32_matmul_precision=None,
            deterministic_algorithms=None,
            deterministic_algorithms_warn_only=None,
            cuda_matmul_allow_tf32=None,
            cudnn_allow_tf32=None,
            cudnn_deterministic=None,
            cudnn_benchmark=None,
            cudnn_enabled=None,
            flash_sdp_enabled=None,
            mem_efficient_sdp_enabled=None,
            math_sdp_enabled=None,
            attestation_ineligible_context=True,
        )
    diagnostics = _deduplicate_diagnostics(diagnostics)
    preflight = ProducerPreflight(passed=not diagnostics, diagnostics=tuple(diagnostics))
    descriptor = SparseRunDescriptor(
        capability=RUNNABLE_TLSPEC_SCHEMA_VERSION,
        backend=backend,
        call_recipe=RUNNABLE_CALL_RECIPE_VERSION,
        callable_ref_schema=RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
        state_binding="module_path_role_v1",
        input_binding="model_site_io_role_v1",
        control_witness="scalar_bool_and_arm_entry_v1",
        initializer_policy_version=RUNNABLE_INITIALIZER_POLICY_VERSION,
        payload_layers=PayloadLayersDescriptor(
            weights=PayloadLayerDescriptor(present=False, schema="state_dict_v1"),
            nonpersistent_buffers=PayloadLayerDescriptor(
                present=any(
                    draft.state_binding is not None and not draft.state_binding.persistent
                    for draft in slot_drafts.values()
                ),
                schema="runnable_nonpersistent_buffer_v1",
            ),
            activations=PayloadLayerDescriptor(
                present=False,
                schema=RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION,
            ),
        ),
        callable_registry=tuple(registry_entries),
        calls=tuple(calls),
        tensor_slots=tuple(draft.freeze() for draft in slot_drafts.values()),
        control_witnesses=tuple(witnesses),
        witness_completeness=completeness,
        rng_profile=_build_rng_profile(trace),
        ambient_context=ambient_context,
        compatibility=RunnableCompatibility(
            torchlens_version=TORCHLENS_VERSION,
            python_version=platform.python_version(),
            backend_version=str(torch.__version__),
            descriptor_version=RUNNABLE_TLSPEC_SCHEMA_VERSION,
            call_recipe_version=RUNNABLE_CALL_RECIPE_VERSION,
            callable_ref_schema_version=RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
            initializer_policy_version=RUNNABLE_INITIALIZER_POLICY_VERSION,
        ),
        preflight=preflight,
        unsupported_sites=tuple(diagnostics),
    )
    assert_sparse_core_has_no_tensor_payload(descriptor)
    return descriptor


def _normalize_trace_numpy_scalar_metadata(trace: Any) -> None:
    """Replace NumPy scalar metadata with equivalent Python values before saving.

    NumPy scalar subclasses are not admitted by the safe metadata unpickler.
    Runnable recipes already use a frozen Python-literal grammar, so preserving a
    raw NumPy scalar in duplicate trace metadata would make an otherwise valid
    runnable artifact save successfully but fail during load.

    Parameters
    ----------
    trace:
        Cooked trace whose runnable projection is being built.
    """

    for op in getattr(trace, "layer_list", ()):
        for field_name in (
            "non_tensor_pos_args",
            "non_tensor_kwargs",
            "func_non_tensor_args",
            "args_template",
            "kwargs_template",
        ):
            if hasattr(op, field_name):
                setattr(op, field_name, _normalize_numpy_scalars(getattr(op, field_name)))
    leaves = getattr(trace, "_runnable_input_nontensor_leaves", None)
    if leaves is not None:
        trace._runnable_input_nontensor_leaves = _normalize_numpy_scalars(leaves)


def _normalize_numpy_scalars(value: Any) -> Any:
    """Recursively convert NumPy scalar leaves to their Python equivalents.

    Parameters
    ----------
    value:
        Arbitrary captured non-tensor metadata.

    Returns
    -------
    Any
        Equivalent metadata with no ``numpy.generic`` leaves.
    """

    if isinstance(value, np.generic):
        return _normalize_numpy_scalars(value.item())
    if isinstance(value, LiteralValue):
        return replace(value, value=_normalize_numpy_scalars(value.value))
    if isinstance(value, CapturedArgTemplate):
        return replace(
            value,
            args=tuple(_normalize_numpy_scalars(item) for item in value.args),
            kwargs=tuple((key, _normalize_numpy_scalars(item)) for key, item in value.kwargs),
        )
    if isinstance(value, list):
        return [_normalize_numpy_scalars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_numpy_scalars(item) for item in value)
    if isinstance(value, set):
        return {_normalize_numpy_scalars(item) for item in value}
    if isinstance(value, Mapping):
        return {
            _normalize_numpy_scalars(key): _normalize_numpy_scalars(item)
            for key, item in value.items()
        }
    return value


def _build_rng_profile(trace: Any) -> RunnableRngProfile:
    """Record host-RNG reproducibility metadata from the capture-time trace.

    ``_runnable_host_rng_consumed`` is stamped during exhaustive capture by
    bracketing the user forward with side-effect-free host-RNG snapshots;
    ``random_seed`` is the concrete effective seed every capture is seeded with.
    """

    consumed = bool(getattr(trace, "_runnable_host_rng_consumed", False))
    seed = getattr(trace, "random_seed", None)
    capture_seed = int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None
    return RunnableRngProfile(host_rng_consumed=consumed, capture_seed=capture_seed)


def require_sparse_run_descriptor(trace: Any) -> SparseRunDescriptor:
    """Build a descriptor and reject any failed whole-graph preflight.

    Parameters
    ----------
    trace:
        Cooked Trace to project.

    Returns
    -------
    SparseRunDescriptor
        Passed sparse descriptor.

    Raises
    ------
    RunnablePreflightError
        If any call, slot, input, output, or control site is unsupported.
    """

    descriptor = build_sparse_run_descriptor(trace)
    if not descriptor.preflight.passed:
        raise RunnablePreflightError(
            "Sparse runnable producer preflight failed.",
            code=RunnableErrorCode.SPARSE_PREFLIGHT_FAILED.value,
            diagnostics=descriptor.preflight.diagnostics,
        )
    return descriptor


def with_weight_payload(descriptor: SparseRunDescriptor) -> SparseRunDescriptor:
    """Return a descriptor declaring the optional state-dict payload present.

    Parameters
    ----------
    descriptor:
        Value-free sparse descriptor produced for the runnable artifact.

    Returns
    -------
    SparseRunDescriptor
        Descriptor with only the separately stored weight-layer presence flag
        changed; the sparse tensor-slot recipe remains value-free.
    """

    return replace(
        descriptor,
        payload_layers=replace(
            descriptor.payload_layers,
            weights=replace(descriptor.payload_layers.weights, present=True),
        ),
    )


def with_activation_payload(
    descriptor: SparseRunDescriptor,
    *,
    members: tuple[ActivationPayloadMember, ...],
    original_input_digests: tuple[SlotByteDigest, ...],
    capture_state_digests: tuple[StateByteDigest, ...],
    input_fingerprints: tuple[InputAttestationFingerprint, ...],
) -> SparseRunDescriptor:
    """Declare one separately stored selected-activation payload family.

    Parameters
    ----------
    descriptor:
        Value-free sparse descriptor produced for the runnable artifact.
    members:
        Exact capture-selected payload membership and logical byte digests.
    original_input_digests:
        Available capture-input slot digests used only for attestation eligibility.
    capture_state_digests:
        Capture-time state digests used to recognize equivalent real state.
    input_fingerprints:
        Physical identity fingerprints of the live capture-time input slots
        (required in ``selected_activation_v2``).

    Returns
    -------
    SparseRunDescriptor
        Descriptor with activation-layer metadata; the sparse recipe remains value-free.
    """

    return replace(
        descriptor,
        payload_layers=replace(
            descriptor.payload_layers,
            activations=ActivationPayloadLayerDescriptor(
                present=True,
                schema=RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION,
                members=members,
                original_input_digests=original_input_digests,
                capture_state_digests=capture_state_digests,
                input_fingerprints=input_fingerprints,
            ),
        ),
    )


def sparse_descriptor_to_json(descriptor: SparseRunDescriptor) -> dict[str, Any]:
    """Convert a frozen sparse descriptor to deterministic JSON values.

    Parameters
    ----------
    descriptor:
        Passed Stage-0 sparse descriptor.

    Returns
    -------
    dict[str, Any]
        JSON-ready descriptor preserving dataclass field names and enum values.
    """

    value = _json_value(descriptor)
    if not isinstance(value, dict):
        raise TypeError("Sparse descriptor JSON projection must be an object.")
    assert_sparse_core_has_no_tensor_payload(value)
    return value


def detach_sparse_core_nested_trace_backrefs(value: Any) -> None:
    """Detach runtime-only nested-``Trace`` back-references from a scrub product.

    A conditional arm records a private ``_trace`` back-reference to its owning
    ``Trace`` (bound in postprocess finalization to serve the ``evaluation_ops`` /
    ``execution_ops`` convenience accessors). ``ConditionalAccessor`` has no
    ``PORTABLE_STATE_SPEC``, so the bundle scrub returns it verbatim and the arm's
    ``_trace`` still points at the LIVE trace -- dragging that trace's
    ``_runnable_capture_state`` (and every other live tensor field) into the value-free
    sparse core through ``conditionals._list.<i>.arms.<j>._trace``. The top-level
    capture-state is dropped by ``Trace.PORTABLE_STATE_SPEC`` (``FieldPolicy.DROP``) and
    routed to the separate ``state_dict_v1`` blob; the arm back-reference must get the
    SAME treatment. Arm replay is driven entirely by the descriptor's recorded
    ``conditional_arm_entry_edges`` control witnesses and top-level state, never by this
    back-reference, so it is dropped (not routed): the sparse core stays value-free.

    This mutates only the passed SCRUB-PRODUCT container (a throwaway dict built by the
    bundle scrub for pickling) -- it rebuilds the ``conditionals`` entry from detached
    shallow arm copies and leaves the LIVE ``Trace`` fully intact (its own accessor
    object is untouched, so ``evaluation_ops`` keeps working after ``save``). It is a
    no-op for the frozen ``SparseRunDescriptor`` projection (which carries no nested
    trace back-reference), and it does NOT weaken the tensor-payload tripwire: any
    OTHER stray tensor still fails :func:`assert_sparse_core_has_no_tensor_payload`.
    """

    import copy as _copy

    from ..data_classes.trace import Conditional, ConditionalAccessor

    if not isinstance(value, MutableMapping):
        return
    accessor = value.get("conditionals")
    if not isinstance(accessor, ConditionalAccessor):
        return
    detached: list[Conditional] = []
    for conditional in accessor.values():
        detached_arms = []
        for arm in conditional.arms:
            arm_copy = _copy.copy(arm)
            arm_copy._trace = None
            detached_arms.append(arm_copy)
        conditional_copy = _copy.copy(conditional)
        conditional_copy.arms = detached_arms
        detached.append(conditional_copy)
    value["conditionals"] = ConditionalAccessor(detached)


def assert_sparse_core_has_no_tensor_payload(value: Any) -> None:
    """Assert that a sparse core contains no tensor or tensor-blob value.

    Parameters
    ----------
    value:
        Descriptor, JSON projection, or scrubbed sparse metadata tree.

    Raises
    ------
    AssertionError
        If a tensor, parameter, or portable tensor blob reference is present.
    """

    from . import BlobRef

    # Detach runtime-only nested-Trace back-references (conditional arm ``_trace``)
    # from the scrub product BEFORE the value-free invariant is enforced. These are
    # not sparse-core payload; leaving them bound would drag the live trace's
    # capture-state tensors into the core (mirrors the top-level DROP-and-route). The
    # tensor-payload walk below is unchanged and still fails on any genuine stray.
    detach_sparse_core_nested_trace_backrefs(value)

    seen: set[int] = set()

    def visit(node: Any, path: tuple[str, ...]) -> None:
        """Visit one node in the sparse-core invariant walk."""

        if isinstance(node, (torch.Tensor, BlobRef)):
            dotted_path = ".".join(path) or "<root>"
            raise AssertionError(f"Sparse core tensor payload at {dotted_path}.")
        if node is None or isinstance(node, (str, bytes, bool, int, float, Enum)):
            return
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        if is_dataclass(node) and not isinstance(node, type):
            for field in fields(node):
                try:
                    field_value = getattr(node, field.name)
                except AttributeError:
                    continue
                visit(field_value, (*path, field.name))
            return
        if isinstance(node, Mapping):
            for key, item in node.items():
                visit(key, (*path, "<key>"))
                visit(item, (*path, str(key)))
            return
        if isinstance(node, (list, tuple, set, frozenset)):
            for index, item in enumerate(node):
                visit(item, (*path, str(index)))
            return
        for field_name, field_value in state_items(node):
            visit(field_value, (*path, str(field_name)))

    visit(value, ())


def _build_op_slot_drafts(
    trace: Any,
    ops: Sequence[Any],
    diagnostics: list[RunnableDiagnostic],
) -> tuple[dict[str, _SlotDraft], dict[int, str]]:
    """Build source, intermediate, and output slot drafts for cooked ops."""

    drafts: dict[str, _SlotDraft] = {}
    slot_for_op: dict[int, str] = {}
    for op in ops:
        slot_id = f"slot:{op.label}"
        slot_for_op[id(op)] = slot_id
        shape = _shape_tuple(getattr(op, "shape", None))
        dtype = _dtype_name(op)
        if shape is None or dtype is None:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_TENSOR_SLOT,
                    "Cooked tensor op is missing shape or dtype metadata.",
                    affected_ops=(str(op.label),),
                    detection_stage="producer_tensor_slot",
                )
            )
            shape = shape or ()
            dtype = dtype or "unknown"
        role = _op_slot_role(op)
        if role is TensorSlotRole.BUFFER:
            buffer_names = set((getattr(trace, "_buffer_initial_values", {}) or {}).keys())
            address = getattr(op, "address", None)
            if address not in buffer_names:
                role = TensorSlotRole.CONSTANT_LIKE_TENSOR
                diagnostics.append(
                    _diagnostic(
                        RunnableErrorCode.UNSUPPORTED_TENSOR_CONSTANT,
                        "Internal tensor source is not present in captured named buffer state "
                        "and has no reproducible initializer recipe.",
                        affected_ops=(str(op.label),),
                        detection_stage="producer_tensor_constant",
                        details=(("address", str(address)),),
                    )
                )
        try:
            output_path = _normalize_container_path(getattr(op, "container_path", ()))
        except ValueError:
            # An output/container path with a non-str/int key (tuple, float, ...)
            # cannot be represented in the frozen slot-path vocabulary. Reject
            # honestly with a typed diagnostic instead of a raw ValueError crash;
            # the run is refused rather than advertised-then-broken.
            output_path = ()
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT,
                    "Output container path uses a key that cannot be represented in the "
                    "runnable slot-path vocabulary (only str/int keys are supported).",
                    affected_ops=(str(op.label),),
                    detection_stage="producer_output_binding",
                )
            )
        input_binding = None
        if role is TensorSlotRole.MODEL_INPUT:
            input_binding = _input_binding_for_op(trace, op, diagnostics)
        producer_slot_id = None
        if role is TensorSlotRole.OUTPUT:
            parents = list(getattr(op, "parents", ()))
            if parents:
                parent_op = _resolve_op(trace, parents[0])
                if parent_op is not None:
                    producer_slot_id = f"slot:{parent_op.label}"
        device_type, device_index = _device_parts(getattr(op, "device_ref", None))
        drafts[slot_id] = _SlotDraft(
            slot_id=slot_id,
            role=role,
            shape=shape,
            dtype=dtype,
            device_type=device_type,
            device_index=device_index,
            mutable=bool(getattr(op, "is_inplace", False)),
            producer_slot_id=producer_slot_id,
            output_path=output_path
            if output_path
            else (() if role is TensorSlotRole.OUTPUT else None),
            input_binding=input_binding,
            state_binding=(_buffer_binding(trace, op) if role is TensorSlotRole.BUFFER else None),
            use_sites=[],
        )
    return drafts, slot_for_op


def _build_parameter_slot_drafts(trace: Any) -> dict[str, _SlotDraft]:
    """Build value-free parameter slots from cooked Param metadata."""

    drafts: dict[str, _SlotDraft] = {}
    param_logs = getattr(trace, "param_logs", {})
    values = getattr(param_logs, "values", lambda: ())()
    for param in values:
        device_type, device_index = _device_parts(getattr(param, "device_ref", None))
        addresses = tuple(str(address) for address in getattr(param, "all_addresses", ()))
        if not addresses:
            addresses = (str(param.address),)
        alias_group = f"alias:{param.barcode}" if len(addresses) > 1 else None
        for address in addresses:
            module_path, separator, _name = address.rpartition(".")
            slot_id = f"state:{address}"
            drafts[slot_id] = _SlotDraft(
                slot_id=slot_id,
                role=TensorSlotRole.PARAMETER,
                shape=tuple(int(dim) for dim in param.shape),
                dtype=str(param.dtype),
                device_type=device_type,
                device_index=device_index,
                state_binding=StateSlotBinding(
                    module_path=module_path if separator else "self",
                    state_dict_name=address,
                    semantic_role=_parameter_role(param),
                    trainable=bool(param.is_trainable),
                    persistent=True,
                    alias_group=alias_group,
                ),
                use_sites=[],
            )
    return drafts


def _add_persistent_buffer_slot_drafts(
    trace: Any,
    drafts: dict[str, _SlotDraft],
) -> None:
    """Add every persistent source-model buffer to the value-free state map.

    Parameters
    ----------
    trace:
        Live cooked Trace whose weak source-model reference is inspection-only.
    drafts:
        Mutable descriptor slots, including any buffers already represented by
        graph source ops.
    """

    source_ref = getattr(trace, "_source_model_ref", None)
    model = source_ref() if callable(source_ref) else None
    state_dict_method = getattr(model, "state_dict", None)
    named_parameters_method = getattr(model, "named_parameters", None)
    named_buffers_method = getattr(model, "named_buffers", None)
    if (
        not callable(state_dict_method)
        or not callable(named_parameters_method)
        or not callable(named_buffers_method)
    ):
        return
    state = state_dict_method()
    if not isinstance(state, Mapping):
        return
    parameter_names = {
        str(name) for name, _value in named_parameters_method(remove_duplicate=False)
    }
    buffers = {str(name): value for name, value in named_buffers_method(remove_duplicate=False)}
    buffer_names = tuple(
        name
        for name, value in state.items()
        if name not in parameter_names and name in buffers and isinstance(value, torch.Tensor)
    )
    names_by_object: dict[int, list[str]] = defaultdict(list)
    for name in buffer_names:
        names_by_object[id(buffers[name])].append(name)
    alias_by_name = {
        name: (f"buffer_alias:{min(names)}" if len(names) > 1 else None)
        for names in names_by_object.values()
        for name in names
    }
    existing_by_name = {
        draft.state_binding.state_dict_name: draft
        for draft in drafts.values()
        if draft.state_binding is not None
    }
    for name in buffer_names:
        alias_group = alias_by_name[name]
        existing = existing_by_name.get(name)
        if existing is not None:
            binding = existing.state_binding
            assert binding is not None
            existing.state_binding = replace(
                binding,
                persistent=True,
                alias_group=alias_group,
            )
            continue
        value = cast(torch.Tensor, state[name])
        module_path, separator, leaf_name = name.rpartition(".")
        device_type, device_index = _device_parts(value.device)
        slot_id = f"state:{name}"
        drafts[slot_id] = _SlotDraft(
            slot_id=slot_id,
            role=TensorSlotRole.BUFFER,
            shape=tuple(int(dim) for dim in value.shape),
            dtype=str(value.dtype),
            device_type=device_type,
            device_index=device_index,
            state_binding=StateSlotBinding(
                module_path=module_path if separator else "self",
                state_dict_name=name,
                semantic_role=_buffer_role(leaf_name or name),
                trainable=False,
                persistent=True,
                alias_group=alias_group,
            ),
            use_sites=[],
        )


def _build_child_version_slot_drafts(
    trace: Any,
    ops: Sequence[Any],
    drafts: dict[str, _SlotDraft],
    slot_for_op: Mapping[int, str],
) -> None:
    """Add value-free per-child tensor-version identities from cooked mapping keys."""

    aliases = _op_alias_index(trace, ops)
    for parent in ops:
        versions = getattr(parent, "out_versions_by_child", {}) or {}
        if not isinstance(versions, Mapping):
            continue
        base_slot_id = slot_for_op[id(parent)]
        base = drafts[base_slot_id]
        for child_label in versions:
            child = aliases.get(str(child_label))
            stable_child_label = str(getattr(child, "label", child_label))
            version_slot_id = f"{base_slot_id}:use:{stable_child_label}"
            drafts[version_slot_id] = _SlotDraft(
                slot_id=version_slot_id,
                role=base.role,
                shape=base.shape,
                dtype=base.dtype,
                device_type=base.device_type,
                device_index=base.device_index,
                mutable=True,
                version_of=base_slot_id,
                producer_slot_id=base_slot_id,
                output_path=base.output_path,
                input_binding=base.input_binding,
                state_binding=base.state_binding,
                use_sites=[],
            )


def _child_version_slot_id(
    parent: Any,
    child: Any,
    base_slot_id: str,
    drafts: Mapping[str, _SlotDraft],
) -> str:
    """Return the child-specific version slot when the cooked core records one."""

    exact_slot_id = f"{base_slot_id}:use:{child.label}"
    if exact_slot_id in drafts:
        return exact_slot_id
    versions = getattr(parent, "out_versions_by_child", {}) or {}
    child_aliases = {
        str(getattr(child, "label", "")),
        str(getattr(child, "layer_label", "")),
    }
    matching_label = next((label for label in versions if str(label) in child_aliases), None)
    if matching_label is None:
        return base_slot_id
    fallback_slot_id = f"{base_slot_id}:use:{matching_label}"
    return fallback_slot_id if fallback_slot_id in drafts else base_slot_id


def _build_call_arguments(
    op: Any,
    template: CapturedArgTemplate,
    *,
    call_id: str,
    registry_id: str,
    op_by_alias: Mapping[str, Any],
    slot_for_op: Mapping[int, str],
    slot_drafts: dict[str, _SlotDraft],
    diagnostics: list[RunnableDiagnostic],
) -> tuple[list[TensorArgumentRef], list[LiteralArgumentRef], bool]:
    """Build tensor and literal call leaves from one cooked argument template."""

    tensor_args: list[TensorArgumentRef] = []
    literal_args: list[LiteralArgumentRef] = []
    parameter_candidates = list(getattr(op, "_param_logs", ()) or ())
    non_tensor_positional = iter(getattr(op, "non_tensor_pos_args", ()) or ())
    has_unmodelled_host_write = False

    for index, component in enumerate(template.args):
        path: tuple[str | int, ...] = ("args", index)
        if _should_recover_removed_inplace_receiver(op, component, path):
            if _append_first_parent_tensor_argument(
                op,
                path=path,
                call_id=call_id,
                op_by_alias=op_by_alias,
                slot_for_op=slot_for_op,
                slot_drafts=slot_drafts,
                tensor_args=tensor_args,
            ):
                has_unmodelled_host_write = True
                continue
        if _should_recover_unattributed_inplace_receiver(op, component, path):
            if _append_unattributed_inplace_receiver(
                op,
                path=path,
                call_id=call_id,
                op_by_alias=op_by_alias,
                slot_for_op=slot_for_op,
                slot_drafts=slot_drafts,
                tensor_args=tensor_args,
            ):
                has_unmodelled_host_write = True
                continue
        override = None
        if not _component_contains_tensor(component):
            override = next(non_tensor_positional, _NO_OVERRIDE)
        _append_argument_component(
            component,
            path=path,
            literal_override=override,
            op=op,
            call_id=call_id,
            registry_id=registry_id,
            op_by_alias=op_by_alias,
            slot_for_op=slot_for_op,
            slot_drafts=slot_drafts,
            parameter_candidates=parameter_candidates,
            tensor_args=tensor_args,
            literal_args=literal_args,
            diagnostics=diagnostics,
        )
    non_tensor_kwargs = getattr(op, "non_tensor_kwargs", {}) or {}
    for key, component in template.kwargs:
        path = ("kwargs", str(key))
        override = non_tensor_kwargs.get(key, _NO_OVERRIDE)
        _append_argument_component(
            component,
            path=path,
            literal_override=override,
            op=op,
            call_id=call_id,
            registry_id=registry_id,
            op_by_alias=op_by_alias,
            slot_for_op=slot_for_op,
            slot_drafts=slot_drafts,
            parameter_candidates=parameter_candidates,
            tensor_args=tensor_args,
            literal_args=literal_args,
            diagnostics=diagnostics,
        )
    return tensor_args, literal_args, has_unmodelled_host_write


def _should_recover_removed_inplace_receiver(
    op: Any,
    component: Any,
    path: tuple[str | int, ...],
) -> bool:
    """Return whether an in-place receiver was removed as an unmodelled host alias.

    Parameters
    ----------
    op:
        Cooked op whose sparse call recipe is being built.
    component:
        Captured argument component at ``path``.
    path:
        Sparse argument path for ``component``.

    Returns
    -------
    bool
        True when an in-place call's receiver was formerly a ``ParentRef`` but
        cleanup replaced it with an unsupported marker.
    """

    return (
        path == ("args", 0)
        and bool(getattr(op, "is_inplace", False))
        and isinstance(component, Unsupported)
        and component.reason == "removed_parent_ref"
        and component.value_type == "ParentRef"
    )


def _should_recover_unattributed_inplace_receiver(
    op: Any,
    component: Any,
    path: tuple[str | int, ...],
) -> bool:
    """Return whether an in-place receiver was captured as an unattributed literal.

    Parameters
    ----------
    op:
        Cooked op whose sparse call recipe is being built.
    component:
        Captured argument component at ``path``.
    path:
        Sparse argument path for ``component``.

    Returns
    -------
    bool
        True when the first argument of an in-place call is an unbound tensor
        literal, as produced by labelled-RHS ``.data`` writes.
    """

    return (
        path == ("args", 0)
        and bool(getattr(op, "is_inplace", False))
        and isinstance(component, LiteralTensor)
    )


def _append_first_parent_tensor_argument(
    op: Any,
    *,
    path: tuple[str | int, ...],
    call_id: str,
    op_by_alias: Mapping[str, Any],
    slot_for_op: Mapping[int, str],
    slot_drafts: dict[str, _SlotDraft],
    tensor_args: list[TensorArgumentRef],
) -> bool:
    """Append the first cooked parent as a recovered in-place receiver.

    Parameters
    ----------
    op:
        Cooked in-place op.
    path:
        Sparse argument path for the receiver.
    call_id:
        Sparse call identifier.
    op_by_alias:
        Lookup table from raw/final labels to cooked ops.
    slot_for_op:
        Lookup table from cooked op identity to tensor slot id.
    slot_drafts:
        Mutable tensor slot descriptors.
    tensor_args:
        Accumulator receiving tensor arguments.

    Returns
    -------
    bool
        True when the receiver was recovered and appended.
    """

    parents = tuple(str(parent) for parent in getattr(op, "parents", ()) or ())
    if not parents:
        return False
    parent = op_by_alias.get(parents[0])
    if parent is None:
        return False
    base_slot_id = slot_for_op.get(id(parent))
    if base_slot_id is None:
        return False
    slot_id = _child_version_slot_id(parent, op, base_slot_id, slot_drafts)
    _append_tensor_argument(
        tensor_args,
        path,
        slot_id,
        call_id=call_id,
        slot_drafts=slot_drafts,
    )
    return True


def _append_unattributed_inplace_receiver(
    op: Any,
    *,
    path: tuple[str | int, ...],
    call_id: str,
    op_by_alias: Mapping[str, Any],
    slot_for_op: Mapping[int, str],
    slot_drafts: dict[str, _SlotDraft],
    tensor_args: list[TensorArgumentRef],
) -> bool:
    """Append the unique graph receiver for an unattributed in-place mutation.

    Parameters
    ----------
    op:
        Cooked in-place op.
    path:
        Sparse argument path for the receiver.
    call_id:
        Sparse call identifier.
    op_by_alias:
        Lookup table from raw/final labels to cooked ops.
    slot_for_op:
        Lookup table from cooked op identity to tensor slot id.
    slot_drafts:
        Mutable tensor slot descriptors.
    tensor_args:
        Accumulator receiving tensor arguments.

    Returns
    -------
    bool
        True when exactly one cooked slot can be identified as the missing
        receiver.
    """

    matches = _unattributed_inplace_receiver_candidates(op, op_by_alias, slot_for_op)
    if len(matches) != 1:
        return False
    slot_id = matches[0]
    _append_tensor_argument(
        tensor_args,
        path,
        slot_id,
        call_id=call_id,
        slot_drafts=slot_drafts,
    )
    return True


def _unattributed_inplace_receiver_candidates(
    op: Any,
    op_by_alias: Mapping[str, Any],
    slot_for_op: Mapping[int, str],
) -> tuple[str, ...]:
    """Return candidate receiver slots for a labelled-RHS ``.data`` write.

    Parameters
    ----------
    op:
        Cooked in-place op whose receiver lacks graph provenance.
    op_by_alias:
        Lookup table from raw/final labels to cooked ops.
    slot_for_op:
        Lookup table from cooked op identity to tensor slot id.

    Returns
    -------
    tuple[str, ...]
        Unique slot IDs whose recorded output-version snapshot matches the
        mutation result.
    """

    target_digest = _tensor_digest(getattr(op, "out", None))
    if target_digest is None:
        return ()
    parent_labels = {str(parent) for parent in getattr(op, "parents", ()) or ()}
    candidates: dict[str, str] = {}
    seen_ops: set[int] = set()
    for candidate in op_by_alias.values():
        candidate_id = id(candidate)
        if candidate_id in seen_ops or candidate_id == id(op):
            continue
        seen_ops.add(candidate_id)
        if str(getattr(candidate, "label", "")) in parent_labels:
            continue
        slot_id = slot_for_op.get(candidate_id)
        if slot_id is None:
            continue
        versions = getattr(candidate, "out_versions_by_child", {}) or {}
        for value in versions.values():
            if _tensor_digest(value) == target_digest:
                candidates[slot_id] = slot_id
                break
    return tuple(candidates)


def _tensor_digest(value: Any) -> str | None:
    """Return a byte digest for a tensor-like value, if available."""

    if not isinstance(value, torch.Tensor):
        return None
    return runnable_tensor_byte_digest(value)


_NO_OVERRIDE = object()


def _append_argument_component(
    component: Any,
    *,
    path: tuple[str | int, ...],
    literal_override: Any,
    op: Any,
    call_id: str,
    registry_id: str,
    op_by_alias: Mapping[str, Any],
    slot_for_op: Mapping[int, str],
    slot_drafts: dict[str, _SlotDraft],
    parameter_candidates: list[Any],
    tensor_args: list[TensorArgumentRef],
    literal_args: list[LiteralArgumentRef],
    diagnostics: list[RunnableDiagnostic],
) -> None:
    """Append one captured argument component to a sparse call recipe."""

    if not _component_contains_tensor(component):
        value = component.value if isinstance(component, LiteralValue) else component
        if literal_override is not _NO_OVERRIDE:
            value = literal_override
        try:
            literal_args.append(LiteralArgumentRef(path, _encode_literal(value)))
        except _UnsupportedLiteralError as exc:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.UNSUPPORTED_LITERAL,
                    str(exc),
                    registry_id=registry_id,
                    affected_ops=(str(op.label),),
                    detection_stage="producer_literal",
                    details=(("argument_path", repr(path)),),
                )
            )
        return
    if isinstance(component, (list, tuple)):
        try:
            literal_args.append(LiteralArgumentRef(path, _tensor_container_skeleton(component)))
        except _UnsupportedLiteralError as exc:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
                    str(exc),
                    registry_id=registry_id,
                    affected_ops=(str(op.label),),
                    detection_stage="producer_call_recipe",
                    details=(("argument_path", repr(path)),),
                )
            )
            return
        for index, item in enumerate(component):
            _append_argument_component(
                item,
                path=(*path, index),
                literal_override=_NO_OVERRIDE,
                op=op,
                call_id=call_id,
                registry_id=registry_id,
                op_by_alias=op_by_alias,
                slot_for_op=slot_for_op,
                slot_drafts=slot_drafts,
                parameter_candidates=parameter_candidates,
                tensor_args=tensor_args,
                literal_args=literal_args,
                diagnostics=diagnostics,
            )
        return
    if isinstance(component, Mapping):
        try:
            literal_args.append(LiteralArgumentRef(path, _tensor_container_skeleton(component)))
        except _UnsupportedLiteralError as exc:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
                    str(exc),
                    registry_id=registry_id,
                    affected_ops=(str(op.label),),
                    detection_stage="producer_call_recipe",
                    details=(("argument_path", repr(path)),),
                )
            )
            return
        for key, item in component.items():
            if not isinstance(key, (str, int)):
                diagnostics.append(
                    _diagnostic(
                        RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
                        "A tensor-containing mapping argument has a key outside the sparse "
                        "argument-path grammar.",
                        registry_id=registry_id,
                        affected_ops=(str(op.label),),
                        detection_stage="producer_call_recipe",
                        details=(("argument_path", repr(path)),),
                    )
                )
                continue
            _append_argument_component(
                item,
                path=(*path, key),
                literal_override=_NO_OVERRIDE,
                op=op,
                call_id=call_id,
                registry_id=registry_id,
                op_by_alias=op_by_alias,
                slot_for_op=slot_for_op,
                slot_drafts=slot_drafts,
                parameter_candidates=parameter_candidates,
                tensor_args=tensor_args,
                literal_args=literal_args,
                diagnostics=diagnostics,
            )
        return
    if isinstance(component, ParentRef):
        parent = op_by_alias.get(component.parent_label)
        if parent is None:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_TENSOR_SLOT,
                    f"No cooked parent slot matches {component.parent_label!r}.",
                    registry_id=registry_id,
                    affected_ops=(str(op.label),),
                    detection_stage="producer_tensor_argument",
                )
            )
            return
        base_slot_id = slot_for_op[id(parent)]
        slot_id = _child_version_slot_id(parent, op, base_slot_id, slot_drafts)
        _append_tensor_argument(
            tensor_args,
            path,
            slot_id,
            call_id=call_id,
            slot_drafts=slot_drafts,
        )
        return
    if isinstance(component, LiteralTensor):
        param = _match_parameter(component.value, path, op, parameter_candidates)
        if param is None:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.UNSUPPORTED_TENSOR_CONSTANT,
                    "Tensor literal is not a named parameter, registered buffer, model input, "
                    "or reproducible source call.",
                    registry_id=registry_id,
                    affected_ops=(str(op.label),),
                    detection_stage="producer_tensor_constant",
                    details=(("argument_path", repr(path)),),
                )
            )
            return
        parameter_candidates.remove(param)
        slot_id = f"state:{param.address}"
        draft = slot_drafts.get(slot_id)
        if draft is not None:
            draft.device_type, draft.device_index = _device_parts(
                getattr(component.value, "device", None)
            )
        _append_tensor_argument(
            tensor_args,
            path,
            slot_id,
            call_id=call_id,
            slot_drafts=slot_drafts,
        )
        return
    if isinstance(component, Unsupported):
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.UNSUPPORTED_LITERAL,
                component.reason,
                registry_id=registry_id,
                affected_ops=(str(op.label),),
                detection_stage="producer_literal",
                details=(("argument_path", repr(path)), ("value_type", component.value_type)),
            )
        )
        return

    diagnostics.append(
        _diagnostic(
            RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
            "A tensor-containing argument container lacks a preserved list/tuple/mapping "
            "contract in the cooked projection.",
            registry_id=registry_id,
            affected_ops=(str(op.label),),
            detection_stage="producer_call_recipe",
            details=(("argument_path", repr(path)),),
        )
    )


def _append_tensor_argument(
    tensor_args: list[TensorArgumentRef],
    path: tuple[str | int, ...],
    slot_id: str,
    *,
    call_id: str,
    slot_drafts: dict[str, _SlotDraft],
) -> None:
    """Append a tensor reference and its reverse use-site metadata."""

    tensor_args.append(TensorArgumentRef(argument_path=path, slot_id=slot_id))
    draft = slot_drafts.get(slot_id)
    if draft is not None:
        if draft.use_sites is None:
            draft.use_sites = []
        draft.use_sites.append(TensorUseSite(call_id=call_id, argument_path=path))


def _match_parameter(
    tensor: Any,
    path: tuple[str | int, ...],
    op: Any,
    candidates: Sequence[Any],
) -> Any | None:
    """Match a template tensor literal to one cooked named parameter."""

    shape = _shape_tuple(getattr(tensor, "shape", None))
    dtype = str(getattr(tensor, "dtype", ""))
    matches = [
        param for param in candidates if tuple(param.shape) == shape and str(param.dtype) == dtype
    ]
    identity_matches = [param for param in matches if _same_tensor_identity(tensor, param)]
    if len(identity_matches) == 1:
        return identity_matches[0]
    argument_names = tuple(getattr(op, "arg_names", ()) or ())
    top_index = path[1] if len(path) > 1 and isinstance(path[1], int) else None
    argument_name = (
        str(argument_names[top_index])
        if top_index is not None and top_index < len(argument_names)
        else None
    )
    named_matches = [param for param in matches if str(param.name) == argument_name]
    if len(named_matches) == 1:
        return named_matches[0]
    return matches[0] if len(matches) == 1 else None


def _same_tensor_identity(tensor: Any, param: Any) -> bool:
    """Return whether ``tensor`` is the exact captured parameter object/storage."""

    return any(_same_tensor_ref(tensor, param_ref) for param_ref in _parameter_refs(param))


def _parameter_refs(param: Any) -> tuple[Any, ...]:
    """Return live parameter objects associated with one cooked ``Param`` record."""

    refs: list[Any] = []
    param_ref = getattr(param, "_param_ref", None)
    if param_ref is not None:
        refs.append(param_ref)
    source_trace_ref = getattr(param, "_source_trace_ref", None)
    trace = source_trace_ref() if callable(source_trace_ref) else None
    source_model_ref = getattr(trace, "_source_model_ref", None)
    model = source_model_ref() if callable(source_model_ref) else None
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        addresses = {str(getattr(param, "address", ""))}
        addresses.update(str(address) for address in getattr(param, "all_addresses", ()) or ())
        try:
            for name, value in named_parameters(remove_duplicate=False):
                if str(name) in addresses:
                    refs.append(value)
        except TypeError:
            for name, value in named_parameters():
                if str(name) in addresses:
                    refs.append(value)
    return tuple(dict.fromkeys(refs))


def _same_tensor_ref(tensor: Any, reference: Any) -> bool:
    """Return whether two tensors are the same object or share the same data pointer."""

    if tensor is reference or id(tensor) == id(reference):
        return True
    tensor_ptr = _tensor_data_ptr(tensor)
    reference_ptr = _tensor_data_ptr(reference)
    return tensor_ptr is not None and tensor_ptr == reference_ptr


def _tensor_data_ptr(tensor: Any) -> int | None:
    """Return a tensor data pointer without raising for non-tensors."""

    data_ptr = getattr(tensor, "data_ptr", None)
    if not callable(data_ptr):
        return None
    try:
        return int(data_ptr())
    except RuntimeError:
        return None


def _group_computational_ops(ops: Sequence[Any]) -> list[tuple[int, list[Any]]]:
    """Group cooked computational output tensors by capture call ID."""

    groups: dict[int, list[Any]] = defaultdict(list)
    first_index: dict[int, int] = {}
    for index, op in enumerate(ops):
        call_number = getattr(op, "func_call_id", None)
        if call_number is None or bool(getattr(op, "is_input", False)):
            continue
        if bool(getattr(op, "is_buffer", False)) or bool(getattr(op, "is_output", False)):
            continue
        call_number = int(call_number)
        groups[call_number].append(op)
        first_index.setdefault(call_number, index)
    return sorted(groups.items(), key=lambda item: first_index[item[0]])


def _op_alias_index(trace: Any, ops: Sequence[Any]) -> dict[str, Any]:
    """Index cooked ops under raw, layer, pass-qualified, and short labels."""

    aliases: dict[str, Any] = {}
    raw_to_final = getattr(trace, "_raw_to_final_op_labels", {}) or {}
    for op in ops:
        for value in (
            getattr(op, "label", None),
            getattr(op, "layer_label", None),
            getattr(op, "label_short", None),
            getattr(op, "layer_label_short", None),
        ):
            if isinstance(value, str):
                aliases[value] = op
        label = str(getattr(op, "label", ""))
        if ":" in label:
            aliases.setdefault(label.rsplit(":", 1)[0], op)
    for raw_label, final_label in raw_to_final.items():
        final_op = aliases.get(str(final_label))
        if final_op is not None:
            aliases[str(raw_label)] = final_op
    return aliases


def _resolve_op(trace: Any, label: str) -> Any | None:
    """Resolve one cooked op label through the Trace lookup tables."""

    layer_dict = getattr(trace, "layer_dict_all_keys", {}) or {}
    if label in layer_dict:
        return layer_dict[label]
    try:
        layer = trace[label]
    except (KeyError, TypeError, ValueError):
        return None
    ops = getattr(layer, "ops", None)
    if ops:
        return ops[0]
    return layer


def _input_binding_for_op(
    trace: Any,
    op: Any,
    diagnostics: list[RunnableDiagnostic],
) -> InputSlotBinding | None:
    """Resolve an input source op to a captured model-boundary container site."""

    containers = getattr(trace, "__dict__", {}).get("_containers", {}) or {}
    raw_candidates = {
        str(getattr(op, "label", "")),
        str(getattr(op, "layer_label", "")),
    }
    final_to_raw = getattr(trace, "_final_to_raw_layer_labels", {}) or {}
    raw_label = final_to_raw.get(getattr(op, "layer_label", None))
    if isinstance(raw_label, str):
        raw_candidates.add(raw_label)
    for record in containers.values():
        for snapshot in getattr(record, "snapshots", ()):
            if getattr(snapshot, "role", None) is not Role.MODEL_INPUT:
                continue
            site = getattr(snapshot, "site", None)
            if not isinstance(site, ModelSite):
                continue
            for occurrence in getattr(snapshot, "leaf_occurrences", ()):
                producer = getattr(occurrence, "producer_op_label", None)
                if not isinstance(producer, str):
                    continue
                cooked = (getattr(trace, "_raw_to_final_op_labels", {}) or {}).get(producer)
                if producer not in raw_candidates and cooked not in raw_candidates:
                    continue
                position = _normalize_model_site_position(site.position)
                if position is None:
                    break
                return InputSlotBinding(
                    io_role="model_input",
                    model_ref=site.model_ref,
                    model_site_position=position,
                    container_record_id=int(record.ordinal),
                    container_path=_normalize_container_path(occurrence.path),
                )

    io_role = str(getattr(op, "io_role", ""))
    if io_role.count(".") > 1:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.MISSING_INPUT_CONTAINER_CONTRACT,
                "Nested model input has no captured ContainerRecord/ModelSite contract; "
                "capture with capture_container_structure=True.",
                affected_ops=(str(op.label),),
                detection_stage="producer_input_binding",
            )
        )
        return None
    input_ops = [item for item in getattr(trace, "layer_list", ()) if item.is_input]
    input_index = next((index for index, item in enumerate(input_ops) if item is op), 0)
    return InputSlotBinding(
        io_role="model_input",
        model_ref="self:1",
        model_site_position=("arg", input_index),
        container_record_id=-1,
        container_path=(),
    )


def _build_control_witnesses(
    trace: Any,
    ops: Sequence[Any],
    diagnostics: list[RunnableDiagnostic],
) -> tuple[list[ControlWitness], WitnessCompleteness]:
    """Build ordered scalar, loop, arm-entry, and structure witnesses."""

    witnesses: list[ControlWitness] = []
    completeness = WitnessCompleteness.COMPLETE
    for op in ops:
        if not bool(getattr(op, "is_scalar_bool", False)):
            continue
        bool_value = getattr(op, "bool_value", None)
        context_kind = getattr(op, "conditional_context_kind", None)
        if bool_value is None:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_CONTROL_CLASSIFICATION,
                    "Scalar-bool op has no recorded bool_value.",
                    affected_ops=(str(op.label),),
                    detection_stage="producer_control_witness",
                )
            )
            completeness = WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
            continue
        if context_kind in {None, "unknown"} and bool(getattr(op, "is_terminal_bool", False)):
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_CONTROL_CLASSIFICATION,
                    "Terminal scalar-bool escaped without a classified consumer.",
                    affected_ops=(str(op.label),),
                    detection_stage="producer_control_witness",
                )
            )
            completeness = WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        kind = (
            ControlWitnessKind.LOOP_PREDICATE
            if context_kind == "while"
            else ControlWitnessKind.SCALAR_BOOL
        )
        call_number = getattr(op, "func_call_id", None)
        witnesses.append(
            ControlWitness(
                witness_id=f"witness:{len(witnesses) + 1}",
                kind=kind,
                order=len(witnesses),
                call_id=None if call_number is None else f"call:{call_number}",
                site_label=str(op.label),
                observed_value=_encode_literal(bool(bool_value)),
            )
        )

    arm_edges = getattr(trace, "conditional_arm_entry_edges", {}) or {}
    for (conditional_id, arm_kind), edges in sorted(
        arm_edges.items(), key=lambda item: (int(item[0][0]), str(item[0][1]))
    ):
        for parent, child in edges:
            witnesses.append(
                ControlWitness(
                    witness_id=f"witness:{len(witnesses) + 1}",
                    kind=ControlWitnessKind.CONDITIONAL_ARM_ENTRY,
                    order=len(witnesses),
                    call_id=None,
                    site_label=f"conditional:{conditional_id}:{arm_kind}:{parent}->{child}",
                    observed_value=_encode_literal(True),
                )
            )

    witnesses.extend(_container_structure_witnesses(trace, start_order=len(witnesses)))
    return witnesses, completeness


def _op_retained_tensor(trace: Any, op: Any) -> torch.Tensor | None:
    """Return one op's retained capture-time output tensor, or ``None``.

    Reads the retained capture-time output activation for any shape/dtype. A
    missing or non-tensor activation yields ``None`` so an unsaved slot is treated
    conservatively as un-witnessable rather than being forced to an unverifiable
    run.
    """

    label = getattr(op, "label", None)
    if label is None:
        return None
    try:
        value = trace[label].out
    except (KeyError, AttributeError, RuntimeError, TypeError, IndexError, ValueError):
        # A selectively-captured op whose activation was not retained raises when
        # its ``out`` is read; treat it as un-witnessable rather than failing build.
        return None
    return value if isinstance(value, torch.Tensor) else None


_TENSOR_SOURCE_ESCAPE_MAX_SEQUENCE_NUMEL = 4096
"""Upper bound on element count scanned for a value-equality sequence bake match."""


def _collect_baked_literal_values(
    calls: Sequence[RunnableCallDescriptor],
) -> tuple[set[int], set[float], set[tuple[Any, ...]]]:
    """Collect baked non-bool scalar and flat-numeric sequence literal values.

    Returns the set of int literals, float literals, and flat numeric sequence
    tuples that appear anywhere in a downstream call's literal arguments -- the
    exact host-side footprints a tensor->Python escape leaves when its value is
    baked verbatim into a later op.
    """

    ints: set[int] = set()
    floats: set[float] = set()
    sequences: set[tuple[Any, ...]] = set()

    def visit(node: Any) -> None:
        if isinstance(node, LiteralAtom):
            if node.kind is LiteralAtomKind.INT and isinstance(node.value, int):
                ints.add(int(node.value))
            elif node.kind is LiteralAtomKind.FLOAT and isinstance(node.value, float):
                floats.add(float(node.value))
            return
        if isinstance(node, LiteralSequence):
            flat: list[Any] = []
            numeric = True
            for item in node.items:
                if (
                    isinstance(item, LiteralAtom)
                    and item.kind in {LiteralAtomKind.INT, LiteralAtomKind.FLOAT}
                    and not isinstance(item.value, bool)
                ):
                    flat.append(item.value)
                else:
                    numeric = False
                visit(item)
            if numeric and flat:
                sequences.add(tuple(flat))

    for call in calls:
        for literal in call.literal_arguments:
            visit(literal.value)
    return ints, floats, sequences


def _value_matches_baked_literal(
    value: torch.Tensor,
    ints: set[int],
    floats: set[float],
    sequences: set[tuple[Any, ...]],
    sequence_lengths: set[int],
) -> bool:
    """Return whether a tensor's exact value was baked verbatim into a literal.

    This is the value-equality net for a DUAL-USE escape: an op whose output also
    feeds the traced graph (so it is not an internal sink) but whose Python-escaped
    value was baked verbatim into a downstream op. Matches a scalar against baked
    int/float atoms and a small tensor against a baked flat-numeric sequence.
    """

    numel = int(value.numel())
    if numel == 1:
        try:
            scalar = value.item()
        except (RuntimeError, ValueError):
            return False
        if isinstance(scalar, bool):
            return False
        if isinstance(scalar, int):
            return scalar in ints
        if isinstance(scalar, float):
            return scalar in floats
        return False
    if 1 < numel <= _TENSOR_SOURCE_ESCAPE_MAX_SEQUENCE_NUMEL and numel in sequence_lengths:
        try:
            flat = tuple(value.detach().flatten().tolist())
        except (RuntimeError, ValueError):
            return False
        return flat in sequences
    return False


def _has_input_metadata_view_read(trace: Any) -> bool:
    """Return whether a metadata predicate was read on a DERIVED VIEW of a model input.

    The completeness-witness scoped patch records (in a weak-keyed module table) any trace
    that read a layout/autograd predicate (``is_contiguous`` / ``stride`` / ``storage_offset``
    / ``requires_grad`` / ``grad_fn`` / ``is_leaf``) on a pure view of a model-input leaf
    (``x.t().is_contiguous()``). That view is an orphan-pruned intermediate the sparse replay
    never re-derives, so the read cannot be re-verified against the runtime input and the
    producer must downgrade witness completeness to keep the run honest.
    """

    from ..backends.torch.completeness_witness import input_metadata_view_read

    return bool(input_metadata_view_read(trace))


def _has_pruned_rng_control_flow(trace: Any) -> bool:
    """Return whether a torch-RNG op that steered control flow was orphan-pruned.

    Postprocess orphan removal records (in a weak-keyed side table) any
    ``nondeterministic_seeded`` torch-RNG op whose result drove a pure-Python
    control decision but was input-disconnected and pruned from the visible
    graph (see ``graph_traversal._record_pruned_rng_control_flow``). Such an op
    never reaches the runnable descriptor, so the producer consults this fact to
    downgrade witness completeness and keep the model honestly UNVERIFIABLE +
    NOT_APPLICABLE rather than falsely VERIFIED + ATTESTED. A deterministic model
    (or one with only genuinely-dead RNG draws) records nothing here.
    """

    from ..backends.torch.completeness_witness import pruned_rng_control_source_labels

    return bool(pruned_rng_control_source_labels(trace))


def _has_pruned_alias_mutation(trace: Any) -> bool:
    """Return whether an in-place op mutating an unlabelled alias was orphan-pruned.

    Postprocess orphan removal records (in a weak-keyed side table) any in-place op whose
    mutation target carried no resolvable capture label -- an invisible ``.data`` / foreign
    alias (``y.data.add_(5.0)``) -- that was actually dropped from the visible graph (see
    ``graph_traversal._record_pruned_alias_mutation``). The dropped write never reaches the
    runnable descriptor, so a replay recomputes the pre-mutation value: the producer consults
    this fact to downgrade witness completeness and keep the model honestly UNVERIFIABLE +
    NOT_APPLICABLE rather than falsely VERIFIED with the mutation lost. A model whose in-place
    ops all target graph-connected (labelled) tensors records nothing here.
    """

    from ..backends.torch.completeness_witness import pruned_alias_mutation_source_labels

    return bool(pruned_alias_mutation_source_labels(trace))


def _has_forward_value_override_intervention(trace: Any) -> bool:
    """Return whether the capture applied a forward-modifying value-override.

    A forward intervention that REPLACED an op's output value (``zero_ablate``,
    ``replace_with``, ``scale``, ``mean_ablate``, ...) makes the captured forward
    diverge from what the recorded sparse DAG ops recompute: the DAG stores only the
    original op recipe, never the value substitution. Such an artifact cannot
    faithfully re-run the intervention-captured forward, so the producer downgrades
    witness completeness to keep the run honestly UNVERIFIABLE + NOT_APPLICABLE
    rather than falsely VERIFIED (with a contradicting NumericAttestationError when
    activations are archived).

    Only forward-direction, value-replacing interventions are flagged. An
    observe-only intervention (``replaced=False``) or a backward/grad intervention
    (``direction != "forward"``) leaves the forward output reproducible byte-for-byte
    and is intentionally NOT flagged, so it still saves and VERIFIES. A plain,
    non-intervened capture records no such op and is unchanged.
    """

    for op in getattr(trace, "layer_list", ()) or ():
        if not getattr(op, "intervention_replaced", False):
            continue
        for record in getattr(op, "interventions", ()) or ():
            direction = getattr(record, "direction", None)
            if bool(getattr(record, "replaced", False)) and direction == "forward":
                return True
    return False


def _escape_witnesses(
    trace: Any,
    ops: Sequence[Any],
    calls: Sequence[RunnableCallDescriptor],
    slot_drafts: Mapping[str, _SlotDraft],
    *,
    start_order: int,
) -> tuple[list[ControlWitness], bool]:
    """Witness the SOURCE of every tensor->host escape in one exhaustive fail-closed pass.

    A tensor->Python escape (``.item()`` / ``int()`` / ``float()`` / ``__index__``
    / ``.tolist()`` / ``.numpy()`` / ``aten._local_scalar_dense``) reads a captured
    op's output tensor and hands a host value to Python. That host value is then
    consumed as a baked op-arg literal (verbatim OR after arbitrary Python
    arithmetic) or as a pure-Python control-flow predicate. The escape breaks the
    tensor graph, so the sparse DAG never recomputes it: on a CHANGED input the
    baked literal / taken branch is STALE while the run would otherwise falsely
    report VERIFIED (+ATTESTED). This is the honesty tripwire.

    This single pass closes the whole (SOURCE class x ESCAPE mechanism x USE) matrix
    so no per-net seam can leave a recognized escape un-witnessed-and-not-flagged:

    * SOURCE class -- model INPUT, INTERNAL op output, BOUND param, BOUND buffer, and
      UNBOUND param/buffer are ALL witnessed. A bound state slot that also feeds a
      graph op still gets an ESCAPE witness (its capture-time state digest): bound-ness
      only exempts the UNBOUND-state net, never the escape witness.
    * ESCAPE mechanism -- census-VISIBLE ``.item()`` / ``int()`` / ``float()`` /
      ``__index__`` / ``bool()`` (``aten._local_scalar_dense``) AND census-INVISIBLE
      ``.tolist()`` / ``.numpy()`` / ``__array__`` (observed at the torch-function
      layer, recorded into the SAME source tables). The witness keys on the SOURCE
      tensor's digest, so host ARITHMETIC on the escaped value (``s*2+1`` /
      ``sum(...)`` / ``.sum()``) is irrelevant -- the source is what changes.
    * USE -- verbatim literal, host-arithmetic literal, and pure-Python control flow
      are all covered by witnessing the SOURCE rather than correlating a baked value.

    PASS A witnesses state slots by their capture-time state digest: every UNBOUND
    state slot (the host-only-read net) PLUS every state slot that is an escape source
    by name (bound or unbound) PLUS a state slot whose scalar value equals an
    unattributable (unlabelled-source) escaped value. PASS B witnesses non-state
    tensor-op escape sources (input / internal) by their retained-output digest, plus
    the value-equality OPTIMIZATION for a dual-use verbatim bake.

    At run time each witnessed slot is re-digested; a differing digest (a CHANGED input
    or CHANGED staged state) means the escaped value / branch may be stale -> the run
    reports UNVERIFIABLE + NOT_APPLICABLE rather than a false VERIFIED. Capture-equivalent
    input+state re-digests byte-identically -> still VERIFIED (+ATTESTED where eligible).
    A source that genuinely cannot be witnessed (an orphan-pruned census label, an
    unattributable bool control predicate, an unattributable census-invisible escape,
    or an unattributable value matching no sink/state) makes the witness set INCOMPLETE
    (fail closed), never a silent pass. Scalar-*bool* escapes with a resolvable source
    are the control-witness net's domain and excluded from the tensor-op gate here.
    """

    from ..backends.torch.completeness_witness import (
        host_escape_has_mutable_writeback,
        host_escape_has_raw_pointer,
        host_escape_has_unattributable_bool,
        host_escape_has_unattributable_opaque,
        host_escape_state_source_names,
        host_escape_unattributable_values,
    )

    witnesses: list[ControlWitness] = []
    escaped_labels, unresolvable_escape = _host_escape_source_labels(trace)
    state_names = host_escape_state_source_names(trace)
    # Fail closed for the genuinely-unwitnessable escape shapes: an orphan-pruned census
    # tensor-op source (:_host_escape_source_labels), an unattributable (``.data`` alias) bool
    # control predicate covered by no net, an unattributable census-invisible
    # (``.tolist``/``.numpy``) escape with no source slot, and a detected host WRITE-BACK through
    # a mutable zero-copy alias (``.numpy()[0] = 99``) -- the write mutates the source bytes with
    # no dispatch and no version bump, so the sparse replay recomputes the pre-write value and the
    # source digest cannot witness it; keep the run honestly UNVERIFIABLE. A raw ``data_ptr()``
    # pointer escape is likewise unobservable (r15-H1) and fails closed here too.
    incomplete = unresolvable_escape
    if host_escape_has_unattributable_bool(trace):
        incomplete = True
    if host_escape_has_unattributable_opaque(trace):
        incomplete = True
    if host_escape_has_mutable_writeback(trace):
        incomplete = True
    # A raw ``Tensor.data_ptr()`` pointer escape (r15-H1) leaves the source tensor's subsequent
    # value unobservable (a raw ctypes read/write bypasses every dispatch and byte watch), so the
    # run must fail closed to UNVERIFIABLE rather than a false VERIFIED.
    if host_escape_has_raw_pointer(trace):
        incomplete = True

    ints, floats, sequences = _collect_baked_literal_values(calls)
    # An UNLABELLED-source non-bool escape (a ``.data`` alias) leaves a scalar value but
    # no source-op label. Treat those values as baked-literal candidates (internal sink)
    # AND as capture-state candidates (PASS A). A value matching NEITHER -> INCOMPLETE.
    unattr_values: set[Any] = {
        value for value in host_escape_unattributable_values(trace) if not isinstance(value, bool)
    }
    unmatched_unattr: set[Any] = set(unattr_values)
    for escaped in unattr_values:
        if isinstance(escaped, int):
            ints.add(escaped)
        elif isinstance(escaped, float):
            floats.add(escaped)
    sequence_lengths = {len(item) for item in sequences}

    call_id_by_slot: dict[str, str] = {}
    for call in calls:
        for slot_id in call.output_slot_ids:
            call_id_by_slot.setdefault(slot_id, call.call_id)
    bound_slot_ids: set[str] = set()
    for call in calls:
        for argument in call.tensor_arguments:
            bound_slot_ids.add(argument.slot_id)
    # State NAMES that feed at least one traced call as a tensor argument. A registered buffer
    # whose value is consumed by a traced op (BatchNorm ``running_mean`` / ``running_var`` /
    # ``num_batches_tracked``, read by the ``batch_norm`` call) is graph-connected: its in-place
    # running-stat update is a TRACKED side effect the replay reproduces natively, so its extra
    # post-update orphan buffer VERSION must NOT be treated as an untraced host-path escape
    # (r15-C3). The unbound-state-escape net is for buffers/params consumed by NO traced call.
    bound_state_names: set[str] = set()
    for slot_id, draft in slot_drafts.items():
        binding = draft.state_binding
        if binding is not None and slot_id in bound_slot_ids:
            bound_state_names.add(binding.state_dict_name)
    capture_state = trace.__dict__.get("_runnable_capture_state")

    # ---- PASS A: state-slot escape/host-path witnesses (bound-or-unbound) ----
    for slot_id, draft in slot_drafts.items():
        binding = draft.state_binding
        if binding is None:
            continue
        name = binding.state_dict_name
        captured = capture_state.get(name) if isinstance(capture_state, Mapping) else None
        matched_value: Any = None
        if isinstance(captured, torch.Tensor) and unmatched_unattr and int(captured.numel()) == 1:
            try:
                scalar = captured.detach().item()
            except (RuntimeError, ValueError, TypeError):
                scalar = None
            if not isinstance(scalar, bool) and scalar in unmatched_unattr:
                matched_value = scalar
        is_named_escape = name in state_names
        # Name-level, not slot-level: a buffer with ANY slot consumed by a traced call is
        # graph-connected (r15-C3), so a normal tracked in-place running-stat update is
        # replayable and stays VERIFIED. A buffer/param read only on an untraced host path has
        # NO bound slot for its name and is still witnessed here (fails closed on changed state).
        is_unbound = slot_id not in bound_slot_ids and name not in bound_state_names
        if not (is_named_escape or is_unbound or matched_value is not None):
            continue
        if not isinstance(captured, torch.Tensor):
            # A state slot that must be witnessed but whose capture value is not
            # available cannot be re-verified: fail closed (UNVERIFIABLE).
            incomplete = True
            continue
        try:
            digest = runnable_tensor_byte_digest(captured)
        except (RuntimeError, ValueError, TypeError):
            incomplete = True
            continue
        fact = {
            UNBOUND_STATE_ESCAPE_FACT_KEY: True,
            "state_dict_name": name,
            "slot_id": slot_id,
            "digest": digest,
        }
        try:
            observed = _encode_literal(fact)
        except _UnsupportedLiteralError:
            incomplete = True
            continue
        order = start_order + len(witnesses)
        witnesses.append(
            ControlWitness(
                witness_id=f"witness:{order + 1}",
                kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
                order=order,
                call_id=None,
                site_label=f"{UNBOUND_STATE_ESCAPE_SITE_PREFIX}{name}",
                observed_value=observed,
            )
        )
        if matched_value is not None:
            unmatched_unattr.discard(matched_value)

    # ---- PASS B: non-state tensor-op escape sources (input / internal) ----
    seen_slots: set[str] = set()
    covered_labels: set[str] = set()
    for op in ops:
        is_input = bool(getattr(op, "is_input", False))
        is_output = bool(getattr(op, "is_output", False))
        is_escape = str(op.label) in escaped_labels
        # A bound param/buffer escape source (state address recorded) is witnessed by
        # PASS A's state digest; do NOT re-witness it as a tensor-op slot, and never
        # treat it as INCOMPLETE here (its state digest already covers the escape).
        address = getattr(op, "address", None)
        if address is not None and str(address) in state_names:
            if is_escape:
                covered_labels.add(str(op.label))
            continue
        # An input/output BOUNDARY op is witnessed ONLY when the census recorded a host
        # escape reading it; a non-escape boundary op is skipped (witnessing the always-
        # present output/un-escaped input would falsely downgrade every changed run).
        if (is_input or is_output) and not is_escape:
            continue
        if bool(getattr(op, "is_scalar_bool", False)) or bool(
            getattr(op, "is_terminal_bool", False)
        ):
            continue
        slot_id = f"slot:{op.label}"
        if slot_id in seen_slots:
            continue
        call_id = call_id_by_slot.get(slot_id)
        if call_id is None and not (is_input or is_output):
            continue
        is_sink = bool(getattr(op, "is_internal_sink", False))
        if not is_escape and not is_sink:
            continue
        value = _op_retained_tensor(trace, op)
        if value is None:
            if is_escape:
                incomplete = True
            continue
        matched = is_escape
        # Value-equality OPTIMIZATION (secondary): a dual-use internal sink whose exact
        # value was baked verbatim into a downstream literal (or equals an unattributable
        # escaped value). Never the ONLY net -- the source-digest witness above is primary.
        if not matched and is_sink:
            matched = _value_matches_baked_literal(value, ints, floats, sequences, sequence_lengths)
        if not matched:
            continue
        matched_value_here: Any = None
        if unmatched_unattr and int(value.numel()) == 1:
            try:
                sink_scalar = value.item()
            except (RuntimeError, ValueError):
                sink_scalar = None
            if not isinstance(sink_scalar, bool) and sink_scalar in unmatched_unattr:
                matched_value_here = sink_scalar
        try:
            digest = runnable_tensor_byte_digest(value)
        except (RuntimeError, ValueError, TypeError):
            if is_escape:
                incomplete = True
            continue
        try:
            observed = _encode_literal(digest)
        except _UnsupportedLiteralError:
            if is_escape:
                incomplete = True
            continue
        order = start_order + len(witnesses)
        witnesses.append(
            ControlWitness(
                witness_id=f"witness:{order + 1}",
                kind=ControlWitnessKind.TENSOR_DERIVED_SCALAR_LITERAL,
                order=order,
                call_id=call_id,
                site_label=slot_id,
                observed_value=observed,
            )
        )
        seen_slots.add(slot_id)
        if is_escape:
            covered_labels.add(str(op.label))
        if matched_value_here is not None:
            unmatched_unattr.discard(matched_value_here)

    # ---- Structural invariant: every recorded escape fact is witnessed OR INCOMPLETE ----
    # No net-exclusion may silently drop a recognized escape.
    if unmatched_unattr:
        # An unattributable non-bool escape value matched neither a witnessed internal
        # sink nor a capture-state slot: its source cannot be witnessed -> fail closed.
        incomplete = True
    if not incomplete:
        from ..backends.torch.completeness_witness import host_escape_bool_source_labels

        raw_bool = host_escape_bool_source_labels(trace)
        raw_to_final = getattr(trace, "_raw_to_final_op_labels", {}) or {}
        bool_final_labels = {
            raw_to_final[raw] for raw in raw_bool if isinstance(raw_to_final.get(raw), str)
        }
        for label in escaped_labels:
            if label in covered_labels or label in bool_final_labels:
                continue
            # A resolvable non-bool escape source that PASS A/B did not witness would
            # otherwise slip through a net seam: fail closed instead.
            incomplete = True
            break
    return witnesses, incomplete


def _host_escape_source_labels(trace: Any) -> tuple[frozenset[str], bool]:
    """Return final escape-source labels and whether any census label is unresolvable.

    The dispatch census records, for each ``aten._local_scalar_dense`` escape, the
    RAW capture label of the source tensor's producing op (see
    ``completeness_witness._record_host_escape_source``). This resolves those raw
    labels to their final cooked op labels via the trace's raw->final map so the
    descriptor can witness the escape's source slot.

    A census raw label that does NOT resolve to a final op is a real escape whose
    source chain was orphan-PRUNED -- a host-only chain that reached neither an
    input nor an output (e.g. a param-rooted ``float((w + 1).sum())``). Such a slot
    can never be digest-witnessed, so the second return value flags it and the
    caller must fail honest (mark the witness set INCOMPLETE). Silently dropping it
    (the pre-R10 behavior) left completeness COMPLETE -> false VERIFIED on changed
    state. The value-equality net and the unbound-state net remain independent
    safety nets for the resolvable cases.
    """

    from ..backends.torch.completeness_witness import (
        host_escape_source_labels,
        host_escape_state_source_labels,
    )

    raw_labels = host_escape_source_labels(trace)
    if not raw_labels:
        return frozenset(), False
    state_labels = host_escape_state_source_labels(trace)
    raw_to_final = getattr(trace, "_raw_to_final_op_labels", {}) or {}
    final_labels: set[str] = set()
    unresolvable = False
    for raw_label in raw_labels:
        if not isinstance(raw_label, str):
            unresolvable = True
            continue
        final = raw_to_final.get(raw_label)
        if isinstance(final, str):
            final_labels.add(final)
        elif raw_label not in state_labels:
            # An unresolved source has no witnessable slot -> fail honest. This covers an
            # orphan-pruned TENSOR-OP host-only chain (``float((w + 1).sum())``) AND a
            # PRUNED BOOL control predicate (``bool(self.gate.data > 0.5)`` -- the ``.data``
            # alias severs the graph link so the gt is orphan-pruned and NO control-witness
            # net can see it; spec point 3). A RESOLVED bool predicate is a real captured
            # conditional and the control-witness net covers it, so it is not flagged here.
            # An unresolved STATE source (a buffer/param read only on the host) is covered
            # by the state net, so it does not close as a pruned chain here.
            unresolvable = True
    return frozenset(final_labels), unresolvable


UNBOUND_STATE_ESCAPE_SITE_PREFIX = "unbound_state_escape:"
"""``site_label`` prefix marking a witnessed unbound state (buffer/param) escape."""

UNBOUND_STATE_ESCAPE_FACT_KEY = "unbound_state_escape"
"""Discriminator key present in every unbound-state escape fact.

The site prefix and fact key are shared by the unified ``_escape_witnesses`` PASS A,
which witnesses UNBOUND state slots (host-only reads) AND bound state slots that are
escape sources (a ``self.gate.item()`` on a buffer that also feeds a graph op), both by
their capture-time state digest. bound-ness exempts a state slot from the unbound net
only, never from the escape witness.
"""


def _container_structure_witnesses(trace: Any, *, start_order: int) -> list[ControlWitness]:
    """Encode captured model-boundary container facts as non-tensor witnesses."""

    witnesses: list[ControlWitness] = []
    containers = getattr(trace, "__dict__", {}).get("_containers", {}) or {}
    for record_id, record in sorted(containers.items()):
        for snapshot_index, snapshot in enumerate(getattr(record, "snapshots", ())):
            if getattr(snapshot, "role", None) not in {Role.MODEL_INPUT, Role.MODEL_OUTPUT}:
                continue
            spec = getattr(snapshot, "spec", None)
            fact = {
                "record_id": int(record_id),
                "snapshot": snapshot_index,
                "role": snapshot.role.value,
                "kind": getattr(spec, "kind", "unknown"),
                "reconstructable": bool(getattr(snapshot, "reconstructable", False)),
                "leaf_paths": [
                    list(normalized)
                    for occurrence in getattr(snapshot, "leaf_occurrences", ())
                    if (normalized := _safe_normalize_container_path(occurrence.path)) is not None
                ],
            }
            try:
                observed = _encode_literal(fact)
            except _UnsupportedLiteralError:
                continue
            order = start_order + len(witnesses)
            witnesses.append(
                ControlWitness(
                    witness_id=f"witness:{order + 1}",
                    kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
                    order=order,
                    call_id=None,
                    site_label=f"container:{record_id}:{snapshot_index}",
                    observed_value=observed,
                )
            )
    return witnesses


MODEL_INPUT_LITERAL_SITE_PREFIX = "model_input_literal:"
"""``site_label`` prefix marking a witnessed non-tensor model-input leaf."""

MODEL_INPUT_LITERAL_FACT_KEY = "model_input_literal"
"""Discriminator key present in every non-tensor model-input leaf fact."""

MODULE_TRAINING_MODE_SITE_PREFIX = "module_training_mode:"
"""``site_label`` prefix marking the declared capture-time per-module train/eval mode."""

MODULE_TRAINING_MODE_FACT_KEY = "module_training_mode"
"""Discriminator key present in the declared per-module train/eval mode fact."""


def _module_training_mode_witnesses(
    trace: Any,
    *,
    start_order: int,
) -> list[ControlWitness]:
    """Declare the capture-time per-module ``training`` mode as a structure fact.

    ``self.training`` is module state outside the ``state_dict``, but it steers
    mode-sensitive ops (BatchNorm running-stats vs batch-stats, Dropout on/off). The
    runnable VERIFIED oracle is a fresh instance IN THE CAPTURED MODE on the given inputs,
    so the captured mode is DECLARED state the replay reproduces. Emitting it as a single
    ``SHAPE_STRUCTURE_FACT`` witness lets the executor anchor VERIFIED to the recorded mode;
    a mode-sensitive op replayed without this fact is downgraded to UNVERIFIABLE (fail
    closed). No tensors are recorded.
    """

    modes = getattr(trace, "__dict__", {}).get("_runnable_module_training_modes", None)
    if not isinstance(modes, Mapping) or not modes:
        return []
    fact = {
        MODULE_TRAINING_MODE_FACT_KEY: True,
        "modes": {str(address): bool(training) for address, training in modes.items()},
    }
    try:
        observed = _encode_literal(fact)
    except _UnsupportedLiteralError:
        return []
    return [
        ControlWitness(
            witness_id=f"witness:{start_order + 1}",
            kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
            order=start_order,
            call_id=None,
            site_label=MODULE_TRAINING_MODE_SITE_PREFIX,
            observed_value=observed,
        )
    ]


MODEL_INPUT_METADATA_SITE_PREFIX = "model_input_metadata:"
"""``site_label`` prefix marking a witnessed model-input metadata-predicate read."""

MODEL_INPUT_METADATA_FACT_KEY = "model_input_metadata"
"""Discriminator key present in every model-input metadata-predicate fact."""

_INPUT_METADATA_FACT_NAMES = frozenset(
    {
        "is_contiguous",
        "stride",
        "storage_offset",
        "requires_grad",
        "grad_fn",
        "is_leaf",
        "storage_nbytes",
        "retains_grad",
        "_base",
        "_is_view",
        "is_conj",
        "is_neg",
        "is_inference",
        "is_pinned",
        "is_shared",
        "is_coalesced",
        "grad",
        "_grad",
        "_version",
        "output_nr",
    }
)
"""Metadata predicates the capture-time observer records for model-input receivers (r27-H2,
extended r29-C1 with ``storage_offset`` / ``grad_fn`` / ``is_leaf`` / ``storage_nbytes``; r31
adds the capability-driven surface ``retains_grad`` / ``_base`` / ``_is_view`` / ``is_conj`` /
``is_neg`` / ``is_inference`` / ``is_pinned`` / ``is_shared`` / ``is_coalesced``; r33 adds
``grad`` / ``_grad`` presence + ``_version`` / ``output_nr`` int facts)."""


def _input_metadata_witnesses(
    trace: Any,
    *,
    start_order: int,
) -> list[ControlWitness]:
    """Witness capture-time metadata-predicate reads on model-input leaves (r27-H2).

    A forward that reads ``x.is_contiguous()`` / ``x.stride()`` / ``x.requires_grad`` on a
    model input steers Python control flow on facts the input contract does NOT check
    (only shape+dtype): a same-shape runtime input differing in layout or grad flag would
    silently replay the wrong recorded arm as a false VERIFIED+ATTESTED. Each observed
    (site, predicate, value) fact -- recorded by the completeness-witness scoped patch
    ONLY when such a read actually happened -- becomes a ``SHAPE_STRUCTURE_FACT`` witness
    the executor compares against the RAW runtime input (before the detach-clone that
    erases ``requires_grad``), diverging on mismatch. A model that never reads input
    metadata records no facts and emits no witnesses: zero over-trigger by construction.
    """

    reads = getattr(trace, "__dict__", {}).get("_runnable_input_metadata_reads", None)
    if not isinstance(reads, Mapping) or not reads:
        return []
    witnesses: list[ControlWitness] = []
    for (position, path), site_facts in reads.items():
        if not isinstance(site_facts, Mapping):
            continue
        recorded = {
            str(name): (list(value) if isinstance(value, tuple) else value)
            for name, value in site_facts.items()
            if str(name) in _INPUT_METADATA_FACT_NAMES
        }
        if not recorded:
            continue
        fact = {
            MODEL_INPUT_METADATA_FACT_KEY: True,
            "position": list(position) if isinstance(position, tuple) else position,
            "path": list(path),
            "facts": recorded,
        }
        try:
            observed = _encode_literal(fact)
        except _UnsupportedLiteralError:
            # Defensive: a fact site under a non-encodable container key cannot be
            # re-resolved at run time. The literal-leaf walker independently records
            # such a subtree as an OPAQUE leaf, downgrading the run to UNVERIFIABLE,
            # so dropping the witness here can never produce a false VERIFIED.
            continue
        order = start_order + len(witnesses)
        witnesses.append(
            ControlWitness(
                witness_id=f"witness:{order + 1}",
                kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
                order=order,
                call_id=None,
                site_label=f"{MODEL_INPUT_METADATA_SITE_PREFIX}{position!r}:{list(path)!r}",
                observed_value=observed,
            )
        )
    return witnesses


def _input_literal_witnesses(
    trace: Any,
    *,
    start_order: int,
) -> tuple[list[ControlWitness], bool]:
    """Witness capture-time non-tensor model-input leaves as structure facts.

    The runnable executor binds only tensor input leaves; a changed non-tensor
    Python input (bool/int/float/str/None) can silently steer control flow that
    was never captured as an op, making the recorded taken path wrong. Each
    grammar-encodable non-tensor leaf is recorded as a ``SHAPE_STRUCTURE_FACT``
    witness carrying its site position, container path, and literal value so the
    executor can diverge on a changed non-tensor input rather than falsely
    reporting a verified, attested result.

    A leaf *outside* the frozen literal grammar (enum, dataclass, set, bytes,
    complex, numpy scalar, ...) cannot be
    compared across save/load, so its value is recorded ``None`` -- a value-free
    fact. Such a leaf can still steer unobserved Python control flow, and because
    the executor cannot re-verify it, the run's witness coverage is genuinely
    INCOMPLETE: the caller must downgrade ``witness_completeness`` so the run
    reports ``UNVERIFIABLE`` + ``NOT_APPLICABLE`` instead of a false
    ``VERIFIED``/``ATTESTED`` over a possibly-wrong replayed path. This function
    signals that condition by returning ``saw_opaque_leaf=True`` so the caller can
    downgrade ``witness_completeness``. It does *not* fail producer preflight: an
    opaque-leaf artifact still saves and runs, but honestly reports
    ``UNVERIFIABLE`` rather than a false ``VERIFIED``. No tensors are recorded.

    Parameters
    ----------
    trace:
        Cooked Trace carrying the capture-time non-tensor input leaf stash.
    start_order:
        First dense witness order to assign, continuing the existing sequence.

    Returns
    -------
    tuple[list[ControlWitness], bool]
        Ordered non-tensor model-input leaf witnesses, and whether any leaf was
        value-free (opaque), signalling incomplete witness coverage.
    """

    leaves = getattr(trace, "__dict__", {}).get("_runnable_input_nontensor_leaves", ())
    witnesses: list[ControlWitness] = []
    saw_opaque_leaf = False
    for position, path, value in leaves:
        encodable = _is_encodable_model_input_leaf(value)
        if not encodable:
            saw_opaque_leaf = True
        fact = {
            MODEL_INPUT_LITERAL_FACT_KEY: True,
            "position": list(position) if isinstance(position, tuple) else position,
            "path": list(path),
            "encodable": encodable,
            "value": value if encodable else None,
        }
        try:
            observed = _encode_literal(fact)
        except _UnsupportedLiteralError:
            # Defensive: an exotic path/position component that cannot be encoded
            # cannot be witnessed. Tensor-slot binding still constrains arity.
            continue
        order = start_order + len(witnesses)
        witnesses.append(
            ControlWitness(
                witness_id=f"witness:{order + 1}",
                kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
                order=order,
                call_id=None,
                site_label=f"{MODEL_INPUT_LITERAL_SITE_PREFIX}{position!r}:{list(path)!r}",
                observed_value=observed,
            )
        )
    return witnesses, saw_opaque_leaf


def _is_encodable_model_input_leaf(value: Any) -> bool:
    """Return whether a non-tensor model-input leaf is runtime-comparable.

    Non-finite Python floats are intentionally outside the comparable input-leaf
    subset even though the literal grammar can serialize them for call recipes.
    A ``nan``/``inf`` input leaf cannot honestly support value attestation, so it
    must be recorded as opaque and force incomplete witness coverage.
    """

    if isinstance(value, float) and not math.isfinite(value):
        return False
    try:
        _encode_literal(value)
    except _UnsupportedLiteralError:
        return False
    return True


def _preflight_output_contracts(trace: Any, ops: Sequence[Any]) -> list[RunnableDiagnostic]:
    """Report structured model outputs whose container contract is unavailable."""

    diagnostics: list[RunnableDiagnostic] = []
    output_ops = [op for op in ops if bool(getattr(op, "is_output", False))]
    # r35 I1 (decision B, subsumes r33 R32-B1): a runnable save requires a POSITIVE
    # capture-time losslessness proof of the model output -- exact root kind,
    # recursively supported children, encodable literal leaves, and a tensor-leaf/
    # typed-path bijection. Refuse-unless-proved: an absent or failed proof refuses
    # the runnable save uniformly (bare/nested/one-tensor/empty sets, frozensets and
    # subclasses, opaque tensor holders, duplicate paths, BFS fallback), never a
    # save-then-UNVERIFIABLE landmine and never an advertise-then-crash artifact.
    losslessness = getattr(trace, "__dict__", {}).get("_runnable_output_losslessness")
    if not isinstance(losslessness, Mapping) or not losslessness.get("lossless", False):
        reason = (
            str(losslessness.get("reason", "unknown"))
            if isinstance(losslessness, Mapping)
            else "losslessness_not_proven"
        )
        root_type = (
            str(losslessness.get("root_type", "unknown"))
            if isinstance(losslessness, Mapping)
            else "unknown"
        )
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT,
                "Model output is not provably lossless for runnable replay "
                f"(root {root_type!r}, reason {reason!r}); runnable replay could "
                "silently drop elements or mis-report the container kind, so the "
                "save is refused. Ordinary analysis save levels remain available.",
                affected_ops=tuple(str(op.label) for op in output_ops),
                detection_stage="producer_output_binding",
                details=(("reason", reason), ("root_type", root_type)),
            )
        )
        return diagnostics
    containers = getattr(trace, "__dict__", {}).get("_containers", {}) or {}
    model_output_snapshots = [
        snapshot
        for record in containers.values()
        for snapshot in getattr(record, "snapshots", ())
        if getattr(snapshot, "role", None) is Role.MODEL_OUTPUT
    ]
    # A recorded model-output container that is non-reconstructable (opaque custom
    # Mapping, unsafe defaultdict, unknown dict subclass, or an unrepresentable
    # non-tensor leaf) must NOT advertise runnable regardless of how many tensors it
    # holds: otherwise the loaded run silently returns a bare tensor / plain dict and
    # reports VERIFIED. Honest-reject at save closes that class.
    if any(
        not getattr(snapshot, "reconstructable", True)
        or getattr(getattr(snapshot, "spec", None), "kind", None) == "opaque"
        for snapshot in model_output_snapshots
    ):
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT,
                "Model output is a non-reconstructable container; runnable replay cannot "
                "restore its exact type and non-tensor leaves.",
                affected_ops=tuple(str(op.label) for op in output_ops),
                detection_stage="producer_output_binding",
            )
        )
        return diagnostics
    if len(output_ops) <= 1:
        return diagnostics
    if any(getattr(op, "container_path", None) for op in output_ops):
        return diagnostics
    has_model_output = bool(model_output_snapshots)
    if not has_model_output:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT,
                "Multi-tensor model output has no cooked container contract.",
                affected_ops=tuple(str(op.label) for op in output_ops),
                detection_stage="producer_output_binding",
            )
        )
    return diagnostics


def _mark_inplace_versions(
    calls: Sequence[RunnableCallDescriptor], slot_drafts: Mapping[str, _SlotDraft]
) -> None:
    """Attach version relations for in-place call outputs."""

    for call in calls:
        version_of = _mutation_target_slot_id(call)
        if not call.is_inplace or version_of is None:
            continue
        for output_slot_id in call.output_slot_ids:
            draft = slot_drafts.get(output_slot_id)
            if draft is not None:
                draft.mutable = True
                draft.version_of = version_of


def _mutation_target_slot_id(call: RunnableCallDescriptor) -> str | None:
    """Return the tensor slot mutated by one recorded call.

    Parameters
    ----------
    call:
        Frozen runnable call descriptor.

    Returns
    -------
    str | None
        The ``out=`` tensor slot when present, otherwise the first tensor
        argument for conventional in-place operators.
    """

    for argument in call.tensor_arguments:
        if argument.argument_path == ("kwargs", "out"):
            return argument.slot_id
    return call.tensor_arguments[0].slot_id if call.tensor_arguments else None


def _buffer_binding(trace: Any, op: Any) -> StateSlotBinding | None:
    """Build a named buffer binding from a cooked source op.

    Persistence is defined by canonical ``state_dict`` membership. Registered
    buffer membership is captured on the trace while the source model is alive,
    so save-time classification never depends on weak-reference or GC state.
    Buffers excluded with ``persistent=False`` retain a buffer binding for replay,
    but never claim or require a canonical state-dict key.
    """

    address = getattr(op, "address", None)
    if not isinstance(address, str) or not address:
        return None
    persistence = getattr(trace, "_buffer_persistence", {}) or {}
    persistent = bool(persistence.get(address, False))
    module_path, _, name = address.rpartition(".")
    return StateSlotBinding(
        module_path=module_path or "self",
        state_dict_name=address,
        semantic_role=_buffer_role(name or address),
        trainable=False,
        persistent=persistent,
        alias_group=None,
    )


def _parameter_role(param: Any) -> StateSlotRole:
    """Classify one cooked parameter into the frozen initializer role table."""

    name = str(getattr(param, "name", ""))
    module_address = str(getattr(param, "module_address", ""))
    trace = getattr(param, "source_trace", None)
    modules = getattr(trace, "modules", {}) if trace is not None else {}
    module = modules.get(module_address) if isinstance(modules, Mapping) else None
    class_name = str(getattr(module, "class_name", module_address)).lower()
    is_norm = "norm" in class_name
    if name == "weight":
        return StateSlotRole.NORM_SCALE if is_norm else StateSlotRole.WEIGHT
    if name == "bias":
        return StateSlotRole.NORM_OFFSET if is_norm else StateSlotRole.BIAS
    return StateSlotRole.WEIGHT


def _buffer_role(name: str) -> StateSlotRole:
    """Classify a named buffer into the frozen state role vocabulary."""

    if name == "running_mean":
        return StateSlotRole.RUNNING_MEAN
    if name in {"running_var", "running_variance"}:
        return StateSlotRole.RUNNING_VAR
    if name in {"num_batches_tracked", "counter", "step"}:
        return StateSlotRole.COUNTER
    return StateSlotRole.GENERIC_BUFFER


def _op_slot_role(op: Any) -> TensorSlotRole:
    """Classify one cooked op tensor into the frozen slot role vocabulary."""

    if bool(getattr(op, "is_input", False)):
        return TensorSlotRole.MODEL_INPUT
    if bool(getattr(op, "is_buffer", False)):
        return TensorSlotRole.BUFFER
    if bool(getattr(op, "is_output", False)):
        return TensorSlotRole.OUTPUT
    if str(getattr(op, "func_name", "")) in {
        "rand",
        "randn",
        "randint",
        "rand_like",
        "randn_like",
        "bernoulli",
    }:
        return TensorSlotRole.RNG_SOURCE
    return TensorSlotRole.INTERMEDIATE


def _component_contains_tensor(component: Any) -> bool:
    """Return whether a captured component contains a tensor-valued leaf."""

    if isinstance(component, (ParentRef, LiteralTensor)):
        return True
    if isinstance(component, tuple):
        return any(_component_contains_tensor(item) for item in component)
    if isinstance(component, list):
        return any(_component_contains_tensor(item) for item in component)
    if isinstance(component, Mapping):
        return any(_component_contains_tensor(item) for item in component.values())
    return False


def _tensor_container_skeleton(component: Any) -> NonTensorLiteral:
    """Encode a mutable sparse-call container shell without tensor values.

    Tensor leaves are represented by ``None`` placeholders and overwritten
    with ``ParentRef`` slots during sparse execution. Captured tensor
    containers are projected as tuples by the eager capture layer, so their
    runnable shell intentionally uses a list, which is accepted by the
    relevant variadic torch operators and supports leaf replacement.

    Parameters
    ----------
    component:
        Captured argument component containing at least one tensor leaf.

    Returns
    -------
    NonTensorLiteral
        Value-free literal shell used to reconstruct the argument tree.
    """

    if isinstance(component, (ParentRef, LiteralTensor)):
        return LiteralAtom(LiteralAtomKind.NONE, None)
    if isinstance(component, LiteralValue):
        return _encode_literal(component.value)
    if isinstance(component, Unsupported):
        raise _UnsupportedLiteralError(component.reason)
    if isinstance(component, (list, tuple)):
        return LiteralSequence(
            LiteralSequenceKind.LIST,
            tuple(_tensor_container_skeleton(item) for item in component),
        )
    if isinstance(component, Mapping):
        return LiteralMapping(
            tuple(
                LiteralMappingEntry(
                    _encode_literal_key(key),
                    _tensor_container_skeleton(item),
                )
                for key, item in component.items()
            )
        )
    return _encode_literal(component)


def _encode_literal(value: Any) -> NonTensorLiteral:
    """Encode a Python value using only the frozen safe literal grammar."""

    if value is None:
        return LiteralAtom(LiteralAtomKind.NONE, None)
    if isinstance(value, bool):
        return LiteralAtom(LiteralAtomKind.BOOL, value)
    if isinstance(value, int):
        # Normalize integer subclasses (e.g. ``IntEnum``) to a plain ``int`` so
        # the stored literal round-trips through JSON / the safe unpickler.
        return LiteralAtom(LiteralAtomKind.INT, int(value))
    if isinstance(value, float):
        # Finiteness is a pure host check on a Python float; use ``math.isfinite`` rather
        # than ``torch.isfinite(torch.tensor(value)).item()`` so encoding a float literal
        # key/leaf during capture emits NO ``aten._local_scalar_dense`` dispatch. That
        # internal read would otherwise be recorded by the host-escape witness as a
        # spurious (bool) user escape and falsely downgrade an exotic-key model (e.g. a
        # ``dict[float, int]`` branch) to UNVERIFIABLE on the unchanged input.
        if not math.isfinite(value):
            return LiteralAtom(LiteralAtomKind.NONFINITE_FLOAT, _nonfinite_float_payload(value))
        # Normalize float subclasses (e.g. ``numpy.float64``) to a plain
        # ``float`` so the recorded literal round-trips to a grammar-native value
        # the safe metadata unpickler admits and value-equality can verify.
        return LiteralAtom(LiteralAtomKind.FLOAT, float(value))
    if isinstance(value, str):
        return LiteralAtom(LiteralAtomKind.STR, value)
    if value is Ellipsis:
        return LiteralAtom(LiteralAtomKind.ELLIPSIS, None)
    if isinstance(value, slice):
        return LiteralSlice(
            start=_encode_slice_component(value.start, "start"),
            stop=_encode_slice_component(value.stop, "stop"),
            step=_encode_slice_component(value.step, "step"),
        )
    torch_symbol = _torch_symbol_qualname(value)
    if torch_symbol is not None:
        return LiteralTorchSymbol(torch_symbol)
    if isinstance(value, list):
        return LiteralSequence(
            LiteralSequenceKind.LIST,
            tuple(_encode_literal(item) for item in value),
        )
    if isinstance(value, tuple):
        return LiteralSequence(
            LiteralSequenceKind.TUPLE,
            tuple(_encode_literal(item) for item in value),
        )
    if isinstance(value, Mapping):
        return LiteralMapping(
            tuple(
                LiteralMappingEntry(_encode_literal_key(key), _encode_literal(item))
                for key, item in value.items()
            )
        )
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    raise _UnsupportedLiteralError(
        f"Value of type {value_type} is outside the frozen non-tensor literal grammar."
    )


def _nonfinite_float_payload(value: float) -> str:
    """Return the stable string payload for a non-finite float literal.

    Parameters
    ----------
    value:
        Python float known to be non-finite.

    Returns
    -------
    str
        One of ``"nan"``, ``"inf"``, or ``"-inf"``.
    """

    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _encode_slice_component(value: Any, field_name: str) -> LiteralAtom:
    """Encode one ``slice.start``/``.stop``/``.step`` component.

    A Python ``slice`` component is always ``None`` or an integer (indices with
    ``__index__`` are normalized to ``int`` by the interpreter before the ``slice``
    object is constructed), so the encoded shape is restricted to those two atom
    kinds -- no callables, no arbitrary objects.
    """

    if value is None:
        return LiteralAtom(LiteralAtomKind.NONE, None)
    if isinstance(value, bool) or not isinstance(value, int):
        value_type = f"{type(value).__module__}.{type(value).__qualname__}"
        raise _UnsupportedLiteralError(
            f"Slice component {field_name!r} of type {value_type} is outside the frozen "
            "non-tensor literal grammar; only int or None slice components are supported."
        )
    return LiteralAtom(LiteralAtomKind.INT, int(value))


EMPTY_CONTAINER_PATH_MARKER = "\x00tl_empty_container"
"""Reserved terminal path component marking an EMPTY non-tensor input container (r29-C2).

An empty dict/list/tuple contributes NO leaf path, so the non-tensor leaf-path SET witness
(H1) is blind to an EXTRA empty container a model branches on (``if not d.get('flag', {})``).
Both the capture and runtime walks emit a synthetic leaf at ``(*container_path, MARKER)``
carrying the container KIND string, so an added/removed/kind-changed empty container diverges
the run. The null-byte prefix makes collision with a real string dict key effectively
impossible.
"""

BOOL_KEY_PATH_TAG = "\x00tl_bool_key"
"""Reserved tag distinguishing a BOOL mapping key from the equal-valued int (r29-C2, F6).

``bool`` is a subclass of ``int`` and ``hash(True) == hash(1)``, so a raw ``(True,)`` path
component compares equal to ``(1,)`` in the leaf-path set and ``_value_at_path`` resolves both
against either key. Encoding a bool key as ``(BOOL_KEY_PATH_TAG, bool(key))`` keeps the key
TYPE distinct across capture/runtime, matching the type-strict ``_literal_leaf_equal`` used for
values.
"""


def input_path_key_component(key: Any) -> Any:
    """Return a type-strict non-tensor path component for one mapping key (r29-C2, F6)."""

    if isinstance(key, bool):
        return (BOOL_KEY_PATH_TAG, bool(key))
    return key


def empty_container_kind(value: Any) -> str | None:
    """Return the KIND string of an EMPTY non-tensor container, else ``None`` (r29-C2).

    ``None`` for non-containers and for NON-empty containers (whose leaves are witnessed
    ordinarily). Namedtuples are treated as sequences by field arity; an empty namedtuple has
    no fields.
    """

    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return "namedtuple" if len(value._fields) == 0 else None
    if isinstance(value, Mapping):
        return "mapping" if len(value) == 0 else None
    if isinstance(value, (list, tuple)):
        return "sequence" if len(value) == 0 else None
    return None


def _encode_literal_key(value: Any) -> LiteralAtom | LiteralTupleKey:
    """Encode a mapping key using the frozen safe key subset."""

    encoded = _encode_literal(value)
    if isinstance(encoded, LiteralAtom):
        return encoded
    if isinstance(encoded, LiteralSequence) and encoded.kind is LiteralSequenceKind.TUPLE:
        items: list[LiteralAtom | LiteralTupleKey] = []
        for item in encoded.items:
            if isinstance(item, LiteralAtom):
                items.append(item)
            elif isinstance(item, LiteralSequence) and item.kind is LiteralSequenceKind.TUPLE:
                items.append(_encode_literal_key(_literal_sequence_to_python(item)))
            else:
                raise _UnsupportedLiteralError("Mapping tuple keys may contain only scalar atoms.")
        return LiteralTupleKey(tuple(items))
    raise _UnsupportedLiteralError("Mapping keys must be scalar atoms or safe tuples.")


def _literal_sequence_to_python(value: LiteralSequence) -> tuple[Any, ...]:
    """Convert an encoded scalar tuple key back to a Python tuple for recursion."""

    result: list[Any] = []
    for item in value.items:
        if isinstance(item, LiteralAtom):
            result.append(item.value)
        elif isinstance(item, LiteralSequence):
            result.append(_literal_sequence_to_python(item))
        else:
            raise _UnsupportedLiteralError("Mapping tuple keys may contain only scalar atoms.")
    return tuple(result)


def _torch_symbol_qualname(value: Any) -> str | None:
    """Return an allowlisted torch symbolic name for a non-callable value."""

    if isinstance(value, torch.device):
        return f"torch.device({value})"
    for name, candidate in vars(torch).items():
        if callable(candidate):
            continue
        if candidate is value and isinstance(
            value, (torch.dtype, torch.layout, torch.memory_format)
        ):
            return f"torch.{name}"
    return None


def _runtime_fingerprint(
    op: Any,
    func_id: FunctionRegistryKey,
    call_ops: Sequence[Any],
    execution_context: CallExecutionContext,
) -> str:
    """Hash the recorded runtime call signature without tensor values.

    The canonical serialized per-call execution context participates in the
    fingerprint (v2): a context change is a signature-relevant replay fact.
    """

    payload = {
        "callable": {
            "namespace": func_id.namespace,
            "qualname": func_id.qualname,
            "dispatch_kind": func_id.dispatch_kind,
            "version": func_id.version,
            "import_path": func_id.import_path,
        },
        "argument_names": list(getattr(op, "arg_names", ()) or ()),
        "num_positional_args": int(getattr(op, "num_pos_args", 0)),
        "num_keyword_args": int(getattr(op, "num_kwargs", 0)),
        "outputs": [
            {"shape": list(_shape_tuple(item.shape) or ()), "dtype": _dtype_name(item)}
            for item in call_ops
        ],
        "execution_context": {
            "autocast": [
                {
                    "device_type": entry.device_type,
                    "enabled": entry.enabled,
                    "dtype": entry.dtype,
                }
                for entry in execution_context.autocast
            ],
            "grad_enabled": execution_context.grad_enabled,
            "inference_mode": execution_context.inference_mode,
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _diagnostic(
    code: RunnableErrorCode,
    message: str,
    *,
    registry_id: str | None = None,
    affected_ops: tuple[str, ...] = (),
    detection_stage: str,
    details: tuple[tuple[str, str], ...] = (),
) -> RunnableDiagnostic:
    """Construct one structured producer diagnostic."""

    return RunnableDiagnostic(
        code=code,
        message=message,
        registry_id=registry_id,
        affected_op_labels=affected_ops,
        recorded_runtime=str(torch.__version__),
        current_runtime=str(torch.__version__),
        detection_stage=detection_stage,
        resolver_provenance=None,
        analysis_load_available=True,
        details=details,
    )


def _deduplicate_diagnostics(
    diagnostics: Iterable[RunnableDiagnostic],
) -> list[RunnableDiagnostic]:
    """Deduplicate diagnostics while retaining deterministic first occurrence."""

    result: list[RunnableDiagnostic] = []
    seen: set[tuple[Any, ...]] = set()
    for diagnostic in diagnostics:
        key = (
            diagnostic.code,
            diagnostic.registry_id,
            diagnostic.affected_op_labels,
            diagnostic.detection_stage,
            diagnostic.details,
        )
        if key not in seen:
            seen.add(key)
            result.append(diagnostic)
    return result


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    """Normalize shape metadata to a tuple of integers."""

    if value is None:
        return None
    try:
        return tuple(int(dim) for dim in value)
    except (TypeError, ValueError):
        return None


def _dtype_name(op: Any) -> str | None:
    """Return canonical dtype metadata for one cooked op."""

    dtype_ref = getattr(op, "dtype_ref", None)
    if dtype_ref is not None:
        return str(dtype_ref)
    dtype = getattr(op, "dtype", None)
    return None if dtype is None else str(dtype)


def _device_parts(value: Any) -> tuple[str, int | None]:
    """Split cooked device metadata into type and optional index."""

    text = "cpu" if value is None else str(value)
    device = torch.device(text)
    return device.type, device.index


def _safe_normalize_container_path(path: Iterable[Any]) -> tuple[str | int, ...] | None:
    """Return the frozen container path, or ``None`` when a key is unrepresentable.

    Used where an unrepresentable output-container path (non-str/int key) must
    degrade gracefully -- the enclosing container is recorded opaque and rejected
    at preflight, so its leaf paths are advisory and must never crash descriptor
    build with a raw ``ValueError``.
    """

    try:
        return _normalize_container_path(path)
    except ValueError:
        return None


def _normalize_container_path(path: Iterable[Any]) -> tuple[str | int, ...]:
    """Convert cooked container path components to frozen string/int paths."""

    normalized: list[str | int] = []
    for component in path:
        if isinstance(component, TupleIndex):
            normalized.append(component.index)
        elif isinstance(component, (DictKey, HFKey)):
            if not isinstance(component.key, (str, int)):
                raise ValueError("Runnable container paths require string or integer keys.")
            normalized.append(component.key)
        elif isinstance(component, (NamedField, DataclassField)):
            normalized.append(component.name)
        elif isinstance(component, (str, int)):
            normalized.append(component)
        else:
            raise ValueError(f"Unsupported runnable container path component {component!r}.")
    return tuple(normalized)


def _normalize_model_site_position(value: Any) -> str | int | tuple[str | int, ...] | None:
    """Normalize a captured ModelSite position to the frozen path vocabulary."""

    if isinstance(value, (str, int)):
        return value
    if isinstance(value, tuple) and all(isinstance(item, (str, int)) for item in value):
        return value
    return None


def _json_value(value: Any) -> Any:
    """Recursively convert dataclasses, enums, tuples, and mappings to JSON values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Sparse descriptor contains non-JSON value {type(value).__qualname__}.")


from .runnable_load import (  # noqa: E402 - keep producer helpers grouped above
    attach_sparse_run_readiness,
    parse_sparse_run_descriptor,
    preflight_sparse_run_descriptor,
)


__all__ = [
    "assert_sparse_core_has_no_tensor_payload",
    "attach_sparse_run_readiness",
    "build_sparse_run_descriptor",
    "parse_sparse_run_descriptor",
    "preflight_sparse_run_descriptor",
    "require_sparse_run_descriptor",
    "sparse_descriptor_to_json",
    "with_activation_payload",
    "with_weight_payload",
]
