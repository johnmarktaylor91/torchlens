"""Safe sparse runnable descriptor loading and torch callable reattachment.

This module performs readiness preflight only. It never imports artifact-selected
modules, binds state, constructs runtime calls, or executes a recorded graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import inspect
import operator
import platform
from typing import Any, cast

import torch

from .. import _state
from ..constants import get_orig_torch_funcs
from ..intervention.types import FunctionRegistryKey
from ..runnable import (
    LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS,
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
    ReadinessReport,
    ReadinessStatus,
    ResolverRecord,
    ResolverStatus,
    RunProvider,
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
    StateSource,
    TensorArgumentRef,
    TensorSlotDescriptor,
    TensorSlotRole,
    TensorUseSite,
    WitnessCompleteness,
)
from ..utils._callable_safety import is_pure_forward_callable, unsafe_callable_reason
from ..utils._torch_compat import resolve_runnable_torch_alias


_ALLOWED_EXACT_ROOTS: Mapping[str, Any] = {
    "torch": torch,
    "torch.Tensor": torch.Tensor,
    "torch.nn.functional": torch.nn.functional,
    "operator": operator,
}
_ENUMERATED_TORCH_NAMESPACES = frozenset(
    {
        "torch.fft",
        "torch.linalg",
        "torch.nested",
        "torch.special",
        "torch.nn.functional",
        "torch.Tensor",
        "torch._C._nn",
        "torch._C._special",
        "torch._C._VariableFunctions",
        "torch._C._VariableFunctionsClass",
        "torch._C._TensorBase",
        "torch._C.TensorBase",
        "torch._VF",
    }
)
_REMOVED_TORCH_CALLABLES = frozenset({"torch.gesv"})
_SAFE_TENSOR_PROPERTY_NAMES = frozenset({"T", "mT", "real", "imag"})


@dataclass(frozen=True, slots=True)
class _ReverseCandidate:
    """One reverse-index callable candidate and its stable display path."""

    namespace: str
    qualname: str
    func: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class _Resolution:
    """Internal result of resolving one callable registry entry."""

    record: ResolverRecord
    func: Callable[..., Any] | None


def parse_sparse_run_descriptor(value: Mapping[str, Any]) -> SparseRunDescriptor:
    """Parse a schema-validated manifest run object into frozen descriptor types.

    Parameters
    ----------
    value:
        Decoded ``manifest.run`` mapping after structural manifest validation.

    Returns
    -------
    SparseRunDescriptor
        Typed sparse descriptor. Version values are retained verbatim so the
        readiness preflight can report unsupported ceilings without executing.
    """

    registry = tuple(
        CallableRegistryEntry(
            registry_id=_string(entry, "registry_id"),
            key=_parse_registry_key(_mapping(entry, "key")),
        )
        for entry in _mapping_sequence(value, "callable_registry")
    )
    calls = tuple(_parse_call(item) for item in _mapping_sequence(value, "calls"))
    slots = tuple(_parse_slot(item) for item in _mapping_sequence(value, "tensor_slots"))
    witnesses = tuple(
        _parse_witness(item) for item in _mapping_sequence(value, "control_witnesses")
    )
    payload = _mapping(value, "payload_layers")
    compatibility = _mapping(value, "compatibility")
    preflight = _mapping(value, "preflight")
    # v2: the capture-scoped ambient execution context is REQUIRED and EXPLICIT.
    # Absence is a parse failure (fail closed), never a defaulted context.
    ambient = _parse_ambient_context(_mapping(value, "ambient_context"))
    return SparseRunDescriptor(
        capability=cast(Any, _string(value, "capability")),
        backend=_string(value, "backend"),
        call_recipe=cast(Any, _string(value, "call_recipe")),
        callable_ref_schema=cast(Any, _integer(value, "callable_ref_schema")),
        state_binding=cast(Any, _string(value, "state_binding")),
        input_binding=cast(Any, _string(value, "input_binding")),
        control_witness=cast(Any, _string(value, "control_witness")),
        initializer_policy_version=cast(Any, _string(value, "initializer_policy_version")),
        payload_layers=PayloadLayersDescriptor(
            weights=_parse_payload_layer(_mapping(payload, "weights")),
            nonpersistent_buffers=_parse_nonpersistent_buffer_payload_layer(payload),
            activations=_parse_activation_payload_layer(_mapping(payload, "activations")),
        ),
        callable_registry=registry,
        calls=calls,
        tensor_slots=slots,
        control_witnesses=witnesses,
        witness_completeness=WitnessCompleteness(_string(value, "witness_completeness")),
        rng_profile=_parse_rng_profile(value.get("rng_profile")),
        ambient_context=ambient,
        compatibility=RunnableCompatibility(
            torchlens_version=_string(compatibility, "torchlens_version"),
            python_version=_string(compatibility, "python_version"),
            backend_version=_string(compatibility, "backend_version"),
            descriptor_version=_string(compatibility, "descriptor_version"),
            call_recipe_version=_string(compatibility, "call_recipe_version"),
            callable_ref_schema_version=_integer(compatibility, "callable_ref_schema_version"),
            initializer_policy_version=_string(compatibility, "initializer_policy_version"),
        ),
        preflight=ProducerPreflight(
            passed=_boolean(preflight, "passed"),
            diagnostics=tuple(
                _parse_diagnostic(item) for item in _mapping_sequence(preflight, "diagnostics")
            ),
        ),
        unsupported_sites=tuple(
            _parse_diagnostic(item) for item in _mapping_sequence(value, "unsupported_sites")
        ),
    )


class ContextFieldInvalidError(ValueError):
    """A persisted execution-context field failed closed-vocabulary validation (INV-4).

    Raised at PARSE time -- before readiness, staging, or any torch setter/callable
    can observe the attacker-controllable bytes -- and surfaced as the frozen
    ``context_field_invalid`` readiness diagnostic.
    """

    def __init__(self, field: str, detail: str) -> None:
        super().__init__(f"Persisted execution-context field {field!r} is invalid: {detail}")
        self.field = field
        self.detail = detail


_MATMUL_PRECISION_VOCABULARY = frozenset({"highest", "high", "medium"})
"""Closed vocabulary for ``float32_matmul_precision`` (torch's exact accepted set)."""

_DEVICE_TYPE_VOCABULARY = frozenset(
    {
        "cpu",
        "cuda",
        "meta",
        "mps",
        "xpu",
        "hpu",
        "mtia",
        "ipu",
        "xla",
        "vulkan",
        "lazy",
        "privateuseone",
    }
)
"""Closed vocabulary of device TYPE literals a persisted context may name."""


def _validated_device_literal(field: str, raw: str) -> str:
    """Validate a persisted device literal against the closed type[:index] grammar."""

    device_type, separator, index = raw.partition(":")
    if device_type not in _DEVICE_TYPE_VOCABULARY:
        raise ContextFieldInvalidError(
            field, f"device type {device_type!r} is outside the closed device vocabulary"
        )
    if separator and not index.isdigit():
        raise ContextFieldInvalidError(
            field, f"device index {index!r} is not a nonnegative integer"
        )
    return raw


def _validated_dtype_literal(field: str, raw: str) -> str:
    """Validate a persisted ``torch.<dtype>`` literal against the live dtype table."""

    name = raw.removeprefix("torch.")
    resolved = getattr(torch, name, None)
    if not isinstance(resolved, torch.dtype):
        raise ContextFieldInvalidError(field, f"{raw!r} does not name a torch dtype")
    return raw


def _parse_ambient_context(value: Mapping[str, Any]) -> AmbientExecutionContext:
    """Parse the required v2 capture-scoped ambient execution context.

    Every field must be present explicitly; ``None`` means the producing runtime
    did not expose that control. Missing keys raise (fail closed). r37 INV-4:
    every VALUE validates here against a closed vocabulary -- device literals
    against the type[:index] grammar, dtype literals against the live dtype
    table, matmul precision against torch's exact accepted set, Booleans
    strictly -- so no torch setter (and no callable) ever receives an
    unvalidated persisted byte. The ambient-apply guards remain as a second
    belt.
    """

    def _optional_bool_field(name: str) -> bool | None:
        raw = value[name]
        if raw is None:
            return None
        if not isinstance(raw, bool):
            raise ContextFieldInvalidError(
                f"ambient_context.{name}", f"{raw!r} is not a strict boolean or null"
            )
        return raw

    def _optional_str_field(name: str) -> str | None:
        raw = value[name]
        if raw is None:
            return None
        return _string_item(raw, f"ambient_context.{name}")

    precision = _optional_str_field("float32_matmul_precision")
    if precision is not None and precision not in _MATMUL_PRECISION_VOCABULARY:
        raise ContextFieldInvalidError(
            "ambient_context.float32_matmul_precision",
            f"{precision!r} is outside {sorted(_MATMUL_PRECISION_VOCABULARY)}",
        )
    return AmbientExecutionContext(
        default_dtype=_validated_dtype_literal(
            "ambient_context.default_dtype", _string(value, "default_dtype")
        ),
        default_device=_validated_device_literal(
            "ambient_context.default_device", _string(value, "default_device")
        ),
        float32_matmul_precision=precision,
        deterministic_algorithms=_optional_bool_field("deterministic_algorithms"),
        deterministic_algorithms_warn_only=_optional_bool_field(
            "deterministic_algorithms_warn_only"
        ),
        cuda_matmul_allow_tf32=_optional_bool_field("cuda_matmul_allow_tf32"),
        cudnn_allow_tf32=_optional_bool_field("cudnn_allow_tf32"),
        cudnn_deterministic=_optional_bool_field("cudnn_deterministic"),
        cudnn_benchmark=_optional_bool_field("cudnn_benchmark"),
        cudnn_enabled=_optional_bool_field("cudnn_enabled"),
        flash_sdp_enabled=_optional_bool_field("flash_sdp_enabled"),
        mem_efficient_sdp_enabled=_optional_bool_field("mem_efficient_sdp_enabled"),
        math_sdp_enabled=_optional_bool_field("math_sdp_enabled"),
        attestation_ineligible_context=_boolean(value, "attestation_ineligible_context"),
    )


def _parse_call_execution_context(value: Mapping[str, Any]) -> CallExecutionContext:
    """Parse the required v2 per-call execution context (explicit, never defaulted)."""

    autocast_entries: list[AutocastDeviceContext] = []
    for item in _mapping_sequence(value, "autocast"):
        dtype = item.get("dtype")
        autocast_entries.append(
            AutocastDeviceContext(
                device_type=_validated_device_literal(
                    "execution_context.autocast.device_type", _string(item, "device_type")
                ),
                enabled=_boolean(item, "enabled"),
                dtype=(
                    None
                    if dtype is None
                    else _validated_dtype_literal(
                        "execution_context.autocast.dtype",
                        _string_item(dtype, "autocast.dtype"),
                    )
                ),
            )
        )
    return CallExecutionContext(
        autocast=tuple(autocast_entries),
        grad_enabled=_boolean(value, "grad_enabled"),
        inference_mode=_boolean(value, "inference_mode"),
    )


def _parse_input_fingerprint(value: Mapping[str, Any]) -> InputAttestationFingerprint:
    """Parse one required physical input fingerprint (``selected_activation_v2``)."""

    device_index = value.get("device_index")
    return InputAttestationFingerprint(
        slot_id=_string(value, "slot_id"),
        byte_digest=_string(value, "byte_digest"),
        device_type=_string(value, "device_type"),
        device_index=None if device_index is None else _integer_item(device_index, "device_index"),
        layout=_string(value, "layout"),
        sizes=tuple(_integer_item(item, "sizes") for item in _sequence(value, "sizes")),
        strides=tuple(_integer_item(item, "strides") for item in _sequence(value, "strides")),
        storage_offset=_integer(value, "storage_offset"),
        is_contiguous=_boolean(value, "is_contiguous"),
        is_channels_last=_boolean(value, "is_channels_last"),
        is_channels_last_3d=_boolean(value, "is_channels_last_3d"),
        is_conj=_boolean(value, "is_conj"),
        is_neg=_boolean(value, "is_neg"),
        tensor_class=_string(value, "tensor_class"),
        requires_grad=_boolean(value, "requires_grad"),
        is_inference=_boolean(value, "is_inference"),
        alignment_class=_integer(value, "alignment_class"),
    )


def _parse_rng_profile(value: Any) -> RunnableRngProfile:
    """Parse the optional host-RNG profile, defaulting legacy manifests to deterministic.

    Manifests written before host-RNG honesty tracking omit this object; they are
    treated as ``host_rng_consumed=False`` (deterministic) because their capture
    predates the recorded signal and cannot be recovered.
    """

    if not isinstance(value, Mapping):
        return RunnableRngProfile(host_rng_consumed=False, capture_seed=None)
    consumed = value.get("host_rng_consumed")
    seed = value.get("capture_seed")
    return RunnableRngProfile(
        host_rng_consumed=bool(consumed),
        capture_seed=int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None,
    )


def attach_sparse_run_readiness(
    trace: Any,
    raw_descriptor: Mapping[str, Any] | None,
) -> ReadinessReport:
    """Parse, resolve, and atomically attach runnable callables to a loaded Trace.

    Parameters
    ----------
    trace:
        Analysis Trace already rehydrated by the ordinary portable loader.
    raw_descriptor:
        Decoded manifest run descriptor, or ``None`` for an analysis artifact.

    Returns
    -------
    ReadinessReport
        Complete non-executing readiness report. Resolution failure never
        prevents the already-loaded analysis Trace from being returned.
    """

    if raw_descriptor is None:
        report = _analysis_only_readiness(trace)
        _store_readiness(trace, None, report, None)
        return report
    capability = raw_descriptor.get("capability")
    if isinstance(capability, str) and capability in LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS:
        # Decision G: a legacy v1 descriptor carries no execution-context records
        # (per-call context, ambient backend context, input fingerprints), and
        # "absent" is never interpreted as a default. Legacy artifacts load for
        # analysis with a typed readiness refusal naming the missing context class;
        # re-capture under v2 is the actionable remedy.
        diagnostic = _diagnostic(
            RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
            f"Legacy runnable capability {capability!r} is analysis-only: it "
            "records no per-call execution context, capture-scoped ambient "
            "backend context, or physical input fingerprints, and absent "
            "context is never defaulted. Re-capture and re-save with this "
            f"TorchLens version to produce {RUNNABLE_TLSPEC_SCHEMA_VERSION!r}.",
            backend=str(getattr(trace, "backend", "unknown")),
            detection_stage="descriptor_legacy_version",
            details=(
                ("capability", capability),
                ("supported", RUNNABLE_TLSPEC_SCHEMA_VERSION),
                ("missing_context_class", "execution_context/ambient_context"),
            ),
        )
        report = ReadinessReport(
            status=ReadinessStatus.UNAVAILABLE,
            provider=RunProvider.LOADED_SPARSE,
            backend=str(getattr(trace, "backend", "unknown")),
            capability=capability,
            resolver_records=(),
            state_sources_available=(),
            witness_completeness=None,
            diagnostics=(diagnostic,),
        )
        _store_readiness(trace, None, report, None)
        return report
    try:
        descriptor = parse_sparse_run_descriptor(raw_descriptor)
    except ContextFieldInvalidError as exc:
        # r37 INV-4: an invalid persisted context VALUE is its own frozen refusal
        # class -- refused at parse, before readiness/staging, so no torch setter
        # or resolved callable ever observes the bytes.
        diagnostic = _diagnostic(
            RunnableErrorCode.CONTEXT_FIELD_INVALID,
            str(exc),
            backend=str(getattr(trace, "backend", "unknown")),
            detection_stage="context_parse_validation",
            details=(("context_field", exc.field), ("reason", exc.detail)),
        )
        report = ReadinessReport(
            status=ReadinessStatus.UNAVAILABLE,
            provider=RunProvider.LOADED_SPARSE,
            backend=str(getattr(trace, "backend", "unknown")),
            capability=cast(str | None, raw_descriptor.get("capability")),
            resolver_records=(),
            state_sources_available=(),
            witness_completeness=None,
            diagnostics=(diagnostic,),
        )
        _store_readiness(trace, None, report, None)
        return report
    except (KeyError, TypeError, ValueError) as exc:
        diagnostic = _diagnostic(
            RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
            f"Sparse runnable descriptor could not be parsed: {exc}",
            backend=str(getattr(trace, "backend", "unknown")),
            detection_stage="descriptor_parse",
        )
        report = ReadinessReport(
            status=ReadinessStatus.UNAVAILABLE,
            provider=RunProvider.LOADED_SPARSE,
            backend=str(getattr(trace, "backend", "unknown")),
            capability=cast(str | None, raw_descriptor.get("capability")),
            resolver_records=(),
            state_sources_available=(),
            witness_completeness=None,
            diagnostics=(diagnostic,),
        )
        _store_readiness(trace, None, report, None)
        return report

    report, attachments = preflight_sparse_run_descriptor(descriptor)
    _store_readiness(trace, descriptor, report, attachments)
    return report


def preflight_sparse_run_descriptor(
    descriptor: SparseRunDescriptor,
) -> tuple[ReadinessReport, Mapping[str, Callable[..., Any]] | None]:
    """Resolve a parsed descriptor once and return atomic call attachments.

    Parameters
    ----------
    descriptor:
        Parsed sparse descriptor.

    Returns
    -------
    tuple[ReadinessReport, Mapping[str, Callable[..., Any]] | None]
        Readiness plus a call-id mapping only when every required computational
        group resolved successfully.
    """

    version_diagnostics = _descriptor_version_diagnostics(descriptor)
    if descriptor.backend != "torch":
        diagnostic = _diagnostic(
            RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY,
            f"Sparse runnable replay is unavailable for backend {descriptor.backend!r}.",
            descriptor=descriptor,
            detection_stage="resolver_backend",
        )
        report = _readiness_report(
            descriptor,
            records=(),
            diagnostics=(*version_diagnostics, diagnostic),
            ready=False,
        )
        return report, None

    registry_by_id = {entry.registry_id: entry for entry in descriptor.callable_registry}
    labels_by_registry: dict[str, list[str]] = {}
    structural_diagnostics: list[RunnableDiagnostic] = []
    for call in descriptor.calls:
        labels_by_registry.setdefault(call.registry_id, []).extend(call.op_labels)
        if call.registry_id not in registry_by_id:
            structural_diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_CALLABLE_REF,
                    "Computational group references a missing callable registry ID.",
                    descriptor=descriptor,
                    registry_id=call.registry_id,
                    affected_ops=call.op_labels,
                    detection_stage="resolver_registry_validation",
                )
            )

    resolutions: dict[str, _Resolution] = {}
    if not version_diagnostics:
        with _state.pause_logging():
            for entry in descriptor.callable_registry:
                resolutions[entry.registry_id] = _resolve_registry_entry(
                    entry,
                    descriptor=descriptor,
                    affected_ops=tuple(
                        dict.fromkeys(labels_by_registry.get(entry.registry_id, ()))
                    ),
                    calls=tuple(
                        call for call in descriptor.calls if call.registry_id == entry.registry_id
                    ),
                )

    records = tuple(resolution.record for resolution in resolutions.values())
    diagnostics = [*version_diagnostics, *structural_diagnostics]
    diagnostics.extend(diagnostic for record in records for diagnostic in record.diagnostics)
    ready = (
        not version_diagnostics
        and not structural_diagnostics
        and len(resolutions) == len(descriptor.callable_registry)
        and all(item.func is not None for item in resolutions.values())
        and all(call.registry_id in resolutions for call in descriptor.calls)
    )
    report = _readiness_report(
        descriptor,
        records=records,
        diagnostics=tuple(diagnostics),
        ready=ready,
    )
    if not ready:
        return report, None
    attachments = {
        call.call_id: cast(Callable[..., Any], resolutions[call.registry_id].func)
        for call in descriptor.calls
    }
    return report, attachments


def _resolve_registry_entry(
    entry: CallableRegistryEntry,
    *,
    descriptor: SparseRunDescriptor,
    affected_ops: tuple[str, ...],
    calls: tuple[RunnableCallDescriptor, ...],
) -> _Resolution:
    """Resolve one unique registry entry through the locked torch ladder."""

    key = entry.key
    stock_path = _stock_path_from_key(key)
    if key.namespace == "custom" and stock_path is None:
        diagnostic = _diagnostic(
            RunnableErrorCode.UNTRUSTED_CUSTOM_IMPORT,
            "Resolver-only readiness refuses custom module imports.",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_security",
            provenance="custom_import_default_deny",
            details=(("import_path", str(key.import_path)),),
        )
        return _unavailable_resolution(entry, diagnostic, "custom_import_default_deny")

    exact = _resolve_exact_key(key, stock_path)
    if exact is not None:
        func, resolved_qualname = exact
        return _resolved_callable(
            entry,
            func,
            resolved_qualname=resolved_qualname,
            provenance=f"exact_getattr:{resolved_qualname}",
            status=ResolverStatus.RESOLVED_EXACT,
            descriptor=descriptor,
            affected_ops=affected_ops,
            calls=calls,
            moved=False,
        )

    if _is_captured_internal_torch_builtin_key(key):
        diagnostic = _diagnostic(
            RunnableErrorCode.PRIVATE_API_UNAVAILABLE,
            "Captured internal torch builtin is unavailable; its public wrapper is not a "
            "safe replay substitute for the recorded call recipe.",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_exact",
            provenance="internal_builtin_identity_required",
            details=(("recorded_key", _key_display_path(key)),),
        )
        return _unavailable_resolution(entry, diagnostic, "internal_builtin_identity_required")

    alias = resolve_runnable_torch_alias(
        stock_path or _key_display_path(key), descriptor.compatibility.backend_version
    )
    if alias is not None:
        namespace, qualname, provenance = alias
        alias_func = _getattr_allowlisted(namespace, qualname)
        if alias_func is None:
            private = _is_private_path(stock_path or _key_display_path(key))
            code = (
                RunnableErrorCode.PRIVATE_API_UNAVAILABLE
                if private
                else RunnableErrorCode.CALLABLE_REMOVED
            )
            diagnostic = _diagnostic(
                code,
                "Recorded callable has an explicit successor, but that successor is unavailable.",
                descriptor=descriptor,
                registry_id=entry.registry_id,
                affected_ops=affected_ops,
                detection_stage="resolver_alias",
                provenance=provenance,
                details=(("target", f"{namespace}.{qualname}"),),
            )
            return _unavailable_resolution(entry, diagnostic, provenance)
        return _resolved_callable(
            entry,
            alias_func,
            resolved_qualname=f"{namespace}.{qualname}",
            provenance=provenance,
            status=ResolverStatus.RESOLVED_ALIAS,
            descriptor=descriptor,
            affected_ops=affected_ops,
            calls=calls,
            moved=True,
        )

    if (stock_path or _key_display_path(key)) in _REMOVED_TORCH_CALLABLES:
        diagnostic = _diagnostic(
            RunnableErrorCode.CALLABLE_REMOVED,
            "Recorded callable was removed and has no supported sparse-run successor.",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_tombstone",
            provenance="explicit_removed_callable_table",
        )
        return _unavailable_resolution(entry, diagnostic, "explicit_removed_callable_table")

    candidates = _reverse_candidates(key.qualname, calls)
    if candidates:
        ranked = _best_ranked_candidates(candidates, key, stock_path)
        if len(ranked) > 1:
            candidate_names = tuple(
                sorted(f"{candidate.namespace}.{candidate.qualname}" for candidate in ranked)
            )
            diagnostic = _diagnostic(
                RunnableErrorCode.AMBIGUOUS_QUALNAME,
                "Multiple reverse-index candidates survived namespace and arity guards.",
                descriptor=descriptor,
                registry_id=entry.registry_id,
                affected_ops=affected_ops,
                detection_stage="resolver_reverse_index",
                provenance="reverse_index_ambiguous",
                details=tuple(("candidate", name) for name in candidate_names),
            )
            return _unavailable_resolution(entry, diagnostic, "reverse_index_ambiguous")
        candidate = ranked[0]
        return _resolved_callable(
            entry,
            candidate.func,
            resolved_qualname=f"{candidate.namespace}.{candidate.qualname}",
            provenance="reverse_index:namespace_rank+name+arity",
            status=ResolverStatus.RESOLVED_ALIAS,
            descriptor=descriptor,
            affected_ops=affected_ops,
            calls=calls,
            moved=True,
        )

    code = (
        RunnableErrorCode.PRIVATE_API_UNAVAILABLE
        if stock_path is not None and _is_private_path(stock_path)
        else RunnableErrorCode.UNRESOLVED_QUALNAME
    )
    diagnostic = _diagnostic(
        code,
        "No exact, explicit alias, or unambiguous reverse-index callable was found.",
        descriptor=descriptor,
        registry_id=entry.registry_id,
        affected_ops=affected_ops,
        detection_stage="resolver_reverse_index",
        provenance="ladder_exhausted_no_guess",
        details=(("recorded_key", _key_display_path(key)),),
    )
    return _unavailable_resolution(entry, diagnostic, "ladder_exhausted_no_guess")


def _resolved_callable(
    entry: CallableRegistryEntry,
    func: Callable[..., Any],
    *,
    resolved_qualname: str,
    provenance: str,
    status: ResolverStatus,
    descriptor: SparseRunDescriptor,
    affected_ops: tuple[str, ...],
    calls: tuple[RunnableCallDescriptor, ...],
    moved: bool,
) -> _Resolution:
    """Unwrap and validate one resolved callable before recording success."""

    original = _unwrap_decorated(func)
    if id(original) in _state._decorated_to_orig:
        diagnostic = _diagnostic(
            RunnableErrorCode.WRAPPER_SHADOWED,
            "Resolved callable remains a TorchLens wrapper after translation.",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_wrapper_translation",
            provenance=provenance,
        )
        return _unavailable_resolution(entry, diagnostic, provenance)
    # SECURITY BOUNDARY (tripwire). A bundle is UNTRUSTED input. The torch /
    # torch.Tensor / torch.nn.functional / operator namespaces also expose
    # side-effecting callables -- above all torch.load / torch.save (both in
    # torch.serialization), which unpickle attacker files (RCE) or write to
    # arbitrary paths BEFORE any downstream path/signature check fires. Every
    # success ladder rung (exact, alias, reverse-index) funnels through here, so
    # this gate closes the whole class: only pure, side-effect-free forward/
    # tensor ops may resolve. Note torch.load IS in get_orig_torch_funcs(), so
    # gating on the wrapped-op inventory would NOT suffice.
    if not is_pure_forward_callable(original):
        diagnostic = _diagnostic(
            RunnableErrorCode.UNTRUSTED_CUSTOM_IMPORT,
            "Resolved callable is not a pure forward/tensor op and is refused "
            "for security (side-effecting or dangerous namespace).",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_security",
            provenance="nonforward_callable_denied",
            details=(("resolved_callable", unsafe_callable_reason(original)),),
        )
        return _unavailable_resolution(entry, diagnostic, "nonforward_callable_denied")
    if not all(_signature_accepts_call(original, call) for call in calls):
        diagnostic = _diagnostic(
            RunnableErrorCode.SIGNATURE_DRIFT,
            "Inspectable current callable cannot accept the recorded positional arity.",
            descriptor=descriptor,
            registry_id=entry.registry_id,
            affected_ops=affected_ops,
            detection_stage="resolver_signature",
            provenance=provenance,
            details=tuple(
                ("call_arity", f"{call.call_id}:{call.num_positional_args}+{call.num_keyword_args}")
                for call in calls
            ),
        )
        return _unavailable_resolution(entry, diagnostic, provenance)
    diagnostics: tuple[RunnableDiagnostic, ...] = ()
    if moved:
        diagnostics = (
            _diagnostic(
                RunnableErrorCode.CALLABLE_MOVED_OR_RENAMED,
                "Recorded callable resolved through an explicit compatibility successor.",
                descriptor=descriptor,
                registry_id=entry.registry_id,
                affected_ops=affected_ops,
                detection_stage="resolver_alias",
                provenance=provenance,
                details=(("resolved_qualname", resolved_qualname),),
            ),
        )
    return _Resolution(
        record=ResolverRecord(
            registry_id=entry.registry_id,
            status=status,
            recorded_key=entry.key,
            resolved_qualname=resolved_qualname,
            provenance=provenance,
            diagnostics=diagnostics,
        ),
        func=original,
    )


def _unavailable_resolution(
    entry: CallableRegistryEntry,
    diagnostic: RunnableDiagnostic,
    provenance: str,
) -> _Resolution:
    """Build one unavailable per-registry resolution result."""

    return _Resolution(
        record=ResolverRecord(
            registry_id=entry.registry_id,
            status=ResolverStatus.UNAVAILABLE,
            recorded_key=entry.key,
            resolved_qualname=None,
            provenance=provenance,
            diagnostics=(diagnostic,),
        ),
        func=None,
    )


def _resolve_exact_key(
    key: FunctionRegistryKey,
    stock_path: str | None,
) -> tuple[Callable[..., Any], str] | None:
    """Resolve one key by getattr on allowlisted roots and namespaces only."""

    property_getter = _safe_tensor_property_getter(key)
    if property_getter is not None:
        return property_getter, f"torch.Tensor.{key.qualname}"
    if key.namespace in _ALLOWED_EXACT_ROOTS:
        func = getattr(_ALLOWED_EXACT_ROOTS[key.namespace], key.qualname, None)
        if callable(func):
            return cast(Callable[..., Any], func), f"{key.namespace}.{key.qualname}"
    if stock_path is None:
        return None
    namespace, _, qualname = stock_path.rpartition(".")
    if namespace not in _ENUMERATED_TORCH_NAMESPACES:
        return None
    func = _getattr_allowlisted(namespace, qualname)
    if func is None:
        return None
    return func, f"{namespace}.{qualname}"


def _safe_tensor_property_getter(key: FunctionRegistryKey) -> Callable[..., Any] | None:
    """Return a callable getter for an allowlisted tensor property key."""

    if (
        key.namespace != "torch.Tensor"
        or key.dispatch_kind != "method"
        or key.qualname not in _SAFE_TENSOR_PROPERTY_NAMES
    ):
        return None

    def getter(tensor: torch.Tensor) -> torch.Tensor:
        """Return one safe tensor property value."""

        return cast(torch.Tensor, getattr(tensor, key.qualname))

    getter.__name__ = str(key.qualname)
    getter.__qualname__ = f"Tensor.{key.qualname}"
    getter.__module__ = "torch._tensor"
    return getter


def _getattr_allowlisted(namespace: str, qualname: str) -> Callable[..., Any] | None:
    """Read a callable from one explicitly allowlisted in-memory namespace."""

    if namespace in _ALLOWED_EXACT_ROOTS:
        root = _ALLOWED_EXACT_ROOTS[namespace]
    elif namespace in _ENUMERATED_TORCH_NAMESPACES:
        root = _torch_namespace(namespace)
    else:
        return None
    if root is None:
        return None
    value = getattr(root, qualname, None)
    return cast(Callable[..., Any], value) if callable(value) else None


def _torch_namespace(namespace: str) -> Any | None:
    """Traverse an enumerated torch namespace without importing modules."""

    current: Any = torch
    for part in namespace.removeprefix("torch.").split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _stock_path_from_key(key: FunctionRegistryKey) -> str | None:
    """Return a normalized stock torch path without importing custom modules."""

    if key.namespace != "custom":
        return f"{key.namespace}.{key.qualname}"
    if not key.import_path:
        return None
    module_name, separator, qualname = key.import_path.partition(":")
    if not separator or not (module_name == "torch" or module_name.startswith("torch.")):
        return None
    return f"{module_name}.{qualname}"


def _is_captured_internal_torch_builtin_key(key: FunctionRegistryKey) -> bool:
    """Return whether ``key`` identifies a recipe-sensitive torch builtin.

    Parameters
    ----------
    key:
        Saved callable registry key.

    Returns
    -------
    bool
        Whether the key was emitted for a canonical
        ``_VariableFunctionsClass`` builtin.
    """

    if key.namespace != "custom" or key.import_path is None:
        return False
    module_name, separator, qualname = key.import_path.partition(":")
    return (
        separator == ":"
        and module_name == "torch._C._VariableFunctionsClass"
        and qualname == key.qualname
    )


def _key_display_path(key: FunctionRegistryKey) -> str:
    """Return a stable display path for one recorded registry key."""

    return (
        key.import_path.replace(":", ".") if key.import_path else f"{key.namespace}.{key.qualname}"
    )


def _is_private_path(path: str) -> bool:
    """Return whether a recorded torch path traverses a private namespace."""

    return any(part.startswith("_") for part in path.split(".")[1:-1])


def _unwrap_decorated(func: Callable[..., Any]) -> Callable[..., Any]:
    """Translate a TorchLens decorated function to its original callable."""

    current = func
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        original = _state._decorated_to_orig.get(id(current))
        if original is None:
            break
        current = original
    return current


@lru_cache(maxsize=1)
def _reverse_index() -> Mapping[str, tuple[_ReverseCandidate, ...]]:
    """Build the cached reverse index over TorchLens's torch wrapper inventory."""

    index: dict[str, list[_ReverseCandidate]] = {}
    seen: set[tuple[str, str]] = set()
    for namespace, qualname in get_orig_torch_funcs(include_torchvision=False):
        marker = (namespace, qualname)
        if marker in seen:
            continue
        seen.add(marker)
        root = _torch_namespace(namespace) if namespace != "torch" else torch
        if root is None:
            continue
        value = getattr(root, qualname, None)
        if not callable(value):
            continue
        candidate = _ReverseCandidate(
            namespace=namespace,
            qualname=qualname,
            func=_unwrap_decorated(cast(Callable[..., Any], value)),
        )
        index.setdefault(qualname, []).append(candidate)
    return {name: tuple(candidates) for name, candidates in index.items()}


def _reverse_candidates(
    qualname: str,
    calls: tuple[RunnableCallDescriptor, ...],
) -> tuple[_ReverseCandidate, ...]:
    """Return name- and arity-compatible reverse-index candidates."""

    func_name = qualname.rsplit(".", maxsplit=1)[-1]
    return tuple(
        candidate
        for candidate in _reverse_index().get(func_name, ())
        if all(_signature_accepts_call(candidate.func, call) for call in calls)
    )


def _best_ranked_candidates(
    candidates: tuple[_ReverseCandidate, ...],
    key: FunctionRegistryKey,
    stock_path: str | None,
) -> tuple[_ReverseCandidate, ...]:
    """Keep only candidates at the best deterministic namespace rank."""

    expected_namespace: str = key.namespace
    if stock_path is not None:
        expected_namespace = stock_path.rpartition(".")[0]
    public_rank = {
        "torch": 1,
        "torch.nn.functional": 2,
        "torch.Tensor": 3,
        "operator": 4,
    }

    def rank(candidate: _ReverseCandidate) -> int:
        """Return one candidate's deterministic namespace rank."""

        if candidate.namespace == expected_namespace:
            return 0
        return public_rank.get(candidate.namespace, 10)

    best = min(rank(candidate) for candidate in candidates)
    ranked = [candidate for candidate in candidates if rank(candidate) == best]
    deduplicated: dict[int, _ReverseCandidate] = {}
    for candidate in ranked:
        deduplicated.setdefault(id(candidate.func), candidate)
    return tuple(deduplicated.values())


def _signature_accepts_call(
    func: Callable[..., Any],
    call: RunnableCallDescriptor,
) -> bool:
    """Return whether an inspectable callable accepts recorded positional arity."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    placeholders = (object(),) * call.num_positional_args
    argument_paths = (
        *(argument.argument_path for argument in call.tensor_arguments),
        *(argument.argument_path for argument in call.literal_arguments),
    )
    keyword_names = {
        path[1]
        for path in argument_paths
        if len(path) >= 2 and path[0] == "kwargs" and isinstance(path[1], str)
    }
    keyword_placeholders = {name: object() for name in keyword_names}
    try:
        signature.bind_partial(*placeholders, **keyword_placeholders)
    except TypeError:
        return False
    return True


def _descriptor_version_diagnostics(
    descriptor: SparseRunDescriptor,
) -> tuple[RunnableDiagnostic, ...]:
    """Return structured readiness failures for unsupported descriptor ceilings."""

    diagnostics: list[RunnableDiagnostic] = []
    expected_strings = (
        ("capability", descriptor.capability, RUNNABLE_TLSPEC_SCHEMA_VERSION),
        ("call_recipe", descriptor.call_recipe, RUNNABLE_CALL_RECIPE_VERSION),
        (
            "initializer_policy_version",
            descriptor.initializer_policy_version,
            RUNNABLE_INITIALIZER_POLICY_VERSION,
        ),
        ("state_binding", descriptor.state_binding, "module_path_role_v1"),
        ("input_binding", descriptor.input_binding, "model_site_io_role_v1"),
        (
            "control_witness",
            descriptor.control_witness,
            "scalar_bool_and_arm_entry_v1",
        ),
        (
            "compatibility.descriptor_version",
            descriptor.compatibility.descriptor_version,
            RUNNABLE_TLSPEC_SCHEMA_VERSION,
        ),
        (
            "compatibility.call_recipe_version",
            descriptor.compatibility.call_recipe_version,
            RUNNABLE_CALL_RECIPE_VERSION,
        ),
        (
            "compatibility.initializer_policy_version",
            descriptor.compatibility.initializer_policy_version,
            RUNNABLE_INITIALIZER_POLICY_VERSION,
        ),
    )
    for field_name, actual, expected in expected_strings:
        if actual == expected:
            continue
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
                f"Unsupported runnable {field_name}={actual!r}; runtime supports {expected!r}.",
                descriptor=descriptor,
                detection_stage="descriptor_version_validation",
                details=(("field", field_name), ("supported", str(expected))),
            )
        )
    activations_layer = descriptor.payload_layers.activations
    if activations_layer.schema != RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
                f"Unsupported activation payload schema {activations_layer.schema!r}; "
                f"runtime supports {RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION!r}.",
                descriptor=descriptor,
                detection_stage="descriptor_version_validation",
                details=(
                    ("field", "payload_layers.activations.schema"),
                    ("supported", RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION),
                ),
            )
        )
    nonpersistent_layer = descriptor.payload_layers.nonpersistent_buffers
    has_nonpersistent_slots = any(
        slot.role is TensorSlotRole.BUFFER
        and slot.state_binding is not None
        and not slot.state_binding.persistent
        for slot in descriptor.tensor_slots
    )
    if (
        nonpersistent_layer.schema != "runnable_nonpersistent_buffer_v1"
        or nonpersistent_layer.present != has_nonpersistent_slots
    ):
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
                "Non-persistent buffer payload declaration disagrees with its slots.",
                descriptor=descriptor,
                detection_stage="descriptor_payload_validation",
                details=(
                    ("schema", nonpersistent_layer.schema),
                    ("declared_present", str(nonpersistent_layer.present)),
                    ("has_nonpersistent_slots", str(has_nonpersistent_slots)),
                ),
            )
        )
    if descriptor.callable_ref_schema != RUNNABLE_CALLABLE_REF_SCHEMA_VERSION:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.UNSUPPORTED_REF_SCHEMA,
                "Callable reference schema is newer or otherwise unsupported.",
                descriptor=descriptor,
                detection_stage="descriptor_version_validation",
                details=(
                    ("recorded", str(descriptor.callable_ref_schema)),
                    ("supported", str(RUNNABLE_CALLABLE_REF_SCHEMA_VERSION)),
                ),
            )
        )
    incompatible_key_ids = tuple(
        entry.registry_id
        for entry in descriptor.callable_registry
        if entry.key.version != RUNNABLE_CALLABLE_REF_SCHEMA_VERSION
    )
    if (
        descriptor.compatibility.callable_ref_schema_version != RUNNABLE_CALLABLE_REF_SCHEMA_VERSION
        or incompatible_key_ids
    ):
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.UNSUPPORTED_REF_SCHEMA,
                "Compatibility metadata or a registry key uses an unsupported ref schema.",
                descriptor=descriptor,
                detection_stage="descriptor_version_validation",
                details=(
                    (
                        "compatibility_recorded",
                        str(descriptor.compatibility.callable_ref_schema_version),
                    ),
                    ("registry_ids", ",".join(incompatible_key_ids)),
                ),
            )
        )
    return tuple(diagnostics)


def _readiness_report(
    descriptor: SparseRunDescriptor,
    *,
    records: tuple[ResolverRecord, ...],
    diagnostics: tuple[RunnableDiagnostic, ...],
    ready: bool,
) -> ReadinessReport:
    """Build the frozen readiness report shape for one sparse descriptor."""

    state_sources = [StateSource.RANDOM_INITIALIZATION]
    if descriptor.payload_layers.weights.present:
        state_sources.insert(0, StateSource.EMBEDDED_CAPTURE_STATE)
    return ReadinessReport(
        status=ReadinessStatus.READY if ready else ReadinessStatus.UNAVAILABLE,
        provider=RunProvider.LOADED_SPARSE,
        backend=descriptor.backend,
        capability=descriptor.capability,
        resolver_records=records,
        state_sources_available=tuple(state_sources),
        witness_completeness=descriptor.witness_completeness,
        diagnostics=diagnostics,
    )


def _analysis_only_readiness(trace: Any) -> ReadinessReport:
    """Build unsupported readiness for a loaded analysis-only Trace."""

    backend = str(getattr(trace, "backend", "unknown"))
    diagnostic = _diagnostic(
        RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
        "Loaded analysis artifact has no sparse runnable descriptor.",
        backend=backend,
        detection_stage="descriptor_presence",
    )
    return ReadinessReport(
        status=ReadinessStatus.UNAVAILABLE,
        provider=RunProvider.LOADED_ANALYSIS,
        backend=backend,
        capability=None,
        resolver_records=(),
        state_sources_available=(),
        witness_completeness=None,
        diagnostics=(diagnostic,),
    )


def _store_readiness(
    trace: Any,
    descriptor: SparseRunDescriptor | None,
    report: ReadinessReport,
    attachments: Mapping[str, Callable[..., Any]] | None,
) -> None:
    """Store transient descriptor/readiness state with atomic callable attachment."""

    trace.__dict__["_runnable_descriptor"] = descriptor
    trace.__dict__["_runnable_readiness"] = report
    if attachments is None:
        trace.__dict__.pop("_runnable_callables_by_call_id", None)
    else:
        trace.__dict__["_runnable_callables_by_call_id"] = dict(attachments)


def _diagnostic(
    code: RunnableErrorCode,
    message: str,
    *,
    descriptor: SparseRunDescriptor | None = None,
    backend: str | None = None,
    registry_id: str | None = None,
    affected_ops: tuple[str, ...] = (),
    detection_stage: str,
    provenance: str | None = None,
    details: tuple[tuple[str, str], ...] = (),
) -> RunnableDiagnostic:
    """Build one complete Stage 3 readiness diagnostic record."""

    recorded_runtime = descriptor.compatibility.backend_version if descriptor is not None else None
    current_backend = descriptor.backend if descriptor is not None else backend
    current_runtime = (
        str(torch.__version__) if current_backend == "torch" else platform.python_version()
    )
    return RunnableDiagnostic(
        code=code,
        message=message,
        registry_id=registry_id,
        affected_op_labels=affected_ops,
        recorded_runtime=recorded_runtime,
        current_runtime=current_runtime,
        detection_stage=detection_stage,
        resolver_provenance=provenance,
        analysis_load_available=True,
        details=details,
    )


def _parse_registry_key(value: Mapping[str, Any]) -> FunctionRegistryKey:
    """Parse one FunctionRegistryKey-shaped JSON object."""

    import_path = value.get("import_path")
    if import_path is not None and not isinstance(import_path, str):
        raise TypeError("FunctionRegistryKey.import_path must be a string or null.")
    return FunctionRegistryKey(
        namespace=cast(Any, _string(value, "namespace")),
        qualname=_string(value, "qualname"),
        dispatch_kind=cast(Any, _string(value, "dispatch_kind")),
        version=_integer(value, "version"),
        import_path=import_path,
    )


def _parse_call(value: Mapping[str, Any]) -> RunnableCallDescriptor:
    """Parse one runnable computational call descriptor."""

    return RunnableCallDescriptor(
        call_id=_string(value, "call_id"),
        op_labels=_string_tuple(value, "op_labels"),
        registry_id=_string(value, "registry_id"),
        dispatch_kind=cast(Any, _string(value, "dispatch_kind")),
        argument_names=_string_tuple(value, "argument_names"),
        num_positional_args=_integer(value, "num_positional_args"),
        num_keyword_args=_integer(value, "num_keyword_args"),
        tensor_arguments=tuple(
            TensorArgumentRef(
                argument_path=_path(item, "argument_path"),
                slot_id=_string(item, "slot_id"),
            )
            for item in _mapping_sequence(value, "tensor_arguments")
        ),
        literal_arguments=tuple(
            LiteralArgumentRef(
                argument_path=_path(item, "argument_path"),
                value=_parse_literal(item["value"]),
            )
            for item in _mapping_sequence(value, "literal_arguments")
        ),
        output_slot_ids=_string_tuple(value, "output_slot_ids"),
        parent_call_ids=_string_tuple(value, "parent_call_ids"),
        is_inplace=_boolean(value, "is_inplace"),
        runtime_fingerprint=_string(value, "runtime_fingerprint"),
        # v2: the per-call execution context is REQUIRED; absence is a parse
        # failure (legacy artifacts are analysis-only), never "assume disabled".
        execution_context=_parse_call_execution_context(_mapping(value, "execution_context")),
    )


def _parse_slot(value: Mapping[str, Any]) -> TensorSlotDescriptor:
    """Parse one value-free tensor slot descriptor."""

    input_value = value.get("input_binding")
    state_value = value.get("state_binding")
    output_path = value.get("output_path")
    version_of = value.get("version_of")
    producer_slot_id = value.get("producer_slot_id")
    device_index = value.get("device_index")
    return TensorSlotDescriptor(
        slot_id=_string(value, "slot_id"),
        role=TensorSlotRole(_string(value, "role")),
        use_sites=tuple(
            TensorUseSite(
                call_id=_string(item, "call_id"),
                argument_path=_path(item, "argument_path"),
            )
            for item in _mapping_sequence(value, "use_sites")
        ),
        shape=tuple(_integer_item(item, "shape") for item in _sequence(value, "shape")),
        dtype=_string(value, "dtype"),
        rank=_integer(value, "rank"),
        device_type=_string(value, "device_type"),
        device_index=None if device_index is None else _integer_item(device_index, "device_index"),
        mutable=_boolean(value, "mutable"),
        version_of=_optional_string(version_of, "version_of"),
        producer_slot_id=_optional_string(producer_slot_id, "producer_slot_id"),
        output_path=None if output_path is None else _path_value(output_path, "output_path"),
        input_binding=(
            None
            if input_value is None
            else _parse_input_binding(_mapping_item(input_value, "input_binding"))
        ),
        state_binding=(
            None
            if state_value is None
            else _parse_state_binding(_mapping_item(state_value, "state_binding"))
        ),
    )


def _parse_input_binding(value: Mapping[str, Any]) -> InputSlotBinding:
    """Parse one model-input slot binding."""

    position = value["model_site_position"]
    if isinstance(position, list):
        parsed_position: str | int | tuple[str | int, ...] = _path_value(
            position, "model_site_position"
        )
    elif isinstance(position, (str, int)) and not isinstance(position, bool):
        parsed_position = position
    else:
        raise TypeError("model_site_position must be a string, integer, or path array.")
    return InputSlotBinding(
        io_role=cast(Any, _string(value, "io_role")),
        model_ref=_string(value, "model_ref"),
        model_site_position=parsed_position,
        container_record_id=_integer(value, "container_record_id"),
        container_path=_path(value, "container_path"),
    )


def _parse_state_binding(value: Mapping[str, Any]) -> StateSlotBinding:
    """Parse one named parameter or buffer binding."""

    return StateSlotBinding(
        module_path=_string(value, "module_path"),
        state_dict_name=_string(value, "state_dict_name"),
        semantic_role=StateSlotRole(_string(value, "semantic_role")),
        trainable=_boolean(value, "trainable"),
        persistent=_boolean(value, "persistent"),
        alias_group=_optional_string(value.get("alias_group"), "alias_group"),
    )


def _parse_witness(value: Mapping[str, Any]) -> ControlWitness:
    """Parse one ordered control-flow witness."""

    return ControlWitness(
        witness_id=_string(value, "witness_id"),
        kind=ControlWitnessKind(_string(value, "kind")),
        order=_integer(value, "order"),
        call_id=_optional_string(value.get("call_id"), "call_id"),
        site_label=_string(value, "site_label"),
        observed_value=_parse_literal(value["observed_value"]),
    )


def _parse_payload_layer(value: Mapping[str, Any]) -> PayloadLayerDescriptor:
    """Parse one optional payload-layer declaration."""

    return PayloadLayerDescriptor(
        present=_boolean(value, "present"),
        schema=_string(value, "schema"),
    )


def _parse_nonpersistent_buffer_payload_layer(
    payload_layers: Mapping[str, Any],
) -> PayloadLayerDescriptor:
    """Parse capture-embedded non-persistent buffers with a legacy default."""

    value = payload_layers.get("nonpersistent_buffers")
    if not isinstance(value, Mapping):
        return PayloadLayerDescriptor(
            present=False,
            schema="runnable_nonpersistent_buffer_v1",
        )
    return _parse_payload_layer(value)


def _parse_activation_payload_layer(
    value: Mapping[str, Any],
) -> PayloadLayerDescriptor | ActivationPayloadLayerDescriptor:
    """Parse selected-activation membership and eligibility digests."""

    if not _boolean(value, "present"):
        return _parse_payload_layer(value)
    return ActivationPayloadLayerDescriptor(
        present=_boolean(value, "present"),
        schema=cast(Any, _string(value, "schema")),
        input_fingerprints=tuple(
            _parse_input_fingerprint(item)
            for item in _mapping_sequence(value, "input_fingerprints")
        ),
        members=tuple(
            ActivationPayloadMember(
                blob_id=_string(item, "blob_id"),
                slot_id=_string(item, "slot_id"),
                call_id=_optional_string(item.get("call_id"), "call_id"),
                op_label=_string(item, "op_label"),
                field=cast(Any, _string(item, "field")),
                byte_digest=_string(item, "byte_digest"),
            )
            for item in _mapping_sequence(value, "members")
        ),
        original_input_digests=tuple(
            SlotByteDigest(
                slot_id=_string(item, "slot_id"),
                byte_digest=_string(item, "byte_digest"),
            )
            for item in _mapping_sequence(value, "original_input_digests")
        ),
        capture_state_digests=tuple(
            StateByteDigest(
                state_dict_name=_string(item, "state_dict_name"),
                byte_digest=_string(item, "byte_digest"),
            )
            for item in _mapping_sequence(value, "capture_state_digests")
        ),
    )


def _parse_diagnostic(value: Mapping[str, Any]) -> RunnableDiagnostic:
    """Parse one persisted producer diagnostic."""

    return RunnableDiagnostic(
        code=RunnableErrorCode(_string(value, "code")),
        message=_string(value, "message"),
        registry_id=_optional_string(value.get("registry_id"), "registry_id"),
        affected_op_labels=_string_tuple(value, "affected_op_labels"),
        recorded_runtime=_optional_string(value.get("recorded_runtime"), "recorded_runtime"),
        current_runtime=_optional_string(value.get("current_runtime"), "current_runtime"),
        detection_stage=_string(value, "detection_stage"),
        resolver_provenance=_optional_string(
            value.get("resolver_provenance"), "resolver_provenance"
        ),
        analysis_load_available=_boolean(value, "analysis_load_available"),
        details=tuple(
            (_string_item(item[0], "details key"), _string_item(item[1], "details value"))
            for item in _sequence(value, "details")
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2
        ),
    )


def _validate_literal_atom_value(kind: "LiteralAtomKind", value: Any) -> None:
    """Enforce that an atom's JSON value matches its declared ``kind``.

    ROBUSTNESS (secC informational note). ``LiteralAtom`` is a *scalar* grammar
    node: downstream witness / decode logic assumes an atom carries the scalar its
    ``kind`` promises (e.g. ``bool(...)`` / ``.get(...)`` on a witness
    ``observed_value``). The encoder (``_io/runnable.py``) only ever emits one JSON
    type per kind, but a hand-edited ``manifest.json`` could tag ``kind="int"`` on a
    list / dict / string, which decodes inertly yet makes scalar-assuming logic
    inconsistent. Reject the type mismatch at parse so the grammar invariant is
    upheld. (A genuine container value has its own ``LiteralSequence`` /
    ``LiteralMapping`` node; it never rides on a scalar atom.) ``bool`` is a subclass
    of ``int`` in Python, so ``INT`` explicitly excludes ``bool`` (and vice versa) to
    match the encoder's distinct ``BOOL`` / ``INT`` kinds; JSON ``float`` never
    parses as ``int``/``bool`` so ``FLOAT`` needs no such exclusion.
    """

    if kind in (LiteralAtomKind.NONE, LiteralAtomKind.ELLIPSIS):
        if value is not None:
            raise ValueError(
                f"Runnable literal atom kind {kind.value!r} requires a null value, "
                f"got {type(value).__name__}."
            )
    elif kind is LiteralAtomKind.BOOL:
        if not isinstance(value, bool):
            raise ValueError(
                f"Runnable literal atom kind 'bool' requires a bool value, "
                f"got {type(value).__name__}."
            )
    elif kind is LiteralAtomKind.INT:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"Runnable literal atom kind 'int' requires an int value, "
                f"got {type(value).__name__}."
            )
    elif kind is LiteralAtomKind.FLOAT:
        if not isinstance(value, float):
            raise ValueError(
                f"Runnable literal atom kind 'float' requires a float value, "
                f"got {type(value).__name__}."
            )
    elif kind in (LiteralAtomKind.STR, LiteralAtomKind.NONFINITE_FLOAT):
        if not isinstance(value, str):
            raise ValueError(
                f"Runnable literal atom kind {kind.value!r} requires a str value, "
                f"got {type(value).__name__}."
            )


def _parse_literal(value: Any) -> NonTensorLiteral:
    """Parse one recursively tagged non-tensor literal."""

    mapping = _mapping_item(value, "literal")
    keys = set(mapping)
    if keys == {"kind", "value"}:
        kind = LiteralAtomKind(_string(mapping, "kind"))
        atom_value = mapping["value"]
        _validate_literal_atom_value(kind, atom_value)
        return LiteralAtom(
            kind=kind,
            value=cast(Any, atom_value),
        )
    if keys == {"qualname"}:
        return LiteralTorchSymbol(qualname=_string(mapping, "qualname"))
    if keys == {"start", "stop", "step"}:
        return LiteralSlice(
            start=_parse_literal_slice_component(mapping["start"], "start"),
            stop=_parse_literal_slice_component(mapping["stop"], "stop"),
            step=_parse_literal_slice_component(mapping["step"], "step"),
        )
    if keys == {"kind", "items"}:
        return LiteralSequence(
            kind=LiteralSequenceKind(_string(mapping, "kind")),
            items=tuple(_parse_literal(item) for item in _sequence(mapping, "items")),
        )
    if keys == {"entries"}:
        return LiteralMapping(
            entries=tuple(
                LiteralMappingEntry(
                    key=_parse_literal_key(item["key"]),
                    value=_parse_literal(item["value"]),
                )
                for item in _mapping_sequence(mapping, "entries")
            )
        )
    raise ValueError(f"Unsupported tagged runnable literal fields: {sorted(keys)}")


def _parse_literal_slice_component(value: Any, field_name: str) -> LiteralAtom:
    """Parse one ``slice.start``/``.stop``/``.step`` tagged atom.

    A slice component is restricted to the ``NONE`` or ``INT`` atom kinds on the
    encode side (see ``_io/runnable.py::_encode_slice_component``); reject anything
    else here rather than silently widening what a loaded bundle may claim.
    """

    parsed = _parse_literal(value)
    if not isinstance(parsed, LiteralAtom) or parsed.kind not in (
        LiteralAtomKind.NONE,
        LiteralAtomKind.INT,
    ):
        raise ValueError(
            f"Runnable literal slice component {field_name!r} must be a NONE or INT atom."
        )
    return parsed


def _parse_literal_key(value: Any) -> LiteralAtom | LiteralTupleKey:
    """Parse one safe mapping-key literal."""

    mapping = _mapping_item(value, "literal mapping key")
    if set(mapping) == {"kind", "value"}:
        parsed = _parse_literal(mapping)
        if isinstance(parsed, LiteralAtom):
            return parsed
    if set(mapping) == {"items"}:
        return LiteralTupleKey(
            items=tuple(_parse_literal_key(item) for item in _sequence(mapping, "items"))
        )
    raise ValueError("Runnable literal mapping key is not an atom or tuple key.")


def _mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    """Return one required mapping field."""

    return _mapping_item(value[field], field)


def _mapping_item(value: Any, label: str) -> Mapping[str, Any]:
    """Validate and return one arbitrary mapping value."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    return cast(Mapping[str, Any], value)


def _sequence(value: Mapping[str, Any], field: str) -> Sequence[Any]:
    """Return one required non-string sequence field."""

    item = value[field]
    if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
        raise TypeError(f"{field} must be an array.")
    return item


def _mapping_sequence(value: Mapping[str, Any], field: str) -> tuple[Mapping[str, Any], ...]:
    """Return one required array of object values."""

    return tuple(_mapping_item(item, field) for item in _sequence(value, field))


def _string(value: Mapping[str, Any], field: str) -> str:
    """Return one required string field."""

    return _string_item(value[field], field)


def _string_item(value: Any, label: str) -> str:
    """Validate and return one arbitrary string value."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    """Validate and return one optional string value."""

    if value is None:
        return None
    return _string_item(value, label)


def _string_tuple(value: Mapping[str, Any], field: str) -> tuple[str, ...]:
    """Return one required array of strings."""

    return tuple(_string_item(item, field) for item in _sequence(value, field))


def _integer(value: Mapping[str, Any], field: str) -> int:
    """Return one required non-boolean integer field."""

    return _integer_item(value[field], field)


def _integer_item(value: Any, label: str) -> int:
    """Validate and return one arbitrary non-boolean integer value."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer.")
    return value


def _boolean(value: Mapping[str, Any], field: str) -> bool:
    """Return one required boolean field."""

    item = value[field]
    if not isinstance(item, bool):
        raise TypeError(f"{field} must be a boolean.")
    return item


def _path(value: Mapping[str, Any], field: str) -> tuple[str | int, ...]:
    """Return one required string/integer path field."""

    return _path_value(value[field], field)


def _path_value(value: Any, label: str) -> tuple[str | int, ...]:
    """Validate and return one arbitrary container path array."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be an array.")
    result: list[str | int] = []
    for item in value:
        # ``bool`` is a subclass of ``int`` and is accepted as a faithful,
        # round-trippable container-path key (``True`` stays distinct from ``1``
        # as a mapping key), matching the producer's normalized path grammar so a
        # bool-keyed output/input container never advertises runnable then loses
        # it on load.
        if not isinstance(item, (str, int)):
            raise TypeError(f"{label} entries must be strings or integers.")
        result.append(item)
    return tuple(result)


__all__ = [
    "attach_sparse_run_readiness",
    "parse_sparse_run_descriptor",
    "preflight_sparse_run_descriptor",
]
