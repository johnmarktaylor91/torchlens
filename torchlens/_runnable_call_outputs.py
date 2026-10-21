"""Sparse-call output binding and mutation checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from ._runnable_state import (
    RunResourceCeiling,
)
from .errors import (
    RunPreconditionError,
)
from .ir.container import (
    CONTAINER_KIND_CAPABILITIES,
    ContainerReconstructionError,
    ContainerSpec,
    _reconstruction_would_substitute_plain,
    namedtuple_type_can_carry_instance_state,
    rebuild_container_from_spec,
    reconstruction_is_lossy_by_type,
    resolve_container_type,
)
from .runnable import (
    ContractCheck,
    RunnableCallDescriptor,
    RunnableErrorCode,
    SparseRunDescriptor,
    TensorSlotDescriptor,
    TensorSlotRole,
)
from .utils._torch_compat import tensor_version_or_none

if TYPE_CHECKING:
    from ._runnable_execution import (
        _canonicalize_structseq_output_paths,
        _container_from_paths,
        _contract_check,
        _op_for_label,
        _op_for_slot,
        _raw_runtime_output,
        _recorded_structseq_output_type_matches,
        _resolve_setter_output,
        _tensor_leaf_paths,
        _torch_structseq_field_names,
        _value_at_path,
    )

__all__ = (
    "_bind_call_outputs",
    "_mutation_contract_checks",
    "_mutation_target_slot_id",
    "_out_argument_slot_id",
    "_reconstruct_output",
    "_output_container_spec",
    "_output_not_reproduced",
    "_container_spec_reconstruction_lossy",
    "_spec_node_reconstruction_lossy",
    "_fresh_bare_tensor_root",
)


def _bind_call_outputs(
    call: RunnableCallDescriptor,
    output: Any,
    slot_values: dict[str, torch.Tensor],
    fork: Any,
    *,
    ceiling: RunResourceCeiling,
    before_versions: Mapping[str, int],
    attestation_slot_ids: frozenset[str],
    attestation_slot_values: dict[str, torch.Tensor],
    slots: Mapping[str, TensorSlotDescriptor],
    state_slot_ids: frozenset[str],
    version_alias_ids: Mapping[str, tuple[str, ...]],
    witness_slot_ids: frozenset[str] = frozenset(),
    witness_source_snapshots: dict[str, torch.Tensor] | None = None,
) -> tuple[ContractCheck, ...]:
    """Slice, validate, and stage one grouped call's tensor outputs.

    ``slots``, ``state_slot_ids``, and ``version_alias_ids`` are the run-invariant
    ``descriptor.tensor_slots`` indexes, built ONCE per run by the transaction and
    threaded in: the descriptor is a frozen dataclass whose ``tensor_slots`` tuple and
    per-slot ``role``/version topology cannot change mid-replay, so rebuilding or
    rescanning them per call was pure O(calls x slots) waste.
    """

    checks: list[ContractCheck] = []
    output = _resolve_setter_output(call, output, slot_values)
    expected_paths = tuple(slots[slot_id].output_path or () for slot_id in call.output_slot_ids)
    actual_paths = _tensor_leaf_paths(output)
    # r59 gate 2 (backstop): a projection-skipped call (data-dependent fail-open, or no
    # numeric literal) cannot smuggle an unbounded realized-output tree past the
    # front-line count projection. Honest realized == recorded, so this is pure headroom.
    ceiling.charge_realized_outputs(call, len(actual_paths))
    expected_structure_paths = _canonicalize_structseq_output_paths(output, expected_paths)
    actual_structure_paths = _canonicalize_structseq_output_paths(output, actual_paths)
    output_type_matches = _recorded_structseq_output_type_matches(fork, call, output)
    checks.append(
        _contract_check(
            f"output_structure:{call.call_id}",
            len(call.output_slot_ids) == len(call.op_labels)
            and output_type_matches
            and actual_structure_paths == expected_structure_paths,
            RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
            f"Call {call.call_id!r} output tensor paths disagree with the recorded container.",
            affected_op_labels=call.op_labels,
            details=(
                ("expected_paths", repr(expected_paths)),
                ("actual_paths", repr(tuple(actual_paths))),
                ("canonical_expected_paths", repr(expected_structure_paths)),
                ("canonical_actual_paths", repr(actual_structure_paths)),
            ),
        )
    )
    for slot_id, op_label in zip(call.output_slot_ids, call.op_labels):
        slot = slots[slot_id]
        try:
            value = _value_at_path(output, slot.output_path or ())
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            checks.append(
                _contract_check(
                    f"slot_production:{slot_id}",
                    False,
                    RunnableErrorCode.SLOT_PRODUCTION_MISMATCH,
                    f"Call {call.call_id!r} output lacks path {slot.output_path!r}: {exc}",
                    affected_op_labels=call.op_labels,
                    details=(("slot_id", slot_id),),
                )
            )
            continue
        if not isinstance(value, torch.Tensor):
            checks.append(
                _contract_check(
                    f"slot_production:{slot_id}",
                    False,
                    RunnableErrorCode.SLOT_PRODUCTION_MISMATCH,
                    f"Call {call.call_id!r} output is not a tensor at {slot.output_path!r}.",
                    affected_op_labels=call.op_labels,
                    details=(("slot_id", slot_id),),
                )
            )
            continue
        slot_values[slot_id] = value
        produced_slot_ids = {slot_id}
        out_argument_slot_id = _out_argument_slot_id(call) if call.is_inplace else None
        if out_argument_slot_id is not None:
            # ``out=`` calls mutate their explicit destination, which may have
            # been produced by an earlier call. Keep that aliased slot current
            # for downstream reads and activation attestation.
            slot_values[out_argument_slot_id] = value
            produced_slot_ids.add(out_argument_slot_id)
        for version_slot_id in version_alias_ids.get(slot_id, ()):
            slot_values[version_slot_id] = value
            produced_slot_ids.add(version_slot_id)
        for produced_slot_id in produced_slot_ids & attestation_slot_ids:
            # r59 gate 3: byte-guard every op-output snapshot before it allocates.
            attestation_slot_values[produced_slot_id] = ceiling.guarded_clone(
                value,
                call_id=call.call_id,
                slot_id=produced_slot_id,
                affected_op_labels=call.op_labels,
            )
        if witness_source_snapshots is not None:
            # Snapshot every escape-witness source slot at its production point so a
            # later in-place mutation of the live tensor cannot restale the digest
            # comparison (H3): the run-digest then matches the pre-mutation save-digest.
            for produced_slot_id in produced_slot_ids & witness_slot_ids:
                witness_source_snapshots[produced_slot_id] = ceiling.guarded_clone(
                    value,
                    call_id=call.call_id,
                    slot_id=produced_slot_id,
                    affected_op_labels=call.op_labels,
                )
        op = _op_for_label(fork, op_label)
        if op is not None and bool(getattr(op, "has_saved_activation", False)):
            # Snapshot only what the capture saved: the result trace keeps the capture's
            # saved set, and an unsaved output (a huge ``expand`` view, say) is never
            # copied out. r59 gate 3: byte-guard the fork ``Op.out`` snapshot BEFORE it
            # materializes a tampered view (r58 free_2) and before the shape check below.
            op._internal_set(
                "out",
                ceiling.guarded_clone(
                    value,
                    call_id=call.call_id,
                    slot_id=slot_id,
                    affected_op_labels=(op_label,),
                ),
            )
        else:
            ceiling.check_unretained_extent(
                value, call_id=call.call_id, slot_id=slot_id, affected_op_labels=(op_label,)
            )
        shape_ok = tuple(value.shape) == slot.shape
        dtype_ok = str(value.dtype) == slot.dtype
        checks.append(
            _contract_check(
                f"slot_production:{slot_id}",
                True,
                RunnableErrorCode.SLOT_PRODUCTION_MISMATCH,
                f"Call {call.call_id!r} did not produce slot {slot_id!r}.",
                affected_op_labels=(op_label,),
            )
        )
        checks.append(
            _contract_check(
                f"output_shape:{slot_id}",
                shape_ok,
                RunnableErrorCode.OUTPUT_SHAPE_MISMATCH,
                f"Output slot {slot_id!r} has shape {tuple(value.shape)}, expected {slot.shape}.",
                affected_op_labels=(op_label,),
                details=(
                    ("slot_id", slot_id),
                    ("expected_shape", repr(slot.shape)),
                    ("actual_shape", repr(tuple(value.shape))),
                ),
            )
        )
        checks.append(
            _contract_check(
                f"output_dtype:{slot_id}",
                dtype_ok,
                RunnableErrorCode.OUTPUT_DTYPE_MISMATCH,
                f"Output slot {slot_id!r} has dtype {value.dtype}, expected {slot.dtype}.",
                affected_op_labels=(op_label,),
                details=(
                    ("slot_id", slot_id),
                    ("expected_dtype", slot.dtype),
                    ("actual_dtype", str(value.dtype)),
                ),
            )
        )
    checks.extend(
        _mutation_contract_checks(call, output, slot_values, before_versions, state_slot_ids)
    )
    return tuple(checks)


def _mutation_contract_checks(
    call: RunnableCallDescriptor,
    output: Any,
    slot_values: Mapping[str, torch.Tensor],
    before_versions: Mapping[str, int],
    state_slot_ids: frozenset[str] = frozenset(),
) -> tuple[ContractCheck, ...]:
    """Validate recorded in-place aliasing and tensor-version expectations.

    A NON-inplace call may legitimately bump the ``_version`` of a STATE buffer/parameter
    slot -- a mode-sensitive norm layer (InstanceNorm/BatchNorm with
    ``track_running_stats=True``) updates its ``running_mean``/``running_var`` running stats
    inside the functional ``instance_norm``/``batch_norm`` call, which TorchLens does not
    record as a separate in-place op (r29-C3, codex-F3). That running-stat update is a
    declared-state side effect reproduced identically on a fresh replay from the captured
    state, NOT the input/activation-tensor mutation this check guards against, so state slots
    are excluded from the non-inplace ``changed`` set. Value correctness of the updated buffer
    is separately enforced by state/output attestation, so this exclusion cannot mask a real
    numeric divergence.
    """

    changed = {
        slot_id
        for slot_id, before in before_versions.items()
        if slot_id in slot_values and tensor_version_or_none(slot_values[slot_id]) != before
    }
    if not call.is_inplace:
        non_state_changed = changed - state_slot_ids
        return (
            _contract_check(
                f"mutation:{call.call_id}",
                not non_state_changed,
                RunnableErrorCode.MUTATION_VERSION_MISMATCH,
                f"Non-mutating call {call.call_id!r} changed an input tensor version.",
                affected_op_labels=call.op_labels,
                details=(("changed_slot_ids", repr(tuple(sorted(non_state_changed)))),),
            ),
        )
    input_slot_id = _mutation_target_slot_id(call)
    if input_slot_id is None or not call.output_slot_ids:
        return (
            _contract_check(
                f"mutation:{call.call_id}",
                False,
                RunnableErrorCode.MUTATION_VERSION_MISMATCH,
                f"In-place call {call.call_id!r} lacks an input/output version relation.",
                affected_op_labels=call.op_labels,
            ),
        )
    input_value = slot_values.get(input_slot_id)
    try:
        output_value = _value_at_path(output, ()) if len(call.output_slot_ids) == 1 else output
        if len(call.output_slot_ids) > 1:
            output_value = _value_at_path(output, (0,))
    except (KeyError, IndexError, TypeError):
        output_value = None
    aliases = (
        isinstance(input_value, torch.Tensor)
        and isinstance(output_value, torch.Tensor)
        and (input_value is output_value or input_value.data_ptr() == output_value.data_ptr())
    )
    # Version leg: enforced only when a baseline exists. An inference tensor has no
    # version counter, so its in-place relation is proven by alias identity alone
    # (hon1_4); a versioned tensor keeps the full version-bump requirement.
    version_leg = input_slot_id in changed if input_slot_id in before_versions else True
    return (
        _contract_check(
            f"mutation:{call.call_id}",
            version_leg and aliases,
            RunnableErrorCode.MUTATION_VERSION_MISMATCH,
            f"In-place call {call.call_id!r} violated its alias/version expectation.",
            affected_op_labels=call.op_labels,
            details=(
                ("input_slot_id", input_slot_id),
                ("version_changed", repr(input_slot_id in changed)),
                ("output_aliases_input", repr(aliases)),
            ),
        ),
    )


def _mutation_target_slot_id(call: RunnableCallDescriptor) -> str | None:
    """Return the tensor slot expected to alias an in-place call's output.

    Parameters
    ----------
    call:
        Frozen runnable call descriptor.

    Returns
    -------
    str | None
        The explicit ``out=`` tensor slot when present, otherwise the first
        tensor argument for conventional in-place operators.
    """

    out_argument_slot_id = _out_argument_slot_id(call)
    if out_argument_slot_id is not None:
        return out_argument_slot_id
    return call.tensor_arguments[0].slot_id if call.tensor_arguments else None


def _out_argument_slot_id(call: RunnableCallDescriptor) -> str | None:
    """Return an explicit ``out=`` tensor slot, if the call has one.

    Parameters
    ----------
    call:
        Frozen runnable call descriptor.

    Returns
    -------
    str | None
        The explicit output tensor slot or ``None`` for conventional in-place
        calls that mutate their first tensor argument.
    """

    return next(
        (
            argument.slot_id
            for argument in call.tensor_arguments
            if argument.argument_path == ("kwargs", "out")
        ),
        None,
    )


def _reconstruct_output(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
    fork: Any,
    *,
    ceiling: RunResourceCeiling,
    call_outputs: Mapping[str, Any],
) -> Any:
    """Reconstruct the model-output container and populate synthetic output Ops."""

    output_slots = tuple(
        slot for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.OUTPUT
    )
    values: list[tuple[tuple[str | int, ...], torch.Tensor]] = []
    for slot in output_slots:
        source_id = slot.producer_slot_id or slot.version_of
        if source_id is None or source_id not in slot_values:
            raise RunPreconditionError(
                f"Output slot {slot.slot_id!r} has no produced source slot.",
                code=RunnableErrorCode.SLOT_PRODUCTION_MISMATCH.value,
                slot_id=slot.slot_id,
                remedy=(
                    "re-save the runnable artifact with the current torchlens "
                    "producer so every model-output slot names a produced source "
                    "slot; do not hand-edit producer_slot_id/version_of links"
                ),
            )
        value = slot_values[source_id]
        slot_values_dict = cast(dict[str, torch.Tensor], slot_values)
        slot_values_dict[slot.slot_id] = value
        op = _op_for_slot(fork, slot.slot_id)
        if op is not None:
            # r59 gate 3: byte-guard the reconstructed model-output snapshot too.
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
    container_spec = _output_container_spec(fork)
    if container_spec is not None:
        raw_output = _raw_runtime_output(descriptor, None, call_outputs)
        if (
            container_spec.type_module == "torch.return_types"
            and _torch_structseq_field_names(raw_output) == container_spec.fields
        ):
            return raw_output
        try:
            return rebuild_container_from_spec(container_spec, [value for _, value in values])
        except (TypeError, ValueError) as exc:
            raise RunPreconditionError(
                f"Recorded output container could not be reconstructed: {exc}",
                code=RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH.value,
                remedy=(
                    "re-capture with the output container type importable in this "
                    "process (same library versions) and re-save the runnable "
                    "artifact so its container spec matches the produced leaves"
                ),
            ) from exc
    if len(values) == 1 and not values[0][0]:
        # Genuine bare-tensor model output: nothing to reconstruct.
        return values[0][1]
    if values:
        # r35 I1 defense in depth: a multi-leaf output with NO recorded container
        # spec can only come from a legacy/tampered artifact (new saves are refused
        # unless losslessness is proved). Approximating the container here would be
        # a silent lossy substitution, so fail typed instead.
        raise RunPreconditionError(
            "Model output has multiple leaves but no recorded lossless container "
            "contract; this artifact predates (or violates) the v2 output "
            "losslessness proof and cannot be reconstructed faithfully.",
            code=RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT.value,
            remedy=(
                "re-capture and re-save the runnable artifact with the current "
                "torchlens producer (v2 schema), which records the lossless "
                "output-container contract at save time"
            ),
        )
    return _container_from_paths(values)


def _output_container_spec(trace: Any) -> ContainerSpec | None:
    """Return the shared recorded model-output container specification."""

    for label in getattr(trace, "output_layers", ()):
        op = _op_for_label(trace, label)
        spec = getattr(op, "container_spec", None)
        if isinstance(spec, ContainerSpec):
            return spec
    return None


def _output_not_reproduced(
    descriptor: SparseRunDescriptor, container_spec: ContainerSpec | None
) -> bool:
    """Return whether the captured model output was NOT reproduced by the sparse replay.

    A model whose forward returns a HOST-ESCAPED non-tensor Python scalar (``float(x.sum())``,
    ``x.item()``, ``int(...)``, ``bool(...)``) has NO output tensor slots AND no reconstructable
    output container spec, so the sparse DAG cannot emit that value: ``_reconstruct_output``
    returns a dropped ``None``. The escape-witness class applies to the OUTPUT too -- an output
    the replay never produced or compared must never be blessed VERIFIED. Stage-6 downgrades
    such a run to UNVERIFIABLE.

    A normal tensor output has >= 1 ``OUTPUT`` tensor slot; a container output (even one whose
    leaves are all literals) carries a ``ContainerSpec``; both are genuinely produced/compared
    and stay eligible for VERIFIED. Only the "no output slot AND no container spec" shape -- the
    host-escaped / dropped output -- is flagged here.

    Parameters
    ----------
    descriptor:
        Runnable descriptor whose tensor slots include the model-output slots.
    container_spec:
        The shared recorded output ``ContainerSpec`` (``None`` when none was recorded).

    Returns
    -------
    bool
        True when replay produced no output tensor and no reconstructable container.
    """

    has_output_slot = any(slot.role is TensorSlotRole.OUTPUT for slot in descriptor.tensor_slots)
    return not has_output_slot and container_spec is None


def _container_spec_reconstruction_lossy(spec: ContainerSpec | None) -> bool:
    """Return whether ``spec`` (or any nested child) reconstructs lossily.

    A dataclass / ``ModelOutput`` output whose live instance carried computed
    non-field/non-key state, a ``__slots__`` layout, or a data-descriptor field is
    flagged ``lossy_reconstruction`` at capture: the non-invoking rebuild cannot restore
    that state and must NOT be blessed VERIFIED. Any lossy node anywhere in the output
    container tree downgrades the whole run to UNVERIFIABLE.

    The persisted ``lossy_reconstruction`` flag is attacker-controlled in an untrusted bundle,
    so we NEVER trust a ``False`` alone: for every dataclass / ``hf_model_output`` node we ALSO
    recompute lossiness INDEPENDENTLY from the RESOLVED type at load time (``__slots__`` /
    data-descriptor field / dropped non-field state), so a forged ``lossy_reconstruction=False``
    cannot force a false VERIFIED. The persisted flag stays as a supplementary signal (kept for
    the purely instance-level custom-``ModelOutput`` case that is not type-observable at load),
    but it can only ADD lossiness, never suppress the independent recompute.
    """

    if spec is None:
        return False
    if _spec_node_reconstruction_lossy(spec):
        return True
    return any(_container_spec_reconstruction_lossy(child) for _, child in spec.child_specs)


def _spec_node_reconstruction_lossy(spec: ContainerSpec) -> bool:
    """Return whether one dataclass / ``hf_model_output`` node reconstructs lossily.

    Combines the persisted (supplementary) flag with an INDEPENDENT load-time recompute from
    the resolved type. A tampered spec naming a type that fails default-deny admissibility, or
    one whose type is not loaded (reconstruction then falls back to a plain namespace / mapping
    that is NOT the captured container type), is treated as lossy -- never a false VERIFIED.
    """

    if getattr(spec, "lossy_reconstruction", False):
        return True
    # r37 R11: the load-time recompute follows the capability table's per-kind
    # instance-state rule. ``type_recompute`` kinds re-derive lossiness from the
    # RESOLVED type so a forged ``lossy_reconstruction=False`` cannot suppress it;
    # builtin-stateless kinds short-circuit False; ``instance_refused`` /
    # ``declaration_required`` kinds are policed at save (their persisted flag is
    # supplementary and honest captures never produce a stateful instance).
    rule = CONTAINER_KIND_CAPABILITIES.get(spec.kind, {}).get("instance_state_rule")
    if rule != "type_recompute":
        return False
    if spec.kind == "namedtuple":
        try:
            container_type = resolve_container_type(spec)
        except ContainerReconstructionError:
            return True
        if container_type is None:
            # Unresolved namedtuple type: reconstruction substitutes a synthesized
            # type -- a lossy type substitution, never a false VERIFIED.
            return True
        # secB_1 forged-flag defense: a resolved namedtuple type that CAN carry
        # per-instance state (no ``__slots__ = ()``) is treated as lossy even when
        # the persisted flag is false. Plain ``collections.namedtuple`` /
        # ``typing.NamedTuple`` / slotted subclasses stay VERIFIED-eligible.
        # r49 secB_1: ALSO couple to reconstruction's substitution criterion -- a namedtuple
        # type that is neither a generated namedtuple nor a trusted structseq rebuilds as a
        # plain ``tuple`` (type substitution), which the instance-dict signal alone misses
        # (the r48 non-``FunctionType``-``__new__`` hole).
        return namedtuple_type_can_carry_instance_state(
            container_type
        ) or _reconstruction_would_substitute_plain(container_type, "namedtuple", spec.fields, spec)
    captured_names = spec.fields if spec.kind == "dataclass" else spec.keys
    try:
        container_type = resolve_container_type(spec)
    except ContainerReconstructionError:
        # A tampered / non-admissible type is refused at reconstruction; for the honesty
        # gate treat it as lossy so it can never be blessed VERIFIED.
        return True
    if container_type is None:
        # The captured type is not loaded, so reconstruction returns a plain namespace / mapping
        # rather than the recorded container type: a lossy substitution, never a false VERIFIED.
        return True
    return reconstruction_is_lossy_by_type(container_type, captured_names, spec.kind, spec)


def _fresh_bare_tensor_root(trace: Any) -> bool:
    """Return whether the fresh refresh proof positively attests a bare-tensor output root (r39).

    Consumes the ``_runnable_output_losslessness`` proof ``save_new_outs`` copies from the FRESH
    refresh forward onto the projected fork (corr2_5). The live bare-tensor fast path is honest
    ONLY when that proof is present, lossless, its root kind is exactly ``bare_tensor``, it walked
    a single leaf, and it used no fallback/duplicate traversal -- otherwise an opaque
    set/frozenset/custom container (which yields the same one-leaf/no-spec/no-path signature)
    could be wrongly blessed. A missing or malformed proof fails closed (returns ``False``).
    """

    runnable_state = getattr(trace, "__dict__", {}).get("_runnable")
    proof = getattr(runnable_state, "output_losslessness", None)
    if not isinstance(proof, Mapping):
        return False
    return (
        bool(proof.get("lossless"))
        and bool(proof.get("bare_tensor_root"))
        and proof.get("root_kind") == "bare_tensor"
        and int(proof.get("leaf_count", 0)) <= 1
        and not bool(proof.get("used_fallback"))
        and not bool(proof.get("duplicate_paths"))
    )
