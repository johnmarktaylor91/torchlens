"""Save-entry gate helpers for ``torchlens._io.bundle.save``.

Three entry-time guards that run before (or instead of) the per-trace save
pipeline: the two PERMANENT erasure-prevention invariants (edge-substitution
carriers and the shard-local disclosure, both keyed on the ACTIVE SCHEMA so a
regression re-fires them), and the Bundle container-door delegation that
routes ``bundle.save(path, ...)`` through ``Bundle.save`` (foldB D8/D9); plus
the blob-loop refusal of a tensor a strict (portable) save cannot write. Split
out of ``torchlens/_io/bundle.py`` to respect its size ceiling (R43
split-never-raise); ``bundle.py`` re-exports these names, so the monkeypatch
surface ``torchlens._io.bundle._refuse_*`` is unchanged.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

import torch

from .._runnable import (
    ActivationPayloadMember,
    RunnableErrorCode,
    SlotByteDigest,
    StateByteDigest,
)
from .._runnable.input_aliases import build_input_attestation_fingerprint
from .._runnable.state import runnable_tensor_byte_digest
from ..errors._base import TorchLensWarning
from ..errors.runnable import RunnablePreflightError
from .format_errors import ArtifactError
from .scrub import BlobSpec

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from .._runnable import SparseRunDescriptor

from .._data_classes._trace_missing_attrs import is_live_trace
from .._errors import InvalidArgumentError
from .._runnable import RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION, PayloadLayerDescriptor
from .format3.runnable import _WriteContext
from .runnable import (
    require_sparse_run_descriptor,
    sparse_descriptor_to_json,
    with_activation_payload,
    with_weight_payload,
)

#: Save preparation shares the bundle assembly stratum.
__tl_layer__ = "L3"


def _refuse_edge_intervened_save(trace: Any) -> None:
    """PERMANENT erasure-prevention invariant for edge substitutions (L6 4.3).

    The tlspec v8 coordinated bump persists ``Op.edge_substitutions``
    (BLOB_RECURSIVE) and ``Op.edge_replacement_stamps`` (KEEP), so ordinary
    saves of edge-intervened traces now proceed. Like the shard-local guard
    below, the predicate keys on the ACTIVE SCHEMA rather than being deleted:
    it refuses typed IFF tier-(ii) entries are present AND the active policy
    would drop them (a schema regression re-dropping the carrier, with the S3
    switch inactive). Any such regression re-fires this refusal instead of
    silently presenting post-edit values with zero edge provenance.
    """

    from .._data_classes.trace import Trace

    if not isinstance(trace, Trace):
        # A Recording or other non-Trace save subject holds no Ops, so it
        # cannot carry edge substitutions; its own refusal fires later.
        return
    carriers = [
        op.label
        for op in (trace._layer_list if is_live_trace(trace) else ()) or ()
        if getattr(op, "edge_substitutions", None)
    ]
    if not carriers:
        return
    from .._data_classes.op import Op as _Op
    from . import FieldPolicy
    from .prerelease import prerelease_fields_active

    policy_entry = _Op._FIELD_POLICY.get("edge_substitutions")
    portable_policy = getattr(policy_entry, "portable_policy", policy_entry)
    if portable_policy is not None and portable_policy is not FieldPolicy.DROP:
        return
    if prerelease_fields_active():
        return
    raise InvalidArgumentError(
        "this trace carries edge-substitution interventions, and the active "
        "schema has no occurrence-granular carrier at ANY save level: the "
        "artifact would present post-edit values with zero edge provenance. "
        "Edge-intervened traces are session-only under such a schema.",
        code="edge_intervention_save_unsupported",
        remedy="analyze in-session, or re-capture without the edge edit before saving",
        carriers=tuple(carriers),
    )


def _refuse_shard_local_erasure(trace: Any) -> None:
    """PERMANENT erasure-prevention invariant for the shard-local disclosure.

    L8/F6 (census plan 3.2b): an ordinary bundle write must NEVER complete if
    it would silently drop the shard-local disclosure -- a saved shard-local
    trace reloading as a plain dense-looking trace is the marker-free-artifact
    class this invariant keeps EMPTY BY CONSTRUCTION. Refuses typed IFF the
    trace carries ``distributed_scope == "rank_local_shard"`` AND the marker
    is not persisted by the active schema (still ``FieldPolicy.DROP`` with the
    S3 pre-release switch inactive). The wave-3 coordinated bump changes the
    ENVIRONMENT, not this predicate: once the policy persists, the second
    conjunct goes false by construction and ordinary saves proceed; the code,
    predicate, and forced-DROP tamper red all REMAIN so any schema regression
    that would re-drop the disclosure re-fires the invariant. Never deleted,
    never a narrowing, no D-ruling owed.
    """

    if (
        trace._provenance_distributed_scope if is_live_trace(trace) else None
    ) != "rank_local_shard":
        return
    from .._data_classes.trace import Trace as _Trace
    from . import FieldPolicy
    from .prerelease import prerelease_fields_active

    policy_entry = _Trace._FIELD_POLICY.get("_provenance_distributed_scope")
    portable_policy = getattr(policy_entry, "portable_policy", policy_entry)
    if portable_policy is not None and portable_policy is not FieldPolicy.DROP:
        return
    if prerelease_fields_active():
        return
    raise InvalidArgumentError(
        "this trace is a shard-local capture (distributed_scope == "
        "'rank_local_shard'), and the active schema does not persist the "
        "shard-local disclosure: an ordinary save would reload as a plain "
        "trace with silently mis-stated parameter geometry. Shard-local "
        "traces are session-only until the coordinated schema bump persists "
        "the marker and its dual-geometry evidence.",
        code="shard_local_persistence_unsupported",
        remedy="analyze in-session; persistence lands with the coordinated schema bump",
    )


# trace.save() defaults for the per-trace payload options the Bundle container
# door does not offer (its surface is exactly path/level/overwrite). An
# EXPLICIT non-default value on a bundle save refuses typed below instead of
# being silently dropped.
_BUNDLE_TRACE_ONLY_SAVE_DEFAULTS: dict[str, bool | None] = {
    "include_outs": None,
    "include_grads": None,
    "include_saved_args": None,
    "include_rng_states": None,
    "include_weights": False,
    "include_activations": False,
    "include_source": True,
    "include_custom_attributes": True,
    "include_buffer_values": None,
    "strict": True,
}


def _save_bundle_via_container_door(
    bundle: Any,
    path: str | Path,
    *,
    level: str,
    overwrite: bool,
    trace_only_options: dict[str, bool | None],
) -> None:
    """Route ``bundle.save(path, ...)`` through the one container door.

    Both public save doors resolve to ``Bundle.save`` (the container door),
    so they agree on every settled bundle by construction: the N1 capture-
    outcome gate runs PER MEMBER inside each member's own save (foldB D8/D9),
    never against the container. Per-trace payload options the container door
    cannot honor refuse typed rather than being silently dropped.

    Parameters
    ----------
    bundle:
        The ``torchlens.bundle.Bundle`` instance passed to ``Trace.save``.
    path:
        Destination ``.tlspec`` directory path.
    level:
        Save level forwarded to the container door.
    overwrite:
        Whether an existing destination may be replaced.
    trace_only_options:
        The received values of every per-trace-only ``Trace.save`` option,
        keyed by parameter name.

    Raises
    ------
    InvalidArgumentError
        ``bundle_save_option_unsupported`` when a per-trace payload option
        was set away from its default.
    """

    unsupported = sorted(
        name
        for name, value in trace_only_options.items()
        if value != _BUNDLE_TRACE_ONLY_SAVE_DEFAULTS[name]
    )
    if unsupported:
        raise InvalidArgumentError(
            "bundle.save(path, ...) routes through the Bundle container door, "
            "which supports only 'level' and 'overwrite'; it cannot honor "
            f"per-trace save options: {', '.join(unsupported)}.",
            code="bundle_save_option_unsupported",
            remedy=(
                "Drop the per-trace options (or call bundle.save(path, "
                "level=..., overwrite=...)); for per-member payload control, "
                "save one member with bundle[name].save(path, ...)."
            ),
            options=tuple(unsupported),
        )
    bundle.save(path, level=level, overwrite=overwrite)


def _validated_descriptor_for_resave(trace: Any) -> SparseRunDescriptor | None:
    """Return the descriptor a loaded trace resaves from, or ``None`` when the producer must run.

    ``Op.args_template`` is runtime-only and never persists, so a trace loaded from an
    artifact cannot re-cook its call recipe; the descriptor the load validated is that
    recipe, carried through exactly. Its optional payload layers reset to absent so the
    save stamps only what it writes. A trace whose ops carry a cooked template (a live
    capture, or a loaded trace refreshed by a rerun) builds afresh, and a trace with neither
    still runs the producer and takes its typed refusal: the recipe is never optional.
    """

    descriptor = trace._runnable.descriptor
    if descriptor is None or not getattr(trace, "_loaded_from_bundle", False):
        return None
    if any(getattr(op, "args_template", None) is not None for op in trace._layer_list):
        return None
    layers = descriptor.payload_layers
    return replace(
        descriptor,
        payload_layers=replace(
            layers,
            weights=replace(layers.weights, present=False),
            activations=PayloadLayerDescriptor(
                present=False, schema=RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION
            ),
        ),
    )


def _stamped_descriptor(
    trace: Any, *, include_weights: bool, include_activations: bool
) -> tuple[Any, list[Any], list[Any], list[Any]]:
    """Produce and payload-stamp the descriptor exactly as the tlspec 11 save did.

    Returns the descriptor and the buffer, weight and activation blob specs.
    """

    descriptor = _validated_descriptor_for_resave(trace) or require_sparse_run_descriptor(trace)
    buffer_specs = _capture_nonpersistent_buffer_blob_specs(trace, descriptor)
    if buffer_specs:
        _warn_nonpersistent_buffer_disclosure_once()
    weight_specs: list[Any] = []
    activation_specs: list[Any] = []
    if include_weights:
        weight_specs = _capture_weight_blob_specs(trace, descriptor)
        descriptor = with_weight_payload(descriptor)
    if include_activations:
        (activation_specs, members, input_digests, state_digests, fingerprints) = (
            _capture_activation_blob_specs(trace, descriptor)
        )
        if members and not input_digests:
            _warn_unattestable_activation_archive_once()
        descriptor = with_activation_payload(
            descriptor,
            members=members,
            original_input_digests=input_digests,
            capture_state_digests=state_digests,
            input_fingerprints=fingerprints,
        )
    return descriptor, buffer_specs, weight_specs, activation_specs


def _runnable_write_context(
    prepared: tuple[Any, list[Any], list[Any], list[Any]], counter: list[int]
) -> _WriteContext:
    """Project a prepared runnable save onto the pure wire carrier's input."""

    descriptor, buffers, weights, activations = prepared
    layers = descriptor.payload_layers
    return _WriteContext(
        counter=counter,
        descriptor=sparse_descriptor_to_json(descriptor),
        families=(
            ("state_dict", weights, str(layers.weights.schema), False),
            ("buffers", buffers, str(layers.nonpersistent_buffers.schema), False),
            ("activations", activations, str(layers.activations.schema), True),
        ),
    )


_RUNNABLE_WEIGHT_KIND = "runnable_weight"
_RUNNABLE_NONPERSISTENT_BUFFER_KIND = "runnable_nonpersistent_buffer"
_RUNNABLE_ACTIVATION_KIND = "runnable_activation"


def _require_canonical_runnable_labels(names: Iterable[object], *, family: str) -> None:
    """Refuse a runnable payload family whose entry labels load would reject.

    r79 (r78 free LOW): ``self._parameters[""] = nn.Parameter(...)`` bypasses
    ``register_parameter``'s empty-name validation, captures, and SAVED an
    artifact whose weight tensor entry carried label ``""`` -- which
    ``validate_tlspec`` categorically refuses at load ("requires a canonical
    label"). A save door must never produce a stillborn artifact, so this
    preflight mirrors the load-side canonical-label predicate exactly
    (non-``str`` or empty refuses; nothing wider).

    Parameters
    ----------
    names:
        Candidate canonical entry labels for one payload family.
    family:
        Human-readable payload family name for the refusal message.

    Raises
    ------
    RunnablePreflightError
        Typed save refusal for any label the load door would refuse.
    """

    for name in names:
        if not isinstance(name, str) or name == "":
            raise RunnablePreflightError(
                f"Runnable {family} entry requires a canonical label; got {name!r}. "
                "An artifact with this label would be refused wholesale at load "
                "(empty state names bypass register_parameter/register_buffer "
                "validation and have no canonical state_dict identity).",
                code=RunnableErrorCode.SPARSE_PREFLIGHT_FAILED.value,
            )


def _capture_weight_blob_specs(
    trace: Trace,
    descriptor: SparseRunDescriptor,
) -> list[BlobSpec]:
    """Collect one full source-model state dict as runnable weight blobs.

    Parameters
    ----------
    trace:
        Live capture Trace retaining a weak reference to its source model.
    descriptor:
        Sparse descriptor whose canonical names, roles, and aliases define the
        strict state contract.

    Returns
    -------
    list[BlobSpec]
        Separately named state-only blobs ordered by canonical state name.

    Raises
    ------
    ArtifactError
        If the live source model or its state mapping is unavailable.
    StateBindingError
        If the full state dict disagrees with the descriptor contract.
    RunnablePreflightError
        If a state name is not a canonical (non-empty) label (r79 save-side
        mirror of the load door's canonical-label check).
    """

    from .._runnable.state import validate_state_mapping_for_descriptor

    state = _capture_source_state(trace, option_name="include_weights")
    validate_state_mapping_for_descriptor(descriptor, state)
    _require_canonical_runnable_labels(
        cast(Mapping[str, torch.Tensor], state).keys(), family="weight tensor"
    )
    return [
        BlobSpec(
            blob_id=f"runnable_weight_{index:05d}",
            value=state[name].detach().clone(),
            kind=_RUNNABLE_WEIGHT_KIND,
            label=name,
            logical_backend="torch",
        )
        for index, name in enumerate(sorted(cast(Mapping[str, torch.Tensor], state)))
    ]


def _capture_nonpersistent_buffer_blob_specs(
    trace: Trace,
    descriptor: SparseRunDescriptor,
) -> list[BlobSpec]:
    """Collect used non-persistent buffers as mandatory runnable payloads.

    Parameters
    ----------
    trace:
        Live capture Trace retaining the captured initial buffer values.
    descriptor:
        Sparse descriptor identifying buffer bindings excluded from ``state_dict``.

    Returns
    -------
    list[BlobSpec]
        One separately stored value blob per used non-persistent buffer name.

    Raises
    ------
    StateBindingError
        If a required captured value is missing or violates its slot contract.
    RunnablePreflightError
        If a buffer name is not a canonical (non-empty) label (r79 save-side
        mirror of the load door's canonical-label check).
    """

    from .._runnable.state import validate_nonpersistent_buffer_mapping_for_descriptor

    names = sorted(
        {
            slot.state_binding.state_dict_name
            for slot in descriptor.tensor_slots
            if slot.role.value == "buffer"
            and slot.state_binding is not None
            and not slot.state_binding.persistent
        }
    )
    _require_canonical_runnable_labels(names, family="non-persistent buffer")
    captured_values = getattr(trace, "_buffer_initial_values", {}) or {}
    values = {name: captured_values[name] for name in names if name in captured_values}
    validate_nonpersistent_buffer_mapping_for_descriptor(descriptor, values)
    return [
        BlobSpec(
            blob_id=f"runnable_nonpersistent_buffer_{index:05d}",
            value=cast(torch.Tensor, values[name]).detach().clone(),
            kind=_RUNNABLE_NONPERSISTENT_BUFFER_KIND,
            label=name,
            logical_backend="torch",
        )
        for index, name in enumerate(names)
    ]


def _captured_input_attestation(slot_id: str, out: torch.Tensor) -> tuple[SlotByteDigest, Any]:
    """Build the original-input digest and physical fingerprint from the retained value."""

    digest = runnable_tensor_byte_digest(out)
    # The live retained input seeded the forward; serialized payloads lose strides.
    # The run side fingerprints its executed clone on this same basis.
    return (
        SlotByteDigest(slot_id=slot_id, byte_digest=digest),
        build_input_attestation_fingerprint(slot_id, out, byte_digest=digest),
    )


# Complexity: keep activation archive membership and slot witnesses in one preflight.
def _capture_activation_blob_specs(
    trace: Trace,
    descriptor: SparseRunDescriptor,
) -> tuple[
    list[BlobSpec],
    tuple[ActivationPayloadMember, ...],
    tuple[SlotByteDigest, ...],
    tuple[StateByteDigest, ...],
    tuple[Any, ...],
]:
    """Collect capture-selected activation blobs and attestation eligibility digests.

    Parameters
    ----------
    trace:
        Completed capture retaining the existing ``save=`` payload decisions.
    descriptor:
        Sparse descriptor whose slots and calls identify archived membership.

    Returns
    -------
    tuple
        Blob specs, exact activation membership, original-input digests available
        from selected input payloads, and capture-state digests.

    Raises
    ------
    ArtifactError
        If a selected payload is unavailable or is not a dense torch tensor.
    """

    slot_ids = {slot.slot_id for slot in descriptor.tensor_slots}
    call_id_by_label = {
        op_label: call.call_id for call in descriptor.calls for op_label in call.op_labels
    }
    blob_specs: list[BlobSpec] = []
    members: list[ActivationPayloadMember] = []
    input_digests: list[SlotByteDigest] = []
    input_fingerprints: list[Any] = []
    for op in trace._layer_list:
        op_label = str(op.label)
        slot_id = f"slot:{op_label}"
        if slot_id not in slot_ids:
            continue
        out = _physical_op_payload(op, "out")
        if bool(getattr(op, "is_input", False)) and isinstance(out, torch.Tensor):
            evidence, fingerprint = _captured_input_attestation(slot_id, out)
            input_digests.append(evidence)
            input_fingerprints.append(fingerprint)
        if not bool(getattr(op, "has_saved_activation", False)):
            continue
        found_payload = False
        for field_name in ("out", "transformed_out"):
            payload = _physical_op_payload(op, field_name)
            if payload is None:
                continue
            found_payload = True
            if not isinstance(payload, torch.Tensor) or payload.layout is not torch.strided:
                raise ArtifactError(
                    "include_activations=True requires every selected out/transformed_out "
                    f"payload to be a dense torch.Tensor; {op_label}.{field_name} is "
                    f"{type(payload).__name__}.",
                    code="artifact_tensor_unsupported",
                )
            blob_id = f"runnable_activation_{len(blob_specs):05d}"
            digest = runnable_tensor_byte_digest(payload)
            blob_specs.append(
                BlobSpec(
                    blob_id=blob_id,
                    value=payload,
                    kind=_RUNNABLE_ACTIVATION_KIND,
                    label=op_label,
                    logical_backend="torch",
                )
            )
            members.append(
                ActivationPayloadMember(
                    blob_id=blob_id,
                    slot_id=slot_id,
                    call_id=call_id_by_label.get(op_label),
                    op_label=op_label,
                    field=cast(Any, field_name),
                    byte_digest=digest,
                )
            )
        if not found_payload:
            raise ArtifactError(
                "include_activations=True found a capture-selected activation without a "
                f"materialized out/transformed_out payload at {op_label!r}.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            )

    state = _capture_source_state(trace, option_name="include_activations")
    state_digests = tuple(
        StateByteDigest(
            state_dict_name=name,
            byte_digest=runnable_tensor_byte_digest(value),
        )
        for name, value in sorted(cast(Mapping[str, torch.Tensor], state).items())
    )
    return (
        blob_specs,
        tuple(members),
        tuple(input_digests),
        state_digests,
        tuple(input_fingerprints),
    )


def _physical_op_payload(op: Any, field_name: str) -> Any:
    """Read one stored Op payload without invoking missing-payload access errors."""

    slot = getattr(op, "_slot", None)
    if callable(slot):
        return slot(field_name)
    return getattr(op, field_name, None)


def _capture_source_state(trace: Trace, *, option_name: str) -> Mapping[str, torch.Tensor]:
    """Return the snapshot of the model state used by the captured forward pass.

    The snapshot is taken at the runnable capture boundary, before user model
    execution. It must never be replaced with a save-time read of the live
    model because that could embed a drifted state under the capture-state
    label.
    """

    state = trace._runnable.capture_state
    if state is None and not is_live_trace(trace):
        # The reader binds this family only after validating its capture-state
        # slot universe and byte digests. User-staged state is not capture state.
        state = trace._runnable.embedded_state
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ArtifactError(
            f"{option_name}=True requires a tensor-only capture-time state snapshot. "
            "This Trace cannot embed state as embedded_capture_state.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        )
    return cast(Mapping[str, torch.Tensor], state)


_NONPERSISTENT_DISCLOSURE_WARNED = False
"""One-time process flag for the non-persistent buffer save disclosure."""

_UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED = False
"""One-time process flag for the unattestable-activation-archive save disclosure (r6 H2)."""


def _warn_nonpersistent_buffer_disclosure_once() -> None:
    """Emit the one-time REQUIRED-family privacy disclosure (contract section 5).

    A default runnable save of a model with used non-persistent buffers carries
    their capture-time tensor values in the required
    ``runnable_nonpersistent_buffer_v1`` family even with both include flags
    false -- declared state without which the artifact cannot replay. The family
    is manifest-visible; this warning makes the disclosure active.
    """

    global _NONPERSISTENT_DISCLOSURE_WARNED
    if _NONPERSISTENT_DISCLOSURE_WARNED:
        return
    _NONPERSISTENT_DISCLOSURE_WARNED = True
    warnings.warn(
        "This runnable save includes capture-time values of used NON-persistent "
        "buffers (the required runnable_nonpersistent_buffer_v1 family): they are "
        "declared state the artifact cannot replay without, and they are written "
        "even with include_weights/include_activations false. Review the buffers "
        "before sharing the artifact if they may hold sensitive data.",
        TorchLensWarning,
        stacklevel=3,
    )


def _warn_unattestable_activation_archive_once() -> None:
    """Disclose an activation archive saved without any original-input eligibility (r6 H2).

    ``_capture_activation_blob_specs`` can only record ``original_input_digests`` /
    ``input_fingerprints`` for a model input that the capture actually SELECTED. With a
    ``save=`` selector -- the recommended idiom -- the input op is usually unselected, so
    the archive ships blobs and digests but ZERO input eligibility, and numeric attestation
    is permanently ``NOT_APPLICABLE`` no matter what the caller later passes to ``.run()``.
    Left silent that is indistinguishable from "you changed the input", so disclose it at
    SAVE time. The run report carries the matching named
    ``numeric_attestation:not_applicable:no_recorded_original_input_eligibility`` entry.

    This is the behavior-preserving half only: the status itself is deliberately NOT
    promoted to ``ATTESTED``.
    """

    global _UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED
    if _UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED:
        return
    _UNATTESTABLE_ACTIVATION_DISCLOSURE_WARNED = True
    warnings.warn(
        "This runnable save archives activations but records no original-input "
        "digests/fingerprints, because the capture selection did not include the model "
        "input. Numeric attestation on a later .run() will report not_applicable for that "
        "reason -- not because the inputs changed. Capture the model input alongside the "
        "selected activations if byte-exact attestation is wanted.",
        TorchLensWarning,
        stacklevel=3,
    )


def _refuse_unsupported_tensor(error_text: str, blob_spec: Any, reason: str) -> NoReturn:
    """Refuse a torch tensor the strict (portable) save cannot write.

    Parameters
    ----------
    error_text:
        Refusal text naming the blob's label, kind and the codec's reason.
    blob_spec:
        The refused blob (its ``label`` and ``kind`` ride on the error).
    reason:
        The payload codec's refusal reason.

    Raises
    ------
    ArtifactError
        With ``fields["code"] == "artifact_tensor_unsupported"``.
    """

    raise ArtifactError(
        error_text,
        code="artifact_tensor_unsupported",
        remedy="save at level='audit', or make the tensor dense and strided",
        label=blob_spec.label,
        kind=blob_spec.kind,
        reason=reason,
    )
