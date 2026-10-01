"""Control, shape, and host-escape witness checks."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ._runnable_state import (
    runnable_tensor_byte_digest,
)
from .ir.container import (
    ContainerSpec,
)
from .runnable import (
    ControlWitness,
    ControlWitnessKind,
    RunnableCallDescriptor,
    SparseRunDescriptor,
)

if TYPE_CHECKING:
    from ._runnable_execution import (
        _STATE_METADATA_FACT_SITE_PREFIX,
        _UNBOUND_STATE_ESCAPE_FACT_KEY,
        _UNBOUND_STATE_ESCAPE_SITE_PREFIX,
        _codec_component,
        _decode_literal,
        _op_for_label,
        _registered_flatten_children,
        _value_at_path,
    )

__all__ = (
    "_tensor_leaf_paths",
    "_canonicalize_structseq_output_paths",
    "_canonicalize_structseq_output_path",
    "_recorded_structseq_output_type_matches",
    "_call_has_recorded_torch_structseq_output",
    "_container_leaf_paths",
    "_container_kind",
    "_is_hf_model_output",
    "_container_field_names",
    "_torch_structseq_field_names",
    "_scalar_literal_equal",
    "_tensor_derived_scalar_witness_slot_ids",
    "_tensor_derived_scalar_stale",
    "_is_unbound_state_escape_witness",
    "_is_state_metadata_fact_witness",
    "_unbound_state_escape_stale",
)


def _hybrid_registered_flatten_children(value: Any) -> list[Any] | None:
    """Flatten children for a GENERIC-TYPED registered container, else ``None``.

    ``_registered_flatten_children`` deliberately short-circuits Mapping /
    ``list`` / ``tuple`` instances before the registry lookup, so a HYBRID
    registered container (a registered namedtuple, list, dict, or tuple
    subclass) never reached the registered lane at run time while the
    capture-side ``ContainerSpec`` builder dispatches registered-FIRST with
    no generic exclusion (T11.6). This covers exactly that excluded slice; a
    throwing flatten returns ``None`` (the caller falls back to the generic
    walk, same as the sibling helper).
    """

    if not isinstance(value, (Mapping, list, tuple)):
        return None
    from torchlens.ir.container import get_registered_container

    registration = get_registered_container(type(value))
    if registration is None:
        return None
    try:
        return list(registration.flatten(value)[0])
    except Exception:
        return None


def _tensor_leaf_paths(
    value: Any, path: tuple[str | int, ...] = ()
) -> tuple[tuple[str | int, ...], ...]:
    """Return ordered tensor-leaf paths for a runtime container."""

    if isinstance(value, torch.Tensor):
        return (path,)
    registered_children = _registered_flatten_children(value)
    if registered_children is None:
        # T11.6: _registered_flatten_children excludes generic-typed values
        # (Mapping/list/tuple), but a HYBRID registered container must walk its
        # registration's OWN flatten children exactly as capture did -- generic
        # indexing only matches when flatten happens to preserve order. Local
        # import: this function is rebound into _runnable_execution's globals.
        from torchlens._runnable_witness_contracts import (
            _hybrid_registered_flatten_children,
        )

        registered_children = _hybrid_registered_flatten_children(value)
    if registered_children is not None:
        paths: list[tuple[str | int, ...]] = []
        for index, child in enumerate(registered_children):
            paths.extend(_tensor_leaf_paths(child, (*path, index)))
        return tuple(paths)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        paths = []
        for field in dataclasses.fields(value):
            paths.extend(_tensor_leaf_paths(getattr(value, field.name), (*path, field.name)))
        return tuple(paths)
    field_names = _container_field_names(value)
    if field_names:
        paths = []
        for name in field_names:
            paths.extend(_tensor_leaf_paths(getattr(value, name), (*path, str(name))))
        return tuple(paths)
    if isinstance(value, Mapping):
        paths = []
        # r67 C2 (hon1-F1): mapping keys route through the ONE type-strict codec so a
        # tensor child under a grammar key (2.5 / None / bool / safe tuple) enters the
        # leaf accounting; a non-grammar key is skipped here (its subtree is already
        # opaque-ceilinged / save-refused by the structure spine).
        for key, child in value.items():
            component = _codec_component(key)
            if component is not None:
                paths.extend(_tensor_leaf_paths(child, (*path, component)))
        return tuple(paths)
    if isinstance(value, (list, tuple)):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_tensor_leaf_paths(child, (*path, index)))
        return tuple(paths)
    return ()


def _canonicalize_structseq_output_paths(
    output: Any,
    paths: Sequence[Sequence[str | int]],
) -> tuple[tuple[str | int, ...], ...]:
    """Canonicalize only ``torch.return_types`` named/positional path components.

    Parameters
    ----------
    output:
        Runtime output container used to interpret path components.
    paths:
        Tensor leaf paths to canonicalize.

    Returns
    -------
    tuple[tuple[str | int, ...], ...]
        Paths where field names and positional indexes are equivalent only while
        traversing a ``torch.return_types.*`` structseq.
    """

    return tuple(_canonicalize_structseq_output_path(output, tuple(path)) for path in paths)


def _canonicalize_structseq_output_path(
    output: Any,
    path: tuple[str | int, ...],
) -> tuple[str | int, ...]:
    """Canonicalize one path through runtime ``torch.return_types`` containers.

    Parameters
    ----------
    output:
        Runtime output container used to interpret path components.
    path:
        Tensor leaf path to canonicalize.

    Returns
    -------
    tuple[str | int, ...]
        Path with structseq fields represented by their positional index.
    """

    current = output
    canonical: list[str | int] = []
    for component in path:
        canonical_component = component
        field_names = _torch_structseq_field_names(current)
        if field_names:
            if isinstance(component, str) and component in field_names:
                canonical_component = field_names.index(component)
            elif isinstance(component, int) and 0 <= component < len(field_names):
                canonical_component = component
        canonical.append(canonical_component)
        try:
            current = _value_at_path(current, (canonical_component,))
        except (AttributeError, KeyError, IndexError, TypeError):
            break
    return tuple(canonical)


def _recorded_structseq_output_type_matches(
    trace: Any,
    call: RunnableCallDescriptor,
    output: Any,
) -> bool:
    """Return whether a recorded torch structseq call produced a torch structseq.

    Parameters
    ----------
    trace:
        Runtime fork containing recorded op metadata.
    call:
        Runnable call descriptor being bound.
    output:
        Runtime output produced by the resolved callable.

    Returns
    -------
    bool
        False only when the recorded call's output container was a
        ``torch.return_types.*`` structseq but runtime produced another tuple
        shape, such as a plain positional tuple.
    """

    if not _call_has_recorded_torch_structseq_output(trace, call):
        return True
    return _torch_structseq_field_names(output) != ()


def _call_has_recorded_torch_structseq_output(
    trace: Any,
    call: RunnableCallDescriptor,
) -> bool:
    """Return whether any call output op recorded a torch structseq container.

    Parameters
    ----------
    trace:
        Runtime fork containing recorded op metadata.
    call:
        Runnable call descriptor being inspected.

    Returns
    -------
    bool
        True when any output op for the call has a ``torch.return_types`` root
        container specification.
    """

    for op_label in call.op_labels:
        op = _op_for_label(trace, op_label)
        container_spec = getattr(op, "container_spec", None)
        if (
            isinstance(container_spec, ContainerSpec)
            and container_spec.type_module == "torch.return_types"
        ):
            return True
    return False


def _container_leaf_paths(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[str | int, ...], ...]:
    """Return producer-compatible tensor-leaf paths for a boundary container."""

    if isinstance(value, torch.Tensor):
        return (path,)
    registered_children = _registered_flatten_children(value)
    if registered_children is None:
        # T11.6: same hybrid registered-first rule (and same rebind-safe local
        # import) as _tensor_leaf_paths.
        from torchlens._runnable_witness_contracts import (
            _hybrid_registered_flatten_children,
        )

        registered_children = _hybrid_registered_flatten_children(value)
    if registered_children is not None:
        paths: list[tuple[str | int, ...]] = []
        for index, child in enumerate(registered_children):
            paths.extend(_container_leaf_paths(child, (*path, index)))
        return tuple(paths)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        paths = []
        for field in dataclasses.fields(value):
            paths.extend(_container_leaf_paths(getattr(value, field.name), (*path, field.name)))
        return tuple(paths)
    field_names = _container_field_names(value)
    if field_names:
        paths = []
        for name in field_names:
            paths.extend(_container_leaf_paths(getattr(value, name), (*path, str(name))))
        return tuple(paths)
    if isinstance(value, Mapping):
        paths = []
        # r67 C2 (hon1-F1): same ONE type-strict key codec as the capture-side path
        # normalization -- lockstep by construction.
        for key, child in value.items():
            component = _codec_component(key)
            if component is not None:
                paths.extend(_container_leaf_paths(child, (*path, component)))
        return tuple(paths)
    if isinstance(value, (list, tuple)):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_container_leaf_paths(child, (*path, index)))
        return tuple(paths)
    return ()


def _container_kind(value: Any) -> str:
    """Return the sparse witness vocabulary name for a runtime container.

    Dispatch order mirrors the capture-side ``ContainerSpec`` builder
    (T11.6): REGISTERED wins over every generic kind, exactly as it does at
    capture and in the sibling leaf-path walkers above. The r67 C2 fix added
    the registered branch BELOW the generic kinds, so a HYBRID registered
    container (a registered dataclass/namedtuple/tuple subclass) recorded
    ``"registered"`` at capture but reported its generic kind here -- an
    honest identical run then false-DIVERGED with OUTPUT_STRUCTURE_MISMATCH.
    """

    from torchlens.ir.container import get_registered_container

    if isinstance(value, torch.Tensor):
        return "tensor"
    # Registration is checked by TYPE (mirroring classify_input_container and the
    # capture-side spec builder), NOT through _registered_flatten_children, whose
    # generic-type exclusion (Mapping/list/tuple) is exactly what hid hybrids.
    if not isinstance(value, type) and get_registered_container(type(value)) is not None:
        return "registered"
    # HF ModelOutput subclasses ARE dataclasses, so the HF check must run
    # FIRST, exactly as the capture-side spec builder dispatches (F32 rider:
    # the inverted order classified a real GPT-2 CausalLMOutput as
    # "dataclass" against its recorded "hf_model_output" witness, so every
    # honest identical run false-DIVERGED with OUTPUT_STRUCTURE_MISMATCH --
    # the same bug class as the r67 C2 registered-branch fix above).
    if _is_hf_model_output(value):
        return "hf_model_output"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return "dataclass"
    if _container_field_names(value):
        return "namedtuple"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "dict"
    return type(value).__name__


def _is_hf_model_output(value: Any) -> bool:
    """Return whether ``value`` looks like a HuggingFace ``ModelOutput``."""

    cls = type(value)
    if any(
        base.__module__.startswith("transformers") and base.__name__ == "ModelOutput"
        for base in cls.__mro__
    ):
        return True
    return (
        (cls.__module__.startswith("transformers") or cls.__name__.endswith("ModelOutput"))
        and hasattr(value, "keys")
        and hasattr(value, "__getitem__")
    )


def _container_field_names(value: Any) -> tuple[str, ...]:
    """Return the stable field names for namedtuple-like runtime containers.

    Routes namedtuple field resolution through the ONE capture-side authority,
    ``_input_walk._instance_fields`` (raw MRO, tuple-of-str required). The historical
    ``hasattr(value, "_fields")`` + ``tuple(value._fields)`` spelling was a LIVE instance
    read that executed a user property and accepted any iterable, so the same container
    was field-addressable to this walker and zero-field to the capture walker: the
    runtime contract check then crashed with an untyped ``AttributeError`` (only
    ``KeyError``/``IndexError``/``TypeError`` were guarded downstream), and on shapes
    that did not crash the two walkers keyed the same leaf under different paths, so the
    leaf-path set contract compared apples to oranges.

    Parameters
    ----------
    value:
        Candidate runtime container.

    Returns
    -------
    tuple[str, ...]
        Namedtuple fields or torch structseq fields; empty when ``value`` is
        not a field-addressable container.
    """

    from torchlens._input_walk import _instance_fields, declares_namedtuple_fields

    if not isinstance(value, tuple):
        return ()
    if declares_namedtuple_fields(value):
        return _instance_fields(value)
    return _torch_structseq_field_names(value)


def _torch_structseq_field_names(value: Any) -> tuple[str, ...]:
    """Return producer-compatible field names for a torch structseq value.

    r35 hon1_4: delegates to the shared repr-independent helper in
    ``utils/_torch_compat.py`` -- the exact same source the capture side uses,
    so capture and replay stay behavior-identical. Console wrap position,
    dtype/device suffixes, and tensor rendering can never create or destroy a
    structural field.

    Parameters
    ----------
    value:
        Candidate tuple-like torch return value.

    Returns
    -------
    tuple[str, ...]
        Public structseq field names, or an empty tuple when ``value`` is not
        a fully named ``torch.return_types`` value.
    """

    from .utils._torch_compat import torch_structseq_field_names

    return torch_structseq_field_names(value)


def _scalar_literal_equal(actual: Any, expected: Any) -> bool:
    """Return exact scalar equality, treating two matching NaNs as equal.

    Bool and non-bool numeric types are kept distinct so a recomputed ``True`` is
    never mistaken for a numeric ``1``. A capture-time NaN scalar recomputes to
    NaN on the original input, so ``NaN == NaN`` must read as equal here.
    """

    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    if isinstance(actual, float) and isinstance(expected, float):
        if actual != actual and expected != expected:  # both NaN
            return True
    return bool(actual == expected)


def _tensor_derived_scalar_witness_slot_ids(descriptor: SparseRunDescriptor) -> frozenset[str]:
    """Return the runtime slot ids of every tensor->host escape-source witness.

    Each ``TENSOR_DERIVED_SCALAR_LITERAL`` witness digests its source slot at a
    mutation-consistent snapshot (the op's capture-time production value). At run
    time the source value must be snapshotted at the SAME logical point -- when the
    slot is first produced -- before any later in-place op can mutate the live
    tensor, so the run-digest compares the same logical value the save-digest did.
    """

    return frozenset(
        witness.site_label
        for witness in descriptor.control_witnesses
        if witness.kind is ControlWitnessKind.TENSOR_DERIVED_SCALAR_LITERAL
    )


def _tensor_derived_scalar_stale(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
    witness_source_snapshots: Mapping[str, torch.Tensor] | None = None,
) -> bool:
    """Return whether a tensor->host escape source slot is stale for this run.

    A ``TENSOR_DERIVED_SCALAR_LITERAL`` witness records the runtime slot of the op
    whose output tensor escaped to the Python host (via ``.item()`` / ``int()`` /
    ``.tolist()`` / ``aten._local_scalar_dense`` / etc.) together with the SHA-256
    byte digest of that tensor at capture time. The escaped value was baked into a
    downstream literal or steered pure-Python control flow -- neither of which the
    sparse DAG can recompute. If the source slot recomputes to a different value
    than at capture (a CHANGED input), the baked literal / taken branch may be
    stale, so the run must not be blessed VERIFIED/ATTESTED. A slot that recomputes
    the exact capture-time bytes (the ORIGINAL input) keeps the run faithful. A
    missing source slot is treated as stale: the dependency cannot be re-confirmed,
    so the honest ceiling is UNVERIFIABLE.

    A legacy witness whose ``observed_value`` is a scalar literal (rather than a
    byte digest) is compared by exact scalar equality for backward compatibility.
    """

    snapshots = witness_source_snapshots or {}
    # r71 A2: the OWNER-RECORD obligations (``TensorSlotDescriptor.host_escape``) are
    # the required staleness domain -- replay-consumed structure, never the witness
    # stream. An obligated slot with no surviving witness cannot be re-confirmed:
    # stale (fail closed). A gap-discharged obligation also reads stale here, which
    # coincides with the UNVERIFIABLE floor its gap already guarantees.
    witnessed_slot_ids = {
        witness.site_label
        for witness in descriptor.control_witnesses
        if witness.kind is ControlWitnessKind.TENSOR_DERIVED_SCALAR_LITERAL
    }
    for slot in descriptor.tensor_slots:
        if slot.host_escape and slot.slot_id not in witnessed_slot_ids:
            return True
    for witness in descriptor.control_witnesses:
        if witness.kind is not ControlWitnessKind.TENSOR_DERIVED_SCALAR_LITERAL:
            continue
        # Prefer the production-time snapshot: a later in-place op (``y.add_(...)``)
        # could have mutated the live slot value after the escape read it, and the
        # save-digest was taken at the pre-mutation production point. The snapshot
        # compares the SAME logical value; the live slot is the honest fallback.
        recomputed = snapshots.get(witness.site_label)
        if recomputed is None:
            recomputed = slot_values.get(witness.site_label)
        if not isinstance(recomputed, torch.Tensor):
            return True
        expected = _decode_literal(witness.observed_value)
        if isinstance(expected, str):
            # Digest-based witness (value-free, any shape/dtype): re-digest the
            # recomputed source slot and require byte-exact equality with capture.
            try:
                if runnable_tensor_byte_digest(recomputed) != expected:
                    return True
            except (RuntimeError, ValueError, TypeError):
                return True
            continue
        # Legacy scalar-literal witness.
        if recomputed.numel() != 1:
            return True
        try:
            actual = recomputed.item()
        except (RuntimeError, ValueError):
            return True
        if not _scalar_literal_equal(actual, expected):
            return True
    return False


def _is_unbound_state_escape_witness(witness: ControlWitness) -> bool:
    """Return whether a structure witness records an unbound state escape."""

    return (
        witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
        and witness.site_label.startswith(_UNBOUND_STATE_ESCAPE_SITE_PREFIX)
    )


def _is_state_metadata_fact_witness(witness: ControlWitness) -> bool:
    """Return whether a structure witness records a declared state-metadata fact (r65)."""

    return (
        witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
        and witness.site_label.startswith(_STATE_METADATA_FACT_SITE_PREFIX)
    )


def _unbound_state_escape_stale(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> bool:
    """Return whether an unbound state slot differs from its capture-time value.

    An unbound state slot (a registered buffer/param consumed by NO traced call)
    influenced the forward only through an untraced host path -- a Python
    truth-test, an ``.item()`` comparison, or other pure-Python control flow. The
    sparse DAG cannot recompute that dependency, so a staged value that differs
    from capture may have flipped a branch or restaled a literal. Each unbound
    state escape witness records the state name, its runtime slot, and the SHA-256
    byte digest of its capture-time value. This run re-digests the effective staged
    /embedded value; a differing (or missing) value means the untraced dependency
    changed, so the honest ceiling is UNVERIFIABLE + NOT_APPLICABLE. State that is
    byte-identical to capture keeps the run faithful.
    """

    name_to_slot_id: dict[str, str] = {}
    for slot in descriptor.tensor_slots:
        binding = slot.state_binding
        if binding is not None:
            name_to_slot_id.setdefault(binding.state_dict_name, slot.slot_id)
    # r71 A2: the OWNER-RECORD dispositions (``StateSlotBinding.host_escape_disposition
    # == "escaped"``) are the required staleness domain. An escaped-claimed slot with
    # no surviving witness cannot re-confirm its capture digest: stale (fail closed);
    # a gap-discharged claim coincides with its guaranteed UNVERIFIABLE floor.
    witnessed_members: set[tuple[str, str]] = set()
    for witness in descriptor.control_witnesses:
        if not _is_unbound_state_escape_witness(witness):
            continue
        fact = _decode_literal(witness.observed_value)
        if isinstance(fact, Mapping):
            name = fact.get("state_dict_name")
            slot_id = fact.get("slot_id")
            if isinstance(name, str) and isinstance(slot_id, str):
                witnessed_members.add((name, slot_id))
    for slot in descriptor.tensor_slots:
        binding = slot.state_binding
        if binding is None or binding.host_escape_disposition != "escaped":
            continue
        if (binding.state_dict_name, slot.slot_id) not in witnessed_members:
            return True
    for witness in descriptor.control_witnesses:
        if not _is_unbound_state_escape_witness(witness):
            continue
        fact = _decode_literal(witness.observed_value)
        if not isinstance(fact, Mapping) or fact.get(_UNBOUND_STATE_ESCAPE_FACT_KEY) is not True:
            continue
        name = fact.get("state_dict_name")
        expected = fact.get("digest")
        if not isinstance(name, str) or not isinstance(expected, str):
            return True
        slot_id = name_to_slot_id.get(name)
        value = slot_values.get(slot_id) if isinstance(slot_id, str) else None
        if not isinstance(value, torch.Tensor):
            return True
        try:
            if runnable_tensor_byte_digest(value) != expected:
                return True
        except (RuntimeError, ValueError, TypeError):
            return True
    return False
