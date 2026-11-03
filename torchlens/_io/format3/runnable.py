"""The 3.0 runnable component: descriptor JSON plus safetensors state, no pickle.

Contract: ``docs/maintainers/saved_format_3.md``, section 7 "Runnable" and the
runnable example of section 8. A runnable save adds one directory,
``runnable/``, beside the trace records:

* ``descriptor.json``: the sparse taken-path descriptor as canonical JSON
  (sorted keys, no NaN or infinity), byte for byte the descriptor the tlspec 11
  manifest carried under ``run``. It holds the call recipe, the callables as
  callable registry records (a registry key per callable, never a callable
  object) and the initializer policy (``initializer_policy_version``).
* ``state_dict-NNNN.safetensors``: the capture-time state dict, in files of
  at most 256 MiB numbered from 0001.
* ``buffers.safetensors``: the non-persistent buffer values.
* ``activations-NNNN.safetensors``: the capture-selected activations (only
  when activations were archived).

Every tensor is one opaque key ``t<k>`` in one file; the manifest's
``runnable`` object maps each state name, buffer name or activation blob id to
a contract tensor reference (``runnable_families``).

The writer door returns the manifest's ``runnable`` object (component versions
and file paths) and the ``files`` rows (sha256 and byte length) for every file
it wrote; the trace writer merges both into ``manifest.json``. The reader door
takes the same two values back, checks every listed file against its row,
checks the versions and file names, and binds the families onto a loaded trace
through the ``_runnable`` binders the tlspec 11 loader used, so the readiness,
attestation and path faithfulness checks prove exactly what they proved before.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch

from ..._runnable import (
    RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION,
    RUNNABLE_CALL_RECIPE_VERSION,
    RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
    RUNNABLE_INITIALIZER_POLICY_VERSION,
    RUNNABLE_TLSPEC_SCHEMA_VERSION,
    WITNESS_FAMILY_REGISTRY_VERSION,
    ActivationPayloadLayerDescriptor,
    ArchivedActivation,
    RunProvider,
)
from ..._runnable.state import (
    bind_embedded_nonpersistent_buffers,
    bind_embedded_trace_state,
    runnable_tensor_byte_digest,
)
from .. import bundle as tlspec_bundle
from .._json import read_bounded
from ..paths import reject_symlink_path
from ..runnable import (
    require_sparse_run_descriptor,
    sparse_descriptor_to_json,
    with_activation_payload,
    with_weight_payload,
)
from ..runnable_load import attach_sparse_run_readiness
from .contract import REFUSAL_CODES, RUNNABLE_DIR
from .runnable_families import (
    ALTERED as _ALTERED,
    FamilySink,
    checked_file,
    file_row,
    read_family,
    refuse as _refuse,
    write_family,
)

#: The runnable carrier binds through ``_runnable`` and ``_io.runnable`` and sits in
#: their stratum (the layer map's ``_io.runnable`` row), like ``_io.bundle_container``.
__tl_layer__ = "L3"

#: Descriptor file of the runnable component.
DESCRIPTOR_PATH = f"{RUNNABLE_DIR}/descriptor.json"
#: Families: manifest key to (file stem under ``runnable/``, chunked, bound family name).
FAMILIES: dict[str, tuple[str, bool, str]] = {
    "state_dict": ("state_dict", True, "state_dict"),
    "buffers": ("buffers", False, "nonpersistent_buffers"),
    "activations": ("activations", True, "activations"),
}
#: Required keys of the manifest's ``runnable`` object (versions plus the descriptor).
_REQUIRED_KEYS = frozenset(
    {
        "descriptor",
        "descriptor_version",
        "call_recipe_version",
        "callable_ref_version",
        "initializer_policy_version",
        "witness_registry_version",
    }
)
#: Keys a present family adds to the ``runnable`` object: files, schema, tensor references.
_FAMILY_KEYS = frozenset(
    key for family in FAMILIES for key in (family, f"{family}_family", f"{family}_tensors")
)


@dataclass(frozen=True)
class RunnableWrite:
    """What the runnable writer door hands the trace writer.

    Attributes
    ----------
    manifest:
        The manifest's ``runnable`` object: component versions and file paths.
    files:
        Artifact-relative path to ``{"sha256", "bytes"}`` for every file written.
    """

    manifest: dict[str, Any]
    files: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RunnableComponent:
    """A runnable component read from disk, validated but not yet bound.

    Attributes
    ----------
    descriptor:
        Decoded descriptor JSON object.
    families:
        Family name (``state_dict``, ``nonpersistent_buffers``,
        ``activations``) to ``{member key: tensor}``; only present families appear.
    """

    descriptor: Mapping[str, Any]
    families: Mapping[str, Mapping[str, torch.Tensor]] = field(default_factory=dict)


def component_versions() -> dict[str, Any]:
    """Return the component versions a runnable save stamps.

    Returns
    -------
    dict[str, Any]
        The runnable rows of the writer contract envelope, by component name.
    """

    return {
        "nonpersistent_buffer_family": "runnable_nonpersistent_buffer_v1",
        "runnable_activation_payload": RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION,
        "runnable_call_recipe": RUNNABLE_CALL_RECIPE_VERSION,
        "runnable_callable_ref": RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
        "runnable_descriptor": RUNNABLE_TLSPEC_SCHEMA_VERSION,
        "runnable_initializer_policy": RUNNABLE_INITIALIZER_POLICY_VERSION,
        "state_dict_blob_family": "state_dict_v1",
        "witness_family_registry": WITNESS_FAMILY_REGISTRY_VERSION,
    }


def canonical_json_text(value: Any) -> str:
    """Render a JSON value canonically: sorted keys, compact, no NaN, trailing newline.

    Parameters
    ----------
    value:
        JSON-ready value.

    Returns
    -------
    str
        The canonical text.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


def _write_private_text(path: Path, text: str) -> None:
    """Write a text file readable by its owner only."""

    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


def manifest_versions() -> dict[str, str]:
    """Return the version fields every manifest ``runnable`` object carries.

    Returns
    -------
    dict[str, str]
        Manifest key to the component version this runtime writes and reads.
    """

    return {
        "descriptor_version": RUNNABLE_TLSPEC_SCHEMA_VERSION,
        "call_recipe_version": RUNNABLE_CALL_RECIPE_VERSION,
        "callable_ref_version": str(RUNNABLE_CALLABLE_REF_SCHEMA_VERSION),
        "initializer_policy_version": RUNNABLE_INITIALIZER_POLICY_VERSION,
        "witness_registry_version": WITNESS_FAMILY_REGISTRY_VERSION,
    }


def _spec_members(specs: Sequence[Any], *, key_by_blob: bool) -> list[tuple[str, torch.Tensor]]:
    """Turn tlspec blob specs into ``(name, tensor)`` members named by state name or blob id."""

    return [(str(spec.blob_id) if key_by_blob else str(spec.label), spec.value) for spec in specs]


def _stamped_descriptor(
    trace: Any, *, include_weights: bool, include_activations: bool
) -> tuple[Any, list[Any], list[Any], list[Any]]:
    """Produce and payload-stamp the descriptor exactly as the tlspec 11 save did.

    Returns the descriptor and the buffer, weight and activation blob specs.
    """

    descriptor = require_sparse_run_descriptor(trace)
    buffer_specs = tlspec_bundle._capture_nonpersistent_buffer_blob_specs(trace, descriptor)
    if buffer_specs:
        tlspec_bundle._warn_nonpersistent_buffer_disclosure_once()
    weight_specs: list[Any] = []
    activation_specs: list[Any] = []
    if include_weights:
        weight_specs = tlspec_bundle._capture_weight_blob_specs(trace, descriptor)
        descriptor = with_weight_payload(descriptor)
    if include_activations:
        (activation_specs, members, input_digests, state_digests, fingerprints) = (
            tlspec_bundle._capture_activation_blob_specs(trace, descriptor)
        )
        if members and not input_digests:
            tlspec_bundle._warn_unattestable_activation_archive_once()
        descriptor = with_activation_payload(
            descriptor,
            members=members,
            original_input_digests=input_digests,
            capture_state_digests=state_digests,
            input_fingerprints=fingerprints,
        )
    return descriptor, buffer_specs, weight_specs, activation_specs


@dataclass(frozen=True)
class _WriteContext:
    """Prepared runnable data and the shared tensor-key counter of one artifact."""

    counter: list[int]
    prepared: tuple[Any, list[Any], list[Any], list[Any]] | None = None


def write_runnable(
    trace: Any,
    path: str | Path,
    *,
    include_weights: bool = True,
    include_activations: bool = False,
    context: _WriteContext | None = None,
) -> RunnableWrite:
    """Write the runnable component of a trace into an artifact directory.

    The descriptor is produced, validated and payload-stamped exactly as the
    tlspec 11 runnable save did; only the carrier changes.

    Parameters
    ----------
    trace:
        Live trace carrying runnable state.
    path:
        Artifact directory (the component lands under ``runnable/``).
    include_weights:
        Write the capture-time state dict family.
    include_activations:
        Write the capture-selected activation family.
    context:
        Shared artifact counter and optionally the already stamped descriptor and
        family specifications. Omit to prepare a standalone component.

    Returns
    -------
    RunnableWrite
        The manifest ``runnable`` object and the ``files`` rows to merge.

    Raises
    ------
    ArtifactError
        ``bundle_save_failed`` when the component directory already exists;
        the descriptor producer's own typed refusals otherwise.
    """

    root = Path(path)
    component_dir = root / RUNNABLE_DIR
    if component_dir.exists() or component_dir.is_symlink():
        _refuse(
            "target_exists",
            f"The runnable component directory {component_dir} already exists.",
            "write the runnable component once, into a fresh artifact directory",
        )
    context = context if context is not None else _WriteContext([0])
    prepared = context.prepared
    descriptor, buffer_specs, weight_specs, activation_specs = (
        prepared
        if prepared is not None
        else _stamped_descriptor(
            trace, include_weights=include_weights, include_activations=include_activations
        )
    )
    sink = _create_component(root, descriptor, counter=context.counter)
    manifest: dict[str, Any] = {"descriptor": DESCRIPTOR_PATH, **manifest_versions()}
    layers = descriptor.payload_layers
    _write_families(
        sink,
        manifest,
        (
            ("state_dict", weight_specs, layers.weights.schema, False),
            ("buffers", buffer_specs, layers.nonpersistent_buffers.schema, False),
            ("activations", activation_specs, layers.activations.schema, True),
        ),
    )
    return RunnableWrite(manifest=manifest, files=sink.files)


def _create_component(
    root: Path, descriptor: Any, *, counter: list[int] | None = None
) -> FamilySink:
    """Create ``runnable/`` owner-only, write the descriptor and return the family sink."""

    component_dir = root / RUNNABLE_DIR
    component_dir.mkdir(parents=True)
    os.chmod(component_dir, 0o700)
    _write_private_text(
        root / DESCRIPTOR_PATH, canonical_json_text(sparse_descriptor_to_json(descriptor))
    )
    return FamilySink(
        root,
        {DESCRIPTOR_PATH: file_row(root, DESCRIPTOR_PATH)},
        counter if counter is not None else [0],
    )


def _write_families(
    sink: FamilySink, manifest: dict[str, Any], families: Sequence[tuple[str, Any, Any, bool]]
) -> None:
    """Write each non-empty family and record its three manifest keys."""

    for family, specs, schema, key_by_blob in families:
        if not specs:
            continue
        stem, chunked, _bound = FAMILIES[family]
        members = _spec_members(specs, key_by_blob=key_by_blob)
        paths, references = write_family(sink, stem, members, chunked=chunked)
        manifest[family] = paths if chunked else paths[0]
        manifest[f"{family}_family"] = str(schema)
        manifest[f"{family}_tensors"] = references


def _checked_manifest(manifest: Any) -> Mapping[str, Any]:
    """Check the ``runnable`` object's key set, versions and family pairing."""

    keys = set(manifest) if isinstance(manifest, Mapping) else set()
    if (
        not isinstance(manifest, Mapping)
        or not keys >= _REQUIRED_KEYS
        or not keys <= _REQUIRED_KEYS | _FAMILY_KEYS
        or any(
            len({family, f"{family}_family", f"{family}_tensors"} & keys) not in (0, 3)
            for family in FAMILIES
        )
    ):
        _refuse(
            "manifest_invalid",
            "The manifest's runnable object must hold the descriptor, the five component "
            "versions and, per present family, its files, schema and tensor references; "
            f"found {sorted(keys)}.",
            _ALTERED,
        )
    if manifest["descriptor"] != DESCRIPTOR_PATH:
        _refuse(
            "manifest_invalid", f"The runnable descriptor must be {DESCRIPTOR_PATH!r}.", _ALTERED
        )
    stale = sorted(key for key, value in manifest_versions().items() if manifest[key] != value)
    if stale:
        _refuse(
            "manifest_invalid",
            f"The runnable component versions {stale} differ from this runtime's.",
            "load it with the TorchLens release that wrote it, or save the trace again",
        )
    return manifest


def _family_paths(family: str, listed: Any) -> list[str]:
    """Check one family's listed files against the fixed names and numbering."""

    stem, chunked, _bound = FAMILIES[family]
    if chunked:
        count = len(listed) if isinstance(listed, list) else 0
        expected: Any = [f"{RUNNABLE_DIR}/{stem}-{n:04d}.safetensors" for n in range(1, count + 1)]
    else:
        expected = f"{RUNNABLE_DIR}/{stem}.safetensors"
    if not listed or listed != expected:
        _refuse(
            "manifest_invalid",
            f"The runnable {family!r} files must be {expected!r} (chunks numbered from 0001); "
            f"found {listed!r}.",
            _ALTERED,
        )
    return list(listed) if chunked else [listed]


def _read_descriptor(root: Path, files: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read the verified descriptor file; refuse unreadable or non-object JSON."""

    descriptor_path = checked_file(root, DESCRIPTOR_PATH, files)
    try:
        descriptor = read_bounded(descriptor_path)
    except (OSError, ValueError) as exc:
        _refuse(
            "record_file_corrupt",
            f"The runnable descriptor {descriptor_path} is not readable JSON "
            f"({type(exc).__name__}).",
            _ALTERED,
        )
    if not isinstance(descriptor, Mapping):
        _refuse(
            "manifest_invalid",
            f"The runnable descriptor {descriptor_path} must be an object.",
            _ALTERED,
        )
    return descriptor


def load_runnable(
    path: str | Path,
    manifest: Mapping[str, Any] | None,
    files: Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> RunnableComponent | None:
    """Load the runnable component after the artifact's manifest validated.

    Parameters
    ----------
    path:
        Artifact directory.
    manifest:
        The manifest's ``runnable`` object, or ``None`` when it has none.
    files:
        The manifest's ``files`` rows.
    map_location:
        Device for the family tensors.

    Returns
    -------
    RunnableComponent | None
        The validated component, or ``None`` when the artifact has none.

    Raises
    ------
    ArtifactError
        A typed refusal from ``REFUSAL_CODES`` for a symlinked, missing,
        altered or malformed component, or a component directory the manifest
        does not list.
    """

    root = Path(path)
    component_dir = root / RUNNABLE_DIR
    reject_symlink_path(
        component_dir, context="runnable component", code=REFUSAL_CODES["symlink_path"]
    )
    if manifest is None:
        if component_dir.exists():
            _refuse(
                "manifest_invalid",
                f"The artifact holds {component_dir} but its manifest lists no runnable object.",
                _ALTERED,
            )
        return None
    checked = _checked_manifest(manifest)
    descriptor = _read_descriptor(root, files)
    families: dict[str, Mapping[str, torch.Tensor]] = {}
    for family, (_stem, _chunked, bound_as) in FAMILIES.items():
        if family not in checked:
            continue
        paths = _family_paths(family, checked[family])
        families[bound_as] = read_family(
            root, paths, checked[f"{family}_tensors"], files, map_location
        )
    return RunnableComponent(descriptor=descriptor, families=families)


def attach_runnable(
    trace: Any,
    path: str | Path,
    manifest: Mapping[str, Any] | None,
    files: Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Attach an artifact's runnable component (or its absence) to a loaded trace.

    The reader calls this once per loaded trace, after the records rebuilt.
    An artifact without a component attaches the analysis-only readiness,
    as the tlspec 11 loader did for an analysis artifact.

    Parameters
    ----------
    trace:
        Trace rebuilt by the reader.
    path:
        Artifact directory.
    manifest:
        The manifest's ``runnable`` object, or ``None``.
    files:
        The manifest's ``files`` rows.
    map_location:
        Device for the family tensors.

    Returns
    -------
    ReadinessReport
        The non-executing readiness report attached to ``trace``.
    """

    component = load_runnable(path, manifest, files, map_location=map_location)
    if component is None:
        return attach_sparse_run_readiness(trace, None)
    report = attach_sparse_run_readiness(trace, component.descriptor)
    bind_runnable_families(trace, component.families)
    return report


def _family_disposition(trace: Any, present: bool) -> str:
    """Return ``bind``, ``skip`` or ``error`` for one family (the r39 corr2_3 rule)."""

    if trace._sparse_run_descriptor is not None:
        return "bind"
    readiness = trace._runnable.readiness
    if getattr(readiness, "provider", None) is RunProvider.LOADED_SPARSE:
        return "skip"
    return "error" if present else "skip"


def _declared_layer(trace: Any, family: str) -> Any:
    """Return the descriptor payload layer that declares ``family``."""

    layers = trace._sparse_run_descriptor.payload_layers
    return {
        "state_dict": layers.weights,
        "nonpersistent_buffers": layers.nonpersistent_buffers,
        "activations": layers.activations,
    }[family]


def _declared_present(family: str, declared: Any, values: Any, schema: str) -> bool:
    """Return whether the descriptor declares ``family`` present, checking it agrees.

    Refuses a family present against an absent declaration, or a declaration
    with a schema this reader does not bind.
    """

    if not declared.present:
        if values is not None:
            _refuse(
                "record_file_corrupt",
                f"The runnable family {family!r} is present while the descriptor "
                "declares it absent.",
                "save the trace again; the runnable component was altered",
            )
        return False
    if declared.schema != schema:
        _refuse(
            "record_file_corrupt",
            f"The runnable family {family!r} declares unsupported schema {declared.schema!r}.",
            "load it with the TorchLens release that wrote it, or save the trace again",
        )
    return True


def bind_runnable_families(trace: Any, families: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
    """Bind the loaded tensor families onto a trace whose descriptor attached.

    Each family is checked against the descriptor's payload declaration
    (presence and schema) before it binds through the ``_runnable`` binders,
    which apply the strict slot contracts.

    Parameters
    ----------
    trace:
        Trace with its readiness attached.
    families:
        Family name to ``{member name: tensor}``.

    Raises
    ------
    ArtifactError
        When a family contradicts the descriptor's payload declaration.
    StateBindingError
        When a value violates its slot contract.
    """

    expected_schema = {
        "state_dict": "state_dict_v1",
        "nonpersistent_buffers": "runnable_nonpersistent_buffer_v1",
        "activations": component_versions()["runnable_activation_payload"],
    }
    for family in ("nonpersistent_buffers", "state_dict", "activations"):
        values = families.get(family)
        disposition = _family_disposition(trace, values is not None)
        if disposition == "skip":
            continue
        if disposition == "error":
            _refuse(
                "structure_only_incoherent",
                f"The runnable family {family!r} is present on an artifact without a "
                "runnable descriptor.",
                "save the trace again; the runnable component was altered",
            )
        declared = _declared_layer(trace, family)
        if not _declared_present(family, declared, values, expected_schema[family]):
            if family == "activations":
                trace._runnable.archived_activations = {}
            continue
        if family == "state_dict":
            bind_embedded_trace_state(trace, values or {})
        elif family == "nonpersistent_buffers":
            bind_embedded_nonpersistent_buffers(trace, values or {})
        else:
            _bind_activations(trace, declared, values or {})


def _bind_activations(trace: Any, declared: Any, values: Mapping[str, torch.Tensor]) -> None:
    """Bind the inspection-only activation family, checking each member's digest."""

    if not isinstance(declared, ActivationPayloadLayerDescriptor):
        _refuse(
            "record_file_corrupt",
            "The runnable activation payload declaration is incomplete.",
            "save the trace again; the runnable component was altered",
        )
    if set(values) != {member.blob_id for member in declared.members}:
        _refuse(
            "record_file_corrupt",
            "The runnable activation family and the declared membership disagree.",
            "save the trace again; the runnable component was altered",
        )
    archived: dict[str, Any] = {}
    for member in declared.members:
        tensor = cast(torch.Tensor, values[member.blob_id])
        if runnable_tensor_byte_digest(tensor) != member.byte_digest:
            _refuse(
                "record_file_corrupt",
                f"The archived activation {member.blob_id!r} does not match its byte digest.",
                "save the trace again; the runnable component was altered",
            )
        archive_key = f"{member.slot_id}:{member.field}"
        if archive_key in archived:
            _refuse(
                "record_file_corrupt",
                f"The runnable activation family repeats {archive_key!r}.",
                "save the trace again; the runnable component was altered",
            )
        archived[archive_key] = ArchivedActivation(
            slot_id=member.slot_id,
            call_id=member.call_id,
            op_label=member.op_label,
            field=member.field,
            byte_digest=member.byte_digest,
            value=tensor,
        )
    trace._runnable.archived_activations = archived
