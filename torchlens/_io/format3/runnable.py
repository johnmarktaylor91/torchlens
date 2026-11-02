"""The 3.0 runnable component: descriptor JSON plus safetensors state, no pickle.

Contract: ``docs/maintainers/saved_format_3.md``, section 7 "Runnable" and the
runnable example of section 8. A runnable save adds one directory,
``runnable/``, beside the trace records:

* ``descriptor.json``: the sparse taken-path descriptor as canonical JSON
  (sorted keys, no NaN or infinity), byte for byte the descriptor the tlspec 11
  manifest carried under ``run``. It holds the call recipe, the callables as
  callable registry records (a registry key per callable, never a callable
  object) and the initializer policy (``initializer_policy_version``).
* ``state_dict-NNNN.safetensors``: the capture-time state dict, keyed by
  canonical state name, packed into files of at most ``CHUNK_BYTES``.
* ``buffers.safetensors``: the non-persistent buffer values, keyed by name.
* ``activations-NNNN.safetensors``: the capture-selected activations, keyed by
  the activation member's blob id (only when activations were archived).

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
from typing import Any, NoReturn, cast

import torch
from safetensors.torch import load_file, save_file

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
from ..._utils.hashing import sha256_file
from .. import bundle as tlspec_bundle
from .._json import read_bounded
from ..format_errors import ArtifactError
from ..paths import reject_symlink_path
from ..runnable import (
    require_sparse_run_descriptor,
    sparse_descriptor_to_json,
    with_activation_payload,
    with_weight_payload,
)
from ..runnable_load import attach_sparse_run_readiness
from .contract import REFUSAL_CODES, RUNNABLE_DIR

#: Descriptor file of the runnable component.
DESCRIPTOR_PATH = f"{RUNNABLE_DIR}/descriptor.json"
#: Non-persistent buffer file of the runnable component.
BUFFERS_PATH = f"{RUNNABLE_DIR}/buffers.safetensors"
#: Largest safetensors file a chunked family writes, in bytes (contract section 3).
CHUNK_BYTES = 256 * 1024 * 1024
#: Chunked families: manifest key to file stem under ``runnable/``.
CHUNKED_FAMILIES: dict[str, str] = {"state_dict": "state_dict", "activations": "activations"}
#: Manifest key of each family's payload schema.
FAMILY_SCHEMA_KEYS: dict[str, str] = {
    "state_dict": "state_dict_family",
    "buffers": "buffers_family",
    "activations": "activations_family",
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
#: Keys a present family adds to the ``runnable`` object.
_FAMILY_KEYS = frozenset(FAMILY_SCHEMA_KEYS) | frozenset(FAMILY_SCHEMA_KEYS.values())
#: Keys of one ``files`` row.
_FILE_ROW_KEYS = frozenset({"sha256", "bytes"})
#: Remedy for an altered component.
_ALTERED = "save the trace again at level='runnable'; the runnable component was altered"


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


def _refuse(situation: str, message: str, remedy: str) -> NoReturn:
    """Raise the contract's typed refusal for one situation.

    Parameters
    ----------
    situation:
        Key of ``REFUSAL_CODES``.
    message:
        What was refused and where.
    remedy:
        What the caller can do about it.
    """

    raise ArtifactError(message, code=REFUSAL_CODES[situation], remedy=remedy)


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


def _host_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the host-resident, contiguous, materialized bytes of one tensor."""

    return tensor.detach().resolve_conj().resolve_neg().contiguous().cpu()


def _unshared_host_tensors(
    members: Sequence[tuple[str, str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Return host tensors keyed by member key, each with storage of its own.

    Tied weights and view activations share storage; safetensors refuses
    shared storage, so every tensor after the first on a storage is copied.
    """

    host: dict[str, torch.Tensor] = {}
    seen_storages: set[int] = set()
    for _, key, tensor in members:
        prepared = _host_tensor(tensor)
        storage = prepared.untyped_storage().data_ptr()
        if storage in seen_storages:
            prepared = prepared.clone()
        else:
            seen_storages.add(storage)
        host[key] = prepared
    return host


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


def _file_row(root: Path, relative: str) -> dict[str, Any]:
    """Return the manifest ``files`` row of one written file."""

    target = root / relative
    os.chmod(target, 0o600)
    return {"sha256": sha256_file(target), "bytes": target.stat().st_size}


def _chunked(
    members: Sequence[tuple[str, torch.Tensor]],
) -> list[list[tuple[str, torch.Tensor]]]:
    """Split ``(key, tensor)`` members into chunks of at most ``CHUNK_BYTES``.

    A tensor larger than the limit gets a chunk of its own.
    """

    chunks: list[list[tuple[str, torch.Tensor]]] = []
    current: list[tuple[str, torch.Tensor]] = []
    size = 0
    for key, tensor in members:
        nbytes = tensor.numel() * tensor.element_size()
        if current and size + nbytes > CHUNK_BYTES:
            chunks.append(current)
            current, size = [], 0
        current.append((key, tensor))
        size += nbytes
    if current:
        chunks.append(current)
    return chunks


def _write_safetensors(
    root: Path, relative: str, members: Sequence[tuple[str, torch.Tensor]]
) -> dict[str, Any]:
    """Write one safetensors file and return its ``files`` row."""

    save_file(_unshared_host_tensors(members), str(root / relative))
    return _file_row(root, relative)


def _write_chunked(
    root: Path, stem: str, members: Sequence[tuple[str, torch.Tensor]], files: dict[str, Any]
) -> list[str]:
    """Write one chunked family; record each file in ``files``; return the paths."""

    paths: list[str] = []
    for number, chunk in enumerate(_chunked(members), start=1):
        relative = f"{RUNNABLE_DIR}/{stem}-{number:04d}.safetensors"
        files[relative] = _write_safetensors(root, relative, chunk)
        paths.append(relative)
    return paths


def _spec_members(specs: Sequence[Any], *, key_by_blob: bool) -> list[tuple[str, torch.Tensor]]:
    """Turn tlspec blob specs into ``(key, tensor)`` members keyed by name or blob id."""

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


def write_runnable(
    trace: Any,
    path: str | Path,
    *,
    include_weights: bool = True,
    include_activations: bool = False,
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
    descriptor, buffer_specs, weight_specs, activation_specs = _stamped_descriptor(
        trace, include_weights=include_weights, include_activations=include_activations
    )
    layers = descriptor.payload_layers
    component_dir.mkdir(parents=True)
    os.chmod(component_dir, 0o700)
    _write_private_text(
        root / DESCRIPTOR_PATH, canonical_json_text(sparse_descriptor_to_json(descriptor))
    )
    files: dict[str, dict[str, Any]] = {DESCRIPTOR_PATH: _file_row(root, DESCRIPTOR_PATH)}
    manifest: dict[str, Any] = {"descriptor": DESCRIPTOR_PATH, **manifest_versions()}
    if weight_specs:
        manifest["state_dict"] = _write_chunked(
            root, "state_dict", _spec_members(weight_specs, key_by_blob=False), files
        )
        manifest["state_dict_family"] = str(layers.weights.schema)
    if buffer_specs:
        files[BUFFERS_PATH] = _write_safetensors(
            root, BUFFERS_PATH, _spec_members(buffer_specs, key_by_blob=False)
        )
        manifest["buffers"] = BUFFERS_PATH
        manifest["buffers_family"] = str(layers.nonpersistent_buffers.schema)
    if activation_specs:
        manifest["activations"] = _write_chunked(
            root, "activations", _spec_members(activation_specs, key_by_blob=True), files
        )
        manifest["activations_family"] = str(layers.activations.schema)
    return RunnableWrite(manifest=manifest, files=files)


def _checked_path(root: Path, relative: Any, allowed: str, files: Mapping[str, Any]) -> Path:
    """Resolve one listed component file and verify it against its ``files`` row.

    ``allowed`` is the exact path, or a ``{n}`` pattern for a chunk number.
    """

    if not isinstance(relative, str) or not _path_allowed(relative, allowed):
        _refuse(
            "manifest_invalid",
            f"The runnable component names the file {relative!r}; it must be {allowed!r}.",
            _ALTERED,
        )
    row = files.get(relative)
    if not isinstance(row, Mapping) or set(row) != _FILE_ROW_KEYS:
        _refuse(
            "manifest_invalid",
            f"The runnable component file {relative!r} has no sha256 and bytes row in the "
            "manifest's files.",
            _ALTERED,
        )
    target = root / relative
    reject_symlink_path(
        target, context="runnable component file", code=REFUSAL_CODES["symlink_path"]
    )
    if not target.is_file():
        _refuse(
            "blob_missing",
            f"The runnable component file {target} is missing.",
            "copy the whole artifact directory; the runnable component is incomplete",
        )
    if target.stat().st_size != row["bytes"] or sha256_file(target) != row["sha256"]:
        _refuse(
            "record_file_corrupt",
            f"The runnable component file {target} does not match its sha256 and byte length.",
            _ALTERED,
        )
    return target


def _path_allowed(relative: str, allowed: str) -> bool:
    """Return whether ``relative`` is ``allowed`` (a ``{n}`` stands for four digits)."""

    if "{n}" not in allowed:
        return relative == allowed
    head, tail = allowed.split("{n}")
    number = relative[len(head) : len(relative) - len(tail)]
    return (
        relative.startswith(head)
        and relative.endswith(tail)
        and len(number) == 4
        and number.isdigit()
    )


def _read_safetensors(target: Path, map_location: str | torch.device) -> dict[str, torch.Tensor]:
    """Load one safetensors file; refuse an unreadable one."""

    try:
        return load_file(str(target), device=str(map_location))
    except (OSError, ValueError, RuntimeError) as exc:
        _refuse(
            "record_file_corrupt",
            f"The runnable family file {target} is not a readable safetensors file "
            f"({type(exc).__name__}).",
            _ALTERED,
        )


def _read_family(
    root: Path,
    manifest: Mapping[str, Any],
    family: str,
    files: Mapping[str, Any],
    map_location: str | torch.device,
) -> dict[str, torch.Tensor] | None:
    """Read one family's files into ``{key: tensor}``; ``None`` when it is absent."""

    listed = manifest.get(family)
    if listed is None:
        return None
    if family in CHUNKED_FAMILIES:
        stem = CHUNKED_FAMILIES[family]
        if not isinstance(listed, list) or not listed:
            _refuse(
                "manifest_invalid", f"The runnable {family!r} entry must list its files.", _ALTERED
            )
        expected = [f"{RUNNABLE_DIR}/{stem}-{n:04d}.safetensors" for n in range(1, len(listed) + 1)]
        if listed != expected:
            _refuse(
                "manifest_invalid",
                f"The runnable {family!r} files must be numbered from 0001 in order.",
                _ALTERED,
            )
        paths = [
            _checked_path(root, item, f"{RUNNABLE_DIR}/{stem}-{{n}}.safetensors", files)
            for item in listed
        ]
    else:
        paths = [_checked_path(root, listed, BUFFERS_PATH, files)]
    values: dict[str, torch.Tensor] = {}
    for target in paths:
        chunk = _read_safetensors(target, map_location)
        if set(chunk) & set(values):
            _refuse("record_file_corrupt", f"The runnable {family!r} files repeat a key.", _ALTERED)
        values.update(chunk)
    return values


def _checked_manifest(manifest: Any) -> Mapping[str, Any]:
    """Check the ``runnable`` object's key set, versions and family pairing."""

    keys = set(manifest) if isinstance(manifest, Mapping) else set()
    if (
        not isinstance(manifest, Mapping)
        or not keys >= _REQUIRED_KEYS
        or not keys <= _REQUIRED_KEYS | _FAMILY_KEYS
        or any(
            (family in keys) != (schema in keys) for family, schema in FAMILY_SCHEMA_KEYS.items()
        )
    ):
        _refuse(
            "manifest_invalid",
            "The manifest's runnable object must hold the descriptor, the five component "
            f"versions and a schema beside each family file; found {sorted(keys)}.",
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
    descriptor_path = _checked_path(root, checked["descriptor"], DESCRIPTOR_PATH, files)
    try:
        descriptor = read_bounded(descriptor_path)
    except (OSError, ValueError) as exc:
        _refuse(
            "record_file_corrupt",
            f"The runnable descriptor {descriptor_path} is not readable JSON ({type(exc).__name__}).",
            _ALTERED,
        )
    if not isinstance(descriptor, Mapping):
        _refuse(
            "manifest_invalid",
            f"The runnable descriptor {descriptor_path} must be an object.",
            _ALTERED,
        )
    families = {}
    for family, bound_as in (
        ("state_dict", "state_dict"),
        ("buffers", "nonpersistent_buffers"),
        ("activations", "activations"),
    ):
        values = _read_family(root, checked, family, files, map_location)
        if values is not None:
            families[bound_as] = values
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
        if not declared.present:
            if values is not None:
                _refuse(
                    "record_file_corrupt",
                    f"The runnable family {family!r} is present while the descriptor "
                    "declares it absent.",
                    "save the trace again; the runnable component was altered",
                )
            if family == "activations":
                trace._runnable.archived_activations = {}
            continue
        if declared.schema != expected_schema[family]:
            _refuse(
                "record_file_corrupt",
                f"The runnable family {family!r} declares unsupported schema {declared.schema!r}.",
                "load it with the TorchLens release that wrote it, or save the trace again",
            )
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
