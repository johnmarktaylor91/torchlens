"""The 3.0 runnable component: descriptor JSON plus safetensors state, no pickle.

Contract: ``docs/maintainers/saved_format_3.md``, section "Runnable". A runnable
save adds one directory, ``runnable/``, beside the trace records:

* ``component.json``: the component index. It carries the component versions
  of the writer contract (descriptor, call recipe, callable reference,
  initializer policy, the two state families, the activation payload, the
  witness registry), the descriptor file and its sha256, and one entry per
  tensor family: schema, safetensors file, sha256 and one tensor reference
  per member (``name`` plus the contract's tensor reference fields).
* ``descriptor.json``: the sparse taken-path descriptor as canonical JSON
  (sorted keys, no NaN or infinity). Callables are carried as the
  descriptor's callable registry records (a registry key per callable),
  never as a callable object; the initializer policy is the descriptor's
  ``initializer_policy_version`` field.
* ``state_dict.safetensors``, ``nonpersistent_buffers.safetensors``,
  ``activations.safetensors``: one file per present family, keyed by the
  canonical state name (weights, buffers) or the activation member's blob id.

The writer door returns the index record the trace manifest lists for the
component (``component`` path and sha256). The reader door validates the
index, the hashes and the family membership against the parsed descriptor
and binds the families onto a loaded trace through the same ``_runnable``
binders the tlspec 11 loader used, so the readiness, attestation and path
faithfulness checks prove exactly what they proved before.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, cast

import torch

from .._json import read_bounded
from ..format_errors import ArtifactError
from ..paths import reject_symlink_path
from .contract import REFUSAL_CODES, RUNNABLE_DIR

#: Index file of the runnable component, relative to ``runnable/``.
COMPONENT_FILE = "component.json"
#: Descriptor file of the runnable component, relative to ``runnable/``.
DESCRIPTOR_FILE = "descriptor.json"
#: ``format`` value of the component index; anything else is not this component.
COMPONENT_FORMAT = "torchlens.runnable_component"
#: Integer version of the component index layout.
COMPONENT_LAYOUT_VERSION = 1
#: Payload codec every runnable family is written with.
FAMILY_CODEC = "torch_safetensors_v1"
#: The three tensor families: name, file, and the descriptor payload layer.
FAMILY_FILES: dict[str, str] = {
    "state_dict": "state_dict.safetensors",
    "nonpersistent_buffers": "nonpersistent_buffers.safetensors",
    "activations": "activations.safetensors",
}
#: Keys of the component index, all required.
_COMPONENT_KEYS = frozenset(
    {"format", "layout_version", "component_versions", "descriptor", "families"}
)
#: Keys of one family entry, all required.
_FAMILY_KEYS = frozenset({"schema", "file", "sha256", "members"})
#: Keys of one family member: its name plus the contract's tensor reference fields.
_MEMBER_KEYS = frozenset(
    {"name", "blob", "key", "dtype", "shape", "device", "requires_grad", "codec", "complex"}
)


@dataclass(frozen=True)
class RunnableComponent:
    """A runnable component read from disk, validated but not yet bound.

    Attributes
    ----------
    component_versions:
        Component versions the writer stamped (the writer contract's envelope rows).
    descriptor:
        Decoded descriptor JSON object.
    families:
        Family name to ``{member name: tensor}``; only present families appear.
    """

    component_versions: Mapping[str, Any]
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

    from ..._runnable import (
        RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION,
        RUNNABLE_CALL_RECIPE_VERSION,
        RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
        RUNNABLE_INITIALIZER_POLICY_VERSION,
        RUNNABLE_TLSPEC_SCHEMA_VERSION,
        WITNESS_FAMILY_REGISTRY_VERSION,
    )

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


def _member_record(name: str, key: str, file_name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Build one family member: its name plus a contract tensor reference."""

    return {
        "name": name,
        "blob": f"{RUNNABLE_DIR}/{file_name}",
        "key": key,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": [int(dim) for dim in tensor.shape],
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
        "codec": FAMILY_CODEC,
        "complex": False,
    }


def _write_family(
    component_dir: Path,
    family: str,
    schema: str,
    members: Sequence[tuple[str, str, torch.Tensor]],
) -> dict[str, Any]:
    """Write one tensor family as one safetensors file and return its index entry.

    Parameters
    ----------
    component_dir:
        The ``runnable/`` directory being written.
    family:
        Key of ``FAMILY_FILES``.
    schema:
        The payload schema the descriptor declares for the family.
    members:
        ``(name, key, tensor)`` per member, in index order.
    """

    from safetensors.torch import save_file

    from ..._utils.hashing import sha256_file

    file_name = FAMILY_FILES[family]
    host = {key: _host_tensor(tensor) for _, key, tensor in members}
    target = component_dir / file_name
    save_file(host, str(target))
    os.chmod(target, 0o600)
    return {
        "schema": schema,
        "file": file_name,
        "sha256": sha256_file(target),
        "members": [_member_record(name, key, file_name, tensor) for name, key, tensor in members],
    }


def _specs_as_members(specs: Sequence[Any], *, key_by_blob: bool) -> list[tuple[str, str, Any]]:
    """Turn tlspec blob specs into family members keyed by name or blob id."""

    return [
        (str(spec.label), str(spec.blob_id) if key_by_blob else str(spec.label), spec.value)
        for spec in specs
    ]


def write_runnable(
    trace: Any,
    path: str | Path,
    *,
    include_weights: bool = True,
    include_activations: bool = False,
) -> dict[str, Any]:
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
    dict[str, Any]
        The manifest record of the component: ``component`` (artifact-relative
        index path) and ``sha256`` of the index file.

    Raises
    ------
    ArtifactError
        ``bundle_save_failed`` when the component directory already exists;
        the descriptor producer's own typed refusals otherwise.
    """

    from ..._utils.hashing import sha256_file
    from .. import bundle as tlspec_bundle
    from ..runnable import (
        require_sparse_run_descriptor,
        sparse_descriptor_to_json,
        with_activation_payload,
        with_weight_payload,
    )

    component_dir = Path(path) / RUNNABLE_DIR
    if component_dir.exists() or component_dir.is_symlink():
        _refuse(
            "target_exists",
            f"The runnable component directory {component_dir} already exists.",
            "write the runnable component once, into a fresh artifact directory",
        )
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
    descriptor_json = sparse_descriptor_to_json(descriptor)
    layers = descriptor.payload_layers

    component_dir.mkdir(parents=True)
    os.chmod(component_dir, 0o700)
    _write_private_text(component_dir / DESCRIPTOR_FILE, canonical_json_text(descriptor_json))
    families: dict[str, Any] = {}
    if buffer_specs:
        families["nonpersistent_buffers"] = _write_family(
            component_dir,
            "nonpersistent_buffers",
            str(layers.nonpersistent_buffers.schema),
            _specs_as_members(buffer_specs, key_by_blob=False),
        )
    if weight_specs:
        families["state_dict"] = _write_family(
            component_dir,
            "state_dict",
            str(layers.weights.schema),
            _specs_as_members(weight_specs, key_by_blob=False),
        )
    if activation_specs:
        families["activations"] = _write_family(
            component_dir,
            "activations",
            str(layers.activations.schema),
            _specs_as_members(activation_specs, key_by_blob=True),
        )
    index = {
        "format": COMPONENT_FORMAT,
        "layout_version": COMPONENT_LAYOUT_VERSION,
        "component_versions": component_versions(),
        "descriptor": {
            "file": DESCRIPTOR_FILE,
            "sha256": sha256_file(component_dir / DESCRIPTOR_FILE),
        },
        "families": families,
    }
    index_path = component_dir / COMPONENT_FILE
    _write_private_text(index_path, canonical_json_text(index))
    return {"component": f"{RUNNABLE_DIR}/{COMPONENT_FILE}", "sha256": sha256_file(index_path)}


def _require_mapping(value: Any, where: str, keys: frozenset[str]) -> Mapping[str, Any]:
    """Return ``value`` when it is an object with exactly ``keys``; refuse otherwise."""

    if not isinstance(value, Mapping) or set(value) != keys:
        found = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        _refuse(
            "manifest_invalid",
            f"The runnable component {where} must be an object with exactly the keys "
            f"{sorted(keys)}; found {found}.",
            "save the trace again at level='runnable'; the component index was altered",
        )
    return value


def _checked_file(component_dir: Path, name: Any, expected_sha: Any, where: str) -> Path:
    """Resolve one component file by its fixed name and verify its sha256."""

    from ..._utils.hashing import sha256_file

    allowed = {DESCRIPTOR_FILE, *FAMILY_FILES.values()}
    if not isinstance(name, str) or name not in allowed or not isinstance(expected_sha, str):
        _refuse(
            "manifest_invalid",
            f"The runnable component {where} names {name!r}; only {sorted(allowed)} are allowed.",
            "save the trace again at level='runnable'; the component index was altered",
        )
    target = component_dir / name
    reject_symlink_path(target, context=f"runnable {where}", code=REFUSAL_CODES["symlink_path"])
    if not target.is_file():
        _refuse(
            "blob_missing",
            f"The runnable component file {target} is missing.",
            "copy the whole artifact directory; the runnable component is incomplete",
        )
    if sha256_file(target) != expected_sha:
        _refuse(
            "record_file_corrupt",
            f"The runnable component file {target} does not match its recorded sha256.",
            "save the trace again; the runnable component was altered after it was written",
        )
    return target


def _read_json_file(target: Path, where: str) -> Any:
    """Read one bounded JSON file; refuse an unreadable one."""

    try:
        return read_bounded(target)
    except (OSError, ValueError) as exc:
        _refuse(
            "record_file_corrupt",
            f"The runnable {where} {target} is not readable JSON ({type(exc).__name__}).",
            "save the trace again; the runnable component was altered after it was written",
        )


def _read_family(
    component_dir: Path, family: str, entry: Any, map_location: str | torch.device
) -> dict[str, torch.Tensor]:
    """Validate one family entry, load its file and return ``{member name: tensor}``."""

    from safetensors.torch import load_file

    record = _require_mapping(entry, f"family {family!r}", _FAMILY_KEYS)
    if record["file"] != FAMILY_FILES[family]:
        _refuse(
            "manifest_invalid",
            f"The runnable family {family!r} must live in {FAMILY_FILES[family]!r}.",
            "save the trace again at level='runnable'; the component index was altered",
        )
    target = _checked_file(component_dir, record["file"], record["sha256"], f"family {family!r}")
    members = record["members"]
    if not isinstance(members, list):
        _refuse(
            "manifest_invalid",
            f"The runnable family {family!r} members must be a list.",
            "save the trace again at level='runnable'; the component index was altered",
        )
    by_key: dict[str, str] = {}
    names: set[str] = set()
    for member in members:
        row = _require_mapping(member, f"family {family!r} member", _MEMBER_KEYS)
        name, key = row["name"], row["key"]
        if not isinstance(name, str) or not isinstance(key, str) or name in names or key in by_key:
            _refuse(
                "manifest_invalid",
                f"The runnable family {family!r} repeats or mistypes member {name!r}.",
                "save the trace again at level='runnable'; the component index was altered",
            )
        names.add(name)
        by_key[key] = name
    try:
        tensors = load_file(str(target), device=str(map_location))
    except (OSError, ValueError, RuntimeError) as exc:
        _refuse(
            "record_file_corrupt",
            f"The runnable family file {target} is not a readable safetensors file "
            f"({type(exc).__name__}).",
            "save the trace again; the runnable component was altered after it was written",
        )
    if set(tensors) != set(by_key):
        _refuse(
            "record_file_corrupt",
            f"The runnable family file {target} holds keys that disagree with its index.",
            "save the trace again; the runnable component was altered after it was written",
        )
    return {by_key[key]: tensor for key, tensor in tensors.items()}


def load_runnable(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> RunnableComponent | None:
    """Load the runnable component after the artifact validated.

    Parameters
    ----------
    path:
        Artifact directory.
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
        altered or malformed component.
    """

    component_dir = Path(path) / RUNNABLE_DIR
    reject_symlink_path(
        component_dir, context="runnable component", code=REFUSAL_CODES["symlink_path"]
    )
    if not component_dir.exists():
        return None
    index_path = component_dir / COMPONENT_FILE
    reject_symlink_path(index_path, context="runnable index", code=REFUSAL_CODES["symlink_path"])
    if not index_path.is_file():
        _refuse(
            "blob_missing",
            f"The runnable component index {index_path} is missing.",
            "copy the whole artifact directory; the runnable component is incomplete",
        )
    index = _require_mapping(_read_json_file(index_path, "index"), "index", _COMPONENT_KEYS)
    if index["format"] != COMPONENT_FORMAT or index["layout_version"] != COMPONENT_LAYOUT_VERSION:
        _refuse(
            "manifest_invalid",
            f"The runnable component index {index_path} is not a layout "
            f"{COMPONENT_LAYOUT_VERSION} {COMPONENT_FORMAT!r} index.",
            "load it with the TorchLens release that wrote it, or save the trace again",
        )
    if not isinstance(index["component_versions"], Mapping):
        _refuse(
            "manifest_invalid",
            "The runnable component versions must be an object.",
            "save the trace again at level='runnable'; the component index was altered",
        )
    descriptor_entry = _require_mapping(
        index["descriptor"], "descriptor", frozenset({"file", "sha256"})
    )
    if descriptor_entry["file"] != DESCRIPTOR_FILE:
        _refuse(
            "manifest_invalid",
            f"The runnable descriptor must live in {DESCRIPTOR_FILE!r}.",
            "save the trace again at level='runnable'; the component index was altered",
        )
    descriptor_path = _checked_file(
        component_dir, descriptor_entry["file"], descriptor_entry["sha256"], "descriptor"
    )
    descriptor = _read_json_file(descriptor_path, "descriptor")
    if not isinstance(descriptor, Mapping):
        _refuse(
            "manifest_invalid",
            f"The runnable descriptor {descriptor_path} must be a JSON object.",
            "save the trace again at level='runnable'; the descriptor was altered",
        )
    raw_families = index["families"]
    if not isinstance(raw_families, Mapping) or not set(raw_families) <= set(FAMILY_FILES):
        _refuse(
            "manifest_invalid",
            f"The runnable families must be an object over {sorted(FAMILY_FILES)}.",
            "save the trace again at level='runnable'; the component index was altered",
        )
    families = {
        family: _read_family(component_dir, family, entry, map_location)
        for family, entry in raw_families.items()
    }
    return RunnableComponent(
        component_versions=dict(index["component_versions"]),
        descriptor=descriptor,
        families=families,
    )


def attach_runnable(
    trace: Any,
    path: str | Path,
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
    map_location:
        Device for the family tensors.

    Returns
    -------
    ReadinessReport
        The non-executing readiness report attached to ``trace``.
    """

    from ..runnable_load import attach_sparse_run_readiness

    component = load_runnable(path, map_location=map_location)
    if component is None:
        return attach_sparse_run_readiness(trace, None)
    report = attach_sparse_run_readiness(trace, component.descriptor)
    bind_runnable_families(trace, component.families)
    return report


def _family_disposition(trace: Any, present: bool) -> str:
    """Return ``bind``, ``skip`` or ``error`` for one family (the r39 corr2_3 rule)."""

    from ..._runnable import RunProvider

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

    from ..._runnable.state import bind_embedded_nonpersistent_buffers, bind_embedded_trace_state

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

    from ..._runnable import ActivationPayloadLayerDescriptor, ArchivedActivation
    from ..._runnable.state import runnable_tensor_byte_digest

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
