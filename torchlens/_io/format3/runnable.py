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
from types import MappingProxyType
from typing import Any

import torch

from .._json import read_bounded
from ..paths import reject_symlink_path
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

#: Pure wire carrier below record restoration and execution-state binding.
__tl_layer__ = "L1"

#: Descriptor file of the runnable component.
DESCRIPTOR_PATH = f"{RUNNABLE_DIR}/descriptor.json"
#: Families: manifest key to (file stem under ``runnable/``, chunked, bound family name).
FAMILIES: Mapping[str, tuple[str, bool, str]] = MappingProxyType(
    {
        "state_dict": ("state_dict", True, "state_dict"),
        "buffers": ("buffers", False, "nonpersistent_buffers"),
        "activations": ("activations", True, "activations"),
    }
)
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

    # Deferred: the frozen wire versions live in the L3 runnable vocabulary.
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


def manifest_versions() -> dict[str, str]:
    """Return the version fields every manifest ``runnable`` object carries.

    Returns
    -------
    dict[str, str]
        Manifest key to the component version this runtime writes and reads.
    """

    versions = component_versions()
    return {
        "descriptor_version": versions["runnable_descriptor"],
        "call_recipe_version": versions["runnable_call_recipe"],
        "callable_ref_version": str(versions["runnable_callable_ref"]),
        "initializer_policy_version": versions["runnable_initializer_policy"],
        "witness_registry_version": versions["witness_family_registry"],
    }


def _spec_members(specs: Sequence[Any], *, key_by_blob: bool) -> list[tuple[str, torch.Tensor]]:
    """Turn tlspec blob specs into ``(name, tensor)`` members named by state name or blob id."""

    return [(str(spec.blob_id) if key_by_blob else str(spec.label), spec.value) for spec in specs]


@dataclass(frozen=True)
class _WriteContext:
    """Prepared wire descriptor, families and shared tensor counter of one artifact."""

    counter: list[int]
    descriptor: Mapping[str, Any]
    families: tuple[tuple[str, Sequence[Any], str, bool], ...]


def write_runnable(path: str | Path, *, context: _WriteContext) -> RunnableWrite:
    """Write one prepared runnable component through the artifact's shared tensor counter.

    Parameters
    ----------
    path:
        Artifact directory; the component lands under ``runnable/``.
    context:
        Wire descriptor, payload specifications and shared artifact counter,
        prepared by the save preflight before any component bytes are written.

    Returns
    -------
    RunnableWrite
        The manifest ``runnable`` object and ``files`` rows.
    """

    root = Path(path)
    component_dir = root / RUNNABLE_DIR
    if component_dir.exists() or component_dir.is_symlink():
        _refuse(
            "target_exists",
            f"The runnable component directory {component_dir} already exists.",
            "write the runnable component once, into a fresh artifact directory",
        )
    sink = _create_component(root, context.descriptor, counter=context.counter)
    manifest: dict[str, Any] = {"descriptor": DESCRIPTOR_PATH, **manifest_versions()}
    _write_families(sink, manifest, context.families)
    return RunnableWrite(manifest=manifest, files=sink.files)


def _create_component(
    root: Path, descriptor: Any, *, counter: list[int] | None = None
) -> FamilySink:
    """Create ``runnable/`` owner-only, write the descriptor and return the family sink."""

    component_dir = root / RUNNABLE_DIR
    component_dir.mkdir(parents=True)
    os.chmod(component_dir, 0o700)
    _write_private_text(root / DESCRIPTOR_PATH, canonical_json_text(descriptor))
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
    if not isinstance(descriptor, Mapping) or (
        descriptor.get("capability") == component_versions()["runnable_descriptor"]
        and not isinstance(descriptor.get("ambient_context"), Mapping)
    ):
        _refuse(
            "manifest_invalid",
            f"The runnable descriptor {descriptor_path} must be an object with "
            "an explicit ambient_context for the current capability.",
            _ALTERED,
        )
    return descriptor


def _checked_descriptor(
    root: Path, manifest: Mapping[str, Any], files: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate component metadata before record allocation or family reads."""

    return _checked_manifest(manifest), _read_descriptor(root, files)


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
    checked, descriptor = _checked_descriptor(root, manifest, files)
    families: dict[str, Mapping[str, torch.Tensor]] = {}
    for family, (_stem, _chunked, bound_as) in FAMILIES.items():
        if family not in checked:
            continue
        paths = _family_paths(family, checked[family])
        families[bound_as] = read_family(
            root, paths, checked[f"{family}_tensors"], files, map_location
        )
    return RunnableComponent(descriptor=descriptor, families=families)
