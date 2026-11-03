"""Runnable binding of a 3.0 artifact: attach the component the carrier read, bind its families.

``format3/runnable.py`` is the pure wire carrier (``runnable/descriptor.json`` and the
tensor families under ``runnable/``, listed in ``manifest["runnable"]``). This module sits
above it and the ``_runnable`` binders: :func:`attach_runnable` attaches the descriptor's
readiness and :func:`bind_runnable_families` checks each family against the descriptor's
payload declaration before the strict slot binders take it. The 2.x payload index and keyed
chunk readers below serve the 2.x binders in ``bundle.py``. Design:
``docs/maintainers/saved_format_3_design.md``, section 12.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple, cast

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file

from .._runnable import ActivationPayloadLayerDescriptor, ArchivedActivation, RunProvider
from .._utils.hashing import sha256_file
from ._chunk_digest import chunk_sha256
from .format3.runnable import component_versions, load_runnable
from .format3.runnable_families import refuse as _refuse
from .format_errors import ArtifactError

if TYPE_CHECKING:
    from .._runnable import SparseRunDescriptor

#: Key of the one tensor in a 2.x one-tensor blob file.
ONE_TENSOR_KEY: Final = "data"

#: Blob-id prefix of each runnable payload family and the ``kind`` the binders filter on.
RUNNABLE_FAMILIES: Final[dict[str, str]] = {
    "runnable_weight_": "runnable_weight",
    "runnable_nonpersistent_buffer_": "runnable_nonpersistent_buffer",
    "runnable_activation_": "runnable_activation",
}

#: ``TensorSlotRole.BUFFER`` (a str enum), compared by value so this module stays below
#: the runnable package in the import layering.
_BUFFER_ROLE: Final = "buffer"
_INDEX_CODE: Final = "manifest_schema_invalid"
_INTEGRITY_CODE: Final = "bundle_metadata_integrity_refused"
_RESAVE: Final = (
    "re-save the trace with trace.save(level='runnable'); the artifact was edited or corrupted"
)


class RunnableBinders(NamedTuple):
    """The execution-side binders this module calls, installed by ``bundle`` at import.

    The binders live in the ``_io.runnable_load`` and ``_runnable.state`` stratum, which
    imports back into ``bundle``; receiving them here (like the reader's load hooks) keeps
    this module out of that import cycle.
    """

    attach_readiness: Callable[[Any, Mapping[str, Any] | None], Any]
    bind_state: Callable[[Any, Mapping[str, torch.Tensor]], None]
    bind_buffers: Callable[[Any, Mapping[str, torch.Tensor]], None]
    byte_digest: Callable[[torch.Tensor], str]


_BINDERS: RunnableBinders | None = None


def install_runnable_binders(binders: RunnableBinders) -> None:
    """Install the execution-side binders (called once when ``bundle`` imports)."""

    global _BINDERS
    _BINDERS = binders


def _binders() -> RunnableBinders:
    if _BINDERS is None:  # pragma: no cover - bundle installs them at import
        raise RuntimeError("runnable binders are installed when torchlens._io.bundle imports")
    return _BINDERS


class RunnablePayloadRef(NamedTuple):
    """One runnable payload of an artifact, as the binders read it.

    Parameters
    ----------
    blob_id:
        Blob id the descriptor and the save named the payload by.
    kind:
        Family kind (a value of :data:`RUNNABLE_FAMILIES`).
    label:
        Canonical state name (weights, buffers) or op label (activations).
    relative_path:
        Artifact-relative path of the chunk file holding the tensor.
    sha256:
        Expected digest of that chunk file.
    key:
        Tensor key inside the chunk; ``None`` for a 2.x one-tensor blob.
    """

    blob_id: str
    kind: str
    label: str
    relative_path: str
    sha256: str
    key: str | None


class RunnablePayloadIndex(NamedTuple):
    """The runnable payload references of one artifact (the binders' ``manifest.tensors``)."""

    tensors: tuple[RunnablePayloadRef, ...]


def _family_kind(blob_id: str) -> str | None:
    """Return the runnable family kind a blob id names, or ``None`` for a record payload."""

    for prefix, kind in RUNNABLE_FAMILIES.items():
        if blob_id.startswith(prefix):
            return kind
    return None


def _index_error(blob_id: str, detail: str, *, code: str = _INDEX_CODE) -> ArtifactError:
    """Return the manifest-index refusal for a runnable blob id."""

    return ArtifactError(
        f"Runnable payload {blob_id!r} does not satisfy the 3.0 saved-format contract: {detail}.",
        code=code,
        remedy=_RESAVE,
        blob_id=blob_id,
        detail=detail,
    )


def _sorted_state_names(descriptor: SparseRunDescriptor, *, persistent: bool) -> list[str]:
    """Return the state names of one binding family in the save's blob-id order."""

    # Several slots may bind one state tensor; the save wrote each name once.
    return sorted(
        {
            slot.state_binding.state_dict_name
            for slot in descriptor.tensor_slots
            if slot.state_binding is not None
            and slot.state_binding.persistent is persistent
            and (persistent or slot.role == _BUFFER_ROLE)
        }
    )


def runnable_labels(descriptor: SparseRunDescriptor | None) -> dict[str, str]:
    """Return blob id to label for every runnable payload the descriptor declares.

    Parameters
    ----------
    descriptor:
        Parsed sparse run descriptor, or ``None`` when the artifact has none (then no
        label is known and the binders refuse or skip the stray blobs themselves).

    Returns
    -------
    dict[str, str]
        ``runnable_weight_<i>`` to the i-th sorted persistent state name,
        ``runnable_nonpersistent_buffer_<i>`` to the i-th sorted non-persistent buffer
        name, and each declared activation member's blob id to its op label.
    """

    if descriptor is None:
        return {}
    labels: dict[str, str] = {}
    layers = descriptor.payload_layers
    if layers.weights.present:
        for index, name in enumerate(_sorted_state_names(descriptor, persistent=True)):
            labels[f"runnable_weight_{index:05d}"] = name
    if layers.nonpersistent_buffers.present:
        for index, name in enumerate(_sorted_state_names(descriptor, persistent=False)):
            labels[f"runnable_nonpersistent_buffer_{index:05d}"] = name
    # Only ``ActivationPayloadLayerDescriptor`` carries ``members``.
    for member in getattr(layers.activations, "members", ()):
        labels[member.blob_id] = member.op_label
    return labels


def runnable_payload_index(
    manifest: Mapping[str, Any], descriptor: SparseRunDescriptor | None
) -> RunnablePayloadIndex:
    """Build the runnable payload references of a 3.0 artifact from its manifest.

    Parameters
    ----------
    manifest:
        Decoded 3.0 manifest (``tensors`` blob-id index, ``files`` digests).
    descriptor:
        Parsed sparse run descriptor attached to the loaded trace, or ``None``.

    Returns
    -------
    RunnablePayloadIndex
        One reference per runnable blob id, in blob-id order; record payloads
        (``out``, ``grad`` and the rest) are not runnable payloads and are skipped.

    Raises
    ------
    ArtifactError
        ``manifest_schema_invalid`` when a runnable blob id names a chunk the manifest does
        not list, or when the descriptor declares the family but not that blob id.
    """

    index = manifest.get("tensors") or {}
    files = manifest.get("files") or {}
    labels = runnable_labels(descriptor)
    refs: list[RunnablePayloadRef] = []
    for blob_id in sorted(index):
        kind = _family_kind(blob_id)
        if kind is None:
            continue
        fields = index[blob_id]
        blob, key = str(fields.get("blob", "")), fields.get("key")
        digest = files.get(blob, {}).get("sha256") if isinstance(files.get(blob), Mapping) else None
        if not blob or not isinstance(key, str) or not isinstance(digest, str):
            raise _index_error(blob_id, "its chunk is not a listed artifact file", code=_INDEX_CODE)
        label = labels.get(blob_id)
        if label is None and descriptor is not None:
            raise _index_error(
                blob_id, "the run descriptor declares no slot for it", code=_INDEX_CODE
            )
        refs.append(
            RunnablePayloadRef(
                blob_id, kind, label if label is not None else blob_id, blob, digest, key
            )
        )
    return RunnablePayloadIndex(tuple(refs))


def _integrity_error(blob_path: Path, detail: str, *, code: str = _INTEGRITY_CODE) -> ArtifactError:
    """Return the integrity refusal for a runnable payload chunk."""

    return ArtifactError(
        f"Runnable payload chunk {blob_path} failed its integrity check: {detail}.",
        code=code,
        remedy=_RESAVE,
        detail=detail,
    )


def _read_one_tensor_blob(blob_path: Path, map_location: str | torch.device) -> torch.Tensor:
    """Read the tensor of a 2.x one-tensor blob file (key ``data``)."""

    try:
        tensor = load_file(blob_path, device=str(map_location)).get(ONE_TENSOR_KEY)
    except ImportError as exc:
        raise ArtifactError(
            "Portable bundle load requires the safetensors backend. Install safetensors>=0.4.",
            code=_INTEGRITY_CODE,
            remedy="install safetensors>=0.4",
        ) from exc
    except (OSError, SafetensorError, ValueError) as exc:
        raise _integrity_error(
            blob_path, f"the file could not be read ({exc})", code=_INTEGRITY_CODE
        ) from exc
    if tensor is None:
        raise _integrity_error(
            blob_path, f"it lacks the {ONE_TENSOR_KEY!r} tensor", code=_INTEGRITY_CODE
        )
    return tensor


def read_runnable_payload(
    blob_path: Path, entry: Any, map_location: str | torch.device
) -> torch.Tensor:
    """Read one runnable payload tensor after verifying its chunk digest.

    Parameters
    ----------
    blob_path:
        Resolved chunk path (already contained under the artifact's ``blobs/``).
    entry:
        The payload's reference: ``sha256`` of the chunk and ``key`` inside it
        (``None`` selects the 2.x one-tensor layout).
    map_location:
        Device the tensor is loaded to.

    Returns
    -------
    torch.Tensor
        The stored tensor (a complex value is stored as its real view).

    Raises
    ------
    ArtifactError
        ``bundle_metadata_integrity_refused`` when the chunk digest differs from the
        manifest, the key is absent, or the file cannot be read.
    """

    key = getattr(entry, "key", None)
    if key is None:
        if sha256_file(blob_path) != entry.sha256:
            raise _integrity_error(
                blob_path,
                f"sha256 differs from the manifest ({entry.blob_id})",
                code=_INTEGRITY_CODE,
            )
        return _read_one_tensor_blob(blob_path, map_location)
    try:
        observed = chunk_sha256(blob_path)
    except OSError as exc:
        raise _integrity_error(
            blob_path, f"the file could not be read ({exc})", code=_INTEGRITY_CODE
        ) from exc
    if observed != entry.sha256:
        raise _integrity_error(
            blob_path, f"sha256 differs from the manifest ({entry.blob_id})", code=_INTEGRITY_CODE
        )
    try:
        with safe_open(str(blob_path), framework="pt", device=str(map_location)) as handle:
            tensor_keys = handle.keys()
            if key not in tensor_keys:
                raise _integrity_error(
                    blob_path, f"it lacks key {key!r} ({entry.blob_id})", code=_INTEGRITY_CODE
                )
            return handle.get_tensor(key)
    except (OSError, SafetensorError, ValueError) as exc:
        raise _integrity_error(
            blob_path, f"key {key!r} could not be read ({exc})", code=_INTEGRITY_CODE
        ) from exc


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
        return _binders().attach_readiness(trace, None)
    report = _binders().attach_readiness(trace, component.descriptor)
    bind_runnable_families(trace, component.families)
    return report


def _family_disposition(trace: Any, present: bool) -> str:
    """Return ``bind``, ``skip`` or ``error`` for one family."""

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
            _binders().bind_state(trace, values or {})
        elif family == "nonpersistent_buffers":
            _binders().bind_buffers(trace, values or {})
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
        if _binders().byte_digest(tensor) != member.byte_digest:
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
