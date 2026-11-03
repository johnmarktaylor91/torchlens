"""Runnable payloads of a 3.0 artifact: the blob-id index as payload references, keyed chunk reads.

The 3.0 writer indexes every stored tensor as ``manifest["tensors"][blob_id] = {"blob", "key"}``
and keeps the sparse run descriptor in ``manifest["run"]``. The three runnable payload
families are named by their blob-id prefixes (``runnable_weight_``,
``runnable_nonpersistent_buffer_``, ``runnable_activation_``); the labels the binders key on
come from the parsed descriptor in the order the save used: sorted persistent state names,
sorted non-persistent buffer names, and the declared activation members. Design:
``docs/maintainers/saved_format_3_design.md``, section 12.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file

from .._utils.hashing import sha256_file
from ._chunk_digest import chunk_sha256
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


def _index_error(blob_id: str, detail: str) -> ArtifactError:
    return ArtifactError(
        f"Runnable payload {blob_id!r} does not satisfy the 3.0 saved-format contract: {detail}.",
        code=_INDEX_CODE,
        remedy=_RESAVE,
        blob_id=blob_id,
        detail=detail,
    )


def _sorted_state_names(descriptor: SparseRunDescriptor, *, persistent: bool) -> list[str]:
    """Return the state names of one binding family in the save's blob-id order."""

    return sorted(
        slot.state_binding.state_dict_name
        for slot in descriptor.tensor_slots
        if slot.state_binding is not None
        and slot.state_binding.persistent is persistent
        and (persistent or slot.role == _BUFFER_ROLE)
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
            raise _index_error(blob_id, "its chunk is not a listed artifact file")
        label = labels.get(blob_id)
        if label is None and descriptor is not None:
            raise _index_error(blob_id, "the run descriptor declares no slot for it")
        refs.append(
            RunnablePayloadRef(
                blob_id, kind, label if label is not None else blob_id, blob, digest, key
            )
        )
    return RunnablePayloadIndex(tuple(refs))


def _integrity_error(blob_path: Path, detail: str) -> ArtifactError:
    return ArtifactError(
        f"Runnable payload chunk {blob_path} failed its integrity check: {detail}.",
        code=_INTEGRITY_CODE,
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
        raise _integrity_error(blob_path, f"the file could not be read ({exc})") from exc
    if tensor is None:
        raise _integrity_error(blob_path, f"it lacks the {ONE_TENSOR_KEY!r} tensor")
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
            raise _integrity_error(blob_path, f"sha256 differs from the manifest ({entry.blob_id})")
        return _read_one_tensor_blob(blob_path, map_location)
    try:
        observed = chunk_sha256(blob_path)
    except OSError as exc:
        raise _integrity_error(blob_path, f"the file could not be read ({exc})") from exc
    if observed != entry.sha256:
        raise _integrity_error(blob_path, f"sha256 differs from the manifest ({entry.blob_id})")
    try:
        with safe_open(str(blob_path), framework="pt", device=str(map_location)) as handle:
            if key not in handle.keys():
                raise _integrity_error(blob_path, f"it lacks key {key!r} ({entry.blob_id})")
            return handle.get_tensor(key)
    except (OSError, SafetensorError, ValueError) as exc:
        raise _integrity_error(blob_path, f"key {key!r} could not be read ({exc})") from exc
