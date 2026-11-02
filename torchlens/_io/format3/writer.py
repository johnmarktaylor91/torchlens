"""The 3.0 writer: a trace or bundle to typed JSON records plus safetensors blobs.

Contract: ``docs/maintainers/saved_format_3.md`` (envelope, records, tensor
references, unencodable values, the intervention recipe, refusals).

``Trace.save`` keeps its one chokepoint (``torchlens._io.bundle.save``): the
entry refusals, the save gates, the level and payload-flag resolution, the
scrub, the blob writes and the atomic publish are unchanged. Where that
chokepoint used to pickle the scrubbed state it now calls
:func:`write_trace_envelope`, which encodes the state into one JSON table per
record kind (``records/<kind>.json``), writes any tensor the scrub left inline
into ``blobs/inline.safetensors``, digests every file and writes the 3.0
``manifest.json``. No pickle file is written at any level. The save preflight
(reload every save once) runs the 3.0 reader's ``validate_artifact`` on the
staged directory before it is published.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .. import _json
from ..format_errors import ArtifactError
from ..manifest import TensorEntry
from ..writer_contract import writer_contract_digest
from . import reader as _reader
from ._encode import StateEncoder
from .contract import (
    BLOBS_DIR,
    FORMAT_MAJOR,
    FORMAT_NAME,
    MANIFEST_FILE,
    PAYLOAD_CODEC_IDS,
    RECORD_CONTRACT_VERSION,
    RECORD_INDEX,
    RECORDS_DIR,
    refuse_writer_not_built,
)

#: Tensor key inside every payload blob file (one tensor per file).
BLOB_TENSOR_KEY = "data"
#: File that holds tensors the scrub left inline in the metadata.
INLINE_BLOB_FILE = f"{BLOBS_DIR}/inline.safetensors"
#: Root files the digest table leaves out: the manifest itself and the
#: staging sentinel that comes off right before publish.
_UNDIGESTED_ROOT_FILES = frozenset({MANIFEST_FILE, "PARTIAL"})
#: Prefix of the unbuilt-door refusal message (``contract.refuse_*_not_built``).
_UNBUILT_DOOR_PREFIX = "The 3.0 saved-format door "


def tensor_reference_fields(entry: TensorEntry) -> dict[str, Any]:
    """Return the tensor reference fields of one written blob entry.

    Parameters
    ----------
    entry:
        Manifest entry of a written payload blob.

    Returns
    -------
    dict[str, Any]
        ``contract.TENSOR_REFERENCE_FIELDS`` values for the reference object.
    """

    return {
        "blob": entry.relative_path,
        "key": BLOB_TENSOR_KEY,
        "dtype": entry.logical_dtype or entry.dtype,
        "shape": list(entry.shape),
        "device": entry.logical_device or entry.device_at_save,
        "requires_grad": bool(entry.requires_grad),
        "codec": entry.codec or PAYLOAD_CODEC_IDS[0],
        "complex": False,
    }


class _InlineTensorSink:
    """Collect tensors the scrub left inline so they land in one blob file."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}

    def add(self, tensor: torch.Tensor) -> dict[str, Any]:
        """Store one inline tensor and return its reference fields."""

        key = f"t{len(self.tensors):06d}"
        host = tensor.detach().resolve_conj().resolve_neg().contiguous().cpu()
        self.tensors[key] = host
        return {
            "blob": INLINE_BLOB_FILE,
            "key": key,
            "dtype": str(host.dtype).replace("torch.", ""),
            "shape": [int(dim) for dim in host.shape],
            "device": str(tensor.device),
            "requires_grad": bool(tensor.requires_grad),
            "codec": PAYLOAD_CODEC_IDS[0],
            "complex": False,
        }

    def write(self, root: Path) -> None:
        """Write the collected tensors (nothing when none were collected)."""

        if self.tensors:
            save_file(self.tensors, str(root / INLINE_BLOB_FILE))


def encode_state(
    scrubbed_state: Mapping[str, Any], tensor_entries: Sequence[TensorEntry]
) -> tuple[StateEncoder, _InlineTensorSink]:
    """Encode scrubbed trace state into record tables.

    Parameters
    ----------
    scrubbed_state:
        Scrubbed trace state from the save scrub.
    tensor_entries:
        Entries of every blob the save wrote.

    Returns
    -------
    tuple[StateEncoder, _InlineTensorSink]
        The filled encoder (tables and disclosures) and the inline tensors.
    """

    sink = _InlineTensorSink()
    refs = {entry.blob_id: tensor_reference_fields(entry) for entry in tensor_entries}
    encoder = StateEncoder(refs, inline_tensor=sink.add)
    encoder.encode_trace_state(scrubbed_state)
    return encoder, sink


def _dump(value: Any) -> bytes:
    """Return the canonical bytes of one JSON file of the artifact."""

    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _write_records(root: Path, encoder: StateEncoder) -> dict[str, dict[str, Any]]:
    """Write one JSON file per non-empty record table; return the index."""

    (root / RECORDS_DIR).mkdir(exist_ok=True)
    index: dict[str, dict[str, Any]] = {}
    for kind, class_path, _ in RECORD_INDEX:
        rows = encoder.tables[kind]
        if not rows:
            continue
        relative = f"{RECORDS_DIR}/{kind}.json"
        payload = {
            "record": kind,
            "class": class_path,
            "tlspec_version": RECORD_CONTRACT_VERSION,
            "rows": rows,
        }
        (root / relative).write_bytes(_dump(payload))
        index[kind] = {"file": relative, "rows": len(rows)}
    return index


def _file_digests(root: Path) -> dict[str, str]:
    """Return ``relative path -> sha256`` for every file except the manifest."""

    digests: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.parent == root and path.name in _UNDIGESTED_ROOT_FILES:
            continue
        digests[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def write_trace_envelope(
    root: Path,
    *,
    base_manifest: Mapping[str, Any],
    scrubbed_state: Mapping[str, Any],
    tensor_entries: Sequence[TensorEntry],
    save_level: str,
    payload_policy: Mapping[str, bool],
    kind: str = "trace",
) -> dict[str, Any]:
    """Write the records, inline blob and 3.0 manifest of one staged trace.

    Parameters
    ----------
    root:
        Staged artifact directory (blobs already written under ``blobs/``).
    base_manifest:
        The unified manifest fields the save already builds (writer identity,
        tensor entries, provenance, disclosures, sites, runnable descriptor).
    scrubbed_state:
        Scrubbed trace state.
    tensor_entries:
        Entries of every written payload blob.
    save_level:
        Resolved save level.
    payload_policy:
        Resolved payload flags (``include_outs`` and the rest).
    kind:
        Artifact kind recorded in the manifest.

    Returns
    -------
    dict[str, Any]
        The manifest written to ``manifest.json``.
    """

    encoder, sink = encode_state(scrubbed_state, tensor_entries)
    sink.write(root)
    records = _write_records(root, encoder)
    manifest: dict[str, Any] = dict(base_manifest)
    manifest.update(
        envelope_fields(
            root,
            kind=kind,
            save_level=save_level,
            payload_policy=payload_policy,
            records=records,
            unencodable=encoder.unencodable,
        )
    )
    (root / MANIFEST_FILE).write_bytes(_dump(manifest))
    return manifest


def envelope_fields(
    root: Path,
    *,
    kind: str,
    save_level: str,
    payload_policy: Mapping[str, bool],
    records: Mapping[str, Mapping[str, Any]],
    unencodable: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the 3.0 envelope fields of a staged artifact.

    Call it last, after every other file is written: the ``files`` table
    digests what is on disk.

    Parameters
    ----------
    root:
        Staged artifact directory.
    kind:
        One of ``contract.ARTIFACT_KINDS``.
    save_level:
        Resolved save level.
    payload_policy:
        Resolved payload flags.
    records:
        Record table index (``kind -> {"file", "rows"}``).
    unencodable:
        Disclosures of values stringified because they had no typed encoding.

    Returns
    -------
    dict[str, Any]
        The envelope keys ``contract.MANIFEST_REQUIRED_FIELDS`` names beyond
        the writer identity, plus the disclosure list.
    """

    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_MAJOR,
        "tlspec_version": RECORD_CONTRACT_VERSION,
        "kind": kind,
        "save_level": save_level,
        "payload_policy": dict(payload_policy),
        "contract_digest": writer_contract_digest(),
        "codecs": list(PAYLOAD_CODEC_IDS),
        "records": {name: dict(entry) for name, entry in records.items()},
        "unencodable": [dict(item) for item in unencodable],
        "files": _file_digests(root),
    }


def check_written_json(root: Path) -> None:
    """Re-read every JSON file of a staged artifact through the bounded reader.

    Parameters
    ----------
    root:
        Staged artifact directory.

    Raises
    ------
    ArtifactError
        ``bundle_save_failed`` when a written file would not read back.
    """

    for path in [root / MANIFEST_FILE, *sorted((root / RECORDS_DIR).glob("*.json"))]:
        try:
            _json.read_bounded(path)
        except (OSError, ValueError) as exc:
            raise ArtifactError(
                f"The 3.0 writer produced {path.name}, which does not read back: {exc}",
                code="bundle_save_failed",
                remedy="report this; the writer and the bounded reader disagree",
            ) from exc


def preflight_reload(root: Path) -> None:
    """Reload a staged artifact once before it is published (late.21).

    The writer re-reads its own JSON through the bounded reader, then the
    3.0 reader validates every record against the contract. Until the reader
    lane's ``validate_artifact`` is merged, its stub refusal is the only
    refusal this step tolerates.

    Parameters
    ----------
    root:
        Staged artifact directory.
    """

    check_written_json(root)
    try:
        _reader.validate_artifact(root)
    except ArtifactError as exc:
        if not str(exc).startswith(_UNBUILT_DOOR_PREFIX):
            raise


def write_trace(
    trace: Any,
    path: str | Path,
    *,
    level: str = "portable",
    overwrite: bool = False,
    payload_policy: dict[str, bool] | None = None,
) -> None:
    """Write one trace as a 3.0 artifact directory.

    This is ``trace.save``: the save chokepoint writes the 3.0 envelope.

    Parameters
    ----------
    trace:
        Live trace to persist.
    path:
        Target directory; refused when it exists unless ``overwrite``.
    level:
        One of ``torchlens._io.format3.contract.SAVE_LEVELS``.
    overwrite:
        Replace an existing artifact atomically.
    payload_policy:
        Explicit payload flags (``include_outs`` and the rest); conflicts with
        the level refuse typed.
    """

    trace.save(path, level=level, overwrite=overwrite, **dict(payload_policy or {}))


def write_bundle(bundle: Any, path: str | Path, *, level: str = "portable") -> None:
    """Write a bundle: one member artifact per trace plus the relation table.

    This is ``bundle.save``; every member is written by :func:`write_trace`'s
    chokepoint.

    Parameters
    ----------
    bundle:
        Live bundle to persist.
    path:
        Target directory.
    level:
        Save level applied to every member.
    """

    bundle.save(path, level=level)


def write_intervention_recipe(spec: Any, path: str | Path) -> None:
    """Write a staged intervention spec as the typed recipe file.

    The same grammar serves the recipe inside a trace artifact and the
    separate recipe ``save_intervention`` writes.

    Parameters
    ----------
    spec:
        Staged intervention spec (rules and built-in edits).
    path:
        Target file.
    """

    refuse_writer_not_built("write_intervention_recipe", "torchlens/_io/format3/writer.py")
