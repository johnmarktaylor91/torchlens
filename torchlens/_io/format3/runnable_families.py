"""Tensor families of the 3.0 runnable component: safetensors files and tensor references.

Contract: ``docs/maintainers/saved_format_3.md``, sections 3 and 7. Each family
(state dict, non-persistent buffers, archived activations) is written as
safetensors files under ``runnable/``; every tensor is one opaque key ``t<k>``
(monotone per component) in exactly one file, and the manifest's ``runnable``
object maps each member name to a contract tensor reference
(``{"$tl": "tensor", "blob", "key", "dtype", "shape", "device",
"requires_grad", "codec"}``, plus ``"complex": true`` for a complex tensor
stored as its real view). The reader checks every reference against the file
it names: the key exists, dtype and shape agree, and no tensor in a file goes
unreferenced.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import torch
from safetensors.torch import load_file, save_file

from ..._utils.hashing import sha256_file
from ..format_errors import ArtifactError
from ..paths import reject_symlink_path
from .contract import REFUSAL_CODES, RUNNABLE_DIR, TENSOR_REFERENCE_FIELDS, TYPED_VALUE_KEY

#: Largest safetensors file a chunked family writes, in bytes (contract section 3).
CHUNK_BYTES = 256 * 1024 * 1024
#: Payload codec every runnable family is written with.
FAMILY_CODEC = "torch_safetensors_v1"
#: Keys of one ``files`` row.
FILE_ROW_KEYS = frozenset({"sha256", "bytes"})
#: Keys of a tensor reference; ``complex`` is optional.
_REFERENCE_KEYS = frozenset({TYPED_VALUE_KEY, *TENSOR_REFERENCE_FIELDS})
_REQUIRED_REFERENCE_KEYS = _REFERENCE_KEYS - {"complex"}
#: Remedy for an altered component.
ALTERED = "save the trace again at level='runnable'; the runnable component was altered"


def refuse(situation: str, message: str, remedy: str) -> NoReturn:
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


def file_row(root: Path, relative: str) -> dict[str, Any]:
    """Make one written file owner-only and return its manifest ``files`` row."""

    target = root / relative
    os.chmod(target, 0o600)
    return {"sha256": sha256_file(target), "bytes": target.stat().st_size}


def _stored_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the host-resident bytes of one tensor as safetensors stores them."""

    host = tensor.detach().resolve_conj().resolve_neg().contiguous().cpu()
    return torch.view_as_real(host).contiguous() if host.is_complex() else host


def _reference(blob: str, key: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Build the contract tensor reference of one written tensor."""

    reference: dict[str, Any] = {
        TYPED_VALUE_KEY: "tensor",
        "blob": blob,
        "key": key,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": [int(dim) for dim in tensor.shape],
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
        "codec": FAMILY_CODEC,
    }
    if tensor.is_complex():
        reference["complex"] = True
    return reference


def _chunks(
    members: Sequence[tuple[str, torch.Tensor]], *, chunked: bool
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Group members into files of at most ``CHUNK_BYTES`` (one group when unchunked).

    A tensor larger than the limit gets a file of its own.
    """

    current: list[tuple[str, torch.Tensor]] = []
    size = 0
    for name, tensor in members:
        nbytes = tensor.numel() * tensor.element_size()
        if chunked and current and size + nbytes > CHUNK_BYTES:
            yield current
            current, size = [], 0
        current.append((name, tensor))
        size += nbytes
    if current:
        yield current


@dataclass(frozen=True)
class FamilySink:
    """Where a component's families are written.

    Attributes
    ----------
    root:
        Artifact directory.
    files:
        Manifest ``files`` rows; one row is added per written file.
    keys:
        The component's key counter; each tensor takes the next ``t<k>``.
    """

    root: Path
    files: dict[str, dict[str, Any]]
    keys: Iterator[int]


def write_family(
    sink: FamilySink,
    stem: str,
    members: Sequence[tuple[str, torch.Tensor]],
    *,
    chunked: bool,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Write one family and return its file paths and its name-to-reference map.

    Parameters
    ----------
    sink:
        Artifact directory, ``files`` rows and key counter.
    stem:
        File stem under ``runnable/``: chunked families write
        ``<stem>-NNNN.safetensors`` from 0001, an unchunked one ``<stem>.safetensors``.
    members:
        ``(name, tensor)`` per member, in order.
    chunked:
        Whether the family splits at ``CHUNK_BYTES``.
    """

    paths: list[str] = []
    references: dict[str, dict[str, Any]] = {}
    for number, group in enumerate(_chunks(members, chunked=chunked), start=1):
        suffix = f"-{number:04d}" if chunked else ""
        relative = f"{RUNNABLE_DIR}/{stem}{suffix}.safetensors"
        stored: dict[str, torch.Tensor] = {}
        storages: set[int] = set()
        for name, tensor in group:
            key = f"t{next(sink.keys)}"
            value = _stored_tensor(tensor)
            storage = value.untyped_storage().data_ptr()
            # Tied weights and view activations share storage, which safetensors refuses.
            stored[key] = value.clone() if storage in storages else value
            storages.add(storage)
            references[name] = _reference(relative, key, tensor)
        save_file(stored, str(sink.root / relative))
        sink.files[relative] = file_row(sink.root, relative)
        paths.append(relative)
    return paths, references


def checked_file(root: Path, relative: str, files: Mapping[str, Any]) -> Path:
    """Resolve one listed component file and verify it against its ``files`` row."""

    row = files.get(relative)
    if not isinstance(row, Mapping) or set(row) != FILE_ROW_KEYS:
        refuse(
            "manifest_invalid",
            f"The runnable component file {relative!r} has no sha256 and bytes row in the "
            "manifest's files.",
            ALTERED,
        )
    target = root / relative
    reject_symlink_path(
        target, context="runnable component file", code=REFUSAL_CODES["symlink_path"]
    )
    if not target.is_file():
        refuse(
            "blob_missing",
            f"The runnable component file {target} is missing.",
            "copy the whole artifact directory; the runnable component is incomplete",
        )
    if target.stat().st_size != row["bytes"] or sha256_file(target) != row["sha256"]:
        refuse(
            "record_file_corrupt",
            f"The runnable component file {target} does not match its sha256 and byte length.",
            ALTERED,
        )
    return target


def _load_safetensors(target: Path, map_location: str | torch.device) -> dict[str, torch.Tensor]:
    """Load one safetensors file; refuse an unreadable one."""

    try:
        return load_file(str(target), device=str(map_location))
    except (OSError, ValueError, RuntimeError) as exc:
        refuse(
            "record_file_corrupt",
            f"The runnable family file {target} is not a readable safetensors file "
            f"({type(exc).__name__}).",
            ALTERED,
        )


def _checked_reference(name: Any, reference: Any, paths: Sequence[str]) -> Mapping[str, Any]:
    """Check one tensor reference's shape of fields before any tensor is read."""

    keys = set(reference) if isinstance(reference, Mapping) else set()
    if (
        not isinstance(name, str)
        or not isinstance(reference, Mapping)
        or not _REQUIRED_REFERENCE_KEYS <= keys <= _REFERENCE_KEYS
        or reference[TYPED_VALUE_KEY] != "tensor"
        or reference["codec"] != FAMILY_CODEC
        or reference["blob"] not in paths
        or not isinstance(reference["key"], str)
        or not isinstance(reference["shape"], list)
        or reference.get("complex", True) is not True
    ):
        refuse(
            "manifest_invalid",
            f"The runnable tensor reference for {name!r} is not a contract tensor reference "
            "into one of its family's files.",
            ALTERED,
        )
    return reference


def _decoded(name: str, reference: Mapping[str, Any], stored: torch.Tensor) -> torch.Tensor:
    """Return the tensor a reference names, checking dtype and shape."""

    value = torch.view_as_complex(stored) if reference.get("complex") else stored
    dtype = str(value.dtype).replace("torch.", "")
    if dtype != reference["dtype"] or list(value.shape) != reference["shape"]:
        refuse(
            "record_file_corrupt",
            f"The runnable tensor {name!r} has dtype {dtype} and shape {list(value.shape)}; "
            f"its reference records {reference['dtype']} and {reference['shape']}.",
            ALTERED,
        )
    return value


def read_family(
    root: Path,
    paths: Sequence[str],
    references: Any,
    files: Mapping[str, Any],
    map_location: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Read one family's files and resolve every member reference.

    Parameters
    ----------
    root:
        Artifact directory.
    paths:
        The family's file paths, already checked against the fixed names.
    references:
        The manifest's name-to-tensor-reference map for the family.
    files:
        The manifest's ``files`` rows.
    map_location:
        Device the tensors load onto.

    Returns
    -------
    dict[str, torch.Tensor]
        Member name to tensor.
    """

    if not isinstance(references, Mapping) or not references:
        refuse("manifest_invalid", "A runnable family must reference its tensors.", ALTERED)
    checked = {name: _checked_reference(name, ref, paths) for name, ref in references.items()}
    stored = {
        path: _load_safetensors(checked_file(root, path, files), map_location) for path in paths
    }
    claimed = [(ref["blob"], ref["key"]) for ref in checked.values()]
    present = [(path, key) for path, tensors in stored.items() for key in tensors]
    if len(set(claimed)) != len(claimed) or set(claimed) != set(present):
        refuse(
            "record_file_corrupt",
            "The runnable family files and their tensor references disagree: a key is "
            "missing, repeated or unreferenced.",
            ALTERED,
        )
    return {
        name: _decoded(name, ref, stored[ref["blob"]][ref["key"]]) for name, ref in checked.items()
    }
