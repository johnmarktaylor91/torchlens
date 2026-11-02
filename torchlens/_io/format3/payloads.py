"""Tensor payloads of the 3.0 saved format: chunked safetensors files and lazy references.

Every tensor is one opaque key (``t<k>``) in one chunk
``blobs/<codec>-<nnnn>.safetensors``; a chunk closes when it would pass
``PAYLOAD_CHUNK_BYTES`` so streaming saves append by opening the next chunk,
and relabeling a row edits its record cell, never a blob. No tensor is written
inline. Design:
``docs/maintainers/saved_format_3_design.md``, section 6.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..._utils.hashing import sha256_file
from .._durability import fsync_file
from ..format_errors import ArtifactError
from ..paths import reject_symlink_path
from .contract import BLOBS_DIR, PAYLOAD_CODEC_IDS

if TYPE_CHECKING:
    from ..manifest import TensorEntry

#: Bytes after which the chunk writer closes the current chunk and opens the next.
PAYLOAD_CHUNK_BYTES: Final = 256 << 20
#: Private test override of the chunk cap (bytes); never documented for users.
_CHUNK_BYTES_ENV: Final = "TORCHLENS_PAYLOAD_CHUNK_BYTES"
#: Codec id of torch tensors written as themselves.
TORCH_CODEC: Final = PAYLOAD_CODEC_IDS[0]
#: Codec id of backend payloads carried as numpy-transport tensors.
NUMPY_CODEC: Final = PAYLOAD_CODEC_IDS[1]
#: File name stem of the intervention recipe's tensor chunks.
RECIPE_CHUNK_STEM: Final = "recipe"


def chunk_bytes_cap() -> int:
    """Return the chunk cap in bytes (the private environment override wins).

    Returns
    -------
    int
        ``PAYLOAD_CHUNK_BYTES`` unless ``TORCHLENS_PAYLOAD_CHUNK_BYTES`` holds a
        positive integer.
    """

    text = os.environ.get(_CHUNK_BYTES_ENV, "").strip()
    if text.isdigit() and int(text) > 0:
        return int(text)
    return PAYLOAD_CHUNK_BYTES


def host_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the contiguous host copy of a tensor as it is written to a chunk.

    Parameters
    ----------
    tensor:
        Payload tensor (any device; conjugate and negative views resolved).

    Returns
    -------
    torch.Tensor
        Contiguous CPU tensor that owns its storage exactly (safetensors
        refuses two keys of one file sharing memory, and views of one
        activation do); a complex tensor becomes its real view.
    """

    source = tensor.detach().resolve_conj().resolve_neg()
    host = torch.view_as_real(source) if source.is_complex() else source
    host = host.contiguous().cpu()
    owns_storage = host.untyped_storage().nbytes() == host.numel() * host.element_size()
    if owns_storage and host.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr():
        return host
    return host.clone()


class ChunkWriter:
    """Write tensors of one codec into capped safetensors chunks.

    Parameters
    ----------
    root:
        Artifact directory (``blobs/`` is created under it).
    codec:
        One of ``contract.PAYLOAD_CODEC_IDS``.
    counter:
        Shared one-element key counter, so keys stay monotone across the
        codecs of one artifact.
    directory:
        Directory, relative to ``root``, that holds the chunks.
    stem:
        File name stem of the chunks; the codec id unless a series of its own
        (the intervention recipe's ``recipe-<nnnn>``) is wanted.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        codec: str,
        counter: list[int] | None = None,
        directory: str = BLOBS_DIR,
        stem: str | None = None,
    ) -> None:
        if codec not in PAYLOAD_CODEC_IDS:
            raise ArtifactError(
                f"Unknown payload codec {codec!r}; the 3.0 format writes {PAYLOAD_CODEC_IDS}.",
                code="metadata_value_unportable",
                remedy="save the tensor with a supported codec",
            )
        self.root = Path(root)
        self.codec = codec
        self.directory = directory
        self.stem = stem or codec
        self._counter = counter if counter is not None else [0]
        self._cap = chunk_bytes_cap()
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._index = 0
        self.written: list[str] = []

    def _chunk_name(self) -> str:
        """Return the relative path of the chunk being filled."""

        return f"{self.directory}/{self.stem}-{self._index:04d}.safetensors"

    def submit(self, tensor: torch.Tensor) -> tuple[str, str]:
        """Queue one tensor into the open chunk.

        Parameters
        ----------
        tensor:
            Host tensor from :func:`host_tensor`.

        Returns
        -------
        tuple[str, str]
            ``(chunk relative path, key)`` of the stored tensor.
        """

        size = int(tensor.numel() * tensor.element_size())
        if self._pending and self._pending_bytes + size > self._cap:
            self.flush()
        key = f"t{self._counter[0]}"
        self._counter[0] += 1
        self._pending[key] = tensor
        self._pending_bytes += size
        return self._chunk_name(), key

    def flush(self) -> None:
        """Write the open chunk to disk and start the next one."""

        if not self._pending:
            return
        relative = self._chunk_name()
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            save_file(self._pending, str(path))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ArtifactError(
                f"A tensor payload could not be written to {relative}: {exc}",
                code="metadata_value_unportable",
                remedy="save without that payload, or convert it to a dense tensor first",
            ) from exc
        fsync_file(path)
        self.written.append(relative)
        self._pending = {}
        self._pending_bytes = 0
        self._index += 1


class PayloadStore:
    """All chunk writers of one artifact, with one shared key counter.

    Parameters
    ----------
    root:
        Artifact directory.
    directory:
        Chunk directory relative to ``root`` (``blobs`` or ``runnable``).
    """

    def __init__(self, root: str | Path, *, directory: str = BLOBS_DIR) -> None:
        self.root = Path(root)
        self.directory = directory
        self._counter = [0]
        self._writers: dict[str, ChunkWriter] = {}
        self._recipe: ChunkWriter | None = None

    def recipe_writer(self) -> ChunkWriter:
        """Return the writer of the intervention recipe's tensors (``recipe-<nnnn>``).

        It shares the artifact's key counter, so recipe keys stay monotone
        with the record payload keys.
        """

        if self._recipe is None:
            self._recipe = ChunkWriter(
                self.root,
                codec=TORCH_CODEC,
                counter=self._counter,
                directory=self.directory,
                stem=RECIPE_CHUNK_STEM,
            )
        return self._recipe

    def writer(self, codec: str) -> ChunkWriter:
        """Return the chunk writer of one codec, opening it on first use."""

        if codec not in self._writers:
            self._writers[codec] = ChunkWriter(
                self.root, codec=codec, counter=self._counter, directory=self.directory
            )
        return self._writers[codec]

    def submit(self, tensor: torch.Tensor, *, codec: str = TORCH_CODEC) -> tuple[str, str]:
        """Store one host tensor; return ``(chunk relative path, key)``."""

        return self.writer(codec).submit(tensor)

    def close(self) -> list[str]:
        """Flush every open chunk; return every chunk written, in order."""

        written: list[str] = []
        for codec in sorted(self._writers):
            writer = self._writers[codec]
            writer.flush()
            written.extend(writer.written)
        if self._recipe is not None:
            self._recipe.flush()
            written.extend(self._recipe.written)
        return written

    @property
    def codecs(self) -> list[str]:
        """Codec ids that wrote at least one tensor, in contract order."""

        used = set(self._writers)
        if self._recipe is not None and self._recipe.written:
            used.add(self._recipe.codec)
        return [codec for codec in PAYLOAD_CODEC_IDS if codec in used]


def open_chunk_writer(root: str | Path, *, codec: str) -> ChunkWriter:
    """Return the chunk writer for one codec family under ``root/blobs``.

    Parameters
    ----------
    root:
        Artifact directory.
    codec:
        One of ``contract.PAYLOAD_CODEC_IDS``.

    Returns
    -------
    ChunkWriter
        A writer with its own key counter (use :class:`PayloadStore` when one
        artifact writes more than one codec).
    """

    return ChunkWriter(root, codec=codec)


def tensor_reference(
    tensor: torch.Tensor,
    *,
    blob: str | None,
    key: str | None,
    codec: Any = TORCH_CODEC,
) -> dict[str, Any]:
    """Return the tensor reference fields of one stored tensor.

    Parameters
    ----------
    tensor:
        The tensor as captured (dtype, shape and device are taken from it).
    blob, key:
        Chunk relative path and key; both ``None`` in a structure-only
        artifact (no payload written).
    codec:
        Codec id, or for a backend payload ``{"id": codec id, **hints}``.

    Returns
    -------
    dict[str, Any]
        ``contract.TENSOR_REFERENCE_FIELDS`` values (``blob`` and ``key`` only
        when a payload was written; ``complex`` only when true).
    """

    fields: dict[str, Any] = {}
    if blob is not None and key is not None:
        fields["blob"] = blob
        fields["key"] = key
    fields.update(
        {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": [int(dim) for dim in tensor.shape],
            "device": str(tensor.device),
            "requires_grad": bool(tensor.requires_grad),
            "codec": codec,
        }
    )
    if tensor.is_complex():
        fields["complex"] = True
    return fields


def contained_chunk(root: Path, blob: str) -> Path:
    """Resolve a record's chunk path, refusing any path that leaves its chunk directory.

    The artifact is untrusted: a reference must name ``<directory>/<file>``
    under the artifact root, with no absolute path, no ``..`` and no symlinked
    directory or file redirecting outside (a digest does not help, the same
    hand wrote it).

    Parameters
    ----------
    root:
        Artifact directory.
    blob:
        Chunk path relative to ``root``, as stored in the reference.

    Returns
    -------
    Path
        The resolved chunk path.

    Raises
    ------
    ArtifactError
        ``bundle_load_failed`` for an absolute, traversing or escaping path.
    """

    candidate = Path(blob)
    problem = None
    if candidate.is_absolute():
        problem = "an absolute path"
    elif ".." in candidate.parts:
        problem = "a parent traversal"
    elif len(candidate.parts) != 2:
        problem = "not <directory>/<chunk file>"
    if problem is None:
        directory = root / candidate.parts[0]
        reject_symlink_path(directory, context="payload chunk directory", code="bundle_load_failed")
        try:
            resolved = (root / candidate).resolve()
            resolved.relative_to(directory.resolve())
        except (OSError, ValueError):
            problem = "a path traversal outside its chunk directory"
        else:
            return resolved
    raise ArtifactError(
        f"Payload chunk reference {blob!r} is {problem}; refusing to read it.",
        code="bundle_load_failed",
        remedy="the artifact was altered; load only artifacts you trust, or re-save it",
    )


def read_tensor(root: str | Path, *, blob: str, key: str) -> torch.Tensor:
    """Read one tensor by chunk file and key; the eager half of a lazy reference.

    Parameters
    ----------
    root:
        Artifact directory.
    blob:
        Chunk path relative to ``root``.
    key:
        Tensor key inside the chunk.

    Returns
    -------
    torch.Tensor
        The stored tensor on CPU (a complex payload is returned as stored, its
        real view; the caller applies ``view_as_complex`` from the reference).

    Raises
    ------
    ArtifactError
        ``bundle_load_failed`` when the chunk or key is missing or unreadable.
    """

    path = contained_chunk(Path(root), blob)
    try:
        with safe_open(str(path), framework="pt") as handle:
            if key not in handle.keys():
                raise ArtifactError(
                    f"Tensor key {key!r} is missing from {blob}.",
                    code="bundle_load_failed",
                    remedy="the artifact is incomplete; re-save it",
                )
            return handle.get_tensor(key)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactError(
            f"The payload chunk {blob} could not be read: {exc}",
            code="bundle_load_failed",
            remedy="the artifact is incomplete or damaged; re-save it",
        ) from exc


def submit_tensor_payload(
    *,
    store: PayloadStore,
    tensor_refs: dict[str, dict[str, Any]],
    blob_id: str,
    tensor: torch.Tensor,
    kind: str,
    label: str,
) -> dict[str, Any]:
    """Queue one torch tensor payload into the artifact's chunk files.

    Parameters
    ----------
    store:
        Chunk store of the staged artifact (``blobs/<codec>-<nnnn>.safetensors``).
    tensor_refs:
        Blob id to tensor reference fields; this payload's entry is added.
    blob_id:
        Opaque blob identifier the scrubbed state references.
    tensor:
        Tensor payload to persist.
    kind:
        Logical tensor kind.
    label:
        Human-readable TorchLens layer label.

    Returns
    -------
    dict[str, Any]
        Fields of the manifest ``TensorEntry``; ``sha256`` is filled by
        ``seal_tensor_entries`` once the chunk is on disk.
    """

    host = host_tensor(tensor)
    blob, key = store.submit(host, codec=TORCH_CODEC)
    tensor_refs[blob_id] = tensor_reference(tensor, blob=blob, key=key)
    return {
        "blob_id": blob_id,
        "kind": kind,
        "label": label,
        "relative_path": blob,
        "backend": "safetensors",
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device_at_save": str(tensor.device),
        "layout": str(tensor.layout).replace("torch.", ""),
        "bytes": int(host.numel() * host.element_size()),
        "sha256": "",
        "requires_grad": bool(tensor.requires_grad),
    }


def seal_tensor_entries(tmp_path: Path, entries: list[TensorEntry]) -> list[TensorEntry]:
    """Fill each entry's ``sha256`` with the digest of its closed chunk file.

    Parameters
    ----------
    tmp_path:
        Staged artifact directory.
    entries:
        Entries queued while the chunks were open.

    Returns
    -------
    list[TensorEntry]
        The entries with chunk digests, in the same order.
    """

    digests: dict[str, str] = {}
    sealed: list[TensorEntry] = []
    for entry in entries:
        if entry.relative_path not in digests:
            digests[entry.relative_path] = sha256_file(tmp_path / entry.relative_path)
        sealed.append(replace(entry, sha256=digests[entry.relative_path]))
    return sealed
