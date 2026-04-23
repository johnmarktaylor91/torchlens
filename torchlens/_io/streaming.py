"""Streaming bundle writer used during forward-pass activation capture."""

from __future__ import annotations

import pickle
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from . import IO_FORMAT_VERSION, TorchLensIOError
from .manifest import Manifest, TensorEntry, sha256_of_file
from .tensor_policy import FailReason, Ok, SkipReason, is_supported_for_save
from .._state import pause_logging
from .. import __version__ as TORCHLENS_VERSION

PARTIAL_SENTINEL = "PARTIAL"
REASON_SENTINEL = "REASON.txt"
_BLOB_TENSOR_KEY = "data"
BlobSpec = tuple[str, torch.Tensor, str, str]


def next_blob_id(blob_index: int) -> str:
    """Return the canonical zero-padded blob id for one monotonic counter.

    Parameters
    ----------
    blob_index:
        One-based blob counter.

    Returns
    -------
    str
        Zero-padded blob id.
    """

    return f"{blob_index:010d}"


class BundleStreamWriter:
    """Persist activation blobs incrementally into a temp TorchLens bundle.

    Parameters
    ----------
    path:
        Final bundle directory path.
    strict:
        Streaming bundles are always strict. Passing ``False`` is rejected.
    """

    def __init__(self, path: str | Path, *, strict: bool = True) -> None:
        """Create the temp bundle directory used for streaming writes.

        Parameters
        ----------
        path:
            Final bundle directory path.
        strict:
            Streaming bundles are always strict. Passing ``False`` is rejected.

        Raises
        ------
        TorchLensIOError
            If the target path is invalid or the temp directory cannot be created.
        """

        if not strict:
            raise TorchLensIOError("Streaming activation save is always strict.")

        self.final_path = Path(path)
        if self.final_path.is_symlink():
            raise TorchLensIOError(f"Refusing symlinked save target: {self.final_path}.")
        if self.final_path.exists():
            raise TorchLensIOError(f"Bundle path already exists: {self.final_path}")

        self.tmp_path = self.final_path.parent / f"{self.final_path.name}.tmp.{uuid.uuid4().hex}"
        self.blobs_path = self.tmp_path / "blobs"
        self._blob_counter = 0
        self._saw_first_payload = False
        self._closed = False
        self._finalized = False
        self._tensor_entries: list[TensorEntry] = []
        self._entries_by_blob_id: dict[str, TensorEntry] = {}

        try:
            self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
            self.tmp_path.mkdir()
            self.blobs_path.mkdir()
        except OSError as exc:
            raise TorchLensIOError(
                f"Failed to create streaming temp bundle at {self.tmp_path}."
            ) from exc

    def next_blob_id(self) -> str:
        """Return the next monotonic blob id for this writer.

        Returns
        -------
        str
            Zero-padded blob id.
        """

        self._blob_counter += 1
        return next_blob_id(self._blob_counter)

    def write_blob(
        self,
        blob_id: str,
        tensor: torch.Tensor,
        *,
        kind: str,
        label: str,
    ) -> TensorEntry:
        """Write one tensor blob into ``blobs/`` and record its manifest entry.

        Parameters
        ----------
        blob_id:
            Opaque zero-padded blob identifier.
        tensor:
            Tensor payload to persist.
        kind:
            Logical tensor kind.
        label:
            Human-readable or provisional label for the tensor owner.

        Returns
        -------
        TensorEntry
            Recorded manifest entry.

        Raises
        ------
        TorchLensIOError
            If the tensor is unsupported or writing fails.
        """

        self._ensure_writable()
        if not isinstance(tensor, torch.Tensor):
            reason = (
                "Streaming activation save requires activation_postfunc outputs to be torch.Tensor "
                f"instances, but blob_id={blob_id} ({label}) received {type(tensor).__name__}."
            )
            self.abort(reason)
            raise TorchLensIOError(reason)

        self._saw_first_payload = True
        decision = is_supported_for_save(tensor, strict=True)
        if not isinstance(decision, Ok):
            if isinstance(decision, (SkipReason, FailReason)):
                reason_text = decision.text
            else:
                reason_text = "unsupported tensor"
            reason = (
                f"Unsupported tensor for streaming activation save at {label} "
                f"(blob_id={blob_id}, kind={kind}): {reason_text}"
            )
            self.abort(reason)
            raise TorchLensIOError(reason)
        if blob_id in self._entries_by_blob_id:
            reason = f"Duplicate streaming blob_id={blob_id} for {label}."
            self.abort(reason)
            raise TorchLensIOError(reason)

        try:
            entry = self._write_tensor_blob(blob_id=blob_id, tensor=tensor, kind=kind, label=label)
        except OSError as exc:
            reason = f"Failed to write streaming blob_id={blob_id} for {label}: {exc}"
            self.abort(reason)
            raise TorchLensIOError(reason) from exc

        self._tensor_entries.append(entry)
        self._entries_by_blob_id[blob_id] = entry
        return entry

    def finalize(
        self,
        scrubbed_state: dict[str, Any],
        blob_specs: list[BlobSpec],
        unsupported: list[dict[str, str]],
    ) -> Path:
        """Finish the bundle by writing remaining blobs, manifest, and metadata.

        Parameters
        ----------
        scrubbed_state:
            Portable scrubbed metadata state.
        blob_specs:
            Remaining blob specs that were not already streamed during the pass.
        unsupported:
            Unsupported tensor records for the manifest.

        Returns
        -------
        Path
            Final bundle directory path.

        Raises
        ------
        TorchLensIOError
            If finalization fails.
        """

        self._ensure_writable()
        try:
            for blob_id, tensor, kind, label in blob_specs:
                if blob_id in self._entries_by_blob_id:
                    continue
                self.write_blob(blob_id, tensor, kind=kind, label=label)

            manifest = self._build_manifest(scrubbed_state=scrubbed_state, unsupported=unsupported)
            manifest.write(self.tmp_path / "manifest.json")
            with (self.tmp_path / "metadata.pkl").open("wb") as handle:
                pickle.dump(scrubbed_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except TorchLensIOError:
            raise
        except (OSError, ValueError, pickle.PickleError) as exc:
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            raise TorchLensIOError(reason) from exc

        try:
            self.tmp_path.rename(self.final_path)
        except OSError as exc:
            self._closed = True
            raise TorchLensIOError(
                f"Failed to atomically rename {self.tmp_path} to {self.final_path}."
            ) from exc

        self._closed = True
        self._finalized = True
        return self.final_path

    def abort(self, reason: str) -> None:
        """Mark the temp bundle as partial and stop accepting writes.

        Parameters
        ----------
        reason:
            Human-readable failure reason written to ``REASON.txt``.
        """

        if self._finalized:
            return
        self._closed = True
        self._mark_partial(reason)

    def relabel_blob(self, blob_id: str, label: str) -> None:
        """Update the manifest label for an already-written blob.

        Parameters
        ----------
        blob_id:
            Blob identifier to relabel.
        label:
            Final human-readable label.
        """

        entry = self._entries_by_blob_id.get(blob_id)
        if entry is None:
            return
        updated_entry = TensorEntry(
            blob_id=entry.blob_id,
            kind=entry.kind,
            label=label,
            relative_path=entry.relative_path,
            backend=entry.backend,
            shape=entry.shape,
            dtype=entry.dtype,
            device_at_save=entry.device_at_save,
            layout=entry.layout,
            bytes=entry.bytes,
            sha256=entry.sha256,
        )
        self._entries_by_blob_id[blob_id] = updated_entry
        for index, existing_entry in enumerate(self._tensor_entries):
            if existing_entry.blob_id == blob_id:
                self._tensor_entries[index] = updated_entry
                break

    def get_entry(self, blob_id: str) -> TensorEntry:
        """Return the manifest entry recorded for one blob id.

        Parameters
        ----------
        blob_id:
            Blob identifier to look up.

        Returns
        -------
        TensorEntry
            Recorded manifest entry.

        Raises
        ------
        TorchLensIOError
            If the blob id is unknown.
        """

        if blob_id not in self._entries_by_blob_id:
            raise TorchLensIOError(f"Streaming bundle is missing blob_id={blob_id}.")
        return self._entries_by_blob_id[blob_id]

    def _write_tensor_blob(
        self,
        *,
        blob_id: str,
        tensor: torch.Tensor,
        kind: str,
        label: str,
    ) -> TensorEntry:
        """Write one supported tensor blob and return its manifest entry."""

        with pause_logging():
            contiguous_tensor = tensor.contiguous()
        relative_path = Path("blobs") / f"{blob_id}.safetensors"
        blob_path = self.tmp_path / relative_path
        save_file({_BLOB_TENSOR_KEY: contiguous_tensor}, str(blob_path))
        return TensorEntry(
            blob_id=blob_id,
            kind=kind,
            label=label,
            relative_path=relative_path.as_posix(),
            backend="safetensors",
            shape=[int(dim) for dim in contiguous_tensor.shape],
            dtype=str(contiguous_tensor.dtype).replace("torch.", ""),
            device_at_save=str(tensor.device),
            layout=str(contiguous_tensor.layout).replace("torch.", ""),
            bytes=int(contiguous_tensor.numel() * contiguous_tensor.element_size()),
            sha256=sha256_of_file(blob_path),
        )

    def _build_manifest(
        self,
        *,
        scrubbed_state: dict[str, Any],
        unsupported: list[dict[str, str]],
    ) -> Manifest:
        """Build the final manifest for the streamed bundle."""

        tensor_entries = list(self._tensor_entries)
        n_activation_blobs = sum(1 for entry in tensor_entries if entry.kind == "activation")
        n_gradient_blobs = sum(1 for entry in tensor_entries if entry.kind == "gradient")
        n_auxiliary_blobs = len(tensor_entries) - n_activation_blobs - n_gradient_blobs
        layer_list = scrubbed_state.get("layer_list", [])
        n_layers = len(layer_list) if isinstance(layer_list, list) else 0
        return Manifest(
            io_format_version=IO_FORMAT_VERSION,
            torchlens_version=TORCHLENS_VERSION,
            torch_version=torch.__version__,
            python_version=(
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            platform=f"{platform.system().lower()}-{platform.machine().lower()}",
            created_at=datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            bundle_format="directory",
            n_layers=n_layers,
            n_activation_blobs=n_activation_blobs,
            n_gradient_blobs=n_gradient_blobs,
            n_auxiliary_blobs=n_auxiliary_blobs,
            tensors=tensor_entries,
            unsupported_tensors=unsupported,
        )

    def _ensure_writable(self) -> None:
        """Raise if the writer has already been closed."""

        if self._closed:
            raise TorchLensIOError("Streaming bundle writer is already closed.")

    def _mark_partial(self, reason: str) -> None:
        """Best-effort write the partial sentinel and human-readable reason."""

        try:
            if self.tmp_path.exists():
                (self.tmp_path / PARTIAL_SENTINEL).write_text("", encoding="utf-8")
                (self.tmp_path / REASON_SENTINEL).write_text(reason, encoding="utf-8")
        except OSError:
            return
