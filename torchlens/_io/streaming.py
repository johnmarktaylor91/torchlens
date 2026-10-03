"""Streaming bundle writer used during forward-pass out capture.

This module implements the strict streaming writer behind
``trace(storage=tl.to_disk(...))``. It writes one safetensors blob
per saved out into a temporary bundle during the forward pass, then
finalizes ``manifest.json`` and ``metadata.pkl`` at postprocess time so the
returned log can stay memory-backed or disk-backed with the same on-disk
bundle.
"""

from __future__ import annotations

import io
import os
import pickle
import platform
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from .. import __version__ as TORCHLENS_VERSION
from .._state import pause_logging
from . import TLSPEC_VERSION, TorchLensIOError
from ._canonical_pickle import dump_canonical_metadata
from ._durability import fsync_dir, fsync_tree
from .manifest import Manifest, TensorEntry, sha256_of_file
from .scrub import BlobSpec
from .streaming_async import (
    DEFAULT_MAX_PENDING_BYTES,
    AsyncWriteEngine,
    AsyncWriteFailedError,
    PendingWrite,
)
from .tensor_policy import FailReason, Ok, SkipReason, is_supported_for_save
from .tlspec import _TlSpecWriter

PARTIAL_SENTINEL = "PARTIAL"
REASON_SENTINEL = "REASON.txt"
_BLOB_TENSOR_KEY = "data"


@dataclass(frozen=True)
class _PreparedPayload:
    """One serialization-ready payload plus its origin-tensor manifest facts.

    Parameters
    ----------
    payload:
        Contiguous tensor to serialize (CPU-resident when prepared for the
        async worker).
    device_at_save:
        Device string of the ORIGINAL tensor at submission time.
    requires_grad:
        ``requires_grad`` of the original tensor at submission time.
    """

    payload: torch.Tensor
    device_at_save: str
    requires_grad: bool


def _restrict_mode(path: Path, mode: int) -> None:
    """Best-effort tighten a streamed bundle path's permissions (POSIX only).

    Twin of ``_io/bundle.py::_restrict_mode``: ``mkdir``/``open`` honor the
    ambient umask, so under umask 002 the streaming bundle dir and its metadata
    sidecars were left group-writable while the core bundle writer tightens
    them. Best-effort: a filesystem that ignores mode bits is not a save
    failure.

    Parameters
    ----------
    path:
        Bundle directory or file to tighten.
    mode:
        Target permission bits (``0o700`` for directories, ``0o600`` for files).
    """

    if os.name != "posix":
        return
    try:
        path.chmod(mode)
    except OSError:
        pass


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
    """Persist out blobs incrementally into a temp TorchLens bundle.

    Parameters
    ----------
    path:
        Final bundle directory path.
    strict:
        Streaming bundles are always strict. Passing ``False`` is rejected.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        strict: bool = True,
        include_custom_attributes: bool = True,
        include_buffer_values: bool = True,
    ) -> None:
        """Create the temp bundle directory used for streaming writes.

        Writes are synchronous unless ``arm_async_writes`` is called before
        the first submission.

        Parameters
        ----------
        path:
            Final bundle directory path.
        strict:
            Streaming bundles are always strict. Passing ``False`` is rejected.
        include_custom_attributes:
            Whether harvested module attributes are persisted in the streamed
            bundle (the tl.save opt-out, mirrored for streaming -- R62).
        include_buffer_values:
            Whether captured pre-forward buffer values are persisted in the
            streamed bundle (the tl.save opt-out, mirrored for streaming --
            R62 buffer extension).

        Raises
        ------
        TorchLensIOError
            If the target path is invalid or the temp directory cannot be created.
        """

        if not strict:
            raise TorchLensIOError("Streaming out save is always strict.")

        self.include_custom_attributes = include_custom_attributes
        self.include_buffer_values = include_buffer_values
        self.final_path = Path(path)
        if self.final_path.is_symlink():
            raise TorchLensIOError(f"Refusing symlinked save target: {self.final_path}.")
        if self.final_path.exists():
            raise TorchLensIOError(f"Bundle path already exists: {self.final_path}")

        self.tmp_path = self.final_path.parent / f"{self.final_path.name}.tmp.{uuid.uuid4().hex}"
        self.blobs_path = self.tmp_path / "blobs"
        self._blob_counter = 0
        self._closed = False
        self._finalized = False
        self._tensor_entries: list[TensorEntry] = []
        self._entries_by_blob_id: dict[str, TensorEntry] = {}
        # blob_id -> position in ``_tensor_entries`` (r8 R29): ``relabel_blob``
        # linear-scanned the entry list once per streamed payload, O(B^2)
        # across a streamed save (~1.2e9 compares at 50k saved ops).
        self._tensor_entry_indexes: dict[str, int] = {}
        # Entry structures are worker-mutated when the async engine is active;
        # every read/write of the entry maps and the id reservation set goes
        # through this lock.
        self._entries_lock = threading.Lock()
        self._known_blob_ids: set[str] = set()
        self._async_engine: AsyncWriteEngine | None = None
        # WT1 A-IV item 18 (lane A08): streamed bundles enter settlement. The
        # postprocess step-18 finalize now STAGES the bundle (blobs +
        # manifest, held scrubbed state) and the tmp->final publish happens
        # no earlier than settlement, carrying the settled capture-outcome
        # attestation into metadata.pkl. A capture that settles FAILED never
        # publishes: the pending writer is aborted (PARTIAL debris).
        self._staged_scrubbed_state: dict[str, Any] | None = None

        try:
            self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
            self.tmp_path.mkdir()
            self.blobs_path.mkdir()
            # Permission parity with tl.save (B8-10): mkdir honors the ambient
            # umask, so under umask 002 the streaming bundle dir and its blobs/
            # dir were left group-writable/readable while the core bundle
            # writer tightens them to 0700. The rename at finalize preserves
            # tmp_path's mode, so tightening here also tightens the published
            # bundle directory.
            _restrict_mode(self.tmp_path, 0o700)
            _restrict_mode(self.blobs_path, 0o700)
        except OSError as exc:
            raise TorchLensIOError(
                f"Failed to create streaming temp bundle at {self.tmp_path}."
            ) from exc

    def arm_async_writes(self, max_pending_bytes: int | None = None) -> None:
        """Arm the bounded async write pipeline (DOCUMENTED-UNSTABLE spelling).

        Must run before the first blob submission: ``submit_blob`` then
        overlaps blob writes with capture through a single-worker FIFO
        pipeline (ordering, backpressure, failure latching, and the finalize
        drain barrier are described in ``torchlens/_io/streaming_async.py``).
        Direct ``write_blob`` calls stay synchronous either way.

        Parameters
        ----------
        max_pending_bytes:
            Pending snapshot byte budget for the async pipeline (defaults to
            ``streaming_async.DEFAULT_MAX_PENDING_BYTES``). Submissions block
            once the budget is full, so a slow disk slows capture instead of
            accumulating unbounded RAM.

        Raises
        ------
        TorchLensIOError
            If the writer is closed, already armed, or already wrote blobs.
        """

        self._ensure_writable()
        if self._async_engine is not None:
            raise TorchLensIOError("Async streaming writes are already armed.")
        with self._entries_lock:
            already_submitted = bool(self._known_blob_ids)
        if already_submitted:
            raise TorchLensIOError(
                "Cannot arm async streaming writes after blobs were already submitted."
            )
        self._async_engine = AsyncWriteEngine(
            max_pending_bytes=(
                DEFAULT_MAX_PENDING_BYTES if max_pending_bytes is None else max_pending_bytes
            )
        )

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

        self._validate_blob_submission(blob_id, tensor, kind=kind, label=label)

        try:
            entry = self._write_tensor_blob(blob_id=blob_id, tensor=tensor, kind=kind, label=label)
        except OSError as exc:
            reason = f"Failed to write streaming blob_id={blob_id} for {label}: {exc}"
            self.abort(reason)
            raise TorchLensIOError(reason) from exc
        except BaseException as exc:
            # Safety-net catch-all mirroring bundle.py's ``save()`` handler
            # (round-8 F3): a hand-enumerated except clause can always miss
            # the next not-yet-seen failure shape (e.g. a bare ``KeyError``
            # from ``safetensors.torch.save_file()`` for an allow-listed-but-
            # actually-unwritable dtype, cert round 8 BLOCKER) or a
            # KeyboardInterrupt/SystemExit/GeneratorExit unwinding mid-write.
            # This guarantees the ``.tmp`` dir is always marked PARTIAL --
            # and thus sweepable by ``cleanup_tmp()`` -- for any failure,
            # known or not, while re-raising non-``Exception``
            # ``BaseException``s unwrapped so control-flow semantics are
            # preserved.
            reason = f"Failed to write streaming blob_id={blob_id} for {label}: {exc}"
            self.abort(reason)
            if isinstance(exc, Exception):
                raise TorchLensIOError(reason) from exc
            raise

        self._record_entry(entry)
        return entry

    def submit_blob(
        self,
        blob_id: str,
        tensor: torch.Tensor,
        *,
        kind: str,
        label: str,
    ) -> None:
        """Persist one tensor blob, overlapping the write with capture when armed.

        The capture-time spelling of ``write_blob``: with the async engine
        active the payload is snapshotted on THIS thread (value-at-call-time
        semantics survive later in-place mutation of the source tensor) and
        the serialize + write + sha256 work runs on the single worker thread,
        FIFO, under the pending-bytes budget. Without the engine this is
        exactly ``write_blob``. A failed queued write latches and re-raises
        as ``TorchLensIOError`` at the next writer interaction; ``finalize``
        drains every pending write before the bundle can publish.

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

        Raises
        ------
        TorchLensIOError
            If the tensor is unsupported, snapshotting fails, or an earlier
            queued write failed.
        """

        engine = self._async_engine
        if engine is None:
            self.write_blob(blob_id, tensor, kind=kind, label=label)
            return

        self._validate_blob_submission(blob_id, tensor, kind=kind, label=label)
        try:
            prepared = self._prepare_deferred_snapshot(tensor)
        except BaseException as exc:
            # Same safety-net contract as write_blob (round-8 F3): any
            # snapshot failure marks the temp dir PARTIAL and re-raises
            # non-Exception BaseExceptions unwrapped.
            reason = f"Failed to snapshot streaming blob_id={blob_id} for {label}: {exc}"
            self.abort(reason)
            if isinstance(exc, Exception):
                raise TorchLensIOError(reason) from exc
            raise

        def _job() -> None:
            """Write the prepared snapshot and record its entry (worker thread)."""

            entry = self._write_prepared_blob(
                blob_id=blob_id,
                prepared=prepared,
                kind=kind,
                label=label,
            )
            self._record_entry(entry)

        try:
            engine.submit(
                PendingWrite(
                    blob_id=blob_id,
                    nbytes=int(prepared.payload.numel() * prepared.payload.element_size()),
                    job=_job,
                )
            )
        except AsyncWriteFailedError as exc:
            reason = str(exc)
            self.abort(reason)
            raise TorchLensIOError(reason) from (exc.__cause__ or exc)

    def finalize(
        self,
        scrubbed_state: dict[str, Any],
        blob_specs: list[BlobSpec],
        unsupported: list[dict[str, str]],
        *,
        trace: Any,
    ) -> Path:
        """Finish the bundle: stage everything, then publish immediately.

        Post-settlement callers only (the deferred grad-streaming tail): the
        settled capture-outcome attestation is read off ``trace`` and
        persisted. Pre-settlement finalization goes through
        :meth:`stage_for_settlement` + :meth:`publish_staged` so the
        tmp->final publish happens no earlier than settlement (WT1 A-IV
        item 18).

        Parameters
        ----------
        scrubbed_state:
            Portable scrubbed metadata state.
        blob_specs:
            Remaining blob specs that were not already streamed during the pass.
        unsupported:
            Unsupported tensor records for the manifest.
        trace:
            Source ``Trace`` being streamed to disk.

        Returns
        -------
        Path
            Final bundle directory path.

        Raises
        ------
        TorchLensIOError
            If finalization fails.
        """

        from ..capture.outcome import outcome_for

        self.stage_for_settlement(scrubbed_state, blob_specs, unsupported, trace=trace)
        settled = outcome_for(trace)
        return self.publish_staged(None if settled is None else settled.to_payload())

    def stage_for_settlement(
        self,
        scrubbed_state: dict[str, Any],
        blob_specs: list[BlobSpec],
        unsupported: list[dict[str, str]],
        *,
        trace: Any,
    ) -> Path:
        """Write remaining blobs and the manifest; hold the state for publish.

        The bundle stays in its temp directory: :meth:`publish_staged` writes
        ``metadata.pkl`` (with the settled capture-outcome attestation
        injected), fsyncs, and renames. A capture failure between staging and
        settlement aborts the writer instead, so a FAILED capture never
        leaves a publishable artifact (WT1 A-IV item 18).

        Parameters
        ----------
        scrubbed_state:
            Portable scrubbed metadata state.
        blob_specs:
            Remaining blob specs that were not already streamed during the pass.
        unsupported:
            Unsupported tensor records for the manifest.
        trace:
            Source ``Trace`` being streamed to disk. Used to write the same
            unified ``.tlspec`` manifest fields (``kind``, ``model_signature``,
            ``sites``, ``body_index``, ...) that ``Trace.save()``/``tl.save()``
            write, so streaming bundles are detected as ``"v2.0_unified"`` and
            go through the same ``validate_tlspec()`` schema validation.

        Returns
        -------
        Path
            The FUTURE final bundle directory path (not yet published).

        Raises
        ------
        TorchLensIOError
            If staging fails; the temp bundle is marked PARTIAL.
        """

        self._ensure_writable()
        self._settle_async_engine()
        try:
            for blob_id, tensor, kind, label in blob_specs:
                if blob_id in self._entries_by_blob_id:
                    continue
                self.write_blob(blob_id, tensor, kind=kind, label=label)

            legacy_manifest = self._build_manifest(
                scrubbed_state=scrubbed_state, unsupported=unsupported, trace=trace
            )
            _TlSpecWriter.write_trace_manifest(
                path=self.tmp_path / "manifest.json",
                trace=trace,
                legacy_manifest=legacy_manifest,
                save_level="portable",
            )
            _restrict_mode(self.tmp_path / "manifest.json", 0o600)
        except TorchLensIOError:
            raise
        except (OSError, TypeError, ValueError, pickle.PickleError) as exc:
            # See torchlens/_io/bundle.py's ``save()`` handler: ``TypeError``
            # is included alongside ``pickle.PickleError`` because
            # ``pickle.dump()`` raises a bare ``TypeError`` (not the
            # ``PickleError`` subclass) for many live-resource objects.
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            raise TorchLensIOError(reason) from exc
        except BaseException as exc:
            # Safety-net catch-all closing the same bug class as
            # bundle.py's ``save()`` (round-8 F3): a hand-enumerated except
            # tuple can always miss the next not-yet-discovered exception
            # shape, or a KeyboardInterrupt/SystemExit/GeneratorExit
            # unwinding mid-finalize.
            # Guarantees the ``.tmp`` dir is always marked PARTIAL -- and
            # thus sweepable by ``cleanup_tmp()`` -- for any failure, while
            # re-raising non-``Exception`` ``BaseException``s unwrapped so
            # KeyboardInterrupt/SystemExit/GeneratorExit control flow is
            # preserved.
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            if isinstance(exc, Exception):
                raise TorchLensIOError(reason) from exc
            raise
        self._staged_scrubbed_state = scrubbed_state
        return self.final_path

    @property
    def staged_for_settlement(self) -> bool:
        """Whether the bundle is staged and awaiting its settlement publish."""

        return self._staged_scrubbed_state is not None and not self._finalized

    def _write_staged_metadata(
        self,
        scrubbed_state: dict[str, Any],
        outcome_payload: dict[str, Any] | None,
    ) -> None:
        """Write ``metadata.pkl`` with the settled outcome attestation injected."""

        try:
            scrubbed_state["_capture_outcome"] = outcome_payload
            from ._portability_preflight import preflight_metadata_portability
            from .bundle import _RenameAwareUnpickler

            # B3R4-R21-2: canonical container bytes (set/frozenset members
            # sorted); persisted metadata must not vary with PYTHONHASHSEED.
            # Dumped once through the ``dump_canonical_metadata`` seam.
            metadata_buffer = io.BytesIO()
            dump_canonical_metadata(scrubbed_state, metadata_buffer)
            metadata_bytes = metadata_buffer.getvalue()
            # Write/read symmetry (AUD-CODE 2.20): dry-run exactly these bytes
            # through the loader's restricted unpickler before writing.
            preflight_metadata_portability(
                scrubbed_state,
                unpickler_factory=_RenameAwareUnpickler,
                bundle_path=self.tmp_path,
                data=metadata_bytes,
            )
            with (self.tmp_path / "metadata.pkl").open("wb") as handle:
                handle.write(metadata_bytes)
            _restrict_mode(self.tmp_path / "metadata.pkl", 0o600)
        except TorchLensIOError:
            raise
        except (OSError, TypeError, ValueError, pickle.PickleError) as exc:
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            raise TorchLensIOError(reason) from exc
        except BaseException as exc:
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            if isinstance(exc, Exception):
                raise TorchLensIOError(reason) from exc
            raise

    def publish_staged(self, outcome_payload: dict[str, Any] | None) -> Path:
        """Publish the staged bundle with its settled capture-outcome attestation.

        Parameters
        ----------
        outcome_payload:
            String-only settled-outcome payload (``CaptureOutcome.to_payload``)
            injected into the persisted state as ``_capture_outcome`` -- the
            SAME key and codec ordinary ``tl.save()`` bundles persist, so
            loads adopt it through the identical parse + coherence matrix.
            ``None`` persists no attestation (loads derive structurally).

        Returns
        -------
        Path
            Final bundle directory path.

        Raises
        ------
        TorchLensIOError
            If nothing is staged or the publish fails; failures mark the temp
            bundle PARTIAL.
        """

        scrubbed_state = self._staged_scrubbed_state
        if scrubbed_state is None:
            raise TorchLensIOError(
                "Streaming publish requires a staged bundle; call stage_for_settlement() first."
            )
        if self._finalized:
            raise TorchLensIOError(f"Streaming bundle already published: {self.final_path}")
        if self._closed:
            # An abort between staging and settlement already marked the temp
            # bundle PARTIAL; publishing it would rename failure debris into a
            # final-named artifact.
            raise TorchLensIOError(
                f"Streaming bundle at {self.tmp_path} was aborted; refusing to publish it."
            )
        try:
            self._write_staged_metadata(scrubbed_state, outcome_payload)
        finally:
            self._staged_scrubbed_state = None

        # Crash-durability before publish: fsync every written blob/sidecar and
        # the staged directories so a power/OS crash after the rename below
        # cannot publish a final-named bundle holding zero-length/partial files
        # with no PARTIAL sentinel (cleanup_tmp would never sweep it). Mirrors
        # the tl.save writer (_io/bundle.py:568-586).
        try:
            fsync_tree(self.tmp_path)
        except OSError as exc:
            reason = f"Failed to flush streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            raise TorchLensIOError(reason) from exc

        # Re-check target absence at finalize, not just at __init__ (R59 TOCTOU):
        # a streaming writer never overwrites, but a concurrent writer could have
        # created ``final_path`` after the start-of-stream check. A bare rename
        # would then replace an empty concurrent target or surface a confusing
        # ENOTEMPTY; refuse it cleanly instead. (The narrow residual window
        # between this check and the rename cannot be closed without an atomic
        # exclusive-directory create, matching the other writers.)
        if self.final_path.exists():
            reason = f"Bundle path already exists: {self.final_path}"
            self.abort(reason)
            raise TorchLensIOError(reason)
        try:
            self.tmp_path.rename(self.final_path)
        except OSError as exc:
            reason = f"Failed to atomically rename {self.tmp_path} to {self.final_path}."
            self.abort(reason)
            raise TorchLensIOError(reason) from exc
        # Make the rename itself durable before declaring the save complete.
        fsync_dir(self.final_path.parent)

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
        engine = self._async_engine
        if engine is not None:
            # Queued-but-unwritten blobs are pointless in an aborted bundle;
            # the bounded join means a wedged disk cannot hang the unwind.
            self._async_engine = None
            engine.shutdown(discard=True)
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

        self._drain_async()
        with self._entries_lock:
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
            requires_grad=entry.requires_grad,
            logical_backend=entry.logical_backend,
            codec=entry.codec,
            logical_dtype=entry.logical_dtype,
            logical_device=entry.logical_device,
            transport_backend=entry.transport_backend,
            transport_dtype=entry.transport_dtype,
            codec_metadata=entry.codec_metadata,
        )
        with self._entries_lock:
            self._entries_by_blob_id[blob_id] = updated_entry
            entry_index = self._tensor_entry_indexes.get(blob_id)
            if entry_index is not None:
                self._tensor_entries[entry_index] = updated_entry

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

        self._drain_async()
        with self._entries_lock:
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
            contiguous_tensor = tensor.resolve_conj().resolve_neg().contiguous()
            if contiguous_tensor.device.type != "cpu":
                contiguous_tensor = contiguous_tensor.cpu()
        return self._write_prepared_blob(
            blob_id=blob_id,
            prepared=_PreparedPayload(
                payload=contiguous_tensor,
                device_at_save=str(tensor.device),
                requires_grad=bool(tensor.requires_grad),
            ),
            kind=kind,
            label=label,
        )

    def _write_prepared_blob(
        self,
        *,
        blob_id: str,
        prepared: _PreparedPayload,
        kind: str,
        label: str,
    ) -> TensorEntry:
        """Serialize one prepared payload to ``blobs/`` and build its entry.

        Worker-thread safe by construction: a CPU-contiguous payload reaches
        safetensors' raw-pointer serialization without invoking any wrapped
        torch function, and the sha256 hash is pure hashlib.

        Parameters
        ----------
        blob_id:
            Opaque zero-padded blob identifier.
        prepared:
            Serialization-ready payload plus its origin-tensor facts.
        kind:
            Logical tensor kind.
        label:
            Human-readable or provisional label for the tensor owner.

        Returns
        -------
        TensorEntry
            Recorded manifest entry.
        """

        payload = prepared.payload
        relative_path = Path("blobs") / f"{blob_id}.safetensors"
        blob_path = self.tmp_path / relative_path
        save_file({_BLOB_TENSOR_KEY: payload}, str(blob_path))
        return TensorEntry(
            blob_id=blob_id,
            kind=kind,
            label=label,
            relative_path=relative_path.as_posix(),
            backend="safetensors",
            shape=[int(dim) for dim in payload.shape],
            dtype=str(payload.dtype).replace("torch.", ""),
            device_at_save=prepared.device_at_save,
            layout=str(payload.layout).replace("torch.", ""),
            bytes=int(payload.numel() * payload.element_size()),
            sha256=sha256_of_file(blob_path),
            requires_grad=prepared.requires_grad,
        )

    def _prepare_deferred_snapshot(self, tensor: torch.Tensor) -> _PreparedPayload:
        """Return a private CPU-contiguous byte snapshot of one payload.

        Runs on the SUBMITTING thread under ``pause_logging`` so the deferred
        write preserves the sync path's value-at-call-time semantics: a later
        in-place mutation of the source tensor (or of storage it aliases)
        must never reach the artifact. Whenever the resolve/contiguous chain
        made no copy, the result still aliases the caller's storage and is
        cloned; non-CPU payloads move to CPU here so the worker never runs a
        wrapped torch op.

        Parameters
        ----------
        tensor:
            Tensor payload to snapshot.

        Returns
        -------
        _PreparedPayload
            CPU-contiguous snapshot owning storage no caller can mutate,
            plus the origin tensor's manifest facts.
        """

        with pause_logging():
            snapshot = tensor.resolve_conj().resolve_neg().contiguous()
            if snapshot.device.type != "cpu":
                snapshot = snapshot.cpu()
            elif snapshot is tensor:
                snapshot = snapshot.clone()
        return _PreparedPayload(
            payload=snapshot,
            device_at_save=str(tensor.device),
            requires_grad=bool(tensor.requires_grad),
        )

    def _validate_blob_submission(
        self,
        blob_id: str,
        tensor: torch.Tensor,
        *,
        kind: str,
        label: str,
    ) -> None:
        """Run the synchronous admission checks shared by both write paths.

        Aborts the bundle and raises on an unwritable submission, and
        reserves ``blob_id`` so a duplicate id refuses even while the first
        write is still queued on the async worker.

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

        Raises
        ------
        TorchLensIOError
            If the writer is closed, the payload is not a supported tensor,
            or the blob id was already submitted.
        """

        self._ensure_writable()
        if not isinstance(tensor, torch.Tensor):
            transform_name = "grad_transform" if kind == "grad" else "activation_transform"
            reason = (
                f"Streaming {kind} save requires {transform_name} outputs to be torch.Tensor "
                f"instances, but blob_id={blob_id} ({label}) received {type(tensor).__name__}."
            )
            self.abort(reason)
            raise TorchLensIOError(reason)

        decision = is_supported_for_save(tensor, strict=True)
        if not isinstance(decision, Ok):
            if isinstance(decision, (SkipReason, FailReason)):
                reason_text = decision.text
            else:
                reason_text = "unsupported tensor"
            reason = (
                f"Unsupported tensor for streaming {kind} save at {label} "
                f"(blob_id={blob_id}, kind={kind}): {reason_text}"
            )
            self.abort(reason)
            raise TorchLensIOError(reason)
        with self._entries_lock:
            duplicate = blob_id in self._known_blob_ids
            if not duplicate:
                self._known_blob_ids.add(blob_id)
        if duplicate:
            reason = f"Duplicate streaming blob_id={blob_id} for {label}."
            self.abort(reason)
            raise TorchLensIOError(reason)

    def _record_entry(self, entry: TensorEntry) -> None:
        """Append one completed manifest entry (worker- or caller-thread)."""

        with self._entries_lock:
            self._tensor_entry_indexes[entry.blob_id] = len(self._tensor_entries)
            self._tensor_entries.append(entry)
            self._entries_by_blob_id[entry.blob_id] = entry

    def _drain_async(self) -> None:
        """Barrier: wait for every queued write; convert a latched failure.

        Raises
        ------
        TorchLensIOError
            If any queued write failed. The bundle is marked PARTIAL first.
        """

        engine = self._async_engine
        if engine is None:
            return
        try:
            engine.drain()
        except AsyncWriteFailedError as exc:
            reason = str(exc)
            self.abort(reason)
            raise TorchLensIOError(reason) from (exc.__cause__ or exc)

    def _settle_async_engine(self) -> None:
        """FINALIZATION barrier: land every capture-time write, retire the engine.

        Every accepted write must land before the manifest is built and the
        bundle can publish; a latched write failure aborts here, typed, so a
        bundle with a silently missing blob can never be presented as
        complete. The engine then retires so finalize's remaining blob specs
        take the ordinary synchronous path.

        Raises
        ------
        TorchLensIOError
            If any queued write failed. The bundle is marked PARTIAL first.
        """

        self._drain_async()
        engine = self._async_engine
        if engine is not None:
            engine.shutdown(discard=False)
            self._async_engine = None

    def _build_manifest(
        self,
        *,
        scrubbed_state: dict[str, Any],
        unsupported: list[dict[str, str]],
        trace: Any,
    ) -> Manifest:
        """Build the final manifest for the streamed bundle."""

        tensor_entries = list(self._tensor_entries)
        n_out_blobs = sum(1 for entry in tensor_entries if entry.kind == "out")
        n_grad_blobs = sum(1 for entry in tensor_entries if entry.kind == "grad")
        n_auxiliary_blobs = len(tensor_entries) - n_out_blobs - n_grad_blobs
        layer_list = scrubbed_state.get("layer_list", [])
        n_layers = len(layer_list) if isinstance(layer_list, list) else 0
        # Disclose the harvested module-attribute channel (R62): the documented
        # invariant is that EVERY save writes a custom_attributes_disclosure
        # entry, and the streaming path now honors the same opt-out and emits
        # the same embedding warning as tl.save (the reopened hf_token class:
        # a canary token used to ship in the streamed bundle with zero
        # warnings and no way to withhold it).
        from .bundle import (
            _buffer_values_disclosure,
            _custom_attributes_disclosure,
            _warn_buffer_value_embedding,
            _warn_custom_attribute_embedding,
        )

        disclosure = _custom_attributes_disclosure(trace, included=self.include_custom_attributes)
        _warn_custom_attribute_embedding(disclosure)
        buffer_disclosure = _buffer_values_disclosure(trace, included=self.include_buffer_values)
        _warn_buffer_value_embedding(buffer_disclosure)

        return Manifest(
            tlspec_version=TLSPEC_VERSION,
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
            n_out_blobs=n_out_blobs,
            n_grad_blobs=n_grad_blobs,
            n_auxiliary_blobs=n_auxiliary_blobs,
            tensors=tensor_entries,
            unsupported_tensors=unsupported,
            custom_attributes_disclosure=disclosure,
            buffer_values_disclosure=buffer_disclosure,
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
