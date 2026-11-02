"""Streaming bundle writer used during forward-pass out capture.

This module implements the strict streaming writer behind
``trace(storage=tl.to_disk(...))``. During the forward pass every saved
payload is queued as one opaque key (``t<k>``) into the open chunk
``blobs/<codec>-<nnnn>.safetensors`` of a temporary bundle; a chunk is written
once it reaches ``format3.payloads.PAYLOAD_CHUNK_BYTES``. Staging closes the
chunks and digests them; publishing writes the 3.0 envelope (typed JSON record
tables plus ``manifest.json``) through ``format3.writer``, reloads it once, and
renames the bundle into place, so the returned log can stay memory-backed or
disk-backed with the same on-disk bundle. Relabeling a streamed payload edits
its manifest entry only: the chunk key is opaque and never rewritten.
"""

from __future__ import annotations

import dataclasses
import platform
import threading
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .. import __version__ as TORCHLENS_VERSION
from .._state import pause_logging
from .._utils.hashing import sha256_file
from . import TLSPEC_VERSION, ArtifactError, refuse_bundle_exists
from ._durability import _restrict_mode, fsync_dir, fsync_tree
from .format3.payloads import TORCH_CODEC, PayloadStore, host_tensor, tensor_reference
from .format3.writer import EnvelopeSpec, TraceTables, preflight_reload, write_trace_envelope
from .manifest import Manifest, TensorEntry, _runtime_python_version
from .paths import _make_tmp_bundle_path, reject_symlink_path
from .scrub import BlobSpec
from .streaming_async import (
    DEFAULT_MAX_PENDING_BYTES,
    AsyncWriteEngine,
    AsyncWriteFailedError,
    PendingWrite,
    _PreparedPayload,
)
from .tensor_policy import FailReason, Ok, SkipReason, is_supported_for_save
from .tlspec import _TlSpecWriter

PARTIAL_SENTINEL = "PARTIAL"
REASON_SENTINEL = "REASON.txt"


def _weak_reference(value: Any) -> Any:
    """Return a weak reference to ``value``, or ``None`` when it cannot hold one."""

    try:
        return weakref.ref(value)
    except TypeError:
        return None


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
        payload_per_file: bool = False,
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
        include_custom_attributes, include_buffer_values:
            Whether harvested module attributes and captured pre-forward buffer
            values are persisted (the Trace.save opt-outs, mirrored -- R62).
        payload_per_file:
            Close and digest each payload's chunk as it is written, for readers
            that consume entries synchronously (the fastlog recorder).

        Raises
        ------
        ArtifactError
            If the target path is invalid or the temp directory cannot be created.
        """

        if not strict:
            raise ArtifactError("Streaming out save is always strict.")

        self.include_custom_attributes = include_custom_attributes
        self.include_buffer_values = include_buffer_values
        self._payload_per_file = payload_per_file
        self.final_path = Path(path)
        reject_symlink_path(self.final_path, context="save target")
        if self.final_path.exists():
            refuse_bundle_exists(self.final_path)

        self.tmp_path = _make_tmp_bundle_path(self.final_path)
        self.blobs_path = self.tmp_path / "blobs"
        self._blob_counter = 0
        self._closed = False
        self._finalized = False
        self._tensor_entries: list[TensorEntry] = []
        self._entries_by_blob_id: dict[str, TensorEntry] = {}
        self._tensor_entry_indexes: dict[str, int] = {}  # r8 R29: no O(B^2) relabel scan
        # Entry maps, tensor references and the id reservation set are
        # worker-mutated when the async engine is active: guarded by this lock.
        self._entries_lock = threading.Lock()
        self._known_blob_ids: set[str] = set()
        self._tensor_refs: dict[str, dict[str, Any]] = {}
        # One chunk store per bundle (one key counter); its own lock because
        # direct ``write_blob`` calls stay synchronous while the worker runs.
        self._payload_store = PayloadStore(self.tmp_path)
        self._store_lock = threading.Lock()
        self._async_engine: AsyncWriteEngine | None = None
        # WT1 A-IV item 18: finalize STAGES the bundle (chunks closed, manifest
        # base built, scrubbed state held) and the publish, which writes the
        # record tables with the settled capture outcome, waits for settlement.
        # A capture that settles FAILED never publishes (PARTIAL debris).
        self._staged_scrubbed_state: dict[str, Any] | None = None
        self._staged_base_manifest: dict[str, Any] = {}
        self._staged_payload_policy: dict[str, bool] = {}
        self._staged_trace: Any = None

        try:
            self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
            self.tmp_path.mkdir()
            self.blobs_path.mkdir()
            # Permission parity with Trace.save (B8-10): tighten against the
            # ambient umask; the publish rename preserves the mode.
            _restrict_mode(self.tmp_path, 0o700)
            _restrict_mode(self.blobs_path, 0o700)
        except OSError as exc:
            raise ArtifactError(
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
        ArtifactError
            If the writer is closed, already armed, or already wrote blobs.
        """

        self._ensure_writable()
        if self._async_engine is not None:
            raise ArtifactError("Async streaming writes are already armed.")
        with self._entries_lock:
            already_submitted = bool(self._known_blob_ids)
        if already_submitted:
            raise ArtifactError(
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

    def _issued_blob_count(self) -> int:
        """Return how many ids :meth:`next_blob_id` issued; save-time scrub ids start past it."""

        return self._blob_counter

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
        ArtifactError
            If the tensor is unsupported or writing fails.
        """

        self._validate_blob_submission(blob_id, tensor, kind=kind, label=label)

        try:
            entry = self._write_tensor_blob(blob_id=blob_id, tensor=tensor, kind=kind, label=label)
        except OSError as exc:
            reason = f"Failed to write streaming blob_id={blob_id} for {label}: {exc}"
            self.abort(reason)
            raise ArtifactError(reason) from exc
        except BaseException as exc:
            # Safety net (round-8 F3): any other failure (a bare KeyError from
            # safetensors, a SystemExit mid-write) marks the ``.tmp`` dir PARTIAL;
            # non-Exception BaseExceptions re-raise unwrapped.
            reason = f"Failed to write streaming blob_id={blob_id} for {label}: {exc}"
            self.abort(reason)
            if isinstance(exc, Exception):
                raise ArtifactError(reason) from exc
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
        as ``ArtifactError`` at the next writer interaction; ``finalize``
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
        ArtifactError
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
                raise ArtifactError(reason) from exc
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
            raise ArtifactError(reason) from (exc.__cause__ or exc)

    def stage_for_settlement(
        self,
        scrubbed_state: dict[str, Any],
        blob_specs: list[BlobSpec],
        unsupported: list[dict[str, str]],
        *,
        trace: Any,
        payload_policy: dict[str, bool] | None = None,
        record_pairs: list[tuple[Any, Any]] | None = None,
    ) -> Path:
        """Write remaining payloads, close the chunks, build the manifest; hold the state.

        The bundle stays in its temp directory: :meth:`publish_staged` writes
        the record tables (with the settled capture-outcome attestation
        injected) and the 3.0 manifest, reloads them once, fsyncs, and
        renames. A capture failure between staging and settlement aborts the
        writer instead, so a FAILED capture never leaves a publishable
        artifact (WT1 A-IV item 18). Until the publish, ``manifest.json``
        holds the manifest base (no record tables yet).

        Parameters
        ----------
        scrubbed_state:
            Portable scrubbed metadata state.
        blob_specs:
            Remaining blob specs that were not already streamed during the pass.
        unsupported:
            Unsupported tensor records for the manifest.
        trace:
            Source ``Trace`` being streamed to disk; feeds the same unified
            manifest fields (``kind``, ``model_signature``, ``sites``, ...)
            that ``trace.save()`` writes.
        payload_policy:
            Include flags the scrub used (``include_outs``, ``include_grads``,
            ...); ``None`` derives them from the written payloads and the
            scrubbed options.
        record_pairs:
            ``(scrubbed copy, live record)`` pairs from the scrub (live cells).

        Returns
        -------
        Path
            The FUTURE final bundle directory path (not yet published).

        Raises
        ------
        ArtifactError
            If staging fails; the temp bundle is marked PARTIAL.
        """

        self._ensure_writable()
        self._settle_async_engine()
        try:
            for blob_id, tensor, kind, label in blob_specs:
                if blob_id in self._entries_by_blob_id:
                    # Reused blobs were filtered and scrub ids start past the stream's, so
                    # an already-written id would resolve to ANOTHER op's payload.
                    existing = self._entries_by_blob_id[blob_id]
                    raise ArtifactError(
                        f"Streaming bundle blob id {blob_id} for {label!r} collides with "
                        f"the blob already streamed for {existing.label!r}."
                    )
                self.write_blob(blob_id, tensor, kind=kind, label=label)
            self._seal_payload_chunks()
            self._staged_payload_policy = self._resolve_payload_policy(
                scrubbed_state, payload_policy
            )
            self._write_manifest(scrubbed_state, unsupported, trace)
        except ArtifactError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            raise ArtifactError(reason) from exc
        except BaseException as exc:
            # Safety net (round-8 F3): any failure marks the ``.tmp`` dir PARTIAL.
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            if isinstance(exc, Exception):
                raise ArtifactError(reason) from exc
            raise
        self._staged_record_pairs = [*(record_pairs or ()), (scrubbed_state, trace)]
        self._staged_scrubbed_state = scrubbed_state
        self._staged_trace = _weak_reference(trace)
        return self.final_path

    def _resolve_payload_policy(
        self, scrubbed_state: dict[str, Any], payload_policy: dict[str, bool] | None
    ) -> dict[str, bool]:
        """Return the manifest payload policy of this bundle (contract flag set)."""

        if payload_policy is None:
            with self._entries_lock:
                has_grads = any(entry.kind == "grad" for entry in self._tensor_entries)
            payload_policy = {
                "include_grads": has_grads,
                "include_saved_args": scrubbed_state.get("_options_save_args") is True,
                "include_rng_states": scrubbed_state.get("_options_save_rng") is True,
            }
        flags = ("include_grads", "include_saved_args", "include_rng_states")
        return {
            "include_outs": bool(payload_policy.get("include_outs", True)),
            **{flag: bool(payload_policy.get(flag, False)) for flag in flags},
            "include_buffer_values": bool(self.include_buffer_values),
            "include_custom_attributes": bool(self.include_custom_attributes),
            "structure_only": scrubbed_state.get("_options_structure_only") is True,
        }

    @property
    def staged_for_settlement(self) -> bool:
        """Whether the bundle is staged and awaiting its settlement publish."""

        return self._staged_scrubbed_state is not None and not self._finalized

    def _write_staged_metadata(
        self,
        scrubbed_state: dict[str, Any],
        outcome_payload: dict[str, Any] | None,
    ) -> None:
        """Write the record tables and 3.0 manifest with the settled outcome; reload once."""

        try:
            scrubbed_state["_capture_outcome"] = outcome_payload
            policy, codecs = self._staged_payload_policy, self._payload_store.codecs
            write_trace_envelope(
                self.tmp_path,
                base_manifest=self._staged_base_manifest,
                tables=TraceTables(scrubbed_state, self._tensor_refs, self._staged_record_pairs),
                envelope=EnvelopeSpec("trace", "portable", payload_policy=policy, codecs=codecs),
            )
            for path in sorted(self.tmp_path.rglob("*")):
                _restrict_mode(path, 0o700 if path.is_dir() else 0o600)
            preflight_reload(self.tmp_path)  # late.21: one JSON parse plus the typed decode
        except ArtifactError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            raise ArtifactError(reason) from exc
        except BaseException as exc:
            reason = f"Failed to finalize streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            if isinstance(exc, Exception):
                raise ArtifactError(reason) from exc
            raise

    def _refresh_trace_manifest_digest(self) -> None:
        """Point the staged trace's manifest digest at the published manifest.

        Staging may hand the trace a digest of the manifest base before a
        deferred publish rewrites ``manifest.json`` with the record tables.
        """

        trace = self._staged_trace() if self._staged_trace is not None else None
        self._staged_trace = None
        if trace is None or getattr(trace, "_source_bundle_path", None) != self.final_path:
            return
        setattr(
            trace,
            "_source_bundle_manifest_sha256",
            sha256_file(self.final_path / "manifest.json"),
        )

    def publish_staged(self, outcome_payload: dict[str, Any] | None) -> Path:
        """Publish the staged bundle with its settled capture-outcome attestation.

        Parameters
        ----------
        outcome_payload:
            String-only settled-outcome payload (``CaptureOutcome.to_payload``)
            injected into the persisted state as ``_capture_outcome`` -- the
            SAME key and codec ordinary ``trace.save()`` bundles persist, so
            loads adopt it through the identical parse + coherence matrix.
            ``None`` persists no attestation (loads derive structurally).

        Returns
        -------
        Path
            Final bundle directory path.

        Raises
        ------
        ArtifactError
            If nothing is staged or the publish fails; failures mark the temp
            bundle PARTIAL.
        """

        scrubbed_state = self._staged_scrubbed_state
        if scrubbed_state is None:
            raise ArtifactError(
                "Streaming publish requires a staged bundle; call stage_for_settlement() first."
            )
        if self._finalized:
            raise ArtifactError(f"Streaming bundle already published: {self.final_path}")
        if self._closed:
            # An abort between staging and settlement already marked the temp
            # bundle PARTIAL; publishing it would rename failure debris into a
            # final-named artifact.
            raise ArtifactError(
                f"Streaming bundle at {self.tmp_path} was aborted; refusing to publish it."
            )
        try:
            self._write_staged_metadata(scrubbed_state, outcome_payload)
        finally:
            self._staged_scrubbed_state, self._staged_record_pairs = None, []

        # Crash-durability before publish: fsync every written file and the
        # staged directories so a crash after the rename cannot publish a
        # final-named bundle holding partial files with no PARTIAL sentinel.
        try:
            fsync_tree(self.tmp_path)
        except OSError as exc:
            reason = f"Failed to flush streaming bundle at {self.tmp_path}: {exc}"
            self.abort(reason)
            raise ArtifactError(reason) from exc

        # Re-check target absence at publish (R59 TOCTOU): a concurrent writer
        # could have created ``final_path`` after the start-of-stream check.
        if self.final_path.exists():
            reason = f"Bundle path already exists: {self.final_path}"
            self.abort(reason)
            refuse_bundle_exists(self.final_path)
        try:
            self.tmp_path.rename(self.final_path)
        except OSError as exc:
            reason = f"Failed to atomically rename {self.tmp_path} to {self.final_path}."
            self.abort(reason)
            raise ArtifactError(reason) from exc
        # Make the rename itself durable before declaring the save complete.
        fsync_dir(self.final_path.parent)

        self._closed = True
        self._finalized = True
        self._refresh_trace_manifest_digest()
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

        The chunk key is opaque, so a relabel edits the entry only; the
        record cell that names the payload carries the final label.

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
        updated_entry = dataclasses.replace(entry, label=label)
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
        ArtifactError
            If the blob id is unknown.
        """

        self._drain_async()
        with self._entries_lock:
            if blob_id not in self._entries_by_blob_id:
                raise ArtifactError(f"Streaming bundle is missing blob_id={blob_id}.")
            return self._entries_by_blob_id[blob_id]

    def get_tensor_reference(self, blob_id: str) -> dict[str, Any]:
        """Return the tensor reference fields (chunk ``blob``, ``key``, ...) of one blob id.

        Parameters
        ----------
        blob_id:
            Blob identifier to look up.

        Returns
        -------
        dict[str, Any]
            A copy of the reference the record tables carry for this payload.

        Raises
        ------
        ArtifactError
            If the blob id is unknown (raised by :meth:`get_entry`).
        """

        self.get_entry(blob_id)
        with self._entries_lock:
            return dict(self._tensor_refs[blob_id])

    def _seal_payload_chunks(self) -> None:
        """Close every open chunk and stamp each entry with its chunk's digest."""

        with self._store_lock:
            self._payload_store.close()
        digests: dict[str, str] = {}
        with self._entries_lock:
            for index, entry in enumerate(self._tensor_entries):
                if entry.relative_path not in digests:
                    digests[entry.relative_path] = sha256_file(self.tmp_path / entry.relative_path)
                sealed = dataclasses.replace(entry, sha256=digests[entry.relative_path])
                self._tensor_entries[index] = sealed
                self._entries_by_blob_id[entry.blob_id] = sealed

    def _write_tensor_blob(
        self,
        *,
        blob_id: str,
        tensor: torch.Tensor,
        kind: str,
        label: str,
    ) -> TensorEntry:
        """Queue one supported tensor into the open chunk and return its manifest entry."""

        return self._write_prepared_blob(
            blob_id=blob_id,
            prepared=self._prepare_deferred_snapshot(tensor),
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
        """Queue one prepared payload into the open chunk and build its entry.

        Worker-thread safe: the payload is a private CPU-contiguous tensor, so
        queueing and writing it run no wrapped torch function. ``sha256`` is the
        chunk digest, set at staging (:meth:`_seal_payload_chunks`) or at once
        when each payload gets its own file.

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

        payload, reference = prepared.payload, prepared.reference
        digest = ""
        with self._store_lock:
            blob, key = self._payload_store.submit(payload, codec=TORCH_CODEC)
            if self._payload_per_file:
                self._payload_store.writer(TORCH_CODEC).flush()
                digest = sha256_file(self.tmp_path / blob)
        with self._entries_lock:
            self._tensor_refs[blob_id] = {"blob": blob, "key": key, **reference}
        return TensorEntry(
            blob_id=blob_id,
            kind=kind,
            label=label,
            relative_path=blob,
            backend="safetensors",
            shape=list(reference["shape"]),  # the origin tensor's (complex stays complex)
            dtype=reference["dtype"],
            device_at_save=prepared.device_at_save,
            layout=str(payload.layout).replace("torch.", ""),
            bytes=int(payload.numel() * payload.element_size()),
            sha256=digest,
            requires_grad=prepared.requires_grad,
        )

    def _prepare_deferred_snapshot(self, tensor: torch.Tensor) -> _PreparedPayload:
        """Return a private CPU-contiguous byte snapshot of one payload.

        Runs on the SUBMITTING thread under ``pause_logging`` so the deferred
        write preserves the sync path's value-at-call-time semantics: a later
        in-place mutation of the source tensor (or of storage it aliases)
        must never reach the artifact. Whenever the resolve/contiguous chain
        made no copy, the result still aliases the caller's storage and is
        cloned (``payloads.host_tensor``); non-CPU payloads move to CPU here
        so the worker never runs a wrapped torch op. Both write paths use it:
        a queued chunk holds the payload until the chunk is written.

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
            snapshot = host_tensor(tensor)
            reference = tensor_reference(tensor, blob=None, key=None)
        return _PreparedPayload(
            payload=snapshot,
            device_at_save=str(tensor.device),
            requires_grad=bool(tensor.requires_grad),
            reference=reference,
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
        ArtifactError
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
            raise ArtifactError(reason)

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
            raise ArtifactError(reason)
        with self._entries_lock:
            duplicate = blob_id in self._known_blob_ids
            if not duplicate:
                self._known_blob_ids.add(blob_id)
        if duplicate:
            reason = f"Duplicate streaming blob_id={blob_id} for {label}."
            self.abort(reason)
            raise ArtifactError(reason)

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
        ArtifactError
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
            raise ArtifactError(reason) from (exc.__cause__ or exc)

    def _settle_async_engine(self) -> None:
        """FINALIZATION barrier: land every capture-time write, retire the engine.

        Every accepted write must land before the manifest is built and the
        bundle can publish; a latched write failure aborts here, typed, so a
        bundle with a silently missing blob can never be presented as
        complete. The engine then retires so finalize's remaining blob specs
        take the ordinary synchronous path.

        Raises
        ------
        ArtifactError
            If any queued write failed. The bundle is marked PARTIAL first.
        """

        self._drain_async()
        engine = self._async_engine
        if engine is not None:
            engine.shutdown(discard=False)
            self._async_engine = None

    def _write_manifest(
        self,
        scrubbed_state: dict[str, Any],
        unsupported: list[dict[str, str]],
        trace: Any,
    ) -> None:
        """Build the manifest base of the streamed bundle; hold it and write ``manifest.json``.

        The publish adds the 3.0 envelope fields (records, files, payload
        index, policy) on top of this base.
        """

        tensor_entries = list(self._tensor_entries)
        n_out_blobs = sum(1 for entry in tensor_entries if entry.kind == "out")
        n_grad_blobs = sum(1 for entry in tensor_entries if entry.kind == "grad")
        n_auxiliary_blobs = len(tensor_entries) - n_out_blobs - n_grad_blobs
        layer_list = scrubbed_state.get("_layer_list", [])
        n_layers = len(layer_list) if isinstance(layer_list, list) else 0
        # Disclose the harvested module-attribute and buffer channels (R62): every save
        # writes both disclosures and honors the same opt-outs and warnings as Trace.save.
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

        legacy_manifest = Manifest(
            tlspec_version=TLSPEC_VERSION,
            torchlens_version=TORCHLENS_VERSION,
            torch_version=torch.__version__,
            python_version=_runtime_python_version(),
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
        self._staged_base_manifest = _TlSpecWriter.build_trace_manifest(
            trace=trace,
            legacy_manifest=legacy_manifest,
            save_level="portable",
        )
        _TlSpecWriter.write_json(self.tmp_path / "manifest.json", self._staged_base_manifest)
        _restrict_mode(self.tmp_path / "manifest.json", 0o600)

    def _ensure_writable(self) -> None:
        """Raise if the writer has already been closed."""

        if self._closed:
            raise ArtifactError("Streaming bundle writer is already closed.")

    def _mark_partial(self, reason: str) -> None:
        """Best-effort write the partial sentinel and human-readable reason."""

        try:
            if self.tmp_path.exists():
                (self.tmp_path / PARTIAL_SENTINEL).write_text("", encoding="utf-8")
                (self.tmp_path / REASON_SENTINEL).write_text(reason, encoding="utf-8")
        except OSError:
            return
