"""Paddle capture modes that keep payloads out of RAM: structure-only and streaming.

Structure-only capture (``structure_only=True``) runs the real eager forward, so
concrete control flow is taken exactly as in a full capture, but no op keeps a
payload. The trace carries the structure-only marker, so every value-bearing
surface (validation, value exports, runnable saves) refuses through the one
structure-only chokepoint instead of reporting a check it has no witnesses for.

Streaming capture (``storage=tl.to_disk(path)``) hands each op payload to the
shared bundle stream writer as the wrapped call returns, through the bounded
async pipeline (``max_pending_bytes``), and releases it from RAM unless
``retain_in_memory=True``. Payloads travel as the Paddle payload codec encodes
them for ``Trace.save``, so the published bundle loads back as Paddle tensors.
The bundle is staged and published by the shared streamed-bundle finalizer once
the capture outcome is settled; a capture that raises leaves a PARTIAL temp
bundle, never a published one.

Both modes are honored only once the Paddle backend spec opens the matching
capability gate (``structure_only_capture`` and ``has_streaming``); until then
the options stay with the extra-keyword policy, which refuses them as before.
Shared modules that sit in the package import cycle (the writer class, the
finalizer) are passed in by the backend, so this module stays outside it.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ... import _state
from ..._errors import InvalidArgumentError
from ..._io import ArtifactError, BlobRef, refuse_bundle_exists
from ..._io._numpy_payload_codec import numpy_to_transport_tensor
from ..._io.format3.payloads import NUMPY_CODEC
from ..._io.manifest import TensorEntry
from ..._io.payload_codec import get_payload_codec
from ..._utils.hashing import sha256_file
from ..registry import get_backend_spec
from . import _retention

MODE_KEY = "_paddle_capture_mode"
"""Trace ``__dict__`` key holding the capture's :class:`PaddleCaptureMode`."""


@dataclass(frozen=True)
class PaddleCaptureMode:
    """Payload handling of one Paddle capture.

    Attributes
    ----------
    structure_only
        Whether no op keeps a payload.
    storage
        The ``storage=`` options payloads stream to, or ``None``.
    retained_selector
        The static-label ``save=`` selector whose unselected payloads leave RAM
        during capture, or ``None``.
    """

    structure_only: bool = False
    storage: Any = None
    retained_selector: Any = None


def pop_capture_mode(
    capture: Any,
    extra_kwargs: dict[str, Any],
    retained_selector: object = None,
    save_predicate: object = None,
) -> PaddleCaptureMode:
    """Take the payload-mode options the Paddle spec admits out of ``extra_kwargs``.

    Parameters
    ----------
    capture
        The call's ``CaptureOptions``, or ``None``.
    extra_kwargs
        Keyword options not bound by ``capture_trace``; admitted ones are removed.
    retained_selector
        The static-label ``save=`` selector capture-time retention may use.
    save_predicate
        The call's ``save=`` predicate, or ``None``.

    Returns
    -------
    PaddleCaptureMode
        The admitted mode; options the spec does not admit stay for the refusal policy.

    Raises
    ------
    InvalidArgumentError
        If ``storage=`` is combined with ``structure_only=True`` or ``save=``.
    """

    capabilities = get_backend_spec("paddle").capabilities
    structure_only = False
    if bool(getattr(capabilities, "structure_only_capture", False)):
        flag = extra_kwargs.pop("structure_only", None)
        structure_only = bool(flag) or bool(getattr(capture, "structure_only", False))
    storage = None
    if bool(getattr(capabilities, "has_streaming", False)):
        storage = _pop_storage(extra_kwargs)
    if storage is not None and (structure_only or save_predicate is not None):
        partner = "structure_only=True" if structure_only else "save="
        raise InvalidArgumentError(
            f"the Paddle backend does not combine storage= with {partner} yet",
            code="trace_option_unsupported",
            remedy=f"pass storage= or {partner}, not both",
            argument="storage",
        )
    keep_selector = None if structure_only or storage is not None else retained_selector
    return PaddleCaptureMode(structure_only, storage, keep_selector)


def _pop_storage(extra_kwargs: dict[str, Any]) -> Any:
    """Pop ``storage=`` (or its ``streaming=`` spelling); ``None`` when nothing streams."""

    storage = extra_kwargs.pop("storage", None)
    streaming = extra_kwargs.pop("streaming", None)
    if storage is not None and streaming is not None and storage is not streaming:
        raise InvalidArgumentError(
            "storage= and streaming= name the same option; pass one of them",
            code="trace_option_unsupported",
            remedy="pass storage=tl.to_disk(path) only",
            argument="storage",
        )
    chosen = storage if storage is not None else streaming
    if _storage_path(chosen) is None and getattr(chosen, "on_save", None) is None:
        return None
    return chosen


def _storage_path(storage: Any) -> Any:
    """The bundle path of ``storage`` (``StorageOptions.path``, or a bare path)."""

    return getattr(storage, "path", storage)


def start_capture_mode(trace: Any, mode: PaddleCaptureMode, writer_base: type) -> None:
    """Arm ``mode`` on the trace under construction.

    Parameters
    ----------
    trace
        The trace under construction.
    mode
        The admitted payload mode.
    writer_base
        ``BundleStreamWriter``; streamed payloads go through a Paddle-codec subclass.

    Raises
    ------
    ArtifactError
        If the ``storage=`` bundle path already exists (``bundle_path_exists``).
    """

    trace.__dict__[MODE_KEY] = mode
    trace._options_structure_only = mode.structure_only
    if mode.retained_selector is not None:
        _retention.start_retention(trace, mode.retained_selector)
    if mode.storage is None:
        return
    path = _storage_path(mode.storage)
    if path is not None and Path(path).exists():
        refuse_bundle_exists(path)
    trace.__dict__[_retention.STREAM_KEY] = PaddleStream(
        storage=mode.storage, writer_base=writer_base, entry_exception=sys.exc_info()[1]
    )


def saves_payloads(session: Any) -> bool:
    """Whether ops of the capture ``session`` keep payloads (False for structure-only)."""

    mode = session.__dict__.get(MODE_KEY)
    return mode is None or not mode.structure_only


@dataclass
class PaddleStream:
    """Streaming state of one capture; the writer itself is ``trace._out_writer``.

    Attributes
    ----------
    storage
        The admitted ``storage=`` options (or bare bundle path).
    writer_base
        ``BundleStreamWriter``, passed in by the backend.
    entry_exception
        The exception being handled when capture started, so teardown can tell
        a failed capture from a capture made inside an ``except`` block.
    """

    storage: Any
    writer_base: type
    entry_exception: BaseException | None

    def writer(self, trace: Any) -> Any:
        """Return the capture's writer, opening it on first use (``None``: no bundle path)."""

        writer = getattr(trace, "_out_writer", None)
        path = _storage_path(self.storage)
        if writer is not None or path is None:
            return writer
        writer = paddle_writer_class(self.writer_base)(
            path,
            include_custom_attributes=bool(
                getattr(self.storage, "include_custom_attributes", True)
            ),
            include_buffer_values=bool(getattr(self.storage, "include_buffer_values", True)),
        )
        if getattr(self.storage, "async_writes", None) is not False:
            try:
                writer.arm_async_writes(getattr(self.storage, "max_pending_bytes", None))
            except BaseException as exc:
                writer.abort(f"Failed to arm async streaming writes: {exc}")
                raise
        trace._out_writer = writer
        return writer

    def submit(self, trace: Any, ref: Any, payload: Any, label: str) -> Any:
        """Stream one op payload; return its tensor reference with the bundle blob attached.

        Parameters
        ----------
        trace
            The trace under construction.
        ref
            The op output's ``TensorRef`` carrying ``payload``.
        payload
            The Paddle tensor to persist.
        label
            The op's raw label (the shared finalizer relabels the blob).

        Returns
        -------
        Any
            ``ref`` with the blob reference, and no payload unless ``retain_in_memory``.
        """

        on_save = getattr(self.storage, "on_save", None)
        if callable(on_save):
            with _state.pause_logging():
                on_save(label, payload)
        writer = self.writer(trace)
        if writer is None:
            return ref
        blob_id = writer.next_blob_id()
        writer.submit_blob(blob_id, payload, kind="out", label=label)
        kept = payload if bool(getattr(self.storage, "retain_in_memory", False)) else None
        return replace(ref, payload=kept, blob_ref=BlobRef(blob_id=blob_id, kind="out"))


def end_capture_mode(trace: Any) -> None:
    """Drop capture-only state at teardown; a failed capture's bundle is aborted.

    Runs in the capture's ``finally``. On success the stream stays for
    :func:`settle`; when an exception is propagating the retention spool is
    removed and the temp bundle is marked PARTIAL (never published).

    Parameters
    ----------
    trace
        The trace under construction.
    """

    trace.__dict__.pop(MODE_KEY, None)
    _retention.discard_retention(trace)
    stream = trace.__dict__.get(_retention.STREAM_KEY)
    failure = sys.exc_info()[1]
    if stream is None or failure is stream.entry_exception:
        return
    trace.__dict__.pop(_retention.STREAM_KEY, None)
    writer = getattr(trace, "_out_writer", None)
    if writer is not None:
        writer.abort(f"Paddle capture failed: {type(failure).__name__}: {failure}")
        trace._out_writer = None


def settle(trace: Any, stamp: Callable[[Any], Any], finalize_bundle: Callable[[Any], None]) -> None:
    """Settle the capture outcome, then publish its streamed bundle.

    Parameters
    ----------
    trace
        The finished trace.
    stamp
        ``stamp_backend_finalized``: settles the capture outcome.
    finalize_bundle
        ``_finalize_streamed_bundle``: stages the bundle and, the outcome being
        settled, publishes it with the outcome attestation.
    """

    stamp(trace)
    stream = trace.__dict__.pop(_retention.STREAM_KEY, None)
    if stream is not None and stream.writer(trace) is not None:
        finalize_bundle(trace)


class PaddleCodecWrites:
    """Writer mixin: Paddle payloads travel as the Paddle codec's NumPy transport.

    ``Trace.save`` writes a Paddle payload as its codec-encoded transport tensor
    in the NumPy-codec chunk with the codec's manifest fields, which is what
    makes ``tl.load`` rebuild Paddle tensors. The mixin does the same for the
    stream writer: encoding runs on the submitting thread (the host copy is the
    value-at-call-time snapshot), and the entry is recorded with the codec
    fields. Torch payloads pass through to the base writer unchanged.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._paddle_codec_fields: dict[str, tuple[dict[str, Any], str]] = {}

    def write_blob(self, blob_id: str, tensor: Any, *, kind: str, label: str) -> Any:
        """Write one blob; a Paddle tensor is encoded first (see the class docstring)."""

        transport = self._paddle_transport(blob_id, tensor, label)
        return super().write_blob(blob_id, transport, kind=kind, label=label)  # type: ignore[misc]

    def submit_blob(self, blob_id: str, tensor: Any, *, kind: str, label: str) -> None:
        """Submit one blob to the bounded pipeline; a Paddle tensor is encoded first."""

        transport = self._paddle_transport(blob_id, tensor, label)
        super().submit_blob(blob_id, transport, kind=kind, label=label)  # type: ignore[misc]

    def _paddle_transport(self, blob_id: str, value: Any, label: str) -> Any:
        """Return the transport tensor of a Paddle payload (other values unchanged)."""

        codec = get_payload_codec("paddle")
        if not codec.can_encode(value):
            return value
        try:
            with _state.pause_logging():
                encoded = codec.to_numpy(value)
                fields = dict(codec.manifest_fields(value, encoded))
            transport = numpy_to_transport_tensor(encoded.array)
        except Exception as exc:
            reason = (
                f"Unsupported Paddle payload for streaming at {label} (blob_id={blob_id}): {exc}"
            )
            self.abort(reason)  # type: ignore[attr-defined]
            raise ArtifactError(reason) from exc
        self._paddle_codec_fields[blob_id] = (fields, str(encoded.logical_device))
        return transport

    def _write_prepared_blob(self, *, blob_id: str, prepared: Any, kind: str, label: str) -> Any:
        """Queue a prepared payload; Paddle transports go to the NumPy-codec chunk."""

        codec_entry = self._paddle_codec_fields.get(blob_id)
        if codec_entry is None:
            return super()._write_prepared_blob(  # type: ignore[misc]
                blob_id=blob_id, prepared=prepared, kind=kind, label=label
            )
        fields, logical_device = codec_entry
        payload, reference = prepared.payload, prepared.reference
        digest = ""
        with self._store_lock:  # type: ignore[attr-defined]
            store = self._payload_store  # type: ignore[attr-defined]
            blob, key = store.submit(payload, codec=NUMPY_CODEC)
            if self._payload_per_file:  # type: ignore[attr-defined]
                store.writer(NUMPY_CODEC).flush()
                digest = sha256_file(self.tmp_path / blob)  # type: ignore[attr-defined]
        codec_reference = {
            "blob": blob,
            "key": key,
            **reference,
            "codec": {"id": NUMPY_CODEC, **fields},
        }
        with self._entries_lock:  # type: ignore[attr-defined]
            self._tensor_refs[blob_id] = codec_reference  # type: ignore[attr-defined]
        return TensorEntry(
            blob_id=blob_id,
            kind=kind,
            label=label,
            relative_path=blob,
            backend="safetensors",
            shape=list(reference["shape"]),
            dtype=reference["dtype"],
            device_at_save=logical_device,
            layout=str(payload.layout).replace("torch.", ""),
            bytes=int(payload.numel() * payload.element_size()),
            sha256=digest,
            requires_grad=False,
            **fields,
        )


@functools.cache
def paddle_writer_class(writer_base: type) -> type:
    """Return ``writer_base`` with :class:`PaddleCodecWrites` mixed in (one class per base)."""

    return type(
        "PaddleBundleStreamWriter",
        (PaddleCodecWrites, writer_base),
        {"__module__": __name__, "__doc__": PaddleCodecWrites.__doc__},
    )
