"""Dependency-free sinks: in-memory (tests/inspection) and JSONL (item 11).

``MemorySink`` is the reference receiver the engine tests run against.
``JSONLSink`` is the future-viewer plumbing: persisted rows over the same
protocol, one JSON object per line, self-describing header row first. Both
carry every non-image capability, so the full engine contract is CI-tested
with zero optional dependencies.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

from ._errors import SinkDeliveryError
from ._records import EMISSION_SCHEMA_VERSION, HistogramPoint, ScalarPoint, TextPoint

__tl_layer__ = "L8"

#: JSONL artifact identity: the first row of every file names this format.
JSONL_FORMAT = "torchlens.trackers.jsonl.v1"

_MEMORY_CAPABILITIES = frozenset(
    {"scalar", "raw_histogram", "text_manifest", "graph", "structured_event", "run_metadata"}
)


class MemorySink:
    """In-memory reference sink; records every accepted point verbatim."""

    def __init__(self, *, capabilities: frozenset[str] | None = None) -> None:
        """Optionally narrow the advertised capability set (test knob)."""

        self._capabilities = capabilities if capabilities is not None else _MEMORY_CAPABILITIES
        self.scalars: list[ScalarPoint] = []
        self.histograms: list[HistogramPoint] = []
        self.texts: list[TextPoint] = []
        self.graphs: list[Any] = []
        self.flush_count = 0
        self.closed = False

    def capabilities(self) -> frozenset[str]:
        """Advertised capability set."""

        return self._capabilities

    def emit_scalar(self, point: ScalarPoint) -> None:
        """Record one scalar point."""

        self.scalars.append(point)

    def emit_histogram(self, point: HistogramPoint) -> None:
        """Record one histogram point."""

        self.histograms.append(point)

    def emit_text(self, point: TextPoint) -> None:
        """Record one text point."""

        self.texts.append(point)

    def emit_graph(self, payload: Any) -> None:
        """Record one graph payload."""

        self.graphs.append(payload)

    def flush(self) -> None:
        """Count flushes (the engine flushes at close)."""

        self.flush_count += 1

    def close(self) -> None:
        """Mark the sink closed; idempotent."""

        self.closed = True


class JSONLSink:
    """Line-delimited JSON sink over the shared protocol (build item 11).

    Row shapes: a header row (``{"format": ..., "version": ...}``) first,
    then ``{"kind": "scalar" | "histogram" | "text" | "graph", ...}`` rows in
    emission order. A write failure latches the sink failed and raises
    typed; the absent ``{"kind": "closed"}`` footer marks a partial file.

    Every row is ONE unbuffered write and the file is truncated back to the
    last fully written row on failure, so "rows already written remain valid
    JSONL" is a property of the file, not a hope: the old default-buffered
    handle left a TORN last line whenever the filesystem died mid-flush
    (AUD-CODE 2.14).
    """

    def __init__(self, path: str | Path) -> None:
        """Open ``path`` for writing and emit the self-describing header."""

        self.path = Path(path)
        self._failed = False
        self._closed = False
        self._committed_bytes = 0
        self._handle: BinaryIO = self.path.open("wb", buffering=0)
        self._write_row({"format": JSONL_FORMAT, "version": EMISSION_SCHEMA_VERSION})

    def capabilities(self) -> frozenset[str]:
        """Everything except images (a JSON row has no pixel channel)."""

        return frozenset({"scalar", "raw_histogram", "text_manifest", "graph", "structured_event"})

    def _write_row(self, row: dict[str, Any]) -> None:
        """Serialize one row; latch failed and refuse typed on any OS error."""

        if self._failed:
            raise SinkDeliveryError(
                f"JSONLSink({self.path}) already latched failed; a latched "
                "sink never accepts more rows -- retrying into a torn file "
                "would present a partial run as a complete one.",
                code="tracker_sink_delivery_failed",
                sink="JSONLSink",
                path=str(self.path),
                remedy="Construct a fresh sink on a healthy filesystem path.",
            )
        line = (json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n").encode("ascii")
        try:
            written = 0
            while written < len(line):
                # An unbuffered raw handle may accept a PARTIAL write (ENOSPC
                # mid-row); loop until the row is whole or the OS refuses.
                count = self._handle.write(line[written:])
                written += int(count or 0)
        except (OSError, ValueError) as exc:
            # ValueError covers writes on a handle the OS/user already
            # closed -- the same delivery failure as a raw OSError.
            self._failed = True
            self._truncate_torn_tail()
            raise SinkDeliveryError(
                f"JSONLSink({self.path}) write failed: {exc}. The sink is "
                "latched failed; rows already written remain valid JSONL and "
                "the missing 'closed' footer marks the file partial.",
                code="tracker_sink_delivery_failed",
                sink="JSONLSink",
                path=str(self.path),
                remedy="Free disk space or point the sink at a writable path.",
            ) from exc
        self._committed_bytes += len(line)

    def _truncate_torn_tail(self) -> None:
        """Best-effort: cut the file back to the last fully written row.

        Shrinking never needs free space, so this succeeds on the ENOSPC path
        that tore the row; a handle the caller already closed wrote nothing
        past the committed prefix, so there is nothing to cut.
        """

        with contextlib.suppress(OSError, ValueError):
            os.ftruncate(self._handle.fileno(), self._committed_bytes)

    def emit_scalar(self, point: ScalarPoint) -> None:
        """Write one scalar row."""

        self._write_row(
            {"kind": "scalar", "tag": point.tag, "step": point.step, "value": point.value}
        )

    def emit_histogram(self, point: HistogramPoint) -> None:
        """Write one histogram row (exact counts, explicit edges)."""

        self._write_row(
            {
                "kind": "histogram",
                "tag": point.tag,
                "step": point.step,
                "counts": list(point.counts),
                "edges": list(point.edges),
                "summary": point.summary,
            }
        )

    def emit_text(self, point: TextPoint) -> None:
        """Write one text row."""

        self._write_row({"kind": "text", "tag": point.tag, "step": point.step, "text": point.text})

    def emit_graph(self, payload: Any) -> None:
        """Write one graph row (the dep-free graph IR serializes as JSON)."""

        as_dict = getattr(payload, "as_dict", None)
        self._write_row({"kind": "graph", "graph": as_dict() if callable(as_dict) else payload})

    def flush(self) -> None:
        """Flush the underlying handle."""

        if not self._closed:
            self._handle.flush()

    def close(self) -> None:
        """Write the footer row and close the handle; idempotent."""

        if self._closed:
            return
        if not self._failed:
            self._write_row({"kind": "closed"})
        self._closed = True
        self._handle.close()


__all__ = ["JSONL_FORMAT", "JSONLSink", "MemorySink"]
