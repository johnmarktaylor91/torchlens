"""Bounded single-worker write pipeline behind the async streaming bundle writer.

``BundleStreamWriter`` submits prepared blob-write jobs here so the slow parts
of a streamed disk save (safetensors serialization, the page-cache write, and
the sha256 integrity hash) overlap forward capture instead of pausing it. The
engine guarantees the four properties the async path must never trade away:

- ORDERING: exactly one worker thread executes jobs strictly FIFO, so manifest
  entries land in submission (= blob id) order.
- BACKPRESSURE: submission blocks while the pending snapshot bytes exceed the
  budget, so a disk slower than capture slows CAPTURE down instead of piling
  unbounded tensor snapshots into the RAM the disk save exists to spare.
- FAILURE: the first job failure latches, discards the now-pointless queue,
  and re-raises at the next writer interaction (submit or drain) so a broken
  write can never be silently absorbed.
- FINALIZATION: ``drain()`` is a hard barrier -- it returns only after every
  accepted job landed, or raises the latched failure.

The worker thread never runs wrapped torch functions: jobs receive
caller-thread-prepared CPU-contiguous snapshots and call only safetensors
serialization plus hashlib (both non-torch), and the torch-function wrapper
gate additionally passes non-owner threads through unlogged.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_MAX_PENDING_BYTES = 256 * 1024 * 1024
"""Default pending-snapshot byte budget for the async streaming writer."""

_ABORT_JOIN_TIMEOUT_S = 10.0
"""Bounded best-effort join on abort so a wedged disk cannot hang teardown."""


class AsyncWriteFailedError(Exception):
    """Internal signal that a queued blob write failed.

    ``BundleStreamWriter`` catches this and converts it into its typed
    ``ArtifactError`` after marking the temp bundle PARTIAL; it never
    escapes to users.
    """


@dataclass(frozen=True)
class PendingWrite:
    """One accepted blob-write job awaiting the worker.

    Parameters
    ----------
    blob_id:
        Blob identifier, used in failure reasons.
    nbytes:
        Snapshot payload bytes charged against the pending budget.
    job:
        Zero-argument callable performing the write and entry recording.
    """

    blob_id: str
    nbytes: int
    job: Callable[[], None]


class AsyncWriteEngine:
    """Single-worker FIFO write pipeline with a pending-bytes budget.

    Parameters
    ----------
    max_pending_bytes:
        Budget for queued + in-flight snapshot bytes. A submission that would
        exceed it blocks until the worker frees space. One oversized job is
        always admitted when the pipeline is idle so a single blob larger
        than the budget cannot deadlock.
    """

    def __init__(self, *, max_pending_bytes: int) -> None:
        """Start the worker thread with an empty queue."""

        if max_pending_bytes <= 0:
            raise ValueError("max_pending_bytes must be positive.")
        self._max_pending_bytes = max_pending_bytes
        self._cond = threading.Condition()
        self._pending: deque[PendingWrite] = deque()
        self._active: PendingWrite | None = None
        self._pending_bytes = 0
        self._peak_pending_bytes = 0
        self._failure_reason: str | None = None
        self._failure_cause: BaseException | None = None
        self._shutdown = False
        self._worker = threading.Thread(
            target=self._run, name="torchlens-stream-writer", daemon=True
        )
        self._worker.start()

    @property
    def peak_pending_bytes(self) -> int:
        """Return the high-water mark of queued + in-flight snapshot bytes."""

        with self._cond:
            return self._peak_pending_bytes

    def submit(self, item: PendingWrite) -> None:
        """Accept one write job, blocking while the byte budget is exhausted.

        Parameters
        ----------
        item:
            Prepared write job.

        Raises
        ------
        AsyncWriteFailedError
            If an earlier queued write already failed (latched failure).
        RuntimeError
            If the engine was already shut down.
        """

        with self._cond:
            while True:
                self._raise_if_failed()
                if self._shutdown:
                    raise RuntimeError("Async streaming write engine is already shut down.")
                if self._pending_bytes == 0 or (
                    self._pending_bytes + item.nbytes <= self._max_pending_bytes
                ):
                    break
                self._cond.wait()
            self._pending.append(item)
            self._pending_bytes += item.nbytes
            self._peak_pending_bytes = max(self._peak_pending_bytes, self._pending_bytes)
            self._cond.notify_all()

    def drain(self) -> None:
        """Block until every accepted job landed; raise the latched failure.

        Raises
        ------
        AsyncWriteFailedError
            If any queued write failed. The queue behind the failed write was
            discarded, never silently retried.
        """

        with self._cond:
            while self._failure_reason is None and (self._pending or self._active is not None):
                self._cond.wait()
            self._raise_if_failed()

    def shutdown(self, *, discard: bool) -> None:
        """Stop the worker after the queue empties (or immediately).

        Parameters
        ----------
        discard:
            When ``True`` (abort path) queued jobs are dropped unexecuted and
            the join is bounded; when ``False`` the worker finishes the queue
            first. Callers on the success path drain BEFORE shutting down.
        """

        with self._cond:
            self._shutdown = True
            if discard:
                self._pending_bytes -= sum(item.nbytes for item in self._pending)
                self._pending.clear()
            self._cond.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=_ABORT_JOIN_TIMEOUT_S if discard else None)

    def _raise_if_failed(self) -> None:
        """Raise the latched failure, if any. Caller holds the lock."""

        if self._failure_reason is not None:
            raise AsyncWriteFailedError(self._failure_reason) from self._failure_cause

    def _run(self) -> None:
        """Worker loop: execute jobs FIFO; latch the first failure and stop."""

        while True:
            with self._cond:
                while not self._pending and not self._shutdown:
                    self._cond.wait()
                if not self._pending:
                    self._cond.notify_all()
                    return
                item = self._pending.popleft()
                self._active = item
            try:
                item.job()
            except BaseException as exc:  # noqa: BLE001 - latched and re-raised typed
                with self._cond:
                    self._failure_reason = (
                        f"Async streaming write failed for blob_id={item.blob_id}: {exc}"
                    )
                    self._failure_cause = exc
                    self._active = None
                    self._pending_bytes = 0
                    self._pending.clear()
                    self._cond.notify_all()
                return
            with self._cond:
                self._pending_bytes -= item.nbytes
                self._active = None
                self._cond.notify_all()
