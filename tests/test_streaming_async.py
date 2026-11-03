"""Async streaming disk-write tests: ordering, backpressure, failure, finalization.

Failure and finalization are proven by FAULT INJECTION (writes actually break
mid-capture), never by code inspection: a dropped write nobody notices is the
exact silent-loss class this feature must not introduce.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

import torchlens as tl
from torchlens._io import ArtifactError, streaming as streaming_module
from torchlens._io.streaming import BundleStreamWriter
from torchlens._io.streaming_async import (
    AsyncWriteEngine,
    AsyncWriteFailedError,
    PendingWrite,
)
from torchlens.errors import RecordingConfigError


def _model() -> nn.Module:
    """Return a small deterministic model for streaming tests."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8), nn.Tanh())


def _manifest(bundle: Path) -> dict:
    """Load a bundle manifest."""

    with (bundle / "manifest.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _tmp_dirs(parent: Path, name: str) -> list[Path]:
    """Return leftover streaming temp dirs for one bundle name."""

    return sorted(parent.glob(f"{name}.tmp.*"))


# ---------------------------------------------------------------------------
# Engine unit tests: ordering, backpressure, failure latch
# ---------------------------------------------------------------------------


def test_engine_backpressure_bounds_pending_bytes_and_blocks_submitters() -> None:
    """The byte budget is a hard bound and full budgets block capture (BACKPRESSURE)."""

    engine = AsyncWriteEngine(max_pending_bytes=100)
    job_seconds = 0.02
    started = time.perf_counter()
    for index in range(5):
        engine.submit(
            PendingWrite(blob_id=str(index), nbytes=60, job=lambda: time.sleep(job_seconds))
        )
    submit_elapsed = time.perf_counter() - started
    engine.drain()
    engine.shutdown(discard=False)
    # 60 + 60 > 100: at most one item may be pending at a time, so the peak
    # never exceeds the budget and later submits waited on the slow worker.
    assert engine.peak_pending_bytes <= 100
    assert submit_elapsed >= 3 * job_seconds


def test_engine_latches_first_failure_and_discards_queue() -> None:
    """A failed job latches, discards later jobs, and re-raises on interaction."""

    engine = AsyncWriteEngine(max_pending_bytes=1 << 20)
    landed: list[int] = []
    gate = time.sleep

    def _boom() -> None:
        raise OSError("injected disk failure")

    engine.submit(PendingWrite(blob_id="0", nbytes=8, job=lambda: gate(0.01)))
    engine.submit(PendingWrite(blob_id="1", nbytes=8, job=_boom))
    engine.submit(PendingWrite(blob_id="2", nbytes=8, job=lambda: landed.append(2)))
    with pytest.raises(AsyncWriteFailedError, match="blob_id=1"):
        engine.drain()
    with pytest.raises(AsyncWriteFailedError):
        engine.submit(PendingWrite(blob_id="3", nbytes=8, job=lambda: landed.append(3)))
    engine.shutdown(discard=True)
    assert landed == []


# ---------------------------------------------------------------------------
# Writer-level semantics
# ---------------------------------------------------------------------------


def test_deferred_snapshot_preserves_value_at_call_time(tmp_path: Path) -> None:
    """A later in-place mutation of the source tensor never reaches the blob."""

    writer = BundleStreamWriter(tmp_path / "bundle.tlspec")
    writer.arm_async_writes()
    tensor = torch.ones(64, 64)
    expected = tensor.clone()
    writer.submit_blob("0000000001", tensor, kind="out", label="probe")
    tensor.add_(41.0)
    writer._drain_async()
    entry = writer.get_entry("0000000001")
    written = load_file(writer.tmp_path / entry.relative_path)["data"]
    assert torch.equal(written, expected)
    writer.abort("test cleanup")


def test_duplicate_blob_id_refuses_while_write_is_still_queued(tmp_path: Path) -> None:
    """The id reservation catches duplicates even before the worker lands them."""

    writer = BundleStreamWriter(tmp_path / "bundle.tlspec")
    writer.arm_async_writes()
    writer.submit_blob("0000000001", torch.ones(4), kind="out", label="first")
    with pytest.raises(ArtifactError, match="Duplicate streaming blob_id"):
        writer.submit_blob("0000000001", torch.ones(4), kind="out", label="second")


# ---------------------------------------------------------------------------
# Trace-level equivalence
# ---------------------------------------------------------------------------


def test_record_refuses_explicit_async_writes(tmp_path: Path) -> None:
    """record() streaming cannot honor the async pipeline and refuses typed."""

    with pytest.raises(RecordingConfigError, match="async_writes"):
        tl.record(
            _model(),
            torch.randn(4, 8),
            save=tl.func("relu"),
            storage=tl.to_disk(tmp_path / "r.tlspec", async_writes=True),
        )


# ---------------------------------------------------------------------------
# FAULT INJECTION: failure mid-capture and finalization guarantees
# ---------------------------------------------------------------------------


class _FailingSaveFile:
    """save_file stand-in that breaks on one configured call."""

    def __init__(self, fail_on_call: int, real=streaming_module.save_file) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.real = real

    def __call__(self, tensors, filename) -> None:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise OSError(28, "No space left on device (injected)")
        self.real(tensors, filename)


def _count_sync_blobs(tmp_path: Path) -> int:
    """Return the blob count a sync streamed capture of the test model writes."""

    reference = tmp_path / "reference.tlspec"
    tl.trace(_model(), torch.randn(4, 8), storage=tl.to_disk(reference, async_writes=False))
    return len(_manifest(reference)["tensors"])


@pytest.mark.parametrize("fail_position", ["second", "last"])
def test_injected_write_failure_is_typed_and_marks_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_position: str
) -> None:
    """A write that breaks mid-capture raises typed and never publishes (FAILURE).

    The ``last`` case is the finalization tripwire: the failure lands after
    the forward already finished, so only the drain barrier can catch it.
    """

    n_blobs = _count_sync_blobs(tmp_path)
    assert n_blobs >= 3
    fail_on_call = 2 if fail_position == "second" else n_blobs
    fake = _FailingSaveFile(fail_on_call)
    monkeypatch.setattr(streaming_module, "save_file", fake)

    bundle = tmp_path / "broken.tlspec"
    with pytest.raises((ArtifactError, Exception)) as excinfo:
        tl.trace(_model(), torch.randn(4, 8), storage=tl.to_disk(bundle))
    chain_text = ""
    err: BaseException | None = excinfo.value
    while err is not None:
        chain_text += f"{type(err).__name__}: {err}\n"
        err = err.__cause__
    assert "injected" in chain_text
    assert any(isinstance(e, ArtifactError) for e in _exception_chain(excinfo.value))

    # Never a silently incomplete artifact presented as complete:
    assert not bundle.exists()
    leftovers = _tmp_dirs(tmp_path, "broken.tlspec")
    assert leftovers, "expected a PARTIAL-marked temp dir"
    for leftover in leftovers:
        assert (leftover / "PARTIAL").exists()
        reason = (leftover / "REASON.txt").read_text(encoding="utf-8")
        assert "injected" in reason


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return ``exc`` and its ``__cause__``/``__context__`` chain."""

    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain
