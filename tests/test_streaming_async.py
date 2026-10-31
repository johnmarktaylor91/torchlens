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


def test_engine_executes_jobs_fifo() -> None:
    """Jobs land strictly in submission order (ORDERING)."""

    engine = AsyncWriteEngine(max_pending_bytes=1 << 20)
    landed: list[int] = []
    for index in range(50):
        engine.submit(
            PendingWrite(blob_id=str(index), nbytes=16, job=lambda i=index: landed.append(i))
        )
    engine.drain()
    engine.shutdown(discard=False)
    assert landed == list(range(50))


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


def test_engine_admits_single_oversized_item_when_idle() -> None:
    """One blob larger than the whole budget cannot deadlock submission."""

    engine = AsyncWriteEngine(max_pending_bytes=10)
    landed: list[str] = []
    engine.submit(PendingWrite(blob_id="big", nbytes=1000, job=lambda: landed.append("big")))
    engine.drain()
    engine.shutdown(discard=False)
    assert landed == ["big"]


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


def test_async_and_sync_bundles_are_byte_identical(tmp_path: Path) -> None:
    """Async writes produce the same blobs, ids, order, and hashes as sync."""

    x = torch.randn(4, 8)
    tl.trace(_model(), x, storage=tl.to_disk(tmp_path / "async.tlspec"))
    tl.trace(_model(), x, storage=tl.to_disk(tmp_path / "sync.tlspec", async_writes=False))
    async_entries = _manifest(tmp_path / "async.tlspec")["tensors"]
    sync_entries = _manifest(tmp_path / "sync.tlspec")["tensors"]
    assert [e["blob_id"] for e in async_entries] == [e["blob_id"] for e in sync_entries]
    assert [e["blob_id"] for e in async_entries] == sorted(e["blob_id"] for e in async_entries)
    assert {e["blob_id"]: e["sha256"] for e in async_entries} == {
        e["blob_id"]: e["sha256"] for e in sync_entries
    }
    loaded_async = tl.load(tmp_path / "async.tlspec")
    loaded_sync = tl.load(tmp_path / "sync.tlspec")
    for index in range(len(loaded_async._layer_list)):
        assert torch.equal(loaded_async[index].out, loaded_sync[index].out)


def test_async_worker_logs_no_spurious_ops(tmp_path: Path) -> None:
    """Worker-thread serialization must not add ops to the captured graph."""

    x = torch.randn(4, 8)
    plain = tl.trace(_model(), x)
    streamed = tl.trace(_model(), x, storage=tl.to_disk(tmp_path / "b.tlspec"))
    assert len(streamed._layer_list) == len(plain._layer_list)
    assert [op.layer_label for op in streamed._layer_list] == [
        op.layer_label for op in plain._layer_list
    ]


def test_record_refuses_explicit_async_writes(tmp_path: Path) -> None:
    """record() streaming cannot honor the async pipeline and refuses typed."""

    with pytest.raises(RecordingConfigError, match="async_writes"):
        tl.record(
            _model(),
            torch.randn(4, 8),
            save=tl.func("relu"),
            streaming=tl.to_disk(tmp_path / "r.tlspec", async_writes=True),
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


def test_finalize_waits_for_every_slow_pending_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slow writes all land before the bundle publishes (FINALIZATION).

    Every write is delayed so the worker is guaranteed to still be behind
    when the forward finishes; the published bundle must nevertheless hold
    every blob with intact hashes and loadable payloads.
    """

    real_save_file = streaming_module.save_file

    def slow_save_file(tensors, filename) -> None:
        time.sleep(0.02)
        real_save_file(tensors, filename)

    monkeypatch.setattr(streaming_module, "save_file", slow_save_file)
    bundle = tmp_path / "slow.tlspec"
    x = torch.randn(4, 8)
    tl.trace(_model(), x, storage=tl.to_disk(bundle))

    manifest = _manifest(bundle)
    assert manifest["tensors"], "expected streamed blobs"
    from torchlens._io.manifest import sha256_of_file

    for entry in manifest["tensors"]:
        blob_path = bundle / entry["relative_path"]
        assert blob_path.exists()
        assert sha256_of_file(blob_path) == entry["sha256"]
    loaded = tl.load(bundle)
    reference = tl.trace(_model(), x)
    for index in range(len(loaded._layer_list)):
        assert torch.equal(loaded[index].out, reference[index].out)


def test_streamed_capture_backpressure_peak_respects_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real streamed capture against a slow disk keeps pending bytes bounded."""

    real_save_file = streaming_module.save_file

    def slow_save_file(tensors, filename) -> None:
        time.sleep(0.01)
        real_save_file(tensors, filename)

    monkeypatch.setattr(streaming_module, "save_file", slow_save_file)
    budget = 4096
    engines: list[AsyncWriteEngine] = []
    original_init = AsyncWriteEngine.__init__

    def tracking_init(self, *, max_pending_bytes: int) -> None:
        original_init(self, max_pending_bytes=max_pending_bytes)
        engines.append(self)

    monkeypatch.setattr(AsyncWriteEngine, "__init__", tracking_init)
    bundle = tmp_path / "bounded.tlspec"
    tl.trace(
        _model(),
        torch.randn(4, 8),
        storage=tl.to_disk(bundle, max_pending_bytes=budget),
    )
    assert len(engines) == 1
    # Every payload here is <= budget, so the bound must hold exactly.
    assert engines[0].peak_pending_bytes <= budget
    assert _manifest(bundle)["tensors"]
