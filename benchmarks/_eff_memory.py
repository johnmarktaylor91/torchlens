"""Private bounded-memory and minimal-journal falsification experiments."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import statistics
import sys
import tempfile
import time
import tracemalloc
import weakref
from pathlib import Path

import torch
from _eff_capture import _SelectedCapture, _Tape, _TierRefusal
from _eff_workloads import _build, _Runner
from safetensors.torch import load_file, save_file
from torch import nn

import torchlens as tl


def _checks() -> dict:
    class Repeated(nn.Module):
        def __init__(self):
            super().__init__()
            self.site = nn.Identity()

        def forward(self, x):
            a = self.site(x + 1)
            a.add_(3)
            return self.site(a * 2)

    model = Repeated()
    x = torch.ones(2, 3)
    collector = _SelectedCapture(model, (tl.module("site"),))
    with torch.no_grad(), collector:
        model(x)
    assert [c for _, c, _ in collector.records] == [1, 2]
    assert torch.equal(collector.records[0][2], torch.full_like(x, 2))
    assert torch.equal(collector.records[1][2], torch.full_like(x, 10))
    refused = []
    for mode in ("grad", "identity"):
        try:
            if mode == "identity":
                model.site = nn.Identity()
                with torch.no_grad(), collector:
                    model(x)
            else:
                with collector:
                    model(x)
        except _TierRefusal as exc:
            refused.append(exc.code)
    assert refused == ["gradient_capture_requires_full_engine", "module_identity_changed"]
    assert not collector.handles
    # Strong references do not snapshot mutable tensor values.
    with torch.no_grad(), _Tape(strong=True) as tape:
        a = x + 1
        a.add_(3)
    first = tape.records[0][2][0]
    assert torch.equal(first, torch.full_like(x, 5))
    # Weak references cannot supply replay ground truth after the tensor dies.
    with torch.no_grad(), _Tape() as weak:
        transient = x + 9
    del transient
    gc.collect()
    assert weak.records[0][2][0]() is None
    unchanged_validation = tl.validate(
        nn.Sequential(nn.Linear(3, 4), nn.ReLU()), x, scope="forward"
    )
    assert unchanged_validation is True
    return {
        "repeated_and_mutated_selection_exact": True,
        "refusals": refused,
        "strong_reference_loses_historical_value": True,
        "weak_reference_loses_ground_truth": True,
        "unchanged_full_validation": unchanged_validation,
    }


def _memory(mode: str, batches: int, seq: int) -> dict:
    model, sites, _ = _build("qwen")
    ids = torch.randint(0, 1000, (1, seq), generator=torch.Generator().manual_seed(2))
    collector = _SelectedCapture(model, tuple(tl.module(s) for s in sites))
    retained = []
    bytes_written = 0
    roundtrip = True
    with tempfile.TemporaryDirectory(prefix="selected-cache-") as directory:

        def sink(address, count, tensor):
            nonlocal bytes_written, roundtrip
            path = str(Path(directory) / (address + ".safetensors"))
            save_file({"activation": tensor}, path)
            roundtrip = roundtrip and torch.equal(load_file(path)["activation"], tensor)
            bytes_written += tensor.numel() * tensor.element_size()
            # Keep the benchmark bounded on disk too; a real cache gives each batch a shard.
            os.unlink(path)

        if mode == "stream":
            collector.sink = sink
        with torch.no_grad(), collector:
            model(ids)
        start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        times = []
        for _ in range(3):
            retained.clear()
            gc.collect()
            start = time.perf_counter()
            for step in range(batches):
                with torch.no_grad(), collector:
                    model((ids + step) % 1000)
                if mode == "retain":
                    retained.extend(value for _, _, value in collector.records)
            times.append(time.perf_counter() - start)
    stored = sum(t.numel() * t.element_size() for t in retained)
    return {
        "mode": mode,
        "batches": batches,
        "seq": seq,
        "times_s": times,
        "median_s": statistics.median(times),
        "start_rss_kib": start_rss,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "retained_payload_bytes": stored,
        "streamed_payload_bytes_total": bytes_written,
        "roundtrip_exact": roundtrip,
        "dtype": "float32",
    }


def _journal(strong: bool) -> dict:
    model, sites, _ = _build("qwen")
    ids = torch.randint(0, 1000, (1, 128))
    with torch.no_grad(), _Tape(strong=strong) as tape:
        model(ids)
    gc.collect()
    records = tape.records[: tape.count]
    # Retained Python structure accounting, explicitly excluding shared callable objects.
    python_bytes = sys.getsizeof(tape.records) + sys.getsizeof(tape.producers)
    tensors = {}
    dead = 0
    for record in records:
        python_bytes += sum(sys.getsizeof(v) for v in (record, record[1], record[2]))
        for ref in record[2]:
            tensor = ref if strong else ref()
            if tensor is None:
                dead += 1
            else:
                storage = tensor.untyped_storage()
                tensors[storage.data_ptr()] = storage.nbytes()
            if isinstance(ref, weakref.ReferenceType):
                python_bytes += sys.getsizeof(ref)
    python_bytes += sum(sys.getsizeof(v) for v in tape.producers.values())
    return {
        "strong": strong,
        "n_ops": tape.count,
        "python_bytes_partial": python_bytes,
        "python_bytes_partial_per_op": python_bytes / tape.count,
        "live_output_storage_bytes": sum(tensors.values()),
        "dead_output_references": dead,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _allocations(mode: str) -> dict:
    model, sites, hidden = _build("qwen")
    ids = torch.randint(0, 1000, (1, 32))
    runner = _Runner(model, sites, hidden, mode)
    runner.call(ids)
    runner.latest = None
    gc.collect()
    tracemalloc.start()
    runner.call(ids)
    current, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    rows = []
    for stat in snapshot.statistics("filename")[:30]:
        filename = stat.traceback[0].filename
        if "/torchlens/" in filename:
            filename = "torchlens/" + filename.rsplit("/torchlens/", 1)[1]
        rows.append({"file": filename, "bytes": stat.size, "count": stat.count})
    tracemalloc.stop()
    return {
        "mode": mode,
        "python_retained_bytes": current,
        "python_peak_bytes": peak,
        "python_retained_bytes_per_full_trace_op": current / 718,
        "top_allocators": rows,
        "selected_bytes": runner.bytes,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--seq", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == "checks":
        result = _checks()
    elif args.mode.startswith("alloc_"):
        result = _allocations(args.mode.removeprefix("alloc_"))
    elif args.mode in ("weak", "strong"):
        result = _journal(args.mode == "strong")
    else:
        result = _memory(args.mode, args.batches, args.seq)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
