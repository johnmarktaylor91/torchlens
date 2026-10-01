"""A09 memory-shaped acceptance for the scoped non-attaching payload reader.

Record-shaped tests alone are exactly how the silent-None/attach class of
bug shipped (agent memo P0 part 5); this is the memory-shaped half: retained
RSS stays flat across 200 sequential scoped payload reads against a cached
lazily-loaded trace. Heavy tier: it deliberately builds a ~72 MB artifact.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.payload_reader import read_op_payload

pytestmark = pytest.mark.heavy


def _resident_bytes() -> int:
    """Return this process's resident set size in bytes (Linux)."""

    fields = Path("/proc/self/statm").read_text().split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def test_memory_shaped_scoped_reader_rss_stays_flat(tmp_path: Path) -> None:
    """200 sequential scoped reads hold ONE payload at a time, RSS flat.

    The attach hazard this pins: reading every site through the interactive
    ``op.out`` door caches each payload on the op, so repeated whole-artifact
    scans grow the resident set by the artifact's total payload bytes
    (~72 MB here). The scoped reader must stay within an allocator-aware
    tolerance instead.
    """

    torch.manual_seed(0)
    # 8 relu sites x (4 x 512 x 1024 float32 = 8 MB) >= 64 MB saved payloads.
    layers: list[nn.Module] = []
    for _ in range(8):
        layers.extend([nn.Linear(1024, 1024, bias=False), nn.ReLU()])
    model = nn.Sequential(*layers)
    trace = tl.trace(model, torch.randn(4, 512, 1024), save=tl.func("relu"))
    destination = tmp_path / "big.tlspec"
    tl.save(trace, destination)
    del trace

    loaded = tl.load(destination, lazy=True)
    assert loaded.payload_load_status == "loaded_lazy"
    saved_ops = [op for op in loaded.layer_list if getattr(op, "has_saved_activation", False)]
    assert len(saved_ops) >= 8
    total_payload_bytes = 8 * 4 * 512 * 1024 * 4

    # Warm one read so allocator arenas/import costs are paid before baseline.
    checksum = float(read_op_payload(saved_ops[0]).sum())
    baseline = _resident_bytes()
    rounds = max(1, 200 // len(saved_ops) + 1)
    for _ in range(rounds):  # >= 200 sequential reads across the sites
        for op in saved_ops:
            payload = read_op_payload(op)
            checksum += float(payload.sum())
            del payload
    growth = _resident_bytes() - baseline
    assert checksum == checksum  # keep the reads observable
    # Allocator-aware tolerance: half the artifact's payload bytes. An
    # attaching reader retains the full ~72 MB and fails; a leaking one grows
    # without bound; the scoped reader peaks at ONE 8 MB payload.
    assert growth < total_payload_bytes // 2, f"RSS grew {growth / 1e6:.1f} MB across scoped reads"
    # And the payloads did NOT attach: rows are still lazy.
    still_lazy = [op for op in saved_ops if getattr(op, "_slot")("out") is None]
    assert len(still_lazy) == len(saved_ops)
