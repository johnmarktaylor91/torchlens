"""Closed-form oracles for the summary's two activation-payload byte totals.

``SummaryTotals.at_capture_bytes`` is the bytes of every activation the capture
saved; ``SummaryTotals.retained_now_bytes`` the bytes of those whose payload is
still here (resident or lazily loadable). Expected values are derived from the
fixture's shapes in the comments, never read from TorchLens. These are the I2
oracles the census (``tests/oracles/data/legacy_quantities.tsv``) names for
those doors.
"""

from __future__ import annotations

import torch

import torchlens as tl
from tests.oracles.test_legacy_graph_counts import Grad

FP32 = 4
# Grad on a batch of 2, everything saved: input (2, 4), linear a (2, 8),
# relu (2, 8), linear b (2, 2), output (2, 2).
SAVED_BYTES = (2 * 4 + 2 * 8 + 2 * 8 + 2 * 2 + 2 * 2) * FP32  # 192


def test_payload_byte_totals_at_capture_and_now(tmp_path) -> None:
    torch.manual_seed(0)
    trace = tl.trace(Grad().eval(), torch.randn(2, 4))
    try:
        totals = trace.summary().totals
        assert totals.at_capture_bytes == SAVED_BYTES
        assert totals.retained_now_bytes == SAVED_BYTES  # every payload is resident
        path = tmp_path / "stripped.tlspec"
        tl.save(trace, path, include_outs=False)
    finally:
        trace.cleanup()
    loaded = tl.load(path)
    totals = loaded.summary().totals
    assert totals.at_capture_bytes == SAVED_BYTES  # saved at capture
    assert totals.retained_now_bytes == 0  # no payload travelled with the artifact
