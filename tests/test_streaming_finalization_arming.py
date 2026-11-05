"""Liveness killers for the streamed-bundle finalization step (executor step 18).

r7 R74 executor-family survivor, found by the family's own first sample
campaign: forcing ``_should_run_finalize_streamed_bundle`` to ``False`` (the streamed bundle
never finalizes) survived the ENTIRE 472-test arming suite -- no suite file
captured with ``storage=tl.to_disk(...)`` and then relied on the finalized
bundle. These plants close that margin: skipping finalization (or the
finalizer body) must fail here, in an ordinary pytest run, with no driver.

This file is enrolled in the mutation driver's ``SUITE`` list; removing it
from there resurrects the proven survivor.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


def _capture_streamed(tmp_path):
    """Stream a small relu capture to disk and return the bundle path."""

    path = tmp_path / "streamed.tlspec"
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    trace = tl.trace(model, torch.randn(2, 4), save=tl.func("relu"), storage=tl.to_disk(str(path)))
    return path, trace


def test_streamed_bundle_saved_payload_reads_back(tmp_path) -> None:
    """A saved activation streamed to disk reads back numerically intact."""

    torch.manual_seed(0)
    path, trace = _capture_streamed(tmp_path)
    live = trace["relu_1_2"].out
    loaded = tl.load(str(path))
    reread = loaded["relu_1_2"].out
    assert reread is not None, "streamed saved payload lost"
    assert torch.equal(reread, live)
