"""Persistence honesty: streamed bundles enter settlement.

WT1 A-IV item 18 (lane A08): ``tl.trace(storage=tl.to_disk(...))`` finalized
AND published its bundle at postprocess step 18 -- BEFORE the capture settled
-- so (a) a live COMPLETE capture's artifact carried no outcome attestation
and loaded UNATTESTED, and (b) a postprocess failure after step 18 left a
fully published artifact the N1 export gate would have refused. Step 18 now
STAGES the bundle; the tmp->final publish happens at the settlement seam with
the settled ``_capture_outcome`` payload injected (the same key and codec
ordinary saves persist), and a capture that fails after staging aborts the
writer (PARTIAL debris, never a published artifact).
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens.capture.outcome import CaptureStatus
from torchlens.data_classes.trace import Trace

pytestmark = pytest.mark.smoke


def _model():
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU())


def test_streamed_complete_capture_loads_complete(tmp_path):
    path = tmp_path / "streamed.tlspec"
    trace = tl.trace(_model(), torch.randn(2, 4), storage=tl.to_disk(str(path)))
    assert trace.outcome.status is CaptureStatus.COMPLETE
    loaded = tl.load(path)
    assert loaded.outcome.status is CaptureStatus.COMPLETE


def test_streamed_synchronous_writes_load_complete(tmp_path):
    path = tmp_path / "sync.tlspec"
    trace = tl.trace(
        _model(),
        torch.randn(2, 4),
        storage=tl.to_disk(str(path), async_writes=False),
    )
    assert trace.outcome.status is CaptureStatus.COMPLETE
    assert tl.load(path).outcome.status is CaptureStatus.COMPLETE


def test_streamed_payloads_still_readable_after_publish(tmp_path):
    path = tmp_path / "payloads.tlspec"
    trace = tl.trace(_model(), torch.randn(2, 4), storage=tl.to_disk(str(path)))
    # Live streamed traces evict in-memory outs (historical behavior); the
    # attached lazy ref must point at the PUBLISHED bundle.
    ref = trace["relu_1_2"].ops[0].out_ref
    assert ref is not None and ref.source_bundle_path == path
    loaded = tl.load(path)
    assert loaded["relu_1_2"].out.shape == (2, 3)


def test_streamed_halted_capture_loads_halted(tmp_path):
    """A halted streamed capture publishes at settlement with the HALTED
    attestation (previously it loaded UNATTESTED like every streamed bundle)."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 3), nn.ReLU())
    path = tmp_path / "halted.tlspec"
    trace = tl.trace(
        model,
        torch.randn(2, 4),
        halt=tl.func("linear"),
        storage=tl.to_disk(str(path)),
    )
    assert trace.outcome.status is CaptureStatus.HALTED
    loaded = tl.load(path)
    assert loaded.outcome.status is CaptureStatus.HALTED


def test_post_stage_failure_never_publishes(tmp_path):
    """A failure AFTER step 18 (here: step 20) must not leave a published
    artifact; the staged temp bundle is aborted into PARTIAL debris."""

    path = tmp_path / "failed.tlspec"
    with mock.patch.object(
        Trace, "release_param_refs", side_effect=RuntimeError("boom in step 20")
    ):
        # The streaming failure handler wraps the propagating step-20 error.
        with pytest.raises(TorchLensIOError):
            tl.trace(_model(), torch.randn(2, 4), storage=tl.to_disk(str(path)))
    assert not path.exists()
    debris = [p for p in tmp_path.iterdir() if p.name.startswith("failed.tlspec.tmp.")]
    assert debris, "expected the staged temp bundle to remain as sweepable debris"
    assert (debris[0] / "PARTIAL").exists()


def test_forward_failure_never_publishes(tmp_path):
    class Boom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc(x))
            raise RuntimeError("mid-forward failure")

    path = tmp_path / "fwd_failed.tlspec"
    # The streaming failure handler wraps the forward error (historical
    # streamed-capture behavior); the original rides the exception chain.
    with pytest.raises(TorchLensIOError) as excinfo:
        tl.trace(Boom(), torch.randn(2, 4), storage=tl.to_disk(str(path)))
    assert "mid-forward failure" in str(excinfo.value.__cause__)
    assert not path.exists()
