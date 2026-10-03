"""Persistence honesty: the outcome gate, recover(), and loaded-backward fixes.

WT1 A-IV item 21 (lane A08), three defects in one bundle:

* The N1-N5 capability chokepoint read only ``__dict__["_capture_outcome"]``,
  so every slots-backed ``Recording`` -- whose settled outcome lives in the
  ``_outcome`` slot behind the validating ``outcome`` property -- read as
  UNKNOWN with a false "hand-built object" warning. ``outcome_for`` now
  honors the sanctioned ``_OUTCOME_SELF_AUTHORITY`` marker.
* ``tl.fastlog.recover()`` rebuilt aborted bundles (PARTIAL sentinel +
  REASON.txt debris) as ``failed=False`` recordings with no error evidence --
  laundering the failure record. Recovery now carries the abort evidence and
  the derived outcome is FAILED; corruption salvage itself is unchanged.
* ``log_backward`` on a bundle-loaded Trace hooked whatever foreign autograd
  graph the loss came from, half-mutated the trace (backward passes +
  grad_fns from an unrelated forward), then died untyped -- after which
  ``draw_backward`` silently rendered the wrong graph. It now refuses typed
  BEFORE any mutation.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.capture.outcome import (
    CaptureOutcomeError,
    CaptureStatus,
    outcome_for,
    require_capture_capability,
)
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.options import CaptureOptions


class _Boom(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc(x))
        raise RuntimeError("mid-forward failure")


# --- gate learns Recording -------------------------------------------------


def test_gate_reads_settled_complete_recording_without_warning():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    rec = tl.record(model, torch.randn(2, 4), save=tl.func("relu"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        settled = outcome_for(rec)
        gate_outcome = require_capture_capability(rec, "save_analysis")
    assert settled is not None and settled.status is CaptureStatus.COMPLETE
    assert gate_outcome.status is CaptureStatus.COMPLETE
    hand_built = [w for w in caught if "hand-built" in str(w.message)]
    assert hand_built == []


def test_gate_refuses_failed_recording_with_its_real_status():
    failed = tl.record(
        _Boom(),
        torch.randn(2, 4),
        save=tl.func("relu"),
        on_forward_error="return_partial",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(CaptureOutcomeError) as excinfo:
            require_capture_capability(failed, "save_analysis")
    assert excinfo.value.fields["code"] == "N1"
    assert excinfo.value.fields["status"] == "failed"
    hand_built = [w for w in caught if "hand-built" in str(w.message)]
    assert hand_built == []


def test_hand_built_object_still_warns_unknown():
    class NotAProduct:
        pass

    with (
        pytest.warns(RuntimeWarning, match="hand-built"),
        pytest.raises(CaptureOutcomeError) as excinfo,
    ):
        require_capture_capability(NotAProduct(), "save_analysis")
    assert excinfo.value.fields["status"] == "unknown"


# --- recover() stops laundering ---------------------------------------------


@pytest.fixture()
def aborted_bundle(tmp_path):
    rec = tl.record(
        _Boom(),
        torch.randn(2, 4),
        save=tl.func("relu"),
        storage=tl.to_disk(str(tmp_path / "bundle")),
        on_forward_error="return_partial",
    )
    assert rec.failed and rec.bundle_path is not None
    assert (rec.bundle_path / "PARTIAL").exists()
    return rec.bundle_path


@pytest.mark.smoke
def test_recover_carries_abort_failure_evidence(aborted_bundle):
    recovered = tl.fastlog.recover(aborted_bundle)
    assert recovered.recovered is True
    assert recovered.failed is True
    assert "aborted" in (recovered.error_repr or "")
    assert recovered.outcome.status is CaptureStatus.FAILED
    assert recovered.outcome.recovered is True
    assert any("aborted mid-write" in w for w in recovered.recovery_warnings)


@pytest.mark.smoke
def test_recover_still_salvages_clean_crash_debris_as_unknown(tmp_path):
    """Corruption salvage unchanged: index debris WITHOUT abort evidence stays
    a non-failed recovered recording with a conservative UNKNOWN outcome."""

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    rec = tl.record(
        model, torch.randn(2, 4), save=tl.func("relu"), storage=tl.to_disk(str(tmp_path / "ok"))
    )
    bundle = rec.bundle_path
    assert bundle is not None
    # Simulate a hard crash: strip the finalized manifest so only the index
    # remains, with no PARTIAL/REASON abort debris.
    (bundle / "manifest.json").unlink()
    recovered = tl.fastlog.recover(bundle)
    assert recovered.recovered is True
    assert recovered.failed is False
    assert recovered.outcome.status is CaptureStatus.UNKNOWN


# --- loaded log_backward refuses before mutating -----------------------------


def test_loaded_log_backward_refuses_before_mutation(tmp_path):
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(backward_ready=True))
    tl.save(trace, tmp_path / "bundle")
    loaded = tl.load(tmp_path / "bundle")
    live_loss = model(x.requires_grad_(True)).sum()
    with pytest.raises(RunCapabilityUnavailableError, match="bundle-loaded"):
        loaded.log_backward(live_loss)
    assert len(getattr(loaded, "backward_passes", []) or []) == 0
    assert len(loaded.grad_fns) == 0


def test_loaded_recording_backward_refuses(tmp_path):
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    x = torch.randn(2, 4)
    trace = tl.trace(model, x)
    tl.save(trace, tmp_path / "bundle")
    loaded = tl.load(tmp_path / "bundle")
    with pytest.raises(RunCapabilityUnavailableError, match="bundle-loaded"):
        loaded.recording_backward()


def test_live_log_backward_still_works():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(backward_ready=True))
    loss = trace["relu_1_2"].out.sum()
    trace.log_backward(loss)
    assert len(trace.backward_passes) == 1
