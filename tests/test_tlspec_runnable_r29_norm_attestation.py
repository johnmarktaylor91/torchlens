"""Norm-layer attestation/mutation over-trigger relaxation (r29-C4).

Two over-triggers that penalised a VALUE-CORRECT, path-faithful norm-layer replay:

* codex-F3 (false DIVERGE -- the harmful one): ``InstanceNorm``/``BatchNorm`` with
  ``track_running_stats=True`` updates its running-stat buffers inside the functional
  ``instance_norm``/``batch_norm`` call, bumping the buffer ``_version`` without TorchLens
  recording a separate in-place op. The non-inplace mutation check flagged that version bump
  as a changed-input tensor and raised a false ``PathDivergenceError``. State buffer/param
  version bumps are now excluded from the non-inplace mutation check (value correctness stays
  enforced by state/output attestation).
* codex-F2 (false NOT_APPLICABLE): a read-only running stat under eval is read at several
  points, producing repeated same-state buffer members that were treated as journaled and
  suppressed activation attestation. A repeated buffer member is journaled only when its
  archived bytes DIFFER from the capture state (an actual mid-forward write); a stable
  read-only buffer stays byte-attestable.

The fail-closed side is preserved: a genuinely journaled (written) running stat under train
still reports NOT_APPLICABLE, never a false ATTESTED.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness


def _save(model: nn.Module, x: torch.Tensor, path: Path) -> Path:
    """Capture and save a runnable artifact with embedded weights + activations."""

    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    trace.save(path, level="runnable", include_activations=True, include_weights=True)
    return path


@pytest.mark.smoke
def test_r29_instance_norm_tracking_does_not_falsely_diverge(tmp_path: Path) -> None:
    """InstanceNorm with running stats must VERIFY, not raise a false mutation divergence."""

    for mode in ("eval", "train"):
        model = nn.InstanceNorm1d(4, track_running_stats=True)
        getattr(model, mode)()
        x = torch.randn(2, 4, 8)
        path = _save(model, x, tmp_path / f"in_{mode}.tlspec")

        result = tl.load(path).run(inputs=x.clone())
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert not result.report.poisoned
        # The replayed output is numerically correct (the fix does not mask a real divergence).
        fresh = nn.InstanceNorm1d(4, track_running_stats=True)
        fresh.load_state_dict(model.state_dict())
        getattr(fresh, mode)()
        torch.testing.assert_close(result.output, fresh(x.clone()), atol=1e-6, rtol=1e-6)


@pytest.mark.smoke
def test_r29_batchnorm_eval_read_only_stats_are_attested(tmp_path: Path) -> None:
    """BatchNorm eval (read-only running stats) must ATTEST, not falsely NOT_APPLICABLE."""

    model = nn.BatchNorm1d(4)
    model.eval()
    x = torch.randn(8, 4)
    path = _save(model, x, tmp_path / "bn_eval.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


@pytest.mark.smoke
def test_r29_instance_norm_eval_read_only_stats_are_attested(tmp_path: Path) -> None:
    """InstanceNorm eval with read-only running stats must ATTEST."""

    model = nn.InstanceNorm1d(4, track_running_stats=True)
    model.eval()
    x = torch.randn(2, 4, 8)
    path = _save(model, x, tmp_path / "in_eval.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


@pytest.mark.smoke
def test_r29_batchnorm_train_journaled_stats_stay_conservative(tmp_path: Path) -> None:
    """BatchNorm train (journaled running stats) stays NOT_APPLICABLE, never a false ATTESTED."""

    model = nn.BatchNorm1d(4)
    model.train()
    x = torch.randn(8, 4)
    path = _save(model, x, tmp_path / "bn_train.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
