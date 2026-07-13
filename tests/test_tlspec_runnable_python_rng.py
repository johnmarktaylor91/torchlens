"""Honesty regression: Python/NumPy-RNG control flow must never be a false VERIFIED.

A sparse runnable trace records exactly one taken branch. When the traced forward
chose that branch from Python ``random`` or NumPy RNG (an unwitnessed, tensor-free
predicate), a run under a *different* seed than capture cannot be blessed as
VERIFIED/ATTESTED -- the recorded path may not be the branch a fresh seeded call
takes. The honest ceiling is UNVERIFIABLE + NOT_APPLICABLE. Only a run that
reproduces the captured seed is an honest VERIFIED replay. Deterministic models
(no host RNG) are unchanged.

Incident (round 7): capture seed=1 picked ``x * 10``; ``run(seed=2)`` returned the
stale ``20`` while a fresh live call took the other branch (``12``), yet the report
claimed VERIFIED + ATTESTED. The runnable path reseeded only torch and never
recorded that the forward consumed host RNG.
"""

from __future__ import annotations

from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import build_sparse_run_descriptor
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness


class _PythonRandomBranch(nn.Module):
    """Branch on Python ``random`` with no tensor-level predicate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a tensor op from Python RNG (invisible to control witnesses)."""
        if random.random() > 0.5:
            return x + 10
        return x * 10


class _NumpyRandomBranch(nn.Module):
    """Branch on NumPy RNG with no tensor-level predicate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pick a tensor op from NumPy RNG (invisible to control witnesses)."""
        if np.random.random() > 0.5:
            return x + 10
        return x * 10


class _Deterministic(nn.Module):
    """A host-RNG-free model that must stay VERIFIED for any seed."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a deterministic affine transform."""
        return x * 3 + 1


def _capture(model: nn.Module, x: torch.Tensor, *, seed: int) -> tl.Trace:
    """Capture a runnable-ready trace under a fixed seed."""
    return tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
            random_seed=seed,
        ),
    )


def _roundtrip_run(
    model: nn.Module, x: torch.Tensor, *, capture_seed: int, run_seed: int | None, tmp: Path
) -> tl.RunResult:
    """Capture, save runnable, reload, and run under ``run_seed``."""
    trace = _capture(model, x, seed=capture_seed)
    path = tmp / "rng.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_activations=True)
    return tl.load(path).run(inputs=x, seed=run_seed)


def test_python_rng_profile_detects_host_consumption() -> None:
    """The descriptor records host-RNG consumption and the capture seed."""
    random.seed(1)
    trace = _capture(_PythonRandomBranch(), torch.tensor([2.0]), seed=1)
    profile = build_sparse_run_descriptor(trace).rng_profile
    assert profile.host_rng_consumed is True
    assert profile.capture_seed == 1


def test_numpy_rng_profile_detects_host_consumption() -> None:
    """NumPy RNG control flow is detected as host-RNG consumption too."""
    np.random.seed(1)
    trace = _capture(_NumpyRandomBranch(), torch.tensor([2.0]), seed=1)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is True


def test_deterministic_model_reports_no_host_rng() -> None:
    """A deterministic forward must not be flagged as consuming host RNG."""
    trace = _capture(_Deterministic(), torch.tensor([2.0]), seed=1)
    assert build_sparse_run_descriptor(trace).rng_profile.host_rng_consumed is False


def test_changed_seed_python_rng_is_unverifiable_not_attested(tmp_path: Path) -> None:
    """The money bug: off-seed host-RNG replay must never be VERIFIED + ATTESTED."""
    random.seed(1)
    result = _roundtrip_run(
        _PythonRandomBranch(), torch.tensor([2.0]), capture_seed=1, run_seed=2, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_changed_seed_numpy_rng_is_unverifiable_not_attested(tmp_path: Path) -> None:
    """NumPy host-RNG control flow shares the off-seed honesty ceiling."""
    np.random.seed(1)
    result = _roundtrip_run(
        _NumpyRandomBranch(), torch.tensor([2.0]), capture_seed=1, run_seed=2, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_none_seed_python_rng_is_unverifiable(tmp_path: Path) -> None:
    """A seedless run of a host-RNG capture cannot claim the recorded branch."""
    random.seed(1)
    result = _roundtrip_run(
        _PythonRandomBranch(), torch.tensor([2.0]), capture_seed=1, run_seed=None, tmp=tmp_path
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_same_seed_python_rng_is_verified_and_attested(tmp_path: Path) -> None:
    """Reproducing the captured seed is an honest VERIFIED + ATTESTED replay."""
    random.seed(1)
    result = _roundtrip_run(
        _PythonRandomBranch(), torch.tensor([2.0]), capture_seed=1, run_seed=1, tmp=tmp_path
    )
    assert result.output.tolist() == [20.0]
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


@pytest.mark.parametrize("run_seed", [None, 2, 7])
def test_deterministic_model_stays_verified(run_seed: int | None, tmp_path: Path) -> None:
    """A host-RNG-free model is unchanged: VERIFIED + ATTESTED for any run seed."""
    result = _roundtrip_run(
        _Deterministic(), torch.tensor([2.0]), capture_seed=1, run_seed=run_seed, tmp=tmp_path
    )
    assert result.output.tolist() == [7.0]
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
