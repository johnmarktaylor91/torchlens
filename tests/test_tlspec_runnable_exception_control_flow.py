"""r35 hon2_1 (I2): exception-driven control flow can never be a false VERIFIED.

A taken path decided by whether an op RAISED (Cholesky-with-jitter, robust
inversion, safe-log guards) is invisible to every tensor witness. The
event-lifecycle ledger records the caught raise at capture, the producer
downgrades witness completeness, and EVERY run of the artifact -- original or
changed input -- reports ``unverifiable`` + ``not_applicable``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch.completeness_witness import runnable_ledger_facts
from torchlens.errors import RuntimeSignatureDriftError
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    WitnessCompleteness,
)

pytestmark = pytest.mark.smoke


def _capture(model: nn.Module, x: Any) -> Any:
    """Capture an intervention-ready trace."""

    return tl.trace(
        model,
        x,
        capture=CaptureOptions(intervention_ready=True, cache=False),
    )


def _pd_input() -> torch.Tensor:
    """Positive-definite 2x2 matrix (success branch)."""

    return torch.tensor([[2.0, 0.0], [0.0, 3.0]])


def _non_pd_input() -> torch.Tensor:
    """Non-positive-definite, SINGULAR 2x2 matrix (raising branch for both
    the Cholesky and the inversion fallback models)."""

    return torch.tensor([[1.0, 1.0], [1.0, 1.0]])


class _CholeskyFallbackModel(nn.Module):
    """Numerical-stability idiom: try Cholesky, fall back on failure."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            chol = torch.linalg.cholesky(x)
            return (chol @ chol.transpose(-1, -2)).sum(dim=-1)
        except Exception:
            return (x * 2.0).sum(dim=-1)


class _InverseFallbackModel(nn.Module):
    """Second vector: singular-matrix inversion fallback."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return torch.linalg.inv(x).sum(dim=-1)
        except Exception:
            return (x * 2.0).sum(dim=-1)


class _SuppressFallbackModel(nn.Module):
    """``contextlib.suppress`` variant of the same class."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = (x * 2.0).sum(dim=-1)
        with contextlib.suppress(Exception):
            chol = torch.linalg.cholesky(x)
            result = (chol @ chol.transpose(-1, -2)).sum(dim=-1)
        return result


class _CaughtMutatingFailureModel(nn.Module):
    """A mutating op that raises and is caught (mutation-capable unknown)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 1.0
        try:
            y.add_(torch.ones(97))  # shape mismatch raises
        except Exception:
            pass
        return y + 1


@pytest.mark.parametrize(
    "model_cls",
    [_CholeskyFallbackModel, _InverseFallbackModel, _SuppressFallbackModel],
)
def test_capture_on_fallback_never_verifies(tmp_path: Path, model_cls: type[nn.Module]) -> None:
    """Capture on the raising branch: every run is unverifiable, never VERIFIED."""

    model = model_cls().eval()
    x = _non_pd_input()
    trace = _capture(model, x)
    facts = runnable_ledger_facts(trace)
    assert any(fact["kind"] == "caught_exception_control" for fact in facts)
    path = tmp_path / "fallback.tlspec"
    tl.save(trace, str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))
    descriptor = loaded.__dict__["_runnable_descriptor"]
    assert descriptor.witness_completeness is not WitnessCompleteness.COMPLETE

    for runtime_input in (_non_pd_input(), _pd_input()):
        result = loaded.run(inputs=runtime_input)
        assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
        assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_capture_on_success_raising_runtime_input_stays_typed(tmp_path: Path) -> None:
    """Direction asymmetry pinned: replaying the recorded op re-raises typed."""

    model = _CholeskyFallbackModel().eval()
    trace = _capture(model, _pd_input())
    path = tmp_path / "success.tlspec"
    tl.save(trace, str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))
    with pytest.raises(RuntimeSignatureDriftError):
        loaded.run(inputs=_non_pd_input())


def test_raise_free_model_records_zero_facts_and_stays_verified(tmp_path: Path) -> None:
    """No over-trigger: an ordinary model records no facts and verifies."""

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU()).eval()
    x = torch.randn(2, 4)
    trace = _capture(model, x)
    assert runnable_ledger_facts(trace) == ()
    path = tmp_path / "clean.tlspec"
    tl.save(trace, str(path), level="runnable", include_weights=True)
    result = tl.load(str(path)).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_caught_mutating_failure_downgrades_opaque(tmp_path: Path) -> None:
    """A caught raise from a mutating op is a mutation-capable incomplete fact."""

    model = _CaughtMutatingFailureModel().eval()
    x = torch.ones(3)
    trace = _capture(model, x)
    facts = runnable_ledger_facts(trace)
    assert any(fact["kind"] == "caught_exception_control" and fact["mutates"] for fact in facts)
    path = tmp_path / "mutating.tlspec"
    tl.save(trace, str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))
    descriptor = loaded.__dict__["_runnable_descriptor"]
    assert descriptor.witness_completeness is WitnessCompleteness.INCOMPLETE_OPAQUE_SIDE_EFFECT
    result = loaded.run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


class _RawAtenHostReturnModel(nn.Module):
    """Successful unmodeled host-returning aten call (ledger is not exception-only)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A raw, unowned aten dispatch returning a host value: no wrapper owner,
        # not in the exact escape net, no exception involved.
        same = torch.ops.aten.is_same_size(x, x + 1)
        del same
        return x + 2


def test_successful_unmodeled_host_return_also_downgrades(tmp_path: Path) -> None:
    """The ledger discharges by outcome, not exception-only: host returns count."""

    model = _RawAtenHostReturnModel().eval()
    x = torch.ones(3)
    trace = _capture(model, x)
    facts = runnable_ledger_facts(trace)
    assert any(fact["kind"] == "unmodeled_host_return" for fact in facts)
    path = tmp_path / "hostreturn.tlspec"
    tl.save(trace, str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))
    descriptor = loaded.__dict__["_runnable_descriptor"]
    assert descriptor.witness_completeness is not WitnessCompleteness.COMPLETE
    result = loaded.run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
