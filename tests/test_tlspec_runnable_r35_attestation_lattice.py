"""r35 corr2_7 (I3): numeric attestation is downstream of the settled path verdict.

``attested`` implies ``verified`` and unpoisoned -- structurally. Any failed
contract check or ceiling caps attestation at ``not_applicable`` before a single
archive byte is read, and the report constructor makes the contradictory
combination unrepresentable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
)

pytestmark = pytest.mark.smoke


class _AddOneModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class _FlagModel(nn.Module):
    """Model whose non-tensor input leaf steers the computation."""

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        return x * scale + 1


def _save(model: nn.Module, args: Any, path: Path) -> Path:
    trace = tl.trace(
        model,
        args,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    trace.save(path, level="runnable", include_weights=True, include_activations=True)
    return path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_r35_device_diverged_run_is_never_attested(tmp_path: Path) -> None:
    """corr2_7 repro: a device-diverged poisoned run must be not_applicable."""

    x = torch.tensor([1.0, 2.0, 3.0])
    path = _save(_AddOneModel().eval(), x, tmp_path / "dev.tlspec")
    result = tl.load(path).run(inputs=x.cuda(), on_divergence=DivergencePolicy.RETURN_DIVERGED)
    assert result.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert result.report.poisoned is True
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_r35_any_failed_input_contract_caps_attestation(tmp_path: Path) -> None:
    """CPU equivalent: a failed input-literal check caps attestation."""

    x = torch.tensor([1.0, 2.0, 3.0])
    path = _save(_FlagModel().eval(), (x, 2.0), tmp_path / "flag.tlspec")
    result = tl.load(path).run(
        inputs=(x, 3.0),
        on_divergence=DivergencePolicy.RETURN_DIVERGED,
    )
    assert result.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert result.report.poisoned is True


def test_r35_eligible_original_run_stays_attested(tmp_path: Path) -> None:
    """Positive control: the lattice does not widen not_applicable needlessly."""

    x = torch.tensor([1.0, 2.0, 3.0])
    path = _save(_AddOneModel().eval(), x, tmp_path / "ok.tlspec")
    result = tl.load(path).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert result.report.poisoned is False


def test_r35_report_constructor_rejects_attested_nonverified() -> None:
    """The prohibited ATTESTED + non-VERIFIED combination is unrepresentable."""

    from torchlens._runnable_execution import _run_report
    from torchlens._runnable_state import PreparedRunnableState
    from types import MappingProxyType

    from torchlens.runnable import (
        ReadinessReport,
        ReadinessStatus,
        RunProvider,
        StateSource,
        WitnessCompleteness,
    )

    readiness = ReadinessReport(
        status=ReadinessStatus.READY,
        provider=RunProvider.LOADED_SPARSE,
        backend="torch",
        capability="sparse_recorded_taken_path_v2",
        resolver_records=(),
        state_sources_available=(StateSource.RANDOM_INITIALIZATION,),
        witness_completeness=WitnessCompleteness.COMPLETE,
        diagnostics=(),
    )
    state = PreparedRunnableState(
        slot_values=MappingProxyType({}),
        state_source=StateSource.RANDOM_INITIALIZATION,
        initializer_policy_version=None,
        seed=None,
        random_filled_slot_ids=(),
    )
    for verdict in (PathFaithfulness.DIVERGED, PathFaithfulness.UNVERIFIABLE):
        with pytest.raises(RuntimeError, match="attested"):
            _run_report(
                readiness,
                state,
                contract_checks=(),
                path_faithfulness=verdict,
                first_mismatch=None,
                numeric_attestation=NumericAttestationStatus.ATTESTED,
            )


def test_r35_prior_poisoned_fork_mark_caps_attestation(tmp_path: Path) -> None:
    """An inherited monotonic non-verified mark makes attestation ineligible."""

    x = torch.tensor([1.0, 2.0, 3.0])
    path = _save(_AddOneModel().eval(), x, tmp_path / "mono.tlspec")
    loaded = tl.load(path)
    # Simulate an inherited monotonic poisoned mark on the source Trace: forks
    # copy it, so the eligibility verdict must fold it in.
    loaded.__dict__["_runnable_path_faithfulness"] = PathFaithfulness.UNVERIFIABLE
    loaded.__dict__["_runnable_poisoned"] = True
    result = loaded.run(inputs=x)
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
