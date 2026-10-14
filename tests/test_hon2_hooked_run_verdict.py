"""A hooked ``run(inputs=)`` reads unverifiable and not pristine, never verified.

``trace.run(inputs=..., fast=True)`` applies the trace's staged edits at real module
boundaries, so its output is an edited forward, not the model's own. No hook-aware
check exists yet, so the run settles ``unverifiable`` with a typed
``user_intervention_not_replayable`` diagnostic (detection stage
``hooked_run_unverified``) and the trace is no longer pristine. The plain run keeps
its verdict, and the plain ``run(inputs=)`` keeps refusing staged edits it would not
apply.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._trace_state import TraceState
from torchlens.intervention.errors import EngineDispatchError
from torchlens.runnable import PathFaithfulness, RunnableErrorCode

_DIM = 8


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(_DIM, _DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()
        self.head = nn.Linear(_DIM, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(x))


def _setup() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _Net().eval(), torch.randn(_DIM)


def _x(seed: int) -> torch.Tensor:
    return torch.randn(1, 3, _DIM, generator=torch.Generator().manual_seed(seed))


def _steered(model: nn.Module, direction: torch.Tensor) -> tl.Trace:
    spec = tl.when(tl.module("block"), tl.steer(direction, magnitude=3.0, feature_axis=-1))
    return tl.trace(model, _x(0), save=tl.module("block") | tl.module("head"), intervene=spec)


def _report_card(report: object) -> tuple[object, ...]:
    return (
        report.path_faithfulness,  # type: ignore[attr-defined]
        report.first_mismatch,  # type: ignore[attr-defined]
        report.poisoned,  # type: ignore[attr-defined]
        report.contract_checks,  # type: ignore[attr-defined]
        report.numeric_attestation,  # type: ignore[attr-defined]
    )


def test_hooked_fast_run_reports_unverifiable_not_pristine() -> None:
    """The steered fast run settles unverifiable with the typed hooked-run diagnostic."""

    model, direction = _setup()
    trace = _steered(model, direction)
    result = trace.run(inputs=_x(1), fast=True)
    report = result.report
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert report.poisoned is True
    mismatch = report.first_mismatch
    assert mismatch is not None
    assert mismatch.code is RunnableErrorCode.USER_INTERVENTION_NOT_REPLAYABLE
    assert mismatch.detection_stage == "hooked_run_unverified"
    assert dict(mismatch.details)["pristine"] == "false"
    assert dict(mismatch.details)["staged_hooks"] == "1"
    assert trace.state is not TraceState.PRISTINE
    # A second hooked run stays unverifiable (the ceiling is monotonic).
    again = trace.run(inputs=_x(2), fast=True)
    assert again.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


def test_hooked_plain_run_inputs_still_refuses() -> None:
    """The ordinary live run(inputs=) still refuses staged edits it would not apply."""

    model, direction = _setup()
    trace = _steered(model, direction)
    with pytest.raises(EngineDispatchError) as refused:
        trace.run(inputs=_x(1))
    assert refused.value.fields["code"] == "run_staged_spec_unapplied"


@pytest.mark.parametrize("fast", [False, True], ids=["live", "fast"])
def test_plain_run_verdict_is_unchanged(fast: bool) -> None:
    """A trace with nothing staged keeps its verified verdict, run after run."""

    model, _ = _setup()
    trace = tl.trace(model, _x(0), save=tl.module("block") | tl.module("head"))
    first = trace.run(inputs=_x(1), fast=fast)
    second = trace.run(inputs=_x(2), fast=fast)
    for result in (first, second):
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert result.report.first_mismatch is None
        assert result.report.poisoned is False
    assert _report_card(first.report) == _report_card(second.report)


def test_cleared_hooks_on_a_fresh_trace_run_verified() -> None:
    """Detaching the staged edit before any hooked run leaves the plain verdict."""

    model, direction = _setup()
    trace = _steered(model, direction)
    trace.clear_hooks()
    result = trace.run(inputs=_x(1), fast=True)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
