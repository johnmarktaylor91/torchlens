"""``run(inputs=)`` applies a trace's staged intervention spec on the live provider.

The ordinary live door captures the new inputs under the same plan the intervened
rerun uses, hands back a fork that adopted that capture, and settles the fire
receipts (planned, fired, unfired) on the fork's ``last_run``: a run whose staged
entry fired nowhere discloses it under the zero-fire policy instead of returning a
silently un-intervened result. A hook staged downstream also fires when an
upstream ``do`` propagates through it.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import PathFaithfulness, RunnableErrorCode
from torchlens._selection.errors import SiteResolutionError
from torchlens._trace_state import TraceState
from torchlens.errors import TorchLensWarning

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


class _Gated(nn.Module):
    """A sequence-length gate: ``extra`` runs only for more than two positions."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(_DIM, _DIM)
        self.extra = nn.Linear(_DIM, _DIM)
        self.head = nn.Linear(_DIM, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.fc(x))
        if x.shape[1] > 2:
            hidden = hidden + self.extra(hidden)
        return self.head(hidden)


class _Chain(nn.Module):
    """fc1 -> relu -> fc2 -> tanh -> fc3, labelled linear_1_1 .. linear_3_5."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.tanh(self.fc2(torch.relu(self.fc1(x)))))


def _x(seed: int, positions: int = 3) -> torch.Tensor:
    return torch.randn(1, positions, _DIM, generator=torch.Generator().manual_seed(seed))


def _steered(model: nn.Module, direction: torch.Tensor) -> tl.Trace:
    spec = tl.when(tl.module("block"), tl.steer(direction, strength=3.0, dim=-1))
    return tl.trace(model, _x(0), save=tl.module("block") | tl.module("head"), intervene=spec)


def _steered_by_hand(model: _Net, direction: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.head(model.block(x) + direction.reshape(1, 1, -1) * 3.0)


def _fire_records(trace: Any) -> int:
    return sum(
        len(getattr(op, "fire_results", None) or ()) + len(getattr(op, "edits", None) or ())
        for op in trace._layer_list
    )


def _warning_codes(records: list[warnings.WarningMessage]) -> list[str]:
    codes = []
    for record in records:
        message = record.message
        code = getattr(message, "code", None) or (getattr(message, "fields", None) or {}).get(
            "code"
        )
        if code is not None:
            codes.append(str(code))
    return codes


def test_live_run_inputs_applies_the_staged_hook_with_receipts_on_the_fork() -> None:
    """The plain live door applies the steer; the fork carries the fire receipts."""

    torch.manual_seed(0)
    model, direction = _Net().eval(), torch.randn(_DIM)
    trace = _steered(model, direction)
    source_state = trace.state

    result = trace.run(inputs=_x(1))

    assert torch.allclose(result.output, _steered_by_hand(model, direction, _x(1)), atol=1e-6)
    assert result.trace is not trace
    ledger = result.trace.last_run
    assert ledger["engine"] == "live"
    assert ledger["hooks"] == 1
    assert ledger["hooks_fired"] >= 1
    assert ledger["hooks_unfired"] == 0
    assert ledger["unfired_plan_ids"] == []
    assert ledger["divergence_count"] == 0
    assert _fire_records(result.trace) >= 1
    assert result.trace.state is TraceState.RERUN_PROPAGATED
    # An edited forward is unverifiable, never verified, and names why.
    report = result.report
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.first_mismatch is not None
    assert report.first_mismatch.code is RunnableErrorCode.USER_INTERVENTION_NOT_REPLAYABLE
    assert dict(report.first_mismatch.details)["pristine"] == "false"
    # The source trace keeps its own state and spec.
    assert trace.state is source_state
    assert len(trace._intervention_spec.hook_specs) == 1


def test_live_run_inputs_repeats_on_fresh_inputs_and_after_a_fast_run() -> None:
    """Every plain run applies the spec, interleaved with the guarded fast engine."""

    torch.manual_seed(0)
    model, direction = _Net().eval(), torch.randn(_DIM)
    trace = _steered(model, direction)
    for seed in (1, 2):
        plain = trace.run(inputs=_x(seed))
        assert torch.allclose(plain.output, _steered_by_hand(model, direction, _x(seed)), atol=1e-6)
    fast = trace.run(inputs=_x(3), fast=True)
    assert torch.allclose(fast.output, _steered_by_hand(model, direction, _x(3)), atol=1e-6)
    again = trace.run(inputs=_x(4))
    assert torch.allclose(again.output, _steered_by_hand(model, direction, _x(4)), atol=1e-6)
    assert again.trace.last_run["engine"] == "live"


def test_live_run_inputs_discloses_a_staged_entry_that_fired_nowhere() -> None:
    """A gated site the new inputs skip leaves the entry unfired: warned, counted, diverged."""

    torch.manual_seed(0)
    model = _Gated().eval()
    spec = tl.when(tl.module("extra"), tl.zero_ablate())
    trace = tl.trace(model, _x(0, positions=3), intervene=spec)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = trace.run(inputs=_x(1, positions=2), on_divergence="return")
    assert "rerun_zero_fire" in _warning_codes(list(records))
    assert any(isinstance(record.message, TorchLensWarning) for record in records)
    ledger = result.trace.last_run
    assert ledger["hooks"] == 1
    assert ledger["hooks_fired"] == 0
    assert ledger["hooks_unfired"] == 1
    assert len(ledger["unfired_plan_ids"]) == 1
    # The skipped module changed the executed graph: the divergence settles typed.
    assert ledger["divergence_count"] == 1
    assert result.report.path_faithfulness is PathFaithfulness.DIVERGED

    with pytest.raises(SiteResolutionError) as refused:
        trace.run(inputs=_x(2, positions=2), on_divergence="return", on_zero_fire="error")
    assert refused.value.fields["code"] == "rerun_zero_fire"


def test_do_hook_downstream_fires_when_an_upstream_do_propagates() -> None:
    """A hook staged at tanh (L4) fires again when a value edit at relu (L2) propagates."""

    torch.manual_seed(0)
    model = _Chain().eval()
    x = torch.randn(3, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True, save_args=True))
    fork = trace.fork()

    fork.do("tanh_1_4", tl.scale(2.0))
    with torch.no_grad():
        doubled = model.fc3(torch.tanh(model.fc2(torch.relu(model.fc1(x)))) * 2.0)
    assert torch.allclose(fork["linear_3_5"].out, doubled, atol=1e-6)
    assert len(list(fork["tanh_1_4"].ops[0].edits or ())) >= 1

    zeros = torch.zeros_like(fork["relu_1_2"].out)
    fork.do("relu_1_2", zeros)
    with torch.no_grad():
        composed = model.fc3(torch.tanh(model.fc2(zeros)) * 2.0)
    assert torch.allclose(fork["linear_3_5"].out, composed, atol=1e-6)
    assert torch.allclose(fork["tanh_1_4"].out, torch.tanh(model.fc2(zeros)) * 2.0, atol=1e-6)
    # The propagated tanh op carries the hook's edit again: it fired on the new values.
    assert len(list(fork["tanh_1_4"].ops[0].edits or ())) >= 1
    # The source trace is untouched.
    with torch.no_grad():
        assert torch.allclose(trace["linear_3_5"].out, model(x), atol=1e-6)
