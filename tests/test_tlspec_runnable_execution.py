"""Stage 5 sparse DAG execution and unified ``RunResult`` tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._runnable_state import prepare_runnable_state
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, RunProvider, RunResult, StateSource


class RunnableExecutionModel(nn.Module):
    """Small parameterized graph with a persistent buffer."""

    def __init__(self) -> None:
        """Initialize deterministic state-bearing layers."""

        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("scale", torch.tensor([1.5, -0.5]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the recorded static path."""

        return torch.relu(self.linear(value)) * self.scale


@pytest.fixture(scope="module")
def runnable_execution_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, RunnableExecutionModel, tl.Trace]:
    """Build one reusable sparse artifact and its independent live oracle."""

    torch.manual_seed(11)
    model = RunnableExecutionModel().eval()
    trace = tl.trace(
        model,
        torch.ones(2, 3),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path_factory.mktemp("runnable-execution") / "model.tlspec"
    trace.save(path, level="runnable")
    return path, model, trace


@pytest.mark.smoke
def test_loaded_sparse_run_with_user_state_matches_live_values_and_is_transactional(
    runnable_execution_artifact: tuple[Path, RunnableExecutionModel, tl.Trace],
) -> None:
    """Execute new inputs with staged real state and leave the source unchanged."""

    path, model, _ = runnable_execution_artifact
    loaded = tl.load(path)
    loaded.load_state_dict(model.state_dict())
    source_outs = _clone_outs(loaded)
    inputs = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 0.25, 4.0]])

    result = loaded.run(inputs=inputs, seed=73)

    assert isinstance(result, RunResult)
    assert torch.equal(result.output, model(inputs))
    assert result.trace is not loaded
    _assert_outs_equal(loaded, source_outs)
    assert result.report.readiness.provider is RunProvider.LOADED_SPARSE
    assert result.report.state_source is StateSource.USER_STATE_DICT
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.contract_checks
    assert all(check.passed for check in result.report.contract_checks)


def test_loaded_sparse_random_state_runs_have_correct_shape_and_seed_determinism(
    runnable_execution_artifact: tuple[Path, RunnableExecutionModel, tl.Trace],
) -> None:
    """Report every N1-a slot and reproduce state/runtime randomness by seed."""

    path, _, _ = runnable_execution_artifact
    inputs = torch.randn(2, 3)
    first = tl.load(path).run(inputs=inputs, seed=101)
    second = tl.load(path).run(inputs=inputs, seed=101)
    different_state = prepare_runnable_state(tl.load(path), seed=102)

    assert first.output.shape == (2, 2)
    assert torch.equal(first.output, second.output)
    first_state = prepare_runnable_state(tl.load(path), seed=101)
    assert any(
        not torch.equal(first_state.slot_values[slot_id], different_state.slot_values[slot_id])
        for slot_id in first_state.random_filled_slot_ids
    )
    assert first.report.state_source is StateSource.RANDOM_INITIALIZATION
    assert first.report.seed == 101
    assert first.report.random_filled_slot_ids


def test_loaded_sparse_execution_pauses_recursive_capture(
    runnable_execution_artifact: tuple[Path, RunnableExecutionModel, tl.Trace],
) -> None:
    """Invoke every resolved callable while the persistent wrapper gate is paused."""

    path, model, _ = runnable_execution_artifact
    loaded = tl.load(path)
    loaded.load_state_dict(model.state_dict())
    attached = loaded.__dict__["_runnable_callables_by_call_id"]
    observed: list[bool] = []
    for call_id, original in tuple(attached.items()):
        attached[call_id] = _logging_probe(original, observed)

    loaded.run(inputs=torch.ones(2, 3), seed=5)

    assert observed
    assert not any(observed)


def _logging_probe(func: Any, observed: list[bool]) -> Any:
    """Wrap one resolved callable and record the logging toggle at invocation."""

    def probe(*args: Any, **kwargs: Any) -> Any:
        """Record the toggle and forward one sparse call."""

        observed.append(_state._logging_enabled)
        return func(*args, **kwargs)

    return probe


@pytest.mark.smoke
def test_live_run_returns_unified_result_and_matches_save_new_outs_exactly(
    runnable_execution_artifact: tuple[Path, RunnableExecutionModel, tl.Trace],
) -> None:
    """Delegate live execution to the unchanged fast refresh projector."""

    _, model, trace = runnable_execution_artifact
    inputs = torch.tensor([[0.25, 0.5, -1.0], [3.0, -2.0, 1.0]])
    expected = trace.fork(name="expected-save-new-outs")
    expected.save_new_outs(model, inputs, random_seed=37)
    source_outs = _clone_outs(trace)

    result = trace.run(inputs=inputs, seed=37)

    assert isinstance(result, RunResult)
    assert result.report.readiness.provider is RunProvider.LIVE
    assert result.report.state_source is StateSource.LIVE_MODEL_STATE
    assert torch.equal(result.output, model(inputs))
    _assert_outs_equal(trace, source_outs)
    for actual_op, expected_op in zip(result.trace.layer_list, expected.layer_list):
        if expected_op.out is None:
            assert actual_op.out is None
        else:
            assert torch.equal(actual_op.out, expected_op.out)


def test_analysis_only_loaded_trace_raises_typed_capability_error(tmp_path: Path) -> None:
    """Refuse analysis bundles without heuristic runnable promotion."""

    model = RunnableExecutionModel().eval()
    trace = tl.trace(model, torch.ones(2, 3))
    path = tmp_path / "analysis.tlspec"
    trace.save(path)
    loaded = tl.load(path)

    with pytest.raises(RunCapabilityUnavailableError) as captured:
        loaded.run(inputs=torch.ones(2, 3))

    assert captured.value.fields["code"] == "run_capability_unavailable"
    assert captured.value.fields["readiness"] is loaded.readiness


def _clone_outs(trace: tl.Trace) -> tuple[torch.Tensor | None, ...]:
    """Clone one Trace's current activation payloads for mutation assertions."""

    return tuple(None if op.out is None else op.out.detach().clone() for op in trace.layer_list)


def _assert_outs_equal(trace: tl.Trace, expected: tuple[torch.Tensor | None, ...]) -> None:
    """Assert that a Trace retains exactly the snapshotted activation values."""

    for op, value in zip(trace.layer_list, expected):
        if value is None:
            assert op.out is None
        else:
            assert torch.equal(op.out, value)
