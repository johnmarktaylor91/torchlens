"""Stage 5 sparse DAG execution and unified ``RunResult`` tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._runnable_state import prepare_runnable_state
from torchlens.errors import (
    PathDivergenceError,
    PoisonedRunError,
    RunCapabilityUnavailableError,
)
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
    RunProvider,
    RunResult,
    StateSource,
    WitnessCompleteness,
)


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


class HonestyControlModel(nn.Module):
    """Same-shape model with observable loop and conditional witnesses."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Execute one recorded straight-line control-flow schedule."""

        while value.sum() < 0:
            value = value + 1
        if value.sum() > 0:
            value = value * 2
        return value


class InplaceActivationModel(nn.Module):
    """Graph whose later in-place call must not rewrite staged activations."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Mutate a ReLU result only after another result has consumed it."""

        activated = torch.relu(value)
        preserved = activated + 1
        activated.mul_(0)
        return preserved + activated


class RunnableNamedOutput(NamedTuple):
    """Named output container used to verify portable kind reconstruction."""

    primary: torch.Tensor
    score: float
    activated: torch.Tensor


class MixedTupleOutputModel(nn.Module):
    """Model returning tensor leaves around a non-tensor literal."""

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor]:
        """Return a mixed tuple whose literal must survive sparse execution."""

        shifted = value + 1
        return shifted, 3.0, torch.relu(shifted)


class ListOutputModel(nn.Module):
    """Model returning a list rather than a tuple."""

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        """Return two tensor leaves in a list container."""

        shifted = value + 1
        return [shifted, torch.relu(shifted)]


class NamedTupleOutputModel(nn.Module):
    """Model returning a namedtuple with a literal field."""

    def forward(self, value: torch.Tensor) -> RunnableNamedOutput:
        """Return a namedtuple whose type and literal field must survive."""

        shifted = value + 1
        return RunnableNamedOutput(shifted, 3.0, torch.relu(shifted))


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


@pytest.fixture(scope="module")
def honesty_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a sparse artifact carrying complete control witnesses."""

    trace = tl.trace(
        HonestyControlModel(),
        torch.ones(2),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path_factory.mktemp("runnable-honesty") / "control.tlspec"
    trace.save(path, level="runnable")
    return path


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
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert not result.report.poisoned
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


def test_loaded_sparse_inplace_call_does_not_corrupt_staged_activations(
    tmp_path: Path,
) -> None:
    """Keep VERIFIED fork payloads equal to capture-time pre-mutation values."""

    model = InplaceActivationModel()
    inputs = torch.tensor([1.0, -2.0, 3.0])
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "inplace-activation.tlspec"
    trace.save(path, level="runnable")

    result = tl.load(path).run(inputs=inputs.clone())
    live_relu = next(op for op in trace.layer_list if op.func_name == "relu")
    fork_relu = next(op for op in result.trace.layer_list if op.func_name == "relu")
    fork_relu_before_output_edit = fork_relu.out.detach().clone()

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(live_relu.out, torch.tensor([1.0, 0.0, 3.0]))
    assert torch.equal(fork_relu.out, live_relu.out)
    result.output.zero_()
    assert torch.equal(fork_relu.out, fork_relu_before_output_edit)


@pytest.mark.parametrize(
    ("model", "expected_type"),
    [
        (MixedTupleOutputModel(), tuple),
        (ListOutputModel(), list),
        (NamedTupleOutputModel(), RunnableNamedOutput),
    ],
)
def test_loaded_sparse_preserves_output_container_kind_and_literal_leaves(
    tmp_path: Path,
    model: nn.Module,
    expected_type: type[Any],
) -> None:
    """Rebuild faithful mixed outputs without false structure divergence or holes."""

    inputs = torch.tensor([-2.0, 0.0, 2.0])
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / f"{expected_type.__name__}.tlspec"
    trace.save(path, level="runnable")

    loaded = tl.load(path)
    result = loaded.run(inputs=inputs.clone())
    expected = model(inputs)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert type(result.output) is expected_type
    assert torch.equal(result.output[0], expected[0])
    assert torch.equal(result.output[-1], expected[-1])
    if expected_type is not list:
        assert result.output[1] == 3.0
    assert all(check.passed for check in result.report.contract_checks)
    if expected_type is tuple:
        poisoned = loaded.run(
            inputs=torch.ones(4),
            on_divergence=DivergencePolicy.RETURN_DIVERGED,
        )
        assert poisoned.report.path_faithfulness is PathFaithfulness.DIVERGED
        assert poisoned.output[1] == 3.0


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


@pytest.mark.smoke
def test_witness_divergence_raises_and_rolls_back_by_default(honesty_artifact: Path) -> None:
    """Stop at the first flipped witness without exposing transactional updates."""

    loaded = tl.load(honesty_artifact)
    source_outs = _clone_outs(loaded)

    with pytest.raises(PathDivergenceError) as captured:
        loaded.run(inputs=-torch.ones(2), seed=19)

    mismatch = captured.value.fields["first_mismatch"]
    assert mismatch.code.value == "loop_predicate_divergence"
    assert mismatch.affected_op_labels
    _assert_outs_equal(loaded, source_outs)
    assert not bool(loaded.__dict__.get("_runnable_poisoned", False))


def test_shape_divergence_return_mode_finishes_and_poison_marks_result(
    honesty_artifact: Path,
) -> None:
    """Finish an executable mismatch only under the sole explicit poison opt-in."""

    loaded = tl.load(honesty_artifact)
    source_outs = _clone_outs(loaded)
    result = loaded.run(
        inputs=torch.ones(3),
        seed=23,
        on_divergence=DivergencePolicy.RETURN_DIVERGED,
    )

    assert result.output.shape == (3,)
    assert result.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert result.report.poisoned
    assert result.report.first_mismatch is not None
    assert result.report.first_mismatch.code.value == "input_shape_mismatch"
    assert result.trace.__dict__["_runnable_poisoned"] is True
    assert result.trace.__dict__["_runnable_path_faithfulness"] is PathFaithfulness.DIVERGED
    _assert_outs_equal(loaded, source_outs)


def test_poisoned_trace_is_refused_by_faithful_downstream_consumers(
    honesty_artifact: Path,
    tmp_path: Path,
) -> None:
    """Refuse every frozen downstream surface that assumes path identity."""

    poisoned = (
        tl.load(honesty_artifact)
        .run(
            inputs=torch.zeros(2),
            on_divergence=DivergencePolicy.RETURN_DIVERGED,
        )
        .trace
    )

    with pytest.raises(PoisonedRunError):
        poisoned.validate_forward_pass([])
    with pytest.raises(PoisonedRunError):
        poisoned.save(tmp_path / "poisoned.tlspec", level="runnable")
    with pytest.raises(PoisonedRunError):
        poisoned.to_pandas()
    with pytest.raises(PoisonedRunError):
        tl.debug.compare(poisoned, poisoned)
    with pytest.raises(PoisonedRunError):
        poisoned.push_from(poisoned.layer_list[0])
    with pytest.raises(PoisonedRunError):
        poisoned.save_intervention(tmp_path / "poisoned-intervention.tlspec")


def test_incomplete_witness_coverage_is_unverifiable_and_poisoned(
    honesty_artifact: Path,
) -> None:
    """Never promote absence of an observed mismatch to verified without coverage."""

    loaded = tl.load(honesty_artifact)
    descriptor = loaded.__dict__["_runnable_descriptor"]
    loaded.__dict__["_runnable_descriptor"] = replace(
        descriptor,
        witness_completeness=WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE,
    )

    result = loaded.run(inputs=torch.ones(2), seed=29)

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.first_mismatch is None
    assert result.report.poisoned
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert result.trace.__dict__["_runnable_poisoned"] is True


def test_poison_mark_and_first_mismatch_are_monotonic(honesty_artifact: Path) -> None:
    """Retain divergence across a later matching run instead of rehabilitating the Trace."""

    first = tl.load(honesty_artifact).run(
        inputs=torch.zeros(2),
        on_divergence=DivergencePolicy.RETURN_DIVERGED,
    )
    first_mismatch = first.report.first_mismatch

    second = first.trace.run(
        inputs=torch.ones(2),
        on_divergence=DivergencePolicy.RETURN_DIVERGED,
    )

    assert second.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert second.report.first_mismatch == first_mismatch
    assert second.report.poisoned
    with pytest.raises(PathDivergenceError):
        first.trace.run(inputs=torch.ones(2))


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
