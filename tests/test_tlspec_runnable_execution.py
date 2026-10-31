"""Stage 5 sparse DAG execution and unified ``RunResult`` tests."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._errors import TorchLensCaptureGapWarning
from torchlens._runnable_execution import _fresh_bare_tensor_root
from torchlens._runnable_state import prepare_runnable_state
from torchlens.errors import (
    PathDivergenceError,
    PoisonedRunError,
    RunCapabilityUnavailableError,
    RunPreconditionError,
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


def test_fresh_bare_tensor_root_fails_closed_for_duck_trace() -> None:
    """A duck trace without runnable state must fail closed instead of raising."""

    class _DuckTrace:
        """Minimal dict-backed trace stand-in without runnable state."""

    assert _fresh_bare_tensor_root(_DuckTrace()) is False


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


class RandomExecutionModel(nn.Module):
    """Static graph that consumes the seeded PyTorch generator."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Add one seeded random draw to the input."""

        return value + torch.rand_like(value)


class InplaceActivationModel(nn.Module):
    """Graph whose later in-place call must not rewrite staged activations."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Mutate a ReLU result only after another result has consumed it."""

        activated = torch.relu(value)
        preserved = activated + 1
        activated.mul_(0)
        return preserved + activated


class InplaceInputModel(nn.Module):
    """Graph that mutates its model-input tensor directly."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Increment the input in place before returning a derived result."""

        value.add_(2)
        return value * 3


class AliasedViewStateMutationModel(nn.Module):
    """Graph that mutates declared state through a storage-sharing view."""

    def __init__(self) -> None:
        """Register one buffer whose view is mutated during the forward pass."""

        super().__init__()
        self.register_buffer("offset", torch.arange(4.0).reshape(2, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Mutate a flattened state view before consuming the base buffer."""

        flattened = self.offset.view(-1)
        flattened.add_(1)
        return value + self.offset


class FunctionalBatchNormStateMutationModel(nn.Module):
    """Counter-free functional BatchNorm with direct or view-fed running state."""

    def __init__(self, *, composed_views: bool) -> None:
        """Register running statistics in direct or composed-view form.

        Parameters
        ----------
        composed_views:
            Whether one buffer should provide sliced running-stat views.
        """

        super().__init__()
        self.composed_views = composed_views
        if composed_views:
            self.register_buffer("stats", torch.stack((torch.zeros(3), torch.ones(3))))
        else:
            self.register_buffer("running_mean", torch.zeros(3))
            self.register_buffer("running_var", torch.ones(3))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Update running statistics through functional BatchNorm."""

        if self.composed_views:
            running_mean = self.stats[0]
            running_var = self.stats[1]
        else:
            running_mean = self.running_mean
            running_var = self.running_var
        normalized = torch.nn.functional.batch_norm(
            value,
            running_mean,
            running_var,
            training=True,
        )
        return normalized + running_var


class FailingLiveRunModel(nn.Module):
    """Model whose live refresh forward can deliberately raise."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Raise after one captured operation for negative inputs.

        Parameters
        ----------
        value:
            Input that selects the successful or failing path.

        Returns
        -------
        torch.Tensor
            Deterministic successful-path output.

        Raises
        ------
        RuntimeError
            If the input sum is negative.
        """

        staged = value + 1
        if bool(value.sum() < 0):
            raise RuntimeError("intentional live rerun failure")
        return staged * 2


class MultipassFunctionalFastModel(nn.Module):
    """Repeat one functional site enough times for recurrent grouping."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply four value-changing passes through one functional ReLU site."""

        for _ in range(4):
            value = torch.relu(value + 1)
        return value


class MultipassModuleFastModel(nn.Module):
    """Repeat one module instance enough times for recurrent grouping."""

    def __init__(self) -> None:
        """Initialize the shared atomic module."""

        super().__init__()
        self.shared = nn.ReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply four value-changing passes through the shared module."""

        for _ in range(4):
            value = self.shared(value + 1)
        return value


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


class TorchStructSeqOutputModel(nn.Module):
    """Model returning one of PyTorch's field-addressable structseq outputs."""

    def __init__(self, operation: str) -> None:
        """Store the selected torch reduction or ordering operation.

        Parameters
        ----------
        operation:
            One of ``max``, ``min``, ``median``, ``topk``, or ``sort``.
        """

        super().__init__()
        self.operation = operation

    def forward(self, value: torch.Tensor) -> Any:
        """Return the selected field-addressable torch structseq.

        Parameters
        ----------
        value:
            Input matrix reduced or ordered along its first axis.

        Returns
        -------
        Any
            The selected ``torch.return_types`` result.
        """

        if self.operation == "max":
            return torch.max(value, dim=0)
        if self.operation == "min":
            return torch.min(value, dim=0)
        if self.operation == "median":
            return torch.median(value, dim=0)
        if self.operation == "topk":
            return torch.topk(value, k=2, dim=0)
        if self.operation == "sort":
            return torch.sort(value, dim=0)
        raise ValueError(f"Unsupported torch structseq operation {self.operation!r}.")


class SingleTensorContainerOutputModel(nn.Module):
    """Model wrapping one final tensor call with a literal-bearing dictionary."""

    def forward(self, value: torch.Tensor) -> dict[str, Any]:
        """Return one computed tensor and fixed metadata.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        dict[str, Any]
            Tensor result plus a literal metadata leaf.
        """

        return {"value": value + 1, "metadata": None}


@pytest.fixture(scope="module")
def runnable_execution_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, RunnableExecutionModel, tl.Trace]]:
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
    try:
        yield path, model, trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def honesty_artifact(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
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
    try:
        yield path
    finally:
        trace.cleanup()


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


@pytest.mark.filterwarnings(
    # A08 random-role-init disclosure (runnable_random_init_run): this test
    # exercises a weight-free runnable run DELIBERATELY; the warning is the
    # feature under test elsewhere (tests/test_persist_honesty_runnable_random.py).
    "default:This runnable artifact carries no model weights"
)
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


def test_loaded_sparse_sequential_runs_have_unique_fork_labels(
    runnable_execution_artifact: tuple[Path, RunnableExecutionModel, tl.Trace],
) -> None:
    """Assign distinct monotonic labels to sequential transactions on one Trace."""

    path, model, _ = runnable_execution_artifact
    loaded = tl.load(path)
    loaded.load_state_dict(model.state_dict())

    first = loaded.run(inputs=torch.ones(2, 3))
    second = loaded.run(inputs=torch.ones(2, 3))

    assert first.trace.trace_label != second.trace.trace_label


@pytest.mark.smoke
def test_loaded_sparse_fast_run_verifies_once_then_reuses_compiled_result_trace(
    runnable_execution_artifact: tuple[Path, RunnableExecutionModel, tl.Trace],
) -> None:
    """Keep default verification on the first call and reuse one guarded result thereafter."""

    path, model, _ = runnable_execution_artifact
    loaded = tl.load(path)
    loaded.load_state_dict(model.state_dict())
    first_inputs = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 0.25, 4.0]])
    second_inputs = torch.tensor([[0.5, 1.0, -2.0], [1.25, -0.75, 3.0]])

    first = loaded.run(inputs=first_inputs, seed=73, fast=True)
    second = loaded.run(inputs=second_inputs, seed=73, fast=True)

    assert first.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert second.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert second.trace is first.trace
    assert second.trace is not loaded
    assert torch.equal(second.output, model(second_inputs))
    assert loaded.__dict__["_fast_run_session"].prepared_state is not None


def test_loaded_sparse_fast_run_refuses_seed_drift(
    runnable_execution_artifact: tuple[Path, RunnableExecutionModel, tl.Trace],
) -> None:
    """Pin verify-once evidence to the seed that initialized its cached state."""

    path, model, _ = runnable_execution_artifact
    loaded = tl.load(path)
    loaded.load_state_dict(model.state_dict())
    loaded.run(inputs=torch.ones(2, 3), seed=5, fast=True)

    with pytest.raises(RunCapabilityUnavailableError, match="pins the seed"):
        loaded.run(inputs=torch.ones(2, 3), seed=6, fast=True)


def test_loaded_sparse_fast_run_reseeds_rng_after_verify_once(tmp_path: Path) -> None:
    """Reproduce seeded random calls on every compiled trusted iteration."""

    inputs = torch.ones(2, 3)
    captured = tl.trace(
        RandomExecutionModel(),
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
            random_seed=31,
        ),
    )
    path = tmp_path / "random-fast.tlspec"
    captured.save(path, level="runnable")
    loaded = tl.load(path)

    verified = loaded.run(inputs=inputs, seed=31, fast=True)
    repeated = loaded.run(inputs=inputs, seed=31, fast=True)

    assert verified.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert repeated.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(repeated.output, verified.output)


def test_loaded_sparse_fast_run_refuses_training_batchnorm_state_drift(
    tmp_path: Path,
) -> None:
    """Refuse cached execution when BatchNorm functionally updates running stats."""

    model = nn.BatchNorm1d(3).train()
    inputs = torch.randn(4, 3)
    captured = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "training-batchnorm-fast.tlspec"
    captured.save(path, level="runnable", include_weights=True)

    oracle = tl.load(path).run(inputs=inputs)
    assert oracle.report.path_faithfulness is PathFaithfulness.VERIFIED
    loaded = tl.load(path)
    for _ in range(4):
        with pytest.raises(
            RunCapabilityUnavailableError,
            match="may update or mutate declared state",
        ):
            loaded.run(inputs=inputs, fast=True)


@pytest.mark.parametrize("composed_views", [False, True], ids=["direct", "composed-view-fed"])
def test_loaded_sparse_fast_run_refuses_counter_free_functional_batchnorm_state_drift(
    tmp_path: Path,
    *,
    composed_views: bool,
) -> None:
    """Refuse counter-free functional BatchNorm updates, including through state views."""

    inputs = torch.randn(4, 3)
    captured = tl.trace(
        FunctionalBatchNormStateMutationModel(composed_views=composed_views),
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / f"functional-batchnorm-{composed_views}-fast.tlspec"
    captured.save(path, level="runnable", include_weights=True)

    oracle = tl.load(path).run(inputs=inputs)
    assert oracle.report.path_faithfulness is PathFaithfulness.VERIFIED
    with pytest.raises(
        RunCapabilityUnavailableError,
        match="may update or mutate declared state",
    ):
        tl.load(path).run(inputs=inputs, fast=True)


def test_loaded_sparse_fast_run_keeps_eval_batchnorm_static_path(tmp_path: Path) -> None:
    """Keep eval-mode BatchNorm eligible when its running stats are read-only."""

    model = nn.BatchNorm1d(3).eval()
    inputs = torch.randn(4, 3)
    captured = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "eval-batchnorm-fast.tlspec"
    captured.save(path, level="runnable", include_weights=True)

    oracle = tl.load(path).run(inputs=inputs)
    loaded = tl.load(path)
    first = loaded.run(inputs=inputs, fast=True)
    second = loaded.run(inputs=inputs, fast=True)

    assert first.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert second.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(first.output, oracle.output)
    assert torch.equal(second.output, oracle.output)


def test_loaded_sparse_fast_run_refuses_state_view_inplace_mutation(
    tmp_path: Path,
) -> None:
    """Refuse cached execution when an in-place target aliases declared state."""

    model = AliasedViewStateMutationModel().eval()
    inputs = torch.ones(2, 2)
    captured = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "state-view-inplace-fast.tlspec"
    captured.save(path, level="runnable", include_weights=True)

    oracle = tl.load(path).run(inputs=inputs)
    assert oracle.report.path_faithfulness is PathFaithfulness.VERIFIED
    with pytest.raises(
        RunCapabilityUnavailableError,
        match="may update or mutate declared state",
    ):
        tl.load(path).run(inputs=inputs, fast=True)


def test_loaded_sparse_fast_run_does_not_mutate_caller_input(tmp_path: Path) -> None:
    """Mirror runtime inputs before compiled in-place recipes execute."""

    inputs = torch.arange(4.0)
    captured = tl.trace(
        InplaceInputModel(),
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "inplace-input-fast.tlspec"
    captured.save(path, level="runnable")
    loaded = tl.load(path)

    first_input = torch.arange(4.0)
    expected_first = (first_input + 2) * 3
    first = loaded.run(inputs=first_input, fast=True)
    second_input = torch.arange(4.0, 8.0)
    expected_second = (second_input + 2) * 3
    second = loaded.run(inputs=second_input, fast=True)

    assert torch.equal(first_input, torch.arange(4.0))
    assert torch.equal(second_input, torch.arange(4.0, 8.0))
    assert torch.equal(first.output, expected_first)
    assert torch.equal(second.output, expected_second)
    assert second.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_live_fast_run_refuses_same_shape_function_path_divergence() -> None:
    """Never relabel a changed same-shape functional branch as the captured static site."""

    class BranchingFunctionModel(nn.Module):
        """Choose between two same-shape activation functions from tensor data."""

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            """Apply the branch selected by the runtime sum."""

            if bool((value.sum() > 0).item()):
                return torch.relu(value)
            return torch.sigmoid(value)

    model = BranchingFunctionModel().eval()
    captured = tl.trace(model, torch.ones(2), save=tl.func("relu"))

    with pytest.raises(PathDivergenceError):
        captured.run(inputs=-torch.ones(2), fast=True)


@pytest.mark.parametrize("plan_kind", ["functional", "module"])
def test_live_fast_run_refreshes_every_multipass_activation(plan_kind: str) -> None:
    """Refresh each pass distinctly in both functional and module fast plans."""

    model: nn.Module
    save: Any
    if plan_kind == "functional":
        model = MultipassFunctionalFastModel()
        save = tl.func("relu")
    else:
        model = MultipassModuleFastModel()
        save = tl.module("shared")
    captured = tl.trace(model, torch.tensor([-5.0, 1.0]), save=save)
    selected = [
        op for op in captured.layer_list if op.has_saved_activation and op.func_name == "relu"
    ]
    captured_first = selected[0].out.clone()

    runtime_input = torch.tensor([-2.0, 3.0])
    expected_passes = []
    expected = runtime_input
    for _ in range(4):
        expected = torch.relu(expected + 1)
        expected_passes.append(expected.clone())

    result = captured.run(inputs=runtime_input, fast=True)
    refreshed = [
        op for op in result.trace.layer_list if op.has_saved_activation and op.func_name == "relu"
    ]

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert len(refreshed) == 4
    assert len({op.layer_label for op in refreshed}) == 1
    assert not torch.equal(refreshed[0].out, captured_first)
    assert all(
        torch.equal(op.out, expected_value)
        for op, expected_value in zip(refreshed, expected_passes, strict=True)
    )


# Heavy, not unmarked: 9.9 s CPU on a GitHub fast-tier worker (2026-10-03), past the 5 s
# unmarked budget even at reference speed.
@pytest.mark.heavy
def test_loaded_sparse_fast_run_keeps_control_witness_guard(honesty_artifact: Path) -> None:
    """Evaluate recorded control witnesses on every trusted compiled iteration."""

    loaded = tl.load(honesty_artifact)
    verified = loaded.run(inputs=torch.ones(2), seed=19, fast=True)
    assert verified.report.path_faithfulness is PathFaithfulness.VERIFIED

    with pytest.raises(PathDivergenceError):
        loaded.run(inputs=-torch.ones(2), seed=19, fast=True)


def test_loaded_sparse_execution_pauses_recursive_capture(
    runnable_execution_artifact: tuple[Path, RunnableExecutionModel, tl.Trace],
) -> None:
    """Invoke every resolved callable while the persistent wrapper gate is paused."""

    path, model, _ = runnable_execution_artifact
    loaded = tl.load(path)
    loaded.load_state_dict(model.state_dict())
    attached = loaded._runnable.callables_by_call_id
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


class SliceAssignModel(nn.Module):
    """Setter-style in-place mutator: ``__setitem__`` returns ``None``."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Mutate a slice in place and return the mutated tensor."""

        value = value.clone()
        value[0] = value[0] * 4
        return value


def test_loaded_sparse_setitem_runs_and_verifies_on_original_input(
    tmp_path: Path,
) -> None:
    """A ``__setitem__`` (None-returning) mutator replays without a false divergence.

    Regression for the setter-style output aliasing gap: ``Tensor.__setitem__``
    mutates its target in place but returns ``None``, so the recorded tensor
    output slot must alias the mutation target rather than expecting a tensor
    return. Before the fix, original-input replay crashed with a spurious
    PathDivergenceError instead of returning the correct, VERIFIED result.
    """

    model = SliceAssignModel()
    inputs = torch.arange(6.0).reshape(2, 3)
    expected = model(inputs.clone())
    trace = tl.trace(
        model,
        inputs.clone(),
        capture=CaptureOptions(intervention_ready=True, cache=False),
    )
    path = tmp_path / "slice-assign.tlspec"
    trace.save(path, level="runnable")

    result = tl.load(path).run(inputs=inputs.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(result.output, expected)

    # A changed input still replays the mutation faithfully (aliasing/mutation
    # honesty is preserved -- attestation is simply not applicable off-original).
    changed = torch.arange(10.0, 16.0).reshape(2, 3)
    changed_result = tl.load(path).run(inputs=changed.clone())
    assert torch.equal(changed_result.output, model(changed.clone()))
    assert changed_result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


class TorchRngBranchModel(nn.Module):
    """A branch predicate driven by a torch-RNG draw (input-disconnected)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take a nondeterministic arm based on a pruned ``rand -> gt`` predicate."""

        if torch.rand(()) > 0.5:
            return x * 2.0
        return x + 1.0


class DeadRngDrawModel(nn.Module):
    """A deterministic model whose RNG draw's result influences nothing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Draw an RNG value, discard it, and return a deterministic result."""

        _ = torch.rand(()) * 2.0 + 1.0
        return x * 2.0


def test_torch_rng_driven_branch_is_unverifiable_never_attested(tmp_path: Path) -> None:
    """A pruned torch-RNG control predicate must never report VERIFIED + ATTESTED.

    Regression for the orphan-removal honesty gap: ``if torch.rand(()) > 0.5`` is
    input-disconnected, so the ``rand -> gt`` predicate chain is orphaned out of the
    visible graph and the runnable descriptor never sees it. Replay always takes the
    single baked arm while a fresh seeded forward flips ~50% of the time, so the run
    must be honestly UNVERIFIABLE + NOT_APPLICABLE for every seed -- including the
    seed of the live forward -- rather than a false VERIFIED + ATTESTED.
    """

    model = TorchRngBranchModel()
    inputs = torch.tensor([1.0, 2.0])
    path = tmp_path / "torch-rng-branch.tlspec"
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable", include_activations=True)

    # The baked arm is whatever the capture forward took; pick a run seed whose
    # TRUE live forward takes the OTHER arm, then run at that exact seed.
    baked = tl.load(path).run(inputs=inputs, seed=0).output
    flip_seed = 3 if torch.allclose(baked, inputs * 2.0) else 1
    report = tl.load(path).run(inputs=inputs, seed=flip_seed).report
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE

    # Changed input is equally unverifiable (the taken branch is unwitnessed).
    changed = tl.load(path).run(inputs=torch.tensor([5.0, -1.0]), seed=0).report
    assert changed.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert changed.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_dead_torch_rng_draw_stays_verified_and_attested(tmp_path: Path) -> None:
    """A genuinely-dead torch-RNG draw must not over-trigger the honesty downgrade.

    The RNG value here feeds ops but influences neither output nor control, so the
    model is deterministic and its original-input replay must stay VERIFIED +
    ATTESTED. This guards the gate against firing on dead draws.
    """

    model = DeadRngDrawModel()
    inputs = torch.tensor([1.0, 2.0])
    path = tmp_path / "dead-rng-draw.tlspec"
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable", include_activations=True)

    result = tl.load(path).run(inputs=inputs, seed=0)
    assert torch.equal(result.output, inputs * 2.0)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


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


@pytest.mark.parametrize("operation", ("max", "min", "median", "topk", "sort"))
def test_loaded_sparse_preserves_torch_structseq_outputs(
    tmp_path: Path,
    operation: str,
) -> None:
    """Rebuild every supported torch structseq with producer field-name paths."""

    model = TorchStructSeqOutputModel(operation)
    inputs = torch.tensor([[1.0, 4.0, 2.0], [3.0, 2.0, 5.0], [0.0, 6.0, 1.0]])
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / f"{operation}-structseq.tlspec"
    trace.save(path, level="runnable")

    loaded = tl.load(path)
    result = loaded.run(inputs=inputs.clone())
    expected = model(inputs)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert type(result.output) is type(expected)
    assert [
        slot.output_path
        for slot in loaded._runnable.descriptor.tensor_slots
        if slot.role.value == "output"
    ] == [
        ("values",),
        ("indices",),
    ]
    for actual_leaf, expected_leaf in zip(result.output, expected, strict=True):
        assert torch.equal(actual_leaf, expected_leaf)
    assert all(check.passed for check in result.report.contract_checks)


def test_loaded_sparse_verifies_container_wrapping_one_final_tensor_call(tmp_path: Path) -> None:
    """Compare the reconstructed model boundary, not its lone final tensor call."""

    model = SingleTensorContainerOutputModel()
    inputs = torch.tensor([1.0, 2.0])
    trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "single-tensor-container.tlspec"
    trace.save(path, level="runnable")

    result = tl.load(path).run(inputs=inputs.clone())

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert type(result.output) is dict
    assert torch.equal(result.output["value"], model(inputs)["value"])
    assert result.output["metadata"] is None
    assert all(check.passed for check in result.report.contract_checks)


def _return_positional_pair(
    value: torch.Tensor,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a positional pair to simulate a genuinely changed structseq result.

    Parameters
    ----------
    value:
        Input tensor passed to the recorded ``torch.max`` call.
    **_kwargs:
        Recorded keyword arguments, intentionally ignored by this divergent callable.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A plain positional tuple rather than the recorded named torch structseq.
    """

    return value, value


def test_loaded_sparse_still_diverges_for_a_genuinely_changed_output_structure(
    tmp_path: Path,
) -> None:
    """Reject a positional result when the recorded call requires structseq field paths."""

    inputs = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    trace = tl.trace(
        TorchStructSeqOutputModel("max"),
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "changed-structseq-output.tlspec"
    trace.save(path, level="runnable")
    loaded = tl.load(path)
    loaded._runnable.callables_by_call_id["call:1"] = _return_positional_pair

    with pytest.raises(PathDivergenceError):
        loaded.run(inputs=inputs.clone())


def _logging_probe(func: Any, observed: list[bool]) -> Any:
    """Wrap one resolved callable and record the logging toggle at invocation."""

    def probe(*args: Any, **kwargs: Any) -> Any:
        """Record the toggle and forward one sparse call."""

        observed.append(_state._logging_enabled)
        return func(*args, **kwargs)

    return probe


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


def test_live_run_forward_failure_unregisters_transactional_fork() -> None:
    """Discard a registered live-run fork when refresh forward escapes."""

    model = FailingLiveRunModel()
    trace = tl.trace(model, torch.ones(3))
    prior_log_ids = {id(log) for log in _state.list_logs()}

    with pytest.raises(RuntimeError, match="intentional live rerun failure"):
        trace.run(inputs=-torch.ones(3))

    transactional_forks = tuple(
        log
        for log in _state.list_logs()
        if callable(getattr(log, "parent_run", None)) and log.parent_run() is trace
    )
    assert transactional_forks == ()
    assert {id(log) for log in _state.list_logs()} <= prior_log_ids


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


# Heavy, not smoke: 9.0-11.1 s CPU on GitHub rows (2026-10-03), past the smoke tier's 5 s rule.
@pytest.mark.heavy
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
    assert not bool(loaded._runnable.poisoned)


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
    assert result.trace._runnable.poisoned is True
    assert result.trace._runnable.path_faithfulness is PathFaithfulness.DIVERGED
    _assert_outs_equal(loaded, source_outs)


def test_unfinishable_sparse_call_raises_typed_error_without_leaking_fork(
    runnable_execution_artifact: tuple[Path, RunnableExecutionModel, tl.Trace],
) -> None:
    """Rollback registry state when a wrong-shape divergent input is inexecutable.

    r39 corr2_4: an admitted-but-INEXECUTABLE divergent input (wrong FEATURE shape that fails
    the native call) under ``return_diverged`` surfaces as ``PathDivergenceError`` carrying the
    already-failed ``input_shape`` contract check -- NOT ``RuntimeSignatureDriftError`` (which is
    reserved for genuine resolved-callable/torch-version drift with all input checks passing).
    The transactional fork is still rolled back either way (no leaked log).
    """

    path, model, _ = runnable_execution_artifact
    loaded = tl.load(path)
    loaded.load_state_dict(model.state_dict())
    logs_before = set(_state.list_logs())

    with pytest.raises(PathDivergenceError) as caught:
        loaded.run(
            inputs=torch.ones(2, 4),
            on_divergence=DivergencePolicy.RETURN_DIVERGED,
        )

    assert caught.value.fields["path_faithfulness"] is PathFaithfulness.DIVERGED
    check = caught.value.fields.get("contract_check")
    assert check is not None and check.name.startswith("input_shape:")
    assert set(_state.list_logs()) == logs_before


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
    """Never promote absence of an observed mismatch to verified without coverage.

    r71 A3: incompleteness is carried by the typed gap LEDGER (the summary is a
    redundant derived assertion). A surviving gap ceilings the run UNVERIFIABLE;
    a summary flipped WITHOUT its gap is an internally contradictory descriptor
    and refuses typed at run preparation (the floor re-assert belt).
    """

    from torchlens.runnable import (
        WITNESS_GAP_REGISTRY,
        WitnessCoverageGap,
        WitnessGapKind,
    )

    loaded = tl.load(honesty_artifact)
    descriptor = loaded._runnable.descriptor
    gap_spec = WITNESS_GAP_REGISTRY[WitnessGapKind.RNG_MONITOR_UNCERTAIN]
    gap = WitnessCoverageGap(
        gap_kind=WitnessGapKind.RNG_MONITOR_UNCERTAIN,
        source_family=gap_spec.source_family,
        source_member="capture",
        order=0,
        resulting_completeness=gap_spec.resulting_completeness,
    )
    loaded._runnable.descriptor = replace(
        descriptor,
        coverage_gaps=(gap,),
        witness_completeness=gap_spec.resulting_completeness,
    )

    result = loaded.run(inputs=torch.ones(2), seed=29)

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.first_mismatch is None
    assert result.report.poisoned
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert result.trace._runnable.poisoned is True

    # Summary-only flip (no gap): internally contradictory -> typed refusal, never
    # a silently trusted summary in EITHER direction.
    contradictory = tl.load(honesty_artifact)
    contradictory._runnable.descriptor = replace(
        contradictory._runnable.descriptor,
        witness_completeness=WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE,
    )
    with pytest.raises(RunPreconditionError):
        contradictory.run(inputs=torch.ones(2), seed=29)


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


# --------------------------------------------------------------------------- #
# r27-C1: ``_value_at_path`` must never feed an attacker-controlled path string
# to an unconstrained ``getattr``. Path components are reached from untrusted
# bundle fields (``input_binding.container_path`` and the recorded literal-witness
# ``fact["path"]``), so a component like ``"__class__"`` must be refused rather
# than firing descriptor getters or walking a dunder escape chain.
# --------------------------------------------------------------------------- #


class _C1FieldPair(NamedTuple):
    """Namedtuple container with one legitimate structural field."""

    value: torch.Tensor
    meta: int


def test_c1_value_at_path_refuses_dunder_attribute_traversal() -> None:
    """A dunder / non-field attribute component raises AttributeError, never resolves."""

    from torchlens._runnable_execution import _field_getattr, _value_at_path

    pair = _C1FieldPair(torch.zeros(2), 3)

    # Legitimate structural field access still works (no over-trigger).
    assert torch.equal(_value_at_path(pair, ("value",)), pair.value)
    assert _value_at_path(pair, ("meta",)) == 3

    # A dunder escape-chain component is refused BEFORE any getattr fires.
    for attacker in ("__class__", "__init__", "__globals__", "__dict__", "__reduce__"):
        with pytest.raises(AttributeError):
            _value_at_path(pair, (attacker,))
        with pytest.raises(AttributeError):
            _field_getattr(pair, attacker)

    # A non-field public attribute that genuinely exists on the object is still
    # refused: only STRUCTURALLY-declared fields are traversable.
    with pytest.raises(AttributeError):
        _value_at_path(pair, ("count",))  # tuple.count method attribute


def test_c1_value_at_path_mapping_and_index_paths_unaffected() -> None:
    """Mapping-key and integer-index traversal still work (attr guard is scoped)."""

    from torchlens._runnable_execution import _value_at_path

    root = {"weird__key__": torch.ones(3), "nested": [torch.zeros(1), {"k": 5}]}
    # Mapping keys use ``[]`` not getattr, so even a dunder-looking KEY is allowed.
    assert torch.equal(_value_at_path(root, ("weird__key__",)), root["weird__key__"])
    assert _value_at_path(root, ("nested", 1, "k")) == 5


# --------------------------------------------------------------------------- #
# R1-B C1: capture preserves model-input identity/storage semantics. The sparse
# runnable descriptor still cannot encode aliases between distinct model-input
# sites, so an alias-bearing runnable capture is explicitly UNVERIFIABLE rather
# than silently serializing a weaker all-distinct input contract.
# --------------------------------------------------------------------------- #


class _AliasInplaceModel(nn.Module):
    """Mutate one model input in place, then read another input site."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """In-place add on ``a``; a fresh aliased model would see it through ``b``."""

        a.add_(1.0)
        return b * 2.0


def _save_alias_runnable(path: Path, *, aliased_capture: bool = True) -> Path:
    """Capture and save the in-place model with aliased or distinct input sites."""

    t = torch.tensor([1.0, 2.0])
    inputs = [t, t] if aliased_capture else [t, t.clone()]

    def capture_call() -> tl.Trace:
        """Capture the prepared alias topology."""

        return tl.trace(
            _AliasInplaceModel(),
            inputs,
            capture=CaptureOptions(
                intervention_ready=True, capture_container_structure=True, cache=False
            ),
        )

    if aliased_capture:
        with pytest.warns(TorchLensCaptureGapWarning, match="sparse descriptor cannot encode"):
            trace = capture_call()
        assert trace.capture_verified is False
        assert trace.capture_verification_reason == "input_boundary_unverifiable"
    else:
        trace = capture_call()
    trace.save(path, level="runnable", include_activations=True)
    return path


def test_h3_same_object_aliased_input_with_inplace_fails_closed(tmp_path: Path) -> None:
    """An alias-bearing capture runs only with an UNVERIFIABLE verdict."""

    path = _save_alias_runnable(tmp_path / "alias_same.tlspec")
    shared = torch.tensor([1.0, 2.0])

    result = tl.load(path).run(inputs=[shared, shared])
    assert result.output.tolist() == [4.0, 6.0]
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert result.report.poisoned is True


def test_h3_view_aliased_distinct_objects_with_inplace_fails_closed(tmp_path: Path) -> None:
    """Distinct runtime objects that share storage must also fail closed."""

    path = _save_alias_runnable(tmp_path / "alias_view.tlspec")
    base = torch.tensor([1.0, 2.0])
    view = base.view_as(base)
    assert base is not view
    assert base.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()

    result = tl.load(path).run(inputs=[base, view])
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_h3_distinct_inputs_still_verify(tmp_path: Path) -> None:
    """Distinct capture/runtime input sites retain the ordinary VERIFIED path."""

    path = _save_alias_runnable(tmp_path / "alias_distinct.tlspec", aliased_capture=False)
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 2.0])

    result = tl.load(path).run(inputs=[a, b])
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_h3_readonly_aliased_inputs_fail_closed(tmp_path: Path) -> None:
    """Read-only alias identity also ceilings sparse replay at UNVERIFIABLE."""

    class _ReadOnly(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

    z = torch.tensor([3.0, 4.0])
    with pytest.warns(TorchLensCaptureGapWarning, match="sparse descriptor cannot encode"):
        trace = tl.trace(
            _ReadOnly(),
            [z, z],
            capture=CaptureOptions(
                intervention_ready=True, capture_container_structure=True, cache=False
            ),
        )
    assert trace.capture_verified is False
    trace.save(tmp_path / "readonly.tlspec", level="runnable", include_activations=True)

    shared = torch.tensor([3.0, 4.0])
    aliased = tl.load(tmp_path / "readonly.tlspec").run(inputs=[shared, shared])
    assert aliased.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    distinct = tl.load(tmp_path / "readonly.tlspec").run(
        inputs=[torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])]
    )
    assert distinct.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


# --------------------------------------------------------------------------- #
# r27-H5: self.training is module state (not state_dict, not an input) that steers
# mode-sensitive ops (BatchNorm running-stats vs batch-stats). The VERIFIED oracle
# is a fresh instance IN THE CAPTURED MODE on the given inputs, so the captured mode
# is DECLARED state. Every intervention-ready capture now records it; a mode-sensitive
# op replayed WITHOUT a declared mode (an old bundle / capture gap) downgrades to
# UNVERIFIABLE (fail closed). Mode-insensitive models are unaffected.
# --------------------------------------------------------------------------- #


class _BatchNormModel(nn.Module):
    """Linear + BatchNorm1d: BatchNorm is train/eval mode-sensitive."""

    def __init__(self) -> None:
        """Build a linear layer feeding a BatchNorm layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear then the mode-sensitive BatchNorm."""

        return self.bn(self.lin(x))


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.filterwarnings(
    # A08 random-role-init disclosure (runnable_random_init_run): this test
    # exercises a weight-free runnable run DELIBERATELY; the warning is the
    # feature under test elsewhere (tests/test_persist_honesty_runnable_random.py).
    "default:This runnable artifact carries no model weights"
)
def test_h5_batchnorm_records_mode_and_verifies(train: bool, tmp_path: Path) -> None:
    """A BatchNorm model records its mode and still VERIFIES (no over-trigger)."""

    model = _BatchNormModel()
    model.train(train)
    x = torch.randn(8, 4)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    assert trace._runnable.module_training_modes == {
        "self": train,
        "lin": train,
        "bn": train,
    }
    trace.save(tmp_path / "bn.tlspec", level="runnable", include_activations=True)
    result = tl.load(tmp_path / "bn.tlspec").run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.filterwarnings(
    # A08 random-role-init disclosure (runnable_random_init_run): this test
    # exercises a weight-free runnable run DELIBERATELY; the warning is the
    # feature under test elsewhere (tests/test_persist_honesty_runnable_random.py).
    "default:This runnable artifact carries no model weights"
)
def test_h5_dropout_model_in_eval_records_mode_and_verifies(tmp_path: Path) -> None:
    """A Dropout model captured in eval records its mode and still VERIFIES."""

    class _DropModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.drop = nn.Dropout(0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.drop(self.lin(x))

    model = _DropModel().eval()
    x = torch.randn(8, 4)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    assert trace._runnable.module_training_modes == {
        "self": False,
        "lin": False,
        "drop": False,
    }
    trace.save(tmp_path / "drop.tlspec", level="runnable", include_activations=True)
    result = tl.load(tmp_path / "drop.tlspec").run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_h5_mode_sensitive_op_without_declared_mode_is_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BatchNorm capture that records NO mode now fails TYPED at save (r71 A4).

    The r71 mode-domain check moved to parse and the producer runs the SAME validator
    as its save-time self-check, so a capture gap that drops the declared train/eval
    mode (the corr2recover producer-regression lane) refuses the runnable save with
    ``context_field_invalid`` -- strictly stronger than the former ship-then-
    UNVERIFIABLE disposition. ``_mode_sensitive_op_unwitnessed`` remains as the
    runtime belt behind the parse gate.
    """

    import torchlens.capture.trace as capture_trace
    from torchlens.errors import RunnablePreflightError

    # Simulate an old bundle / capture gap: skip the mode recording entirely.
    monkeypatch.setattr(
        capture_trace, "_record_runnable_module_training_modes", lambda trace, model: None
    )
    model = _BatchNormModel().eval()
    x = torch.randn(8, 4)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    assert trace._runnable.module_training_modes is None
    with pytest.raises(RunnablePreflightError) as excinfo:
        trace.save(tmp_path / "bn_nomode.tlspec", level="runnable", include_activations=True)
    assert "context_field_invalid" in str(excinfo.value.fields.get("diagnostics"))
    assert "module_training_mode" in str(excinfo.value.fields.get("diagnostics"))


@pytest.mark.filterwarnings(
    # A08 random-role-init disclosure (runnable_random_init_run): this test
    # exercises a weight-free runnable run DELIBERATELY; the warning is the
    # feature under test elsewhere (tests/test_persist_honesty_runnable_random.py).
    "default:This runnable artifact carries no model weights"
)
def test_h5_mode_insensitive_model_without_mode_still_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model with NO mode-sensitive op verifies even with no declared mode (no over-trigger)."""

    import torchlens.capture.trace as capture_trace

    monkeypatch.setattr(
        capture_trace, "_record_runnable_module_training_modes", lambda trace, model: None
    )

    class _Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    x = torch.randn(8, 4)
    trace = tl.trace(
        _Plain(),
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    trace.save(tmp_path / "plain_nomode.tlspec", level="runnable", include_activations=True)
    result = tl.load(tmp_path / "plain_nomode.tlspec").run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
