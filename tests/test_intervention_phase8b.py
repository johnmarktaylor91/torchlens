"""Phase 8b dispatch, warning, fork, and history tests."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens import RunState
from torchlens.intervention.errors import (
    DirectActivationWriteWarning,
    EngineDispatchError,
    ModelMismatchError,
    MutateInPlaceWarning,
)


class ReluLinear(torch.nn.Module):
    """Small model with stable linear and relu intervention sites."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer followed by relu.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output.
        """

        return torch.relu(self.linear(x))


class WrongModel(torch.nn.Module):
    """Model with mismatched class and weight fingerprint evidence."""

    def __init__(self) -> None:
        """Initialize the incompatible layer."""

        super().__init__()
        self.fc = torch.nn.Linear(99, 99)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the incompatible layer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Linear output.
        """

        return self.fc(x)


def _zero_hook(activation: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Return zeros matching the incoming activation.

    Parameters
    ----------
    activation:
        Matched activation.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Zero activation.
    """

    del hook
    return activation * 0


def _capture(
    model: torch.nn.Module | None = None,
    x: torch.Tensor | None = None,
) -> tuple[Any, torch.Tensor]:
    """Capture an intervention-ready log and return it with its input.

    Parameters
    ----------
    model:
        Optional model to capture.
    x:
        Optional input tensor.

    Returns
    -------
    tuple[Any, torch.Tensor]
        Captured model log and input tensor.
    """

    capture_x = torch.randn(2, 3) if x is None else x
    log = tl.log_forward_pass(
        ReluLinear() if model is None else model,
        capture_x,
        vis_opt="none",
        intervention_ready=True,
    )
    return log, capture_x


@pytest.mark.smoke
def test_mutate_warning_fires_once_and_can_be_suppressed() -> None:
    """Root mutators warn once, with call-level and session-level suppression."""

    log, _ = _capture()
    with pytest.warns(MutateInPlaceWarning):
        log.set(tl.func("relu"), torch.zeros(2, 3))
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        log.set(tl.func("relu"), torch.ones(2, 3))
    assert warnings_record == []

    confirmed, _ = _capture()
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        confirmed.set(tl.func("relu"), torch.zeros(2, 3), confirm_mutation=True)
    assert warnings_record == []

    session, _ = _capture()
    with tl.suppress_mutate_warnings():
        with warnings.catch_warnings(record=True) as warnings_record:
            warnings.simplefilter("always")
            session.set(tl.func("relu"), torch.zeros(2, 3))
    assert warnings_record == []


@pytest.mark.smoke
def test_fork_mutation_does_not_warn_or_mutate_parent_intervention_log() -> None:
    """Forked logs have independent specs and per-pass intervention logs."""

    parent, _ = _capture()
    relu_pass = next(layer for layer in parent.layer_list if layer.func_name == "relu")
    parent_initial_log = list(relu_pass.intervention_log)

    fork = parent.fork("test")
    fork_relu_pass = next(layer for layer in fork.layer_list if layer.func_name == "relu")
    assert fork.parent_run() is parent
    assert fork.name == "test"
    assert fork_relu_pass.activation is relu_pass.activation
    assert fork_relu_pass.intervention_log is not relu_pass.intervention_log
    assert fork_relu_pass.equivalent_operations is not relu_pass.equivalent_operations

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        fork.set(tl.func("relu"), torch.zeros_like(fork_relu_pass.activation))
    assert warnings_record == []
    fork.replay()

    assert list(relu_pass.intervention_log) == parent_initial_log
    assert len(fork_relu_pass.intervention_log) > len(parent_initial_log)
    assert len(fork._intervention_spec.target_value_specs) == 1
    assert len(parent._intervention_spec.target_value_specs) == 0


@pytest.mark.smoke
def test_direct_activation_write_warns_once_and_marks_dirty() -> None:
    """Direct activation writes emit a warning and mark the owning log dirty."""

    log, _ = _capture()
    relu_pass = next(layer for layer in log.layer_list if layer.func_name == "relu")

    with pytest.warns(DirectActivationWriteWarning):
        relu_pass.activation = relu_pass.activation
    assert log._has_direct_writes is True
    assert log.run_state is RunState.DIRECT_WRITE_DIRTY

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        relu_pass.activation = relu_pass.activation
    assert warnings_record == []


@pytest.mark.smoke
def test_do_dispatch_replay_rerun_set_only_and_top_level_alias() -> None:
    """``do`` dispatches to replay, rerun, set-only, and top-level alias paths."""

    replay_log, _ = _capture()
    replay_result = tl.do(replay_log, {tl.func("relu"): _zero_hook}, confirm_mutation=True)
    assert replay_result is replay_log
    assert replay_log.run_state is RunState.REPLAY_PROPAGATED
    assert replay_log.operation_history[-1]["op"] == "replay"
    assert any(record["op"] == "do" for record in replay_log.operation_history)

    rerun_model = ReluLinear()
    rerun_log, x = _capture(rerun_model)
    rerun_result = rerun_log.do(
        {tl.func("relu"): _zero_hook},
        model=ReluLinear(),
        x=x,
        confirm_mutation=True,
    )
    assert rerun_result is rerun_log
    assert rerun_log.run_state is RunState.RERUN_PROPAGATED
    assert rerun_log.operation_history[-1]["op"] == "rerun"

    set_only_log, _ = _capture()
    set_only_log.do(
        tl.func("relu"),
        torch.zeros(2, 3),
        engine="set_only",
        confirm_mutation=True,
    )
    assert set_only_log.run_state is RunState.SPEC_STALE
    assert set_only_log.operation_history[-1]["op"] == "do"


@pytest.mark.smoke
def test_do_ambiguous_dispatch_and_model_mismatch_errors() -> None:
    """``do`` raises targeted errors for ambiguous dispatch and model mismatch."""

    log, x = _capture()

    with pytest.raises(EngineDispatchError):
        log.do({tl.func("relu"): _zero_hook}, x=x, confirm_mutation=True)

    with pytest.raises(EngineDispatchError):
        log.do({tl.func("relu"): _zero_hook}, model=ReluLinear(), confirm_mutation=True)

    history_len = len(log.operation_history)
    spec_revision = log._spec_revision
    with pytest.raises(ModelMismatchError):
        log.do(
            {tl.func("relu"): _zero_hook},
            model=WrongModel(),
            x=torch.randn(2, 99),
            confirm_mutation=True,
        )
    assert len(log.operation_history) == history_len
    assert log._spec_revision == spec_revision


@pytest.mark.smoke
def test_direct_write_propagation_warning_is_one_time() -> None:
    """Replay warns once when propagation overlays direct writes."""

    log, _ = _capture()
    relu_pass = next(layer for layer in log.layer_list if layer.func_name == "relu")
    with pytest.warns(DirectActivationWriteWarning):
        relu_pass.activation = relu_pass.activation

    log.set(tl.func("relu"), torch.zeros_like(relu_pass.activation), confirm_mutation=True)
    with pytest.warns(DirectActivationWriteWarning):
        log.replay()

    relu_pass.activation = relu_pass.activation
    log.set(tl.func("relu"), torch.ones_like(relu_pass.activation), confirm_mutation=True)
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        log.replay()
    assert warnings_record == []
