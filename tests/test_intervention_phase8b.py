"""Phase 8b dispatch, warning, fork, and history tests."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.intervention.errors import (
    DirectActivationWriteWarning,
    EngineDispatchError,
    ModelMismatchError,
    TraceMutationWarning,
)
from torchlens.io import TraceState, suppress_mutate_warnings
from torchlens.options import CaptureOptions, InterventionOptions


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


def _zero_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    """Return zeros matching the incoming out.

    Parameters
    ----------
    out:
        Matched out.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Zero out.
    """

    del ctx
    return out * 0


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
    log = tl.trace(
        ReluLinear() if model is None else model,
        capture_x,
        capture=CaptureOptions(replayable=True),
    )
    return log, capture_x


def test_mutate_warning_fires_once_and_can_be_suppressed() -> None:
    """Root mutators warn once, with call-level and session-level suppression."""

    log, _ = _capture()
    with pytest.warns(TraceMutationWarning):
        log.do(
            tl.func("relu"),
            torch.zeros(2, 3),
            intervention=tl.options.InterventionOptions(engine="set_only"),
        )
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        log.do(
            tl.func("relu"),
            torch.ones(2, 3),
            intervention=tl.options.InterventionOptions(engine="set_only"),
        )
    assert warnings_record == []

    confirmed, _ = _capture()
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        confirmed.do(
            tl.func("relu"),
            torch.zeros(2, 3),
            intervention=tl.options.InterventionOptions(engine="set_only", confirm_mutation=True),
        )
    assert warnings_record == []

    session, _ = _capture()
    with (
        suppress_mutate_warnings(),
        warnings.catch_warnings(record=True) as warnings_record,
    ):
        warnings.simplefilter("always")
        session.do(
            tl.func("relu"),
            torch.zeros(2, 3),
            intervention=tl.options.InterventionOptions(engine="set_only"),
        )
    assert warnings_record == []

    callable_session, _ = _capture()
    suppress_mutate_warnings(False)
    assert suppress_mutate_warnings.is_suppressed is False
    try:
        returned = suppress_mutate_warnings(True)
        assert returned is suppress_mutate_warnings
        assert suppress_mutate_warnings.is_suppressed is True
        with warnings.catch_warnings(record=True) as warnings_record:
            warnings.simplefilter("always")
            callable_session.do(
                tl.func("relu"),
                torch.zeros(2, 3),
                intervention=tl.options.InterventionOptions(engine="set_only"),
            )
        assert warnings_record == []
    finally:
        suppress_mutate_warnings(False)


def test_direct_out_write_warns_once_and_marks_dirty() -> None:
    """Direct out writes emit a warning and mark the owning log dirty."""

    log, _ = _capture()
    relu_pass = next(layer for layer in log._layer_list if layer.func_name == "relu")

    with pytest.warns(DirectActivationWriteWarning):
        relu_pass.out = relu_pass.out
    assert log._has_direct_writes is True
    assert log.state is TraceState.DIRECT_WRITE_DIRTY

    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        relu_pass.out = relu_pass.out
    assert warnings_record == []


# Heavy, not smoke: 11.3-12.4 s CPU on GitHub rows (2026-10-03), about 6 s at
# reference speed, past the smoke tier's 5 s rule.
def test_do_ambiguous_dispatch_and_model_mismatch_errors() -> None:
    """``do`` raises targeted errors for ambiguous dispatch and model mismatch."""

    log, x = _capture()

    with pytest.raises(EngineDispatchError):
        log.do(
            {tl.func("relu"): _zero_hook},
            inputs=x,
            intervention=InterventionOptions(confirm_mutation=True),
        )

    with pytest.raises(EngineDispatchError):
        log.do(
            {tl.func("relu"): _zero_hook},
            model=ReluLinear(),
            intervention=InterventionOptions(confirm_mutation=True),
        )

    history_len = len(log.state_history)
    spec_revision = log._spec_revision
    with pytest.raises(ModelMismatchError):
        log.do(
            {tl.func("relu"): _zero_hook},
            model=WrongModel(),
            inputs=torch.randn(2, 99),
            intervention=InterventionOptions(confirm_mutation=True),
        )
    assert len(log.state_history) == history_len
    assert log._spec_revision == spec_revision


def test_direct_write_propagation_warning_is_one_time() -> None:
    """Replay warns once when propagation overlays direct writes."""

    log, _ = _capture()
    relu_pass = next(layer for layer in log._layer_list if layer.func_name == "relu")
    with pytest.warns(DirectActivationWriteWarning):
        relu_pass.out = relu_pass.out

    log.do(
        tl.func("relu"),
        torch.zeros_like(relu_pass.out),
        intervention=tl.options.InterventionOptions(engine="set_only", confirm_mutation=True),
    )
    with pytest.warns(DirectActivationWriteWarning):
        log.push()

    relu_pass.out = relu_pass.out
    log.do(
        tl.func("relu"),
        torch.ones_like(relu_pass.out),
        intervention=tl.options.InterventionOptions(engine="set_only", confirm_mutation=True),
    )
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        log.push()
    assert warnings_record == []
