"""W051-TRACK / AUD-CODE 2.15: an ``echo=`` sink-open failure must release the
capture reservation on the error path.

The echo sink opens its path eagerly; before the fix the open ran between the
capture-slot claim and the forward's own guard, so holding the exception (REPL
``sys.last_traceback``, pytest ``excinfo``) left the process refusing every
later ``tl.trace`` with ``reentrant_trace`` until the exception was dropped.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.options import EchoOptions


def _model_and_input() -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()), torch.randn(2, 4)


def test_echo_sink_open_failure_releases_capture_slot_while_exception_is_held(
    tmp_path,  # noqa: ANN001
) -> None:
    """The user's sink error propagates AND the next capture is admitted."""

    model, x = _model_and_input()
    missing = tmp_path / "no_such_dir" / "echo.log"
    with pytest.raises(FileNotFoundError) as info:
        tl.trace(model, x, echo=EchoOptions(sink=str(missing)))
    # ``info`` keeps the exception (and its frames) alive: the exact shape
    # that leaked the reservation before the fix.
    assert info.value is not None
    trace = tl.trace(model, x)
    assert len(trace.layer_labels) > 0
    # The released model is still fully instrumentation-free.
    tl.release_model(model)


def test_echo_sink_open_failure_leaves_no_runtime_context(tmp_path) -> None:  # noqa: ANN001
    """The capture-global runtime context is reset on the echo-open error path."""

    from torchlens import _state

    model, x = _model_and_input()
    with pytest.raises(FileNotFoundError):
        tl.trace(model, x, echo=EchoOptions(sink=str(tmp_path / "missing" / "e.log")))
    with _state.capture_reservation():
        pass  # admitted: the slot is free
