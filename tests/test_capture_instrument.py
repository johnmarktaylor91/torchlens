"""``torchlens.backends.torch.instrument.instrumented``: the WRAP bracket.

The bracket composes, in RUN's order, the calls that install capture
machinery on torch and the model, and removes them in reverse on every exit
arm. These tests pin the order, the once-only session cleanup (and its
skip when an arm already retired the label session), the observer restore
before release, and the unwrap that runs only after a completed capture.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from torchlens._utils._tensor_label import (
    active_label_session_token,
    begin_label_session,
    end_label_session,
)
from torchlens.backends.torch import instrument
from torchlens.backends.torch.instrument import WrapPolicy


class _Backend:
    """A backend that logs its calls and runs a real label session."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def prepare_model_once(self, model: object) -> object:
        self.calls.append(("once", model))
        return model

    def prepare_model_session(self, session: object, model: object) -> object:
        self.calls.append(("session", model))
        begin_label_session()
        return model

    def cleanup_model_session(self, session: object, prepared_model: object) -> None:
        self.calls.append(("cleanup", prepared_model))
        end_label_session()


@pytest.fixture
def no_session() -> Any:
    """Start and end each test with no label session installed."""

    if active_label_session_token() is not None:
        end_label_session()
    yield
    if active_label_session_token() is not None:
        end_label_session()


def _model() -> nn.Module:
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    model.eval()
    return model


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/instrument.py.
def test_cleanup_runs_once_with_callable_payload(no_session: Any) -> None:
    backend = _Backend()
    model = _model()
    seen: list[int] = []

    def payload() -> tuple[Any, ...]:
        seen.append(1)
        return (model, (torch.zeros(1),), None)

    with instrument.instrumented(backend, object(), model, payload):
        assert seen == []
    assert seen == [1]
    cleanups = [arg for name, arg in backend.calls if name == "cleanup"]
    assert len(cleanups) == 1
    assert cleanups[0][0] is model


# Kept by shrink-r4 (per-id subset rule): sole coverage of _runnable/state_live.py, backends/torch/instrument.py.
def test_observer_handle_gets_the_trace_on_success(no_session: Any) -> None:
    backend = _Backend()
    model = _model()
    trace = object()
    with instrument.instrumented(
        backend, trace, model, (model, None, None), WrapPolicy(carry_state=False)
    ) as handle:
        assert handle.observer is not None
        assert handle.observer.trace is None
    assert handle.observer is not None
    assert handle.observer.trace is trace


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/instrument.py.
def test_unwrap_only_after_a_completed_capture(
    no_session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    unwraps: list[int] = []
    monkeypatch.setattr(instrument, "_unwrap_after_success", lambda: unwraps.append(1))
    backend = _Backend()
    model = _model()
    with instrument.instrumented(backend, object(), model, (model, None, None)):
        pass
    assert unwraps == []
    unwrap = WrapPolicy(unwrap_when_done=True)
    with instrument.instrumented(backend, object(), model, (model, None, None), unwrap) as handle:
        assert handle.unwrap_when_done is True
        assert unwraps == []
    assert unwraps == [1]
    with (
        pytest.raises(RuntimeError, match="no unwrap"),
        instrument.instrumented(backend, object(), model, (model, None, None), unwrap),
    ):
        raise RuntimeError("no unwrap")
    assert unwraps == [1]
