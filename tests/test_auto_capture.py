"""Phase 4 auto-capture tests."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl


def test_auto_capture_captures_every_nth_call() -> None:
    """Context manager captures every configured forward call."""

    model = torch.nn.Linear(2, 2)
    with tl.experimental.auto_capture(model, every=2) as session:
        model(torch.ones(1, 2))
        model(torch.ones(1, 2))
        model(torch.ones(1, 2))
        model(torch.ones(1, 2))
    assert session.calls == 4
    assert len(session.logs) == 2


def test_torchlens_auto_env_var_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment-variable auto capture is explicitly rejected."""

    monkeypatch.setenv("TORCHLENS_AUTO", "1")
    with pytest.raises(RuntimeError):
        with tl.experimental.auto_capture(torch.nn.Linear(2, 2), every=1):
            pass
