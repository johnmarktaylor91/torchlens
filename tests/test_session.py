"""Phase 5a tests for experimental multi-input sessions."""

from __future__ import annotations

import torch

import torchlens as tl


def test_session_invoke_and_bundle() -> None:
    """Experimental sessions should capture multiple invocations."""

    model = torch.nn.Linear(2, 2)
    session = tl.experimental.session(model)

    first = session.invoke(torch.ones(1, 2))
    second = session.invoke(torch.zeros(1, 2))
    bundle = session.bundle()

    assert first.session_invocation["index"] == 0
    assert second.session_invocation["index"] == 1
    assert len(session.logs) == 2
    assert len(bundle) == 2
    assert first.session_logs == [first, second]
