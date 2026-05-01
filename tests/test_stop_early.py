"""Phase 4 stop-early tests."""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.options import CaptureOptions


def test_stop_after_on_peek_is_accepted() -> None:
    """Experimental stop_after is accepted by peek."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    value = tl.peek(model, torch.ones(1, 2), "relu", stop_after="relu")
    assert isinstance(value, torch.Tensor)


def test_stop_after_context_on_peek_is_accepted() -> None:
    """Experimental stop_after context applies to peek."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    with tl.experimental.stop_after("relu"):
        value = tl.peek(model, torch.ones(1, 2), "relu")
    assert isinstance(value, torch.Tensor)


def test_stop_after_raises_on_log_forward_pass() -> None:
    """stop_after is intentionally unsupported for full capture."""

    model = torch.nn.ReLU()
    with pytest.raises(NotImplementedError):
        tl.log_forward_pass(model, torch.ones(1, 2), capture=CaptureOptions(stop_after="relu"))
