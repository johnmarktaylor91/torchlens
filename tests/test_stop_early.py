"""Stop-early surface tests (A06 rewrite: ``stop_after`` is WIRED).

Historical note: these tests originally pinned the pre-wiring defect --
``pluck`` accepted-and-ignored ``stop_after`` while ``trace`` raised
``NotImplementedError``. A06 wired the option through the halt engine with
brainpipe's inclusive spelling; the full matrix lives in
``tests/test_capopts_truth_stop_after.py``. The pluck KWARG remains
accepted-and-inert until its owning facade lane deletes it (F20).
"""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens.options import CaptureOptions


def test_stop_after_on_peek_is_accepted() -> None:
    """The pluck kwarg is still accepted (inert; deletion is F20's row)."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    value = tl.pluck(model, torch.ones(1, 2), "relu", stop_after="relu")
    assert isinstance(value, torch.Tensor)


def test_stop_after_context_on_peek_is_accepted() -> None:
    """The ambient context manager arms pluck's internal capture."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    with tl.experimental.stop_after("relu"):
        value = tl.pluck(model, torch.ones(1, 2), "relu")
    assert isinstance(value, torch.Tensor)


def test_stop_after_wired_on_trace() -> None:
    """trace(capture=CaptureOptions(stop_after=...)) halts inclusively."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 2))
    log = tl.trace(model, torch.ones(1, 2), capture=CaptureOptions(stop_after="relu"))
    assert log.halted is True
    labels = [op.layer_label for op in log]
    assert any(label.startswith("relu") for label in labels)
    assert not any(label.startswith("linear_2") for label in labels)
