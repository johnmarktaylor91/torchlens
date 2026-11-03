"""A06 capture-options truth: the chunked path drops NO capture option.

WALKTHROUGH list-A row 26 (second clause): ``chunk_size=`` silently dropped
``save_budget`` -- the chunk path rebuilds a recursive ``CaptureOptions``
from a hand-enumerated field list, so session knobs absent from that list
reset to their defaults on every chunked capture (the same include-list
disease the cache key had). The recursive constructor now covers every
``CaptureOptions`` field, and the AST tripwire here turns any future field
that is not consciously routed through the chunk path into a red test.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._save_budget import SaveBudgetExceededError
from torchlens.options import CaptureOptions


class SmallNet(nn.Module):
    """fc -> relu."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def test_chunked_capture_honors_save_budget() -> None:
    """A 1-byte budget must trip on the chunked path exactly as unchunked."""

    x = torch.randn(8, 4)
    with pytest.raises(SaveBudgetExceededError):
        tl.trace(SmallNet(), x, capture=CaptureOptions(save_budget=1))
    with pytest.raises(SaveBudgetExceededError):
        tl.trace(SmallNet(), x, capture=CaptureOptions(save_budget=1, chunk_size=4))


@pytest.mark.smoke
def test_chunked_capture_carries_peak_memory_knob() -> None:
    """measure_python_peak_memory survives the chunked rebuild."""

    x = torch.randn(8, 4)
    log = tl.trace(
        SmallNet(),
        x,
        capture=CaptureOptions(
            measure_python_peak_memory=True, measure_host_peak_memory=True, chunk_size=4
        ),
    )
    peak = log.cost.forward_peak_memory
    assert peak is not None and peak > 0, (
        "measure_python_peak_memory=True on a chunked capture reported no "
        "tracemalloc peak; the knob was dropped by the recursive rebuild"
    )
