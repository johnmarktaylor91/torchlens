"""Dynamic graph coverage for repeated fastlog recording."""

from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl


class DynamicMLP(nn.Module):
    """MLP with input-dependent control flow."""

    def __init__(self) -> None:
        """Initialize three layers."""

        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a variable number of layers based on the input sum."""

        for i in range(int(x.sum().item()) % 3 + 1):
            x = F.relu(self.layers[i](x))
        return x


def test_dynamic_graph_records_1000_passes_with_varying_event_counts() -> None:
    """Recorder handles 1000 input-dependent passes and preserves pass counts."""

    model = DynamicMLP()
    with tl.fastlog.Recorder(model, keep_op=lambda ctx: ctx.kind == "op") as recorder:
        for index in range(1000):
            recorder.log(torch.full((1, 4), float(index)))
    recording = recorder.recording
    op_counts = Counter(record.ctx.pass_index for record in recording if record.ctx.kind == "op")

    assert recording.n_passes == 1000
    assert set(op_counts.values()) == {3, 5, 7}
    assert {op_counts[index] for index in (1, 2, 3)} == {3, 5, 7}
