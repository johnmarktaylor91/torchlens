"""A plain rerun translates staged final labels to the raw capture labels they name.

A live forward matches selectors before postprocess labels exist. A staged edit addressed
by a finalized exact label (``Trace.set(label, ...)``, ``do(label, ...)`` on the rerun engine)
is translated before the forward to the raw labels the same ops carried at capture time, the
translation a mixed gradient plan's activation half already used, so the edit fires on the
rerun instead of being refused. Each pass of a multi-pass op keeps its own raw label, so a
pass-qualified label edits that pass alone; a substring selector on a finalized label is not
translated and still refuses through the live matcher.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import LiveModeLabelError
from torchlens.options import CaptureOptions, InterventionOptions


class _Recurrent(nn.Module):
    """Three unrolled steps; each step's output feeds the next step's add."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(x)
        hidden = gate
        for _ in range(3):
            hidden = torch.tanh(self.lin(gate) + hidden)
        return hidden


_RERUN = InterventionOptions(engine="rerun")


def _recurrent_case() -> tuple[nn.Module, torch.Tensor, tl.Trace, list[str]]:
    torch.manual_seed(0)
    model = _Recurrent().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(replayable=True))
    tanh_labels = sorted(
        str(op.label) for op in trace.ops if op.func_name == "tanh" and op.num_passes == 3
    )
    assert len(tanh_labels) == 3, tanh_labels
    return model, x, trace, tanh_labels


def test_substring_selector_on_a_final_label_still_refuses_live() -> None:
    model, x, trace, _tanh_labels = _recurrent_case()
    gate = next(op for op in trace.ops if op.func_name == "sigmoid")
    assert gate.num_passes == 1
    layer_label = str(gate.layer_label)
    with pytest.raises(LiveModeLabelError):
        trace.fork().do(
            tl.intervention.label_contains(layer_label),
            tl.zero_ablate(),
            model=model,
            inputs=x,
            intervention=_RERUN,
        )
