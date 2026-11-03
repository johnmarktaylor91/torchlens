"""Every node title the default drawing shows is a key of the trace.

The title of a single-pass layer used to rebuild its ordinal from the op's
``step_index``, which postprocess renumbers per op after labeling. In a
recurrent graph the two diverge, so the drawing named ``sum_1_14`` a layer the
trace calls ``sum_1_5``. Titles now read the layer label itself.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

import torchlens as tl

# No smoke mark: the smoke tier sits at its size ceiling (tests/conftest.py).

_TITLE = re.compile(r"<B>([a-z][a-z0-9_]*?_[0-9]+(?:_[0-9]+)?)(?::([0-9]+)| \(x[0-9]+\))?</B>")


class TinyRNN(nn.Module):
    """A four-step recurrence followed by a recorded conditional and a readout."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)
        self.boost = nn.Linear(4, 4)
        self.readout = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros_like(x)
        for _ in range(4):
            h = torch.tanh(self.cell(x + h))
        if h.sum() > 0:
            h = self.boost(h)
        else:
            h = -h
        return self.readout(h)


@pytest.mark.parametrize("rolled", [False, True], ids=["unrolled", "rolled"])
def test_every_drawn_node_title_is_a_trace_key(tmp_path, rolled: bool) -> None:
    torch.manual_seed(0)
    trace = tl.trace(TinyRNN(), torch.randn(1, 4))
    dot = trace.draw(rolled=rolled, path=str(tmp_path / "graph")).dot
    titles = _TITLE.findall(dot)
    assert len(titles) >= 6, dot
    layer_labels = set(trace.layer_labels)
    for base, pass_number in titles:
        assert base in layer_labels, (base, sorted(layer_labels))
        if pass_number:
            assert trace[f"{base}:{pass_number}"].layer_label == base
    # The layers after the loop carry their own ordinals, not op positions.
    single_pass = {base for base, pass_number in titles if not pass_number}
    assert {label for label in layer_labels if label.startswith("sum_")} <= single_pass
