"""Phase 4 multi-pass tests."""

from __future__ import annotations

import torch

import torchlens._utils as utils


# Kept by the shrink-r4 coverage proof: sole branch coverage of _utils/_onramp.py.
def test_trace_streaming_on_iterable_inputs() -> None:
    """Capture iterable inputs into one stacked root log."""

    model = torch.nn.Linear(2, 2)
    traces = utils.trace_streaming(
        model,
        [torch.ones(1, 2), torch.zeros(1, 2)],
        save="none",
    )
    assert len(traces) == 2
