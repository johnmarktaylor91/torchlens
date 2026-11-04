"""One pass/op vocabulary across surfaces (A10) -- 3-pass fixture goldens.

Lane A07 (2026-08-27). Spec: the summary design memo build item 4:
"pass k/N" / "xN passes" spellings, one op denominator, headings follow row kind.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import torch
from torch import nn

import torchlens as tl


class ThreePassModel(nn.Module):
    """Recurrent fixture: one Linear applied three times (3 passes)."""

    def __init__(self) -> None:
        """Initialize the reused layer."""

        super().__init__()
        self.linear = nn.Linear(3, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same layer three times."""

        for _ in range(3):
            x = self.linear(x)
        return x


@pytest.fixture()
def three_pass_log() -> Generator[tl.Trace, None, None]:
    """Metadata-only capture of the 3-pass fixture."""

    log = tl.trace(
        ThreePassModel(),
        torch.randn(2, 3),
        save=None,
    )
    try:
        yield log
    finally:
        log.cleanup()


def test_trace_str_spells_passes_never_ops(three_pass_log: tl.Trace) -> None:
    """print(trace) spells multiplicity in passes, never "(k/N ops)".

    The trace card has no per-layer rows (the house style moves them out): it states the pass count on
    its graph row, and the per-layer rows moved to ``summary()``, which counts passes too.
    """

    text = str(three_pass_log)
    assert "\n  graph     recurrent (up to 3 passes), " in text
    assert " ops)" not in text
    assert "up to 3 passes" in str(three_pass_log.summary())


def test_layer_str_spells_passes_never_ops(three_pass_log: tl.Trace) -> None:
    """Layer str renders aggregate multiplicity as "3 passes", never "(3 ops)"."""

    layer = three_pass_log.layers["linear_1_1"]
    text = str(layer)
    assert text.splitlines()[0] == "Layer linear_1: 3 passes, float32 [2, 3]"
    assert "(3 ops)" not in text
