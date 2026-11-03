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
    """print(trace) renders multi-pass rows as "(pass k/N)", never "(k/N ops)"."""

    text = str(three_pass_log)
    assert "(pass 1/3)" in text
    assert "(pass 3/3)" in text
    assert " ops)" not in text


def test_layer_str_spells_passes_never_ops(three_pass_log: tl.Trace) -> None:
    """Layer str renders aggregate multiplicity as "(x3 passes)", never "(3 ops)"."""

    layer = three_pass_log.layers["linear_1_1"]
    text = str(layer)
    assert "(x3 passes)" in text
    assert "(3 ops)" not in text
