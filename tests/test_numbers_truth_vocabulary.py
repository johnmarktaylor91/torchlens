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
        capture=tl.options.CaptureOptions(layers_to_save=None),
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

    layer = three_pass_log["linear_1_1"]
    text = str(layer)
    assert "(x3 passes)" in text
    assert "(3 ops)" not in text


def test_summary_rows_never_spell_passes_as_ops(three_pass_log: tl.Trace) -> None:
    """Summary rows never spell pass multiplicity as an op count."""

    for text in (str(three_pass_log.summary()), str(three_pass_log.summary(level="op"))):
        assert "(3 ops)" not in text
    # The op denominator stays the executed-op count (3 passes of one layer).
    assert three_pass_log.num_ops == 3


def test_op_str_keeps_pass_vocabulary(three_pass_log: tl.Trace) -> None:
    """Per-pass Op str already spells "(pass k/N)"; pin it so it stays."""

    op = three_pass_log["linear_1_1:2"]
    text = str(op)
    assert "(pass 2/3)" in text
    assert " ops)" not in text.split("\n")[0]


def test_headings_follow_row_kind(three_pass_log: tl.Trace) -> None:
    """The summary name column never heads "Layer"."""

    # The merged name column heads "name (type)", never "Layer".
    rebuilt = three_pass_log.summary()
    assert "name (type)" in rebuilt
    assert "Layer " not in rebuilt
