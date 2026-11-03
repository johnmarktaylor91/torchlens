"""Zero-arithmetic reclassification + the named unknown ledger (costreport D2/D3).

Lane A07 (2026-08-27). Two-sided witnesses: every reclassified
construction name is proven zero-by-rule, AND ambiguous names (RNG draws,
interpolating fills) are proven to STAY unknown -- a confident wrong zero is
worse than an honest unknown. The four-way classification is total.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


@pytest.mark.smoke
def test_unknown_ledger_names_reasons_and_remedies() -> None:
    """The unknown ledger carries counts, examples, shapes, and the remedy."""

    class _PadTwice(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(torch.nn.functional.pad(x, (1, 0)), (0, 1))

    log = tl.trace(_PadTwice(), torch.randn(2, 3), save=None)
    try:
        ledger = log.unknown_flop_ops
        assert len(ledger) == 1
        group = ledger[0]
        assert group.func_name == "pad"
        assert group.count == 2
        assert len(group.example_labels) == 2
        assert all(shape is not None for shape in group.example_shapes)
        assert "register_op_rule('pad'" in group.remedy
        # The footer: unknown count, name, and the remedy survive.
        rebuilt = log.summary()
        assert "pad" in rebuilt
        assert "register_op_rule" in rebuilt
    finally:
        log.cleanup()


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/compute_record.py, _data_classes/trace.py, report/_trace_facts.py.
@pytest.mark.heavy
def test_gpt2_reaches_full_coverage_sentence() -> None:
    """gpt2@16: '541/541 covered; 25 zero-by-rule; 0 unknown' (costreport D3)."""

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config())
    model.eval()
    ids = torch.randint(0, 50257, (1, 16))
    with torch.no_grad():
        log = tl.trace(model, ids, save=None)
    try:
        coverage = log.compute_coverage
        assert coverage["unknown"] == 0
        assert log.unknown_flop_ops == ()
        # The 25 former "unknowns" (tensor x24, arange x1) are zero-by-rule.
        assert coverage["zero_by_rule"] >= 25
        assert coverage["known"] + coverage["zero_by_rule"] == log._num_ops
        # Reclassification adds ZERO FLOPs: the partition pin is unchanged.
        assert int(log.cost.total_flops_forward) == 3_974_725_648
    finally:
        log.cleanup()
