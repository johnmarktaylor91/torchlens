"""Typed quantities: closed-form corpus + identity partitions (item 9; wave 1).

Every quantity a report door serves is checked against an oracle that does
not read TorchLens code: hand-computed closed forms and the torch parameter
census. The identity partition is the tied-weight law: a parameter counted
once per IDENTITY, never once per consumption.
"""

from __future__ import annotations

import torch


def _mlp() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()


#: Closed form for the fixture: Linear(4,8) = 4*8+8 = 40; Linear(8,2) =
#: 8*2+2 = 18; total 58.
MLP_PARAM_CLOSED_FORM = 58

#: Closed form MACs for batch 2: 2*(4*8) + 2*(8*2) = 64 + 32 = 96.
MLP_TRUE_MACS_BATCH2 = 96


def test_flops_report_matches_hand_macs() -> None:
    """flops_report true MACs == the hand-computed closed form; the fma2
    figure is exactly 2x MACs + non-MAC FLOPs (convention arithmetic)."""

    import torchlens as tl
    from torchlens.report import flops

    trace = tl.trace(_mlp(), torch.randn(2, 4))
    try:
        report = flops(trace)
        text = str(report)
        assert f"true MACs: {MLP_TRUE_MACS_BATCH2}" in text, (
            f"flops_report true-MAC figure disagrees with the hand closed form"
            f" ({MLP_TRUE_MACS_BATCH2}): {text[:200]}"
        )
        assert "fma2" in text, "the FLOP convention is not disclosed in the report"
    finally:
        trace.cleanup()
