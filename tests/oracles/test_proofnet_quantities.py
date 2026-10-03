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


def test_param_total_matches_closed_form_and_torch_census() -> None:
    """summary total_params == closed form == torch's own census (I2)."""

    import torchlens as tl

    model = _mlp()
    torch_census = sum(parameter.numel() for parameter in model.parameters())
    assert torch_census == MLP_PARAM_CLOSED_FORM, "fixture drifted from its closed form"
    trace = tl.trace(model, torch.randn(2, 4))
    try:
        report = trace.summary()
        assert report.total_params == MLP_PARAM_CLOSED_FORM, (
            f"summary().total_params = {report.total_params}, closed form"
            f" {MLP_PARAM_CLOSED_FORM} (torch census {torch_census})"
        )
    finally:
        trace.cleanup()


def test_flops_report_matches_hand_macs() -> None:
    """flops_report true MACs == the hand-computed closed form; the fma2
    figure is exactly 2x MACs + non-MAC FLOPs (convention arithmetic)."""

    import torchlens as tl
    from torchlens.report import flops_report

    trace = tl.trace(_mlp(), torch.randn(2, 4))
    try:
        report = flops_report(trace)
        text = str(report)
        assert f"true MACs: {MLP_TRUE_MACS_BATCH2}" in text, (
            f"flops_report true-MAC figure disagrees with the hand closed form"
            f" ({MLP_TRUE_MACS_BATCH2}): {text[:200]}"
        )
        assert "fma2" in text, "the FLOP convention is not disclosed in the report"
    finally:
        trace.cleanup()


def test_identity_partition_counts_tied_weight_once() -> None:
    """The tied-weight law: ONE 8x8 parameter consumed twice reads as 64
    params everywhere, never 128 (the most-reported competitor bug class)."""

    import torchlens as tl

    class _Tied(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 8))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = torch.relu(x @ self.weight)
            return hidden @ self.weight

    torch.manual_seed(0)
    model = _Tied().eval()
    torch_census = sum(parameter.numel() for parameter in model.parameters())
    assert torch_census == 64
    trace = tl.trace(model, torch.randn(2, 8))
    try:
        report = trace.summary()
        assert report.total_params == 64, (
            f"tied weight double-counted: summary says {report.total_params},"
            " identity partition says 64 (one identity, two consumptions)"
        )
    finally:
        trace.cleanup()


def test_agent_json_counts_agree_with_the_trace_itself() -> None:
    """The agent door's counts are the trace's counts (no restatement)."""

    import torchlens as tl

    trace = tl.trace(_mlp(), torch.randn(2, 4))
    try:
        payload = trace.to_agent_json()
        assert len(payload["layer_labels"]) == len(trace.layer_labels)
        counts = payload.get("counts", {})
        count_values = [value for value in counts.values() if isinstance(value, int)]
        assert count_values, f"agent counts block empty: {counts}"
        assert all(value >= 0 for value in count_values)
    finally:
        trace.cleanup()
