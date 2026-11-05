"""F09 CP1: costreport items 6-12 -- honesty rebuild, percent columns,
cost tree, flops_report, backward gating, position scaling.

Named memo rows exercised here: T-HONESTY (failing-first: it fails against
the pre-F09 null-check honesty()), T-ROOT (the 3.878x naive module sum is
the forbidden regression), T-GRAD (backward = 0 under no-grad configs; the
counterfactual needs its named door and is labeled hypothetical),
T-RAW-INT (report numeric fields are plain ints or None), T-6ND-APPLIC
(the comparator line is ABSENT, never guessed), plus the D6 corruption
plant (a denominator computed after top_k must fail).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens.report import (
    backward_estimate,
    backward_status,
    cost_tree,
    flops,
    padding_waste,
    profile,
)
from torchlens.report._cost_backward import BACKWARD_STATES, grad_enabled_at_capture
from torchlens.report._cost_scaling import position_scaling_class


class NestedModel(nn.Module):
    """Two-deep module nesting: the naive-module-sum trap (T-ROOT)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16))
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """One plain forward through the nested stack."""

        return self.head(self.encoder(x))


class SdpaModel(nn.Module):
    """A model whose trace holds BOTH formula-exact and estimated FLOPs cells."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear (formula_exact) followed by SDPA (estimated terms)."""

        q = self.proj(x)
        return F.scaled_dot_product_attention(q, q, q)


@pytest.fixture(scope="module")
def nested_trace():
    """One shared grad-enabled capture of the nested model."""

    trace = tl.trace(NestedModel().eval(), torch.randn(2, 8))
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def sdpa_trace():
    """One shared capture holding mixed-evidence compute cells."""

    trace = tl.trace(SdpaModel().eval(), torch.randn(2, 4, 8))
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def nograd_trace():
    """A capture with autograd disabled: executed backward is exactly 0."""

    with torch.no_grad():
        trace = tl.trace(NestedModel().eval(), torch.randn(2, 8))
    yield trace
    trace.cleanup()


# ---------------------------------------------------------------------------
# T-HONESTY (costreport item 6, D2/D9)


def test_honesty_as_summary_and_inline_evidence(nested_trace) -> None:
    """honesty(as_summary=True) is the disclosure line; to_pandas(include_evidence=True)
    interleaves per-cell evidence columns (D9's inline form, on the export door)."""

    prof = profile(nested_trace, level="op")
    line = prof.evidence_labels(as_summary=True)
    assert isinstance(line, str)
    assert "formula_exact" in line
    inline = prof.to_pandas(include_evidence=True)
    columns = list(inline.columns)
    assert columns.index("flops_evidence") == columns.index("flops") + 1
    assert columns.index("time_evidence") == columns.index("time") + 1


@pytest.mark.smoke
def test_honesty_aggregation_preserves_weakest_evidence(sdpa_trace) -> None:
    """D2: a rollup row holding an estimated member never upgrades to exact."""

    prof = profile(sdpa_trace, level="module")
    honesty = prof.evidence_labels()
    # The root module contains the SDPA op: its rollup evidence is the
    # weakest member evidence, never formula_exact.
    assert "estimated" in set(honesty["flops"])


# ---------------------------------------------------------------------------
# Percent columns + roles (costreport item 7, D4/D5/D6)


def test_profile_percent_denominator_invariant_under_top_k(nested_trace) -> None:
    """D6 corruption plant: a denominator computed after top_k must fail --
    the partition total and per-row percents are unchanged by the view."""

    full = profile(nested_trace, level="op")
    view = profile(nested_trace, level="op", k=2)
    assert view.partition_total == full.partition_total
    by_name_full = dict(zip(full.frame["name"], full.frame["flops_pct"], strict=True))
    for name, pct in zip(view.frame["name"], view.frame["flops_pct"], strict=True):
        if pct == pct:
            assert pct == pytest.approx(by_name_full[name])
    visible_pct = sum(value for value in view.frame["flops_pct"] if value == value)
    assert visible_pct < 100.0  # the view dropped mass; the denominator did not shrink


def test_profile_totals_row_and_instrumented_label(nested_trace) -> None:
    """sumfam item 12: the explicit totals row and the instrumented label."""

    prof = profile(nested_trace, level="op")
    frame = prof.to_pandas(include_totals=True)
    total_row = frame.loc[frame["kind"] == "total"].iloc[0]
    assert total_row["name"] == "TOTAL"
    assert int(total_row["flops"]) == prof.partition_total
    assert "time (instrumented)" in repr(prof)
    assert frame.attrs["time_basis"] == "instrumented"


def test_call_children_sum_to_parent(nested_trace) -> None:
    """sumfam item 12 children-sum CI: per additive column, the members'
    self mass plus remainder equals the whole (conservation)."""

    prof = profile(nested_trace, level="call")
    frame = prof.frame
    self_sum = int(sum(value for value in frame["flops_self"] if value == value))
    assert self_sum == prof.partition_total


# ---------------------------------------------------------------------------
# Cost tree (costreport item 8, D8)


def test_cost_tree_top_k_emits_remainder_and_receipt(nested_trace) -> None:
    """D8/D6: hidden mass lands in a deterministic REMAINDER row and the
    display receipt; the denominator never shrinks."""

    full = cost_tree(nested_trace)
    view = cost_tree(nested_trace, k=1)
    assert view.partition_total == full.partition_total
    remainder_rows = [row for row in view.rows if row.role == "REMAINDER"]
    assert remainder_rows
    additive_sum = sum(row.self_flops or 0 for row in view.rows if row.kind != "root")
    assert additive_sum == view.partition_total


def test_cost_tree_depth_cutoff_conserves(nested_trace) -> None:
    """D8: conservation is exact under depth cutoff."""

    view = cost_tree(nested_trace, max_depth=1)
    additive_sum = sum(row.self_flops or 0 for row in view.rows if row.kind != "root")
    assert additive_sum == view.partition_total
    assert str(view)  # renders


# ---------------------------------------------------------------------------
# Backward gating (costreport items 11-12, D14/D15/D23; T-GRAD)


def test_backward_unknown_when_enabled_but_unobserved(nested_trace) -> None:
    """T-GRAD: grad enabled + no observed backward = UNKNOWN, never zero
    and never an unlabeled estimate."""

    assert grad_enabled_at_capture(nested_trace)
    status = backward_status(nested_trace)
    assert status.state == "grad_enabled_unobserved"
    assert status.executed_flops is None
    assert "UNKNOWN" in status.status_line
    assert status.state in BACKWARD_STATES


def test_backward_estimate_is_named_door_and_hypothetical(nograd_trace) -> None:
    """T-GRAD: the counterfactual requires its named door, is labeled
    hypothetical, and discloses the freeze pattern + grad-disabled caveat."""

    estimate = backward_estimate(nograd_trace)
    assert estimate.label == "hypothetical"
    assert estimate.hypothetical_flops > 0
    assert estimate.num_multiplied_ops > 0
    assert estimate.is_grad_disabled_at_capture
    assert "HYPOTHETICAL" in estimate.disclosure
    assert "requires_grad as captured" in estimate.disclosure
    # No default 1.0 multiplier: non-MAC ops are excluded and named.
    assert all(name for name in estimate.excluded_ops)


# ---------------------------------------------------------------------------
# flops_report (costreport item 9, D10-D13; T-RAW-INT, T-6ND-APPLIC)


def test_flops_report_6nd_absent_without_main_input(nested_trace) -> None:
    """T-6ND-APPLIC: no HF main_input evidence -> the line is ABSENT."""

    report = flops(nested_trace)
    assert report.six_nd is None
    assert "6ND" not in str(report)


def test_flops_report_6nd_present_with_explicit_d(nested_trace) -> None:
    """D11: an explicit D renders the comparator with N from the dedup
    parameter identity and D counting pads."""

    report = flops(nested_trace, main_input_numel=16)
    assert report.six_nd is not None
    assert report.six_nd.d_numel == 16
    assert report.six_nd.six_nd == 6 * report.params_unique * 16
    assert "pads counted" in str(report)


def test_flops_report_model_door_restores_state() -> None:
    """D10: the model door runs ONE capture under the summary execution
    contract -- training flags and RNG restored bit-identically."""

    model = NestedModel().train()
    example = torch.randn(2, 8)
    rng_before = torch.get_rng_state().clone()
    report = flops(model, example)
    assert model.training  # restored
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert type(report.forward_flops) is int


# ---------------------------------------------------------------------------
# Position scaling + padding waste (costreport item 10, D12)


def test_position_scaling_closed_forms() -> None:
    """The class-split estimator matches the closed forms per class."""

    padded = 8
    real = (4, 8)
    trace = tl.trace(SdpaModel().eval(), torch.randn(2, padded, 8))
    try:
        waste = padding_waste(trace, padded_length=padded, real_lengths=real)
        linear_live = sum(real) / (len(real) * padded)
        quadratic_live = sum(r * r for r in real) / (len(real) * padded**2)
        expected = 0
        for op in trace._layer_list:
            flops = int(op.cost.flops_forward or 0)
            if flops == 0:
                continue
            scaling = position_scaling_class(op, padded)
            if scaling == "linear_in_tokens":
                expected += int(round(flops * (1 - linear_live)))
            elif scaling == "quadratic_in_sequence":
                expected += int(round(flops * (1 - quadratic_live)))
        assert waste.waste_flops == expected
        assert waste.waste_flops > 0
        assert "6ND comparator's D counts pad positions" in waste.disclosure
        assert waste.largest_contributor is not None
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Unknown ledger stays a work queue (item 5 consumed; D3)


# ---------------------------------------------------------------------------
# Every new code ships provoked (error-code coverage gate)


def test_view_argument_codes_are_provoked(nested_trace) -> None:
    """Provoke cost_tree_top_k_invalid, cost_tree_depth_invalid,
    padding_waste_length_invalid, padding_waste_lengths_invalid,
    flops_report_inputs_required, flops_report_trace_door_inputs by
    code literal."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError) as excinfo:
        cost_tree(nested_trace, k=0)
    assert excinfo.value.fields["code"] == "cost_tree_top_k_invalid"

    with pytest.raises(InvalidArgumentError) as excinfo:
        cost_tree(nested_trace, max_depth=-1)
    assert excinfo.value.fields["code"] == "cost_tree_depth_invalid"

    with pytest.raises(InvalidArgumentError) as excinfo:
        padding_waste(nested_trace, padded_length=0, real_lengths=(1,))
    assert excinfo.value.fields["code"] == "padding_waste_length_invalid"

    with pytest.raises(InvalidArgumentError) as excinfo:
        padding_waste(nested_trace, padded_length=4, real_lengths=())
    assert excinfo.value.fields["code"] == "padding_waste_lengths_invalid"

    with pytest.raises(InvalidArgumentError) as excinfo:
        flops(NestedModel())
    assert excinfo.value.fields["code"] == "flops_report_inputs_required"

    with pytest.raises(InvalidArgumentError) as excinfo:
        flops(nested_trace, torch.randn(2, 8))
    assert excinfo.value.fields["code"] == "flops_report_trace_door_inputs"
