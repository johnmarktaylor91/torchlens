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
    BACKWARD_STATES,
    backward_estimate,
    backward_status,
    build_profile,
    compute_aggregation,
    cost_tree,
    flops_report,
    grad_enabled_at_capture,
    padding_waste,
    position_scaling_class,
)

pytestmark = pytest.mark.smoke


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


def test_honesty_labels_mixed_evidence_differently(sdpa_trace) -> None:
    """T-HONESTY: honesty() is NOT a function of frame.notna().

    One formula_exact and one estimated non-null value in the same flops
    column must carry different labels. This fails against the pre-F09
    presence-based honesty() (every non-null flops cell read 'estimated').
    """

    profile = build_profile(sdpa_trace, level="op")
    honesty = profile.honesty()
    labels = set(honesty["flops"])
    assert "formula_exact" in labels, labels
    assert "estimated" in labels, labels


def test_honesty_vocabulary_and_qualifiers(nested_trace) -> None:
    """D9 column rules: time is never bare 'measured'; params are declared
    inventory; boundary rows are not_applicable, never fabricated."""

    profile = build_profile(nested_trace, level="op")
    honesty = profile.honesty()
    assert set(honesty["time"]) <= {
        "measured+instrumentation_inclusive",
        "unknown",
        "not_applicable",
    }
    assert "measured" not in set(honesty["time"])
    assert set(honesty["param_count"]) <= {"formula_exact", "not_applicable"}
    boundary = honesty.loc[honesty["name"].str.startswith("input_")].iloc[0]
    assert boundary["flops"] == "not_applicable"
    assert boundary["time"] == "not_applicable"


def test_honesty_as_summary_and_inline_evidence(nested_trace) -> None:
    """honesty(as_summary=True) is the disclosure line; to_pandas(evidence=True)
    interleaves per-cell evidence columns (D9's inline form, on the export door)."""

    profile = build_profile(nested_trace, level="op")
    line = profile.honesty(as_summary=True)
    assert isinstance(line, str)
    assert "formula_exact" in line
    inline = profile.to_pandas(evidence=True)
    columns = list(inline.columns)
    assert columns.index("flops_evidence") == columns.index("flops") + 1
    assert columns.index("time_evidence") == columns.index("time") + 1


def test_honesty_aggregation_preserves_weakest_evidence(sdpa_trace) -> None:
    """D2: a rollup row holding an estimated member never upgrades to exact."""

    profile = build_profile(sdpa_trace, level="module")
    honesty = profile.honesty()
    # The root module contains the SDPA op: its rollup evidence is the
    # weakest member evidence, never formula_exact.
    assert "estimated" in set(honesty["flops"])


# ---------------------------------------------------------------------------
# Percent columns + roles (costreport item 7, D4/D5/D6)


def test_profile_percent_columns_sum_to_100_of_known(nested_trace) -> None:
    """D4: exclusive percents sum to 100% of the KNOWN total."""

    profile = build_profile(nested_trace, level="op")
    total_pct = sum(value for value in profile.frame["flops_pct"] if value == value)
    assert total_pct == pytest.approx(100.0)


def test_profile_percent_denominator_invariant_under_top_k(nested_trace) -> None:
    """D6 corruption plant: a denominator computed after top_k must fail --
    the partition total and per-row percents are unchanged by the view."""

    full = build_profile(nested_trace, level="op")
    view = build_profile(nested_trace, level="op", top_k=2)
    assert view.partition_total == full.partition_total
    by_name_full = dict(zip(full.frame["name"], full.frame["flops_pct"], strict=True))
    for name, pct in zip(view.frame["name"], view.frame["flops_pct"], strict=True):
        if pct == pct:
            assert pct == pytest.approx(by_name_full[name])
    visible_pct = sum(value for value in view.frame["flops_pct"] if value == value)
    assert visible_pct < 100.0  # the view dropped mass; the denominator did not shrink


def test_profile_roles_and_additivity_metadata(nested_trace) -> None:
    """D5: row roles and column additivity ride every export."""

    op_frame = build_profile(nested_trace, level="op").to_pandas()
    assert set(op_frame["role"].dropna()) == {"OWNER"}
    assert op_frame.attrs["column_additivity"]["flops"] is True
    module_profile = build_profile(nested_trace, level="module")
    module_frame = module_profile.to_pandas()
    assert set(module_frame["role"].dropna()) == {"SUBTOTAL"}
    # Inclusive family is marked non-additive at module level.
    assert module_frame.attrs["column_additivity"]["flops"] is False
    assert module_frame.attrs["column_additivity"]["flops_self"] is True


def test_module_self_family_sums_to_partition_total_not_naive(nested_trace) -> None:
    """T-ROOT at module grain: the additive self family sums to the
    partition total; the naive inclusive sum over-counts (forbidden)."""

    profile = build_profile(nested_trace, level="module")
    frame = profile.frame
    partition_total = profile.partition_total
    self_sum = int(sum(value for value in frame["flops_self"] if value == value))
    assert self_sum == partition_total
    naive_sum = int(sum(value for value in frame["flops"].dropna()))
    assert naive_sum > partition_total  # nested modules double-count inclusively


def test_profile_totals_row_and_instrumented_label(nested_trace) -> None:
    """sumfam item 12: the explicit totals row and the instrumented label."""

    profile = build_profile(nested_trace, level="op")
    frame = profile.to_pandas(include_totals=True)
    total_row = frame.loc[frame["kind"] == "total"].iloc[0]
    assert total_row["name"] == "TOTAL"
    assert int(total_row["flops"]) == profile.partition_total
    assert "time (instrumented)" in repr(profile)
    assert frame.attrs["time_basis"] == "instrumented"


def test_call_children_sum_to_parent(nested_trace) -> None:
    """sumfam item 12 children-sum CI: per additive column, the members'
    self mass plus remainder equals the whole (conservation)."""

    profile = build_profile(nested_trace, level="call")
    frame = profile.frame
    self_sum = int(sum(value for value in frame["flops_self"] if value == value))
    assert self_sum == profile.partition_total


# ---------------------------------------------------------------------------
# Cost tree (costreport item 8, D8)


def test_cost_tree_root_row_and_conservation(nested_trace) -> None:
    """T-ROOT: root self cells 0, subtree == partition total; the additive
    column sums to the partition total exactly."""

    tree = cost_tree(nested_trace)
    root = tree.rows[0]
    assert root.kind == "root"
    assert root.self_flops == 0
    assert root.subtree_flops == tree.partition_total
    assert root.label == "NestedModel"  # named model root, never self:1
    additive_sum = sum(row.self_flops or 0 for row in tree.rows if row.row_id != root.row_id)
    assert additive_sum == tree.partition_total
    pct_sum = sum(row.self_pct or 0.0 for row in tree.rows if row.row_id != root.row_id)
    assert pct_sum == pytest.approx(100.0)


def test_cost_tree_parent_child_conservation(nested_trace) -> None:
    """Every SUBTOTAL row: subtree == self + sum(child subtrees)."""

    tree = cost_tree(nested_trace)
    children: dict[str, list] = {}
    for row in tree.rows:
        if row.parent_id is not None:
            children.setdefault(row.parent_id, []).append(row)
    for row in tree.rows:
        if row.role != "SUBTOTAL":
            continue
        own = 0 if row.kind == "root" else (row.self_flops or 0)
        child_sum = sum(child.subtree_flops or 0 for child in children.get(row.row_id, ()))
        assert (row.subtree_flops or 0) == own + child_sum, row.row_id


def test_cost_tree_top_k_emits_remainder_and_receipt(nested_trace) -> None:
    """D8/D6: hidden mass lands in a deterministic REMAINDER row and the
    display receipt; the denominator never shrinks."""

    full = cost_tree(nested_trace)
    view = cost_tree(nested_trace, top_k=1)
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


def test_backward_zero_under_no_grad(nograd_trace) -> None:
    """T-GRAD: executed backward is exactly 0 (formula_exact) when no
    autograd graph was recorded -- never the forward-derived constant."""

    assert not grad_enabled_at_capture(nograd_trace)
    status = backward_status(nograd_trace)
    assert status.state == "grad_disabled"
    assert status.executed_flops == 0
    assert status.executed_evidence == "formula_exact"
    assert "0" in status.status_line


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
    assert estimate.n_ops_multiplied > 0
    assert estimate.grad_disabled_at_capture
    assert "HYPOTHETICAL" in estimate.disclosure
    assert "requires_grad as captured" in estimate.disclosure
    # No default 1.0 multiplier: non-MAC ops are excluded and named.
    assert all(name for name in estimate.excluded_ops)


def test_backward_status_never_prints_counterfactual(nograd_trace) -> None:
    """Memo 3.8: a counterfactual never prints in an actual-cost slot."""

    status = backward_status(nograd_trace)
    estimate = backward_estimate(nograd_trace)
    assert status.executed_flops != estimate.hypothetical_flops


# ---------------------------------------------------------------------------
# flops_report (costreport item 9, D10-D13; T-RAW-INT, T-6ND-APPLIC)


def test_flops_report_trace_door_blocks_and_raw_ints(nested_trace) -> None:
    """D10: six-block screen; T-RAW-INT: numeric fields are plain ints."""

    report = flops_report(nested_trace)
    assert type(report.forward_flops) is int
    assert type(report.true_macs) is int
    text = str(report)
    assert "actual-path analytic" in text  # D13: never "measured"
    assert "measured" not in text.split("backward")[0]
    assert "COMPLETE, not a lower bound" in text  # earned on this model
    assert "parameters:" in text
    assert "backward:" in text
    agg = compute_aggregation(nested_trace)
    assert report.forward_flops == int(agg.partition_total)


def test_flops_report_6nd_absent_without_main_input(nested_trace) -> None:
    """T-6ND-APPLIC: no HF main_input evidence -> the line is ABSENT."""

    report = flops_report(nested_trace)
    assert report.six_nd is None
    assert "6ND" not in str(report)


def test_flops_report_6nd_present_with_explicit_d(nested_trace) -> None:
    """D11: an explicit D renders the comparator with N from the dedup
    parameter identity and D counting pads."""

    report = flops_report(nested_trace, main_input_numel=16)
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
    report = flops_report(model, example)
    assert model.training  # restored
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert type(report.forward_flops) is int


def test_flops_report_trace_door_refuses_inputs(nested_trace) -> None:
    """The trace door takes no inputs; a typed refusal teaches the doors."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError, match="model-door"):
        flops_report(nested_trace, torch.randn(2, 8))


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
        for op in trace.layer_list:
            flops = int(getattr(op, "flops_forward", 0) or 0)
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


def test_padding_waste_refuses_bad_lengths(nested_trace) -> None:
    """Teaching refusals on impossible padding geometry."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError, match="padded_length"):
        padding_waste(nested_trace, padded_length=0, real_lengths=(1,))
    with pytest.raises(InvalidArgumentError, match="real_lengths"):
        padding_waste(nested_trace, padded_length=4, real_lengths=(5,))


def test_sdpa_is_quadratic_linear_is_linear(sdpa_trace) -> None:
    """The scaling table's two anchor classes."""

    ops = {str(op.layer_label): op for op in sdpa_trace.layer_list}
    sdpa_op = next(op for label, op in ops.items() if label.startswith("scaleddotproduct"))
    linear_op = next(op for label, op in ops.items() if label.startswith("linear"))
    assert position_scaling_class(sdpa_op, 4) == "quadratic_in_sequence"
    assert position_scaling_class(linear_op, 4) == "linear_in_tokens"


# ---------------------------------------------------------------------------
# Unknown ledger stays a work queue (item 5 consumed; D3)


def test_unknown_ops_disclosed_as_events_never_percent(nested_trace) -> None:
    """D4: the unknown disclosure is an event count + named ledger."""

    profile = build_profile(nested_trace, level="op")
    assert profile.unknown_events == 0
    text = repr(profile)
    assert "%" not in text.split("unknown")[-1] or "unknown" not in text


# ---------------------------------------------------------------------------
# Every new code ships provoked (error-code coverage gate)


def test_view_argument_codes_are_provoked(nested_trace) -> None:
    """Provoke cost_tree_top_k_invalid, cost_tree_depth_invalid,
    padding_waste_length_invalid, padding_waste_lengths_invalid,
    flops_report_inputs_required, flops_report_trace_door_inputs by
    code literal."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError) as excinfo:
        cost_tree(nested_trace, top_k=0)
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
        flops_report(NestedModel())
    assert excinfo.value.fields["code"] == "flops_report_inputs_required"

    with pytest.raises(InvalidArgumentError) as excinfo:
        flops_report(nested_trace, torch.randn(2, 8))
    assert excinfo.value.fields["code"] == "flops_report_trace_door_inputs"
