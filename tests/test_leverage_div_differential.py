"""B4 + B15: join-guarded differential consumers and the diff projection matrix.

``changed()`` / ``top_changed()`` pair cross-trace occurrences through the
shipped guarded join (labels leave the join key), and ``diff`` settles
the union of both captures into the closed added / removed / unresolved /
excluded_machinery / unreachable / zero / changed vocabulary. Join refusals
are the PRIMARY assertions (gate law).
"""

from __future__ import annotations

import pytest
import torch
from test_leverage_div_fixtures import (
    OptionalBranchNet,
    ReusedReluNet,
    make_insertion_pair,
)

import torchlens as tl
from torchlens.errors import SelectionError
from torchlens.errors._base import TorchLensError
from torchlens.report import diff

_SAVE_ALL = {"capture": tl.options.CaptureOptions(layers_to_save="all")}


def _plain_pair():
    torch.manual_seed(0)
    model = ReusedReluNet()
    x = torch.rand(1, 2, 3, 3)
    return model, x


# ---------------------------------------------------------------------------
# changed()/top_changed() under the join guard.
# ---------------------------------------------------------------------------


def test_changed_refuses_on_insertion_with_join_verdict():
    """PRIMARY: the guarded join's refusal surfaces through changed()."""

    baseline_model, variant_model, x = make_insertion_pair()
    baseline = tl.trace(baseline_model, x, **_SAVE_ALL)
    variant = tl.trace(variant_model, x, **_SAVE_ALL)
    with pytest.raises(SelectionError) as excinfo:
        variant[tl.changed(baseline)]
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "site_join_refused"
    assert excinfo.value.fields["join_verdict"] == "refused_cardinality"


@pytest.mark.smoke
def test_changed_completes_across_label_renumbering_live_edit():
    """G-ALIGN closed: a live edit's label churn no longer breaks changed().

    The raw-hook edit inserts one op mid-graph (renumbering downstream
    labels) and replaces conv2's output; the join pairs every structurally
    intact site by KEY, the ablated cohort's movers select, and the
    subject-only inserted op refuses one-sidedly only when addressed.
    """

    model, x = _plain_pair()
    baseline = tl.trace(model, x, **_SAVE_ALL)
    handle = model.conv2.register_forward_hook(lambda module, args, out: out * 0.0)
    edited = tl.trace(model, x, **_SAVE_ALL)
    handle.remove()
    assert list(edited.op_labels) != list(baseline.op_labels)  # labels DID churn
    shared = tl.in_module("conv1") | tl.in_module("conv2")
    resolved = edited[tl.changed(baseline, within=shared)]
    by_label = {entry.key[0]: int(entry._mask._dense_ro().sum()) for entry in resolved}
    assert by_label["conv2d_1_1"] == 0  # upstream of the edit: unmoved
    assert by_label["conv2d_2_3"] == 0  # the hook replaces AFTER conv2's own output


def test_changed_one_sided_subject_site_refuses_declared():
    """A subject-only site is a DECLARED addition: refused, never guessed."""

    model, x = _plain_pair()
    baseline = tl.trace(model, x, **_SAVE_ALL)
    handle = model.conv2.register_forward_hook(lambda module, args, out: out * 0.0)
    edited = tl.trace(model, x, **_SAVE_ALL)
    handle.remove()
    with pytest.raises(SelectionError) as excinfo:
        edited[tl.changed(baseline)]  # default population includes the mul
    assert excinfo.value.fields["reason"] == "site_join_refused"
    assert excinfo.value.fields["join_verdict"] == "one_sided_subject"


def test_changed_machinery_op_excluded_with_disclosure():
    """D-7: the engine's replacement op rides as a zero-mask disclosed entry."""

    model, x = _plain_pair()
    baseline = tl.trace(model, x, **_SAVE_ALL)
    replacement = torch.zeros(1, 2, 3, 3)
    handle = model.conv2.register_forward_hook(lambda module, args, out: replacement)
    edited = tl.trace(model, x, **_SAVE_ALL)
    handle.remove()
    machinery = [
        label
        for label in edited.op_labels
        if edited.ops[label].layer_type == "interventionreplacement"
    ]
    assert len(machinery) == 1
    resolved = edited[tl.changed(baseline)]
    machinery_entries = [
        entry for entry in resolved if "[machinery_excluded]" in entry.provenance.source
    ]
    assert len(machinery_entries) == 1
    assert int(machinery_entries[0]._mask._dense_ro().sum()) == 0


@pytest.mark.smoke
def test_top_changed_completes_across_live_edit():
    model, x = _plain_pair()
    baseline = tl.trace(model, x, **_SAVE_ALL)
    replacement = torch.ones(1, 2, 3, 3)
    handle = model.conv2.register_forward_hook(lambda module, args, out: replacement)
    edited = tl.trace(model, x, **_SAVE_ALL)
    handle.remove()
    resolved = edited[tl.top_changed(baseline, k=3)]
    assert sum(int(entry._mask._dense_ro().sum()) for entry in resolved) == 3


# ---------------------------------------------------------------------------
# diff: the guarded delta projection (B15 matrix).
# ---------------------------------------------------------------------------


def test_diff_report_insertion_matrix():
    """Touched cohort -> unresolved; untouched -> zero; statuses stay distinct."""

    baseline_model, variant_model, x = make_insertion_pair()
    baseline = tl.trace(baseline_model, x, **_SAVE_ALL)
    variant = tl.trace(variant_model, x, **_SAVE_ALL)
    report = diff(variant, baseline)
    unresolved = report.rows_with_status("unresolved")
    assert unresolved and all("|relu|" in row.site_key for row in unresolved)
    assert all(row.verdict == "refused_cardinality" for row in unresolved)
    zero_keys = {row.site_key for row in report.rows_with_status("zero")}
    assert any("conv1" in key for key in zero_keys)
    assert not report.rows_with_status("changed")


@pytest.mark.smoke
def test_diff_report_added_and_removed_are_declared():
    torch.manual_seed(0)
    without_tail = OptionalBranchNet(use_tail=False)
    with_tail = OptionalBranchNet(use_tail=True)
    with_tail.load_state_dict(without_tail.state_dict())
    x = torch.rand(1, 2, 3, 3)
    lean = tl.trace(without_tail, x, **_SAVE_ALL)
    full = tl.trace(with_tail, x, **_SAVE_ALL)
    report = diff(full, lean)
    added = report.rows_with_status("added")
    assert [row.site_key for row in added] == ["s1|tail|conv2d||1"]
    assert added[0].reference_label is None  # one-sided rows carry one side only
    reverse = diff(lean, full)
    removed = reverse.rows_with_status("removed")
    assert [row.site_key for row in removed] == ["s1|tail|conv2d||1"]
    assert removed[0].subject_label is None


def test_diff_report_unreachable_distinct_from_zero():
    """A missing payload is UNREACHABLE, never zero (the memo's distinctness law)."""

    model, x = _plain_pair()
    subject = tl.trace(model, x, **_SAVE_ALL)
    reference = tl.trace(model, x, save=tl.func("conv2d"))
    report = diff(subject, reference)
    unreachable = report.rows_with_status("unreachable")
    assert unreachable and all("payload not retained on reference" in r.detail for r in unreachable)
    assert {row.site_key for row in report.rows_with_status("zero")}.isdisjoint(
        {row.site_key for row in unreachable}
    )


def test_diff_report_zero_vs_changed_distinct():
    model, x = _plain_pair()
    baseline = tl.trace(model, x, **_SAVE_ALL)
    replacement = torch.ones(1, 2, 3, 3)
    handle = model.conv2.register_forward_hook(lambda module, args, out: replacement)
    edited = tl.trace(model, x, **_SAVE_ALL)
    handle.remove()
    report = diff(edited, baseline)
    machinery = report.rows_with_status("excluded_machinery")
    assert len(machinery) == 1  # exactly one declared machinery addition
    changed_rows = report.rows_with_status("changed")
    assert changed_rows and all(row.max_abs_delta > 0 for row in changed_rows)
    assert all(row.max_abs_delta == 0.0 for row in report.rows_with_status("zero"))


def test_diff_report_self_comparison_refuses():
    model, x = _plain_pair()
    trace = tl.trace(model, x, **_SAVE_ALL)
    with pytest.raises(TorchLensError) as excinfo:
        diff(trace, trace)
    assert excinfo.value.fields["code"] == "differential_subject_is_reference"


def test_diff_report_status_vocabulary_closed():
    model, x = _plain_pair()
    subject = tl.trace(model, x, **_SAVE_ALL)
    reference = tl.trace(model, x, **_SAVE_ALL)
    report = diff(subject, reference)
    with pytest.raises(TorchLensError) as excinfo:
        report.rows_with_status("modified")
    assert excinfo.value.fields["code"] == "differential_status_invalid"
    assert report.counts.get("zero", 0) >= 4  # a clean recapture settles to zeros
