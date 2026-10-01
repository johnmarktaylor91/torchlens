"""C02 detached SummaryReport layer pins (summary memo item 10 / sumfam 10).

SummaryReport(str) with __repr__ == __str__, _repr_html_, to_dict(), and
sections; typed SummaryRow/SummaryTotals/CaptureFacts; the five-value
evidence enum; the raw-numbers pin; the detach pin (survives del model /
del trace / cleanup); profile and summary agreeing through the ONE
aggregation.
"""

import gc

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.report import EVIDENCE_VALUES, SummaryReport, compute_aggregation

pytestmark = pytest.mark.smoke


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()


def test_summary_returns_str_subclass_byte_identical() -> None:
    """Trace.summary text is unchanged; the type gains the data payload."""

    trace = tl.trace(_model(), torch.randn(2, 4))
    report = trace.summary()
    assert isinstance(report, SummaryReport)
    assert isinstance(report, str)
    from torchlens.visualization._summary_internal import render_model_summary

    assert str(report) == render_model_summary(trace)
    assert repr(report) == str(report)  # bare display renders the table (D7)


def test_one_call_door_returns_report_with_disclosure() -> None:
    """tl.summary keeps the execution disclosure AND the typed payload."""

    report = tl.summary(_model(), torch.randn(2, 4))
    assert isinstance(report, SummaryReport)
    assert "eval" in str(report).splitlines()[-1]
    assert report.totals.params_total == 30


def test_raw_numbers_pin() -> None:
    """Every numeric field is a plain int or None (summary 3.8, binding)."""

    report = tl.trace(_model(), torch.randn(2, 4)).summary()
    totals = report.totals
    for name in (
        "params_total",
        "flops_forward_fma2",
        "macs_forward",
        "at_capture_bytes",
        "retained_now_bytes",
    ):
        value = getattr(totals, name)
        assert value is None or type(value) is int, name
        if value is not None:
            assert format(value, ",")  # digits, never human units
    for row in report.rows:
        for name in ("params_exclusive", "flops_exclusive", "macs_exclusive"):
            value = getattr(row, name)
            assert value is None or type(value) is int, (row.row_id, name)
        assert row.evidence in EVIDENCE_VALUES


def test_report_is_detached_survives_gc() -> None:
    """Pin: build report; del model and trace; the data still serves."""

    model = _model()
    trace = tl.trace(model, torch.randn(2, 4))
    report = trace.summary()
    fingerprint = report.capture.capture_fingerprint
    del model, trace
    gc.collect()
    payload = report.to_dict()
    assert payload["schema"] == "torchlens.summary_report.v1"
    assert payload["capture"]["capture_fingerprint"] == fingerprint
    assert len(payload["rows"]) == len(report.rows)
    assert "<table>" in report._repr_html_()
    assert report.sections == ("capture", "totals", "rows")


def test_report_totals_read_the_one_aggregation() -> None:
    """summary totals == the canonical compute face == profile's numbers."""

    trace = tl.trace(_model(), torch.randn(2, 4))
    report = trace.summary()
    aggregation = compute_aggregation(trace)
    assert report.totals.flops_forward_fma2 == int(aggregation.partition_total)
    assert report.totals.macs_forward == int(aggregation.macs_total)
    assert report.totals.compute_coverage["unknown"] == aggregation.coverage.unknown
    # profile consumes the SAME aggregation: its whole-trace total agrees.
    assert int(trace.total_flops_forward) == report.totals.flops_forward_fma2


def test_report_rows_carry_identity_and_evidence() -> None:
    """Rows carry stable identities, site keys, and per-cell evidence."""

    trace = tl.trace(_model(), torch.randn(2, 4))
    report = trace.summary()
    op_rows = [row for row in report.rows if row.kind == "op"]
    assert op_rows
    for row in op_rows:
        assert row.row_id.startswith("op:")
        assert row.flops_exclusive == row.flops_subtree  # families coincide at op grain
    boundary_rows = [row for row in report.rows if row.kind != "op"]
    for row in boundary_rows:
        assert row.flops_exclusive is None  # boundary rows own nothing (D7)
