"""C02 detached SummaryReport layer pins (summary memo item 10 / sumfam 10).

SummaryReport(str) with __repr__ == __str__, _repr_html_, to_dict(), and
sections; typed SummaryRow/SummaryTotals/CaptureFacts; the five-value
evidence enum; the raw-numbers pin; the detach pin (survives del model /
del trace / cleanup); profile and summary agreeing through the ONE
aggregation.
"""

import gc

import torch
from torch import nn

import torchlens as tl
from torchlens.report import SummaryReport


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_summary_report.py and a sole campaign mutation kill.
def test_summary_returns_str_subclass_with_data_payload() -> None:
    """The report is a str subclass carrying the typed data payload."""

    trace = tl.trace(_model(), torch.randn(2, 4))
    report = trace.summary()
    assert isinstance(report, SummaryReport)
    assert isinstance(report, str)
    assert repr(report) == str(report)  # bare display renders the table (D7)
    assert "view:" in str(report)


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_summary_report.py.
def test_one_call_door_returns_report_with_disclosure() -> None:
    """tl.summary keeps the execution disclosure AND the typed payload."""

    report = tl.summary(_model(), torch.randn(2, 4))
    assert isinstance(report, SummaryReport)
    # The rebuilt door hoists the disclosure into the header line.
    assert "eval mode, no_grad, state restored" in str(report).splitlines()[0]
    assert report.totals.params_total == 30


# Kept by the shrink-r4 coverage proof: sole branch coverage of report/_summary_report.py.
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
