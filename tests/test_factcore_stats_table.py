"""C02 stats view substrate (summary(view="stats")) + health-scan sync batching (sumfam 17-18)."""

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.report._stats_table import STATS_ROW_STATES


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()


def test_stats_table_typed_states_never_hollow_zero() -> None:
    """Unsaved payloads get a typed state + reason, never fabricated stats."""

    trace = tl.trace(_model(), torch.randn(2, 4), save=tl.func("relu"))
    table = trace.summary(view="stats")
    states = {row.state for row in table.rows}
    assert "unsaved" in states and "ok" in states
    for row in table.rows:
        assert row.state in STATS_ROW_STATES
        if row.state != "ok":
            assert row.stats is None
            assert row.reason
    saved_labels = {row.label for row in table.rows if row.state == "ok"}
    assert any("relu" in label for label in saved_labels)


def test_stats_view_text_is_the_table_and_pandas_attrs() -> None:
    """str(report) is the disclosed table; the pandas projection carries the policy attrs."""

    trace = tl.trace(_model(), torch.randn(2, 4))
    table = trace.summary(view="stats")
    text = str(table)
    assert text.startswith("Payload statistics (op grain; rows in graph (execution) order)")
    assert "scan policy: summary(view='stats') scans every RETAINED payload" in text
    assert text.isascii()
    header_line = text.splitlines()[5]
    assert header_line.split()[:3] == ["name", "state", "n"]
    assert all(row.label in text for row in table.rows)
    pd = pytest.importorskip("pandas")
    frame = table.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert frame.attrs["basis"].startswith("observations of ONE captured batch")
    assert "scan_cost_policy" in frame.attrs
    assert len(frame) == len(table.rows)


def test_stats_view_budget_only_with_stats_view() -> None:
    """max_scan_elements= belongs to the stats view; any other view refuses typed."""

    from torchlens._errors import InvalidArgumentError

    trace = tl.trace(_model(), torch.randn(2, 4))
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.summary(max_scan_elements=10)
    assert excinfo.value.fields["code"] == "summary_option_invalid"
    budgeted = trace.summary(view="stats", max_scan_elements=4)
    states = [row.state for row in budgeted.rows]
    assert "unscanned" in states
    assert "budget: max_scan_elements=4" in str(budgeted)


def test_stats_view_module_grain_and_columns() -> None:
    """level='module' keeps the module-output payloads; columns= picks stats columns."""

    from torchlens._errors import InvalidArgumentError

    trace = tl.trace(_model(), torch.randn(2, 4))
    op_rows = trace.summary(view="stats").rows
    module_rows = trace.summary("module", view="stats").rows
    assert 0 < len(module_rows) < len(op_rows)
    assert {row.label for row in module_rows} <= {row.label for row in op_rows}
    narrow = trace.summary(view="stats", columns=["mean", "sd"])
    header_line = str(narrow).splitlines()[5]
    assert header_line.split() == ["name", "state", "mean", "sd"]
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.summary(view="stats", columns=["params"])
    assert excinfo.value.fields["code"] == "summary_option_invalid"
    assert "| name | state |" in narrow.to_markdown()
    assert narrow.to_dict()["rows"][0]["state"]
