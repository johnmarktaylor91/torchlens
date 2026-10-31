"""C02 stats view substrate (summary(view="stats")) + health-scan sync batching (sumfam 17-18)."""

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.report import SummaryReport
from torchlens.report._stats_table import STATS_ROW_STATES


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()


def test_stats_table_rows_carry_kernel_numbers() -> None:
    """ok rows serve TensorStats numbers; identities are stable."""

    trace = tl.trace(_model(), torch.randn(2, 4))
    table = trace.summary(view="stats")
    assert isinstance(table, SummaryReport)
    assert len(table.rows) == len(trace.layer_list)
    ok_rows = [row for row in table.rows if row.state == "ok"]
    assert ok_rows
    for row in ok_rows:
        assert row.stats is not None
        assert row.stats.numel > 0
        assert row.row_id == f"op:{row.label}"
    relu_row = next(row for row in table.rows if "relu" in row.label)
    payload = trace[relu_row.label.split(":")[0]].out
    assert relu_row.stats.num_zero == int((payload == 0).sum())
    assert "basis: observations of ONE captured batch" in str(table)


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


def test_full_health_scan_batches_syncs() -> None:
    """sumfam item 17: the full scan never takes the per-op sync path.

    Work counts, never milliseconds: the sequential single-payload verdict
    (one host read per op) must not run at all on the full-scan path; the
    stop-at-first path keeps it (the first sync IS its point).
    """

    from torchlens._data_classes import _nonfinite

    x = torch.randn(2, 4)
    x[0, 0] = float("nan")
    trace = tl.trace(_model(), x)

    calls = {"sequential": 0}
    original = _nonfinite._has_nonfinite

    def counting(out):
        calls["sequential"] += 1
        return original(out)

    _nonfinite._has_nonfinite = counting
    try:
        labels = _nonfinite.nonfinite_op_labels(trace)
        assert labels  # the poison was found by the batched path
        assert calls["sequential"] == 0, "full scan took the per-op sync path"
        _nonfinite.first_nonfinite_layer(trace)
        assert calls["sequential"] > 0  # stop-at-first stays sequential
    finally:
        _nonfinite._has_nonfinite = original


def test_batched_and_sequential_scans_agree() -> None:
    """The batched verdicts equal the per-op verdicts payload for payload."""

    from torchlens._data_classes._nonfinite import _has_nonfinite

    x = torch.randn(2, 4)
    x[0, 1] = float("inf")
    trace = tl.trace(_model(), x)
    table_hits = {
        row.label
        for row in trace.summary(view="stats").rows
        if row.stats is not None and (row.stats.num_nonfinite or 0) > 0
    }
    scan_hits = set(trace.nonfinite_ops)
    sequential_hits = {
        str(op.label)
        for op in trace.layer_list
        if isinstance(getattr(op, "out", None), torch.Tensor)
        and op.out.numel() > 0
        and _has_nonfinite(op.out)
    }
    assert scan_hits
    assert scan_hits <= sequential_hits  # every batched hit is a true per-op hit
    assert table_hits >= scan_hits  # the kernel's exact census agrees


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
