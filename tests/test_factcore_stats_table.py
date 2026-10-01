"""C02 stats_table substrate + health-scan sync batching (sumfam 17-18)."""

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.report import STATS_ROW_STATES, StatsTable

pytestmark = pytest.mark.smoke


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)).eval()


def test_stats_table_rows_carry_kernel_numbers() -> None:
    """ok rows serve TensorStats numbers; identities are stable."""

    trace = tl.trace(_model(), torch.randn(2, 4))
    table = trace.stats_table()
    assert isinstance(table, StatsTable)
    assert len(table.rows) == len(trace.layer_list)
    ok_rows = table.ok_rows
    assert ok_rows
    for row in ok_rows:
        assert row.stats is not None
        assert row.stats.numel > 0
        assert row.row_id == f"op:{row.label}"
    relu_row = next(row for row in table.rows if "relu" in row.label)
    payload = trace[relu_row.label.split(":")[0]].out
    assert relu_row.stats.zero_count == int((payload == 0).sum())
    assert table.basis.startswith("observations of ONE captured batch")


def test_stats_table_typed_states_never_hollow_zero() -> None:
    """Unsaved payloads get a typed state + reason, never fabricated stats."""

    trace = tl.trace(_model(), torch.randn(2, 4), save=tl.func("relu"))
    table = trace.stats_table()
    states = {row.state for row in table.rows}
    assert "unsaved" in states and "ok" in states
    for row in table.rows:
        assert row.state in STATS_ROW_STATES
        if row.state != "ok":
            assert row.stats is None
            assert row.reason
    saved_labels = {row.label for row in table.rows if row.state == "ok"}
    assert any("relu" in label for label in saved_labels)


def test_stats_table_repr_bounded_and_pandas_attrs() -> None:
    """Designed bounded repr; pandas projection carries the policy attrs."""

    trace = tl.trace(_model(), torch.randn(2, 4))
    table = trace.stats_table()
    text = repr(table)
    assert text.startswith("StatsTable(")
    assert "\n" not in text and len(text) < 300
    pd = pytest.importorskip("pandas")
    frame = table.to_pandas()
    assert isinstance(frame, pd.DataFrame)
    assert frame.attrs["basis"] == table.basis
    assert "scan_cost_policy" in frame.attrs
    assert len(frame) == len(table.rows)


def test_full_health_scan_batches_syncs() -> None:
    """sumfam item 17: the full scan never takes the per-op sync path.

    Work counts, never milliseconds: the sequential single-payload verdict
    (one host read per op) must not run at all on the full-scan path; the
    stop-at-first path keeps it (the first sync IS its point).
    """

    from torchlens.data_classes import _nonfinite

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

    from torchlens.data_classes._nonfinite import _has_nonfinite

    x = torch.randn(2, 4)
    x[0, 1] = float("inf")
    trace = tl.trace(_model(), x)
    table_hits = {
        row.label
        for row in trace.stats_table().rows
        if row.stats is not None and row.stats.nonfinite_count > 0
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
