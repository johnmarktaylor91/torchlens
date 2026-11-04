"""F08 result API: detached lifetime, raw-numbers pin, projections, HTML.

Summary memo 3.8 + composition rows 11/14/16: the result survives model
and trace teardown; every numeric field is a plain int or None; renderer
and projection values agree; HTML is escaped and dependency-free;
``Trace.provenance._summary()`` (interim private spelling) serves the relocated preamble byte-for-byte.
"""

from __future__ import annotations

import gc
import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError


class _Toy(nn.Module):
    """Two-Linear toy with an orphan add."""

    def __init__(self) -> None:
        """Two Linears."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fc1 -> relu -> fc2 + slice."""

        a = self.fc1(x)
        return self.fc2(torch.relu(a)) + a[:, :8]


def _fresh_report():
    """A rebuilt report whose model and trace are gone."""

    model = _Toy()
    trace = tl.trace(model, torch.randn(2, 8))
    report = trace.summary()
    trace.cleanup()
    del trace, model
    gc.collect()
    return report


def test_result_survives_model_and_trace_teardown() -> None:
    """Composition row 16: every accessor works after del model/trace."""

    report = _fresh_report()
    assert report.total_params == 280
    assert report.render("unicode")
    assert report.render("html")
    assert report.details()
    assert report.to_dict()["schema"] == "torchlens.summary_report.v1"
    frame = report.to_pandas()
    assert len(frame) == len(report._rebuilt.visible_rows)
    assert report.to_markdown().startswith("| name (type) |")


def test_raw_numbers_pin_on_scalars_and_rows() -> None:
    """Every numeric field is a plain int or None; format(x, ',') digits."""

    report = _fresh_report()
    for name in (
        "total_params",
        "executed_params",
        "unexecuted_params",
        "trainable_params",
        "frozen_params",
        "total_flops_forward",
        "total_macs_forward",
        "unknown_flop_ops",
    ):
        value = getattr(report, name)
        assert value is None or type(value) is int, name
        if value is not None:
            assert re.fullmatch(r"[\d,]+", format(value, ","))
    for row in report._rebuilt.view.rows:
        for name in ("params_display", "params_owned", "flops_display", "flops_owned"):
            value = getattr(row, name)
            assert value is None or type(value) is int, name


def test_payloadless_reports_refuse_rebuilt_methods_typed() -> None:
    """A report assembled from text (no rebuilt payload) teaches instead of half-working."""

    trace = tl.trace(_Toy(), torch.randn(2, 8))
    legacy = tl.report.summary(trace, "assembled text")
    with pytest.raises(InvalidArgumentError) as excinfo:
        legacy.to_html()
    assert excinfo.value.fields["code"] == "summary_result_legacy"
    with pytest.raises(InvalidArgumentError):
        legacy.render("unicode")
    # The canonical ASCII face still serves.
    assert legacy.render("ascii") == str(legacy)


def test_to_pandas_scope_all_serves_op_grain() -> None:
    """scope='all' projects the C02 op-grain rows."""

    report = _fresh_report()
    frame = report.to_pandas(scope="all")
    assert len(frame) == len(report.rows)
    with pytest.raises(InvalidArgumentError):
        report.to_pandas(scope="everything")


def test_parent_compute_row_uses_subtree_macs_like_flops() -> None:
    """A container's MAC cell describes the same population as its FLOP cell."""

    model = nn.Sequential(
        nn.Sequential(nn.Linear(4, 8, bias=False), nn.ReLU()),
        nn.Linear(8, 2, bias=False),
    )
    trace = tl.trace(model, torch.ones(2, 4))
    report = trace.summary(view="compute", units="raw", level="module", fold_repeats=False)
    parent = next(row for row in report._rebuilt.visible_rows if row.class_name == "Sequential")
    assert parent.macs_display == 64
    assert parent.macs_owned == 0
    line = next(line for line in str(report).splitlines() if line.startswith("0 (Sequential)"))
    cells = re.split(r"\s{2,}", line)
    assert cells[4] == "64"
    trace.cleanup()


def test_compute_footer_does_not_upgrade_estimated_evidence() -> None:
    """Known compute counts can include estimates; coverage is not exactness."""

    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1))
    trace = tl.trace(model.eval(), torch.ones(1, 3, 4, 4))
    report = trace.summary(view="compute")
    assert any(row.evidence == "estimated" for row in report._rebuilt.visible_rows)
    footer = next(line for line in str(report).splitlines() if line.startswith("Compute"))
    assert "3/3 ops known" in footer
    assert "formula-exact" not in footer
    assert "formula-exact" not in report.render("html")
    trace.cleanup()
