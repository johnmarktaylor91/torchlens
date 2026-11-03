"""F08 result API: detached lifetime, raw-numbers pin, projections, HTML.

Summary memo 3.8 + composition rows 11/14/16: the result survives model
and trace teardown; every numeric field is a plain int or None; renderer
and projection values agree; HTML is escaped and dependency-free;
``Trace.provenance.summary()`` serves the relocated preamble byte-for-byte.
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
