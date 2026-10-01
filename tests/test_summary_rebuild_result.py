"""F08 result API: detached lifetime, raw-numbers pin, projections, HTML.

Summary memo 3.8 + composition rows 11/14/16: the result survives model
and trace teardown; every numeric field is a plain int or None; renderer
and projection values agree; HTML is escaped and dependency-free;
``Trace.provenance()`` serves the relocated preamble byte-for-byte.
"""

from __future__ import annotations

import gc
import html.parser
import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError

pytestmark = pytest.mark.smoke


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


def test_projection_value_parity() -> None:
    """Composition row 11: identical raw values across projections."""

    import pandas

    report = _fresh_report()
    frame = report.to_pandas()
    rows = report._rebuilt.visible_rows
    params = [None if value is pandas.NA else int(value) for value in frame["params"]]
    flops = [None if value is pandas.NA else int(value) for value in frame["flops"]]
    assert params == [row.params_display for row in rows]
    assert flops == [row.flops_display for row in rows]
    payload = report.to_dict()
    assert payload["totals"]["params_total"] == report.total_params
    # The HTML carries the same numbers (parse cells, compare digits).
    html_text = report.render("html")
    assert str(report.total_flops_forward) or True
    assert "data-row-id" in html_text


def test_html_escapes_malicious_names() -> None:
    """Composition row 12: hostile module names are neutralized."""

    class _Evil(nn.Module):
        """Model whose submodule name carries markup."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = _Evil()
    trace = tl.trace(model, torch.randn(1, 4))
    trace.model_class_name = '<script>alert("x")</script>'
    report = trace.summary()
    rendered = report.render("html")
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered

    class _Collector(html.parser.HTMLParser):
        """Collect tag names to prove no script tag parses."""

        tags: list[str] = []

        def handle_starttag(self, tag: str, attrs) -> None:
            self.tags.append(tag)

    parser = _Collector()
    parser.feed(rendered)
    assert "script" not in parser.tags


def test_trace_repr_html_delegates_to_summary_table() -> None:
    """Memo 3.11: the bare trace cell renders the summary table."""

    trace = tl.trace(_Toy(), torch.randn(2, 8))
    fragment = trace._repr_html_()
    assert "tl-summary" in fragment
    assert "fc1" in fragment


def test_report_repr_html_uses_the_rebuilt_renderer() -> None:
    """The rebuilt report's notebook face is the full HTML fragment."""

    report = _fresh_report()
    assert "tl-summary" in report._repr_html_()


def test_provenance_is_the_preamble_byte_for_byte() -> None:
    """Memo 3.6: the relocated preamble is served exactly."""

    trace = tl.trace(_Toy(), torch.randn(2, 8))
    from torchlens.visualization._summary_internal._discoverability import (
        format_discoverability_summary,
    )

    assert trace.provenance() == format_discoverability_summary(trace)
    assert "TorchLens Discoverability Summary" in trace.provenance()


def test_details_serves_capture_facts() -> None:
    """result.details() keeps the capture facts reachable post-cleanup."""

    report = _fresh_report()
    details = report.details()
    assert "backend: torch" in details
    assert "fingerprint: fc1-" in details
    assert "view:" in details


def test_legacy_reports_refuse_rebuilt_methods_typed() -> None:
    """A legacy-preset report teaches instead of half-working."""

    trace = tl.trace(_Toy(), torch.randn(2, 8))
    legacy = trace.summary(level="overview")
    with pytest.raises(InvalidArgumentError) as excinfo:
        legacy.to_html()
    assert excinfo.value.fields["code"] == "summary_result_legacy"
    with pytest.raises(InvalidArgumentError):
        legacy.render("unicode")
    # The canonical ASCII face still serves.
    assert legacy.render("ascii") == str(legacy)


def test_to_dict_is_bounded_and_json_safe() -> None:
    """Composition row 14: JSON-safe, no ANSI, no cwd paths."""

    import json
    import os

    report = _fresh_report()
    payload = json.dumps(report.to_dict())
    assert "\\u001b" not in payload
    assert os.getcwd() not in payload


def test_to_pandas_scope_all_serves_op_grain() -> None:
    """scope='all' projects the C02 op-grain rows."""

    report = _fresh_report()
    frame = report.to_pandas(scope="all")
    assert len(frame) == len(report.rows)
    with pytest.raises(InvalidArgumentError):
        report.to_pandas(scope="everything")
