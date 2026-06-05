"""Tests for ``torchlens.report.explain``."""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes import FuncCallLocation


class TinyReportModel(nn.Module):
    """Small deterministic model for report tests."""

    def __init__(self) -> None:
        """Initialize deterministic weights."""

        super().__init__()
        self.proj = nn.Linear(2, 2)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
            self.proj.bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a tiny nonlinear forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU-transformed projection.
        """

        return torch.relu(self.proj(x))


def _captured_log() -> tl.Trace:
    """Return a deterministic captured log.

    Returns
    -------
    tl.Trace
        Captured log for ``TinyReportModel``.
    """

    return tl.trace(TinyReportModel(), torch.tensor([[2.0, 3.0]]))


def test_report_namespace_is_not_top_level_all() -> None:
    """``tl.report.explain`` should be reachable without expanding ``tl.__all__``.

    Phase 1a budget was 40; backward-parity added 6 top-level names, and
    post-backward P1 added ``output`` for multi-output module selector
    disambiguation, facets added the top-level ``facets`` namespace, and
    v7 quantity types added five top-level names, and facets P2 added
    ``facet`` / ``head`` selectors.
    """

    assert hasattr(tl.report, "explain")
    assert "report" not in tl.__all__
    assert "explain" not in tl.__all__
    assert len(tl.__all__) == 54


def test_explain_returns_sensible_string_for_each_audience() -> None:
    """All supported audiences should produce a human-readable report."""

    log = _captured_log()
    for audience in ("researcher", "practitioner", "auto"):
        text = tl.report.explain(log, audience=audience)
        assert isinstance(text, str)
        assert "TorchLens report" in text
        assert "Model summary" in text
        assert "Capture summary" in text
        assert "Anomalies" in text
        assert "Interventions" in text
        assert "Notable patterns" in text
        assert "TinyReportModel" in text


def test_explain_reports_nonfinite_out() -> None:
    """The anomaly section should flag saved NaN or Inf outs."""

    log = _captured_log()
    log["linear_1_1"].out[0, 0] = torch.nan
    text = tl.report.explain(log)
    assert "NaN or Inf" in text
    assert "linear_1_1" in text
    assert "vscode://file/" in log.first_nonfinite(link_format="html")


def test_source_locations_render_clickable_terminal_and_html_links() -> None:
    """Source locations should expose OSC 8 terminal links and VS Code HTML links."""

    location = FuncCallLocation(
        file="/tmp/demo.py",
        line_number=12,
        func_name="forward",
        source_loading_enabled=False,
    )
    assert "\033]8;;file://" in repr(location)
    assert "vscode://file/" in location.to_html_link()
