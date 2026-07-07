"""Tests for ``torchlens.report.explain``."""

from __future__ import annotations

from pathlib import Path

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

    The namespace size is checked against the namespace itself so this test
    guards the report names without hard-coding unrelated top-level API churn.
    """

    assert hasattr(tl.report, "explain")
    public_names = set(tl.__all__)
    assert len(tl.__all__) == len(public_names)
    assert "report" not in tl.__all__
    assert "explain" not in tl.__all__


def test_explain_returns_sensible_string_for_each_audience() -> None:
    """All supported audiences should produce a human-readable report."""

    log = _captured_log()
    for audience in ("researcher", "practitioner", "auto"):
        text = tl.report.explain(log, audience=audience)
        assert isinstance(text, str)
        assert "TorchLens report" in text
        assert "Model summary" in text
        assert "Capture summary" in text
        assert "Backward summary" in text
        assert "Anomalies" in text
        assert "Interventions" in text
        assert "Notable patterns" in text
        assert "TinyReportModel" in text
        assert "No backward passes are recorded" in text


def test_operational_status_line_reports_real_streamed_ops_not_a_fake_constant(
    tmp_path: Path,
) -> None:
    """``streamed_ops`` must reflect real streaming state, not a hardcoded ``1``.

    Regression for a bug where ``_operational_status_line`` always printed
    ``streamed_ops=1`` regardless of whether the trace used streaming at all.
    """

    from torchlens.report._explain import _operational_status_line

    plain_log = _captured_log()
    plain_line = _operational_status_line(plain_log)
    assert "streamed_ops=0" in plain_line

    plain_text = tl.report.explain(plain_log, audience="practitioner")
    assert "streamed_ops=0" in plain_text

    bundle_path = tmp_path / "streamed.tlspec"
    streamed_log = tl.trace(
        TinyReportModel(),
        torch.tensor([[2.0, 3.0]]),
        storage=tl.to_disk(bundle_path, retain_in_memory=False),
    )
    streamed_line = _operational_status_line(streamed_log)
    assert "streamed_ops=0" not in streamed_line
    num_layers = len(streamed_log.layer_list)
    assert f"streamed_ops={num_layers}" in streamed_line

    streamed_text = tl.report.explain(streamed_log, audience="practitioner")
    assert f"streamed_ops={num_layers}" in streamed_text


def test_explain_reports_backward_capture() -> None:
    """Backward logs should include pass, GradFn, and saved-gradient counts."""

    x = torch.tensor([[2.0, 3.0]], requires_grad=True)
    log = tl.trace(TinyReportModel(), x, save_grads=True)
    log.log_backward(log[log.output_layers[0]].out.sum())

    text = tl.report.explain(log)

    assert "Backward summary" in text
    assert "Backward passes: 1." in text
    assert "GradFn records:" in text
    assert "Op gradient records saved:" in text


def test_explain_reports_nonfinite_out() -> None:
    """The anomaly section should flag saved NaN or Inf outs."""

    log = _captured_log()
    log["linear_1_1"].out[0, 0] = torch.nan
    text = tl.report.explain(log)
    assert "NaN or Inf" in text
    assert "linear_1_1" in text
    assert "vscode://file/" in log.first_nonfinite(link_format="html")


def test_source_locations_keep_repr_plain_and_expose_html_links() -> None:
    """Source locations should keep repr plain and expose VS Code HTML links."""

    location = FuncCallLocation(
        file="/tmp/demo.py",
        line_number=12,
        func_name="forward",
        source_loading_enabled=False,
    )
    assert "\033]8;;file://" not in repr(location)
    assert "vscode://file/" in location.to_html_link()
