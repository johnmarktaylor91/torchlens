"""Tests for Phase 3 repr and tensor display ergonomics."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


class _InfModel(nn.Module):
    """Tiny model that produces a non-finite activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an infinite tensor."""

        return x / 0


def _log_for_input(x: torch.Tensor) -> tl.ModelLog:
    """Capture an identity model for one input.

    Parameters
    ----------
    x:
        Input tensor.

    Returns
    -------
    tl.ModelLog
        Captured model log.
    """

    return tl.log_forward_pass(nn.Identity(), x)


def test_print_model_log_is_informative(capsys: pytest.CaptureFixture[str]) -> None:
    """Printing a ModelLog gives a concise model summary."""

    log = _log_for_input(torch.randn(1, 3))
    print(log)
    captured = capsys.readouterr()
    assert "Log of" in captured.out
    assert "Tensor info" in captured.out


def test_model_log_repr_html_is_informative() -> None:
    """ModelLog HTML repr returns an informative string or text fallback."""

    log = _log_for_input(torch.randn(1, 3))
    html = log._repr_html_()
    assert isinstance(html, str)
    assert "ModelLog" in html or "TorchLens" in html
    assert "Layers" in html or "layers=" in html


@pytest.mark.parametrize("method", ["auto", "heatmap", "channels", "rgb", "hist"])
def test_layer_log_show_methods_return_output(method: str) -> None:
    """LayerLog.show accepts every Phase 3 display method."""

    x = torch.randn(1, 3, 4, 4) if method == "rgb" else torch.randn(3, 4, 4)
    log = _log_for_input(x)
    output = log.layers[0].show(method=method)
    assert output is not None


def test_layer_pass_log_show_returns_output() -> None:
    """LayerPassLog.show delegates to the tensor display helper."""

    log = _log_for_input(torch.randn(8))
    output = log.layer_list[0].show(method="hist")
    assert output is not None


def test_first_nonfinite_reports_context() -> None:
    """ModelLog.first_nonfinite reports the first saved NaN or Inf site."""

    log = tl.log_forward_pass(_InfModel(), torch.ones(1, 2))
    answer = log.first_nonfinite()
    assert "First non-finite" in answer
    assert "shape=" in answer
    assert "dtype=" in answer
    assert "parents=" in answer
    assert "source=" in answer
