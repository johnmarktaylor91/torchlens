"""Packaging-diet tests for lazy pandas and IPython imports."""

from pathlib import Path
import importlib.util
import subprocess
import sys
from unittest.mock import patch
import zipfile

import pytest
import torch
from torch import nn

import torchlens as tl


class _TinyModel(nn.Module):
    """Small model for packaging smoke tests."""

    def __init__(self) -> None:
        """Initialize the tiny model layers."""

        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single linear layer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        return self.linear(x)


def _make_log() -> tl.Trace:
    """Create a completed model log for packaging tests.

    Returns
    -------
    tl.Trace
        Completed log for a tiny model.
    """

    return tl.trace(_TinyModel(), torch.randn(1, 3))


def test_core_import_capture_and_show_without_optional_tabular_or_notebook_use() -> None:
    """Core import, capture, and show path succeed in the current dev environment."""

    log = _make_log()

    assert len(log.layer_list) > 0
    assert log.show(vis_mode="none") is None


def test_to_pandas_succeeds_when_tabular_extra_is_available() -> None:
    """Trace.to_pandas succeeds when pandas is installed."""

    log = _make_log()
    frame = log.to_pandas()

    assert not frame.empty
    assert "layer_label" in frame.columns


def test_to_pandas_missing_pandas_mentions_tabular_extra() -> None:
    """Trace.to_pandas raises a helpful extra-install hint when pandas is missing."""

    log = _make_log()

    with patch.dict("sys.modules", {"pandas": None}):
        try:
            log.to_pandas()
        except ImportError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected ImportError when pandas is unavailable.")

    assert "pandas is required for this feature" in message
    assert "pip install torchlens[tabular]" in message


def test_repr_html_succeeds_when_notebook_extra_is_available() -> None:
    """Trace._repr_html_ returns the Phase 3 HTML card when IPython is installed."""

    log = _make_log()
    html = log._repr_html_()

    assert html.startswith("<div")
    assert "TorchLens Trace" in html
    assert "NaN/Inf" in html


def test_repr_html_missing_ipython_falls_back_to_text() -> None:
    """Trace._repr_html_ falls back to text when IPython is missing."""

    log = _make_log()

    with patch.dict("sys.modules", {"IPython": None, "IPython.display": None}):
        html = log._repr_html_()

    assert html == repr(log)


@pytest.mark.slow
def test_built_wheel_includes_tlspec_json_schemas(tmp_path: Path) -> None:
    """Built wheels must ship the public ``torchlens/schemas/*.json`` files."""

    project_root = Path(__file__).resolve().parent.parent
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()

    if importlib.util.find_spec("build") is not None:
        command = [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_dir)]
    elif importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(wheel_dir)]
    else:
        pytest.skip("neither build nor pip is importable for wheel construction")

    subprocess.run(command, cwd=project_root, check=True)
    wheels = sorted(wheel_dir.glob("torchlens-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel_zip:
        schema_members = [
            member
            for member in wheel_zip.namelist()
            if member.startswith("torchlens/schemas/") and member.endswith(".json")
        ]

    assert schema_members, "expected at least one torchlens/schemas/*.json wheel member"
