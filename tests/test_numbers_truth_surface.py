"""Numbers-truth surface fixes: front door (A8), escape bytes (A11), memory footer (A7).

Lane A07 (2026-08-27). Spec: the summary design memo build items 1-3.
"""

from __future__ import annotations

import subprocess
import sys
import warnings
from collections.abc import Generator

import pytest
import torch
from torch import nn

import torchlens as tl


class _CondModel(nn.Module):
    """Model with a data-dependent branch, to exercise the control-flow view."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the input mean."""

        if x.mean() > 0:
            return torch.relu(x)
        return torch.sigmoid(x)


@pytest.fixture()
def small_log() -> Generator[tl.Trace, None, None]:
    """Metadata-only capture of a tiny model."""

    model = nn.Sequential(nn.Linear(8, 16, bias=True), nn.ReLU())
    log = tl.trace(model, torch.randn(2, 8), capture=tl.options.CaptureOptions(layers_to_save=None))
    try:
        yield log
    finally:
        log.cleanup()


def test_tl_summary_is_a_first_class_front_door() -> None:
    """tl.summary exists, is exported, RETURNS the text, and never warns (A8)."""

    assert "summary" in tl.__all__
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = tl.summary(model, torch.randn(1, 4))
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations == []
    assert isinstance(text, str)
    assert "Linear" in text


def test_visualization_summary_alias_resolves() -> None:
    """The historically-taught torchlens.visualization.summary spelling resolves (A8)."""

    import torchlens.visualization as visualization

    assert callable(visualization.summary)


@pytest.mark.heavy
def test_readme_idiom_subprocess_zero_warnings() -> None:
    """The README one-call idiom runs warning-free in a fresh process (A8 pin)."""

    code = (
        "import warnings\n"
        "warnings.simplefilter('error', DeprecationWarning)\n"
        "import torch, torch.nn as nn\n"
        "import torchlens as tl\n"
        "text = tl.summary(nn.Linear(4, 4), torch.randn(1, 4))\n"
        "assert isinstance(text, str) and 'Linear' in text\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.smoke
def test_summary_has_no_escape_bytes_any_level(small_log: tl.Trace) -> None:
    """No returned summary string carries ESC bytes, at any level (A11)."""

    for kwargs in ({}, {"level": "module"}, {"level": "op"}, {"view": "compute"}):
        text = str(small_log.summary(**kwargs))
        assert "\x1b" not in text, f"escape byte in summary({kwargs})"


@pytest.mark.smoke
def test_control_flow_summary_has_no_escape_bytes() -> None:
    """A conditional model's summary and provenance emit plain text (A11)."""

    log = tl.trace(
        _CondModel(),
        torch.randn(2, 3) + 1.0,
        capture=tl.options.CaptureOptions(layers_to_save=None),
    )
    try:
        for text in (str(log.summary(level="op")), log.provenance()):
            assert "\x1b" not in text
    finally:
        log.cleanup()


def test_memory_footer_reports_measured_peak(small_log: tl.Trace) -> None:
    """The memory footer prints the measured peak + backend, never 'not tracked' (A7)."""

    from torchlens.observe import pass_peak_facts

    text = str(small_log.summary())
    assert "not tracked" not in text
    backend = small_log.forward_memory_backend
    assert backend in {"cpu", "cuda", "mps"}
    facts = pass_peak_facts(small_log)
    assert facts.value_bytes is not None
    # The basis is always named; a measured value is never called unavailable.
    assert f"({facts.meaning})" in text
    assert "forward peak unavailable" not in text


def test_memory_footer_unavailable_is_distinct(small_log: tl.Trace) -> None:
    """A capture with no recorded backend reports unavailable, never a fake zero (A7)."""

    small_log.forward_memory_backend = "unknown"
    text = str(small_log.summary())
    assert "forward peak unavailable (not measured on this capture)" in text
