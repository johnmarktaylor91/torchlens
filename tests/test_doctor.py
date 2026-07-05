"""Tests for the TorchLens doctor utility."""

from __future__ import annotations

import torchlens as tl
from torchlens.backends.tf._tf_compat import get_tf_capability_snapshot
from torchlens.utils._torch_compat import get_torch_capability_snapshot


def test_doctor_returns_sane_report() -> None:
    """Doctor returns structured checks and a printable report."""

    report = tl.utils.doctor()
    assert report.checks
    names = {check.name for check in report.checks}
    assert {
        "pytorch",
        "runtime capabilities",
        "cuda",
        "graphviz",
        "safetensors",
        "extras",
        "model fingerprint",
    } <= names
    assert all(check.status in {"PASS", "FAIL", "SKIP"} for check in report.checks)
    text = report.show()
    assert "TorchLens doctor report" in text
    assert "pytorch" in text


def test_doctor_surfaces_every_runtime_capability() -> None:
    """Doctor capability row stays in lockstep with defined capability flags."""

    report = tl.utils.doctor()
    row = next(check for check in report.checks if check.name == "runtime capabilities")
    expected = set(get_torch_capability_snapshot()) | set(get_tf_capability_snapshot())
    surfaced = {part.split("=", 1)[0] for part in row.detail.split(";")[0].split(", ")}

    assert surfaced == expected
