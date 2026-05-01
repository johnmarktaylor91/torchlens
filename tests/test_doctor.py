"""Tests for the TorchLens doctor utility."""

from __future__ import annotations

import torchlens as tl


def test_doctor_returns_sane_report() -> None:
    """Doctor returns structured checks and a printable report."""

    report = tl.utils.doctor()
    assert report.checks
    names = {check.name for check in report.checks}
    assert {"pytorch", "cuda", "graphviz", "safetensors", "extras", "model fingerprint"} <= names
    assert all(check.status in {"PASS", "FAIL", "SKIP"} for check in report.checks)
    text = report.show()
    assert "TorchLens doctor report" in text
    assert "pytorch" in text
