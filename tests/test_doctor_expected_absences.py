"""The doctor's runtime-capabilities row: expected absences are info, real gaps still warn."""

from __future__ import annotations

import pytest

from torchlens.report import _environment
from torchlens.report._environment import _classify_capability_absences, _probe_torch_capabilities


def test_flash_attention_4_absence_is_expected_not_missing() -> None:
    """flash-attn 4 is a third-party CUDA package; a CPU install without it is healthy."""

    missing, _unprobed, optional_absent = _classify_capability_absences(
        {"HAS_FLASH_ATTENTION_4_INTERFACE": False, "HAS_SDPA_GQA": True}
    )
    assert missing == []
    assert optional_absent == ["HAS_FLASH_ATTENTION_4_INTERFACE"]


def test_row_passes_with_only_expected_absences_and_lists_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _environment,
        "capability_snapshot",
        lambda: {"HAS_FLASH_ATTENTION_4_INTERFACE": False, "HAS_NAMED_TENSOR_API": False},
    )
    row = _probe_torch_capabilities()
    assert row.status == "PASS"
    assert "optional_absent=" in row.detail
    assert "HAS_FLASH_ATTENTION_4_INTERFACE" in row.detail


def test_a_real_missing_capability_still_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A required, probed capability that is absent is never folded into the expected group."""

    monkeypatch.setattr(
        _environment,
        "capability_snapshot",
        lambda: {"HAS_FLASH_ATTENTION_4_INTERFACE": False, "HAS_ACCUMULATE_GRAD_CLASS": False},
    )
    row = _probe_torch_capabilities()
    assert row.status == "WARN"
    assert "missing=HAS_ACCUMULATE_GRAD_CLASS" in row.detail
