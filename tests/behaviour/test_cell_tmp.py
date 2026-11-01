"""Each behaviour-net cell removes its temp directory when it ends.

A slow-universe run otherwise keeps thousands of cell directories (saved
bundles, streamed payloads) until the session ends and fills the worker disk.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.behaviour import harness
from tests.behaviour.cases import smoke_cases
from tests.behaviour.features import feature
from tests.behaviour.harness import CellResult, Outcome, run_cell_and_clean

CASE = smoke_cases()[0]
FEATURES = (feature("save"), feature("index"))


def _fake_run_cell(status: str) -> Callable[..., CellResult]:
    def fake(case: Any, features: Any, tmp: Path) -> CellResult:
        (tmp / "cell.tlspec").mkdir()
        (tmp / "cell.tlspec" / "payload.bin").write_bytes(b"x" * 1024)
        return CellResult("fake", {"trace": Outcome("pass"), "save": Outcome(status)})

    return fake


def _crashing_run_cell(case: Any, features: Any, tmp: Path) -> CellResult:
    (tmp / "partial.bin").write_bytes(b"x")
    raise RuntimeError("harness crashed")


@pytest.mark.parametrize("keep_failed", [False, True])
@pytest.mark.parametrize("status", ["pass", "refused", "not_applicable"])
def test_satisfied_cell_removes_its_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, keep_failed: bool
) -> None:
    monkeypatch.setattr(harness, "run_cell", _fake_run_cell(status))
    cell_tmp = tmp_path / "cell"
    cell_tmp.mkdir()
    result = run_cell_and_clean(CASE, FEATURES, cell_tmp, keep_failed=keep_failed)
    assert result.oracle("save").status == status
    assert not cell_tmp.exists()


@pytest.mark.parametrize("keep_failed", [False, True])
def test_failed_cell_keeps_tmp_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_failed: bool
) -> None:
    monkeypatch.setattr(harness, "run_cell", _fake_run_cell("fail"))
    cell_tmp = tmp_path / "cell"
    cell_tmp.mkdir()
    result = run_cell_and_clean(CASE, FEATURES, cell_tmp, keep_failed=keep_failed)
    assert result.oracle("save").status == "fail"
    assert cell_tmp.exists() is keep_failed
    if keep_failed:
        assert (cell_tmp / "cell.tlspec" / "payload.bin").stat().st_size == 1024


@pytest.mark.parametrize("keep_failed", [False, True])
def test_crashed_cell_counts_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_failed: bool
) -> None:
    monkeypatch.setattr(harness, "run_cell", _crashing_run_cell)
    cell_tmp = tmp_path / "cell"
    cell_tmp.mkdir()
    with pytest.raises(RuntimeError, match="harness crashed"):
        run_cell_and_clean(CASE, FEATURES, cell_tmp, keep_failed=keep_failed)
    assert cell_tmp.exists() is keep_failed


def test_real_cell_leaves_no_tmp(tmp_path: Path) -> None:
    """A real cell that writes a bundle to disk leaves nothing behind."""

    cell_tmp = tmp_path / "cell"
    cell_tmp.mkdir()
    result = run_cell_and_clean(CASE, FEATURES, cell_tmp)
    assert result.oracle("save").status == "pass"
    assert not cell_tmp.exists()
