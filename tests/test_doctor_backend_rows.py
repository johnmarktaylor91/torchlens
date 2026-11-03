"""``backend_scorecard:<name>`` rows in ``tl.report.doctor()`` (SPEC 4.6).

The rows are built by ``torchlens/backends/_scorecard_rows.py`` from the
scorecard record shipped in each backend package and the backend's registered
``FrameworkRequirement``. They read the shipped JSON only: no framework is
imported at report time and the doctor module's import graph is unchanged.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.conformance._scorecard import (
    PreviewScorecard,
    ScorecardCell,
    backend_scorecards,
    scorecard_from_record,
)
from torchlens.backends import _scorecard_rows as rows_module
from torchlens.backends._scorecard_rows import (
    ROW_KEY_PREFIX,
    backend_scorecard_rows,
    read_shipped_scorecard,
    shipped_scorecard_path,
)
from torchlens.backends.registry import registered_backend_specs
from torchlens.report import _doctor as doctor_module, doctor as report

PREVIEW_FRAMEWORK_MODULES = ("tensorflow", "jax", "mlx", "paddle", "tinygrad")


class _Tiny(nn.Module):
    """Two-op model for the report call."""

    def __init__(self) -> None:
        """One linear layer."""

        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.linear(x))


def _record(
    backend: str,
    *,
    cells: tuple[tuple[str, str, str, str, str], ...],
    highest_tier: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A shipped-record dict in the v1 shape, plus any v2 keys in ``extra``."""

    cell_rows = [
        {"tier": tier, "case_id": case_id, "family": family, "state": state, "detail": detail}
        for tier, case_id, family, state, detail in cells
    ]
    counts = dict.fromkeys(("passed", "failed", "unsupported", "not_run"), 0)
    for row in cell_rows:
        counts[row["state"]] += 1
    counts["denominator"] = len(cell_rows)
    record: dict[str, Any] = {
        "scorecard_version": "torchlens.conformance.preview_scorecard.v1",
        "backend": backend,
        "framework_version": "0.0",
        "suite_version": "test",
        "torchlens_version": tl.__version__,
        "cells": cell_rows,
        "counts": counts,
        "highest_tier": highest_tier,
        "sha256": "",
    }
    record.update(extra or {})
    return record


@pytest.fixture
def shipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Redirect the shipped-record lookup to a temp tree; returns a writer map."""

    def _path(backend: str) -> Path:
        return tmp_path / backend / "scorecard.json"

    monkeypatch.setattr(rows_module, "shipped_scorecard_path", _path)
    paths: dict[str, Path] = {}

    def _write(backend: str, record: dict[str, Any] | None) -> Path:
        target = _path(backend)
        target.parent.mkdir(parents=True, exist_ok=True)
        if record is None:
            if target.exists():
                target.unlink()
        else:
            target.write_text(json.dumps(record), encoding="utf-8")
        paths[backend] = target
        return target

    paths["_write"] = _write  # type: ignore[assignment]
    return paths


@pytest.mark.smoke
def test_report_has_one_scorecard_row_per_registered_backend_without_importing_frameworks() -> None:
    """Every registered backend gets a row; no preview framework module appears in sys.modules."""

    before = {name.split(".")[0] for name in sys.modules}
    doctor_report = report(_Tiny(), torch.randn(2, 4))
    after = {name.split(".")[0] for name in sys.modules}
    newly_imported = (after - before) & set(PREVIEW_FRAMEWORK_MODULES)
    assert not newly_imported, newly_imported

    expected = {f"{ROW_KEY_PREFIX}{spec.name}" for spec in registered_backend_specs()}
    found = {row.key for row in doctor_report.checks if row.key.startswith(ROW_KEY_PREFIX)}
    assert found == expected
    assert {"torch", "tf", "jax", "mlx", "paddle", "tinygrad"} <= {
        key.removeprefix(ROW_KEY_PREFIX) for key in found
    }
    for row in doctor_report.checks:
        if row.key.startswith(ROW_KEY_PREFIX):
            name = row.key.removeprefix(ROW_KEY_PREFIX)
            assert row.label == f"{name} backend scorecard"
            assert row.status in {"pass", "known_broken", "scope", "not_tested"}
            assert row.details.split(";")[0] in {"preview", "stable"}
    cards = backend_scorecards()
    assert isinstance(cards, tuple)
    assert all(isinstance(card, PreviewScorecard) for card in cards)


def test_doctor_module_imports_the_row_builder_lazily() -> None:
    """``torchlens.report._doctor`` reaches the builder inside ``doctor()`` only."""

    module_source = inspect.getsource(doctor_module)
    report_source = inspect.getsource(doctor_module.doctor)
    assert "backends._scorecard_rows import" in report_source
    header = module_source.split("\ndef doctor(", 1)[0]
    assert "_scorecard_rows" not in header


def test_torch_row_reads_detected_and_the_framework_requirement() -> None:
    """Torch is installed: detected, stable, with its pin and range in the details."""

    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}torch")
    assert row.is_detected is True
    assert row.details.startswith("stable; ")
    assert f"installed {torch.__version__}" in row.details
    assert "supported [2.1, " in row.details
    assert "pinned " in row.details


def test_no_shipped_record_reads_not_tested_with_the_write_remedy(shipped: dict[str, Path]) -> None:
    """A backend with no record is ``not_tested``; the suggestion names the scorecard CLI."""

    shipped["_write"]("torch", None)  # type: ignore[operator]
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}torch")
    assert row.status == "not_tested"
    assert row.severity == "info"
    assert "no shipped scorecard" in row.details
    assert "tests.conformance.scorecard --backend torch --write" in row.suggestion
    assert read_shipped_scorecard("torch") is None
    assert all(card.backend != "torch" for card in backend_scorecards())


def test_clean_c2_record_on_an_installed_framework_reads_pass(shipped: dict[str, Path]) -> None:
    """A record with every cell passed at C2 for an installed framework is ``pass``/``ok``."""

    shipped["_write"](  # type: ignore[operator]
        "torch",
        _record(
            "torch",
            cells=(
                ("C0", "c0_protocol_complete", "", "passed", ""),
                ("C1", "c1_capture_completes", "mlp", "passed", ""),
                ("C2", "c2_roundtrip_completes", "mlp", "passed", ""),
            ),
            highest_tier="C2",
            extra={"maturity": "stable"},
        ),
    )
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}torch")
    assert row.status == "pass"
    assert row.severity == "ok"
    assert "stable; 3/3 cells passed, 0 unsupported, 0 failed" in row.details
    assert row.suggestion == ""
    cards = {card.backend: card for card in backend_scorecards()}
    assert cards["torch"].highest_tier == "C2"
    assert cards["torch"].counts["passed"] == 3
    assert isinstance(cards["torch"].cells[0], ScorecardCell)


def test_failed_cell_reads_known_broken_and_names_the_first_failure(
    shipped: dict[str, Path],
) -> None:
    """Any failed cell is ``known_broken``; the suggestion names the first failed cell."""

    shipped["_write"](  # type: ignore[operator]
        "torch",
        _record(
            "torch",
            cells=(
                ("C0", "c0_protocol_complete", "", "passed", ""),
                ("C1", "c1_reference_parity", "cnn", "failed", "max abs diff 0.5"),
                ("C1", "c1_capture_completes", "mlp", "failed", "boom"),
            ),
            highest_tier="C0",
            extra={"maturity": "preview"},
        ),
    )
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}torch")
    assert row.status == "known_broken"
    assert row.severity == "warning"
    assert "1/3 cells passed, 0 unsupported, 2 failed" in row.details
    assert row.suggestion == "first failed cell: c1_reference_parity (cnn): max abs diff 0.5"


def test_stable_backend_failure_is_an_error(shipped: dict[str, Path]) -> None:
    """A stable backend's failed cell is severity ``error``."""

    shipped["_write"](  # type: ignore[operator]
        "torch",
        _record(
            "torch",
            cells=(("C1", "c1_capture_completes", "mlp", "failed", "boom"),),
            highest_tier="",
            extra={"maturity": "stable"},
        ),
    )
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}torch")
    assert (row.status, row.severity) == ("known_broken", "error")


def test_absent_framework_reads_scope_with_the_install_extra(shipped: dict[str, Path]) -> None:
    """A clean record for a framework not installed here is ``scope``; the remedy is the extra."""

    absent = next(
        (
            name
            for name, module in (("tf", "tensorflow"), ("paddle", "paddle"), ("mlx", "mlx"))
            if _distribution_absent(name)
        ),
        None,
    )
    if absent is None:
        pytest.skip("every preview framework with an extra is installed here")
    shipped["_write"](  # type: ignore[operator]
        absent,
        _record(
            absent,
            cells=(("C0", "c0_protocol_complete", "", "passed", ""),),
            highest_tier="C0",
        ),
    )
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}{absent}")
    assert row.is_detected is False
    assert row.status == "scope"
    assert "installed absent" in row.details
    assert row.suggestion == f'install the framework: pip install "torchlens[{absent}]"'


def test_clean_record_below_c2_reads_scope_or_not_tested(shipped: dict[str, Path]) -> None:
    """No failures but below C2: ``not_tested`` when cells did not run, else ``scope``."""

    shipped["_write"](  # type: ignore[operator]
        "torch",
        _record(
            "torch",
            cells=(
                ("C0", "c0_protocol_complete", "", "passed", ""),
                ("C1", "c1_refusal_backward_capture", "", "unsupported", "unsupported[x]"),
            ),
            highest_tier="C0",
        ),
    )
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}torch")
    assert row.status == "scope"
    shipped["_write"](  # type: ignore[operator]
        "torch",
        _record(
            "torch",
            cells=(
                ("C0", "c0_protocol_complete", "", "passed", ""),
                ("C2", "c2_roundtrip_completes", "mlp", "not_run", "weights not cached"),
            ),
            highest_tier="C0",
        ),
    )
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}torch")
    assert row.status == "not_tested"


def test_malformed_or_non_object_record_is_treated_as_absent(shipped: dict[str, Path]) -> None:
    """A broken file never raises inside the doctor report; it reads as no record."""

    target = shipped["_write"]("torch", None)  # type: ignore[operator]
    target.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_shipped_scorecard("torch") is None
    target.write_text("{not json", encoding="utf-8")
    assert read_shipped_scorecard("torch") is None
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}torch")
    assert row.status == "not_tested"


def test_scorecard_from_record_tolerates_later_versions_and_missing_optionals() -> None:
    """Extra (v2) keys load; missing counts and tier default; cells become ScorecardCell."""

    card = scorecard_from_record(
        {
            "scorecard_version": "torchlens.conformance.preview_scorecard.v2",
            "backend": "mlx",
            "cells": [
                {"tier": "C0", "case_id": "c0_x", "family": "", "state": "passed", "extra": 1}
            ],
            "maturity": "preview",
            "graduation": {"items": []},
            "capabilities": {"backward_capture": "unsupported"},
        }
    )
    assert isinstance(card, PreviewScorecard)
    assert card.backend == "mlx"
    assert card.counts == {} and card.highest_tier == ""
    assert card.cells == (ScorecardCell(tier="C0", case_id="c0_x", family="", state="passed"),)


def test_shipped_path_points_inside_the_backend_package() -> None:
    """The record lives at ``torchlens/backends/<name>/scorecard.json``."""

    path = shipped_scorecard_path("tinygrad")
    assert path.name == "scorecard.json"
    assert path.parent.name == "tinygrad"
    assert path.parent.parent.name == "backends"


def _distribution_absent(backend: str) -> bool:
    """Whether the backend's framework distribution is not installed here."""

    from torchlens.backends.registry import get_backend_spec

    spec = get_backend_spec(backend)
    assert spec.protocol is not None
    return spec.protocol().framework._installed_version() is None
