"""``backend_scorecard:<name>`` rows in ``tl.report.doctor()`` (SPEC 4.6).

The rows are built by ``torchlens/backends/_scorecard_rows.py`` from the
scorecard record shipped in each backend package and the backend's registered
``FrameworkRequirement``. They read the shipped JSON only: no framework is
imported at report time and the doctor module's import graph is unchanged.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

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
)
from torchlens.report import _doctor as doctor_module

PREVIEW_FRAMEWORK_MODULES = ("tensorflow", "jax", "mlx", "paddle", "tinygrad")


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


def test_doctor_module_imports_the_row_builder_lazily() -> None:
    """``torchlens.report._doctor`` reaches the builder inside ``doctor()`` only."""

    module_source = inspect.getsource(doctor_module)
    report_source = inspect.getsource(doctor_module.doctor)
    assert "backends._scorecard_rows import" in report_source
    header = module_source.split("\ndef doctor(", 1)[0]
    assert "_scorecard_rows" not in header


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_scorecard_rows.py.
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
        ),
    )
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}torch")
    assert row.status == "pass"
    assert row.severity == "ok"
    assert row.details.startswith("3/3 cells passed, 0 unsupported, 0 failed;")
    assert row.suggestion == ""
    cards = {card.backend: card for card in backend_scorecards()}
    assert cards["torch"].highest_tier == "C2"
    assert cards["torch"].counts["passed"] == 3
    assert isinstance(cards["torch"].cells[0], ScorecardCell)


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_scorecard_rows.py.
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
        pytest.skip("every non-torch framework with an extra is installed here")
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_scorecard_rows.py.
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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _io/_json.py, backends/_scorecard_rows.py.
def test_malformed_or_non_object_record_is_treated_as_absent(shipped: dict[str, Path]) -> None:
    """A broken file never raises inside the doctor report; it reads as no record."""

    target = shipped["_write"]("torch", None)  # type: ignore[operator]
    target.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_shipped_scorecard("torch") is None
    target.write_text("{not json", encoding="utf-8")
    assert read_shipped_scorecard("torch") is None
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}torch")
    assert row.status == "not_tested"


@pytest.mark.parametrize("backend", ["torch", "jax", "tf", "mlx", "paddle", "tinygrad"])
def test_failed_cells_warn_the_same_on_every_backend(
    shipped: dict[str, Path], backend: str
) -> None:
    """A failed cell reads ``known_broken`` at ``warning`` on torch and every other backend.

    The row text starts with the cell counts, never with a label word.
    """

    shipped["_write"](  # type: ignore[operator]
        backend,
        _record(
            backend,
            cells=(
                ("C0", "c0_protocol_complete", "", "passed", ""),
                ("C1", "c1_reference_parity", "mlp", "failed", "planted"),
            ),
            highest_tier="C0",
        ),
    )
    row = next(r for r in backend_scorecard_rows() if r.key == f"{ROW_KEY_PREFIX}{backend}")
    assert row.status == "known_broken"
    assert row.severity == "warning"
    assert row.details.startswith("1/2 cells passed, 0 unsupported, 1 failed;")


def test_shipped_records_never_turn_the_doctor_report_red() -> None:
    """No row built from the scorecards shipped in the package has ``error`` severity."""

    severities = {row.key: row.severity for row in backend_scorecard_rows()}
    assert severities, "no backend_scorecard rows"
    assert "error" not in severities.values(), severities


def test_scorecard_from_record_tolerates_later_versions_and_missing_optionals() -> None:
    """Extra keys load (a v2 record's retired ``maturity`` too); missing counts and tier
    default; cells become ScorecardCell."""

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


def _distribution_absent(backend: str) -> bool:
    """Whether the backend's framework distribution is not installed here."""

    from torchlens.backends.registry import get_backend_spec

    spec = get_backend_spec(backend)
    assert spec.protocol is not None
    return spec.protocol().framework._installed_version() is None
