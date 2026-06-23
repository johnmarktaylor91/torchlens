"""Tests for menagerie validator worker memory caps."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from menagerie.catalog import CatalogRow, load_rows
from menagerie.validate_menagerie import (
    WORKER_MEMORY_CAP_STATUS,
    WORKER_MEMORY_TEST_ALLOC_ENV,
    ValidationResult,
    _resolve_scheduler_memory_budget_gb,
    append_manifest,
    build_parser,
    manifest_records,
    validate_with_timeout,
)


def _row(**overrides: object) -> CatalogRow:
    """Build a compact catalog row fixture for memory-cap tests.

    Parameters
    ----------
    overrides:
        Field overrides for the default catalog row.

    Returns
    -------
    CatalogRow
        Test catalog row.
    """

    data = {
        "model_id": 999001,
        "display_index": 999001,
        "stable_id": "memcap-test",
        "name": "MemoryCapToy",
        "variant": "",
        "family": "toy",
        "family_normalized": "toy",
        "domain": "unit",
        "zoo": "unit-zoo",
        "constructor_call": "torch.nn.Linear(1, 1)",
        "input_shape": "(1, 1)",
        "input_dtype": "float32",
        "era": "2026",
        "verified": False,
        "notes": "",
        "source": "catalog",
        "recipe_revision_sha256": "recipe",
    }
    data.update(overrides)
    return CatalogRow(**data)


def _catalog_row(model_id: int) -> CatalogRow:
    """Return one real catalog row by model ID.

    Parameters
    ----------
    model_id:
        Catalog model identifier.

    Returns
    -------
    CatalogRow
        Matching real catalog row.
    """

    rows_by_id = {row.model_id: row for row in load_rows()}
    return rows_by_id[model_id]


def test_worker_memory_cap_records_honest_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker that exceeds its RSS cap returns a memory-cap validation failure."""

    pytest.importorskip("psutil")
    monkeypatch.setenv(WORKER_MEMORY_TEST_ALLOC_ENV, "192")
    manifest_path = tmp_path / "validation_manifest.tsv"

    result = validate_with_timeout(
        _row(),
        dry_run=True,
        scope="forward",
        device="cpu",
        timeout_sec=20.0,
        worker_memory_cap_gb=0.12,
    )

    assert result.status == WORKER_MEMORY_CAP_STATUS
    assert result.peak_rss_mb is not None
    assert result.peak_rss_mb > 120
    assert "--worker-memory-cap-gb=0.120" in result.error
    append_manifest(manifest_path, result)
    [record] = manifest_records(manifest_path).values()
    assert record["status"] == WORKER_MEMORY_CAP_STATUS
    assert int(record["peak_rss_mb"]) == result.peak_rss_mb


def test_parent_accepts_normal_worker_result_with_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A capped worker result under the cap is parsed normally by the parent."""

    row = _row()
    expected = ValidationResult(
        row.name,
        row.model_id,
        "validated",
        3,
        True,
        "forward",
        0.25,
        "unit-zoo",
        "forward=True",
        "shape",
        stable_id=row.stable_id,
        recipe_revision_sha256=row.recipe_revision_sha256,
        peak_rss_mb=64,
    )
    worker_event = json.dumps({"event": "worker_result", "result": expected.__dict__})
    captured_command: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Capture the worker command and return a successful worker event.

        Parameters
        ----------
        command:
            Subprocess command built by the parent.
        kwargs:
            Additional subprocess options.

        Returns
        -------
        subprocess.CompletedProcess[str]
            Completed process with one worker result event.
        """

        captured_command.extend(command)
        assert kwargs["timeout"] == 5.0
        return subprocess.CompletedProcess(command, 0, stdout=f"{worker_event}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = validate_with_timeout(
        row,
        dry_run=False,
        scope="forward",
        device="cpu",
        timeout_sec=5.0,
        tmp_dir=tmp_path,
        worker_memory_cap_gb=2.0,
    )

    assert result == expected
    assert "--worker-memory-cap-gb" in captured_command
    assert "2.000000" in captured_command


def test_worker_under_memory_cap_validates() -> None:
    """A real worker below its RSS cap returns the normal validated status."""

    result = validate_with_timeout(
        _catalog_row(65),
        dry_run=False,
        scope="forward",
        device="cpu",
        timeout_sec=20.0,
        worker_memory_cap_gb=4.0,
    )

    assert result.status == "validated"
    assert result.peak_rss_mb is not None
    assert result.peak_rss_mb < 4096


def test_worker_memory_cap_sets_scheduler_budget_from_effective_jobs() -> None:
    """A per-worker cap bounds the scheduler's effective in-flight budget."""

    assert _resolve_scheduler_memory_budget_gb(None, 24.0, 3) == 72.0
    assert _resolve_scheduler_memory_budget_gb(64.0, 24.0, 3) == 64.0


def test_worker_memory_cap_default_is_off() -> None:
    """The new worker cap is disabled unless explicitly requested."""

    args = build_parser().parse_args([])
    assert args.worker_memory_cap_gb is None
