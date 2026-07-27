"""Consolidated release-gate invariants spanning all crawler slices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.doctor import DoctorConfig, DoctorError, run_doctor
from menagerie.crawler.status import assert_partition
from menagerie.crawler.tests.conftest import make_model
from menagerie.crawler.tests.test_slice_f_doctor import FakeDoctorProbes
from menagerie.crawler.tools.verify_prompts import verify_prompts
from .support import fabricated_crawler_locks, repository_root


def _terminal_record(stable_id: str, status_code: str) -> dict[str, Any]:
    """Return one synthetic current record in a selected terminal funnel bucket."""

    record = make_model(stable_id, accepted=True, status_code="runs")
    record["status"]["kind"] = status_code.split(":", 1)[0]
    record["status"]["code"] = status_code
    return record


def test_full_funnel_is_an_exact_complete_terminal_partition() -> None:
    """Every intake model appears once in runs/deferred/skipped/failed and nowhere else."""

    records = [
        _terminal_record("m_runs", "runs"),
        _terminal_record("m_deferred", "deferred:needs-cuda"),
        _terminal_record("m_skipped", "skipped:no-description"),
        _terminal_record("m_failed", "failed:forward"),
    ]
    intake_ids = [str(record["stable_id"]) for record in records]
    report = assert_partition(intake_ids, records)
    assert report.valid
    assert set(report.buckets) == {
        "runs",
        "deferred:needs-cuda",
        "skipped:no-description",
        "failed:forward",
    }
    assert sum(len(bucket) for bucket in report.buckets.values()) == len(intake_ids)


def test_frozen_prompts_and_generated_lock_boundary_pass_release_gate() -> None:
    """Frozen prompt bytes verify and no hand-authored target solve output ships."""

    repo_root = repository_root()
    crawler_root = repo_root / "menagerie" / "crawler"
    digests = verify_prompts(
        crawler_root / "PLAN.md",
        crawler_root / "prompts" / "claude_crawler_author_v2.txt",
        crawler_root / "prompts" / "codex_accuracy_checker_v2.txt",
    )
    assert set(digests) == {
        "claude_crawler_author_v2.txt",
        "codex_accuracy_checker_v2.txt",
    }
    assert fabricated_crawler_locks(repo_root) == ()


def test_doctor_is_go_on_clean_preflight_and_no_go_when_tripped(tmp_path: Path) -> None:
    """The release preflight returns go only when every injected host probe is clean."""

    config = DoctorConfig(repository_root(), tmp_path / ".crawl-local", "osx-arm64")
    clean = run_doctor(config, FakeDoctorProbes())
    assert clean.passed
    assert set(clean.checks.values()) == {"pass"}
    tripped = FakeDoctorProbes()
    tripped.tripwires["socket"] = False
    with pytest.raises(DoctorError) as captured:
        run_doctor(config, tripped)
    assert "policy:socket: tripwire self-test failed" in captured.value.failures
