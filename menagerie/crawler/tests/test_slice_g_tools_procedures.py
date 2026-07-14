"""Slice G tool-wrapper and human-procedure tests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from menagerie.crawler.cli import build_parser as build_crawler_parser
from menagerie.crawler.constants import MODEL_SCHEMA_VERSION
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.licenses import (
    LicenseEvidence,
    LicenseEvidenceStatus,
    store_licensed_artifact,
)
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorStore
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.tests.conftest import make_model
from menagerie.crawler.tools.license_sweep import main as license_sweep_main
from menagerie.crawler.tools.rebuild_views import main as rebuild_views_main
from menagerie.crawler.tools.requeue import main as requeue_main
from menagerie.crawler.tools.verify_prompts import main as verify_prompts_main


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write complete canonical test JSONL.

    Parameters
    ----------
    path:
        Destination ledger or manifest.
    rows:
        JSON-compatible object rows.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def test_verify_prompts_passes_shipped_files_and_rejects_mutation(tmp_path: Path) -> None:
    """Shipped prompts equal PLAN while a one-byte mutation fails."""

    assert verify_prompts_main([]) == 0
    crawler_root = Path(__file__).resolve().parents[1]
    author = crawler_root / "prompts" / "claude_crawler_author_v2.txt"
    mutated = tmp_path / author.name
    mutated.write_bytes(author.read_bytes() + b"mutated\n")
    assert verify_prompts_main(["--author-prompt", str(mutated)]) != 0


def test_rebuild_views_is_deterministic_and_ignores_stale_database(tmp_path: Path) -> None:
    """Canonical JSONL yields identical view hashes despite stale derived state."""

    records = tmp_path / "records"
    ledgers = LedgerPaths(
        records / "models" / "current-shard.jsonl",
        records / "attempts" / "local.jsonl",
        records / "gates" / "current-shard.jsonl",
    )
    with JsonlLedger(ledgers.models, MODEL_SCHEMA_VERSION) as ledger:
        ledger.append(make_model(accepted=True))
    intake = tmp_path / "items.jsonl"
    _write_jsonl(intake, [{"stable_id": "m_example"}])
    database = tmp_path / "state.sqlite"
    args = [
        "--intake",
        str(intake),
        "--records-root",
        str(records),
        "--views-root",
        str(tmp_path / "views-a"),
        "--database",
        str(database),
    ]
    assert rebuild_views_main(args) == 0
    first = {
        path.relative_to(tmp_path / "views-a"): path.read_bytes()
        for path in (tmp_path / "views-a").rglob("*")
        if path.is_file()
    }
    database.write_bytes(b"stale derived state")
    args[args.index(str(tmp_path / "views-a"))] = str(tmp_path / "views-b")
    assert rebuild_views_main(args) == 0
    second = {
        path.relative_to(tmp_path / "views-b"): path.read_bytes()
        for path in (tmp_path / "views-b").rglob("*")
        if path.is_file()
    }
    assert first == second


def test_license_sweep_rejects_restricted_staged_artifact(tmp_path: Path) -> None:
    """The wrapper emits a failed report and non-zero code for restricted bytes."""

    mirrors = MirrorStore(tmp_path / "public", tmp_path / "private", tmp_path / "local")
    evidence = (
        LicenseEvidence(
            evidence_id="license-gpl",
            source_id="source-license",
            locator="LICENSE:1",
            excerpt="GPL license text",
            status=LicenseEvidenceStatus.DECLARED,
            spdx="GPL-3.0-only",
        ),
    )
    artifact = store_licensed_artifact(
        mirrors,
        b"restricted",
        staged_path=Path("menagerie/crawler/ports/restricted.py"),
        origin=ArtifactOrigin("https://example.test/restricted", "v1"),
        evidence=evidence,
    )
    artifact_rows = [
        {
            "staged_path": artifact.staged_path.as_posix(),
            "manifest": artifact.manifest.to_dict(),
            "decision": artifact.decision.to_dict(),
        }
    ]
    manifest = tmp_path / "staged.jsonl"
    report = tmp_path / "license-report.json"
    _write_jsonl(manifest, artifact_rows)
    result = license_sweep_main(
        [
            "--artifacts",
            str(manifest),
            "--public-root",
            str(tmp_path / "public"),
            "--private-root",
            str(tmp_path / "private"),
            "--local-root",
            str(tmp_path / "local"),
            "--report",
            str(report),
        ]
    )
    assert result != 0
    assert json.loads(report.read_text(encoding="utf-8"))["passed"] is False


def test_requeue_appends_grant_without_mutating_prior_records(tmp_path: Path) -> None:
    """A bounded grant adds one line and preserves every prior byte."""

    ledger = tmp_path / "requeue-grants.jsonl"
    _write_jsonl(ledger, [{"historical": "fact"}])
    before = ledger.read_bytes()
    result = requeue_main(
        [
            "m_example",
            "--reason",
            "JMT approved one retry",
            "--grant",
            "1",
            "--stage",
            "forward",
            "--ledger",
            str(ledger),
        ]
    )
    assert result == 0
    after = ledger.read_bytes()
    assert after.startswith(before)
    assert len(after.splitlines()) == 2
    grant = json.loads(after.splitlines()[-1])
    assert grant["attempts"] == 1
    assert grant["new_work_generation"] == 2


def _crawler_subcommands() -> set[str]:
    """Return every real CLI subcommand and alias.

    Returns
    -------
    set[str]
        Names accepted by the crawler parser.
    """

    parser = build_crawler_parser()
    action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return set(action.choices)


def test_all_procedures_are_ascii_and_reference_only_real_commands() -> None:
    """Every required procedure is present and uses accepted CLI commands."""

    procedures = Path(__file__).resolve().parents[1] / "procedures"
    required = {
        "QUICKSTART.md",
        "SETUP.md",
        "RUN.md",
        "RESUME.md",
        "TEARDOWN.md",
        "LINUX_SWEEP.md",
    }
    commands = _crawler_subcommands()
    for name in required:
        data = (procedures / name).read_bytes()
        assert data
        data.decode("ascii")
        referenced = re.findall(rb"python -m menagerie\.crawler ([a-z][a-z-]*)", data)
        assert set(item.decode("ascii") for item in referenced) <= commands
