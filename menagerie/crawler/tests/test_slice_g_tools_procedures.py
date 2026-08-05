"""Slice G tool-wrapper and human-procedure tests."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from menagerie.crawler.cli import build_parser as build_crawler_parser
from menagerie.crawler.constants import ACCESS_BLOCKED_STATUS_CODE, MODEL_SCHEMA_VERSION_V3
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.intake import create_intake_snapshot
from menagerie.crawler.licenses import (
    LicenseEvidence,
    LicenseEvidenceStatus,
)
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorStore
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.tests.conftest import (
    make_licensed_artifact_fixture,
    make_model,
)
from menagerie.crawler.tools.license_sweep import main as license_sweep_main
from menagerie.crawler.tools.rebuild_views import (
    _access_blocked_row,
    main as rebuild_views_main,
)
from menagerie.crawler.tools.requeue import main as requeue_main
from menagerie.crawler.tools.verify_pool_prompts import (
    PIN_SECTION_HEADING,
    main as verify_pool_prompts_main,
    verify_prompt_surface,
)
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


def _copy_prompt_pool(destination: Path) -> Path:
    """Copy the shipped pool fragments so a test can mutate them safely.

    Parameters
    ----------
    destination:
        Directory to create as the fragment copy.

    Returns
    -------
    Path
        The populated copy.
    """

    shipped = Path(__file__).resolve().parents[1] / "prompts" / "pool"
    shutil.copytree(shipped, destination)
    return destination


def test_verify_pool_prompts_covers_the_whole_shipped_prompt_surface() -> None:
    """Every top-level prompt and every dispatch-brief fragment is pinned and matches."""

    assert verify_pool_prompts_main([]) == 0
    crawler_root = Path(__file__).resolve().parents[1]
    surface = verify_prompt_surface(
        crawler_root / "PLAN.md",
        crawler_root / "prompts" / "claude_crawler_author_v2.txt",
        crawler_root / "prompts" / "codex_accuracy_checker_v2.txt",
        crawler_root / "prompts" / "pool",
    )
    fragments = {path.name for path in (crawler_root / "prompts" / "pool").glob("*.md")}
    # The executor's own stage prompts drive every production author session; the pool
    # fragments serve the operator-mediated lane it replaced. They were unpinned until
    # 2026-07-29, so this expected set must follow the verifier rather than lag it.
    fragments |= {path.name for path in (crawler_root / "prompts" / "executor").glob("*.md")}
    assert fragments
    assert set(surface) == fragments | {
        "claude_crawler_author_v2.txt",
        "codex_accuracy_checker_v2.txt",
    }


def test_stage_source_request_states_http_links_are_typed_policy_evidence() -> None:
    """The source-request prompt tells authors not to discard policy-refused links."""

    crawler_root = Path(__file__).resolve().parents[1]
    text = (crawler_root / "prompts" / "pool" / "stage_source_request.md").read_text(
        encoding="utf-8"
    )

    assert "policy-refused links are still evidence" in text
    assert "typed policy evidence" in text


def test_verify_pool_prompts_rejects_every_way_a_fragment_can_drift(tmp_path: Path) -> None:
    """A mutated, an unpinned added, and a deleted pinned fragment each fail."""

    mutated_pool = _copy_prompt_pool(tmp_path / "mutated")
    fragment = mutated_pool / "stage_source_request.md"
    fragment.write_bytes(fragment.read_bytes() + b" ")
    assert verify_pool_prompts_main(["--pool-root", str(mutated_pool)]) != 0

    added_pool = _copy_prompt_pool(tmp_path / "added")
    (added_pool / "stage_smuggled.md").write_text("unpinned guidance\n", encoding="utf-8")
    assert verify_pool_prompts_main(["--pool-root", str(added_pool)]) != 0

    deleted_pool = _copy_prompt_pool(tmp_path / "deleted")
    (deleted_pool / "stage_author.md").unlink()
    assert verify_pool_prompts_main(["--pool-root", str(deleted_pool)]) != 0

    empty_pool = tmp_path / "empty"
    empty_pool.mkdir()
    assert verify_pool_prompts_main(["--pool-root", str(empty_pool)]) != 0


def test_verify_pool_prompts_fails_when_the_pinned_section_is_removed(tmp_path: Path) -> None:
    """Deleting the PLAN oracle section is drift, never a vacuous pass."""

    crawler_root = Path(__file__).resolve().parents[1]
    plan = (crawler_root / "PLAN.md").read_text(encoding="utf-8")
    stripped = tmp_path / "PLAN.md"
    stripped.write_text(plan.replace(PIN_SECTION_HEADING, "## 18. Removed"), encoding="utf-8")
    assert verify_pool_prompts_main(["--plan", str(stripped)]) != 0


def test_rebuild_views_is_deterministic_and_ignores_stale_database(tmp_path: Path) -> None:
    """Canonical JSONL yields identical view hashes despite stale derived state."""

    master = tmp_path / "master.jsonl"
    deferred = tmp_path / "deferred.jsonl"
    _write_jsonl(master, [{"name": "Example", "zoo": "fixtures", "variant": "base"}])
    _write_jsonl(deferred, [])
    snapshot = create_intake_snapshot(master, deferred, tmp_path / "intake")
    records = tmp_path / "records"
    ledgers = LedgerPaths(
        records / "models" / "current-shard.jsonl",
        records / "attempts" / "local.jsonl",
        records / "gates" / "current-shard.jsonl",
    )
    # The rebuild path projects the unified v3 ledger; opening it as v2 would
    # exercise a removed shadow schema instead of current-only append behavior.
    with JsonlLedger(ledgers.models, MODEL_SCHEMA_VERSION_V3) as ledger:
        ledger.append(make_model(snapshot.items[0].stable_id, accepted=True))
    intake = snapshot.root / "items.jsonl"
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


def test_blocked_on_access_view_is_an_actionable_recovery_worklist() -> None:
    """Blocked-only-on-access and what was unreachable must be one file read.

    The campaign runs once, and the batch that would recover these models has to be
    assembled from the record months later. A count cannot be actioned; the exact
    locator, the machine-derived identity, when we tried, and what we observed can be.
    """

    record = {
        "stable_id": "m_paywalled",
        "status": {"code": ACCESS_BLOCKED_STATUS_CODE, "kind": "deferred"},
        "source_resolution": {
            "search_report": {"conclusion": "The specifying paper is behind a paywall."},
        },
        "discovery_probes": [
                {
                    "identifier_kind": "doi",
                    "identifier": "10.1109/5.726791",
                    "locator": "https://doi.org/10.1109/5.726791",
                    "attempted_at": "2026-07-30T00:00:00Z",
                    "probe_outcome": "unreachable",
                    "http_status": 403,
                    "author_claimed_class": "access-barrier",
                },
                {
                    "identifier_kind": None,
                    "identifier": None,
                    "locator": "https://example.com/other",
                    "attempted_at": "2026-07-30T00:00:00Z",
                    "probe_outcome": "fetched",
                    "http_status": 200,
                    "author_claimed_class": "not-this-model",
                },
        ],
    }

    row = _access_blocked_row(record)

    assert row["stable_id"] == "m_paywalled"
    assert row["conclusion"] == "The specifying paper is behind a paywall."
    # Only the barriers, not every locator the session happened to look at.
    assert [barrier["locator"] for barrier in row["barriers"]] == [
        "https://doi.org/10.1109/5.726791"
    ]
    barrier = row["barriers"][0]
    assert barrier["identifier_kind"] == "doi"
    assert barrier["identifier"] == "10.1109/5.726791"
    assert barrier["attempted_at"] == "2026-07-30T00:00:00Z"
    assert barrier["http_status"] == 403


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
    artifact = make_licensed_artifact_fixture(
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
            "--intent",
            "graph",
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
    assert grant["target_intent"] == "graph"


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
