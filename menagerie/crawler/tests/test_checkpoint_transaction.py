"""End-to-end fail-closed crawler checkpoint transaction tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

import menagerie.crawler.checkpoint as checkpoint_module
from menagerie.crawler.checkpoint import (
    CheckpointValidationError,
    GitCommandResult,
    RestrictedPublicArtifact,
    WrongCheckpointBranch,
    create_canonical_checkpoint,
)
from menagerie.crawler.cli import main
from menagerie.crawler.constants import MODEL_SCHEMA_VERSION
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.licenses import (
    LicenseEvidence,
    LicenseEvidenceStatus,
    pre_public_merge_sweep,
    store_licensed_artifact,
)
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorStore
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.tests.conftest import make_model
from menagerie.crawler.tools.rebuild_views import rebuild_views


class RecordingGit:
    """Record checkpoint Git commands and emulate an empty index."""

    def __init__(self, staged: Sequence[Path] = ()) -> None:
        """Initialize with optional already-staged repository-relative paths."""

        self.staged = tuple(staged)
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], cwd: Path) -> GitCommandResult:
        """Return deterministic branch, index, and add results."""

        del cwd
        command = tuple(argv)
        self.commands.append(command)
        if command[:4] == ("git", "diff", "--cached", "--name-only"):
            return GitCommandResult(0, "\0".join(path.as_posix() for path in self.staged), "")
        return GitCommandResult(0, "", "")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write canonical newline-terminated JSON objects."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _mirrors(root: Path) -> MirrorStore:
    """Return conventional separated runtime mirror roots."""

    runtime = root / ".crawl-local" / "mirrors"
    return MirrorStore(runtime / "public", runtime / "private", runtime / "local")


def _snapshot(root: Path, count: int = 1) -> IntakeSnapshot:
    """Create one canonical intake snapshot below the records allowlist."""

    master = root / "master.jsonl"
    deferred = root / "deferred.jsonl"
    _write_jsonl(
        master,
        [{"name": f"Example-{index}", "zoo": "tests", "variant": "base"} for index in range(count)],
    )
    _write_jsonl(deferred, [])
    return create_intake_snapshot(
        master,
        deferred,
        root / "menagerie" / "crawler" / "records" / "intake",
    )


def _clean_state(root: Path, *, intake_count: int = 1) -> tuple[IntakeSnapshot, MirrorStore]:
    """Materialize a complete canonical mini-campaign and matching derived facts."""

    snapshot = _snapshot(root, intake_count)
    crawler = root / "menagerie" / "crawler"
    records = crawler / "records"
    with JsonlLedger(records / "models" / "current-shard.jsonl", MODEL_SCHEMA_VERSION) as ledger:
        ledger.append(make_model(snapshot.items[0].stable_id, accepted=True))
    _write_jsonl(records / "attempts" / "local.jsonl", [])
    _write_jsonl(records / "gates" / "current-shard.jsonl", [])
    rebuild_views(
        snapshot.root / "items.jsonl",
        records,
        crawler / "views",
        root / ".crawl-local" / "state.sqlite",
    )
    _write_jsonl(crawler / "mirrors" / "public-manifest.jsonl", [])
    _write_jsonl(crawler / "mirrors" / "private-manifest.jsonl", [])
    mirrors = _mirrors(root)
    report = pre_public_merge_sweep([], mirrors)
    _write_jsonl(crawler / "license_reports" / "current.json", [report.to_dict()])
    return snapshot, mirrors


def test_checkpoint_refuses_wrong_branch_before_staging(tmp_path: Path) -> None:
    """A non-crawler branch is a typed refusal and cannot reach Git add."""

    git = RecordingGit()
    with pytest.raises(WrongCheckpointBranch):
        create_canonical_checkpoint(
            tmp_path,
            tmp_path / "missing-intake",
            branch="main",
            git_runner=git,
        )
    assert all(command[:3] != ("git", "add", "--") for command in git.commands)


def test_checkpoint_cli_wrong_branch_is_nonzero_and_never_claims_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI maps a typed branch refusal to nonzero without a success claim."""

    git = RecordingGit()

    def wrong_branch_git(argv: Sequence[str], cwd: Path) -> GitCommandResult:
        """Return ``main`` for branch inspection and record every Git command."""

        result = git(argv, cwd)
        if tuple(argv) == ("git", "branch", "--show-current"):
            return GitCommandResult(0, "main\n", "")
        return result

    monkeypatch.setattr(checkpoint_module, "_run_git", wrong_branch_git)
    exit_code = main(
        [
            "--repo-root",
            str(tmp_path),
            "checkpoint",
            "--intake",
            str(tmp_path / "missing-intake"),
            "--verify-ledgers",
            "--verify-views",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code != 0
    assert '"verified": true' not in captured.out.lower()
    assert "requires branch" in captured.err
    assert all(command[:3] != ("git", "add", "--") for command in git.commands)


def test_checkpoint_refuses_incomplete_partition(tmp_path: Path) -> None:
    """An intake model without a current terminal record blocks the transaction."""

    snapshot, mirrors = _clean_state(tmp_path, intake_count=2)
    with pytest.raises(CheckpointValidationError, match="incomplete"):
        create_canonical_checkpoint(
            tmp_path,
            snapshot.root,
            mirrors=mirrors,
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )


def test_checkpoint_derives_and_refuses_restricted_public_artifact(tmp_path: Path) -> None:
    """A restricted public-manifest row is found without a caller artifact list."""

    snapshot, mirrors = _clean_state(tmp_path)
    artifact = store_licensed_artifact(
        mirrors,
        b"restricted source",
        staged_path=Path("menagerie/crawler/evidence/restricted.txt"),
        origin=ArtifactOrigin("https://example.test/restricted", "v1"),
        evidence=(
            LicenseEvidence(
                "license-gpl",
                "source-gpl",
                "LICENSE:1",
                "GPL license text",
                LicenseEvidenceStatus.DECLARED,
                "GPL-3.0-only",
            ),
        ),
    )
    row = {
        "staged_path": artifact.staged_path.as_posix(),
        "manifest": artifact.manifest.to_dict(),
        "decision": artifact.decision.to_dict(),
    }
    _write_jsonl(tmp_path / "menagerie" / "crawler" / "mirrors" / "public-manifest.jsonl", [row])
    with pytest.raises(RestrictedPublicArtifact):
        create_canonical_checkpoint(
            tmp_path,
            snapshot.root,
            mirrors=mirrors,
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )


def test_checkpoint_refuses_missing_license_report(tmp_path: Path) -> None:
    """A fresh sweep cannot substitute for its required persisted report."""

    snapshot, mirrors = _clean_state(tmp_path)
    (tmp_path / "menagerie" / "crawler" / "license_reports" / "current.json").unlink()
    with pytest.raises(CheckpointValidationError, match="persisted license report"):
        create_canonical_checkpoint(
            tmp_path,
            snapshot.root,
            mirrors=mirrors,
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )


def test_clean_checkpoint_stages_only_derived_allowlist_and_never_pushes(tmp_path: Path) -> None:
    """Clean canonical state stages every derived fact, no runtime state, and never pushes."""

    snapshot, mirrors = _clean_state(tmp_path)
    runtime_secret = tmp_path / ".crawl-local" / "secret.json"
    runtime_secret.parent.mkdir(parents=True, exist_ok=True)
    runtime_secret.write_text(json.dumps({"token": "must-not-stage"}), encoding="utf-8")
    git = RecordingGit()
    result = create_canonical_checkpoint(
        tmp_path,
        snapshot.root,
        mirrors=mirrors,
        branch="menagerie/crawler-pipeline",
        git_runner=git,
    )
    add = next(command for command in git.commands if command[:3] == ("git", "add", "--"))
    staged = set(add[3:])
    assert staged == {path.as_posix() for path in result.paths}
    assert "menagerie/crawler/license_reports/current.json" in staged
    assert all(not path.startswith(".crawl-local/") for path in staged)
    assert all("push" not in command for command in git.commands)
