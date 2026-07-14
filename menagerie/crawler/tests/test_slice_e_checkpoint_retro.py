"""Slice E never-push checkpoint and retro-audit wave tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from menagerie.crawler.checkpoint import (
    GitCommandResult,
    NonAllowlistedPath,
    RestrictedPublicArtifact,
    WrongCheckpointBranch,
    create_checkpoint_set,
)
from menagerie.crawler.constants import FidelityVerdict
from menagerie.crawler.licenses import (
    LicenseEvidence,
    LicenseEvidenceStatus,
    store_licensed_artifact,
)
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorStore
from menagerie.crawler.retro_audit import (
    CampaignWave,
    LegacyAuditItem,
    ReearnedRecord,
    define_campaign,
)


class RecordingGit:
    """Record argv-only Git commands without mutating a repository."""

    def __init__(self, staged: str = "") -> None:
        """Initialize a fake index response.

        Parameters
        ----------
        staged:
            NUL-delimited already-staged paths.
        """

        self.staged = staged
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], cwd: Path) -> GitCommandResult:
        """Return successful canned Git output.

        Parameters
        ----------
        argv, cwd:
            Command and ignored worktree root.

        Returns
        -------
        GitCommandResult
            Successful result.
        """

        del cwd
        command = tuple(argv)
        self.commands.append(command)
        stdout = self.staged if command[:4] == ("git", "diff", "--cached", "--name-only") else ""
        return GitCommandResult(0, stdout, "")


def _mirrors(tmp_path: Path) -> MirrorStore:
    """Return separated checkpoint mirror roots.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.

    Returns
    -------
    MirrorStore
        Separated store.
    """

    return MirrorStore(tmp_path / "pub", tmp_path / "priv", tmp_path / "local")


def test_checkpoint_stages_clean_allowlisted_set_and_never_pushes(tmp_path: Path) -> None:
    """A clean record path reaches git add and no push command."""

    candidate = Path("menagerie/crawler/records/models/part-1.jsonl")
    absolute = tmp_path / candidate
    absolute.parent.mkdir(parents=True)
    absolute.write_text("", encoding="utf-8")
    git = RecordingGit()
    result = create_checkpoint_set(
        tmp_path,
        [candidate],
        ledger_paths=[absolute],
        derived_view_checks=[lambda: None],
        public_artifacts=[],
        mirrors=_mirrors(tmp_path),
        branch="menagerie/crawler-pipeline",
        git_runner=git,
    )
    assert result.paths == (candidate,)
    assert any(command[:3] == ("git", "add", "--") for command in git.commands)
    assert all("push" not in command for command in git.commands)


def test_checkpoint_refuses_wrong_branch_and_nonallowlisted_path(tmp_path: Path) -> None:
    """Wrong branches and paths outside committed crawler facts fail typed."""

    git = RecordingGit()
    with pytest.raises(WrongCheckpointBranch):
        create_checkpoint_set(
            tmp_path,
            [],
            ledger_paths=[],
            derived_view_checks=[],
            public_artifacts=[],
            mirrors=_mirrors(tmp_path),
            branch="main",
            git_runner=git,
        )
    disallowed = Path(".crawl-local/secrets.json")
    with pytest.raises(NonAllowlistedPath):
        create_checkpoint_set(
            tmp_path,
            [disallowed],
            ledger_paths=[],
            derived_view_checks=[],
            public_artifacts=[],
            mirrors=_mirrors(tmp_path / "other"),
            branch="menagerie/crawler-pipeline",
            git_runner=git,
        )


def test_checkpoint_refuses_restricted_public_artifact(tmp_path: Path) -> None:
    """The checkpoint converts a failed license sweep into typed validation refusal."""

    mirrors = _mirrors(tmp_path)
    restricted = store_licensed_artifact(
        mirrors,
        b"restricted",
        staged_path=Path("menagerie/crawler/mirrors/private-manifest.jsonl"),
        origin=ArtifactOrigin("https://example.test/source", "v1"),
        evidence=(
            LicenseEvidence(
                "license-1",
                "source-1",
                "LICENSE:1",
                "GNU General Public License",
                LicenseEvidenceStatus.DECLARED,
                "GPL-3.0-only",
            ),
        ),
    )
    with pytest.raises(RestrictedPublicArtifact, match="license sweep rejected"):
        create_checkpoint_set(
            tmp_path,
            [],
            ledger_paths=[],
            derived_view_checks=[],
            public_artifacts=[restricted],
            mirrors=mirrors,
            branch="menagerie/crawler-pipeline",
            git_runner=RecordingGit(),
        )


def test_retro_wave_requires_full_reearned_pool_and_current_verdicts() -> None:
    """Legacy hints never award runs and wave completion needs all re-earned facts."""

    items = [
        LegacyAuditItem("m_slop_1", frozenset({"legacy-known-slop"}), frozenset()),
        LegacyAuditItem("m_slop_2", frozenset({"legacy-known-slop"}), frozenset()),
    ]
    assert all(item.legacy_untrusted and not item.inherited_award for item in items)
    wave = define_campaign(items, calibration_ids=[]).wave(CampaignWave.KNOWN_SLOP)
    first = ReearnedRecord(
        "m_slop_1",
        "runs",
        execution_reearned=True,
        current=True,
        fidelity_verdict=FidelityVerdict.SLOP,
        fidelity_current=True,
    )
    partial = wave.completion([first])
    assert not partial.complete
    second = ReearnedRecord(
        "m_slop_2",
        "failed:fidelity",
        execution_reearned=True,
        current=True,
        fidelity_verdict=FidelityVerdict.MAJOR_DRIFT,
        fidelity_current=True,
    )
    assert wave.completion([first, second]).complete
