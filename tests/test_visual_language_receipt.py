"""The deck's environment record names the commit it drew, and its receipt counts whole panels.

``environment.json`` must name the commit whose tree was rendered: under a tree-snapshot
commit (one that copies another commit's tree and names it in a ``Source-Commit`` trailer)
that is the copied commit, and a ``git archive`` export names it through the environment.
The coverage receipt must count a row drawn by whole panels (one panel per variant, or every
panel of a slide that keys no mark) as drawn, not as "claimed but unproven".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.visual_language import deck


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("a\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-qm", "branch commit")
    return tmp_path


def test_snapshot_commit_records_the_commit_it_copies(repo: Path, monkeypatch) -> None:
    """A tree snapshot of a branch commit records the branch commit, and keeps its own as aside."""

    monkeypatch.delenv(deck.COMMIT_ENV, raising=False)
    branch = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    snapshot = _git(
        repo, "commit-tree", tree, "-m", "worker tree snapshot", "-m", f"Source-Commit: {branch}"
    )
    _git(repo, "checkout", "-q", snapshot)
    record = deck.build_commit(repo)
    assert record == {
        "commit": branch,
        "commit_source": "snapshot trailer",
        "checkout_commit": snapshot,
    }


def test_plain_checkout_records_its_head(repo: Path, monkeypatch) -> None:
    monkeypatch.delenv(deck.COMMIT_ENV, raising=False)
    head = _git(repo, "rev-parse", "HEAD")
    assert deck.build_commit(repo) == {"commit": head, "commit_source": "checkout"}


def _records(errored: set[str] = frozenset()) -> list[dict]:
    """Slide records as a build writes them: every panel rendered except ``errored`` stems."""

    out = []
    for slide in deck.deck_slides():
        panels = {}
        for panel in slide.panels:
            stem = f"{slide.id}-{panel.name}"
            panels[stem] = {"error": "boom"} if stem in errored else {"stem": stem}
        out.append({"id": slide.id, "status": "ok", "witnesses": {}, "panels": panels})
    return out


def _row(receipt: dict, row_id: str) -> dict:
    return next(r for r in receipt["rows"] if r["row"] == row_id)
