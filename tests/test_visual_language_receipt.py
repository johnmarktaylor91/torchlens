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


def test_exported_tree_records_the_commit_its_caller_names(tmp_path: Path, monkeypatch) -> None:
    """A tree with no Git data (``git archive``) records the commit named in the environment."""

    monkeypatch.setenv(deck.COMMIT_ENV, "f" * 40)
    assert deck.build_commit(tmp_path) == {"commit": "f" * 40, "commit_source": "environment"}
    monkeypatch.delenv(deck.COMMIT_ENV)
    assert deck.build_commit(tmp_path)["commit"] == "unknown"


def test_environment_record_builds() -> None:
    """The environment record (versions, commit, date) builds on the pinned Python."""

    env = deck._environment()
    assert env["built"].endswith("UTC") and env["commit"] and env["commit_source"]


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


def test_whole_panel_witnesses_count_as_proof() -> None:
    """Rows drawn by one panel per variant, or by every panel of an unkeyed slide, are proven."""

    receipt = deck.receipt(_records())
    skins = _row(receipt, "VT01")
    assert skins["proof"] == "panel" and skins["proven_by_picture"]
    assert skins["variants_unshown"] == []
    assert _row(receipt, "VT02")["proof"] == "panel"
    assert _row(receipt, "VK02")["proof"] == "panel"
    assert _row(receipt, "VO01")["proof"] == "measured"
    unproven = receipt["counts"]["picture_claimed_unproven"]
    assert not {"VT01", "VT02", "VK02", "VO01", "VO02"} & set(unproven)


def test_a_missing_panel_leaves_its_variant_unproven() -> None:
    """A variant whose only panel failed is not drawn, so its row stays unproven."""

    dark = {
        f"{slide.id}-{panel.name}"
        for slide in deck.deck_slides()
        for panel in slide.panels
        if panel.kwargs.get("theme") == "dark"
    }
    assert any(stem.startswith("skins-") for stem in dark)
    receipt = deck.receipt(_records(dark))
    row = _row(receipt, "VT01")
    assert row["variants_unshown"] == ["dark"]
    assert row["proof"] is None
    assert "VT01" in receipt["counts"]["picture_claimed_unproven"]
