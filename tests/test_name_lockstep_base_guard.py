"""The name lockstep ``--base`` guard (``tools/name_lockstep/baseguard.py``).

A lane once ran the gate against a commit inside its own branch and read "0 findings" while
48 stale places existed: every name added before that commit was invisible. A ref the
checkout lacks (``origin/...`` on a worker shipped without remote refs) must fail the same
loud way instead of being guessed around. A throwaway git repository reproduces each case.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from name_lockstep import baseguard  # noqa: E402
from name_lockstep.cli import BAD_BASE, main  # noqa: E402

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "HOME": "/nonexistent",
    "GIT_CONFIG_NOSYSTEM": "1",
    # Keep the caller's PATH so the git that runs is the one the suite runs
    # under; without it /usr/bin/git answers, and an old one rejects `init -b`.
    "PATH": os.environ.get("PATH", os.defpath),
}


def _git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` and return stdout."""

    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=GIT_ENV
    )
    return done.stdout.strip()


def _commit(repo: Path, name: str) -> str:
    """Write a file named ``name``, commit it and return the SHA."""

    (repo / name).write_text(name + "\n", encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def history(tmp_path: Path) -> dict[str, str]:
    """A repo with ``main`` (two commits) and a branch of three commits forked after the first.

    Returns the SHAs: ``fork`` (merge base), ``main_tip``, ``b1``, ``b2``, ``head``.
    """

    repo = tmp_path / "repo"
    (repo / "docs/reference").mkdir(parents=True)
    (repo / "docs/reference/glossary.md").write_text("nothing\n", encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    fork = _commit(repo, "a")
    _git(repo, "checkout", "-q", "-b", "feature")
    b1 = _commit(repo, "b1")
    b2 = _commit(repo, "b2")
    head = _commit(repo, "b3")
    _git(repo, "checkout", "-q", "main")
    main_tip = _commit(repo, "m2")
    _git(repo, "checkout", "-q", "feature")
    return {"repo": str(repo), "fork": fork, "main_tip": main_tip, "b1": b1, "b2": b2, "head": head}


def test_fork_point_is_an_accepted_base(history: dict[str, str]) -> None:
    """The merge base with ``main`` passes and the summary names the target."""

    repo = Path(history["repo"])
    line = baseguard.verify(repo, history["fork"], history["head"], None)
    assert line == f"base: {history['fork'][:12]} is reachable from main"


def test_branch_internal_base_is_refused(history: dict[str, str]) -> None:
    """A commit inside the branch is an ancestor of head but no merge target reaches it."""

    repo = Path(history["repo"])
    with pytest.raises(baseguard.BaseError, match="branch-internal") as caught:
        baseguard.verify(repo, history["b2"], history["head"], None)
    assert history["fork"][:12] in str(caught.value)  # the message names the fork point


def test_base_outside_head_history_is_refused(history: dict[str, str]) -> None:
    """``main``'s tip is not an ancestor of the branch head."""

    repo = Path(history["repo"])
    with pytest.raises(baseguard.BaseError, match="not an ancestor of head"):
        baseguard.verify(repo, history["main_tip"], history["head"], None)


def test_explicit_target_governs(history: dict[str, str]) -> None:
    """``--target`` replaces the ref search; ``none`` skips the reachability check only."""

    repo = Path(history["repo"])
    assert "reachable from --target main" in baseguard.verify(
        repo, history["fork"], history["head"], "main"
    )
    with pytest.raises(baseguard.BaseError, match="not reachable from --target"):
        baseguard.verify(repo, history["b1"], history["head"], "main")
    assert "target check skipped" in baseguard.verify(repo, history["b1"], history["head"], "none")
    with pytest.raises(baseguard.BaseError, match="not an ancestor"):
        baseguard.verify(repo, history["main_tip"], history["head"], "none")


def test_missing_ref_fails_with_a_hint(history: dict[str, str]) -> None:
    """An ``origin/`` ref the checkout lacks is an error that says to pass the SHA."""

    repo = Path(history["repo"])
    with pytest.raises(baseguard.BaseError, match="unknown.*remote refs"):
        baseguard.resolve(repo, "origin/wip/mp-train")
    with pytest.raises(baseguard.BaseError, match="target ref 'nope' is unknown"):
        baseguard.verify(repo, history["fork"], history["head"], "nope")
    assert baseguard.rev(repo, "ref:main") == history["main_tip"]


def test_no_target_ref_in_repo_asks_for_one(tmp_path: Path) -> None:
    """Without ``main`` or a ``*-train`` branch the base cannot be verified."""

    repo = tmp_path / "solo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "topic")
    first = _commit(repo, "a")
    head = _commit(repo, "b")
    assert baseguard.target_refs(repo) == []
    with pytest.raises(baseguard.BaseError, match="pass --target"):
        baseguard.verify(repo, first, head, None)
    _git(repo, "branch", "wip/x-train", first)
    assert baseguard.target_refs(repo) == ["wip/x-train"]
    assert "reachable from wip/x-train" in baseguard.verify(repo, first, head, None)


def _inventory(path: Path, commit: str | None) -> Path:
    """Write an empty inventory recording ``commit``."""

    data: dict = {"schema": "name-lockstep-inventory/3", "rows": []}
    if commit:
        data["commit"] = commit
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_cli_refuses_a_branch_internal_base(
    history: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``check`` exits 2 (never 0) on JSON inventories whose commits make a bad base."""

    repo = history["repo"]
    head = _inventory(tmp_path / "head.json", history["head"])
    bad = _inventory(tmp_path / "bad.json", history["b2"])
    good = _inventory(tmp_path / "good.json", history["fork"])
    assert main(["--repo", repo, "check", "--base", str(bad), "--head", str(head)]) == BAD_BASE
    assert "branch-internal" in capsys.readouterr().err
    assert main(["--repo", repo, "check", "--base", str(good), "--head", str(head)]) == 0
    assert "base:" in capsys.readouterr().err
    missing = _inventory(tmp_path / "missing.json", None)
    assert main(["--repo", repo, "check", "--base", str(missing), "--head", str(head)]) == BAD_BASE
    assert "records no commit" in capsys.readouterr().err


def test_cli_inventory_of_a_missing_ref_fails_loudly(
    history: dict[str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``inventory --ref origin/x`` exits 2 with the hint instead of a traceback."""

    out = tmp_path / "inv.json"
    rc = main(["--repo", history["repo"], "inventory", "--ref", "origin/x", "--out", str(out)])
    assert rc == BAD_BASE
    assert "remote refs" in capsys.readouterr().err
    assert not out.exists()
