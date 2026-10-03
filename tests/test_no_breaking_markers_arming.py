"""Behavioral arming tests for the never-ship-a-major hook script.

``scripts/check_no_breaking_markers.py`` is layers 1 (commit-msg) and 3
(pre-push) of the locked no-major-bump release policy, and until grind r4 it
had ZERO behavioral coverage — ``tests/test_packaging_diet.py`` asserts only
that the pre-commit CONFIG text names the hook. That is how the pre-push
fail-open shipped (b9-opus R70r4-F1): a push to a ref-less remote gets no
``PRE_COMMIT_FROM_REF``/``TO_REF`` from the framework and empty stdin, so
``_push_records()`` returned ``[]`` and the hook printed "Passed" having
scanned nothing — a ``BREAKING CHANGE:`` commit landed.

These tests run the real script as a subprocess against a scratch git repo
(global/system git config suppressed so this machine's ``core.hooksPath``
and identity never leak in), covering every input contract: raw-git stdin
refspecs, the framework's FROM/TO refs, the framework's ref-less-remote
branch-only environment, and the no-range fail-closed guard.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_breaking_markers.py"

_ZEROS = "0" * 40

#: Assembled at runtime so this test file itself can never trip the
#: commit-msg layer of the hook chain.
_BREAKING_FOOTER = "BREAKING" + " CHANGE: everything"


def _clean_env(cwd: Path) -> dict[str, str]:
    """Environment for git/script subprocesses: isolated and unarmed."""

    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("PRE_COMMIT") and name != "TORCHLENS_ALLOW_MAJOR_BUMP"
    }
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "arming-test"
    env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "arming@test.invalid"
    env["HOME"] = str(cwd)  # no user hooks / templates
    return env


def _git(repo: Path, *args: str, env: dict[str, str]) -> str:
    """Run one git command in ``repo`` and return stripped stdout."""

    return subprocess.run(
        ["git", "-C", str(repo), *args],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture()
def push_repo(tmp_path: Path) -> tuple[Path, dict[str, str], str, str, str]:
    """Scratch repo: clean base commit, then a BREAKING-footer commit on a topic branch.

    Returns (repo, env, base_sha, clean_tip_sha, breaking_tip_sha) with
    ``refs/remotes/origin/main`` pinned at the base so the script's
    merge-base scan path resolves exactly the outgoing commits.
    """

    repo = tmp_path / "repo"
    repo.mkdir()
    env = _clean_env(tmp_path)
    # `init -b` needs git >= 2.28; naming the unborn branch with symbolic-ref
    # works on every git the hooks run under.
    _git(repo, "init", "-q", env=env)
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/main", env=env)
    (repo / "f.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "f.txt", env=env)
    _git(repo, "commit", "-q", "-m", "chore: base", env=env)
    base = _git(repo, "rev-parse", "HEAD", env=env)
    # Simulate the tracked remote the script's merge-base path keys on.
    _git(repo, "update-ref", "refs/remotes/origin/main", base, env=env)

    _git(repo, "checkout", "-q", "-b", "clean-topic", env=env)
    (repo / "f.txt").write_text("clean\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "fix: a harmless change", env=env)
    clean_tip = _git(repo, "rev-parse", "HEAD", env=env)

    _git(repo, "checkout", "-q", "-b", "breaking-topic", "main", env=env)
    (repo / "f.txt").write_text("breaking\n", encoding="utf-8")
    _git(
        repo,
        "commit",
        "-q",
        "-am",
        f"feat: rework the API\n\n{_BREAKING_FOOTER}",
        env=env,
    )
    breaking_tip = _git(repo, "rev-parse", "HEAD", env=env)
    return repo, env, base, clean_tip, breaking_tip


def _run_pre_push(
    repo: Path,
    env: dict[str, str],
    *,
    stdin: str = "",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the script in --pre-push mode exactly as the hook layer would."""

    run_env = dict(env)
    if extra_env:
        run_env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--pre-push"],
        cwd=repo,
        env=run_env,
        input=stdin,
        capture_output=True,
        text=True,
    )


def test_raw_git_stdin_blocks_breaking_and_passes_clean(push_repo) -> None:
    """The raw-git refspec contract: BREAKING blocked (rc 1), clean passes (rc 0)."""

    repo, env, base, clean_tip, breaking_tip = push_repo
    blocked = _run_pre_push(
        repo, env, stdin=f"refs/heads/breaking-topic {breaking_tip} refs/heads/x {base}\n"
    )
    assert blocked.returncode == 1, blocked.stderr
    assert "BLOCKED" in blocked.stderr
    clean = _run_pre_push(
        repo, env, stdin=f"refs/heads/clean-topic {clean_tip} refs/heads/x {base}\n"
    )
    assert clean.returncode == 0, clean.stderr


def test_framework_from_to_refs_block_breaking(push_repo) -> None:
    """The pre-commit framework's ref-ful contract (FROM/TO env, empty stdin)."""

    repo, env, base, clean_tip, breaking_tip = push_repo
    blocked = _run_pre_push(
        repo,
        env,
        extra_env={"PRE_COMMIT_FROM_REF": base, "PRE_COMMIT_TO_REF": breaking_tip},
    )
    assert blocked.returncode == 1, blocked.stderr
    clean = _run_pre_push(
        repo,
        env,
        extra_env={"PRE_COMMIT_FROM_REF": base, "PRE_COMMIT_TO_REF": clean_tip},
    )
    assert clean.returncode == 0, clean.stderr


def test_refless_remote_branch_only_env_blocks_breaking(push_repo) -> None:
    """The R70r4-F1 fail-open shape: branch-only framework env must still scan.

    Pushing to a ref-less remote exports PRE_COMMIT_LOCAL_BRANCH but no
    FROM/TO refs, with empty stdin. Pre-fix this scanned nothing and passed;
    the synthesized new-branch record must scan merge-base..tip and block.
    """

    repo, env, _base, clean_tip, _breaking_tip = push_repo
    blocked = _run_pre_push(
        repo,
        env,
        extra_env={"PRE_COMMIT_LOCAL_BRANCH": "refs/heads/breaking-topic"},
    )
    assert blocked.returncode == 1, blocked.stderr
    assert "BLOCKED" in blocked.stderr
    clean = _run_pre_push(
        repo,
        env,
        extra_env={"PRE_COMMIT_LOCAL_BRANCH": "refs/heads/clean-topic"},
    )
    assert clean.returncode == 0, clean.stderr
    assert clean_tip  # the clean branch really is the scanned one


def test_undeterminable_range_fails_closed(push_repo) -> None:
    """Empty stdin + no framework env = rc 2, never a silent pass."""

    repo, env, *_ = push_repo
    result = _run_pre_push(repo, env)
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert "could not determine the push range" in result.stderr


def test_branch_deletion_record_passes(push_repo) -> None:
    """A deletion refspec (local sha all zeros) has nothing to scan."""

    repo, env, base, *_ = push_repo
    result = _run_pre_push(repo, env, stdin=f"(delete) {_ZEROS} refs/heads/gone {base}\n")
    assert result.returncode == 0, result.stderr


def test_commit_msg_layer_blocks_bang_and_passes_clean(push_repo, tmp_path: Path) -> None:
    """Layer 1: a type! marker is refused, a plain conventional message passes."""

    repo, env, *_ = push_repo
    msg = tmp_path / "msg.txt"
    msg.write_text("feat" + "!: breaking spelling\n", encoding="utf-8")
    blocked = subprocess.run(
        [sys.executable, str(_SCRIPT), "--commit-msg", str(msg)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert blocked.returncode == 1, blocked.stderr
    msg.write_text("chore: routine\n", encoding="utf-8")
    clean = subprocess.run(
        [sys.executable, str(_SCRIPT), "--commit-msg", str(msg)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert clean.returncode == 0, clean.stderr


def test_explicit_override_allows_with_loud_notice(push_repo) -> None:
    """TORCHLENS_ALLOW_MAJOR_BUMP=1 (maintainer-authorized) permits but announces."""

    repo, env, base, _clean_tip, breaking_tip = push_repo
    result = _run_pre_push(
        repo,
        env,
        stdin=f"refs/heads/breaking-topic {breaking_tip} refs/heads/x {base}\n",
        extra_env={"TORCHLENS_ALLOW_MAJOR_BUMP": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "active; allowing" in result.stderr


@pytest.mark.parametrize("value", ["FALSE", "no", "yes", "true", "2", " 0 "])
def test_non_exact_override_values_still_block(push_repo, value: str) -> None:
    """Only the exact string "1" authorizes a major marker.

    The former truthy parse (`not in ("", "0", "false", "False")`) authorized
    a breaking push on OVERRIDE=FALSE, =no, or any templated junk — values a
    user sets to DISABLE the override — while the notice claimed "=1 active"
    (grind r5, b10 R86 probe: FALSE -> rc=0).
    """

    repo, env, base, _clean_tip, breaking_tip = push_repo
    result = _run_pre_push(
        repo,
        env,
        stdin=f"refs/heads/breaking-topic {breaking_tip} refs/heads/x {base}\n",
        extra_env={"TORCHLENS_ALLOW_MAJOR_BUMP": value},
    )
    assert result.returncode == 1, (
        f"OVERRIDE={value!r} must NOT authorize a major marker: {result.stderr}"
    )
