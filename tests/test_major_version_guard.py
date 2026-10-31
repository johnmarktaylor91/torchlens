"""Tripwire: a major-version release happens only on the maintainer's explicit command.

Pins the two mechanical stops that need no maintainer action to stay armed:

* release time: ``.github/workflows/release.yml`` runs
  ``scripts/check_release_major.py`` on a side-effect-free next-version
  computation BEFORE ``semantic-release version``, with no token;
* commit time: ``scripts/check_no_breaking_markers.py --commit-msg`` rejects
  ``type!:`` subjects and breaking-change footers unless the approval file
  exists.

The approval file ``.github/MAJOR_RELEASE_APPROVED`` must never exist in the
tree outside a test's temporary directory. Pure file reads and subprocesses on
scratch files: no network, well under 5 s.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.smoke

_ROOT = Path(__file__).resolve().parents[1]
_APPROVAL = _ROOT / ".github" / "MAJOR_RELEASE_APPROVED"
_HOOK = _ROOT / "scripts" / "check_no_breaking_markers.py"
_RELEASE_GUARD = _ROOT / "scripts" / "check_release_major.py"

#: Assembled at runtime so this file itself never trips the commit-msg layer.
_BANG_SUBJECT = "feat" + "!: drop the old API"
_FOOTERS = ("BREAKING" + " CHANGE: gone", "BREAKING" + "-CHANGE: gone")


def _release_steps() -> list[dict]:
    workflow = yaml.safe_load((_ROOT / ".github" / "workflows" / "release.yml").read_text())
    return workflow["jobs"]["release"]["steps"]


def _step_index(steps: list[dict], needle: str) -> int:
    return next(i for i, step in enumerate(steps) if needle in step.get("run", ""))


def test_release_guard_runs_before_semantic_release_version() -> None:
    """The hard stop precedes the version step, computes side-effect-free, holds no token."""

    steps = _release_steps()
    guard = _step_index(steps, "scripts/check_release_major.py")
    version = next(
        i
        for i, step in enumerate(steps)
        if step.get("run", "").strip() == "semantic-release version"
    )
    assert guard < version
    run = steps[guard]["run"]
    assert "semantic-release --noop version --print" in run
    assert "GH_TOKEN" not in str(steps[guard].get("env", {}))
    assert "continue-on-error" not in steps[guard]
    assert "if" not in steps[guard]


def test_approval_file_is_absent_from_the_tree() -> None:
    """Only the maintainer's explicit command may create the approval file."""

    assert not _APPROVAL.exists(), (
        f"{_APPROVAL} exists; it is created only on the maintainer's command"
    )


def _commit_msg(tmp_path: Path, message: str) -> subprocess.CompletedProcess[str]:
    msg = tmp_path / "COMMIT_EDITMSG"
    msg.write_text(message, encoding="utf-8")
    # cwd is a non-repo tmp dir, so the hook's approval lookup cannot see a
    # real checkout's file; the result depends only on the message.
    return subprocess.run(
        [sys.executable, str(_HOOK), "--commit-msg", str(msg)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )


def test_commit_msg_hook_rejects_major_markers(tmp_path: Path) -> None:
    """A ``type!:`` subject and both breaking-footer spellings are each rejected."""

    for message in (_BANG_SUBJECT, *(f"feat: x\n\n{footer}\n" for footer in _FOOTERS)):
        assert _commit_msg(tmp_path, message).returncode == 1, message


def test_commit_msg_hook_accepts_a_normal_message_and_approval_file(tmp_path: Path) -> None:
    assert _commit_msg(tmp_path, "feat: add a thing\n\nBody text.\n").returncode == 0
    approval = tmp_path / ".github" / "MAJOR_RELEASE_APPROVED"
    approval.parent.mkdir()
    approval.write_text("3.0.0\n", encoding="utf-8")
    assert _commit_msg(tmp_path, _BANG_SUBJECT).returncode == 0


def test_release_guard_refuses_unapproved_major(tmp_path: Path) -> None:
    """The release check refuses X+1.0.0 unless the approval file names exactly it."""

    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "2.36.0"\n', encoding="utf-8")

    def run(next_version: str) -> int:
        return subprocess.run(
            [sys.executable, str(_RELEASE_GUARD), "--next", next_version, "--root", str(tmp_path)],
            capture_output=True,
            text=True,
            check=False,
        ).returncode

    assert run("2.37.0") == 0
    assert run("3.0.0") == 1
    assert run("") == 1  # unparsable output fails closed
    approval = tmp_path / ".github" / "MAJOR_RELEASE_APPROVED"
    approval.parent.mkdir()
    approval.write_text("3.1.0\n", encoding="utf-8")
    assert run("3.0.0") == 1  # approval for a different version does not count
    approval.write_text("3.0.0\n", encoding="utf-8")
    assert run("3.0.0") == 0
