"""Tripwire: a major-version release happens only on the maintainer's explicit command.

Releases publish from a maintainer-pushed ``vX.Y.Z`` tag. Two mechanical stops
need no maintainer action to stay armed:

* release time: the ``build`` job of ``.github/workflows/release.yml`` runs
  ``scripts/check_release_major.py`` on every pull request and tag, before the
  build, and ``publish`` needs that job;
* commit and push time: ``scripts/check_no_breaking_markers.py`` rejects
  ``type!:`` subjects and breaking-change footers unless the approval file
  exists.

The approval file ``.github/MAJOR_RELEASE_APPROVED`` must never exist in the
tree outside a test's temporary directory. The end-to-end test runs the
workflow step's own shell text against a scratch repository with a
breaking-change commit range. No network (the PyPI lookup is replaced by an
explicit published version), well under 5 s.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.smoke

_ROOT = Path(__file__).resolve().parents[1]
_APPROVAL_REL = Path(".github") / "MAJOR_RELEASE_APPROVED"
_HOOK = _ROOT / "scripts" / "check_no_breaking_markers.py"
_RELEASE_GUARD = _ROOT / "scripts" / "check_release_major.py"
_STEP_NAME = "Refuse an unapproved major version"

#: Assembled at runtime so this file itself never trips the commit-msg layer.
_BANG_SUBJECT = "feat" + "!: drop the old API"
_FOOTERS = ("BREAKING" + " CHANGE: gone", "BREAKING" + "-CHANGE: gone")


def _release_jobs() -> dict:
    workflow = yaml.safe_load((_ROOT / ".github" / "workflows" / "release.yml").read_text())
    return workflow["jobs"]


def _guard_step(steps: list[dict]) -> tuple[int, dict]:
    return next((i, s) for i, s in enumerate(steps) if s.get("name") == _STEP_NAME)


def test_release_guard_is_wired_and_cannot_be_skipped() -> None:
    """The guard runs unconditionally before the build; publish needs that job; hooks install."""

    jobs = _release_jobs()
    build = jobs["build"]
    assert "if" not in build, "the build job (and its major guard) must run on every trigger"
    steps = build["steps"]
    index, step = _guard_step(steps)
    run = step["run"]
    assert "scripts/check_release_major.py" in run
    for flag in ("--require-tags", "--released-from-pypi", '--next "$GITHUB_REF_NAME"'):
        assert flag in run, flag
    assert "if" not in step and "continue-on-error" not in step
    assert "env" not in step, "the guard needs no token or environment"
    first_install = next(i for i, s in enumerate(steps) if "pip install" in s.get("run", ""))
    build_step = next(i for i, s in enumerate(steps) if "python -m build" in s.get("run", ""))
    assert index < first_install < build_step
    checkout = steps[0]["with"]
    assert checkout["fetch-depth"] == 0 and checkout["fetch-tags"] is True
    assert "build" in jobs["publish"]["needs"]

    assert not (_ROOT / _APPROVAL_REL).exists(), (
        f"{_APPROVAL_REL} exists; it is created only on the maintainer's command"
    )

    config = yaml.safe_load((_ROOT / ".pre-commit-config.yaml").read_text())
    assert {"commit-msg", "pre-push"} <= set(config["default_install_hook_types"])
    entries = {
        hook["entry"]: hook.get("stages")
        for repo in config["repos"]
        for hook in repo["hooks"]
        if "entry" in hook
    }
    assert entries.get("scripts/check_no_breaking_markers.py --commit-msg") == ["commit-msg"]
    assert entries.get("scripts/check_no_breaking_markers.py --pre-push") == ["pre-push"]


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


def test_commit_msg_hook_rejects_markers_and_accepts_normal_messages(tmp_path: Path) -> None:
    """A ``type!:`` subject and both footer spellings are rejected; normal text passes."""

    for message in (_BANG_SUBJECT, *(f"feat: x\n\n{footer}\n" for footer in _FOOTERS)):
        assert _commit_msg(tmp_path, message).returncode == 1, message
    assert _commit_msg(tmp_path, "feat: add a thing\n\nBody text.\n").returncode == 0
    approval = tmp_path / _APPROVAL_REL
    approval.parent.mkdir()
    approval.write_text("3.0.0\n", encoding="utf-8")
    assert _commit_msg(tmp_path, _BANG_SUBJECT).returncode == 0


class _Repo:
    """A scratch repository carrying the version files and the release check."""

    def __init__(self, root: Path, bin_dir: Path) -> None:
        self.root = root
        self.env = {
            "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
            "HOME": str(root),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        }
        (root / "scripts").mkdir(parents=True)
        (root / "torchlens").mkdir()
        shutil.copy(_RELEASE_GUARD, root / "scripts" / _RELEASE_GUARD.name)
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, env=self.env, check=True, capture_output=True, text=True
        ).stdout.strip()

    def release(self, version: str, message: str, *, tag: bool = True) -> str:
        (self.root / "pyproject.toml").write_text(
            f'[project]\nname = "torchlens"\nversion = "{version}"\n', encoding="utf-8"
        )
        (self.root / "torchlens" / "__init__.py").write_text(
            f'__version__ = "{version}"\n', encoding="utf-8"
        )
        self.git("add", "-A")
        self.git("commit", "-q", "--no-verify", "-m", message)
        if tag:
            self.git("tag", f"v{version}")
        return self.git("rev-parse", "HEAD")

    def run_step(self, run: str, ref_type: str, ref_name: str) -> subprocess.CompletedProcess[str]:
        env = {**self.env, "GITHUB_REF_TYPE": ref_type, "GITHUB_REF_NAME": ref_name}
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", run],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def markers(self, base: str, head: str) -> int:
        """Run the pre-push marker scan over ``base..head`` as raw git would feed it."""

        env = {k: v for k, v in os.environ.items() if not k.startswith("PRE_COMMIT")}
        env.update(self.env)
        return subprocess.run(
            [sys.executable, str(_HOOK), "--pre-push", "origin", "unused"],
            cwd=self.root,
            env=env,
            input=f"refs/heads/main {head} refs/heads/main {base}\n",
            capture_output=True,
            text=True,
            check=False,
        ).returncode


def test_release_step_stops_a_breaking_range_and_passes_minor_and_patch(tmp_path: Path) -> None:
    """The workflow's own step refuses 3.0.0 from a breaking range; 2.x releases pass."""

    _, step = _guard_step(_release_jobs()["build"]["steps"])
    assert "--released-from-pypi" in step["run"]
    # Offline stand-in for the PyPI lookup: the published major is 2.
    run = step["run"].replace("--released-from-pypi", "--released 2.36.3")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)

    repo = _Repo(tmp_path / "repo", bin_dir)
    no_tags = repo.release("2.36.3", "chore: first", tag=False)
    assert repo.run_step(run, "branch", "main").returncode == 2  # no tag: fail closed
    repo.git("tag", "v2.36.3", no_tags)

    breaking = f"feat(api): drop the old API\n\n{_FOOTERS[0]}\n"
    major = repo.release("3.0.0", breaking)
    refused = repo.run_step(run, "tag", "v3.0.0")
    assert refused.returncode == 1 and "REFUSING major bump" in refused.stderr
    assert repo.run_step(run, "branch", "main").returncode == 1  # the pull-request dry run
    assert repo.markers(no_tags, major) == 1  # the marker scan refuses the same range

    # A second tag on top of a refused one: the published major still holds the baseline.
    repo.release("3.0.1", "fix: retry")
    assert repo.run_step(run, "tag", "v3.0.1").returncode == 1

    # Approval for exactly the tagged version is the only way through.
    approval = repo.root / _APPROVAL_REL
    approval.parent.mkdir()
    approval.write_text("3.1.0\n", encoding="utf-8")
    assert repo.run_step(run, "tag", "v3.0.1").returncode == 1
    approval.write_text("3.0.1\n", encoding="utf-8")
    assert repo.run_step(run, "tag", "v3.0.1").returncode == 0
    approval.unlink()

    normal = _Repo(tmp_path / "normal", bin_dir)
    base = normal.release("2.36.3", "chore: first")
    head = normal.release("2.37.0", "feat: add a thing")
    assert normal.run_step(run, "tag", "v2.37.0").returncode == 0
    assert normal.markers(base, head) == 0
    normal.release("2.37.1", "fix: a thing")
    assert normal.run_step(run, "tag", "v2.37.1").returncode == 0
    assert normal.run_step(run, "branch", "main").returncode == 0
