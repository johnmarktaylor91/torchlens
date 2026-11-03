"""The release job's major guard against real semantic-release on breaking commit ranges.

``tests/test_major_version_guard.py`` drives the guard step with a stub that
prints a fixed version. This file feeds python-semantic-release itself a
scratch repository and runs the release job's guard step text on what it
computes: a range carrying a breaking-change footer or a ``type!:`` subject
computes 3.0.0 under the stock parser (the case the no-major parser layer
exists for) and the step stops; ``feat`` and ``fix`` ranges compute 2.x and the
step passes. With the repository's own parser the same breaking range is
clamped to a minor. Skips where python-semantic-release is not installed (it
lives in the hash-locked release requirements, not the test extras).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

semantic_release = pytest.importorskip("semantic_release")

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = ("check_release_major.py", "no_major_parser.py")
_STOCK_PARSER = "conventional"
_REPO_PARSER = "scripts/no_major_parser.py:NoMajorAngularParser"
#: Assembled at runtime so this file itself never trips the commit-msg layer.
_BREAKING_FOOTER = "BREAKING" + " CHANGE: the old API is gone"
_BANG_SUBJECT = "feat" + "(api)!: drop the old API"


def _guard_run_text() -> str:
    workflow = yaml.safe_load((_ROOT / ".github" / "workflows" / "release.yml").read_text())
    steps = workflow["jobs"]["release"]["steps"]
    return next(s["run"] for s in steps if "scripts/check_release_major.py" in s.get("run", ""))


def _semantic_release_bin() -> Path:
    found = shutil.which("semantic-release", path=str(Path(sys.executable).parent))
    if found is None:
        pytest.skip("semantic_release imports but its console script is not beside the interpreter")
    return Path(found).parent


class _Repo:
    """A scratch repository released at 2.37.0 whose next commit is under test."""

    def __init__(self, root: Path, bin_dir: Path, parser: str) -> None:
        self.root = root
        self.env = {
            "PATH": os.pathsep.join(
                [str(bin_dir), str(_semantic_release_bin()), os.environ["PATH"]]
            ),
            "HOME": str(root),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        }
        (root / "scripts").mkdir(parents=True)
        for name in _SCRIPTS:
            shutil.copy(_ROOT / "scripts" / name, root / "scripts" / name)
        (root / "pyproject.toml").write_text(
            "[project]\n"
            'name = "torchlens"\n'
            'version = "2.37.0"\n\n'
            "[tool.semantic_release]\n"
            'version_toml = ["pyproject.toml:project.version"]\n'
            'branch = "main"\n'
            "major_on_zero = true\n"
            f'commit_parser = "{parser}"\n',
            encoding="utf-8",
        )
        self.git("init", "-q", "-b", "main")
        self.git("remote", "add", "origin", "https://github.com/example/example.git")
        self.git("add", "-A")
        self.git("commit", "-q", "--no-verify", "-m", "chore(release): 2.37.0")
        self.git("tag", "v2.37.0")

    def git(self, *args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=self.root, env=self.env, check=True, capture_output=True, text=True
        )

    def commit(self, message: str) -> None:
        (self.root / "change.txt").write_text(message, encoding="utf-8")
        self.git("add", "change.txt")
        self.git("commit", "-q", "--no-verify", "-m", message)

    def computed(self) -> str:
        return subprocess.run(
            ["semantic-release", "--noop", "version", "--print"],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def guard_step(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _guard_run_text()],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.fixture()
def bin_dir(tmp_path: Path) -> Path:
    path = tmp_path / "bin"
    path.mkdir()
    (path / "python").symlink_to(sys.executable)
    return path


@pytest.mark.parametrize(
    "message", [f"feat(api): drop the old API\n\n{_BREAKING_FOOTER}\n", _BANG_SUBJECT]
)
def test_breaking_range_computes_a_major_and_the_guard_step_stops(
    tmp_path: Path, bin_dir: Path, message: str
) -> None:
    repo = _Repo(tmp_path / "repo", bin_dir, _STOCK_PARSER)
    repo.commit(message)
    assert repo.computed() == "3.0.0"
    stopped = repo.guard_step()
    assert stopped.returncode == 1, stopped.stdout + stopped.stderr
    assert "REFUSING major bump" in stopped.stderr


@pytest.mark.parametrize(
    ("message", "expected"), [("feat: add a thing", "2.38.0"), ("fix: a thing", "2.37.1")]
)
def test_minor_and_patch_ranges_pass_the_guard_step(
    tmp_path: Path, bin_dir: Path, message: str, expected: str
) -> None:
    repo = _Repo(tmp_path / "repo", bin_dir, _STOCK_PARSER)
    repo.commit(message)
    assert repo.computed() == expected
    passed = repo.guard_step()
    assert passed.returncode == 0, passed.stdout + passed.stderr


def test_repository_parser_clamps_a_breaking_range_to_a_minor(
    tmp_path: Path, bin_dir: Path
) -> None:
    repo = _Repo(tmp_path / "repo", bin_dir, _REPO_PARSER)
    repo.commit(f"feat(api): drop the old API\n\n{_BREAKING_FOOTER}\n")
    assert repo.computed() == "2.38.0"
    assert repo.guard_step().returncode == 0
