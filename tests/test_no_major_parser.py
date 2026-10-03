"""Behavioral tests for layer 3 of the never-ship-a-major defense.

``scripts/no_major_parser.py`` is the ONLY major-bump layer that sees a
squash-merge title/body composed in the GitHub UI (layers 1/2 are local
hooks), and until grind r5 it had ZERO tests anywhere (b10 R86-2): it
depends on three PSR-9.x private-ish API surfaces (``AngularCommitParser``,
``ParsedCommit._replace``, ``LevelBump``) and previously failed only AT
RELEASE TIME, in the job that pushes tags. These tests execute the parser
under the pinned python-semantic-release from the release lock; the lint
workflow's release-defenses job installs exactly that environment, so a PSR
bump that renames any surface now fails the PR that bumps the lock instead
of the release.

Locally the tests skip when PSR is absent (it is deliberately in no dev
extra); both importorskip targets are ledgered release-environment-only.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

semantic_release = pytest.importorskip("semantic_release")
git = pytest.importorskip("git")

from semantic_release.enums import LevelBump  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PARSER_PATH = _PROJECT_ROOT / "scripts" / "no_major_parser.py"


def _load_parser_class():
    """Load NoMajorAngularParser exactly the way PSR's file-path spec does."""

    spec = importlib.util.spec_from_file_location("no_major_parser", _PARSER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NoMajorAngularParser


@pytest.fixture()
def commit_factory(tmp_path: Path):
    """Return a factory minting real git commits for the parser."""

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    env_args = [
        "-c",
        "user.name=probe",
        "-c",
        "user.email=probe@example.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(["git", "init", "-q", str(repo_dir)], check=True)

    def make(message: str):
        (repo_dir / "f.txt").write_text(message)
        subprocess.run(["git", "-C", str(repo_dir), "add", "f.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), *env_args, "commit", "-q", "--no-verify", "-m", message],
            check=True,
        )
        return git.Repo(str(repo_dir)).head.commit

    return make


def _bumps(result) -> list[LevelBump]:
    parsed = result if isinstance(result, list) else [result]
    return [entry.bump for entry in parsed if hasattr(entry, "bump")]


def test_bang_marker_is_clamped_to_minor(commit_factory) -> None:
    """A feat! subject parses as MINOR, never MAJOR."""

    parser = _load_parser_class()()
    result = parser.parse(commit_factory("feat" + "!: breaking spelling"))
    bumps = _bumps(result)
    assert bumps, "parser returned no parsed commit"
    assert LevelBump.MAJOR not in bumps
    assert LevelBump.MINOR in bumps


def test_breaking_change_footer_is_clamped_to_minor(commit_factory) -> None:
    """A BREAKING CHANGE footer parses as MINOR, never MAJOR."""

    parser = _load_parser_class()()
    footer = "BREAKING" + " CHANGE: removes the old API"
    result = parser.parse(commit_factory(f"feat: something\n\n{footer}"))
    bumps = _bumps(result)
    assert bumps, "parser returned no parsed commit"
    assert LevelBump.MAJOR not in bumps
    assert LevelBump.MINOR in bumps


def test_ordinary_bumps_pass_through(commit_factory) -> None:
    """fix stays PATCH and feat stays MINOR — the clamp touches only MAJOR."""

    parser = _load_parser_class()()
    assert LevelBump.PATCH in _bumps(parser.parse(commit_factory("fix: routine repair")))
    assert LevelBump.MINOR in _bumps(parser.parse(commit_factory("feat: routine feature")))


def test_pyproject_names_this_parser() -> None:
    """The PSR config points at exactly this file and class (discovery pin)."""

    pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text()
    assert 'commit_parser = "scripts/no_major_parser.py:NoMajorAngularParser"' in pyproject
