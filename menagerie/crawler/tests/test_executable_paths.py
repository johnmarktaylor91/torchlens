"""Configured executables must keep their virtualenv, not collapse into the base install.

This regression exists because the same defect shipped twice in independent code paths and
each time surfaced as something else entirely:

* the strict doctor reported ``missing version receipts ['checker']`` for a correctly
  configured wrapper;
* the driver died mid-run with ``ModuleNotFoundError: No module named 'jsonschema'`` raised
  from inside the checker subprocess.

Both were ``Path.resolve()`` following a virtualenv's ``bin/python`` symlink chain down to
the base interpreter. CPython finds ``pyvenv.cfg`` next to the path it was *invoked by*, so
resolving the symlink silently swaps the environment and its site-packages.
"""

from __future__ import annotations

import os
from pathlib import Path

from menagerie.crawler.executable_paths import normalize_executable


def _fake_venv(root: Path) -> tuple[Path, Path]:
    """Build a venv-shaped tree whose bin/python symlinks to a 'base' interpreter."""

    base_bin = root / "base" / "bin"
    base_bin.mkdir(parents=True)
    base_python = base_bin / "python3"
    base_python.write_text("#!/bin/sh\nexit 0\n")
    base_python.chmod(0o755)

    venv_bin = root / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (root / "venv" / "pyvenv.cfg").write_text("home = /base\n")
    venv_python = venv_bin / "python"
    venv_python.symlink_to(base_python)
    return venv_python, base_python


def test_absolute_venv_interpreter_is_not_followed_to_its_base(tmp_path: Path) -> None:
    """The configured venv path is returned verbatim, keeping pyvenv.cfg adjacent."""

    venv_python, base_python = _fake_venv(tmp_path)

    resolved = normalize_executable(str(venv_python))

    assert resolved == venv_python
    assert resolved != base_python
    # The decisive property: pyvenv.cfg must sit next to what we hand to subprocess.run,
    # because that is how CPython decides which environment it is running in.
    assert (resolved.parent.parent / "pyvenv.cfg").is_file()


def test_symlinked_venv_directory_still_keeps_its_own_config(tmp_path: Path) -> None:
    """A campaign clone symlinking the whole venv directory must still stay in the venv.

    This is the exact shape of the live failure: the pilot clone's ``.venv-crawler`` was a
    symlink to another checkout's venv, so the interpreter was reachable through two
    symlink hops and ``resolve()`` flattened both.
    """

    venv_python, base_python = _fake_venv(tmp_path)
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / ".venv").symlink_to(tmp_path / "venv")

    resolved = normalize_executable(str(clone / ".venv" / "bin" / "python"))

    assert resolved is not None
    assert resolved != base_python
    assert (resolved.parent.parent / "pyvenv.cfg").is_file()


def test_relative_token_with_parent_resolves_against_cwd_without_following(tmp_path: Path) -> None:
    """A relative command keeps the same guarantee."""

    venv_python, base_python = _fake_venv(tmp_path)

    resolved = normalize_executable("venv/bin/python", cwd=tmp_path)

    assert resolved == venv_python
    assert resolved != base_python


def test_dot_segments_are_normalized(tmp_path: Path) -> None:
    """`.` and `..` are removed even though symlinks are preserved."""

    venv_python, _ = _fake_venv(tmp_path)
    noisy = tmp_path / "venv" / "bin" / ".." / "bin" / "python"

    assert normalize_executable(str(noisy)) == venv_python


def test_missing_executable_is_none(tmp_path: Path) -> None:
    """An unresolvable token is reported, not fabricated."""

    assert normalize_executable(str(tmp_path / "nope" / "python")) is None
    assert normalize_executable("almost-certainly-not-a-real-binary-xyzzy") is None


def test_directory_is_not_accepted_as_an_executable(tmp_path: Path) -> None:
    """Only files count; a directory that happens to exist must not pass."""

    (tmp_path / "bin").mkdir()
    assert normalize_executable(str(tmp_path / "bin")) is None


def test_bare_name_is_looked_up_on_path(tmp_path: Path) -> None:
    """A bare name still goes through PATH lookup and comes back absolute."""

    resolved = normalize_executable("sh")

    assert resolved is not None
    assert resolved.is_absolute()
    assert os.access(resolved, os.X_OK)
