"""Pin the three incident-hardened semantic-release config invariants (R86).

Each of these lines in ``pyproject.toml`` exists because a specific release
incident burned the project, and until fixwave-7 none had a test — a silent
edit (or a "cleanup" that reverts one to its library default) would only
fail AT RELEASE TIME, in the job that pushes tags:

1. ``commit_message`` must carry the literal ``[skip ci]``: the bot pushes
   the release commit to main, and without the skip token GitHub re-triggers
   the Release workflow on that push, which re-releases and pushes again —
   the 1,599-commit runaway loop already happened once.
2. ``[tool.semantic_release.changelog] template_dir`` must stay explicit and
   the capped release-notes template must exist at the exact dotfile name
   PSR discovers (``templates/.release_notes.md.j2``): under the builtin
   uncapped template a large release exceeds GitHub's 125K release-body
   limit and the create-release API 422s, blocking PyPI (the 2.34.0
   incident).
3. ``major_on_zero`` must stay ``true``: flipping it makes PSR silently
   downgrade breaking-change bumps on 0.x versions, splitting version
   semantics from the three-layer never-ship-a-major defense the config
   documents (layers 1-3 assume standard bump semantics as their input).

Parsing is line/regex-based on purpose: ``tomllib`` only exists on 3.11+
and the suite's floor row runs 3.10.
"""

from __future__ import annotations

import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_text() -> str:
    """Return the pyproject.toml text."""

    return (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_release_commit_message_carries_the_skip_ci_literal() -> None:
    """The release commit message keeps the loop-breaking ``[skip ci]`` token."""

    match = re.search(r'^commit_message\s*=\s*"([^"]*)"', _pyproject_text(), re.MULTILINE)
    assert match, "pyproject.toml lost [tool.semantic_release] commit_message"
    assert "[skip ci]" in match.group(1), (
        "the release commit_message must contain the literal '[skip ci]' — "
        "without it the bot's push re-triggers the Release workflow and the "
        "release loops (see the 1,599-commit runaway in git history)"
    )


def test_changelog_template_dir_is_explicit_and_the_capped_template_exists() -> None:
    """``template_dir`` stays pinned and the capped dotfile template is present."""

    match = re.search(r'^template_dir\s*=\s*"([^"]*)"', _pyproject_text(), re.MULTILINE)
    assert match, (
        "pyproject.toml lost the explicit [tool.semantic_release.changelog] "
        "template_dir pin — under the implicit default a PSR default change "
        "or a templates/ rename silently reverts release notes to the "
        "uncapped builtin template (the 2.34.0 422-blocks-PyPI incident)"
    )
    template_dir = _PROJECT_ROOT / match.group(1)
    template = template_dir / ".release_notes.md.j2"
    assert template.is_file(), (
        f"{template} is missing: PSR discovers the capped release-notes "
        "template by this exact dotfile name; without it the uncapped "
        "builtin renders and a large release 422s at create-release"
    )
    # The template must remain the ONLY file there: any sibling *.j2 would
    # silently override further PSR outputs (e.g. a CHANGELOG.md.j2 replaces
    # the bundled changelog template), widening this pin's blast radius.
    extras = sorted(p.name for p in template_dir.iterdir() if p.name != ".release_notes.md.j2")
    assert not extras, (
        f"unexpected files under {template_dir}: {extras} — templates/ is "
        "contractually the sole capped release-notes override (see the "
        "template's own header comment); additions change PSR output "
        "discovery and need an explicit review"
    )
    assert "125,000" in template.read_text(encoding="utf-8") or "100,000" in template.read_text(
        encoding="utf-8"
    ), "the capped template lost its budget documentation/enforcement markers"


def test_major_on_zero_stays_true() -> None:
    """``major_on_zero`` keeps standard bump semantics under the major defense."""

    match = re.search(r"^major_on_zero\s*=\s*(\w+)", _pyproject_text(), re.MULTILINE)
    assert match, "pyproject.toml lost [tool.semantic_release] major_on_zero"
    assert match.group(1) == "true", (
        "major_on_zero must stay true: the three-layer never-ship-a-major "
        "defense (markers hook, arming test, NoMajorAngularParser) assumes "
        "standard bump semantics as its input; flipping this splits version "
        "semantics from the defense instead of strengthening it"
    )
