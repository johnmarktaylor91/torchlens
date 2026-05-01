"""semantic-release commit parser that NEVER produces a MAJOR bump.

Layer 3 of the version-bump prevention mechanism (after the commit-msg and
pre-push hooks in ``scripts/check_no_breaking_markers.py``). TorchLens stays
on the 2.x family per the locked policy in
``~/.claude/projects/-home-jtaylor-projects-torchlens/memory/feedback_version_bumps.md``.
The PyPI 1.0.0 and 2.0.0 slots have already been burned by accidental major
bumps; the 3.0.0 slot was nearly burned a third time on 2026-05-01 (rescued
only by an unrelated workflow bug).

This parser subclasses the stock Angular parser and downgrades any
``LevelBump.MAJOR`` result to ``LevelBump.MINOR``. The hooks make it
impossible for ``!`` markers or ``BREAKING CHANGE:`` footers to enter a
commit in the first place; this parser is the belt-and-suspenders backstop in
case a marker reaches main via a bypassed hook (``--no-verify``, server-side
direct push, or the ``TORCHLENS_ALLOW_MAJOR_BUMP`` override used without the
hooks installed).

To intentionally cut a major release, use:

    semantic-release version --force-level major

(with explicit JMT authorization in the same turn).

Pointed at via ``[tool.semantic_release] commit_parser`` in ``pyproject.toml``.
The ``scripts/`` directory is added to ``sys.path`` by ``conftest.py``-style
import-time machinery embedded in this module's loader path -- semantic-release
imports it via dotted name ``no_major_parser:NoMajorAngularParser`` from the
repo root, where ``scripts/`` is added to ``sys.path`` ahead of release.
"""

from __future__ import annotations

from semantic_release.commit_parser.angular import AngularCommitParser
from semantic_release.commit_parser.token import ParsedCommit, ParseResult
from semantic_release.enums import LevelBump


def _clamp_to_minor(result: ParseResult) -> ParseResult:
    if isinstance(result, ParsedCommit) and result.bump == LevelBump.MAJOR:
        return result._replace(bump=LevelBump.MINOR)
    return result


class NoMajorAngularParser(AngularCommitParser):
    """Angular parser variant that downgrades MAJOR bump signals to MINOR.

    See module docstring for rationale. This is mechanical defense-in-depth:
    even if a ``feat!:`` or ``BREAKING CHANGE:`` somehow reaches main, this
    parser refuses to interpret it as a major bump.
    """

    def parse(self, commit):  # type: ignore[override]
        result = super().parse(commit)
        if isinstance(result, list):
            return [_clamp_to_minor(r) for r in result]
        return _clamp_to_minor(result)
