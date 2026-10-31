"""semantic-release commit parser that NEVER produces a MAJOR bump.

Layer 3 of the version-bump prevention mechanism (after the commit-msg and
pre-push hooks in ``scripts/check_no_breaking_markers.py``). Major-version
bumps happen only on the maintainer's explicit command; the PyPI 1.0.0 and
2.0.0 slots have already been burned by accidental major bumps. The release
job's last check, ``scripts/check_release_major.py``, refuses any computed
major bump that ``.github/MAJOR_RELEASE_APPROVED`` does not name.

This parser subclasses the stock Angular parser and downgrades any
``LevelBump.MAJOR`` result to ``LevelBump.MINOR``. The hooks make it
impossible for ``!`` markers or ``BREAKING CHANGE:`` footers to enter a
commit in the first place; this parser is the belt-and-suspenders backstop in
case a marker reaches main via a bypassed hook (``--no-verify``, server-side
direct push, or a marker committed while the approval file existed).

To intentionally cut a major release, use:

    semantic-release version --force-level major

(only on the maintainer's explicit command, with
``.github/MAJOR_RELEASE_APPROVED`` naming the target version).

Discovery (grind r4 correction -- the old paragraph here described sys.path
"conftest-style machinery" that has never existed): ``[tool.semantic_release]
commit_parser`` in ``pyproject.toml`` names this class with python-semantic-
release's FILE-PATH spec, ``scripts/no_major_parser.py:NoMajorAngularParser``.
PSR loads the class directly from that file at release time; no sys.path
manipulation is involved, and nothing imports this module outside the release
run. It depends on three PSR-9.x API surfaces (AngularCommitParser,
ParsedCommit/ParseResult, LevelBump) and fails AT RELEASE TIME if a PSR bump
renames them -- which is why release.yml pins PSR exactly.
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
