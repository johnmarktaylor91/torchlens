"""Package tripwire: no walk or tracker id in ``torchlens/`` text.

The maintainers' design walks name their rulings with private ids, and the
work tracker names its items the same way. Those ids mean nothing to a user
of this public package, so they never appear in a docstring, an error
message, a warning or any other string literal under ``torchlens/``. Comments
may keep an id only where it points to a public document; none of the id
families below is one, so comments are scanned too.

The refused patterns (``_ID_PATTERNS``) are:

- ``U12.3``: a walk unit (``U`` + 1-3 digits + ``.`` + digits);
- ``R2-08``: a second-round walk row (``R2-`` + two digits);
- ``AGT-2360``: a tracker item (``AGT-`` + digits);
- ``forks.11``, ``neuro.1``, ``names.2``, ``scope.3``, ``errors.3``,
  ``defaults.7``, ``naming.2``: a walk topic and its ruling number.

Text such as version numbers (``torch 2.13``), ``U-Net`` or a module name
like ``neuro.py`` does not match.
"""

from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent.parent / "torchlens"

#: Each refused id family, with an example it matches.
_ID_PATTERNS: dict[str, str] = {
    "walk unit (U12.3)": r"\bU[0-9]{1,3}\.[0-9]+\b",
    "second-round walk row (R2-08)": r"\bR2-[0-9]{2}\b",
    "tracker item (AGT-2360)": r"\bAGT-[0-9]+\b",
    "walk topic ruling (forks.11)": (
        r"\b(?:forks|neuro|names|scope|errors|defaults|naming)\.[0-9]+\b"
    ),
}
_ID_RE = re.compile("|".join(f"(?:{pattern})" for pattern in _ID_PATTERNS.values()))

#: Token kinds whose text is scanned: comments and every string literal
#: (docstrings, messages, f-string text on Python 3.12+).
_TEXT_TOKENS = frozenset(
    kind
    for kind in (
        tokenize.COMMENT,
        tokenize.STRING,
        getattr(tokenize, "FSTRING_MIDDLE", None),
    )
    if kind is not None
)


def _ids_in_source(source: str) -> list[tuple[int, str]]:
    """Return ``(line, id)`` for every id in the comments and strings of ``source``."""

    if _ID_RE.search(source) is None:
        return []
    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in _TEXT_TOKENS:
            continue
        for offset, line in enumerate(token.string.splitlines()):
            found.extend((token.start[0] + offset, m.group(0)) for m in _ID_RE.finditer(line))
    return found


def test_no_walk_or_tracker_id_in_package_text() -> None:
    """No docstring, string or comment under ``torchlens/`` carries a private id."""

    offenders = [
        f"{path.relative_to(_PACKAGE.parent)}:{line}: {found}"
        for path in sorted(_PACKAGE.rglob("*.py"))
        for line, found in _ids_in_source(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "private walk/tracker ids in public package text; describe the behaviour "
        "instead of citing the id:\n" + "\n".join(offenders)
    )


def test_lint_is_red_capable() -> None:
    """Every family is caught in a docstring, a message and a comment; lookalikes pass."""

    source = '''
def f():
    """Restore attributes (U40.1)."""
    raise ValueError(f"bad {x} (R2-08)")  # AGT-2360: tracked
    warn("see forks.11 and neuro.1 row 4")
'''
    found = [found for _, found in _ids_in_source(source)]
    assert found == ["U40.1", "R2-08", "AGT-2360", "forks.11", "neuro.1"]
    clean = '''
def g():
    """torch 2.13 and U-Net; see neuro.py, R2 and AGT."""
    return "v2.37.1"  # names.py
'''
    assert _ids_in_source(clean) == []
