"""Public docs pages carry no internal bookkeeping phrases.

The repository is public and GitHub renders every page under ``docs/``. Only ``docs/internal/``
is for maintainers; every other page is read by users and must not carry planning vocabulary,
placeholder comments or plan step references.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

#: Internal bookkeeping phrases (regular expressions, matched case-insensitively) that must
#: never appear on a public docs page.
BANNED_PHRASES: tuple[str, ...] = (
    r"deprecation shim",
    r"shim owed",
    r"documented-unstable",
    r"JMT writes this",
    r"\(step \d+\)",
    r"<!-- card:",
    r"<!-- Generated from",
)

_BANNED = re.compile("|".join(BANNED_PHRASES), re.IGNORECASE)


def _public_pages() -> list[Path]:
    """Return every Markdown page under ``docs/`` outside ``docs/internal/``."""

    return sorted(
        page for page in DOCS.rglob("*.md") if page.relative_to(DOCS).parts[0] != "internal"
    )


def test_public_docs_pages_carry_no_bookkeeping_phrases() -> None:
    """No public docs page matches a phrase in ``BANNED_PHRASES``."""

    pages = _public_pages()
    assert pages, "found no public docs pages"
    hits = [
        f"{page.relative_to(DOCS.parent).as_posix()}:{number}: {match.group(0)!r}"
        for page in pages
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), start=1)
        for match in _BANNED.finditer(line)
    ]
    assert not hits, "internal bookkeeping phrases on public docs pages:\n" + "\n".join(hits[:50])


def test_banned_phrases_catch_the_known_forms() -> None:
    """Each banned phrase matches the form it exists to catch, and ordinary prose passes."""

    caught = (
        "unstable -- no deprecation shim owed",
        "(spelling DOCUMENTED-UNSTABLE)",
        "<!-- JMT writes this: the pitch. -->",
        "the rename codemod (step 7)",
        "| Intervene | <!-- card: intervene --> |",
        "<!-- Generated from the task cards -->",
    )
    assert all(_BANNED.search(text) for text in caught)
    assert not _BANNED.search("These names are experimental and may change in any release.")
