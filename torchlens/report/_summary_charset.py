"""The summary charset contract (F08; summary memo item 14, 3.9).

``str(result)`` is canonical byte-stable ASCII with no ANSI or OSC-8
bytes -- the equality, logging, snapshot, and issue-paste contract.
Unicode appears only at explicit display boundaries. The two renders
differ ONLY through the declared glyph table below, one unicode character
to one ASCII character, so ``degrade(unicode_render) == ascii_render``
holds byte-for-byte (machine-checked in CI) and column alignment is
identical in both charsets: glyph weight, not design.

Detection ladder (fail-toward-ASCII; the only path to unicode is an
affirmatively interactive, verified terminal):

1. explicit argument (handled by the caller)
2. ``TORCHLENS_SUMMARY_STYLE`` environment override
3. ``CI`` / ``GITHUB_ACTIONS`` set -> ASCII, even on a pty
4. isatty AND TERM != "dumb" AND the stream encoding round-trips the
   exact glyph table -> unicode
5. otherwise ASCII

Spellings DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

import os
from typing import Any

#: The declared glyph table: unicode character -> ASCII character.
#: STRICTLY one-to-one so the degrade map preserves widths and alignment.
GLYPH_TABLE: dict[str, str] = {
    "─": "-",  # horizontal rule
    "×": "x",  # fold multiplier sign
    "█": "#",  # micro-bar filled cell
    "░": ".",  # micro-bar empty cell
}

_DEGRADE_TABLE = str.maketrans(GLYPH_TABLE)


def degrade(unicode_text: str) -> str:
    """Map a unicode render to its ASCII form through the glyph table."""

    return unicode_text.translate(_DEGRADE_TABLE)


def _stream_verifies_glyphs(stream: Any) -> bool:
    """True when the stream encoding round-trips the EXACT glyph table.

    Strictly stronger than "the encoding is named utf-8": a misdeclared
    codec fails the round-trip and falls toward ASCII.
    """

    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    probe = "".join(GLYPH_TABLE)
    try:
        return probe.encode(encoding).decode(encoding) == probe
    except (UnicodeError, LookupError):
        return False


def detect_style(stream: Any = None) -> str:
    """Resolve ``style="auto"`` at one display boundary (never in __str__).

    Every failed or uncertain check falls toward ASCII; detection can be
    wrong only in the harmless direction.
    """

    override = os.environ.get("TORCHLENS_SUMMARY_STYLE", "").strip().lower()
    if override in ("ascii", "unicode"):
        return override
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return "ascii"
    if stream is None:
        import sys

        stream = sys.stdout
    try:
        interactive = bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        interactive = False
    if not interactive:
        return "ascii"
    if os.environ.get("TERM", "") == "dumb":
        return "ascii"
    if not _stream_verifies_glyphs(stream):
        return "ascii"
    return "unicode"


def assert_no_escape_bytes(text: str) -> str:
    """A11 guard: no ESC byte in any returned string, ever."""

    if "\x1b" in text:
        raise AssertionError("summary render produced an ESC byte (A11 violation)")
    return text
