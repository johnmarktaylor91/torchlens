"""Consistent, bounded candidate rows for lookup error messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._display_card import card_row
from ._display_keys import candidates_text, display_key


def candidate_keys(candidates: Sequence[str], records: Mapping[str, Any]) -> list[str]:
    """Resolve suggestions through stored records; do not parse positional suffixes."""

    return list(
        dict.fromkeys(display_key(records[key]) if key in records else key for key in candidates)
    )


def lookup_message(problem: str, candidates: Sequence[str], remedy: str) -> str:
    """Keep the problem first, followed by at most five candidates and the remedy."""

    lines = [problem]
    if candidates:
        lines.append(card_row("candidates", candidates_text(candidates)))
    lines.append(card_row("remedy", remedy))
    return "\n".join(lines)
