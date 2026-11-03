"""Bounded candidate text for merged-rank lookup refusals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._utils._display_errors import lookup_message
from .._utils._display_keys import display_key


def _ambiguous_rank_message(item: str, hits: Mapping[int, Any]) -> str:
    """Keep rank qualification while using each resolved record's display key."""

    return lookup_message(
        f"Label {item!r} resolves on {len(hits)} ranks.",
        [f"r{rank}/{display_key(hits[rank])}" for rank in sorted(hits)],
        "use a rank-qualified label or merged.at() for the cross-rank fan",
    )
