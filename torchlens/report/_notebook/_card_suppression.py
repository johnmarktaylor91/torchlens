"""Card-suppression slot: a renderer bridge claims a card render without the card importing it.

Notebook cards live in the report layer, below the third-party renderer bridges, so a
card never imports a bridge. A bridge that renders TorchLens objects itself (the
treescope bridge) REGISTERS its one-shot suppression predicate here when it loads; a
card asks :func:`card_render_suppressed` whether to return the one-line sentinel
instead of the full card. With nothing registered no bridge has rendered anything, so
the full card always renders.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["card_render_suppressed", "register_card_suppressor"]

CardSuppressor = Callable[[Any], bool]

_SUPPRESSOR: CardSuppressor | None = None


def register_card_suppressor(suppressor: CardSuppressor) -> None:
    """Install ``suppressor`` as the predicate cards consult before rendering.

    Parameters
    ----------
    suppressor:
        ``(obj) -> bool``; True when the bridge just rendered ``obj`` and the card
        should return the sentinel. The last registration wins.
    """

    global _SUPPRESSOR
    _SUPPRESSOR = suppressor


def card_render_suppressed(obj: Any) -> bool:
    """Return whether a registered bridge claims the render of ``obj``.

    Fails SAFE: with no predicate registered, or when the predicate raises, the
    answer is ``False`` and the full card renders; a broken sniff may never
    suppress the user's only good rendering.

    Parameters
    ----------
    obj:
        The object whose ``_repr_html_`` card is about to render.

    Returns
    -------
    bool
        True only when a registered predicate answers True for ``obj``.
    """

    suppressor = _SUPPRESSOR
    if suppressor is None:
        return False
    try:
        return bool(suppressor(obj))
    except Exception:  # noqa: BLE001 - fail-safe is the contract (memo 3.6)
        return False
