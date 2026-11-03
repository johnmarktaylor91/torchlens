"""The ``EncodingChannelRequest`` carrier shared by ``_encoding`` and ``_color_sources``.

A leaf: ``_color_sources`` resolves the non-string ``color_by`` sources and
``_encoding`` compiles the channels; both need the request dataclass without
either importing the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ("EncodingChannelRequest",)


@dataclass(frozen=True)
class EncodingChannelRequest:
    """A color source paired with an explicit transform (lens layer door).

    The lens roster's performance rows pass this as ``color_by`` so the
    view-resolved member arrives WITH its rank transform; a bare string or
    callable ``color_by`` keeps the historical linear default.
    """

    source: Any
    transform: str = "linear"
    display_name: str | None = None
