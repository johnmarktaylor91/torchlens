"""Input and output marks when a colour scale owns the fill (``color_by``).

With no colour channel the input tensor is a pale green oval and the output a
salmon one, as released. Under ``color_by`` the fill channel belongs to the
data: a green fill beside a viridis ramp reads as a value on the scale, not
as "the input". So while a colour scale is active the two endpoint roles
move from the fill to a ring, the device the ``edited`` mark already uses:
the node keeps the page fill (or its scale colour when it carries a value)
and draws its border in the skin's input or output colour at
:data:`ENDPOINT_RING_PENWIDTH`. The key draws the same ring beside the words
``input`` and ``output``. ``size_by`` alone leaves the fills as released.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .themes import RELEASED_SEMANTIC_PALETTE

if TYPE_CHECKING:
    from .._vocab.node_spec import NodeSpec
    from ._header_block import KeySwatch
    from .themes import VisualizationTheme

__all__ = (
    "ENDPOINT_RING_PENWIDTH",
    "color_scale_active",
    "endpoint_ring_ink",
    "endpoint_ring_role",
    "endpoint_ring_spec",
    "endpoint_ring_swatch",
)

#: Ring width of an endpoint under a colour scale, in points (the edit ring's width).
ENDPOINT_RING_PENWIDTH = 2.5

_ENDPOINT_ROLES = ("input", "output")


def color_scale_active(encoding: Any | None) -> bool:
    """Return whether the draw's encoding state carries a colour channel."""

    return encoding is not None and getattr(encoding, "spec", None) is not None


def endpoint_ring_ink(role: str, theme: VisualizationTheme | None) -> str:
    """Return the ring colour for ``role`` (``"input"`` or ``"output"``) under ``theme``.

    The skin's own input or output colour, so a ring matches the fill the
    skin would otherwise paint; the released palette when no skin is given.
    """

    palette: Mapping[str, str] = (
        RELEASED_SEMANTIC_PALETTE if theme is None else theme.semantic_palette
    )
    return str(palette.get(role, RELEASED_SEMANTIC_PALETTE[role]))


def endpoint_ring_spec(spec: NodeSpec, role: str, theme: VisualizationTheme | None) -> NodeSpec:
    """Return ``spec`` with the role moved from its fill to a ring.

    The fill becomes ``"white"`` (the skin maps it to its page fill; a colour
    channel's value fill still replaces it afterwards) and the border takes
    the role colour at :data:`ENDPOINT_RING_PENWIDTH`. The label ink is reset
    so the skin recomputes a legible colour for the new fill.
    """

    return spec.replace(
        fillcolor="white",
        fontcolor="black",
        color=endpoint_ring_ink(role, theme),
        penwidth=ENDPOINT_RING_PENWIDTH,
    )


def endpoint_ring_role(attrs: Mapping[str, Any], theme: VisualizationTheme | None) -> str | None:
    """Return the endpoint role a rendered node's ring carries, or ``None``.

    A ring is a border in the skin's input or output colour at the ring
    width; a thin border in a role colour (a user's ``node_spec_fn``) is not
    one.
    """

    penwidth = attrs.get("penwidth")
    try:
        if penwidth is None or float(penwidth) != ENDPOINT_RING_PENWIDTH:
            return None
    except (TypeError, ValueError):
        return None
    border = str(attrs.get("color", "")).lower()
    for role in _ENDPOINT_ROLES:
        if border == endpoint_ring_ink(role, theme).lower():
            return role
    return None


def endpoint_ring_swatch(theme: VisualizationTheme, swatch: KeySwatch) -> KeySwatch:
    """Return an ``input`` or ``output`` key entry as the ring the picture draws.

    Any other entry comes back unchanged. The ring sits on the skin's page
    fill, like the ``edited`` entry.
    """

    if swatch.text not in _ENDPOINT_ROLES:
        return swatch
    return dataclasses.replace(
        swatch,
        shape="oval",
        fill=theme.default_fill,
        border=endpoint_ring_ink(swatch.text, theme),
        line=ENDPOINT_RING_PENWIDTH,
    )
