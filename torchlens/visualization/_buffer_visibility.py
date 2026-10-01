"""Buffer-node visibility predicates shared across the render pipeline.

Split out of ``_render_edges.py`` under the R43 file-size ratchet
(FIXD03-F13). The tri-state ``show_buffer_layers`` semantics live here;
``_render_edges`` re-exports the names for its historical importers.
"""

from ._render_common import _NOISE_BUFFER_NAMES, BufferVisibilityLiteral, GraphNode
from ._render_leaf import _unwrap_focus_node

__all__ = [
    "_buffer_name_segment",
    "_is_buffer_visible",
    "_is_noise_buffer",
]


def _buffer_name_segment(address: str | None) -> str:
    """Return the last dotted segment of a buffer address.

    Parameters
    ----------
    address:
        Fully qualified buffer address, if available.

    Returns
    -------
    str
        Final dotted address segment, or an empty string for missing addresses.
    """

    if address is None:
        return ""
    return address.split(".")[-1]


def _is_noise_buffer(node: GraphNode) -> bool:
    """Return whether ``node`` is a hardcoded noisy buffer.

    Parameters
    ----------
    node:
        Candidate graph node.

    Returns
    -------
    bool
        True when the node is a buffer whose last address segment is filtered in
        ``"meaningful"`` mode.
    """

    source_node = _unwrap_focus_node(node)
    if not source_node.is_buffer:
        return False
    address = getattr(source_node, "address", None)
    return _buffer_name_segment(address) in _NOISE_BUFFER_NAMES


def _is_buffer_visible(node: GraphNode, show_buffer_layers: BufferVisibilityLiteral) -> bool:
    """Return whether a buffer node should be visible in the current mode.

    Parameters
    ----------
    node:
        Candidate graph node.
    show_buffer_layers:
        Canonical tri-state visibility mode.

    Returns
    -------
    bool
        True when the node is visible. Non-buffer nodes are always visible.
    """

    if not node.is_buffer:
        return True
    if show_buffer_layers == "always":
        return True
    if show_buffer_layers == "never":
        return False
    return not _is_noise_buffer(node)
