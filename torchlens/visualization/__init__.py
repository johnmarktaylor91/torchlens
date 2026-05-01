"""Computational graph visualization via Graphviz.

The experimental Dagua renderer is available from
``torchlens.experimental.dagua`` after explicit opt-in.
"""

from typing import Any

from .node_spec import NodeSpec, render_lines_to_html

_USER_FUNC_EXPORTS = {"show_backward_graph", "show_bundle_graph", "show_model_graph", "summary"}


def __getattr__(name: str) -> Any:
    """Lazily expose user-facing visualization convenience functions.

    Parameters
    ----------
    name:
        Requested visualization attribute.

    Returns
    -------
    Any
        User-facing visualization helper from ``torchlens.user_funcs``.

    Raises
    ------
    AttributeError
        If ``name`` is not exported by this namespace.
    """

    if name in _USER_FUNC_EXPORTS:
        from .. import user_funcs

        return getattr(user_funcs, name)
    raise AttributeError(f"module 'torchlens.visualization' has no attribute {name!r}")


__all__ = [
    "NodeSpec",
    "render_lines_to_html",
    "show_backward_graph",
    "show_bundle_graph",
    "show_model_graph",
    "summary",
]
