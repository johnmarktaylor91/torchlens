"""Computational graph visualization via Graphviz.

The experimental Dagua renderer is available from
``torchlens.experimental.dagua`` after explicit opt-in.
"""

from typing import Any

from .node_spec import NodeSpec, render_lines_to_html

_USER_FUNC_EXPORTS = {
    "draw_backward",
    "draw_combined",
    "show_bundle_graph",
    "show_model_graph",
    "summary",
}

#: Surgery-visuals doors (lane F43; DOCUMENTED-UNSTABLE spellings): lazy so
#: ``import torchlens.visualization`` never pays for the audit derivation.
_SURGERY_EXPORTS = {
    "render_surgery": "surgery_visuals",
    "surgery_census": "surgery_visuals",
    "surgery_facts": "surgery_visuals",
    "surgery_diff": "surgery_diff",
}


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
    if name == "lenses":
        # Documented subpackage (docs/reference/lenses.md): resolve the
        # attribute lazily so a bare `torchlens.visualization.lenses` read
        # works without an eager import. Before this row the attribute
        # existed only after SOMETHING ELSE imported the subpackage -- the
        # docs import-resolution gate passed or failed with collection
        # order (the import-laziness-deletes-read-names class).
        import importlib

        return importlib.import_module(".lenses", __name__)
    if name in _SURGERY_EXPORTS:
        import importlib

        module = importlib.import_module(f".{_SURGERY_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module 'torchlens.visualization' has no attribute {name!r}")


__all__ = [
    "NodeSpec",
    "render_lines_to_html",
    "draw_backward",
    "draw_combined",
    "render_surgery",
    "show_bundle_graph",
    "show_model_graph",
    "summary",
    "surgery_census",
    "surgery_diff",
    "surgery_facts",
]
