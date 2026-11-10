"""Teaching refusal for keywords ``Trace.draw`` does not take.

``Trace.draw`` spells every drawing setting as an explicit keyword, so a wrong one used
to die with CPython's bare ``TypeError: unexpected keyword argument``. The 1.x drawing
keywords (``vis_outpath``, ``vis_nesting_depth``, ...) are the common case; each names its
replacement, and any other keyword gets the closest real spelling.
"""

from __future__ import annotations

import difflib
import functools
import inspect
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from .._errors import ArgumentTypeError

_F = TypeVar("_F", bound=Callable[..., Any])

#: 1.x drawing keyword -> the ``Trace.draw`` keyword that replaced it.
LEGACY_DRAW_KEYWORDS: dict[str, str] = {
    "vis_outpath": "path",
    "vis_fileformat": "file_format",
    "vis_save_only": "open_viewer",
    "vis_nesting_depth": "depth",
    "vis_call_depth": "depth",
    "vis_buffer_layers": "show_buffers",
    "vis_direction": "direction",
    "vis_graph_overrides": "graph_overrides",
    "vis_edge_overrides": "edge_overrides",
    "vis_grad_edge_overrides": "grad_edge_overrides",
    "vis_module_overrides": "module_overrides",
    "vis_node_placement": "layout",
    "vis_renderer": "renderer",
    "vis_theme": "theme",
    "vis_node_mode": "node_style",
    "vis_intervention_mode": "intervention_mode",
    "vis_show_cone": "show_cone",
    "vis_mode": "rolled",
}

#: Remedy notes for legacy keywords whose replacement takes a different value.
_LEGACY_VALUE_NOTES: dict[str, str] = {
    "vis_save_only": " (inverted: vis_save_only=True is open_viewer=False, the default)",
    "vis_mode": " (a bool: vis_mode='rolled' is rolled=True)",
}

VIS_OPT_REMEDY = (
    "vis_opt is gone: capture first, then draw the trace, tl.trace(model, x).draw(), "
    "or draw in one call with tl.draw(model, x)"
)


def draw_keyword_remedy(key: str, accepted: Iterable[str], call: str = "trace.draw") -> str:
    """Return the remedy for one keyword the draw call does not take.

    Parameters
    ----------
    key:
        The rejected keyword.
    accepted:
        Keywords the draw call takes (searched for a close spelling).
    call:
        How the remedy spells the draw call.

    Returns
    -------
    str
        A one-clause remedy naming the working spelling.
    """

    if key == "vis_opt":
        return VIS_OPT_REMEDY
    if key in LEGACY_DRAW_KEYWORDS:
        new = LEGACY_DRAW_KEYWORDS[key]
        return f"{key} is now {new}=: {call}({new}=...){_LEGACY_VALUE_NOTES.get(key, '')}"
    close = difflib.get_close_matches(key, sorted(accepted), n=1, cutoff=0.7)
    if close:
        return f"{key}: did you mean {call}({close[0]}=...)?"
    return f"{key}: see help(tl.Trace.draw) for the keywords it takes"


def teach_unknown_draw_keywords(method: _F) -> _F:
    """Wrap ``Trace.draw`` so an unknown keyword raises a teaching ``ArgumentTypeError``.

    The signature is unchanged (``functools.wraps`` keeps ``inspect.signature``); a call
    with only known keywords costs one set difference.
    """

    accepted = frozenset(inspect.signature(method).parameters) - {"self"}

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Refuse unknown keywords with a teaching error, else call ``Trace.draw``."""
        unknown = sorted(kwargs.keys() - accepted)
        if unknown:
            raise ArgumentTypeError(
                f"Trace.draw() got keyword(s) it does not take: {', '.join(unknown)}",
                code="draw_keyword_unknown",
                remedy="; ".join(draw_keyword_remedy(key, accepted) for key in unknown),
                unknown_keywords=unknown,
            )
        return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]
