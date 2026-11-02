"""Public option members removed with no alias, and the hooks that name their replacement.

A removed field of an options class refuses the same way whether it arrives
as a constructor keyword or as an attribute read: the typed ``facade_redirect``
``FacadeTeachingError`` from ``torchlens._utils.facade.refuse_removed_member``
names the replacement. The tables here are the single place each remedy is
spelled; ``options.py`` applies them, and the ``trace()`` door reads the
capture table so a removed option is never mistaken for a model input.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar, cast

_F = TypeVar("_F", bound=Callable[..., Any])

#: Removed public members by owning options class: name -> remedy naming the replacement.
REMOVED_OPTION_MEMBERS: dict[str, dict[str, str]] = {}

#: Capture fields removed with no alias: the keyword or attribute names its replacement.
_REMOVED_CAPTURE_OPTION_MEMBERS: dict[str, str] = REMOVED_OPTION_MEMBERS.setdefault(
    "CaptureOptions", {}
)
_REMOVED_CAPTURE_OPTION_MEMBERS.update(
    {
        "output_style": (
            "decode at read time with trace.decode_output(style=...) or "
            "trace.output_table(style=...) (hf_text takes tokenizer=) -- output decoding "
            "is no longer a capture option"
        ),
        "output_head": (
            "decode at read time with trace.decode_output(head=...) or "
            "trace.output_table(head=...) -- output decoding is no longer a capture option"
        ),
        "layers_to_save": (
            "select what to save with tl.trace(model, x, save=...) -- the selection is not a "
            "capture option"
        ),
        "save_preview": (
            "select what to save with tl.trace(model, x, save=...) -- the selection is not a "
            "capture option"
        ),
    }
)

#: Alias properties removed by the 2026-10-01 shim removal (no alias kept).
_REMOVED_VISUALIZATION_OPTION_MEMBERS: dict[str, str] = REMOVED_OPTION_MEMBERS.setdefault(
    "VisualizationOptions", {}
)
_REMOVED_VISUALIZATION_OPTION_MEMBERS.update(
    {
        "mode": "use VisualizationOptions.rolled (a bool) -- the mode alias was removed",
        "view": (
            "use VisualizationOptions.rolled (True for the rolled graph); a picture is drawn "
            "whenever visualization options are passed"
        ),
        "save_only": (
            "use VisualizationOptions.open_viewer (True opens the viewer; the default never does)"
        ),
        "for_paper": 'use VisualizationOptions.theme="paper"',
        "return_graph": "read Drawing.dot on the returned Drawing",
        "max_module_depth": "use VisualizationOptions.depth -- the max_module_depth alias was removed",
        "layout_engine": "use VisualizationOptions.layout -- the layout_engine alias was removed",
        "node_mode": "use VisualizationOptions.node_style -- the node_mode alias was removed",
        "animate": "use trace.animate(site) -- the animation is its own door, not a draw keyword",
    }
)


def _refuse_removed_member_of(owner: str, name: str) -> None:
    """Raise the typed redirect when ``name`` is a removed member of ``owner``; else return."""

    from ._utils.facade import refuse_removed_member

    refuse_removed_member(owner, name, REMOVED_OPTION_MEMBERS[owner])


def _refuse_removed_keywords(owner: str) -> Callable[[_F], _F]:
    """Wrap an ``__init__`` so a removed field passed as a keyword names its replacement.

    ``functools.wraps`` keeps the explicit signature visible to ``inspect``
    and the surface tools; any other unknown keyword still fails the
    ordinary ``TypeError`` way inside the wrapped constructor.
    """

    def decorate(init: _F) -> _F:
        @functools.wraps(init)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            for name in kwargs:
                _refuse_removed_member_of(owner, name)
            return init(self, *args, **kwargs)

        return cast(_F, wrapper)

    return decorate
