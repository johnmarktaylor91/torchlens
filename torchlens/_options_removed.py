"""Public option members removed with no alias, with the remedy each one names.

A removed field of an options class refuses the same way whether it arrives
as a constructor keyword or as an attribute read: the typed ``facade_redirect``
``FacadeTeachingError`` from ``torchlens._utils.facade`` names the replacement.
The tables here are the single place each remedy is spelled; ``options.py``
applies them through the facade hooks, and the ``trace()`` door reads the
capture table so a removed option is never mistaken for a model input. This
module imports nothing from the package so the tables stay a leaf.
"""

from __future__ import annotations

#: Capture fields removed with no alias: the keyword or attribute names its replacement.
REMOVED_CAPTURE_MEMBERS: dict[str, str] = {
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
    "module_filter": (
        "select the modules to keep with tl.trace(model, x, save=tl.in_module(...)) or a "
        "save= predicate -- save= is the one site selection"
    ),
    "raise_on_nan": (
        'use CaptureOptions(on_nonfinite="raise") -- the non-finite policy is one field'
    ),
    "track_nonfinite": (
        'use CaptureOptions(on_nonfinite="track") (or "warn") -- the non-finite policy is one field'
    ),
}

#: Alias properties removed by the 2026-10-01 shim removal (no alias kept).
REMOVED_VISUALIZATION_MEMBERS: dict[str, str] = {
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
