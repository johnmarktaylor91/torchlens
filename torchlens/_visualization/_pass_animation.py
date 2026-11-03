"""The ``trace.animate(site)`` picture: an HTML animation of the repeated ops at one site."""

from __future__ import annotations

from html import escape
from typing import Any

from .._utils._record_group_table import record_member


def pass_animation_html(trace: Any, site: Any) -> str:
    """Return a self-contained HTML animation of the passes at ``site``.

    Parameters
    ----------
    trace:
        The trace holding the site.
    site:
        Layer label, pass-qualified layer label, or a record with a
        ``layer_label`` attribute.

    Raises
    ------
    KeyError
        If the requested site cannot be resolved.
    """

    label = str(getattr(site, "layer_label", site))
    base_label = label.split(":", 1)[0]
    if base_label not in trace.layers:
        raise KeyError(f"Unknown layer site {label!r}.")
    layer = trace.layers[base_label]
    pass_entries = list(getattr(layer, "ops", ()) or [])
    if not pass_entries:
        pass_entries = [trace[label]]
    frames = [
        {
            "pass": int(getattr(entry, "pass_index", index + 1) or index + 1),
            "label": str(getattr(entry, "layer_label", base_label)),
            "shape": "x".join(str(dim) for dim in getattr(entry, "shape", ()) or ()),
            "memory": str(record_member(entry, "activation_memory", "")),
        }
        for index, entry in enumerate(pass_entries)
    ]
    frame_markup = "".join(
        "<li data-frame='{idx}'>{label} pass {call_index}: {shape} {memory}</li>".format(
            idx=index,
            label=escape(str(frame["label"])),
            call_index=frame["pass"],
            shape=escape(str(frame["shape"] or "scalar")),
            memory=escape(str(frame["memory"])),
        )
        for index, frame in enumerate(frames)
    )
    return (
        "<div class='tl-pass-animation' data-site='"
        + escape(base_label)
        + "'><button type='button' data-action='play'>Play</button>"
        + "<button type='button' data-action='pause'>Pause</button><ol>"
        + frame_markup
        + "</ol><script>(function(){var root=document.currentScript.parentElement;"
        + "var items=root.querySelectorAll('li');var i=0,t=null;"
        + "function show(){items.forEach(function(x,j){x.style.display=j===i?'':'none';});}"
        + "show();root.querySelector('[data-action=play]').onclick=function(){"
        + "if(t)return;t=setInterval(function(){i=(i+1)%items.length;show();},500);};"
        + "root.querySelector('[data-action=pause]').onclick=function(){clearInterval(t);t=null;};"
        + "})();</script></div>"
    )


__all__ = ["pass_animation_html"]
