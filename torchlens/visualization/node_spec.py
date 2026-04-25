"""NodeSpec helpers for TorchLens graph visualization labels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from html import escape
from typing import Any, cast


@dataclass
class NodeSpec:
    """Graphviz node attributes produced by TorchLens before user customization.

    The dataclass is intentionally mutable because visualization callbacks are
    user ergonomics APIs: mutating and returning the supplied default spec is a
    natural pattern for small display tweaks.

    Attributes
    ----------
    lines:
        Plain-text rows to render in the node label.
    shape:
        Graphviz node shape.
    fillcolor:
        Optional fill color.
    fontcolor:
        Optional font color.
    style:
        Graphviz node style.
    color:
        Optional border color.
    penwidth:
        Optional border width.
    tooltip:
        Optional node tooltip.
    extra_attrs:
        Additional Graphviz node attributes.
    """

    lines: list[str]
    shape: str = "box"
    fillcolor: str | None = None
    fontcolor: str | None = None
    style: str = "filled,rounded"
    color: str | None = None
    penwidth: float | None = None
    tooltip: str | None = None
    extra_attrs: dict[str, str] = field(default_factory=dict)

    def replace(self, **kwargs: Any) -> "NodeSpec":
        """Return a copy of this spec with selected fields replaced.

        Parameters
        ----------
        **kwargs:
            Dataclass fields to replace.

        Returns
        -------
        NodeSpec
            A copied ``NodeSpec`` with the requested field changes.
        """

        return cast("NodeSpec", dataclass_replace(self, **kwargs))


def render_lines_to_html(lines: list[str]) -> str:
    """Render plain-text node rows as a Graphviz HTML-like table label.

    Parameters
    ----------
    lines:
        Plain-text row contents. Special HTML characters are escaped.

    Returns
    -------
    str
        A string suitable for Graphviz ``label=<...>`` syntax.
    """

    rows = [f'<TR><TD ALIGN="CENTER">{escape(str(line), quote=False)}</TD></TR>' for line in lines]
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        + "".join(rows)
        + "</TABLE>>"
    )
