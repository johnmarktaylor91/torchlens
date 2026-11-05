"""Conservative text-width metrics for sizing node labels before Graphviz lays them out.

Node labels draw in Helvetica, which fontconfig resolves to whatever sans face the host
has; on a stock Linux box that is DejaVu Sans, the widest of the common substitutes
(Helvetica, Arial, Liberation Sans and Nimbus Sans all run narrower). The per-glyph
advances below are DejaVu Sans's own, in ems, so an estimate made from them is never
narrower than what Graphviz measures on any of those faces. A bold glyph (the node
title row) is the larger of :data:`BOLD_FACTOR` times its regular advance and its regular
advance plus :data:`BOLD_EXTRA_EM`: DejaVu Sans Bold widens wide glyphs by a ratio and
narrow ones (``i``, ``l``, ``t``, ``r``, ``f``) by a near-constant amount. The sum carries
:data:`HINTING_ALLOWANCE` on top: the renderer snaps glyph advances to its pixel grid and
rounds each text run up, so a drawn row can come out a little wider than its advances.

Leaf module by design: no runtime imports from the visualization package.
"""

from __future__ import annotations

#: DejaVu Sans advance widths in ems, grouped by equal width.
_WIDTH_GROUPS: tuple[tuple[float, str], ...] = (
    (0.278, "ijl|"),
    (0.318, " ,.;'·"),
    (0.337, ":/\\"),
    (0.352, "f"),
    (0.361, "-"),
    (0.39, "()[]{}"),
    (0.392, "t"),
    (0.411, "r"),
    (0.5, '_*"`'),
    (0.521, "s"),
    (0.525, "z"),
    (0.55, "c"),
    (0.557, "L"),
    (0.575, "F"),
    (0.579, "k"),
    (0.592, "vxy"),
    (0.603, "P"),
    (0.611, "TY"),
    (0.612, "o"),
    (0.613, "a"),
    (0.615, "e"),
    (0.632, "E"),
    (0.634, "hnu"),
    (0.635, "bdgpqS"),
    (0.636, "0123456789$"),
    (0.656, "K"),
    (0.684, "AVX"),
    (0.686, "B"),
    (0.695, "R"),
    (0.698, "C"),
    (0.732, "U"),
    (0.748, "N"),
    (0.752, "H"),
    (0.77, "D"),
    (0.775, "G"),
    (0.78, "&"),
    (0.787, "OQ"),
    (0.818, "w"),
    (0.838, "=+<>#~^"),
    (0.863, "M"),
    (0.95, "%"),
    (0.989, "W"),
    (1.0, "@"),
)

#: Per-character advance in ems (``I``/``J``/``Z`` complete the uppercase set).
GLYPH_EM: dict[str, float] = {
    **{char: width for width, chars in _WIDTH_GROUPS for char in chars},
    "I": 0.295,
    "J": 0.295,
    "Z": 0.685,
    "!": 0.4,
    "?": 0.531,
}

#: Advance for a glyph outside the table (other scripts, symbols): a full em, so an
#: unknown character can only make a node larger, never clip its label.
UNKNOWN_GLYPH_EM = 1.0

#: Headroom over the summed advances for pixel-grid hinting and rounded run widths.
HINTING_ALLOWANCE = 1.05

#: Bold-over-regular advance ratio, covering DejaVu Sans Bold's wide glyphs.
BOLD_FACTOR = 1.15
#: Bold-over-regular advance increment in ems, covering its narrow glyphs.
BOLD_EXTRA_EM = 0.09


def text_width_em(text: str, *, bold: bool = False) -> float:
    """Return the drawn width of ``text`` in ems, never narrower than Graphviz's measure.

    Parameters
    ----------
    text:
        Visible text of one label row (markup already removed).
    bold:
        Whether the row draws in the bold face.

    Returns
    -------
    float
        Width in ems; multiply by the font size in points for points.

    Examples
    --------
    >>> from torchlens._visualization._label_width import text_width_em
    >>> text_width_em("ab") < text_width_em("ab", bold=True)
    True
    """

    widths = [GLYPH_EM.get(char, UNKNOWN_GLYPH_EM) for char in text]
    if bold:
        widths = [max(width * BOLD_FACTOR, width + BOLD_EXTRA_EM) for width in widths]
    return sum(widths) * HINTING_ALLOWANCE
