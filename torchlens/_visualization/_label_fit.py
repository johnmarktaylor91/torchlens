"""Keep a long node row inside its box where a browser draws it wider than Graphviz measured.

Graphviz centres each row of a node label from its own measure, which rounds
every glyph advance to a 96 dpi pixel (:mod:`._text_metrics`), and sizes the box
from the widest row plus the cell padding and the node margin. A browser draws
the real advances, so a row runs right by the difference: about 3 % of its
width, which the 10 pt of side padding absorbs for ordinary rows but not for a
long one (GPT-2's ``num_embeddings=50257, embedding_dim=768, padding_idx=-1``
ran 10.5 pt past its centre line and touched the border).

:func:`fitted_cell_width` returns the minimum cell width that keeps every row
:data:`ROW_CLEARANCE` points inside the border, or ``None`` when Graphviz's own
width already does, so ordinary labels keep their bytes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ._text_metrics import graphviz_text_width, text_width

__all__ = ("ROW_CLEARANCE", "Run", "fitted_cell_width")

#: One run of a row: its text, point size and weight.
Run = tuple[str, float, bool]

#: Graphviz's default node margin (0.11 in), in points per side.
_NODE_MARGIN = 0.11 * 72
#: The cell padding node labels give every row, in points per side.
_CELL_PADDING = 2.0
#: Points kept between a row's drawn right end and the node border.
ROW_CLEARANCE = 3.0
#: Rows shorter than this cannot need a wider cell: rounding moves each glyph at
#: most half a 96 dpi pixel (0.375 pt), and a row must run past the side padding
#: less the clearance (6.92 pt), so a node with only short rows is not measured.
_MIN_CHARS = math.ceil((_CELL_PADDING + _NODE_MARGIN - ROW_CLEARANCE) / (0.5 * 72 / 96))


def fitted_cell_width(rows: Sequence[Sequence[Run]]) -> int | None:
    """Return the cell width that keeps every row inside its box, else ``None``.

    A row is laid out from Graphviz's width ``g`` and drawn ``t`` wide, starting
    ``g / 2`` left of the centre, so its right end sits ``t - g / 2`` right of it;
    the border sits ``W / 2 + margin`` right of it for a cell ``W`` wide.

    Parameters
    ----------
    rows:
        Each row's runs (text, point size, bold).

    Returns
    -------
    int | None
        The minimum cell width in points, or ``None`` when the default fits.

    Examples
    --------
    >>> from torchlens._visualization._label_fit import fitted_cell_width
    >>> fitted_cell_width([[("relu_1_2", 14.0, True)], [("(1, 5, 768), 15.4 KB", 14.0, False)]])
    >>> row = "num_embeddings=50257, embedding_dim=768, padding_idx=-1"
    >>> fitted_cell_width([[(row, 14.0, False)]])
    402
    """

    if all(sum(len(text) for text, _size, _bold in row) < _MIN_CHARS for row in rows):
        return None
    default = 0.0
    needed = 0.0
    for row in rows:
        laid_out = sum(graphviz_text_width(text, size, bold) for text, size, bold in row)
        drawn = sum(text_width(text, size, bold) for text, size, bold in row)
        default = max(default, laid_out + 2 * _CELL_PADDING)
        needed = max(needed, 2 * (drawn + ROW_CLEARANCE - _NODE_MARGIN) - laid_out)
    if needed <= default:
        return None
    return math.ceil(needed)
