"""F16 B3: the ported truncation protocol (treescope memo section 4).

Pinned regressions from the memo's test plan: port equivalence against
upstream on the measured shapes AND a non-default ``doubling_bonus``
(dropping the fifth parameter silently changes which cells are visible);
truncation-budget provenance (OUR 4,000/128/5 numbers, not render_array's
10,000/512/5); the truncation-mask law (a band travels as ``mask=False``,
never a data zero); and the adapter transfer bound on a gpt2-shaped
payload.
"""

from __future__ import annotations

from torchlens.report._notebook._truncation import (
    CELL_BUDGET_DEFAULT,
    EDGE_ITEMS_DEFAULT,
    PER_AXIS_BUDGET_DEFAULT,
)

#: The memo's measured fixture shapes plus edge shapes.
SHAPES = [
    (2, 50257),
    (128, 768),
    (8, 12, 64, 64),
    (3, 224, 224),
    (7,),
    (16, 16),
]


def test_budget_constants_are_the_autovisualizer_numbers() -> None:
    """Provenance pin: our budgets are ArrayAutovisualizer's, NOT render_array's."""

    assert (CELL_BUDGET_DEFAULT, PER_AXIS_BUDGET_DEFAULT, EDGE_ITEMS_DEFAULT) == (4_000, 128, 5)
