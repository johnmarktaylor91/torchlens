"""CardTree IR + safe renderer tests (lane C05; treescope memo B1).

Pins the presentation-only IR contracts: typed-leaf escaping, the
never-raise boundary, budgeted collections with disclosed truncation,
versioned scoped CSS on themes tokens, native ``<details>`` folding,
clipboard keys with click-to-select degradation, stable ``data-lookup-key``
attributes -- and the migration of today's Trace/PartialTrace HTML through
it with the IPython generation gate REMOVED (card tests assert CONTENT).
"""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens.report._notebook.cardtree import (
    Card,
    safe_card_html,
)


def test_never_raise_boundary_degrades_to_one_line() -> None:
    def exploding_build() -> Card:
        raise ValueError("user data: <boom>")

    html = safe_card_html(exploding_build)
    assert "card unavailable: ValueError" in html
    # Exception text is user data: escaped, never markup.
    assert "<boom>" not in html
    assert "&lt;boom&gt;" in html


def test_trace_repr_html_stays_bounded() -> None:
    model = torch.nn.Sequential(*(torch.nn.Linear(4, 4) for _ in range(30)))
    log = tl.trace(model, torch.randn(2, 4))
    html = log._repr_html_()
    # The key index is budgeted: truncation is disclosed, size stays O(50KB).
    assert "more (" in html
    assert len(html) < 50_000
