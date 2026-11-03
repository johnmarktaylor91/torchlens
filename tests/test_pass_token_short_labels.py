"""``:last`` and ``:all`` work on the short layer label too.

The bare-label error on a multi-pass layer suggests ``label:last``; the
glossary says pass tokens work wherever a label is accepted, and a short layer
label (``linear_1``) is accepted for lookups, so ``linear_1:last`` must resolve.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl

# No smoke mark: the smoke tier sits at its size ceiling (tests/conftest.py).


class Recurrent(nn.Module):
    """One Linear reused three times."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(self.fc(x))
        return x


def test_short_label_last_and_all_resolve_like_the_full_label() -> None:
    trace = tl.trace(Recurrent(), torch.randn(1, 4))
    layer = trace.layers["linear_1"]
    full = layer.layer_label
    assert full != "linear_1"
    assert trace["linear_1:last"].label == trace[f"{full}:last"].label == f"{full}:3"
    short_sites = [site.key for site in trace["linear_1:all"]]
    assert short_sites == [site.key for site in trace[f"{full}:all"]]
    assert len(short_sites) == 3


def test_bare_label_error_names_a_spelling_that_resolves() -> None:
    trace = tl.trace(Recurrent(), torch.randn(1, 4))
    try:
        trace["linear_1"]
    except Exception as error:  # noqa: BLE001 -- the message is the subject
        message = str(error)
    else:  # pragma: no cover - a bare multi-pass label must not resolve to one op
        raise AssertionError("bare multi-pass label resolved")
    for spelling in ("linear_1:last", "linear_1:all"):
        if f"'{spelling}'" in message:
            assert trace[spelling] is not None
