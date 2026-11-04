"""``:last`` and ``:all`` work on the short layer label too.

A bare label on a multi-pass layer is the Layer; the refusals that need one op
suggest ``label:last``. The glossary says pass tokens work wherever a label is
accepted, and a short layer label (``linear_1``) is accepted for lookups, so
``linear_1:last`` must resolve.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.layer import Layer
from torchlens._data_classes.op import Op
from torchlens._errors import AmbiguousOpLookupError, InvalidArgumentError
from torchlens._selection.errors import AmbiguousSiteError

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


def _pass_spellings(message: str) -> list[str]:
    """Return every pass-qualified spelling (``label:N``, ``:last``, ``:all``) a refusal names."""

    return sorted(set(re.findall(r"[A-Za-z_][\w.]*:(?:\d+|last|all)\b", message)))


def _assert_every_named_spelling_resolves(trace: tl.Trace, message: str) -> None:
    """Every pass spelling the refusal names resolves: one pass to its Op, ``:all`` to all three."""

    spellings = _pass_spellings(message)
    assert spellings, message
    for spelling in spellings:
        resolved = trace[spelling]
        if spelling.endswith(":all"):
            assert len(resolved) == 3
        else:
            assert isinstance(resolved, Op), spelling


def test_bare_label_error_names_a_spelling_that_resolves() -> None:
    """A bare multi-pass label is the Layer; each one-op refusal names spellings that resolve.

    ``trace[label]`` returns the Layer, single- or multi-pass (the indexing
    ruling: a layer label names a Layer); the doors that must give exactly one
    op still refuse the bare label, and so does a single-op read on the
    multi-pass Layer.
    """

    trace = tl.trace(Recurrent(), torch.randn(1, 4))
    layer = trace["linear_1"]
    assert isinstance(layer, Layer)
    assert layer is trace.layers["linear_1"]
    assert layer.num_passes == 3

    with pytest.raises(InvalidArgumentError) as single_read:
        layer.out
    assert single_read.value.fields["code"] == "layer_pass_ambiguous"
    _assert_every_named_spelling_resolves(trace, str(single_read.value))

    with pytest.raises(AmbiguousSiteError) as selector:
        trace[tl.label("linear_1")]
    assert selector.value.fields["code"] == "multipass_bare_label_ambiguous"
    _assert_every_named_spelling_resolves(trace, str(selector.value))

    with pytest.raises(AmbiguousOpLookupError) as ops_door:
        trace.ops["linear_1"]
    _assert_every_named_spelling_resolves(trace, str(ops_door.value))
