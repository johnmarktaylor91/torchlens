"""F12 pins: the six-state nonfinite status pattern channel (N6).

The finite -> +Inf -> -Inf -> NaN corpus toy exercises the state
derivation; the two-mode degrade (zero coverage = one legend line, partial
coverage = per-node not_checked motifs) is the memo's debug-row contract.
NOT-CHECKED-as-FINITE is a zero-tolerance honesty class: an unchecked op
never presents as checked-clean.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

import torchlens as tl
from torchlens._visualization import lenses
from torchlens._visualization.lenses._nonfinite import (
    derive_nonfinite_channel,
    nonfinite_spec_fn,
)
from torchlens._visualization.lenses.audit import CORPUS
from torchlens._visualization.node_spec import NodeSpec


def _member(name: str) -> Any:
    """Return one corpus member by name."""

    return next(member for member in CORPUS if member.name == name)


@pytest.fixture(scope="module")
def nonfinite_log() -> Any:
    """Full-save capture of the nonfinite chain toy."""

    model, x = _member("nonfinite_chain").build()
    log = tl.trace(model, x)
    yield log
    log.cleanup()


def test_motifs_per_shape_table(nonfinite_log: Any) -> None:
    """box -> striped, oval -> wedged, box3d -> border-only degrade."""

    channel = derive_nonfinite_channel(nonfinite_log)
    spec_fn = nonfinite_spec_fn(channel)
    nan_label = next(label for label, state in channel.states.items() if state == "nan")
    layer = nonfinite_log[nan_label.split(":")[0]]

    boxed = spec_fn(layer, NodeSpec(lines=["x"], shape="box"))
    assert "striped" in boxed.style
    assert ":" in str(boxed.fillcolor)

    oval = spec_fn(layer, NodeSpec(lines=["x"], shape="oval"))
    assert "wedged" in oval.style

    degraded = spec_fn(layer, NodeSpec(lines=["x"], shape="box3d"))
    assert "striped" not in degraded.style and "wedged" not in degraded.style
    assert degraded.penwidth == 3.0  # the border motif still marks the state


def test_zero_coverage_channel_level_degrade() -> None:
    """Zero coverage: ONE prominent legend line, no per-node motifs, coded
    warning at the debug resolve."""

    model, x = _member("nonfinite_chain").build()
    with warnings.catch_warnings():
        # The zero-match save selector warning is the POINT of this fixture:
        # a capture retaining no payloads at all.
        warnings.simplefilter("ignore")
        log = tl.trace(model, x, save=tl.func("no_such_op_zzz"))
    try:
        channel = derive_nonfinite_channel(log)
        assert channel.zero_coverage
        assert "NOT CHECKED" in channel.legend_lines[0]
        spec_fn = nonfinite_spec_fn(channel)
        some_label = next(iter(channel.states))
        untouched = spec_fn(log[some_label.split(":")[0]], NodeSpec(lines=["x"], shape="box"))
        assert untouched.penwidth is None  # no per-node motifs in this mode
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolution = lenses.resolve_lens(log, "debug")
        codes = [getattr(w.message, "fields", {}).get("code") for w in caught]
        assert "lens_nonfinite_not_checked" in codes
        assert any("NOT CHECKED" in line for line in resolution.disclosure)
    finally:
        log.cleanup()
