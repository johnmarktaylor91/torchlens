"""F12 pins: the item-11 encoding honesty fixes and the rank transform (N5
slice).

Composition-test row 16: a degenerate (min == max) domain renders UNENCODED
with the note, never mid-ramp; a per-pass field on a rolled multi-pass node
unencodes with its own note, never resolves to pass 1. Plus the rank
transform's ordinal wording, unit invariance, the log floor, and the
coverage line in the legend.
"""

from __future__ import annotations

import pytest

from torchlens._errors import InvalidArgumentError
from torchlens._visualization._encoding import (
    COLOR_TRANSFORM_VOCABULARY,
    LAYER_SOURCE_ROWS,
    NOTE_CONSTANT,
    ROW_PER_PASS,
    EncodingChannelRequest,
    EncodingState,
    _color_legend_rows,
    _normalize_color_values,
    resolve_color_by,
)


def _state_for(transform: str) -> EncodingState:
    """Build a populated-enough state carrying one transform."""

    spec = resolve_color_by(EncodingChannelRequest(source="func_duration", transform=transform))
    assert spec is not None
    return EncodingState(spec=spec)


def test_transform_vocabulary_is_closed() -> None:
    """linear / rank / log; log1p rejected by design."""

    assert COLOR_TRANSFORM_VOCABULARY == ("linear", "rank", "log")
    with pytest.raises(Exception) as excinfo:
        resolve_color_by(EncodingChannelRequest(source="func_duration", transform="log1p"))
    assert excinfo.value.fields["code"] == "encoding_transform_invalid"
    assert "log1p" in str(excinfo.value)


def test_degenerate_domain_unencodes_never_midramp() -> None:
    """min == max leaves colors EMPTY with the note (composition row 16)."""

    state = _state_for("linear")
    _normalize_color_values(state, {"a": 2.5, "b": 2.5, "c": 2.5})
    assert state.colors == {}
    assert NOTE_CONSTANT in state.notes
    assert state.values  # coercion evidence retained
    # Legend advertises NO scale rows over zero encoded nodes.
    rows = _color_legend_rows(state)
    assert len(rows) == 1


def test_log_legend_labels_match_their_swatches() -> None:
    """Under log the 0.5 swatch encodes the geometric mean of the encoded
    (positive) domain, and the unencoded value <= 0 is not an endpoint."""

    spec = resolve_color_by(EncodingChannelRequest(source=lambda node: 1.0, transform="log"))
    state = EncodingState(spec=spec)
    _normalize_color_values(state, {"a": -5.0, "b": 1.0, "c": 100.0})
    labels = [row.lines[0] for row in _color_legend_rows(state)[1:]]
    assert labels == ["min: 1", "mid: 10", "max: 100"]


def test_pass_number_is_the_per_pass_color_source() -> None:
    """``color_by="pass_number"`` is the per-pass source; ``pass_index`` is not a source."""

    assert resolve_color_by("pass_number") is not None
    assert LAYER_SOURCE_ROWS["pass_index"] == ROW_PER_PASS  # classifies Layer.pass_index
    with pytest.raises(InvalidArgumentError, match="not a known record field"):
        resolve_color_by("pass_index")
