"""F12 pins: the item-11 encoding honesty fixes and the rank transform (N5
slice).

Composition-test row 16: a degenerate (min == max) domain renders UNENCODED
with the note, never mid-ramp; a per-pass field on a rolled multi-pass node
unencodes with its own note, never resolves to pass 1. Plus the rank
transform's ordinal wording, unit invariance, the log floor, and the
coverage line in the legend.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization._encoding import (
    COLOR_TRANSFORM_VOCABULARY,
    NOTE_CONSTANT,
    NOTE_LOG_NONPOSITIVE,
    NOTE_PER_PASS_ROLLED,
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


def test_rank_transform_is_ordinal_and_unit_invariant() -> None:
    """Rank fractions depend only on order: seconds vs milliseconds are
    IDENTICAL (the Stage-0 unit-invariance check, by construction)."""

    seconds = _state_for("rank")
    _normalize_color_values(seconds, {"a": 0.001, "b": 0.002, "c": 0.9})
    milliseconds = _state_for("rank")
    _normalize_color_values(milliseconds, {"a": 1.0, "b": 2.0, "c": 900.0})
    assert seconds.colors == milliseconds.colors
    # Ordinal: the huge outlier does NOT compress the others to one end.
    assert len(set(seconds.colors.values())) == 3


def test_rank_legend_wording_is_ordinal_not_ratio() -> None:
    """The legend title line names the mapping honestly."""

    state = _state_for("rank")
    _normalize_color_values(state, {"a": 1.0, "b": 2.0})
    rows = _color_legend_rows(state)
    joined = "\n".join(line for row in rows for line in row.lines)
    assert "ordinal, not ratio" in joined


def test_legend_carries_the_coverage_line() -> None:
    """encoded X of Y falls out of the same computation (N13 slice)."""

    state = _state_for("linear")
    state.eligible_count = 5
    _normalize_color_values(state, {"a": 1.0, "b": 2.0})
    rows = _color_legend_rows(state)
    joined = "\n".join(line for row in rows for line in row.lines)
    assert "encoded 2 of 5 eligible nodes" in joined


def test_log_transform_discloses_the_floor() -> None:
    """Non-positive values unencode under log with the disclosed note."""

    state = _state_for("log")
    _normalize_color_values(state, {"a": 0.0, "b": 1.0, "c": 10.0})
    assert "a" not in state.colors
    assert {"b", "c"} <= set(state.colors)
    assert NOTE_LOG_NONPOSITIVE in state.notes


def test_per_pass_field_on_rolled_unencodes_with_its_own_note(tmp_path: Any) -> None:
    """The measured min:1/max:1 uniform mid-ramp defect is dead: rolled
    multi-pass nodes unencode per-pass fields (composition row 16)."""

    class Loop(nn.Module):
        """Three-pass tied linear loop."""

        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: Any) -> Any:
            for _ in range(3):
                x = torch.relu(self.fc(x))
            return x

    log = tl.trace(Loop(), torch.randn(1, 4))
    try:
        log.draw(
            color_by="pass_index",
            vis_mode="rolled",
            vis_outpath=str(tmp_path / "rolled"),
            vis_fileformat="svg",
            vis_save_only=True,
        )
        state = log._last_encoding_state if hasattr(log, "_last_encoding_state") else None
        if state is None:
            from torchlens.visualization._encoding import EncodingState as _ES

            del _ES  # fall through to the DOT-level assertion below
        graph = log.draw(
            color_by="pass_index",
            vis_mode="rolled",
            vis_outpath=str(tmp_path / "rolled2"),
            vis_fileformat="svg",
            vis_save_only=True,
            return_graph=True,
        )
        assert NOTE_PER_PASS_ROLLED in graph.source
    finally:
        log.cleanup()


def test_plain_string_color_by_keeps_linear_default() -> None:
    """A bare string source resolves with the historical linear transform."""

    spec = resolve_color_by("func_duration")
    assert spec is not None
    assert spec.transform == "linear"
