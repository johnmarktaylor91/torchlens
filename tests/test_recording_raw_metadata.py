"""Pins for ``Recording.raw_metadata()`` (DOCUMENTED-UNSTABLE spelling).

The accessor exports payload-free per-event metadata straight from the raw
capture event stream — every event the recorder saw, retained or not — and
refuses typed (never a silently empty tuple) when the stream is gone.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.recording._exceptions import RecorderStateError
from torchlens.recording._types import _RAW_METADATA_FIELDS


def _model() -> nn.Module:
    """Return the small deterministic demo model.

    Returns
    -------
    torch.nn.Module
        Three-layer sequential model.
    """

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


# Kept by shrink-r4 (per-id subset rule): sole coverage of recording/_types.py.
def test_raw_metadata_covers_every_event_payload_free() -> None:
    """One row per chronological event, closed field set plus retained flag."""

    recording = tl.record(_model(), torch.randn(1, 4), save=tl.func("relu"))

    rows = recording.raw_metadata()

    assert len(rows) == len(recording.lazy_trace.contexts) > len(recording)
    expected_keys = set(_RAW_METADATA_FIELDS) | {"retained"}
    assert all(set(row) == expected_keys for row in rows)
    # Payload-bearing fields never appear: a row cannot force a payload read.
    assert not any("recent_events" in row or "bool_value" in row for row in rows)
    kinds = {row["kind"] for row in rows}
    assert {"input", "op", "module_enter", "module_exit"}.issubset(kinds)
    op_rows = [row for row in rows if row["kind"] == "op"]
    assert all(row["shape"] is not None and row["dtype"] is not None for row in op_rows)


# Kept by shrink-r4 (per-id subset rule): sole coverage of recording/_types.py.
def test_raw_metadata_refuses_when_event_stream_is_gone() -> None:
    """A stream-less recording refuses typed, never an empty tuple."""

    recording = tl.record(_model(), torch.randn(1, 4), save=tl.func("relu"))
    object.__setattr__(recording, "_capture_events", None)

    with pytest.raises(RecorderStateError) as excinfo:
        recording.raw_metadata()
    assert excinfo.value.fields["code"] == "recording_event_stream_unavailable"


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_raw_metadata_works_on_failed_partial_recordings() -> None:
    """Failed partials keep their event stream readable (no cooking needed)."""

    class Exploding(nn.Module):
        """Model that fails mid-forward after one linear op."""

        def __init__(self) -> None:
            """Initialize the leading linear layer."""

            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one op then raise.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Never returns.
            """

            x = self.fc(x)
            raise RuntimeError("boom")

    torch.manual_seed(1)
    recording = tl.record(
        Exploding(),
        torch.randn(1, 4),
        save=tl.func("linear"),
        recording=tl.options.RecordingOptions(on_forward_error="return_partial"),
    )
    assert recording.is_failed

    rows = recording.raw_metadata()

    assert any(row["kind"] == "op" and row["func_name"] == "linear" for row in rows)
