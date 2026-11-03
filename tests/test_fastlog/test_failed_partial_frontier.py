"""Failed-partial ``last_event_*`` frontier metadata (B8-46).

A failed fastlog capture's ``last_event_label`` used to point at the exception
unwind (``root:exit:1``) because the unwind records module-exit events after
the failure point. The stamped best-effort metadata must instead name the
failure frontier: the deepest event that actually ran before the exception.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class _Boom(nn.Module):
    """Submodule that raises mid-forward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raise unconditionally."""

        raise RuntimeError("boom")


class _FrontierModel(nn.Module):
    """Model whose middle submodule fails mid-forward."""

    def __init__(self) -> None:
        """Initialize two linear layers around a failing submodule."""

        super().__init__()
        self.a = nn.Linear(3, 3)
        self.boom = _Boom()
        self.b = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear -> relu -> failing submodule -> linear."""

        x = torch.relu(self.a(x))
        x = self.boom(x)
        return self.b(x)


def _failed_partial() -> tl.Recording:
    """Capture the failed partial recording for the frontier model."""

    with pytest.warns(RuntimeWarning, match="capture attempt failed"):
        return tl.record(
            _FrontierModel(),
            torch.ones(1, 3),
            default_op=True,
            default_module=True,
            on_forward_error="return_partial",
        )


def test_failed_partial_last_event_points_at_failure_frontier() -> None:
    """``last_event_label`` names the failing frontier, not the unwind."""

    recording = _failed_partial()

    assert recording.is_failed is True
    assert recording.status == "partial_error"
    # The failing submodule was ENTERED but never completed: that entry is the
    # frontier. The unwind's trailing module-exit events (boom:exit:1,
    # root:exit:1) must not be reported as the last event.
    assert recording.last_event_label == "boom:enter:1"
    assert recording.last_successful_op_label == "relu_1_3_raw"


def test_failed_partial_keeps_string_only_metadata() -> None:
    """Failed-partial metadata stays best-effort string-only."""

    recording = _failed_partial()

    for field_name in (
        "last_event_label",
        "last_event_func",
        "last_event_source_line",
        "last_event_input_meta",
        "last_successful_op_label",
        "error_repr",
        "error_traceback",
    ):
        value = getattr(recording, field_name)
        assert value is None or isinstance(value, str)
    assert isinstance(recording.num_committed_ops, int)
