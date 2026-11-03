"""R45: the ``_state.active_capture()`` snapshot tracks a live capture.

The raw-access no-growth ratchet and the access-policy declaration check moved
to ``scripts/checks/state_access_ratchet.py``; this behaviour test stays with
the test runner because it captures a model.
"""

from __future__ import annotations


def test_active_capture_snapshot_tracks_live_session() -> None:
    """The accessor reflects an active capture and resets after it."""

    import torch
    from torch import nn

    import torchlens as tl
    from torchlens import _state

    seen: dict[str, object] = {}

    class _Probe(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seen["snapshot"] = _state.active_capture()
            return x + 1

    trace = tl.trace(_Probe(), torch.randn(2, 2))
    mid_trace, mid_enabled = seen["snapshot"]  # type: ignore[misc]
    assert mid_enabled is True
    assert mid_trace is trace
    after_trace, after_enabled = _state.active_capture()
    assert after_trace is None
    assert after_enabled is False
