"""Replay commit rollback restores every field it touched, under its live name.

``_commit_replay_updates`` snapshots a site's saved-out fields before writing
the replayed value and sets them back when anything in the commit raises. A
snapshot keyed by a field name ``Op`` no longer carries turned the error being
unwound into a bare ``AttributeError`` (a fork replay over a payload the digest
could not read surfaced as ``'Op' object has no attribute 'memory'``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention import _replay as replay_module


class _TwoLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


def _site_state(site: tl.Op) -> dict[str, object]:
    return {
        "out": site.out,
        "shape": site.shape,
        "dtype": site.dtype,
        "activation_memory": site.activation_memory,
        "has_saved_activation": site.has_saved_activation,
        "edits": list(site.edits),
        "is_edited": site.is_edited,
    }


def test_commit_rollback_restores_the_site_and_reraises_the_cause(monkeypatch) -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        _TwoLinear(), torch.randn(3, 4), capture=tl.options.CaptureOptions(replayable=True)
    )
    site = trace["relu_1_2"]
    before = _site_state(site)
    replacement = torch.zeros(3, 4, dtype=torch.float64)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("digest unavailable")

    monkeypatch.setattr(replay_module, "_record_capture_digest", _boom)
    with pytest.raises(RuntimeError, match="digest unavailable"):
        replay_module._commit_replay_updates(trace, {site.label: replacement}, {})

    after = _site_state(site)
    assert after["out"] is before["out"]
    assert after["shape"] == before["shape"] == (3, 4)
    assert after["dtype"] == before["dtype"] == torch.float32
    assert after["activation_memory"] == before["activation_memory"]
    assert after["has_saved_activation"] == before["has_saved_activation"]
    assert after["edits"] == before["edits"]
    assert after["is_edited"] == before["is_edited"]


def test_commit_rollback_after_a_partial_write_restores_the_written_fields(monkeypatch) -> None:
    """The failure lands after ``_apply_out_update`` ran: the rollback undoes it."""

    torch.manual_seed(0)
    trace = tl.trace(
        _TwoLinear(), torch.randn(3, 4), capture=tl.options.CaptureOptions(replayable=True)
    )
    site = trace["relu_1_2"]
    before = _site_state(site)
    original_apply = replay_module._apply_out_update

    def _apply_then_fail(target: tl.Op, tensor: torch.Tensor) -> None:
        original_apply(target, tensor)
        raise RuntimeError("after the write")

    monkeypatch.setattr(replay_module, "_apply_out_update", _apply_then_fail)
    with pytest.raises(RuntimeError, match="after the write"):
        replay_module._commit_replay_updates(
            trace, {site.label: torch.zeros(6, 2, dtype=torch.float64)}, {}
        )
    after = _site_state(site)
    assert after["out"] is before["out"]
    assert after["shape"] == (3, 4)
    assert after["dtype"] == torch.float32
    assert after["activation_memory"] == before["activation_memory"]
