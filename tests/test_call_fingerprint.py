"""Capture-time ordered call fingerprint vs its native-forward twin.

Every torch capture stores ``Trace._raw_call_fingerprint``: an ordered rolling
hash over the unpaused owner-thread torch calls and non-root module entries of
the captured forward. ``torchlens._call_fingerprint.fingerprint_native_forward``
reproduces the same stream around a plain forward, so a guarded fast re-run can
prove the same op structure ran (e.g. on a different-length input).
"""

from __future__ import annotations

import threading
import zlib

import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._call_fingerprint import (
    fingerprint_native_forward,
)


class _TwoLayer(nn.Module):
    """``fc1 -> relu -> fc2``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the two layers."""
        return self.fc2(torch.relu(self.fc1(x)))


def _seeded(model: nn.Module) -> nn.Module:
    """Return ``model`` in eval mode (construction already seeded by the caller)."""
    return model.eval()


def _native(model: nn.Module, x: torch.Tensor) -> tuple[int, int]:
    """Return the native-forward fingerprint of ``model(x)``."""
    value, _ = fingerprint_native_forward(model, (x,), {})
    return value


# Kept by shrink-r4 (per-id subset rule): sole coverage of _call_fingerprint.py.
def test_module_intervention_and_sparse_save_keep_the_fingerprint() -> None:
    """Hook arithmetic and save selection never enter the structural fingerprint."""
    torch.manual_seed(0)
    model = _seeded(_TwoLayer())
    x = torch.randn(2, 4)
    plain = tl.trace(model, x)._raw_call_fingerprint  # noqa: SLF001
    direction = torch.linspace(-1.0, 1.0, 8)
    steered = tl.trace(
        model,
        x,
        intervene=tl.when(tl.module("fc1"), tl.steer(direction, strength=2, dim=-1)),
    )
    sparse = tl.trace(model, x, save=tl.module("fc1"))
    assert plain == _native(model, x)
    assert steered._raw_call_fingerprint == plain  # noqa: SLF001
    assert sparse._raw_call_fingerprint == plain  # noqa: SLF001


# Kept by shrink-r4 (per-id subset rule): sole coverage of _state.py.
def test_rolling_hash_is_ordered_and_deterministic() -> None:
    """Token order changes the digest; tokens are stable CRC32 values."""
    first = _state.CallFingerprint(threading.get_ident())
    second = _state.CallFingerprint(threading.get_ident())
    a, b = _state.call_token("relu"), _state.module_token("fc1")
    first.add(a)
    first.add(b)
    second.add(b)
    second.add(a)
    assert first.count == second.count == 2
    assert first.value != second.value
    assert a == zlib.crc32(b"call:relu") and b == zlib.crc32(b"module:fc1")
    assert a != _state.module_token("relu")
