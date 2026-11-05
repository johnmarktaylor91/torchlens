"""``Trace.cost.pinned_autograd_memory`` discloses what the default attached save keeps alive.

The default save keeps autograd history (``detach_saved_activations=False``, unchanged
here), so every graph-connected payload pins the tensors autograd saved for the
backward of every op upstream of it. The disclosure counts those bytes once per
storage, excludes the model's own Parameters (and views of them), reads ``Bytes(0)``
when every payload is detached, and is a session-time field: a loaded trace pins
nothing and reads ``None``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl


def _mlp() -> nn.Sequential:
    """Return a seeded ``Linear -> ReLU`` model."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU()).eval()


def _deep() -> nn.Sequential:
    """Return a seeded four-stage model for the selector comparison."""

    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 4)
    ).eval()


def test_pinned_bytes_do_not_survive_save_and_load(tmp_path: Path) -> None:
    """The field is session-time: a loaded trace pins nothing and reads ``None``."""

    trace = tl.trace(_mlp(), torch.randn(2, 4))
    path = tmp_path / "pinned"
    trace.save(path)

    assert tl.load(path).cost.pinned_autograd_memory is None


def test_forged_restored_state_never_surfaces_a_number() -> None:
    """A state dict carrying a pinned-byte number restores as ``None``."""

    trace = tl.trace(_mlp(), torch.randn(2, 4))
    state = trace.__getstate__()
    state["_cost_pinned_autograd_memory"] = 123456
    restored = object.__new__(type(trace))
    restored.__setstate__(state)

    assert restored.cost.pinned_autograd_memory is None
