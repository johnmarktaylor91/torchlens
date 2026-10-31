"""``Trace.cost.pinned_autograd_memory`` discloses what the default attached save keeps alive (R2-37).

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
from torchlens.options import CaptureOptions


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


def test_default_save_discloses_exact_pinned_bytes() -> None:
    """``addmm`` pins its input (2x4 fp32) and ``relu`` its result (2x3 fp32): 56 bytes."""

    model = _mlp()
    trace = tl.trace(model, torch.randn(2, 4))

    assert trace.options.detach_saved_activations is False
    assert trace["0"].out.grad_fn is not None
    assert int(trace.cost.pinned_autograd_memory) == (2 * 4 + 2 * 3) * 4


def test_detached_and_no_grad_saves_pin_nothing() -> None:
    """A detached save and a no-grad capture both disclose zero pinned bytes."""

    model = _deep()
    x = torch.randn(3, 16)
    detached = tl.trace(model, x, capture=CaptureOptions(detach_saved_activations=True))
    with torch.no_grad():
        no_grad = tl.trace(model, x)

    assert int(detached.pinned_autograd_memory) == 0
    assert int(no_grad.pinned_autograd_memory) == 0


def test_an_early_selector_pins_less_than_the_save_all_default() -> None:
    """Saving only the first stage pins only that stage's upstream graph."""

    model = _deep()
    x = torch.randn(3, 16)
    save_all = tl.trace(model, x)
    early = tl.trace(model, x, save=tl.module("0"))

    assert 0 < int(early.pinned_autograd_memory) < int(save_all.pinned_autograd_memory)


def test_pinned_bytes_do_not_survive_save_and_load(tmp_path: Path) -> None:
    """The field is session-time: a loaded trace pins nothing and reads ``None``."""

    trace = tl.trace(_mlp(), torch.randn(2, 4))
    path = tmp_path / "pinned"
    trace.save(path)

    assert tl.load(path).pinned_autograd_memory is None


def test_pickle_and_deepcopy_read_none() -> None:
    """A pickled or deep-copied trace pins nothing and reads ``None``."""

    import copy
    import pickle

    trace = tl.trace(_mlp(), torch.randn(2, 4))
    assert trace.cost.pinned_autograd_memory is not None

    assert pickle.loads(pickle.dumps(trace)).cost.pinned_autograd_memory is None
    assert copy.deepcopy(trace).pinned_autograd_memory is None


def test_forged_restored_state_never_surfaces_a_number() -> None:
    """A state dict carrying a pinned-byte number restores as ``None``."""

    trace = tl.trace(_mlp(), torch.randn(2, 4))
    state = trace.__getstate__()
    state["pinned_autograd_memory"] = 123456
    restored = object.__new__(type(trace))
    restored.__setstate__(state)

    assert restored.cost.pinned_autograd_memory is None
