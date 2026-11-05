"""Forward replay of call arguments that a later write changes outside any recorded op.

Two capture bugs made isolated replay disagree with the saved out:

* A slice argument holding a tensor bound (``x[i : i + n]`` with a tensor ``i``) was
  saved by reference, so a later ``i += n`` moved the saved bounds and replay indexed
  a different range (an empty one, for the usual running-offset loop).
* The deferred payload (clone-on-write alias) of a tensor handed to a train-mode
  ``batch_norm`` / ``instance_norm`` as running statistics was not copied out before
  the call updated those statistics, so the producer's saved out showed the updated
  value rather than the value it produced.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
import torchlens.backends.torch.wrappers as _wrappers
from torchlens._utils import tensor_utils as _tu
from torchlens._utils.arg_handling import copy_arg_tree


@contextlib.contextmanager
def _payload_clone_mode(defer: bool) -> Iterator[None]:
    """Force deferred (clone-on-write) or eager payload clones inside the block."""

    prev_tu, prev_w = _tu._DEFER_ENABLED, _wrappers._COW_ENABLED
    _tu._DEFER_ENABLED = defer
    _wrappers._COW_ENABLED = defer
    try:
        yield
    finally:
        _tu._DEFER_ENABLED = prev_tu
        _wrappers._COW_ENABLED = prev_w


class _RunningOffsetBags(nn.Module):
    """Pool variable-size bags with a running tensor offset, as attention MIL models do."""

    def forward(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        pooled = []
        start = 0
        for n in counts:
            pooled.append(x[start : start + n].sum(0))
            start += n  # an int first, then a tensor updated in place
        return torch.stack(pooled)


class _RunningOffsetLevels(nn.Module):
    """Slice levels off a flattened sequence with a tuple index, as multi-scale DETRs do."""

    def forward(self, memory: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        cursor = 0
        levels = []
        for size in sizes:
            levels.append(memory[:, cursor : cursor + size].mean(1))
            cursor += size
        return torch.cat(levels, dim=-1)


class _AdaptiveNormRepeat(nn.Module):
    """Adaptive instance norm through a train-mode batch_norm on repeated buffers."""

    def __init__(self, functional: str) -> None:
        super().__init__()
        self.functional = functional
        self.weight = nn.Parameter(torch.randn(8))
        self.bias = nn.Parameter(torch.randn(8))
        self.register_buffer("running_mean", torch.zeros(4))
        self.register_buffer("running_var", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        running_mean = self.running_mean.repeat(n)
        running_var = self.running_var.repeat(n)
        flat = x.contiguous().view(1, n * c, h * w)
        if self.functional == "instance_norm":
            normed = F.instance_norm(
                flat, running_mean, running_var, self.weight, self.bias, True, 0.1, 1e-5
            )
        else:
            normed = F.batch_norm(
                flat, running_mean, running_var, self.weight, self.bias, True, 0.1, 1e-5
            )
        return normed.view(n, c, h, w)


def test_copy_arg_tree_snapshots_tensor_slice_bounds() -> None:
    """A saved slice keeps its tensor bounds' call-time values after an in-place update."""

    start = torch.tensor(4)
    index = (slice(None), slice(start, start + 3))
    saved = copy_arg_tree(index)
    start += 10
    assert saved[1].start.item() == 4
    assert saved[1].stop.item() == 7
    plain = slice(1, 5)
    assert copy_arg_tree(plain) is plain


def test_running_offset_slice_validates() -> None:
    """Replay slices the range the call read, not the range after ``start += n``."""

    torch.manual_seed(0)
    model = _RunningOffsetBags().eval()
    inputs = [torch.randn(20, 4), torch.tensor([8, 12])]
    assert tl.validate(model, inputs, seed=0)


def test_running_offset_tuple_index_validates() -> None:
    """The tuple-index path (validation's own capture copies it item by item) is covered."""

    torch.manual_seed(0)
    model = _RunningOffsetLevels().eval()
    inputs = [torch.randn(2, 10, 3), torch.tensor([4, 6])]
    assert tl.validate(model, inputs, seed=0)


@pytest.mark.parametrize("functional", ["batch_norm", "instance_norm"])
def test_train_mode_norm_keeps_producer_payload(functional: str) -> None:
    """A deferred payload passed as running stats is copied out before the stats update."""

    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 3)
    with _payload_clone_mode(defer=True):
        deferred = tl.trace(_AdaptiveNormRepeat(functional).eval(), x)
    with _payload_clone_mode(defer=False):
        eager = tl.trace(_AdaptiveNormRepeat(functional).eval(), x)
    for label in ("repeat_1_1:1", "repeat_2_2:1"):
        assert torch.equal(deferred[label].out, eager[label].out), label
    assert torch.equal(deferred["repeat_1_1:1"].out, torch.zeros(8))


@pytest.mark.parametrize("functional", ["batch_norm", "instance_norm"])
def test_train_mode_norm_on_repeated_buffers_validates(functional: str) -> None:
    """Replaying the repeat reproduces its saved out under deferred payload clones."""

    torch.manual_seed(0)
    with _payload_clone_mode(defer=True):
        assert tl.validate(_AdaptiveNormRepeat(functional).eval(), torch.randn(2, 4, 3, 3), seed=0)
