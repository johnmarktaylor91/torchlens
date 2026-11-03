"""``native_group_norm`` writes none of its arguments, so no buffer write is journaled for it.

GroupNorm keeps no running statistics: ``F.group_norm`` and ``torch.native_group_norm``
only read their input, weight and bias. A registered buffer handed to either call
keeps its version counter, and the capture journal must agree, in eval and train
mode, whether the buffer is the normalized input or the affine weight. A train-mode
BatchNorm over the same shapes is the control that the journal still sees real
running-statistic writes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl


class _GroupNormModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(2, 4)
        self.register_buffer("shift", torch.rand(2, 4, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x) + self.shift


class _GroupNormOfBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("stored", torch.rand(2, 4, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(self.stored, 2) + x


class _NativeGroupNormOfBuffer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("stored", torch.rand(2, 4, 3, 3))
        self.weight = nn.Parameter(torch.ones(4))
        self.bias = nn.Parameter(torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _mean, _rstd = torch.native_group_norm(
            self.stored, self.weight, self.bias, 2, 4, 9, 2, 1e-5
        )
        return out + x


class _GroupNormBufferAffine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(4))
        self.register_buffer("offset", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(x, 2, self.scale, self.offset)


class _BatchNormControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


def _journal(model: nn.Module, train: bool) -> tuple[set[str], set[str]]:
    """Return (journaled buffer addresses, buffers whose version counter moved)."""

    model.train(train)
    before = {name: buf._version for name, buf in model.named_buffers()}
    torch.manual_seed(0)
    trace = tl.trace(model, torch.rand(2, 4, 3, 3))
    moved = {name for name, buf in model.named_buffers() if buf._version != before[name]}
    journaled = {event.address for event in trace._buffer_write_events}
    assert bool(trace.buffer_write_ops) == bool(journaled)
    return journaled, moved


@pytest.mark.parametrize(
    "model_cls",
    [_GroupNormModule, _GroupNormOfBuffer, _NativeGroupNormOfBuffer, _GroupNormBufferAffine],
)
@pytest.mark.parametrize("train", [False, True])
def test_group_norm_journals_only_real_buffer_writes(
    model_cls: type[nn.Module], train: bool
) -> None:
    journaled, moved = _journal(model_cls(), train)
    assert moved == set()
    assert journaled == set()


def test_batch_norm_control_still_journals_running_statistics() -> None:
    journaled, moved = _journal(_BatchNormControl(), train=True)
    assert moved >= {"bn.running_mean", "bn.running_var"}
    assert journaled >= {"bn.running_mean", "bn.running_var"}
