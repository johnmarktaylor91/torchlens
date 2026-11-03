"""Layer counts of BatchNorm models match an eager census of the forward.

An eval-mode ``batch_norm`` call reads its ``running_mean`` / ``running_var``
buffers and writes nothing: its own ``training`` argument is false. The
exhaustive capture used to journal a "fused" write for both buffers on every
call anyway, so each eval BatchNorm grew two buffer-version layers the eager
computation does not have (resnet18: 151 ops against 111 for a cooked
recording of the same forward). The expected counts here come from a census
taken with a torch function mode, never from a literal. Ops are the census
unit; layers group repeated calls of one module (resnet18 calls each
BasicBlock's ReLU twice) and must agree between the two capture paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_leaves

import torchlens as tl


@dataclass
class _Census:
    """Eager facts of one forward: tensor-producing calls, buffer reads and writes."""

    ops: int = 0
    buffers_read: set[str] = field(default_factory=set)
    buffer_writes: int = 0

    def expected_ops(self, n_inputs: int = 1, n_outputs: int = 1) -> int:
        """Inputs, one op per call, one per buffer read, one per buffer write, outputs."""

        return n_inputs + self.ops + len(self.buffers_read) + self.buffer_writes + n_outputs


class _CensusMode(TorchFunctionMode):
    """Count calls that return a tensor and the registered buffers they read or write.

    A write is a change of the buffer's bytes across the call: native
    ``batch_norm`` updates its running statistics without bumping ``_version``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.named_buffers = list(model.named_buffers())
        self.buffer_names = {id(buf): name for name, buf in self.named_buffers}
        self.census = _Census()

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        before = [buf.detach().clone() for _, buf in self.named_buffers]
        out = func(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            self.census.ops += 1
            for leaf in tree_leaves((args, kwargs)):
                if id(leaf) in self.buffer_names:
                    self.census.buffers_read.add(self.buffer_names[id(leaf)])
            self.census.buffer_writes += sum(
                not torch.equal(buf, old)
                for (_, buf), old in zip(self.named_buffers, before, strict=True)
            )
        return out


def _census(model: nn.Module, x: torch.Tensor) -> _Census:
    mode = _CensusMode(model)
    with torch.no_grad(), mode:
        model(x)
    return mode.census


class _ConvBnBlock(nn.Module):
    """Two conv-BatchNorm stages with a residual add."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(h)) + h)


def _buffer_layers(trace: Any) -> list[Any]:
    return [layer for layer in trace.layers if layer.layer_type == "buffer"]


def _compute_ops(trace: Any) -> list[Any]:
    """Ops of the forward's computation: no inputs, outputs or buffer sources."""

    return [op for op in trace.ops if not (op.is_input or op.is_output or op.is_buffer)]


def _assert_counts_match_census(model: nn.Module, x: torch.Tensor) -> None:
    census = _census(model, x)
    exhaustive = tl.trace(model, x)
    cooked = tl.record(model, x, save=lambda ctx: ctx.kind == "op").to_trace()

    assert census.buffer_writes == 0, "eval-mode BatchNorm wrote a buffer in eager torch"
    assert len(_compute_ops(exhaustive)) == len(_compute_ops(cooked)) == census.ops
    assert len(_buffer_layers(exhaustive)) == len(census.buffers_read)
    assert len(_buffer_layers(cooked)) == len(census.buffers_read)
    assert len(exhaustive) == len(cooked) == census.expected_ops()
    assert len(exhaustive.layers) == len(cooked.layers)
    repeated_passes = sum(layer.num_passes - 1 for layer in exhaustive.layers)
    assert len(exhaustive.layers) == census.expected_ops() - repeated_passes
    assert not exhaustive.buffer_write_ops


def test_eval_batchnorm_block_layer_count_matches_census() -> None:
    torch.manual_seed(0)
    _assert_counts_match_census(_ConvBnBlock().eval(), torch.randn(2, 3, 8, 8))


def test_train_batchnorm_block_still_journals_its_writes() -> None:
    """Training mode writes running_mean, running_var and num_batches_tracked per norm."""

    torch.manual_seed(0)
    model = _ConvBnBlock().train()
    x = torch.randn(2, 3, 8, 8)
    census = _census(model, x)
    trace = tl.trace(_ConvBnBlock().train(), x)
    assert census.buffer_writes == 6
    assert len(trace.buffer_write_ops) == census.buffer_writes


@pytest.mark.heavy
def test_eval_resnet_layer_count_matches_census() -> None:
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = torchvision.models.resnet18().eval()
    _assert_counts_match_census(model, torch.randn(1, 3, 64, 64))
