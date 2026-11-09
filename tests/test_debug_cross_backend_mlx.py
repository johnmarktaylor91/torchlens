"""Module-level comparison of a real torch model against its real MLX twin (same weights).

The twin has an eval-mode dropout (a pass-through call: torch logs an op inside it, MLX
records the returned input), a never-called container (``blocks``: a torch ``ModuleList``
and a plain MLX list) and two blocks. The correct twin must agree everywhere, the
pass-through call included; a twin with one injected bug must be named at the module the
bug is in, localized there because that module's inputs agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = [pytest.mark.backend_mlx, pytest.mark.optional, pytest.mark.rare]

pytest.importorskip("pandas")
mx = pytest.importorskip("mlx.core")
mnn = pytest.importorskip("mlx.nn")

_WIDTH = 8


class _TorchBlock(nn.Module):
    """``act(fc(x))``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc, self.act = nn.Linear(_WIDTH, _WIDTH), nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(x))


class _TorchTwin(nn.Module):
    """``head(blocks(drop(fc_in(x))))`` with an eval-mode dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.fc_in, self.drop = nn.Linear(4, _WIDTH), nn.Dropout(0.1)
        self.blocks = nn.ModuleList([_TorchBlock(), _TorchBlock()])
        self.head = nn.Linear(_WIDTH, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.fc_in(x))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class _MLXBlock(mnn.Module):
    """``act(fc(x))``, plus ``0.1`` when ``bug`` is set."""

    def __init__(self, bug: bool = False) -> None:
        super().__init__()
        self.fc, self.act = mnn.Linear(_WIDTH, _WIDTH), mnn.ReLU()
        self.bug = bug

    def __call__(self, x: Any) -> Any:
        out = self.act(self.fc(x))
        return out + 0.1 if self.bug else out


class _MLXTwin(mnn.Module):
    """The MLX port of :class:`_TorchTwin`; ``bug_block`` injects the bug in one block."""

    def __init__(self, bug_block: int | None = None) -> None:
        super().__init__()
        self.fc_in, self.drop = mnn.Linear(4, _WIDTH), mnn.Dropout(0.1)
        self.blocks = [_MLXBlock(bug=index == bug_block) for index in range(2)]
        self.head = mnn.Linear(_WIDTH, 2)

    def __call__(self, x: Any) -> Any:
        h = self.drop(self.fc_in(x))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def _copy_linear(source: nn.Linear, target: Any) -> None:
    """Copy a torch linear layer's weights into an MLX linear layer (same layout)."""

    target.weight = mx.array(source.weight.detach().numpy())
    target.bias = mx.array(source.bias.detach().numpy())


def _pair(bug_block: int | None = None) -> tuple[tl.Trace, tl.Trace]:
    """Trace the torch model and its MLX twin on one input, weights copied across."""

    torch.manual_seed(0)
    reference = _TorchTwin().eval()
    twin = _MLXTwin(bug_block)
    _copy_linear(reference.fc_in, twin.fc_in)
    _copy_linear(reference.head, twin.head)
    for source, target in zip(reference.blocks, twin.blocks, strict=True):
        _copy_linear(source.fc, target.fc)
    twin.eval()
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 4)
    torch_trace = tl.trace(reference, torch.from_numpy(x))
    mlx_trace = tl.trace(twin, mx.array(x), backend="mlx")
    assert str(mlx_trace.provenance.backend) == "mlx"
    return torch_trace, mlx_trace


def test_correct_twin_agrees_at_every_paired_module() -> None:
    """Every module pairs by address and agrees, the pass-through dropout included."""

    torch_trace, mlx_trace = _pair()
    assert tl.debug.first_divergence(torch_trace, mlx_trace) is None
    frame = tl.debug.compare(torch_trace, mlx_trace).to_pandas()
    rows = frame.set_index("op")
    for label in ("fc_in:1", "drop:1", "blocks.0:1", "blocks.1.fc:1", "blocks.1:1", "head:1"):
        assert rows.loc[label, "module_b"] == label
        assert rows.loc[label, "alignment"] == "supported"
        assert rows.loc[label, "outcome"] == "equal", rows.loc[label, "reason"]
    coverage = frame.attrs["coverage"]
    assert coverage["b"]["boundary_unrecorded"] == 0


@pytest.mark.parametrize("bug_block", [0, 1])
def test_injected_bug_is_named_at_its_module(bug_block: int) -> None:
    """The first divergence is the bugged block, localized because its inputs agree."""

    torch_trace, mlx_trace = _pair(bug_block)
    row = tl.debug.first_divergence(torch_trace, mlx_trace)
    assert row is not None
    assert (row["op"], row["module_b"]) == (f"blocks.{bug_block}:1", f"blocks.{bug_block}:1")
    assert (row["alignment"], row["outcome"]) == ("supported", "different")
    assert row["reason"].startswith("localized inside this module")
    frame = tl.debug.compare(torch_trace, mlx_trace).to_pandas().set_index("op")
    assert frame.loc[f"blocks.{bug_block}.fc:1", "outcome"] == "equal"
    assert frame.loc["drop:1", "outcome"] == "equal"


def test_label_join_across_real_backends_refuses() -> None:
    """``align="label"`` on a torch/MLX pair refuses before reading values."""

    torch_trace, mlx_trace = _pair()
    with pytest.raises(tl.errors.BackendUnsupportedError):
        tl.debug.compare(torch_trace, mlx_trace, align="label")
