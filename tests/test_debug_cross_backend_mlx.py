"""Module-level comparison of a real torch model against its real MLX twin (same weights).

The twin has an identity module (a pass-through call: torch inserts an identity op inside
it, MLX records the returned input), an eval-mode dropout, a never-called container (``blocks``: a torch ``ModuleList``
and a plain MLX list) and two blocks. The correct twin must agree everywhere, the
pass-through call included; a twin with one injected bug must be named at the module the
bug is in, localized there because that module's inputs agree.
"""

from __future__ import annotations

import sys
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
    """``head(blocks(drop(skip(fc_in(x)))))`` with an identity and an eval-mode dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.fc_in, self.skip, self.drop = nn.Linear(4, _WIDTH), nn.Identity(), nn.Dropout(0.1)
        self.blocks = nn.ModuleList([_TorchBlock(), _TorchBlock()])
        self.head = nn.Linear(_WIDTH, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.skip(self.fc_in(x)))
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


class _MLXIdentity(mnn.Module):
    """Returns its input unchanged."""

    def __call__(self, x: Any) -> Any:
        return x


class _MLXTwin(mnn.Module):
    """The MLX port of :class:`_TorchTwin`; ``bug_block`` injects the bug in one block."""

    def __init__(self, bug_block: int | None = None) -> None:
        super().__init__()
        self.fc_in, self.skip, self.drop = mnn.Linear(4, _WIDTH), _MLXIdentity(), mnn.Dropout(0.1)
        self.blocks = [_MLXBlock(bug=index == bug_block) for index in range(2)]
        self.head = mnn.Linear(_WIDTH, 2)

    def __call__(self, x: Any) -> Any:
        h = self.drop(self.skip(self.fc_in(x)))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def _copy_linear(source: nn.Linear, target: Any, dtype: str = "float32") -> None:
    """Copy a torch linear layer's float32 weights into an MLX linear layer in ``dtype``."""

    target.weight = mx.array(source.weight.detach().numpy()).astype(getattr(mx, dtype))
    target.bias = mx.array(source.bias.detach().numpy()).astype(getattr(mx, dtype))


def _pair(bug_block: int | None = None, dtype: str = "float32") -> tuple[tl.Trace, tl.Trace]:
    """Trace the torch model and its MLX twin on one input in ``dtype``, weights copied."""

    torch.manual_seed(0)
    reference = _TorchTwin().eval()
    twin = _MLXTwin(bug_block)
    _copy_linear(reference.fc_in, twin.fc_in, dtype)
    _copy_linear(reference.head, twin.head, dtype)
    for source, target in zip(reference.blocks, twin.blocks, strict=True):
        _copy_linear(source.fc, target.fc, dtype)
    twin.eval()
    reference = reference.to(getattr(torch, dtype))
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 4)
    torch_trace = tl.trace(reference, torch.from_numpy(x).to(getattr(torch, dtype)))
    mlx_trace = tl.trace(twin, mx.array(x).astype(getattr(mx, dtype)), backend="mlx")
    assert str(mlx_trace.provenance.backend) == "mlx"
    return torch_trace, mlx_trace


def test_correct_twin_agrees_at_every_paired_module() -> None:
    """Every module pairs by address and agrees, the pass-through identity included."""

    torch_trace, mlx_trace = _pair()
    assert tl.debug.first_divergence(torch_trace, mlx_trace) is None
    frame = tl.debug.compare(torch_trace, mlx_trace).to_pandas()
    rows = frame.set_index("op")
    labels = ("self:1", "fc_in:1", "skip:1", "drop:1", "blocks.0:1", "blocks.1.fc:1", "head:1")
    for label in labels:
        assert rows.loc[label, "module_b"] == label
        assert rows.loc[label, "alignment"] == "supported"
        assert rows.loc[label, "outcome"] == "equal", rows.loc[label, "reason"]
    assert str(rows.loc["blocks.1:1", "inputs_agree"]) == "True"
    assert (frame.attrs["align"], frame.attrs["backends"]) == ("structure", ("torch", "mlx"))
    coverage = frame.attrs["coverage"]
    assert coverage["a"]["boundary_unrecorded"] == coverage["b"]["boundary_unrecorded"] == 0
    assert coverage["a"]["unpaired"] == coverage["b"]["unpaired"] == 0
    assert "blocks:1" not in rows.index
    assert not frame["reason"].fillna("").str.contains("randomness").any()


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
    assert frame.loc["skip:1", "outcome"] == frame.loc["drop:1", "outcome"] == "equal"


def test_label_join_across_real_backends_refuses() -> None:
    """``align="label"`` on a torch/MLX pair refuses before reading values."""

    torch_trace, mlx_trace = _pair()
    with pytest.raises(tl.errors.BackendUnsupportedError):
        tl.debug.compare(torch_trace, mlx_trace, align="label")


@pytest.mark.skipif(
    sys.platform.startswith("linux") and mx.default_device().type == mx.cpu,
    reason="MLX's Linux CPU AddMM supports float32 only, so a bfloat16 matmul refuses",
)
def test_bfloat16_twin_compares_at_the_bfloat16_tolerance() -> None:
    """MLX bfloat16 values read without crashing and keep their dtype for the tolerance."""

    torch_trace, mlx_trace = _pair(dtype="bfloat16")
    assert tl.debug.first_divergence(torch_trace, mlx_trace) is None
    paired = tl.debug.compare(torch_trace, mlx_trace).to_pandas()
    paired = paired[paired["alignment"] == "supported"]
    assert set(paired["outcome"]) == {"equal"}
    assert set(paired["rtol"]) == {0.75}
    assert paired["reason"].str.contains("half precision").all()


class _TorchEagerAttention(nn.Module):
    """Returns ``(output, weights)``, creating the weights first (HF eager attention)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(_WIDTH, _WIDTH)

    def forward(self, x: torch.Tensor) -> Any:
        weights = torch.softmax(x @ x.transpose(0, 1), dim=-1)
        return self.proj(x), weights


class _TorchAttentionNet(nn.Module):
    """``head(attn(fc_in(x))[0])``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc_in, self.attn = nn.Linear(4, _WIDTH), _TorchEagerAttention()
        self.head = nn.Linear(_WIDTH, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _weights = self.attn(self.fc_in(x))
        return self.head(out)


class _MLXAttention(mnn.Module):
    """Returns only its output, plus ``0.1`` when ``bug`` is set."""

    def __init__(self, bug: bool = False) -> None:
        super().__init__()
        self.proj, self.bug = mnn.Linear(_WIDTH, _WIDTH), bug

    def __call__(self, x: Any) -> Any:
        out = self.proj(x)
        return out + 0.1 if self.bug else out


class _MLXAttentionNet(mnn.Module):
    """The MLX port of :class:`_TorchAttentionNet`."""

    def __init__(self, bug: bool = False) -> None:
        super().__init__()
        self.fc_in, self.attn = mnn.Linear(4, _WIDTH), _MLXAttention(bug)
        self.head = mnn.Linear(_WIDTH, 2)

    def __call__(self, x: Any) -> Any:
        return self.head(self.attn(self.fc_in(x)))


def _attention_pair(bug: bool = False) -> tuple[tl.Trace, tl.Trace]:
    """Trace the eager two-output torch model and its single-output MLX port."""

    torch.manual_seed(0)
    reference, twin = _TorchAttentionNet().eval(), _MLXAttentionNet(bug)
    for source, target in ((reference.fc_in, twin.fc_in), (reference.head, twin.head)):
        _copy_linear(source, target)
    _copy_linear(reference.attn.proj, twin.attn.proj)
    x = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    return tl.trace(reference, torch.from_numpy(x)), tl.trace(twin, mx.array(x), backend="mlx")


@pytest.mark.parametrize("bug", [False, True], ids=["correct", "bug"])
def test_eager_two_output_attention_pairs_its_output(bug: bool) -> None:
    """The torch output (returned first, created second) meets MLX's single output."""

    torch_trace, mlx_trace = _attention_pair(bug)
    row = tl.debug.first_divergence(torch_trace, mlx_trace)
    if not bug:
        assert row is None
        frame = tl.debug.compare(torch_trace, mlx_trace).to_pandas().set_index("op")
        assert frame.loc["attn:1", "outcome"] == "equal"
        assert "1 more on side a not compared" in frame.loc["attn:1", "reason"]
    else:
        assert row is not None and (row["op"], row["outcome"]) == ("attn:1", "different")
