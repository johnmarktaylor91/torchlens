"""Operation fixtures that separate host overhead from device compute.

Each fixture is a stack of identical blocks, each block a module running a few
operations of one kind, followed by a one-operation ``head`` module. Growing the
operation count at a fixed tensor size shows the per-operation host cost; growing the
tensor size at a fixed count shows when device compute dominates. Every fixture names
a middle block as its hook site, a few blocks as its matched retained set, and
``head`` as the site whose output is the model output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

SIZES = {"small": 16, "medium": 256, "large": 1024}
OP_COUNTS = (100, 1000, 10000)
SMOKE_OP_COUNTS = (100, 1000)


class _Scale(nn.Module):
    """One multiply: the model output's own module, so every arm can read it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.0


class _Pointwise(nn.Module):
    ops = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x * 0.5 + 0.01)


class _Matmul(nn.Module):
    ops = 1

    def __init__(self, d: int) -> None:
        super().__init__()
        # Orthogonal weights keep the norm fixed through thousands of matmuls.
        self.w = nn.Parameter(torch.linalg.qr(torch.randn(d, d))[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w)


class _Conv(nn.Module):
    ops = 3

    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The residual inside tanh keeps values bounded without decaying to zero.
        return torch.tanh(self.conv(x) + x)


class _Norm(nn.Module):
    ops = 2

    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) + x


class _AttentionFused(nn.Module):
    ops = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.scaled_dot_product_attention(x, x, x) + x) * 0.5


class _AttentionDecomposed(nn.Module):
    ops = 7

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(x, x.transpose(-2, -1)) * (x.shape[-1] ** -0.5)
        # Halving the residual sum keeps a chain of hundreds of blocks finite.
        return (torch.matmul(torch.softmax(scores, dim=-1), x) + x) * 0.5


class _ViewsInplace(nn.Module):
    ops = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(-2, -1).contiguous().transpose(-2, -1)
        return y.mul_(0.5).add_(x, alpha=0.5)


class _Reduction(nn.Module):
    ops = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=-1, keepdim=True) * 0.5


class _DynamicBranch(nn.Module):
    ops = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x - 0.5
        return x + 0.5


class _ClampGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return -x

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        return -grad.clamp(-1.0, 1.0)


class _CustomAutograd(nn.Module):
    ops = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ClampGrad.apply(x)


@dataclass(frozen=True)
class FixtureKind:
    """How to build one fixture kind."""

    name: str
    make_block: Callable[[int], nn.Module]
    ops_per_block: int
    image: bool = False


KINDS = {
    "pointwise": FixtureKind("pointwise", lambda d: _Pointwise(), _Pointwise.ops),
    "matmul": FixtureKind("matmul", _Matmul, _Matmul.ops),
    "conv": FixtureKind("conv", _Conv, _Conv.ops, image=True),
    "norm": FixtureKind("norm", _Norm, _Norm.ops),
    "attention_fused": FixtureKind("attention_fused", lambda d: _AttentionFused(), 2),
    "attention_decomposed": FixtureKind(
        "attention_decomposed", lambda d: _AttentionDecomposed(), _AttentionDecomposed.ops
    ),
    "views_inplace": FixtureKind("views_inplace", lambda d: _ViewsInplace(), _ViewsInplace.ops),
    "reduction": FixtureKind("reduction", lambda d: _Reduction(), _Reduction.ops),
    "dynamic_branch": FixtureKind("dynamic_branch", lambda d: _DynamicBranch(), _DynamicBranch.ops),
    "custom_autograd": FixtureKind(
        "custom_autograd", lambda d: _CustomAutograd(), _CustomAutograd.ops
    ),
}


class FixtureModel(nn.Module):
    """``blocks`` (a stack of one kind) followed by ``head``."""

    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*blocks)
        self.head = _Scale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))


def fixture_name(kind: str, n_ops: int, size: str) -> str:
    """Return the subject name of one fixture, e.g. ``fx-pointwise-1000-small``."""

    return f"fx-{kind}-{n_ops}-{size}"


def parse_fixture_name(name: str) -> tuple[str, int, str]:
    """Split a fixture subject name into kind, operation count and size."""

    _prefix, rest = name.split("-", 1)
    kind, n_ops, size = rest.rsplit("-", 2)
    return kind, int(n_ops), size


def build_fixture(name: str) -> tuple[nn.Module, torch.Tensor, dict[str, Any]]:
    """Build a fixture model, its input and its site metadata.

    Parameters
    ----------
    name:
        A name from :func:`fixture_name`.

    Returns
    -------
    tuple[nn.Module, torch.Tensor, dict[str, Any]]
        The model in eval mode, a seeded input, and ``site``, ``sites``,
        ``final_site``, ``n_blocks`` and the planned operation count.
    """

    kind_name, n_ops, size = parse_fixture_name(name)
    kind = KINDS[kind_name]
    d = SIZES[size]
    n_blocks = max(2, n_ops // kind.ops_per_block)
    torch.manual_seed(0)
    if kind.image:
        channels = max(4, d // 16)
        blocks = [kind.make_block(channels) for _ in range(n_blocks)]
        x = torch.randn(1, channels, 16, 16)
    else:
        blocks = [kind.make_block(d) for _ in range(n_blocks)]
        x = torch.randn(1, 8, d)
    model = FixtureModel(blocks).eval()
    picks = sorted({n_blocks // 4, n_blocks // 2, (3 * n_blocks) // 4, n_blocks - 1})
    meta = {
        "site": f"blocks.{n_blocks // 2}",
        "sites": [f"blocks.{i}" for i in picks],
        "final_site": "head",
        "n_blocks": n_blocks,
        "planned_ops": n_blocks * kind.ops_per_block + 1,
        "kind": kind_name,
        "size": size,
        "fx_traceable": kind_name != "dynamic_branch",
    }
    return model, x, meta


def fixture_matrix(smoke: bool) -> list[str]:
    """Return the fixture subjects of a run (smoke: small tensors, 100 and 1,000 ops)."""

    counts = SMOKE_OP_COUNTS if smoke else OP_COUNTS
    sizes = ("small",) if smoke else tuple(SIZES)
    return [fixture_name(k, n, s) for k in KINDS for n in counts for s in sizes]
