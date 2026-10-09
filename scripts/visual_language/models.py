"""Tiny fixture models for the visual language deck.

Each model is built so that the one idea a slide teaches is the only new mark on its
picture. Inputs are fixed literals (or seeded draws) so every build draws the same graph.
``FIXTURES`` maps a fixture name to a factory returning ``(model, inputs)``; the slide
table in ``slides.py`` refers to fixtures by name only, so it imports without torch.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.render_encoding_reference import LockstepDecoder  # noqa: E402


class Flow(nn.Module):
    """Input, one leaf-module box, two function ovals, output."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(x)) * 2


class FlowReshape(nn.Module):
    """``Flow`` with one reshape in the middle, for the hiding slide."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x).reshape(2, 2)
        return torch.tanh(h) * 2


class TwoWays(nn.Module):
    """The same matmul drawn once as a module box and once as a function oval."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.w = nn.Parameter(torch.randn(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x)) + F.linear(x, self.w)


class Anatomy(nn.Module):
    """One convolution that carries every default label row."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Params(nn.Module):
    """Trainable, frozen and mixed parameter fills in one chain."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 3)
        self.b = nn.Linear(3, 3)
        self.c = nn.Linear(3, 3)
        self.b.requires_grad_(False)
        self.c.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(self.b(self.a(x)))


class Gate(nn.Module):
    """A branch computed only from a parameter, inside a one-op submodule."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = GateValue()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate()


class GateValue(nn.Module):
    """Returns ``sigmoid(g)`` without touching the input."""

    def __init__(self) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.zeros(3))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.g)


class Nested(nn.Module):
    """Two levels of module boxes; one block called twice."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
        self.head = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.block(self.block(x)))


class ArgOrder(nn.Module):
    """Argument labels on ``sub``, none on the commutative ``add``, two arrows into ``cat``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(x)
        b = torch.sigmoid(x)
        return torch.sub(a, b) + torch.cat([a, a], dim=0).sum(0)


class FanIn(nn.Module):
    """Four distinct producers stacked into one op: midpoint argument labels."""

    def __init__(self) -> None:
        super().__init__()
        self.l0 = nn.Linear(3, 3)
        self.l1 = nn.Linear(3, 3)
        self.l2 = nn.Linear(3, 3)
        self.l3 = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.l0(x), self.l1(x), self.l2(x), self.l3(x)]).sum(0)


class Loop(nn.Module):
    """One weight-tied cell applied three times."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(3):
            h = torch.relu(self.cell(h))
        return h


class LoopGroups(nn.Module):
    """Two separate two-step runs of the same cell, summed: split call groups."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.cell(self.cell(x))
        b = self.cell(self.cell(x * 2))
        return a + b


class LoopShapes(nn.Module):
    """A shape-polymorphic module called on two different shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide = self.act(x)
        narrow = self.act(x[:, :2])
        return wide.sum() + narrow.sum()


class Branch(nn.Module):
    """A data-dependent if/else; traced with ones so the THEN arm runs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return x * 2
        return x - 1


class Stateful(nn.Module):
    """A meaningful registered buffer and BatchNorm's running statistics."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.full((3,), 2.0))
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x) * self.scale


class Clamp(nn.Module):
    """A Parameter changed in place during forward."""

    def __init__(self) -> None:
        super().__init__()
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.temp.clamp_(0.1, 10.0)
        return x / self.temp


class Block(nn.Module):
    """A residual linear block."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.relu(self.fc(x))


class Stack(nn.Module):
    """Stem, four distinct same-class blocks, head."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(3, 3)
        self.blocks = nn.Sequential(*[Block() for _ in range(4)])
        self.head = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class TriBlock(nn.Module):
    """Three ops per block: linear, tanh, residual add."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.fc(x))


class BigStack(nn.Module):
    """Twelve three-op blocks: above the readable band, so ``collapse="auto"`` acts."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[TriBlock() for _ in range(12)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ConvBnRelu2(nn.Module):
    """Two instances of the conv, batch norm, relu idiom."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3, padding=1)
        self.b1 = nn.BatchNorm2d(2)
        self.c2 = nn.Conv2d(2, 2, 3, padding=1)
        self.b2 = nn.BatchNorm2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))))


class AddRelu(nn.Module):
    """Two ``add`` then ``relu`` runs for a user-declared pattern."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(x + 1)
        return torch.relu(h + 1)


class SmallCnn(nn.Module):
    """Shapes and FLOPs that differ node to node."""

    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 4, 3)
        self.c2 = nn.Conv2d(4, 8, 3)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.c2(F.relu(self.c1(x))))
        return self.fc(F.adaptive_avg_pool2d(h, 1).flatten(1))


class DictOut(nn.Module):
    """A dict output for the container slide."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.fc(x)
        return {"logits": h, "probs": torch.softmax(h, -1)}


class ListOut(nn.Module):
    """Fourteen same-shape outputs in a list: more than the inline limit."""

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x * float(k) for k in range(1, 15)]


class Dead(nn.Module):
    """One computed value nobody uses: an orphan."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _unused = torch.exp(x)
        return x * 2


class NanMaker(nn.Module):
    """Produces a NaN: ``log(1 - 2)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x - 2) + 1


class Nonfinite(nn.Module):
    """One branch per nonfinite state, joined so every branch reaches the output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        finite = x + 1
        nan = torch.log(x - 2)
        pos_inf = x / torch.zeros_like(x)
        neg_inf = -x / torch.zeros_like(x)
        mixed = torch.log(x - torch.tensor([0.0, 2.0, 0.0]))
        return torch.stack([finite, nan, pos_inf, neg_inf, mixed])


class Grad(nn.Module):
    """A parameter scales the input; the loss is a sum of squares."""

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.p).square().sum()


class HigherOrder(nn.Module):
    """A nonlinear scalar whose first gradient is differentiated again."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(x) ** 3).sum()


class _DoubleFn(torch.autograd.Function):
    """A custom autograd Function: doubles the value and the gradient."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x * 2

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        return grad * 2


class CustomGrad(nn.Module):
    """A model whose backward runs a custom autograd Function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DoubleFn.apply(x).sum()


class Shapes(nn.Module):
    """Outputs with different shapes and byte counts, for colour transforms."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        small = x[:, :1].clone()
        wide = x.repeat(1, 4)
        big = self.fc(wide[:, :4]).repeat(4, 1)
        return small.sum() + wide.sum() + big.sum()


class Mut(nn.Module):
    """A Parameter shifted in place before it is read."""

    def __init__(self, frozen: bool = False) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.zeros(3), requires_grad=not frozen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.p.add_(1)
        return x * self.p


class Shared(nn.Module):
    """One Linear called twice, for an edit with a fire and a declared target."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.fc(x)))


class ImageNet(nn.Module):
    """A tiny image classifier for raw input previews and decoded outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 3)
        self.fc = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(F.adaptive_avg_pool2d(self.conv(x), 1).flatten(1))


def _seeded(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap a factory so the model and its inputs are built under seed 0."""

    def factory() -> Any:
        torch.manual_seed(0)
        return fn()

    return factory


def _images() -> list[Any]:
    """Two small solid-colour PIL images (PIL is a TorchLens base dependency)."""

    from PIL import Image

    return [Image.new("RGB", (24, 24), (200, 60, 60)), Image.new("RGB", (24, 24), (40, 140, 220))]


FIXTURES: dict[str, Callable[[], tuple[nn.Module, Any]]] = {
    "Flow": _seeded(lambda: (Flow(), torch.randn(1, 4))),
    "FlowReshape": _seeded(lambda: (FlowReshape(), torch.randn(1, 4))),
    "TwoWays": _seeded(lambda: (TwoWays(), torch.randn(1, 4))),
    "Anatomy": _seeded(lambda: (Anatomy(), torch.randn(1, 1, 6, 6))),
    "Params": _seeded(lambda: (Params(), torch.randn(1, 3))),
    "Gate": _seeded(lambda: (Gate(), torch.randn(1, 3))),
    "Nested": _seeded(lambda: (Nested(), torch.randn(1, 3))),
    "ArgOrder": _seeded(lambda: (ArgOrder(), torch.randn(2, 3))),
    "FanIn": _seeded(lambda: (FanIn(), torch.randn(1, 3))),
    "Loop": _seeded(lambda: (Loop(), torch.randn(1, 3))),
    "LoopGroups": _seeded(lambda: (LoopGroups(), torch.randn(1, 3))),
    "LoopShapes": _seeded(lambda: (LoopShapes(), torch.randn(1, 4))),
    "Branch": _seeded(lambda: (Branch(), torch.ones(1, 3))),
    "Stateful": _seeded(lambda: (Stateful().eval(), torch.randn(4, 3))),
    "StatefulTrain": _seeded(lambda: (Stateful().train(), torch.randn(4, 3))),
    "Clamp": _seeded(lambda: (Clamp(), torch.randn(1, 3))),
    "Stack": _seeded(lambda: (Stack(), torch.randn(1, 3))),
    "BigStack": _seeded(lambda: (BigStack(), torch.randn(1, 3))),
    "ConvBnRelu2": _seeded(lambda: (ConvBnRelu2().eval(), torch.randn(1, 1, 6, 6))),
    "AddRelu": _seeded(lambda: (AddRelu(), torch.randn(1, 3))),
    "SmallCnn": _seeded(lambda: (SmallCnn(), torch.randn(1, 1, 8, 8))),
    "DictOut": _seeded(lambda: (DictOut(), torch.randn(1, 3))),
    "ListOut": _seeded(lambda: (ListOut(), torch.randn(1, 3))),
    "Dead": _seeded(lambda: (Dead(), torch.randn(1, 3))),
    "NanMaker": _seeded(lambda: (NanMaker(), torch.ones(1, 3))),
    "Nonfinite": _seeded(lambda: (Nonfinite(), torch.ones(3))),
    "Grad": _seeded(lambda: (Grad(), torch.ones(2, 4, requires_grad=True))),
    "HigherOrder": _seeded(lambda: (HigherOrder(), torch.randn(3, requires_grad=True))),
    "CustomGrad": _seeded(lambda: (CustomGrad(), torch.ones(3, requires_grad=True))),
    "Shapes": _seeded(lambda: (Shapes(), torch.randn(1, 4))),
    "Mut": _seeded(lambda: (Mut(), torch.randn(1, 3))),
    "MutFrozen": _seeded(lambda: (Mut(frozen=True), torch.randn(1, 3))),
    "Shared": _seeded(lambda: (Shared(), torch.randn(1, 3))),
    "ImageNet": _seeded(lambda: (ImageNet(), _images())),
    "LockstepDecoder": _seeded(lambda: (LockstepDecoder(), torch.randn(1, 8))),
}
