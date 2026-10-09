"""Tiny fixture models for the visual language deck.

Each model is built so that the one idea a slide teaches is the only new mark on its
picture. Inputs are fixed literals (or seeded draws) so every build draws the same graph.
``FIXTURES`` maps a fixture name to a factory returning ``(model, inputs)``; the slide
table in ``slides.py`` refers to fixtures by name only, so it imports without torch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class Tiny(nn.Module):
    """Input, one op, output: the smallest graph."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class SubXY(nn.Module):
    """One order-sensitive op on two inputs."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sub(x, y)


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
    """Trainable, frozen and mixed parameter fills in one chain (short rows: no bias on two)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 3, bias=False)
        self.b = nn.Linear(3, 3, bias=False)
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
    """Returns ``2 * sigmoid(g)`` without touching the input (two ops, so a module box)."""

    def __init__(self) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.zeros(3))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.g) * 2


class Inner(nn.Module):
    """A two-op module: linear then relu."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


class Nested(nn.Module):
    """Two levels of module boxes; one block called twice."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(Inner())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.block(x))


class InnerRes(nn.Module):
    """A residual module: its input feeds two ops inside it."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc(x)


class NestedRes(nn.Module):
    """One residual block, so the collapsed box has a doubled input edge."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(InnerRes())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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
    """One weight-tied cell (no bias, so its rows stay short) applied ``steps`` times."""

    def __init__(self, steps: int = 3) -> None:
        super().__init__()
        self.cell = nn.Linear(3, 3, bias=False)
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for _ in range(self.steps):
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


class Pair(nn.Module):
    """A two-op module, so a collapsed call of it is a 3D box."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) * 2


class LoopShapes(nn.Module):
    """One module called on two different shapes: the rolled box's shape changes."""

    def __init__(self) -> None:
        super().__init__()
        self.pair = Pair()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pair(x), self.pair(x[:, :2])


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
        return x * self.temp


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
    """Fifteen three-op blocks (45 ops): above the readable band, so ``collapse="auto"`` acts."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[TriBlock() for _ in range(15)])

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


class DictOut(nn.Module):
    """A dict output for the container slide."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.fc(x)
        return {"logits": h, "probs": torch.softmax(h, -1)}


class ListOut(nn.Module):
    """Three same-shape outputs in a list: more than an inline limit of two."""

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [x * float(k) for k in range(1, 4)]


class Dead(nn.Module):
    """A value computed from a constant that nobody uses: an orphan."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _unused = torch.ones(3).sin()
        return x * 2


class NanMaker(nn.Module):
    """Produces a NaN: ``log(1 - 2)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x - 2)


class Nonfinite(nn.Module):
    """One output per nonfinite state; ``abs`` is left unsaved, so it is not checked."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        finite = x + 1
        nan = torch.log(x - 2)
        pos_inf = x / 0
        neg_inf = -x / 0
        mixed = torch.log(x - torch.tensor([0.0, 2.0, 1.0]))
        return finite, nan, pos_inf, neg_inf, mixed, x.abs()


class Sizes(nn.Module):
    """Tensors of 4, 16 and 2 elements: sizes and FLOPs that differ node to node."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(4, 16)
        self.down = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


class Grad(nn.Module):
    """A parameter scales the input; the loss is the sum."""

    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.p).sum()


class _DoubleFn(torch.autograd.Function):
    """A custom autograd Function: doubles the value and the gradient."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x * 2

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        return grad * 2


class HigherOrder(nn.Module):
    """A custom Function inside a cubic whose first gradient is differentiated again."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (_DoubleFn.apply(x) ** 3).sum()


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
    "Tiny": _seeded(lambda: (Tiny(), torch.randn(1, 3))),
    "SubXY": _seeded(lambda: (SubXY(), (torch.randn(1, 3), torch.randn(1, 3)))),
    "Flow": _seeded(lambda: (Flow(), torch.randn(1, 4))),
    "FlowReshape": _seeded(lambda: (FlowReshape(), torch.randn(1, 4))),
    "TwoWays": _seeded(lambda: (TwoWays(), torch.randn(1, 4))),
    "Anatomy": _seeded(lambda: (Anatomy(), torch.randn(1, 1, 6, 6))),
    "Params": _seeded(lambda: (Params(), torch.randn(1, 3))),
    "Gate": _seeded(lambda: (Gate(), torch.randn(1, 3))),
    "Nested": _seeded(lambda: (Nested(), torch.randn(1, 3))),
    "NestedRes": _seeded(lambda: (NestedRes(), torch.randn(1, 3))),
    "FanIn": _seeded(lambda: (FanIn(), torch.randn(1, 3))),
    "Loop": _seeded(lambda: (Loop(), torch.randn(1, 3))),
    "Loop2": _seeded(lambda: (Loop(steps=2), torch.randn(1, 3))),
    "LoopGroups": _seeded(lambda: (LoopGroups(), torch.randn(1, 3))),
    "LoopShapes": _seeded(lambda: (LoopShapes(), torch.randn(1, 4))),
    "Branch": _seeded(lambda: (Branch(), torch.ones(1, 3))),
    "Stateful": _seeded(lambda: (Stateful().eval(), torch.randn(4, 3))),
    "Clamp": _seeded(lambda: (Clamp(), torch.randn(1, 3))),
    "Stack": _seeded(lambda: (Stack(), torch.randn(1, 3))),
    "BigStack": _seeded(lambda: (BigStack(), torch.randn(1, 3))),
    "ConvBnRelu2": _seeded(lambda: (ConvBnRelu2().eval(), torch.randn(1, 1, 6, 6))),
    "DictOut": _seeded(lambda: (DictOut(), torch.randn(1, 3))),
    "ListOut": _seeded(lambda: (ListOut(), torch.randn(1, 3))),
    "Dead": _seeded(lambda: (Dead(), torch.randn(1, 3))),
    "NanMaker": _seeded(lambda: (NanMaker(), torch.ones(1, 3))),
    "Nonfinite": _seeded(lambda: (Nonfinite(), torch.ones(3))),
    "Sizes": _seeded(lambda: (Sizes(), torch.randn(1, 4))),
    "Grad": _seeded(lambda: (Grad(), torch.ones(2, 4))),
    "HigherOrder": _seeded(lambda: (HigherOrder(), torch.randn(3, requires_grad=True))),
    "Shared": _seeded(lambda: (Shared(), torch.randn(1, 3))),
    "ImageNet": _seeded(lambda: (ImageNet(), _images())),
}
