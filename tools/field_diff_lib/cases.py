"""The fixed case list of the field differential.

Each case builds its model on clean torch, captures it and returns the
traces to dump. The five capture-core benchmark models run under default
options; every option variant runs on the two real models and on short forms
of the three synthetic ones, because the variants exercise option paths, not
length.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

SEED = 20261109


class Chain(nn.Module):
    """``n`` additions in a row, no modules."""

    def __init__(self, n: int) -> None:
        """Remember the chain length."""

        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add one ``n`` times."""

        for _ in range(self.n):
            x = x + 1
        return x


class Varied(nn.Module):
    """``n`` distinct linear layers, each followed by tanh."""

    def __init__(self, n: int) -> None:
        """Build the layers."""

        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every layer in order."""

        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


class LoopCell(nn.Module):
    """One cell applied ``n`` times: the recurrence workload."""

    def __init__(self, n: int) -> None:
        """Build the cell."""

        super().__init__()
        self.n = n
        self.cell = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the cell ``n`` times with a running state."""

        h = torch.zeros_like(x)
        for _ in range(self.n):
            h = torch.tanh(self.cell(x) + h)
        return h


class Branchy(nn.Module):
    """A forward with a Python conditional on a tensor."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the sign of the hidden sum."""

        h = self.fc(x)
        if h.sum() > 0:
            return torch.relu(h) * 2
        return torch.sigmoid(h) - 1


class InPlaceBuffers(nn.Module):
    """In-place ops, an indexed write, and writes to registered buffers."""

    def __init__(self) -> None:
        """Build a batch norm (train mode writes its buffers) and a counter buffer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.act = nn.ReLU(inplace=True)
        self.register_buffer("seen", torch.zeros(1))
        self.register_buffer("scale", torch.full((4,), 2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix in-place tensor methods with buffer reads and writes."""

        h = self.fc(x)
        h = self.bn(h)
        h = self.act(h)
        h.add_(1.0)
        h[0] = 0.0
        self.seen += 1
        h = h * self.scale
        h.mul_(self.seen)
        return h.view(-1, 2).sum(dim=1)


class SparseModel(nn.Module):
    """Sparse tensors as an operand and as an intermediate."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply by a sparse matrix and convert to and from a sparse layout."""

        indices = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        sparse = torch.sparse_coo_tensor(indices, values, (4, 4))
        h = torch.sparse.mm(sparse, self.fc(x).t()).t()
        return torch.relu(h).to_sparse().to_dense() + 1


class AutocastModel(nn.Module):
    """A region under CPU autocast."""

    def __init__(self) -> None:
        """Build two linear layers."""

        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the first layer under autocast and the second outside it."""

        with torch.autocast("cpu", dtype=torch.bfloat16):
            h = torch.relu(self.fc1(x))
        return self.fc2(h.float())


class TransformModel(nn.Module):
    """A ``torch.func`` transform region."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``vmap`` over the batch and a ``grad`` of a scalar function."""

        h = self.fc(x)
        doubled = torch.func.vmap(lambda row: torch.tanh(row) * 2)(h)
        slope = torch.func.grad(lambda v: (v * v).sum())(h)
        return doubled + slope


class TwoRelu(nn.Module):
    """Two relus in a row: their outputs are equal in content and distinct as objects."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply relu twice, then a view and an addition."""

        first = torch.relu(self.fc(x))
        second = torch.relu(first)
        return second.view(-1) + 0


def _resnet18() -> tuple[nn.Module, Any]:
    """Build an untrained ResNet-18 and its input."""

    import torchvision

    return torchvision.models.resnet18(weights=None).eval(), torch.randn(1, 3, 224, 224)


def _gpt2_tiny4() -> tuple[nn.Module, Any]:
    """Build a four-layer random GPT-2 and its token input."""

    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_layer=4, n_head=4, n_embd=64, vocab_size=1000)
    return GPT2LMHeadModel(config).eval(), torch.randint(0, 1000, (1, 32))


#: Model builders, each returning ``(model, input)``.
MODELS: dict[str, Callable[[], tuple[nn.Module, Any]]] = {
    "chain1000": lambda: (Chain(1000), torch.zeros(4)),
    "varied200": lambda: (Varied(200).eval(), torch.randn(2, 16)),
    "loopcell100": lambda: (LoopCell(100).eval(), torch.randn(2, 16)),
    "resnet18": _resnet18,
    "gpt2_tiny4": _gpt2_tiny4,
    "chain40": lambda: (Chain(40), torch.zeros(4)),
    "varied8": lambda: (Varied(8).eval(), torch.randn(2, 16)),
    "loopcell12": lambda: (LoopCell(12).eval(), torch.randn(2, 16)),
    "conditional": lambda: (Branchy().eval(), torch.randn(2, 4)),
    "inplace_buffers": lambda: (InPlaceBuffers().train(), torch.randn(6, 4)),
    "sparse": lambda: (SparseModel().eval(), torch.randn(4, 4)),
    "autocast": lambda: (AutocastModel().eval(), torch.randn(2, 8)),
    "transform": lambda: (TransformModel().eval(), torch.randn(3, 4)),
    "two_relu": lambda: (TwoRelu().eval(), torch.randn(2, 4)),
}

#: The models every option variant runs on, with the layer filter of its selective save.
VARIANT_MODELS: dict[str, str] = {
    "chain40": "add",
    "varied8": "tanh",
    "loopcell12": "tanh",
    "resnet18": "relu",
    "gpt2_tiny4": "add",
}


@dataclass(frozen=True)
class Case:
    """One case of the differential.

    Parameters
    ----------
    name:
        Case identifier, used as the dump file name.
    model:
        Key into ``MODELS``.
    run:
        ``(tl, model, input, scratch folder) -> {trace name: trace}``.
    mutation_probe:
        Whether the dump also records which tensor fields change when each
        saved output is written in place.
    """

    name: str
    model: str
    run: Callable[[Any, nn.Module, Any, str], dict[str, Any]]
    mutation_probe: bool = False


def _capture(tl: Any, **options: Any) -> Any:
    """Return capture options with the fixed seed."""

    return tl.options.CaptureOptions(seed=SEED, **options)


def _default(tl: Any, model: nn.Module, x: Any, scratch: str) -> dict[str, Any]:
    """Capture under default options."""

    return {"trace": tl.trace(model, x, capture=_capture(tl))}


def _save_none(tl: Any, model: nn.Module, x: Any, scratch: str) -> dict[str, Any]:
    """Capture metadata only."""

    return {"trace": tl.trace(model, x, save="none", capture=_capture(tl))}


def _selective(layer_filter: str) -> Callable[[Any, nn.Module, Any, str], dict[str, Any]]:
    """Capture saving only the layers whose label contains ``layer_filter``."""

    def run(tl: Any, model: nn.Module, x: Any, scratch: str) -> dict[str, Any]:
        return {"trace": tl.trace(model, x, save=[layer_filter], capture=_capture(tl))}

    return run


def _save_args(tl: Any, model: nn.Module, x: Any, scratch: str) -> dict[str, Any]:
    """Capture with argument saving."""

    return {"trace": tl.trace(model, x, capture=_capture(tl, save_args=True))}


def _no_recurrence(tl: Any, model: nn.Module, x: Any, scratch: str) -> dict[str, Any]:
    """Capture with recurrence detection off."""

    return {"trace": tl.trace(model, x, capture=_capture(tl, recurrence_detection=False))}


def _fork(tl: Any, model: nn.Module, x: Any, scratch: str) -> dict[str, Any]:
    """Capture, fork, and zero the relus on the fork."""

    parent = tl.trace(model, x, capture=_capture(tl))
    fork = parent.fork()
    fork.do(tl.func("relu"), tl.zero_ablate())
    return {"parent": parent, "fork": fork}


def _backward(tl: Any, model: nn.Module, x: Any, scratch: str) -> dict[str, Any]:
    """Capture with gradients armed and run one backward pass."""

    trace = tl.trace(model, x, capture=_capture(tl, backward_ready=True, save_grads="all"))
    trace.backward(trace.output.sum())
    return {"trace": trace}


def _saved_loaded(tl: Any, model: nn.Module, x: Any, scratch: str) -> dict[str, Any]:
    """Capture, save to disk and load."""

    trace = tl.trace(model, x, capture=_capture(tl))
    path = os.path.join(tempfile.mkdtemp(dir=scratch), "case.tlspec")
    trace.save(path)
    return {"live": trace, "loaded": tl.load(path)}


def case_list() -> list[Case]:
    """Return the fixed case list, in the order it is run."""

    cases = [
        Case(f"default-{m}", m, _default)
        for m in ("chain1000", "varied200", "loopcell100", "resnet18", "gpt2_tiny4")
    ]
    for model, layer_filter in VARIANT_MODELS.items():
        cases.append(Case(f"save_none-{model}", model, _save_none))
        cases.append(Case(f"selective-{model}", model, _selective(layer_filter)))
        cases.append(Case(f"save_args-{model}", model, _save_args))
        cases.append(Case(f"no_recurrence-{model}", model, _no_recurrence))
    cases += [
        Case("conditional", "conditional", _default),
        Case("inplace_buffers", "inplace_buffers", _default),
        Case("inplace_buffers-save_args", "inplace_buffers", _save_args),
        Case("sparse", "sparse", _default),
        Case("autocast", "autocast", _default),
        Case("transform", "transform", _default),
        Case("fork", "two_relu", _fork),
        Case("backward", "conditional", _backward),
        Case("saved_loaded", "conditional", _saved_loaded),
        Case("saved_loaded-varied8", "varied8", _saved_loaded),
        Case("two_relu", "two_relu", _default, mutation_probe=True),
    ]
    return cases


def build_model(name: str) -> tuple[nn.Module, Any]:
    """Build one model and its input under the fixed seed."""

    torch.manual_seed(SEED)
    return MODELS[name]()
