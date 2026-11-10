"""The per-op call-site lookup (``backends/torch/_ops_call_site_fields.py``).

A repeated call site is served from the capture's site table instead of the
full frame walk. These tests pin that a hit returns exactly what the walk
returns (code context and call-site grouping) on the shapes the walk reads
(module hooks, stateless modules, module lists, recursion, deep stacks), that
a site seen before is never walked again, and that the per-op cost does not
grow with stack depth.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from benchmarks import capture_core_floor as floor
from benchmarks.capture_core_counters import code_context_lookups
from torchlens.backends.torch import _ops_call_site_fields as site_module


class _NeverHit(dict):
    """A site table that misses every lookup, so every op takes the full walk."""

    def __getitem__(self, key: Any) -> Any:
        raise KeyError(key)


class _WalkEveryOp(site_module._CallSiteMemo):
    """The capture's site table with lookups disabled."""

    def __init__(self, cache: dict[Any, Any]) -> None:
        super().__init__(cache)
        self.sites = _NeverHit()


def _recurse(fn: Any, depth: int, *args: Any) -> Any:
    """Call ``fn`` under ``depth`` extra Python frames."""

    return fn(*args) if depth <= 0 else _recurse(fn, depth - 1, *args)


class _Chain(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        for _ in range(6):
            x = torch.relu(x) + 1
        return x


class _LoopCell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(5):
            x = torch.tanh(self.cell(x))
        return x


class _Stateless(nn.Module):
    """One ``nn.ReLU`` from two lines, two ``nn.ReLU`` objects from one loop line."""

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.pair = nn.ModuleList([nn.ReLU(), nn.ReLU()])
        self.lin = nn.Linear(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(x)
        x = self.act(x - 1)
        for _ in range(2):
            for act in self.pair:
                x = act(self.lin(x))
        return x


class _Hooked(nn.Module):
    """An op inside a forward pre-hook, and nested stateful and stateless modules."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)
        self.body = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Sequential(nn.ReLU()))
        for lin in (self.first, self.second):
            lin.register_forward_pre_hook(lambda _m, args: (args[0] * 2,))

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            x = self.body(self.second(self.first(x)))
        return x


def _helper(x: Tensor, n: int) -> Tensor:
    return torch.sin(x) if n == 0 else _helper(torch.cos(x), n - 1)


class _Recursive(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return _helper(x, 3) + _helper(x, 1)


class _Deep(nn.Module):
    """The model's submodules called under 30 extra frames inside the root forward."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _LoopCell()

    def forward(self, x: Tensor) -> Tensor:
        return _recurse(self.inner, 30, torch.relu(x))


_MODELS = {
    "chain": _Chain,
    "loopcell": _LoopCell,
    "stateless": _Stateless,
    "hooked": _Hooked,
    "recursive": _Recursive,
    "deep": _Deep,
}


def _projection(trace: Any) -> list[tuple[Any, ...]]:
    """Every layer's label and the user-visible fields of its code context."""

    rows = []
    for label in trace.layer_labels:
        layer = trace[label]
        frames = tuple(
            (f.file, f.line_number, f.func_name, f.func_qualname, f.col_offset)
            for f in (layer.source.frames or ())
        )
        rows.append((label, frames))
    return rows


@pytest.mark.parametrize("name", sorted(_MODELS))
@pytest.mark.parametrize("harness_depth", [0, 30])
def test_site_table_hit_equals_the_full_walk(
    name: str, harness_depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    model = _MODELS[name]()
    x = torch.randn(2, 4)
    served = _projection(_recurse(tl.trace, harness_depth, model, x))
    monkeypatch.setattr(site_module, "_CallSiteMemo", _WalkEveryOp)
    walked = _projection(_recurse(tl.trace, harness_depth, model, x))
    assert served == walked


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_a_seen_site_is_never_walked_again(name: str) -> None:
    model = _MODELS[name]()
    x = torch.randn(2, 4)
    tl.trace(model, x)
    trace, counts = code_context_lookups(lambda: tl.trace(model, x))
    assert counts["repeat_lookups"] == 0
    assert 0 < counts["lookups"] <= len(trace)


def _call_site_per_output(model: nn.Module, x: Tensor, depth: int) -> float:
    run = lambda: _recurse(tl.trace, depth, model, x)  # noqa: E731
    run()
    run()
    _trace, record = floor.count_sections(run)
    return record["calls"]["call_site"] / record["outputs"]


def test_call_site_cost_does_not_grow_with_stack_depth() -> None:
    x = torch.randn(2, 4)
    shallow = _call_site_per_output(_LoopCell(), x, 0)
    deep_harness = _call_site_per_output(_LoopCell(), x, 30)
    deep_model = _call_site_per_output(_Deep(), x, 0)
    assert deep_harness <= shallow * 1.05
    # ``_Deep`` adds one relu at the root and one module level; its 30 extra
    # frames are walked once per module call at most, never per op.
    assert deep_model <= shallow * 1.5
