"""Module entry's incremental call-site path (``_module_entry_site._nested``) is exact.

A nested module call's code context is read from its parent module call's entry
site plus the frames between the two wrappers. These tests run each model twice,
once as shipped and once with that path switched off (every nested entry then
takes the earlier memo or the full walk), and require identical module enter
and exit events, module-call and op code contexts, layer grouping (which keys
on each op's call-site chain), the code-context cache, and the sharing of
location objects across the cache (the postprocess adopts them by identity).
The models cover a repeated cell, buffers, a forward that mutates an argument
list, forward pre-hooks and hooks that call modules, lambda and
``functools.partial`` callers, recursion, an exception raised inside a nested
module and caught by its parent, generators (one consumed after its creator
returned), a module called twice in one line, and a deep harness stack.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from torchlens._ir.capture_events import CaptureEvents
from torchlens._utils.introspection import _CODE_CONTEXT_ANCHOR_KEY
from torchlens.backends.torch import _module_entry_site as entry_module


def _recurse(fn: Any, depth: int, *args: Any) -> Any:
    """Call ``fn`` under ``depth`` extra Python frames."""

    return fn(*args) if depth <= 0 else _recurse(fn, depth - 1, *args)


class _Cell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.lin(x))) + x


class _LoopCell(nn.Module):
    """One cell called many times from one loop line, inside a nested container."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(_Cell(), _Cell())
        self.cell = _Cell()

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(4):
            x = self.cell(x)
        return self.body(x)


class _Buffers(nn.Module):
    """BatchNorm running stats and a buffer registered during the forward."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.block = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(2, 2, 1, 2)
        if not hasattr(self, "made"):
            self.register_buffer("made", torch.ones(1))
        return self.block(self.bn(self.conv(x))).flatten(1) * self.made


class _Appender(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, items: list[Tensor]) -> Tensor:
        out = self.lin(items[-1])
        items.append(out)  # the forward mutates its argument list
        return out


class _MutatesArgs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.app = _Appender()
        self.outer = nn.Sequential(nn.Linear(4, 4))

    def forward(self, x: Tensor) -> Tensor:
        items = [self.outer(x)]
        self.app(items)
        self.app(items)
        return items[-1] + items[0]


class _Hooks(nn.Module):
    """Pre-hooks and hooks that call other modules, at two nesting levels."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.side = nn.Linear(4, 4)
        self.tail = nn.Tanh()
        self.inner[0].register_forward_pre_hook(lambda _m, args: (self.side(args[0]),))
        self.inner[1].register_forward_hook(lambda _m, _a, out: self.tail(out))

    def forward(self, x: Tensor) -> Tensor:
        return self.inner(x) + self.side(x)


class _Callers(nn.Module):
    """A module called from a lambda, a ``functools.partial`` and a local function."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _Cell()
        self.box = nn.Sequential(_Cell())

    def forward(self, x: Tensor) -> Tensor:
        via_lambda = (lambda t: self.inner(t))(x)  # noqa: PLC3002
        via_partial = functools.partial(self.box, x)()

        def local(t: Tensor) -> Tensor:
            return self.inner(t)

        return via_lambda + via_partial + local(x)


class _Recursive(nn.Module):
    def __init__(self, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        self.lin = nn.Linear(4, 4)
        self.child = _Recursive(depth - 1) if depth > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        return x if self.child is None else self.child(x) + self.child(x)


class _Raises(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        self.lin(x)
        raise ValueError("inside a nested module")


class _Catches(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raiser = _Raises()
        self.after = _Cell()

    def forward(self, x: Tensor) -> Tensor:
        try:
            self.raiser(x)
        except ValueError:
            x = self.after(x)
        return self.after(x)


class _Generators(nn.Module):
    """Modules called inside a generator expression and inside a generator run later."""

    def __init__(self) -> None:
        super().__init__()
        self.cells = nn.ModuleList([_Cell() for _ in range(3)])
        self.maker = _GenMaker(self.cells)

    def forward(self, x: Tensor) -> Tensor:
        total = sum(cell(x) for cell in self.cells)
        gen = self.maker(x)  # the generator runs after ``maker``'s forward returned
        return total + sum(gen)


class _GenMaker(nn.Module):
    def __init__(self, cells: nn.ModuleList) -> None:
        super().__init__()
        self.cells = [cells]

    def forward(self, x: Tensor) -> Any:
        return (cell(x) for cell in self.cells[0])


class _TwiceOneLine(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = _Cell()
        self.wrap = nn.Sequential(_Cell())

    def forward(self, x: Tensor) -> Tensor:
        return self.cell(self.cell(x)) + self.wrap(self.wrap(x))


class _Deep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _LoopCell()

    def forward(self, x: Tensor) -> Tensor:
        return _recurse(self.inner, 20, torch.relu(x))


_MODELS: dict[str, Any] = {
    "loop_cell": (_LoopCell, (2, 4)),
    "buffers": (_Buffers, (2, 4)),
    "mutates_args": (_MutatesArgs, (2, 4)),
    "hooks": (_Hooks, (2, 4)),
    "callers": (_Callers, (2, 4)),
    "recursive": (_Recursive, (2, 4)),
    "catches": (_Catches, (2, 4)),
    "generators": (_Generators, (2, 4)),
    "twice_one_line": (_TwiceOneLine, (2, 4)),
    "deep": (_Deep, (2, 4)),
}


def _frames(context: Any) -> tuple[Any, ...]:
    return tuple(
        (f.file, f.line_number, f.func_name, f.func_qualname, f.col_offset, f.code_firstlineno)
        for f in context or ()
    )


def _event(event: Any) -> tuple[Any, ...]:
    """A module enter or exit event with its clock fields dropped and contexts projected."""

    values = []
    for field in dataclasses.fields(event):
        if field.name in ("forward_start_time", "forward_duration"):
            continue
        value = getattr(event, field.name)
        values.append((field.name, _frames(value) if field.name == "code_context" else repr(value)))
    return (type(event).__name__, tuple(values))


def _sharing(cache: dict[Any, Any]) -> tuple[Any, ...]:
    """Which cache entries share location objects: each location's first-appearance index."""

    first: dict[int, int] = {}
    shape = []
    for key, value in cache.items():
        if key == _CODE_CONTEXT_ANCHOR_KEY or not isinstance(value, tuple):
            continue
        if not all(hasattr(v, "line_number") for v in value):
            continue
        shape.append(tuple(first.setdefault(id(v), len(first)) for v in value))
    return tuple(shape)


def _named(obj: Any, modules: dict[int, str]) -> Any:
    """``obj`` with each module id (each run builds its own model) replaced by the module's name."""

    if isinstance(obj, (tuple, list)):
        return type(obj)(_named(item, modules) for item in obj)
    if type(obj) is int and obj in modules:
        return ("module", modules[obj])
    return obj


def _cache(cache: dict[Any, Any], modules: dict[int, str]) -> dict[Any, Any]:
    projected: dict[Any, Any] = {}
    for key, value in cache.items():
        if key == _CODE_CONTEXT_ANCHOR_KEY:
            projected[key] = value[1:]
        elif isinstance(value, tuple) and all(hasattr(v, "line_number") for v in value):
            projected[_named(key, modules)] = _frames(value)
        else:
            projected[_named(key, modules)] = _named(value, modules)
    return projected


def _run(
    name: str, harness_depth: int, incremental: bool, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Trace one model; return everything the incremental path could change, and its hits."""

    torch.manual_seed(0)
    factory, shape = _MODELS[name]
    model = factory()
    x = torch.randn(*shape)
    events: list[Any] = []
    hits = [0]
    orig_enter, orig_exit = CaptureEvents.append_module_enter, CaptureEvents.append_module_exit
    orig_nested = entry_module._nested

    def nested(*args: Any) -> Any:
        found = orig_nested(*args) if incremental else None
        hits[0] += found is not None
        return found

    def enter(self: Any, event: Any) -> None:
        events.append(event)
        orig_enter(self, event)

    def exit_(self: Any, event: Any) -> None:
        events.append(event)
        orig_exit(self, event)

    with monkeypatch.context() as patch:
        patch.setattr(entry_module, "_nested", nested)
        patch.setattr(CaptureEvents, "append_module_enter", enter)
        patch.setattr(CaptureEvents, "append_module_exit", exit_)
        trace = _recurse(tl.trace, harness_depth, model, x)
    cache = trace.__dict__.get("_code_context_cache") or {}
    return {
        "events": [_event(e) for e in events],
        "module_calls": [
            (module.address, index, _frames(call.code_context))
            for module in trace.modules
            for index, call in enumerate(module.ops.values())
        ],
        "ops": [(op.label, _frames(op.source.frames)) for op in trace.ops.values()],
        "layers": [(layer.layer_label, layer.num_passes) for layer in trace.layers.values()],
        "cache": _cache(cache, {id(m): name for name, m in model.named_modules()}),
        "sharing": _sharing(cache),
        "hits": hits[0],
    }


@pytest.mark.parametrize("name", sorted(_MODELS))
@pytest.mark.parametrize("harness_depth", [0, 25])
def test_incremental_module_entry_equals_the_earlier_paths(
    name: str, harness_depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    served = _run(name, harness_depth, True, monkeypatch)
    earlier = _run(name, harness_depth, False, monkeypatch)
    for part in ("events", "module_calls", "ops", "layers", "cache", "sharing"):
        assert served[part] == earlier[part], part
    assert any(frames for _address, _index, frames in served["module_calls"])


# Models whose nested sites repeat within one trace (a loop): a first-time site is a
# code-context cache miss and takes the full walk, so recursion and a line calling one
# module twice (each call a new site) serve nothing incrementally in a single trace.
@pytest.mark.parametrize("name", ["loop_cell", "deep", "buffers"])
def test_nested_entries_take_the_incremental_path(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(name, 0, True, monkeypatch)["hits"] > 0


def test_a_module_called_from_a_hook_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """``side`` runs inside ``inner[0]``'s pre-hook: the hook scope is never served incrementally."""

    seen: list[Any] = []
    orig_nested = entry_module._nested

    def nested(trace: Any, cache: Any, sites: Any, stack: Any, wrapper: Any) -> Any:
        found = orig_nested(trace, cache, sites, stack, wrapper)
        seen.append((stack[-1].address, found is not None))
        return found

    monkeypatch.setattr(entry_module, "_nested", nested)
    tl.trace(_Hooks(), torch.randn(2, 4))
    assert ("side", True) not in seen
    assert any(address == "side" for address, _found in seen)
