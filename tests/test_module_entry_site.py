"""Module entry's code context through the call-site memo (``_module_entry_site.py``).

Module entry looks its code context up by the memo's interned name of the frames
above the module wrapper instead of walking the stack each time. These tests
pin that every module call's code context, and the capture's code-context cache,
equal what the full walk gives (nested, stateless, hooked, looped and recursive
modules, under a deep harness stack), that repeated entry sites skip the walk,
and that two value-equal code objects from different files never share an entry.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from torchlens._utils.introspection import _CODE_CONTEXT_ANCHOR_KEY
from torchlens.backends.torch import _module_entry_site as entry_module, model_prep


def _recurse(fn: Any, depth: int, *args: Any) -> Any:
    """Call ``fn`` under ``depth`` extra Python frames."""

    return fn(*args) if depth <= 0 else _recurse(fn, depth - 1, *args)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.lin(x)) + x


class _Looped(nn.Module):
    """Blocks from one loop line, a stateless module from two lines, a nested Sequential."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])
        self.act = nn.ReLU()
        self.seq = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Sequential(nn.ReLU()))

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            for block in self.blocks:
                x = block(x)
        x = self.act(x)
        x = self.act(x - 1)
        return self.seq(x)


class _Hooked(nn.Module):
    """A module called from a forward pre-hook and a module with no op inside."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.side = nn.Identity()
        self.first.register_forward_pre_hook(lambda _m, args: (self.side(args[0]) * 2,))

    def forward(self, x: Tensor) -> Tensor:
        return self.first(self.side(x)) + self.first(x)


class _Recursive(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()

    def _step(self, x: Tensor, n: int) -> Tensor:
        return self.block(x) if n == 0 else self._step(self.block(x), n - 1)

    def forward(self, x: Tensor) -> Tensor:
        return self._step(x, 3) + self._step(x, 1)


class _Deep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _Looped()

    def forward(self, x: Tensor) -> Tensor:
        return _recurse(self.inner, 30, torch.relu(x))


_MODELS = {"looped": _Looped, "hooked": _Hooked, "recursive": _Recursive, "deep": _Deep}


def _frames(context: Any) -> tuple[Any, ...]:
    return tuple(
        (f.file, f.line_number, f.func_name, f.func_qualname, f.col_offset) for f in context or ()
    )


def _module_contexts(trace: Any) -> list[tuple[Any, ...]]:
    """Every module call's address, pass and the user-visible fields of its code context."""

    return [
        (module.address, index, _frames(call.code_context))
        for module in trace.modules
        for index, call in enumerate(module.ops.values())
    ]


def _cache_projection(cache: dict[Any, Any]) -> dict[Any, Any]:
    """The code-context cache with the anchor's frame id dropped (it differs per run)."""

    projected: dict[Any, Any] = {}
    for key, value in cache.items():
        if key == _CODE_CONTEXT_ANCHOR_KEY:
            projected[key] = value[1:]
        elif isinstance(value, tuple) and all(hasattr(v, "line_number") for v in value):
            projected[key] = _frames(value)
        else:
            projected[key] = value
    return projected


def _always_walk(trace: Any) -> list[Any]:
    """Module entry as before: the full walk every time."""

    cache = trace.__dict__.get("_code_context_cache")
    if cache is None:
        cache = {}
        trace._code_context_cache = cache
    return entry_module._walk(trace, cache)


def _run(
    model: nn.Module, x: Tensor, depth: int, lookup: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Any], dict[Any, Any], int]:
    """Trace with ``lookup`` as module entry's code-context lookup; return what it read."""

    caches: list[dict[Any, Any]] = []
    walks = [0]
    original_walk = entry_module._walk

    def counted_walk(trace: Any, cache: dict[Any, Any]) -> list[Any]:
        walks[0] += 1
        return original_walk(trace, cache)

    def recording(trace: Any) -> list[Any]:
        result = lookup(trace)
        cache = trace.__dict__["_code_context_cache"]
        if not any(seen is cache for seen in caches):
            caches.append(cache)
        return result

    monkeypatch.setattr(entry_module, "_walk", counted_walk)
    monkeypatch.setattr(model_prep, "module_entry_code_context", recording)
    try:
        trace = _recurse(tl.trace, depth, model, x)
    finally:
        monkeypatch.undo()
    assert len(caches) == 1
    return _module_contexts(trace), _cache_projection(caches[0]), walks[0]


@pytest.mark.parametrize("name", sorted(_MODELS))
@pytest.mark.parametrize("harness_depth", [0, 30])
def test_module_entry_context_and_cache_equal_the_full_walk(
    name: str, harness_depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    model = _MODELS[name]()
    x = torch.randn(2, 4)
    served = _run(model, x, harness_depth, entry_module.module_entry_code_context, monkeypatch)
    walked = _run(model, x, harness_depth, _always_walk, monkeypatch)
    assert served[0] == walked[0]
    assert served[1] == walked[1]
    assert any(frames for _address, _index, frames in served[0])


def test_repeated_module_entry_sites_skip_the_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _Looped()
    x = torch.randn(2, 4)
    _contexts, _cache, served_walks = _run(
        model, x, 0, entry_module.module_entry_code_context, monkeypatch
    )
    _contexts, _cache, all_walks = _run(model, x, 0, _always_walk, monkeypatch)
    assert served_walks < all_walks
    # Six block calls from one loop line share one entry site (after the first op).
    assert all_walks - served_walks >= 5


_STEP_SOURCE = "def step(module, x):\n    return module(x)\n"


def _load_step(path: Any) -> Any:
    """``step`` compiled from one source as if it lived in file ``path``."""

    path.write_text(_STEP_SOURCE)
    namespace: dict[str, Any] = {}
    exec(compile(_STEP_SOURCE, str(path), "exec"), namespace)
    return namespace["step"]


class _TwoFiles(nn.Module):
    """Calls one module from each of two value-equal functions in different files, one line."""

    def __init__(self, steps: tuple[Any, Any]) -> None:
        super().__init__()
        self.steps = steps
        self.block = _Block()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(x)  # the first op creates the memo, so both entries use it
        outs = []
        for step in self.steps:
            outs.append(step(self.block, x))
        return outs[0] + outs[1]


def test_value_equal_codes_from_two_files_get_separate_module_entries(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps = (_load_step(tmp_path / "file_a.py"), _load_step(tmp_path / "file_b.py"))
    assert steps[0].__code__ == steps[1].__code__
    assert steps[0].__code__ is not steps[1].__code__
    model = _TwoFiles(steps)
    x = torch.randn(2, 4)
    served = _run(model, x, 0, entry_module.module_entry_code_context, monkeypatch)
    walked = _run(model, x, 0, _always_walk, monkeypatch)
    assert served[0] == walked[0]
    assert served[1] == walked[1]
    block_files = {
        frames[-1][0] for address, _index, frames in served[0] if address == "block" and frames
    }
    assert block_files == {str(tmp_path / "file_a.py"), str(tmp_path / "file_b.py")}
