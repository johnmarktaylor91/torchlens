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
from torchlens.backends.torch import (
    _module_entry_site as entry_module,
    _ops_call_site_fields as site_module,
)


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


class _CountingDict(dict):  # type: ignore[type-arg]
    """The memo's entry table, counting lookups that return a known context."""

    def __init__(self) -> None:
        super().__init__()
        self.hits = 0

    def get(self, key: Any, default: Any = None) -> Any:
        value = super().get(key, default)
        if value is not None:
            self.hits += 1
        return value


def _run(
    model: nn.Module, x: Tensor, depth: int, walk_every_entry: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[Any], dict[Any, Any], int]:
    """Trace; return module calls' contexts, the code-context cache and entry-table hits.

    Nothing is wrapped around the walk itself: a wrapper's frame would sit on the
    stack the walk reads. ``walk_every_entry`` makes every module entry take the
    full walk, as before this lookup existed.
    """

    memos: list[Any] = []

    class _Recorded(site_module._CallSiteMemo):
        def __init__(self, cache: dict[Any, Any], module_ws: Any = None) -> None:
            super().__init__(cache, module_ws)
            self.entry_contexts = _CountingDict()
            memos.append(self)

    with monkeypatch.context() as patch:
        patch.setattr(site_module, "_CallSiteMemo", _Recorded)
        if walk_every_entry:
            patch.setattr(entry_module, "_module_entry", lambda *_args: None)
            patch.setattr(entry_module, "_nested", lambda *_args: None)
        trace = _recurse(tl.trace, depth, model, x)
    caches = {id(memo.cache): memo.cache for memo in memos}
    assert len(caches) == 1
    hits = sum(memo.entry_contexts.hits for memo in memos)
    return _module_contexts(trace), _cache_projection(next(iter(caches.values()))), hits


@pytest.mark.parametrize("name", sorted(_MODELS))
@pytest.mark.parametrize("harness_depth", [0, 30])
def test_module_entry_context_and_cache_equal_the_full_walk(
    name: str, harness_depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    model = _MODELS[name]()
    x = torch.randn(2, 4)
    served = _run(model, x, harness_depth, False, monkeypatch)
    walked = _run(model, x, harness_depth, True, monkeypatch)
    assert served[0] == walked[0]
    assert served[1] == walked[1]
    assert any(frames for _address, _index, frames in served[0])


def test_repeated_module_entry_sites_skip_the_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _Looped()
    x = torch.randn(2, 4)
    # The memo's table is under test: without this, nested calls take the incremental path.
    monkeypatch.setattr(entry_module, "_nested", lambda *_args: None)
    served_hits = _run(model, x, 0, False, monkeypatch)[2]
    assert _run(model, x, 0, True, monkeypatch)[2] == 0
    # Six block calls from one loop line share one entry site (after the first op).
    assert served_hits >= 5


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
    served = _run(model, x, 0, False, monkeypatch)
    walked = _run(model, x, 0, True, monkeypatch)
    assert served[0] == walked[0]
    assert served[1] == walked[1]
    # Each block call (and its lin and act) is a new entry site: a key that merged the
    # two value-equal codes would serve the second call's three entries from the first.
    assert served[2] == 0
    # The walk's whole-context cache names each call's own file too, on either path.
    block_files = [frames[-1][0] for address, _index, frames in served[0] if address == "block"]
    assert block_files == [step.__code__.co_filename for step in steps]
