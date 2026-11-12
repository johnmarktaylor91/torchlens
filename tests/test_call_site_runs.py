"""Run skipping in the per-op site-key walk (``backends/torch/_call_site_runs.py``).

The walk skips a known run of TorchLens frames by code identity and names codes
by ``id``. These tests pin that every op's site key is exactly the key the
frame-by-frame walk makes, that served code contexts equal the full walk on
shapes the run table sees (module hooks, recursion, deep stacks, in-place and
multi-output ops), that every code a stored key names is held by the table, and
the run table's own rules on hand-built frames: a full match skips the run, a
partial match learns a new run, the budget caps learning, a changed set of
module wrappers drops every run, and a run never crosses ``_call_impl`` or a
frame named like a module wrapper.
"""

from __future__ import annotations

from types import CodeType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from torchlens._utils._package_frames import _VERDICTS
from torchlens.backends.torch import (
    _call_site_runs as runs_module,
    _ops_call_site_fields as site_module,
)

_TL_FILE = "/fake-site/torchlens/_fake_internal.py"
_USER_FILE = "/fake-site/user/_fake_user.py"


def _recurse(fn: Any, depth: int, *args: Any) -> Any:
    """Call ``fn`` under ``depth`` extra Python frames."""

    return fn(*args) if depth <= 0 else _recurse(fn, depth - 1, *args)


class _LoopCell(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(4):
            x = torch.tanh(self.cell(x))
        return x


class _Hooked(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.body = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Sequential(nn.ReLU()))
        self.first.register_forward_pre_hook(lambda _m, args: (args[0] * 2,))

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(2):
            x = self.body(self.first(x))
        return x


def _helper(x: Tensor, n: int) -> Tensor:
    return torch.sin(x) if n == 0 else _helper(torch.cos(x), n - 1)


class _Mixed(nn.Module):
    """In-place, multi-output and helper-function ops at the root and in a submodule."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.cell = _LoopCell()

    def forward(self, x: Tensor) -> Tensor:
        y = self.lin(x)
        y.add_(1)
        a, b = torch.chunk(y, 2, dim=1)
        values, _indices = torch.topk(a, 1, dim=1)
        z = _helper(b, 2) + values
        for _ in range(3):
            z = torch.relu(z).mul_(0.5)
        return self.cell(torch.cat([z, z], dim=1))


class _Deep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _Mixed()

    def forward(self, x: Tensor) -> Tensor:
        return _recurse(self.inner, 30, torch.relu(x))


_MODELS = {"deep": _Deep, "hooked": _Hooked, "loopcell": _LoopCell, "mixed": _Mixed}


class _NeverHit(dict):
    def __getitem__(self, key: Any) -> Any:
        raise KeyError(key)


class _WalkEveryOp(site_module._CallSiteMemo):
    """The capture's site table with lookups and the first-time-site builder disabled."""

    build_misses = False

    def __init__(self, cache: dict[Any, Any], module_ws: Any = None) -> None:
        super().__init__(cache, module_ws)
        self.sites = _NeverHit()


_SEEN: list[Any] = []


class _Recording(site_module._CallSiteMemo):
    """The capture's site table, kept for inspection after the capture."""

    def __init__(self, cache: dict[Any, Any], module_ws: Any = None) -> None:
        super().__init__(cache, module_ws)
        _SEEN.append(self)


def _projection(trace: Any) -> list[tuple[Any, ...]]:
    rows = []
    for label in trace.layer_labels:
        for index, op in enumerate(trace[label].ops):
            frames = tuple(
                (f.file, f.line_number, f.func_name, f.func_qualname, f.col_offset)
                for f in (op.source.frames or ())
            )
            rows.append((label, index, frames))
    return rows


@pytest.mark.parametrize("name", sorted(_MODELS))
@pytest.mark.parametrize("harness_depth", [0, 30])
def test_every_site_key_equals_the_frame_by_frame_key(
    name: str, harness_depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = site_module._site_key
    compared = []

    def checking(memo: Any, frame: Any) -> Any:
        runs, budget = memo.runs, memo.run_budget
        memo.runs, memo.run_budget = {}, 0
        try:
            plain = original(memo, frame)
        finally:
            memo.runs, memo.run_budget = runs, budget
        key = original(memo, frame)
        assert key == plain
        compared.append(memo)
        return key

    monkeypatch.setattr(site_module, "_site_key", checking)
    torch.manual_seed(0)
    _recurse(tl.trace, harness_depth, _MODELS[name](), torch.randn(2, 4))
    assert compared
    assert compared[-1].runs, "the run table was never used, so the comparison is vacuous"
    assert compared[-1].run_budget >= 0


@pytest.mark.parametrize("name", sorted(_MODELS))
@pytest.mark.parametrize("harness_depth", [0, 30])
def test_served_code_context_equals_the_full_walk(
    name: str, harness_depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    model = _MODELS[name]()
    x = torch.randn(2, 4)
    served = _projection(_recurse(tl.trace, harness_depth, model, x))
    monkeypatch.setattr(site_module, "_CallSiteMemo", _WalkEveryOp)
    walked = _projection(_recurse(tl.trace, harness_depth, model, x))
    assert served == walked


def _key_codes(key: Any) -> list[tuple[int, str]]:
    """``(id, filename)`` of every code a site key names (its ``(key, id, file, offset)`` links)."""

    found = []
    stack = [key]
    while stack:
        item = stack.pop()
        if not isinstance(item, tuple):
            continue
        if len(item) == 4 and isinstance(item[1], int) and isinstance(item[2], str):
            found.append((item[1], item[2]))
        stack.extend(item)
    return found


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_every_code_a_stored_key_names_is_held(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _SEEN.clear()
    monkeypatch.setattr(site_module, "_CallSiteMemo", _Recording)
    tl.trace(_MODELS[name](), torch.randn(2, 4))
    memo = _SEEN[-1]
    named = [link for key in memo.sites for link in _key_codes(key)]
    assert named
    for code_id, filename in named:
        assert memo.pinned[code_id].co_filename == filename


# Run-table rules on hand-built frames.


def _code(name: str, filename: str) -> CodeType:
    return (lambda: None).__code__.replace(co_name=name, co_filename=filename)


class _Frame:
    def __init__(self, code: CodeType, back: _Frame | None) -> None:
        self.f_code = code
        self.f_back = back


def _stack(*codes: CodeType) -> list[_Frame]:
    """Frames running ``codes``, innermost first."""

    frames: list[_Frame] = []
    back = None
    for code in reversed(codes):
        back = _Frame(code, back)
        frames.append(back)
    return frames[::-1]


def _memo(budget: int = runs_module.RUN_BUDGET) -> Any:
    return SimpleNamespace(
        runs={}, run_budget=budget, run_boundaries=len(runs_module._BOUNDARY_CODES)
    )


@pytest.fixture
def files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_VERDICTS, _TL_FILE, True)
    monkeypatch.setitem(_VERDICTS, _USER_FILE, False)


A, B, C, X = (_code(n, _TL_FILE) for n in "abcx")
USER = _code("forward", _USER_FILE)


@pytest.mark.usefixtures("files")
def test_a_learned_run_is_skipped_by_identity() -> None:
    memo = _memo()
    first = _stack(A, B, C, USER)
    assert runs_module.skip_run(memo, first[0], A) is first[3]
    assert memo.runs[id(A)] == [(A, (B, C))]
    second = _stack(A, B, C, USER)
    assert runs_module.skip_run(memo, second[0], A) is second[3]
    assert memo.run_budget == runs_module.RUN_BUDGET - 1


@pytest.mark.usefixtures("files")
def test_a_partial_match_learns_a_new_run() -> None:
    memo = _memo()
    runs_module.skip_run(memo, _stack(A, B, C, USER)[0], A)
    other = _stack(A, B, X, C, USER)
    assert runs_module.skip_run(memo, other[0], A) is other[4]
    shorter = _stack(A, B, USER)
    assert runs_module.skip_run(memo, shorter[0], A) is shorter[2]
    assert [run for _head, run in memo.runs[id(A)]] == [(B, C), (B, X, C), (B,)]


@pytest.mark.usefixtures("files")
def test_past_the_budget_only_the_frame_is_skipped() -> None:
    memo = _memo(budget=0)
    frames = _stack(A, B, USER)
    assert runs_module.skip_run(memo, frames[0], A) is frames[1]
    assert memo.runs == {}


@pytest.mark.usefixtures("files")
def test_a_changed_wrapper_set_drops_every_run() -> None:
    memo = _memo()
    runs_module.skip_run(memo, _stack(A, B, USER)[0], A)
    memo.run_boundaries -= 1
    memo.run_budget = 0
    frames = _stack(A, B, USER)
    assert runs_module.skip_run(memo, frames[0], A) is frames[1]
    assert memo.runs == {}
    assert memo.run_boundaries == len(runs_module._BOUNDARY_CODES)


@pytest.mark.usefixtures("files")
@pytest.mark.parametrize("stop_name", ["_call_impl", "fake_module_wrapper"])
def test_a_run_never_crosses_call_impl_or_a_wrapper_name(
    stop_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = {*runs_module._BOUNDARY_NAMES, "fake_module_wrapper"}
    monkeypatch.setattr(runs_module, "_BOUNDARY_NAMES", names)
    stop = _code(stop_name, _TL_FILE)
    frames = _stack(A, B, stop, C, USER)
    memo = _memo()
    assert runs_module.skip_run(memo, frames[0], A) is frames[2]
    assert memo.runs[id(A)] == [(A, (B,))]
