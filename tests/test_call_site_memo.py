"""The per-op call-site lookup (``backends/torch/_ops_call_site_fields.py``).

A repeated call site is served from the capture's site table instead of the
full frame walk, and a first-time site is built from the module call's upper
part plus the op's own frames. These tests pin that both return exactly what
the walk returns (code context and call-site grouping) on the shapes the walk
reads (module hooks, stateless modules, module lists, recursion, deep stacks),
that the module record read from the module stack is the wrapper's own, that a
site seen before is never walked again, and that neither the per-op cost nor
the first-time-site cost grows with stack depth.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from benchmarks import capture_core_floor as floor
from benchmarks.capture_core_counters import code_context_lookups
from torchlens.backends.torch import (
    _call_site_parts as parts_module,
    _ops_call_site_fields as site_module,
)


class _NeverHit(dict):
    """A site table that misses every lookup, so every op takes the full walk."""

    def __getitem__(self, key: Any) -> Any:
        raise KeyError(key)


class _WalkEveryOp(site_module._CallSiteMemo):
    """The capture's site table with lookups and the first-time-site builder disabled."""

    build_misses = False

    def __init__(self, cache: dict[Any, Any], module_ws: Any = None) -> None:
        super().__init__(cache, module_ws)
        self.sites = _NeverHit()


class _BuildEveryOp(site_module._CallSiteMemo):
    """The capture's site table with lookups disabled: every op is a first-time site."""

    def __init__(self, cache: dict[Any, Any], module_ws: Any = None) -> None:
        super().__init__(cache, module_ws)
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
    """Every op's layer label, pass and the user-visible fields of its code context."""

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
@pytest.mark.parametrize("harness_depth", [0, 30])
def test_a_first_time_site_built_from_parts_equals_the_full_walk(
    name: str, harness_depth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    model = _MODELS[name]()
    x = torch.randn(2, 4)
    monkeypatch.setattr(site_module, "_CallSiteMemo", _BuildEveryOp)
    built = _projection(_recurse(tl.trace, harness_depth, model, x))
    monkeypatch.setattr(site_module, "_CallSiteMemo", _WalkEveryOp)
    walked = _projection(_recurse(tl.trace, harness_depth, model, x))
    assert built == walked


@pytest.mark.parametrize("name", ["deep", "hooked", "loopcell", "stateless"])
def test_the_module_record_read_from_the_stack_is_the_wrappers_own(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = parts_module.module_record
    checked = []

    def checking(memo: Any, wrapper: Any, depth: int) -> tuple[Any, int]:
        record, next_depth = original(memo, wrapper, depth)
        assert record is wrapper.f_locals.get("frame")
        checked.append(next_depth)
        return record, next_depth

    monkeypatch.setattr(parts_module, "module_record", checking)
    monkeypatch.setattr(site_module, "module_record", checking)
    monkeypatch.setattr(site_module, "_CallSiteMemo", _BuildEveryOp)
    tl.trace(_MODELS[name](), torch.randn(2, 4))
    assert any(next_depth > 0 for next_depth in checked)


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_a_seen_site_is_never_walked_again(name: str) -> None:
    model = _MODELS[name]()
    x = torch.randn(2, 4)
    tl.trace(model, x)
    trace, counts = code_context_lookups(lambda: tl.trace(model, x))
    assert counts["repeat_lookups"] == 0
    assert 0 < counts["lookups"] <= len(trace)


class _LoopN(nn.Module):
    def __init__(self, steps: int) -> None:
        super().__init__()
        self.steps = steps
        self.cell = nn.Linear(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.steps):
            x = torch.tanh(self.cell(x))
        return x


class _DeepN(nn.Module):
    """``_LoopN`` called under ``depth`` extra frames inside the root forward."""

    def __init__(self, steps: int, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.inner = _LoopN(steps)

    def forward(self, x: Tensor) -> Tensor:
        return _recurse(self.inner, self.depth, x)


def _call_site_count(model: nn.Module, x: Tensor, depth: int) -> tuple[int, int]:
    run = lambda: _recurse(tl.trace, depth, model, x)  # noqa: E731
    run()
    run()
    _trace, record = floor.count_sections(run)
    return record["calls"]["call_site"], record["outputs"]


def _per_op(build: Any, depth: int) -> float:
    """Call-site calls per extra op: 40 loop steps against 10, so first-time sites cancel."""

    x = torch.randn(2, 4)
    calls_10, outputs_10 = _call_site_count(build(10), x, depth)
    calls_40, outputs_40 = _call_site_count(build(40), x, depth)
    return (calls_40 - calls_10) / (outputs_40 - outputs_10)


def test_call_site_cost_per_op_does_not_grow_with_stack_depth() -> None:
    """30 extra frames above the capture call, or inside the root forward, add nothing per op."""

    assert _per_op(_LoopN, 30) <= _per_op(_LoopN, 0) + 0.5
    in_model_shallow = _per_op(lambda steps: _DeepN(steps, 0), 0)
    assert _per_op(lambda steps: _DeepN(steps, 30), 0) <= in_model_shallow + 0.5


class _Acts(nn.Module):
    """``count`` distinct stateless modules from one line: every op is a first-time site."""

    def __init__(self, count: int) -> None:
        super().__init__()
        self.acts = nn.ModuleList([nn.ReLU() for _ in range(count)])

    def forward(self, x: Tensor) -> Tensor:
        for act in self.acts:
            x = act(x)
        return x


class _DeepActs(nn.Module):
    """``_Acts`` called under ``depth`` extra frames inside the root forward."""

    def __init__(self, count: int, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.inner = _Acts(count)

    def forward(self, x: Tensor) -> Tensor:
        return _recurse(self.inner, self.depth, x)


def test_first_time_site_cost_does_not_grow_with_stack_depth() -> None:
    """30 extra frames inside the root forward cost the same with 10 or 40 first-time sites."""

    x = torch.randn(2, 4)

    def depth_cost(count: int) -> int:
        deep, _ = _call_site_count(_DeepActs(count, 30), x, 0)
        shallow, _ = _call_site_count(_DeepActs(count, 0), x, 0)
        return deep - shallow

    assert depth_cost(40) <= depth_cost(10) + 30


_PROBE_SOURCE = "def step(probe):\n    return probe()\n"
_RELU_SOURCE = "import torch\n\n\ndef step(x):\n    return torch.relu(x)\n"


def _load_step(path: Any, source: str) -> Any:
    """``step`` compiled from ``source`` as if it lived in file ``path``."""

    path.write_text(source)
    namespace: dict[str, Any] = {}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["step"]


def _value_equal_pair(tmp_path: Any, source: str) -> tuple[Any, Any]:
    """Two functions whose code objects are equal by value but come from two files."""

    step_a = _load_step(tmp_path / "file_a.py", source)
    step_b = _load_step(tmp_path / "file_b.py", source)
    assert step_a.__code__ == step_b.__code__
    assert step_a.__code__ is not step_b.__code__
    assert step_a.__code__.co_filename != step_b.__code__.co_filename
    return step_a, step_b


def test_value_equal_codes_from_two_files_never_share_a_site_key(tmp_path: Any) -> None:
    """Called from one line, two equal codes from different files get different keys."""

    step_a, step_b = _value_equal_pair(tmp_path, _PROBE_SOURCE)
    memo = site_module._CallSiteMemo({})
    keys = []
    for step in (step_a, step_b):
        keys.append(step(lambda: site_module._site_key(memo, sys._getframe(1))))
    assert keys[0] != keys[1]


class _TwoFiles(nn.Module):
    """Runs one op in each of two value-equal functions from different files, from one line."""

    def __init__(self, steps: tuple[Any, Any]) -> None:
        super().__init__()
        self.steps = steps

    def forward(self, x: Tensor) -> Tensor:
        outs = []
        for step in self.steps:
            outs.append(step(x))
        return outs[0] + outs[1]


def test_value_equal_codes_from_two_files_get_separate_sites(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each file's op takes its own site key, and the fields equal the full walk's."""

    steps = _value_equal_pair(tmp_path, _RELU_SOURCE)
    original = site_module._site_key
    seen: dict[str, set[Any]] = {"a": set(), "b": set()}

    def recording(memo: Any, frame: Any) -> Any:
        key = original(memo, frame)
        walk = frame
        while walk is not None:
            for name, step in zip("ab", steps, strict=True):
                if walk.f_code is step.__code__:
                    seen[name].add(key)
                    return key
            walk = walk.f_back
        return key

    model = _TwoFiles(steps)
    x = torch.randn(2, 4)
    monkeypatch.setattr(site_module, "_site_key", recording)
    served = _projection(_recurse(tl.trace, 0, model, x))
    assert seen["a"] and seen["b"]
    assert not seen["a"] & seen["b"]
    monkeypatch.setattr(site_module, "_CallSiteMemo", _WalkEveryOp)
    assert served == _projection(_recurse(tl.trace, 0, model, x))


@pytest.mark.parametrize("memo", [site_module._CallSiteMemo, _BuildEveryOp, _WalkEveryOp])
def test_value_equal_codes_from_two_files_name_their_own_files(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, memo: type
) -> None:
    """Each file's op's code context names its own file, on every lookup path.

    Code equality ignores ``co_filename``: the two ops' kept frames are equal by
    value, line and offset, so a whole-context cache key without the files would
    serve the second op the first file's context.
    """

    steps = _value_equal_pair(tmp_path, _RELU_SOURCE)
    monkeypatch.setattr(site_module, "_CallSiteMemo", memo)
    rows = _projection(_recurse(tl.trace, 0, _TwoFiles(steps), torch.randn(2, 4)))
    innermost = [frames[-1] for _label, _index, frames in rows if frames[-1][2] == "step"]
    files = sorted(file for file, *_rest in innermost)
    assert files == sorted(step.__code__.co_filename for step in steps)
    assert len({(line, col) for _file, line, _name, _qualname, col in innermost}) == 1
