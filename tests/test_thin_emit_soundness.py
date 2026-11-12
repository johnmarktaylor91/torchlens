"""The thin record path against today's emit, both paths captured and every public field compared.

``_thin_emit`` logs an admitted plain call (one bare tensor output in a plain
session) without today's draft, freeze and view. Each case here captures the same
seeded forward twice, with the thin path on and with it forced off
(``set_thin_emit_enabled(False)``: every call takes today's emit, the reference),
and compares the field differential's whole surface (labels, call ids,
version-counter verdicts, alias semantics, buffer records, reprs and tables), the
model state both forwards leave, and the fast-lane admission count. The direct
tests pin the folded admission check and the layering contract.
"""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.test_plain_emit_soundness import (
    _Body,
    _Buffers,
    _Chain,
    _Cond,
    _Loop,
    _Nested,
    _Raises,
    _x,
)
from torchlens._postprocess._step_views import MIN_COLUMN_OPS
from torchlens.backends.torch import _call_plan, _ops_exhaustive, _thin_emit, _thin_settle

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from field_diff_lib.cases import SEED  # noqa: E402
from field_diff_lib.compare import diff  # noqa: E402
from field_diff_lib.mask import Mask  # noqa: E402
from field_diff_lib.surface import snapshot_trace  # noqa: E402

Build = Callable[[], nn.Module]


def _dump(trace: Any) -> dict[str, Any]:
    """Return the trace's canonical field-differential dump, process-local names removed."""

    dump, _canon = snapshot_trace(trace, [(str(REPO_ROOT), "<repo>")], Mask(enabled=True))
    dump["trace"]["event_stream"]["fields"].pop("run_nonce", None)
    text = json.dumps(dump, default=repr, sort_keys=True).replace(trace.name, "<trace>")
    return json.loads(text)


@pytest.fixture
def thin(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """Count thin hits and fast-lane commits; restore the process's thin setting after."""

    counts = {"thin": 0, "fast": 0}
    real_thin = _ops_exhaustive.bound_thin_emit()

    def counted_thin(trace: Any, call: tuple[Any, ...], func_call_id: int) -> bool:
        served = real_thin(trace, call, func_call_id)
        counts["thin"] += served
        return served

    real_fast = _call_plan.commit_fast_call

    def counted_fast(*args: Any) -> Any:
        counts["fast"] += 1
        return real_fast(*args)

    monkeypatch.setitem(_ops_exhaustive._BOUND_THIN, 0, counted_thin)
    monkeypatch.setattr(_call_plan, "commit_fast_call", counted_fast)
    previous = _thin_emit.thin_emit_enabled()
    yield counts
    _thin_emit.set_thin_emit_enabled(previous)


def _capture(build: Build, x: torch.Tensor, on: bool, **kwargs: Any) -> tuple[Any, nn.Module]:
    """Trace a freshly built, seeded model with the thin path on or off."""

    _thin_emit.set_thin_emit_enabled(on)
    torch.manual_seed(SEED)
    model = build()
    capture = kwargs.pop("capture", None) or tl.options.CaptureOptions(seed=SEED)
    return tl.trace(model, x.clone(), capture=capture, **kwargs), model


def _check(counts: dict[str, int], build: Build, x: torch.Tensor, **kwargs: Any) -> tuple[int, Any]:
    """Return the thin hits and the thin trace, after checking both paths agree."""

    _capture(build, x, True, **dict(kwargs))  # warm the process-wide caches
    counts["thin"] = counts["fast"] = 0
    on, on_model = _capture(build, x, True, **dict(kwargs))
    hits, fast_on = counts["thin"], counts["fast"]
    on_dump = _dump(on)
    counts["thin"] = counts["fast"] = 0
    off, off_model = _capture(build, x, False, **dict(kwargs))
    assert counts["thin"] == 0
    assert counts["fast"] == fast_on, "admission counts differ between the paths"
    assert diff(_dump(off), on_dump) == []
    on_state, off_state = on_model.state_dict(), off_model.state_dict()
    assert on_state.keys() == off_state.keys()
    for key, value in on_state.items():
        assert torch.equal(value, off_state[key]), key
    return hits, on


def _body(fn: Callable[[nn.Module, torch.Tensor], Any]) -> Build:
    return lambda: _Body(fn)


def test_inplace_ops(thin: dict[str, int]) -> None:
    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = m.lin(x) * 2
        y.relu_()
        y.add_(1)
        y += x
        return y * m.weight

    hits, trace = _check(thin, _body(body), _x())
    assert hits > 0
    assert any(op.is_inplace for op in trace.ops)


def test_aliasing_views(thin: dict[str, int]) -> None:
    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x + 1
        v = y.view(-1)
        t = y.t()
        v.mul_(2)
        d = y.detach()
        return t.sum() + v[1:4].sum() + d.mean() + y.expand(2, 2, 3).sum()

    hits, trace = _check(thin, _body(body), _x())
    assert hits > 0
    assert [op for op in trace.ops if op.func_name in ("view", "t")]


def test_tensor_mutated_after_its_op(thin: dict[str, int]) -> None:
    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        a = x * 2
        b = a + 1
        a.add_(5)
        c = a * b
        a.zero_()
        return c + a

    hits, _trace = _check(thin, _body(body), _x())
    assert hits > 0


def test_nested_modules_and_repeated_calls(thin: dict[str, int]) -> None:
    hits, trace = _check(thin, _Nested, _x())
    assert hits > 0
    assert any(len(op.modules) > 1 for op in trace.ops if op.modules)


def test_loops(thin: dict[str, int]) -> None:
    hits, _trace = _check(thin, _Loop, _x())
    assert hits > 0


def test_loopcell100(thin: dict[str, int]) -> None:
    """The gate model with a hundred passes of one cell."""

    sys.path.insert(0, str(REPO_ROOT))
    from benchmarks import capture_core_cost as cost

    model, x = cost.build("loopcell100")
    hits, trace = _check(thin, lambda: model, x)
    assert hits > 100
    assert len(trace.ops) >= hits


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_conditionals(thin: dict[str, int], sign: float) -> None:
    hits, _trace = _check(thin, lambda: _Cond(), _x().abs() * sign + sign)
    assert hits > 0


def test_buffers(thin: dict[str, int]) -> None:
    hits, _trace = _check(thin, _Buffers, _x())
    assert hits > 0


def test_backward(thin: dict[str, int]) -> None:
    """A gradient-saving session keeps alias contracts: the thin path declines every call."""

    capture = tl.options.CaptureOptions(seed=SEED, save_grads="all")
    hits, trace = _check(thin, _Nested, _x(), capture=capture)
    assert hits == 0
    assert trace.ops


def test_backward_after_a_thin_capture(thin: dict[str, int]) -> None:
    """Gradients through a forward the thin path logged match today's emit's."""

    grads = []
    for on in (True, False):
        _thin_emit.set_thin_emit_enabled(on)
        torch.manual_seed(SEED)
        model = _Nested()
        x = _x().requires_grad_(True)
        tl.trace(model, x, capture=tl.options.CaptureOptions(seed=SEED))
        model(x).sum().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for left, right in zip(*grads, strict=True):
        assert torch.equal(left, right)


def test_interventions(thin: dict[str, int]) -> None:
    intervene = tl.when(tl.func("relu"), tl.zero_ablate())
    _hits, trace = _check(thin, _Nested, _x(), intervene=intervene)
    assert any(op.func_name == "relu" for op in trace.ops)


def test_halt(thin: dict[str, int]) -> None:
    def halt_on_relu(ctx: Any) -> bool:
        return ctx.kind == "op" and ctx.func_name == "relu"

    _hits, trace = _check(thin, _Nested, _x(), stop_after=halt_on_relu)
    assert trace.is_stopped


def test_exception_mid_forward(thin: dict[str, int]) -> None:
    """A forward that raises after some ops raises the same error on both paths."""

    errors = []
    for on in (True, False):
        with pytest.raises(Exception) as caught:
            _capture(_Raises, _x(), on)
        errors.append((type(caught.value).__name__, str(caught.value)))
    assert errors[0] == errors[1]
    assert "planted failure mid-forward" in errors[0][1]
    hits, _trace = _check(thin, _Nested, _x())
    assert hits > 0


@pytest.mark.parametrize("length", [1, 2, MIN_COLUMN_OPS - 1, MIN_COLUMN_OPS, 400])
def test_both_sides_of_min_column_ops(thin: dict[str, int], length: int) -> None:
    """Below and at the column postprocess threshold, and well past it."""

    hits, trace = _check(thin, lambda: _Chain(length), _x())
    assert hits == length
    assert len(trace.ops) >= length


def test_save_and_load(thin: dict[str, int], tmp_path: Path) -> None:
    """The saved trace of the thin path loads to the same fields as today's emit's."""

    dumps = []
    for on in (True, False):
        trace, _model = _capture(_Nested, _x(), on)
        path = tmp_path / f"thin-{on}.tlspec"
        trace.save(str(path))
        dumps.append(_dump(tl.load(str(path))))
    assert diff(dumps[1], dumps[0]) == []


def test_session_facts_decline(thin: dict[str, int]) -> None:
    """Each plain-session fact on its own sends every call to today's emit."""

    class Facts:
        track_device_memory = False
        _options_aten_ops = "off"
        _options_on_nonfinite = None

    never = lambda _trace: False  # noqa: E731
    assert _thin_settle.plain_session(Facts(), never)
    for name, value in (
        ("track_device_memory", True),
        ("_predicate_save_options", object()),
        ("_options_aten_ops", "values"),
        ("track_nonfinite", True),
        ("_echo_session", object()),
        ("_options_on_nonfinite", "raise"),
    ):
        facts = Facts()
        setattr(facts, name, value)
        assert not _thin_settle.plain_session(facts, never), name
    assert not _thin_settle.plain_session(Facts(), lambda _trace: True)


def test_thin_modules_import_nothing_from_the_capture_cycle() -> None:
    """The ``_thin_*`` modules import only modules outside the capture import cycle."""

    allowed = {
        "..._capture.plan",
        "..._ir._row_journal",
        "..._utils.display",
        "..._utils._internal_read",
        "._ops_freeze_refs",
        "._ops_row_journal",
        "._thin_record",
        "._thin_settle",
        ".aliasing",
    }
    package = REPO_ROOT / "torchlens" / "backends" / "torch"
    for name in ("_thin_emit.py", "_thin_record.py", "_thin_settle.py"):
        tree = ast.parse((package / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                module = "." * node.level + (node.module or "")
                in_type_block = any(
                    isinstance(parent, ast.If) and node in ast.walk(parent)
                    for parent in tree.body
                    if isinstance(parent, ast.If)
                )
                assert in_type_block or module in allowed, (name, module)
