"""The plain-session emit of one bare tensor output against the generic per-output loop.

``_ops_exhaustive.emit_plain_output`` runs the generic loop's steps for a call whose
raw output is one exact tensor, in a session that asks for none of the per-op
extras. Each capture here is compared with a capture of the same forward that
takes the generic loop for every call (``plain_single_output`` answering False),
over the field differential's whole surface (labels, call ids, version-counter
verdicts, alias semantics, buffer records, reprs and tables), and the model state
the two forwards leave must match. The two direct tests pin the bare-output alias
check and the top-level argument hash tokens to the generic helpers they shortcut.
The thin emit (``_thin_emit``) is off here: it serves the same calls ahead of
``emit_plain_output`` and has its own two-path test (``test_thin_emit_soundness``).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._ir.intervention import FunctionEventInput
from torchlens.backends.torch import _arg_hash, _ops_exhaustive, _thin_emit, aliasing

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from field_diff_lib.cases import SEED  # noqa: E402
from field_diff_lib.compare import diff  # noqa: E402
from field_diff_lib.mask import Mask  # noqa: E402
from field_diff_lib.surface import snapshot_trace  # noqa: E402

Build = Callable[[], nn.Module]


@pytest.fixture(autouse=True)
def _thin_emit_off() -> Any:
    """Send every admitted call to today's emit, whose plain-output body is under test."""

    previous = _thin_emit.set_thin_emit_enabled(False)
    yield
    _thin_emit.set_thin_emit_enabled(previous)


def _dump(trace: Any) -> dict[str, Any]:
    """Return the trace's canonical field-differential dump, process-local names removed."""

    dump, _canon = snapshot_trace(trace, [(str(REPO_ROOT), "<repo>")], Mask(enabled=True))
    dump["trace"]["event_stream"]["fields"].pop("run_nonce", None)
    text = json.dumps(dump, default=repr, sort_keys=True).replace(trace.name, "<trace>")
    return json.loads(text)


def _capture(build: Build, x: torch.Tensor, **kwargs: Any) -> tuple[Any, nn.Module]:
    """Trace a freshly built, seeded model and return the trace and the model."""

    torch.manual_seed(SEED)
    model = build()
    capture = kwargs.pop("capture", None) or tl.options.CaptureOptions(seed=SEED)
    return tl.trace(model, x.clone(), capture=capture, **kwargs), model


def _check(
    monkeypatch: pytest.MonkeyPatch, build: Build, x: torch.Tensor, **kwargs: Any
) -> tuple[int, Any]:
    """Return the plain emit's call count and the plain trace, after checking it against the generic one."""

    calls = [0]
    real = _ops_exhaustive.emit_plain_output

    def spy(*args: Any) -> None:
        calls[0] += 1
        real(*args)

    with monkeypatch.context() as patch:
        patch.setattr(_ops_exhaustive, "emit_plain_output", spy)
        _capture(build, x, **dict(kwargs))
    plain, plain_model = _capture(build, x, **dict(kwargs))
    plain_dump = _dump(plain)
    with monkeypatch.context() as patch:
        patch.setattr(_ops_exhaustive, "plain_single_output", lambda *_: False)
        generic, generic_model = _capture(build, x, **dict(kwargs))
        generic_dump = _dump(generic)
    assert diff(generic_dump, plain_dump) == []
    plain_state, generic_state = plain_model.state_dict(), generic_model.state_dict()
    assert plain_state.keys() == generic_state.keys()
    for key, value in plain_state.items():
        assert torch.equal(value, generic_state[key]), key
    return calls[0], plain


class _Body(nn.Module):
    """A module with a Parameter and a buffer, whose forward is ``body``."""

    def __init__(self, body: Callable[[nn.Module, torch.Tensor], Any]) -> None:
        super().__init__()
        self.body = body
        self.weight = nn.Parameter(torch.ones(3))
        self.register_buffer("running", torch.zeros(3))
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> Any:
        return self.body(self, x)


def _x() -> torch.Tensor:
    return torch.arange(6.0).reshape(2, 3) - 2.0


def _body(fn: Callable[[nn.Module, torch.Tensor], Any]) -> Build:
    return lambda: _Body(fn)


def test_inplace_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-place writes on op outputs: the receiver's label advances, ``is_inplace`` holds."""

    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = m.lin(x) * 2
        y.relu_()
        y.add_(1)
        y += x
        return y * m.weight

    calls, trace = _check(monkeypatch, _body(body), _x())
    assert calls > 0
    assert any(op.is_inplace for op in trace.ops)


def test_aliasing_views(monkeypatch: pytest.MonkeyPatch) -> None:
    """Views share their input's storage: the alias semantics name the input position."""

    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x + 1
        v = y.view(-1)
        t = y.t()
        v.mul_(2)
        d = y.detach()
        return t.sum() + v[1:4].sum() + d.mean() + y.expand(2, 2, 3).sum()

    calls, trace = _check(monkeypatch, _body(body), _x())
    assert calls > 0
    aliased = [op for op in trace.ops if op.func_name in ("view", "t")]
    assert aliased


def test_tensor_mutated_after_its_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tensor written after its op and read again: the version counter tells them apart."""

    def body(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        a = x * 2
        b = a + 1
        a.add_(5)
        c = a * b
        a.zero_()
        return c + a

    calls, _trace = _check(monkeypatch, _body(body), _x())
    assert calls > 0


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Sequential(nn.Linear(3, 3), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + x


class _Nested(nn.Module):
    """Nested modules, one block called twice."""

    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()
        self.other = _Block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.other(self.block(self.block(x)))


def test_nested_modules_and_repeated_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _trace = _check(monkeypatch, _Nested, _x())
    assert calls > 0


class _Loop(nn.Module):
    """One cell applied in a loop (the recurrence the grouping steps fold)."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.zeros_like(x)
        for _ in range(6):
            h = torch.tanh(self.cell(h) + x)
        return h


def test_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _trace = _check(monkeypatch, _Loop, _x())
    assert calls > 0


class _Cond(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 3
        y = y - 1 if y.sum() > 0 else y + 1
        if (y > 100).any():
            y = y * 0
        return y.relu()


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_conditionals(monkeypatch: pytest.MonkeyPatch, sign: float) -> None:
    calls, _trace = _check(monkeypatch, _Cond, _x().abs() * sign + sign)
    assert calls > 0


class _Buffers(nn.Module):
    """A buffer read, written in place, and a training batch norm's running stats."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("running", torch.zeros(3))
        self.norm = nn.BatchNorm1d(3)
        self.train()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.running.add_(x.mean(0))
        return self.norm(x) + self.running


def test_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, _trace = _check(monkeypatch, _Buffers, _x())
    assert calls > 0


def test_backward_session_takes_the_generic_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gradient-saving session keeps alias contracts, so no call takes the plain emit."""

    capture = tl.options.CaptureOptions(seed=SEED, save_grads="all")
    calls, _trace = _check(monkeypatch, _Nested, _x(), capture=capture)
    assert calls == 0


def test_interventions(monkeypatch: pytest.MonkeyPatch) -> None:
    intervene = tl.when(tl.func("relu"), tl.zero_ablate())
    _calls, trace = _check(monkeypatch, _Nested, _x(), intervene=intervene)
    assert any(op.func_name == "relu" for op in trace.ops)


def test_halt(monkeypatch: pytest.MonkeyPatch) -> None:
    def halt_on_relu(ctx: Any) -> bool:
        return ctx.kind == "op" and ctx.func_name == "relu"

    _calls, trace = _check(monkeypatch, _Nested, _x(), stop_after=halt_on_relu)
    assert trace.is_stopped


class _Raises(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x * 2).relu() + 1
        if y.shape[0] == 2:
            raise ValueError("planted failure mid-forward")
        return y


def test_exception_mid_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """A forward that raises after some ops raises the same error on both paths."""

    errors = []
    for plain in (True, False):
        with monkeypatch.context() as patch:
            if not plain:
                patch.setattr(_ops_exhaustive, "plain_single_output", lambda *_: False)
            with pytest.raises(Exception) as caught:
                _capture(_Raises, _x())
        errors.append((type(caught.value).__name__, str(caught.value)))
    assert errors[0] == errors[1]
    assert "planted failure mid-forward" in errors[0][1]
    # The next capture after the failure is clean on the plain path.
    calls, _trace = _check(monkeypatch, _Nested, _x())
    assert calls > 0


class _Chain(nn.Module):
    def __init__(self, length: int) -> None:
        super().__init__()
        self.length = length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index in range(self.length):
            x = x * 1.5 if index % 2 else x + 1
        return x


@pytest.mark.parametrize("length", [1, 2, 400])
def test_small_and_large_traces(monkeypatch: pytest.MonkeyPatch, length: int) -> None:
    """One op, two ops and 400 ops: the facade and the column postprocess bodies."""

    calls, trace = _check(monkeypatch, lambda: _Chain(length), _x())
    assert calls == length
    assert len(trace.ops) >= length


def test_save_and_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The saved trace of the plain emit loads to the same fields as the generic one's."""

    dumps = []
    for plain in (True, False):
        with monkeypatch.context() as patch:
            if not plain:
                patch.setattr(_ops_exhaustive, "plain_single_output", lambda *_: False)
            trace, _model = _capture(_Nested, _x())
        path = tmp_path / f"plain-{plain}.tlspec"
        trace.save(str(path))
        loaded = tl.load(str(path))
        dumps.append(_dump(loaded))
    assert diff(dumps[1], dumps[0]) == []


def _generic_output_alias(args: tuple[Any, ...], kwargs: dict[str, Any], out: torch.Tensor) -> Any:
    event = FunctionEventInput(
        func=torch.add,
        func_name="add",
        func_qualname=None,
        args=args,
        kwargs=kwargs,
        raw_output=out,
        arg_copies=args,
        kwarg_copies=kwargs,
        module_stack=(),
        is_bottom_level_func=True,
        func_call_id=0,
        expected_output_count=1,
    )
    return aliasing.detect_torch_output_alias_contract(
        event,
        backend_grad_handle=None,
        grad_fn_class_name="AddBackward0",
        autograd_memory=0,
        num_autograd_tensors=2,
        bytes_delta_at_call=None,
        bytes_peak_at_call=None,
    )


def _alias_cases() -> list[tuple[tuple[Any, ...], dict[str, Any], torch.Tensor]]:
    base = torch.arange(12.0).reshape(3, 4)
    other = torch.ones(3, 4)
    view = base.view(-1)
    meta = torch.empty(3, device="meta")
    return [
        ((base, other), {}, base + other),
        ((base, 2), {}, base.t()),
        ((base,), {}, base),
        (([base, other], 1), {"alpha": 2, "src": view}, view),
        (({"a": base, "b": (other, base)},), {}, view[2:]),
        ((meta, base), {}, meta.view(3)),
        ((), {}, base.clone()),
        ((base, base), {"out": base}, base),
    ]


@pytest.mark.parametrize("case", range(len(_alias_cases())))
def test_bare_output_alias_check_matches_the_generic_one(
    monkeypatch: pytest.MonkeyPatch, case: int
) -> None:
    """Same semantics and the same function-call ids consumed, logging on and off."""

    args, kwargs, out = _alias_cases()[case]
    fields = {
        "grad_fn_handle": None,
        "grad_fn_class_name": "AddBackward0",
        "autograd_memory": 0,
        "num_autograd_tensors": 2,
        "bytes_delta_at_call": None,
        "bytes_peak_at_call": None,
    }
    for logging_enabled in (False, True):
        with monkeypatch.context() as patch:
            patch.setattr(_state, "_logging_enabled", logging_enabled)
            start = _state.next_func_call_id()
            generic = _generic_output_alias(args, kwargs, out)
            middle = _state.next_func_call_id()
            plain = aliasing.plain_output_alias_semantics(args, kwargs, out, fields)
            end = _state.next_func_call_id()
        assert plain == generic
        assert end - middle == middle - start


def test_top_level_arg_hash_tokens_match_the_generic_ones() -> None:
    values: list[Any] = [
        1,
        -3,
        2.5,
        float("nan"),
        True,
        None,
        "text",
        "an object at 0x1234",
        torch.ones(2, 3),
        nn.Parameter(torch.ones(2)),
        [1, torch.zeros(2), (3.0, None)],
        {"k": torch.zeros(1), "j": 4},
        {1, 2},
        torch.Size([2, 3]),
        torch.float32,
        slice(1, 5),
        object(),
    ]
    for index, value in enumerate(values):
        fast: list[Any] = []
        generic: list[Any] = []
        _arg_hash._append_top_arg_hash(value, f"pos{index}", fast)
        _arg_hash._append_arg_hash(value, f"pos{index}", generic)
        assert fast == generic, value
