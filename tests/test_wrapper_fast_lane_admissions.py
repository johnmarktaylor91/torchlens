"""Second set of wrapper fast-lane admission rules, the nested-call exit and the torchvision flag.

Each admission test names one rule of ``backends/torch/_call_plan.py``: the call
it describes either takes the fast commit or must run the wrapper's general
body, and the capture must record it exactly as it does with the fast lane
switched off.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import _call_plan, _ops_exhaustive


class _Forward(nn.Module):
    """A module whose forward is one supplied function of the module and the input.

    It owns Parameters and a buffer only when asked: owned state switches on the
    buffer helpers for every call, which one rule below reads.
    """

    def __init__(self, body: Callable[[nn.Module, torch.Tensor], Any], owned: bool) -> None:
        super().__init__()
        self.body = body
        if owned:
            self.weight = nn.Parameter(torch.ones(3, 4))
            self.bias = nn.Parameter(torch.zeros(4))
            self.register_buffer("table", torch.full((1, 3), 2.0))

    def forward(self, x: torch.Tensor) -> Any:
        return self.body(self, x)


Body = Callable[[nn.Module, torch.Tensor], Any]


def _fast_names(
    monkeypatch: pytest.MonkeyPatch, body: Body, owned: bool = False
) -> tuple[list[str], Any]:
    """Trace ``body`` and return the names the fast lane committed, with the trace."""

    names: list[str] = []
    real_commit = _call_plan.commit_fast_call

    def spy(emit: Any, trace: Any, call: tuple[Any, ...], *rest: Any) -> Any:
        names.append(call[1])
        return real_commit(emit, trace, call, *rest)

    with monkeypatch.context() as patch:
        patch.setattr(_call_plan, "commit_fast_call", spy)
        trace = tl.trace(_Forward(body, owned), torch.ones(2, 3))
    return names, trace


def _op_rows(trace: Any) -> list[tuple[Any, ...]]:
    """Return the per-op facts a fast and a general capture of one forward must share."""

    return [
        (
            op.label,
            op.func_name,
            tuple(op.shape) if op.shape is not None else None,
            str(op.dtype),
            tuple(op.parents),
            op.is_inplace,
            op.in_multi_output,
            tuple(op.container_path or ()),
        )
        for op in trace.ops
    ]


def _assert_same_capture(
    monkeypatch: pytest.MonkeyPatch, body: Body, trace: Any, owned: bool = False
) -> None:
    """Assert ``trace`` records the same ops as a capture with the fast lane off."""

    with monkeypatch.context() as patch:
        patch.setattr(_call_plan, "admit_call", lambda *_: None)
        general = tl.trace(_Forward(body, owned), torch.ones(2, 3))
    assert _op_rows(trace) == _op_rows(general)


def test_reflected_dunders_take_the_fast_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """``2 * x`` and ``1 + x`` read the schemas of ``mul`` and ``add``."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return 1 + 2 * x

    names, trace = _fast_names(monkeypatch, body)
    assert names == ["__rmul__", "__radd__"]
    _assert_same_capture(monkeypatch, body, trace)


def test_multi_output_call_takes_the_fast_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """``torch.chunk`` returns a tuple of new plain tensors: one fast commit, three ops."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        first, second, third = torch.chunk(x * 2, 3, dim=1)
        return first + second + third

    names, trace = _fast_names(monkeypatch, body)
    assert names == ["__mul__", "chunk", "__add__", "__add__"]
    assert sum(1 for op in trace.ops if op.func_name == "chunk") == 3
    _assert_same_capture(monkeypatch, body, trace)


def test_multi_output_returning_an_argument_runs_the_general_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``broadcast_tensors(x, x)`` hands back its arguments: never the fast commit."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        doubled = x * 2
        left, right = torch.broadcast_tensors(doubled, doubled)
        return left + right

    names, trace = _fast_names(monkeypatch, body)
    assert "broadcast_tensors" not in names
    _assert_same_capture(monkeypatch, body, trace)


def test_parameter_first_argument_of_an_out_of_place_call_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``torch.addmm(bias, x, weight)`` only reads its Parameter receiver."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(module.bias, x, module.weight)

    names, trace = _fast_names(monkeypatch, body, owned=True)
    assert names == ["addmm"]
    _assert_same_capture(monkeypatch, body, trace, owned=True)


def test_parameter_receiver_of_a_dunder_runs_the_general_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Parameter receiver of a name that may mutate (every dunder) stays general."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return module.bias + x.sum()

    names, trace = _fast_names(monkeypatch, body, owned=True)
    assert "__add__" not in names
    _assert_same_capture(monkeypatch, body, trace, owned=True)


def test_flat_tensor_list_argument_is_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """``torch.cat([x, x * 2])``: a flat list of plain tensors takes the fast lane."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x * 2], dim=0)

    names, trace = _fast_names(monkeypatch, body)
    assert names == ["__mul__", "cat"]
    _assert_same_capture(monkeypatch, body, trace)


def test_tensor_list_holding_a_buffer_runs_the_general_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A listed buffer needs the buffer helpers, which read today's argument order."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, module.table], dim=0)

    names, trace = _fast_names(monkeypatch, body, owned=True)
    assert "cat" not in names
    _assert_same_capture(monkeypatch, body, trace, owned=True)


def test_nested_outer_call_skips_the_shared_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """``nn.ReLU`` calls ``F.relu`` then ``torch.relu``: only the inner call builds fields.

    ``F.batch_norm`` saves tensors for backward, so its skipped outer emission must
    still spend the same ``func_call_id`` values.
    """

    model = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(3).eval(), nn.ReLU())
    built: list[str] = []
    real_build = _ops_exhaustive._build_shared_fields_dict

    def spy(trace: Any, func: Any, func_name: str, *rest: Any) -> Any:
        built.append(func_name)
        return real_build(trace, func, func_name, *rest)

    with monkeypatch.context() as patch:
        patch.setattr(_ops_exhaustive, "_build_shared_fields_dict", spy)
        # The thin emit binds its helpers once: drop the binding so it rebinds the spy.
        patch.delitem(_ops_exhaustive._BOUND_THIN, "thin", raising=False)
        trace = tl.trace(model, torch.randn(2, 3))
    assert built == ["relu", "batch_norm", "relu"]
    with monkeypatch.context() as patch:
        patch.setattr(_ops_exhaustive, "_nested_call_logged_output", lambda _out: False)
        reference = tl.trace(model, torch.randn(2, 3))
    assert [(op.label, tuple(op.parents), op.func_name, op.call.call_id) for op in trace.ops] == [
        (op.label, tuple(op.parents), op.func_name, op.call.call_id) for op in reference.ops
    ]


def test_torchvision_covered_by_the_first_decoration_is_not_decorated_again() -> None:
    """Torchvision imported before the first wrap: a re-wrap re-registers nothing."""

    pytest.importorskip("torchvision")
    script = textwrap.dedent(
        """
        import torch, torchvision
        import torchlens as tl
        from torchlens.backends.torch import wrappers
        calls = []
        real = wrappers._register_jit_builtin_wrappers
        def spy():
            calls.append(1)
            real()
        wrappers._register_jit_builtin_wrappers = spy
        model = torch.nn.Linear(2, 2)
        for _ in range(3):
            tl.trace(model, torch.ones(1, 2))
            wrappers.unwrap_torch()
        print(len(calls), wrappers._torchvision_ops_ensured)
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=300
    )
    assert done.stdout.split()[-2:] == ["1", "True"]
