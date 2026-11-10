"""Wrapper fast-lane rules for calls that write their receiver or return it.

Each test names one rule of ``_call_plan.build_fast_plan`` and
``_ops_emission.commit_admitted``: the call it describes either takes the fast
commit or must run the wrapper's general body. Either way the capture must match
a capture of the same forward with the fast lane switched off, compared over the
field differential's whole surface (labels, call ids, version-counter verdicts,
buffer-write records, reprs and tables).
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
from torch.nn import functional as F

import torchlens as tl
from torchlens.backends.torch import _call_plan

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from field_diff_lib.cases import SEED  # noqa: E402
from field_diff_lib.compare import diff  # noqa: E402
from field_diff_lib.mask import Mask  # noqa: E402
from field_diff_lib.surface import snapshot_trace  # noqa: E402

Body = Callable[[nn.Module, torch.Tensor], Any]


class _Forward(nn.Module):
    """A module with a Parameter, a buffer and a batch norm, whose forward is ``body``."""

    def __init__(self, body: Body, training: bool) -> None:
        super().__init__()
        self.body = body
        self.weight = nn.Parameter(torch.ones(3))
        self.register_buffer("table", torch.full((3,), 2.0))
        self.norm = nn.BatchNorm1d(3)
        self.train(training)

    def forward(self, x: torch.Tensor) -> Any:
        return self.body(self, x)


def _trace(body: Body, training: bool) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Trace ``body`` and return the trace, its canonical dump and the module's state."""

    torch.manual_seed(SEED)
    model = _Forward(body, training)
    x = torch.arange(6.0).reshape(2, 3) - 2.0
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(seed=SEED))
    dump, _canon = snapshot_trace(trace, [(str(REPO_ROOT), "<repo>")], Mask(enabled=True))
    dump["trace"]["event_stream"]["fields"].pop("run_nonce")
    # The per-process trace counter differs between two traces in one process.
    text = json.dumps(dump, default=repr, sort_keys=True).replace(trace.name, "<trace>")
    state = {name: value.clone() for name, value in model.state_dict().items()}
    return trace, json.loads(text), state


def _fast_names_and_check(
    monkeypatch: pytest.MonkeyPatch, body: Body, training: bool = False
) -> tuple[list[str], Any]:
    """Return the names the fast lane committed, after checking the capture against the general one."""

    names: list[str] = []
    real_commit = _call_plan.commit_fast_call

    def spy(emit: Any, trace: Any, call: tuple[Any, ...], *rest: Any) -> Any:
        names.append(call[1])
        return real_commit(emit, trace, call, *rest)

    with monkeypatch.context() as patch:
        patch.setattr(_call_plan, "commit_fast_call", spy)
        trace, fast_dump, fast_state = _trace(body, training)
    with monkeypatch.context() as patch:
        patch.setattr(_call_plan, "admit_call", lambda *_: None)
        _general, general_dump, general_state = _trace(body, training)
    assert diff(general_dump, fast_dump) == []
    assert fast_state.keys() == general_state.keys()
    for key, value in fast_state.items():
        assert torch.equal(value, general_state[key]), key
    return names, trace


def test_receiver_write_relu_takes_the_fast_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """``y.relu_()``: every ``relu_`` schema writes exactly ``self`` and returns it."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        y.relu_()
        return y + 1

    names, trace = _fast_names_and_check(monkeypatch, body)
    assert "relu_" in names
    assert any(op.func_name == "relu_" and op.is_inplace for op in trace.ops)


def test_augmented_assignment_takes_the_fast_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """``y += x`` reads the schemas of ``add_``."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        y += x
        return y * 3

    names, _trace_ = _fast_names_and_check(monkeypatch, body)
    assert "__iadd__" in names


def test_functional_relu_inplace_commits_its_inner_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """``F.relu(y, inplace=True)``: the inner ``relu_`` is fast, the outer call stays general."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        return F.relu(y, inplace=True) + y

    names, _trace_ = _fast_names_and_check(monkeypatch, body)
    assert "relu_" in names and "relu" not in names


def test_eval_batch_norm_takes_the_fast_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eval ``batch_norm`` writes no running statistics: a plain fast commit."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return module.norm(x * 2)

    names, _trace_ = _fast_names_and_check(monkeypatch, body)
    assert "batch_norm" in names


def test_training_batch_norm_records_its_buffer_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Training ``batch_norm`` takes the fast lane and records its running-statistic writes."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return module.norm(x * 2) + module.norm(x)

    names, _trace_ = _fast_names_and_check(monkeypatch, body, training=True)
    assert names.count("batch_norm") == 2


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("dropout", lambda y: F.dropout(y, 0.5, training=False)),
        ("contiguous", lambda y: y.contiguous()),
        ("to", lambda y: y.to(torch.float32)),
    ],
)
def test_same_object_return_takes_the_fast_lane(
    monkeypatch: pytest.MonkeyPatch, name: str, call: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    """An out-of-place call that hands back its receiver takes the same-object steps."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        return call(y) + 1

    names, _trace_ = _fast_names_and_check(monkeypatch, body)
    assert name in names


def test_zero_receiver_write_runs_the_general_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """``zero_`` rebuilds its receiver's output record: the wrapper's special role, no plan."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        y.zero_()
        return y + x

    names, _trace_ = _fast_names_and_check(monkeypatch, body)
    assert "zero_" not in names


def test_parameter_receiver_write_runs_the_general_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Parameter receiver of a writing call keeps the Parameter mutation steps."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            module.weight.mul_(2.0)
        return x * module.weight

    names, _trace_ = _fast_names_and_check(monkeypatch, body)
    assert "mul_" not in names


def test_out_argument_runs_the_general_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """``torch.relu(y, out=z)`` writes an ``out=`` destination: the admission refuses it."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        z = torch.zeros_like(y)
        torch.mul(y, 3, out=z)
        return z + y

    names, _trace_ = _fast_names_and_check(monkeypatch, body)
    assert "mul" not in names


def test_view_and_buffer_receiver_writes_match_the_general_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writes through a view of an activation and into a buffer record what the general body records."""

    def body(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        row = y[0]
        row.add_(1.0)
        module.table.add_(1.0)
        return y + module.table

    _fast_names_and_check(monkeypatch, body)


def test_inplace_schema_names() -> None:
    """The schema verdicts and the augmented-assignment name map."""

    from torchlens._utils import _torch_compat

    if not _torch_compat.HAS_JIT_SCHEMA_ENUMERATION:
        pytest.skip("no dispatcher schema enumeration")
    names = _call_plan.inplace_schema_names()
    assert names["relu_"] and names["add_"] and names["div_"]
    assert _call_plan._inplace_schema_name("__iadd__") == "add_"
    assert _call_plan._inplace_schema_name("__itruediv__") == "div_"
    assert _call_plan._inplace_schema_name("relu_") == "relu_"
    writing = _call_plan.build_fast_plan(False, torch.Tensor.relu_, "relu_", None, True)
    assert writing is not None and writing.writes
    assert _call_plan.build_fast_plan(True, torch.Tensor.relu_, "relu_", None, True) is None
    assert _call_plan.build_fast_plan(False, torch.Tensor.relu_, "relu_", "set", True) is None
    norm = _call_plan.build_fast_plan(False, torch.batch_norm, "batch_norm", None, False)
    assert norm is not None and norm.writes
    reading = _call_plan.build_fast_plan(False, torch.Tensor.contiguous, "contiguous", None, False)
    assert reading is not None and not reading.writes
