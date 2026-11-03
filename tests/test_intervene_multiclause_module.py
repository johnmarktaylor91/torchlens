"""Multi-clause ``tl.trace(intervene=spec)`` specs fire every rule, module rules included.

A multi-clause spec (``rule_a & rule_b``) used to drop its ``tl.module(...)``
rules at the module-boundary door: the door read ``Recipe.selector``, which is
``None`` for a multi-clause spec, so the module rules fired nowhere and nothing
was disclosed. Zero-fire accounting at the capture door also compared action
reprs, so a single fired callable rule could be reported as unfired and two
same-named closures could hide each other. These tests pin per-rule firing and
per-rule (identity, not repr) zero-fire settlement, and that ``spec.bind`` is
unchanged.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.intervention.errors import SiteResolutionError


class _Net(nn.Module):
    """Two linear blocks, a head and one module that never runs."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        self.unused = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.b(torch.relu(self.a(x)))))


def _setup() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _Net().eval(), torch.randn(2, 4)


def _bump(tag: str, calls: list[str], amount: float = 1.0) -> Callable[..., torch.Tensor]:
    """Return a hook closure; every closure from here shares one ``__qualname__``."""

    def hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        calls.append(tag)
        return out + amount

    return hook


def _eager(model: _Net, x: torch.Tensor, *, a: float = 0.0, b: float = 0.0) -> torch.Tensor:
    with torch.no_grad():
        h = model.a(x) + a
        h = model.b(torch.relu(h)) + b
        return model.head(torch.relu(h))


def _output(trace: Any) -> torch.Tensor:
    return trace[trace.output_layers[0]].out.detach()


@contextmanager
def _caught() -> Iterator[list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _zero_fire_warnings(caught: list[warnings.WarningMessage]) -> list[TorchLensWarning]:
    return [
        w.message
        for w in caught
        if isinstance(w.message, TorchLensWarning)
        and w.message.fields.get("code") == "trace_zero_fire"
    ]


def _capture_row(trace: Any) -> dict[str, Any]:
    rows = [
        row
        for row in trace.state_history
        if isinstance(row, dict)
        and row.get("op") == "intervention_event"
        and row.get("door") == "trace"
    ]
    assert rows, "no capture-door audit row"
    return rows[-1]


def _error_code(exc: BaseException) -> str | None:
    seen: BaseException | None = exc
    while seen is not None:
        fields = getattr(seen, "fields", None)
        if isinstance(fields, dict) and fields.get("code"):
            return str(fields["code"])
        code = getattr(seen, "code", None)
        if isinstance(code, str):
            return code
        seen = seen.__cause__ or seen.__context__
    return None


# --- the reported bug: two module rules under tl.trace -----------------------


def test_two_module_rules_both_fire_under_trace() -> None:
    model, x = _setup()
    calls: list[str] = []
    spec = tl.when(tl.module("a"), _bump("a", calls, 1.0)) & tl.when(
        tl.module("b"), _bump("b", calls, 2.0)
    )
    with _caught() as caught:
        trace = tl.trace(model, x, intervene=spec)
    assert calls == ["a", "b"]
    assert torch.allclose(_output(trace), _eager(model, x, a=1.0, b=2.0))
    assert _zero_fire_warnings(caught) == []
    assert _capture_row(trace).get("zero_fire_rule_ids") in ((), [], None)


def test_mixed_module_and_op_rules_both_fire() -> None:
    model, x = _setup()
    calls: list[str] = []
    spec = tl.when(tl.module("a"), _bump("a", calls, 1.0)) & tl.when(tl.func("relu"), tl.scale(0.5))
    with _caught() as caught:
        trace = tl.trace(model, x, intervene=spec)
    assert calls == ["a"]
    with torch.no_grad():
        h = torch.relu(model.a(x) + 1.0) * 0.5
        expected = model.head(torch.relu(model.b(h)) * 0.5)
    assert torch.allclose(_output(trace), expected)
    assert _zero_fire_warnings(caught) == []


def test_two_module_rules_matching_one_boundary_refuse_overlap() -> None:
    model, x = _setup()
    spec = tl.when(tl.module("a"), tl.scale(0.5)) & tl.when(tl.module("a"), tl.zero_ablate())
    with pytest.raises(Exception) as refused:
        tl.trace(model, x, intervene=spec)
    assert _error_code(refused.value) == "spec_rules_overlap"


# --- zero-fire accounting is per rule identity -------------------------------


def test_single_fired_callable_rule_never_warns() -> None:
    model, x = _setup()
    calls: list[str] = []
    with _caught() as caught:
        trace = tl.trace(model, x, intervene=tl.when(tl.module("b"), _bump("b", calls)))
    assert calls == ["b"]
    assert _zero_fire_warnings(caught) == []
    assert not _capture_row(trace).get("zero_fire_rule_ids")


def test_same_named_closures_settle_per_rule() -> None:
    model, x = _setup()
    calls: list[str] = []
    fired = tl.when(tl.module("a"), _bump("a", calls))
    unfired = tl.when(tl.module("unused"), _bump("unused", calls))
    with _caught() as caught:
        trace = tl.trace(model, x, intervene=fired & unfired)
    assert calls == ["a"]
    found = _zero_fire_warnings(caught)
    assert len(found) == 1
    named = list(found[0].fields.get("zero_fire_rule_ids") or ())
    assert named == [unfired.rules[0].rule_id]
    assert fired.rules[0].rule_id not in str(found[0])
    assert list(_capture_row(trace)["zero_fire_rule_ids"]) == [unfired.rules[0].rule_id]


def test_same_named_closures_on_op_rules_settle_per_rule() -> None:
    model, x = _setup()
    calls: list[str] = []
    fired = tl.when(tl.func("relu"), _bump("relu", calls))
    unfired = tl.when(tl.func("sigmoid"), _bump("sigmoid", calls))
    with _caught() as caught:
        tl.trace(model, x, intervene=fired & unfired)
    assert calls == ["relu", "relu"]
    found = _zero_fire_warnings(caught)
    assert [list(w.fields.get("zero_fire_rule_ids") or ()) for w in found] == [
        [unfired.rules[0].rule_id]
    ]


def _spec(kind: str, calls: list[str]) -> Any:
    a = tl.when(tl.module("a"), _bump("a", calls))
    b = tl.when(tl.module("b"), _bump("b", calls))
    never = tl.when(tl.module("unused"), _bump("unused", calls))
    never_op = tl.when(tl.func("sigmoid"), _bump("sigmoid", calls))
    return {
        "single_fired": a,
        "both_fired": a & b,
        "one_unfired": a & never,
        "none_fired": never & never_op,
    }[kind]


_EXPECT_ZERO = {"single_fired": False, "both_fired": False, "one_unfired": True, "none_fired": True}


@pytest.mark.parametrize("kind", sorted(_EXPECT_ZERO))
@pytest.mark.parametrize("policy", ["warn", "error", "disclose"])
def test_on_zero_fire_matrix(kind: str, policy: str) -> None:
    model, x = _setup()
    calls: list[str] = []
    spec = _spec(kind, calls)
    real_zero = _EXPECT_ZERO[kind]
    if policy == "error" and real_zero:
        with pytest.raises(SiteResolutionError) as refused:
            tl.trace(model, x, intervene=spec, on_zero_fire="error")
        assert _error_code(refused.value) == "trace_zero_fire"
        return
    with _caught() as caught:
        trace = tl.trace(model, x, intervene=spec, on_zero_fire=policy)
    found = _zero_fire_warnings(caught)
    if policy == "warn" and real_zero:
        assert len(found) == 1
    else:
        assert found == []
    unfired = {rule.rule_id for rule in spec.rules} if kind == "none_fired" else set()
    if kind == "one_unfired":
        unfired = {spec.rules[1].rule_id}
    assert set(_capture_row(trace).get("zero_fire_rule_ids") or ()) == unfired


# --- spec.bind is unchanged ---------------------------------------------------


def test_bind_path_fires_both_module_rules() -> None:
    model, x = _setup()
    calls: list[str] = []
    spec = tl.when(tl.module("a"), _bump("a", calls, 1.0)) & tl.when(
        tl.module("b"), _bump("b", calls, 2.0)
    )
    with _caught() as caught:
        bound = spec.bind(model)
        out = bound(x)
    assert calls == ["a", "b"]
    assert torch.allclose(out.detach(), _eager(model, x, a=1.0, b=2.0))
    assert bound.last_report.zero_fire_rule_ids == ()
    assert [w for w in caught if "zero" in str(w.message).lower()] == []


def test_bind_path_names_the_unfired_same_named_closure() -> None:
    model, x = _setup()
    calls: list[str] = []
    fired = tl.when(tl.module("a"), _bump("a", calls))
    unfired = tl.when(tl.module("unused"), _bump("unused", calls))
    bound = (fired & unfired).bind(model, on_zero_fire="disclose")
    bound(x)
    assert calls == ["a"]
    assert bound.last_report.zero_fire_rule_ids == (unfired.rules[0].rule_id,)


# --- an input-routed rule in a multi-clause spec still snapshots in-place inputs ---


class _Inplace(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        h = h.relu_()
        return h * 2


@pytest.mark.parametrize("clauses", [1, 2])
def test_input_routed_rule_sees_pre_inplace_input(clauses: int) -> None:
    torch.manual_seed(0)
    model, x = _Inplace().eval(), torch.randn(3, 4)
    spec = tl.when(tl.func("relu_"), tl.splice_module(nn.Tanh(), input_route="in"))
    if clauses == 2:
        spec = spec & tl.when(tl.func("linear"), tl.scale(1.0))
    with _caught():
        trace = tl.trace(model, x, intervene=spec)
    with torch.no_grad():
        expected = torch.tanh(model.lin(x)) * 2
    assert torch.allclose(_output(trace), expected)
