"""Validation evidence for threshold ops whose output a small perturbation cannot move.

Three holes in the perturbation tripwire, each pinned by a real edge that must
validate True and a planted capture bug (a replay that ignores its recorded
operand) that must validate False:

* a comparison called with keyword arguments (``other=``, ``input=``,
  ``out=``) skipped the comparand straddle and the bool substitution battery
  and took the blanket ``discrete_bool_output`` pass with no evidence;
* ``torch.isclose`` against a comparand outside the battery's values never
  crossed its tolerance, so a real edge looked insensitive;
* integer ``clamp``/``maximum``/``minimum`` saturated against a large constant
  on a one-valued parent stay flat under the small integer draw.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens.validation import last_validation_failure

PAD = 50256
BIG = 50000


def _ids() -> torch.Tensor:
    return torch.tensor([[464, 3139, 286, 4881, 318]])


class _KwargEq(nn.Module):
    """A pad-token test whose comparison is spelled with keyword arguments."""

    def __init__(self, spelling: str) -> None:
        super().__init__()
        self.spelling = spelling
        self.register_buffer("pad", torch.tensor(PAD))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        last = ids[:, -1]
        if self.spelling == "other":
            hit = torch.eq(last, other=self.pad)
        elif self.spelling == "input":
            hit = torch.eq(input=last, other=self.pad)
        else:
            hit = torch.eq(last, self.pad, out=torch.empty(last.shape, dtype=torch.bool))
        return ids.float() * (~hit).float().unsqueeze(-1)


class _CloseCheck(nn.Module):
    """An ``isclose`` test of one value against a reference it does not match."""

    def __init__(self, keyword_tolerance: bool) -> None:
        super().__init__()
        self.keyword_tolerance = keyword_tolerance
        self.register_buffer("ref", torch.tensor([2.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1]
        if self.keyword_tolerance:
            near = torch.isclose(last, self.ref, atol=1e-3)
        else:
            near = torch.isclose(last, self.ref)
        return x * (~near).float().unsqueeze(-1)


def _close_input() -> torch.Tensor:
    return torch.tensor([[1.0, 3.7]])


class _IdSaturate(nn.Module):
    """Integer ids saturated against a large constant (the output is the constant)."""

    def __init__(self, form: str) -> None:
        super().__init__()
        self.form = form
        self.register_buffer("big", torch.tensor(BIG))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        last = ids[:, -1:]
        if self.form == "clamp_min":
            out = last.clamp(min=BIG)
        elif self.form == "clamp_max":
            out = torch.clamp(last, max=-BIG)
        elif self.form == "maximum":
            out = torch.maximum(last, self.big)
        else:
            out = torch.minimum(last, -self.big)
        return ids.float() + out.float()


def _zeroed_first_operand(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``func`` into a capture bug that ignores its recorded first operand."""

    def planted(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("out", None)
        if args:
            args = (torch.zeros_like(args[0]), *args[1:])
        else:
            kwargs["input"] = torch.zeros_like(kwargs["input"])
        return func(*args, **kwargs)

    return planted


def _plant(monkeypatch: pytest.MonkeyPatch, func_name: str, func: Callable[..., Any]) -> list[str]:
    """Replace captured ``func_name`` callables before replay; return the labels planted.

    Only ops whose saved output the planted callable reproduces are planted, so
    the plain replay matches and the perturbation check alone must catch it.
    """

    planted_labels: list[str] = []
    original = Trace.validate_forward_pass
    planted = _zeroed_first_operand(func)

    def validate_with_plant(trace: Trace, *args: Any, **kwargs: Any) -> Any:
        for op in trace._layer_list:
            if op.func_name != func_name:
                continue
            replayed = planted(*op.saved_args, **dict(op.saved_kwargs or {}))
            if isinstance(replayed, torch.Tensor) and torch.equal(replayed, op.out):
                op.func = planted
                planted_labels.append(op.label)
        return original(trace, *args, **kwargs)

    monkeypatch.setattr(Trace, "validate_forward_pass", validate_with_plant)
    return planted_labels


def _assert_planted_bug_fails(planted_labels: list[str], verdict: bool, func_name: str) -> None:
    failure = last_validation_failure()
    assert planted_labels, "the plant reproduced no saved output; the test proves nothing"
    assert verdict is False
    assert failure is not None and failure.check_name == "perturbation"
    assert failure.func_name == func_name


@pytest.mark.parametrize("spelling", ["other", "input", "out"])
def test_keyword_comparison_validates(spelling: str) -> None:
    assert tl.validate(_KwargEq(spelling).eval(), _ids(), scope="forward", seed=0) is True


@pytest.mark.parametrize("spelling", ["other", "input", "out"])
def test_keyword_comparison_planted_bug_fails(
    monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    planted_labels = _plant(monkeypatch, "eq", torch.eq)

    with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(_KwargEq(spelling).eval(), _ids(), scope="forward", seed=0)

    _assert_planted_bug_fails(planted_labels, verdict, "eq")


@pytest.mark.parametrize("keyword_tolerance", [False, True])
def test_isclose_validates(keyword_tolerance: bool) -> None:
    model = _CloseCheck(keyword_tolerance).eval()
    assert tl.validate(model, _close_input(), scope="forward", seed=0) is True


@pytest.mark.parametrize("keyword_tolerance", [False, True])
def test_isclose_planted_bug_fails(
    monkeypatch: pytest.MonkeyPatch, keyword_tolerance: bool
) -> None:
    planted_labels = _plant(monkeypatch, "isclose", torch.isclose)
    model = _CloseCheck(keyword_tolerance).eval()

    with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(model, _close_input(), scope="forward", seed=0)

    _assert_planted_bug_fails(planted_labels, verdict, "isclose")


@pytest.mark.parametrize("form", ["clamp_min", "clamp_max", "maximum", "minimum"])
def test_integer_saturation_validates(form: str) -> None:
    assert tl.validate(_IdSaturate(form).eval(), _ids(), scope="forward", seed=0) is True


@pytest.mark.parametrize(
    ("form", "func_name", "func"),
    [
        ("clamp_min", "clamp", torch.clamp),
        ("clamp_max", "clamp", torch.clamp),
        ("maximum", "maximum", torch.maximum),
        ("minimum", "minimum", torch.minimum),
    ],
)
def test_integer_saturation_planted_bug_fails(
    monkeypatch: pytest.MonkeyPatch, form: str, func_name: str, func: Callable[..., Any]
) -> None:
    planted_labels = _plant(monkeypatch, func_name, func)

    with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(_IdSaturate(form).eval(), _ids(), scope="forward", seed=0)

    _assert_planted_bug_fails(planted_labels, verdict, func_name)
