"""A recorded conditional test is not a scalar escape; every real escape still is.

``if h.sum() > 0:`` reads a 0-dim bool through ``__bool__``. Postprocess step 5
records that read as a ``Conditional``, so the dependence is captured and the
``scalar_escape`` advisory (warning plus ``capture_advisories`` row) must stay
quiet. The carve-out is narrow: a value read as arithmetic, non-bool
truthiness and an ``assert`` test keep warning, also inside a model that
records a conditional.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.errors import ScalarEscapeWarning

# No smoke mark: the smoke tier sits at its size ceiling (tests/conftest.py).


class IfOnly(nn.Module):
    """One recorded ``if`` and no other host read."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        if h.sum() > 0:
            return h * 2
        return -h


class IfAndRealEscape(nn.Module):
    """A recorded ``if`` beside a real escape that feeds arithmetic."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        if h.sum() > 0:
            h = h * 2
        scale = float(h.mean())  # REAL_ESCAPE_LINE
        return h * scale


class Truthiness(nn.Module):
    """``if tensor:`` on a float scalar: not a recorded conditional test."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum():
            return x + 1
        return x - 1


class AssertTest(nn.Module):
    """A bool consumed by ``assert``: not a conditional test."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (x.abs() >= 0).all()
        return x + 1


def _capture(model: nn.Module) -> tuple[Any, list[warnings.WarningMessage]]:
    torch.manual_seed(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model.eval(), torch.ones(1, 4))
    return trace, [w for w in caught if issubclass(w.category, ScalarEscapeWarning)]


def _escape_rows(trace: Any) -> list[dict[str, Any]]:
    rows = (trace.annotations or {}).get("capture_advisories", [])
    return [row for row in rows if row.get("kind") == "scalar_escape"]


def test_recorded_if_raises_no_scalar_escape_advisory() -> None:
    trace, escape_warnings = _capture(IfOnly())
    assert len(trace.graph.conditionals) == 1
    assert escape_warnings == []
    assert _escape_rows(trace) == []


def test_real_escape_still_warns_beside_a_recorded_if() -> None:
    trace, escape_warnings = _capture(IfAndRealEscape())
    assert len(trace.graph.conditionals) == 1
    assert len(escape_warnings) == 1
    rows = _escape_rows(trace)
    assert [row["count"] for row in rows] == [1]
    with open(__file__, encoding="utf-8") as handle:
        real_line = next(
            number for number, text in enumerate(handle, start=1) if "# REAL_ESCAPE_LINE" in text
        )
    assert rows[0]["first_location"].endswith(f":{real_line}")


def test_non_bool_truthiness_still_warns() -> None:
    trace, escape_warnings = _capture(Truthiness())
    assert len(escape_warnings) == 1
    assert [row["count"] for row in _escape_rows(trace)] == [1]


def test_assert_test_still_warns() -> None:
    trace, escape_warnings = _capture(AssertTest())
    assert len(escape_warnings) == 1
    assert [row["count"] for row in _escape_rows(trace)] == [1]


def test_exempt_kinds_match_the_kinds_step5_records() -> None:
    from torchlens._postprocess.control_flow import _BRANCH_CONTEXT_KINDS
    from torchlens.backends.torch._completeness_dispatch_names import (
        _RECORDED_CONDITIONAL_TEST_KINDS,
    )

    assert _RECORDED_CONDITIONAL_TEST_KINDS == _BRANCH_CONTEXT_KINDS
