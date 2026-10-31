"""forks.7 rest: partial-coverage surfaces name track_nonfinite while it is off.

JMT 2026-10-12: "(a) Keep False (no overhead); the not-checked footer teaches
the option." HON's trace footer carries the hint; the summary capture line
and the explain anomaly section carry the same hint here.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.report._health import TRACK_NONFINITE_HINT

_OPTION = "CaptureOptions(track_nonfinite=True)"


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def test_not_checked_surfaces_name_the_option() -> None:
    torch.manual_seed(0)
    trace = tl.trace(_model(), torch.randn(2, 4), save=tl.func("relu"))
    assert trace.nonfinite_verdict == "not_checked"
    assert trace.track_nonfinite is False

    assert f"health NOT-CHECKED ({TRACK_NONFINITE_HINT})" in str(trace.summary())
    assert _OPTION in trace.explain()


def test_the_option_on_drops_the_hint() -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        _model(),
        torch.randn(2, 4),
        save=tl.func("relu"),
        capture=CaptureOptions(track_nonfinite=True),
    )

    assert _OPTION not in str(trace.summary())
    assert _OPTION not in trace.explain()


def test_settled_verdicts_carry_no_hint() -> None:
    torch.manual_seed(0)
    trace = tl.trace(_model(), torch.randn(2, 4))
    assert trace.nonfinite_verdict == "checked_and_clean"

    assert _OPTION not in str(trace.summary())
    assert _OPTION not in trace.explain()
