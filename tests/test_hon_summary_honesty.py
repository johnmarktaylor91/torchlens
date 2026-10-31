"""``print(trace)`` honesty: the unverified-structure guard and the not-checked hint.

U35.1: an unverified capture (``capture_verified is False``, e.g. severed
provenance on a dynamically quantized model) no longer prints "purely
feedforward" and marks its op and parameter counts as lower bounds; a verified
or not-recorded capture prints exactly as before.

forks.7: ``track_nonfinite`` stays ``False`` by default; wherever the summary
prints the NOT-CHECKED health line it names
``CaptureOptions(track_nonfinite=True)``. The test never enables a scan.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl


class _Tiny(nn.Module):
    """Two linear layers around a ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


def _summary(trace: tl.Trace) -> str:
    """The bounded ``print(trace)`` text."""

    return str(trace)


def test_unverified_capture_drops_feedforward_and_bounds_its_counts() -> None:
    """U35.1: "structure unverified" replaces the feedforward claim; counts are bounds."""

    trace = tl.trace(_Tiny(), torch.randn(2, 4))
    trace._provenance_verified = False
    trace._provenance_verification_reason = "escape_rescue_unrecovered"
    text = _summary(trace)
    assert "purely feedforward" not in text
    assert "structure unverified" in text
    assert f"at least {len(trace)} total tensors" in text
    assert f"at least {trace.num_layers_with_params} parameter operations" in text
    assert f"at least {trace.num_params} params total" in text
    assert "lower bound" in text
    assert "Capture verified: NO (escape_rescue_unrecovered)" in text


def test_verified_and_unrecorded_captures_print_unchanged() -> None:
    """A verified (or not-recorded) capture keeps the original wording exactly."""

    trace = tl.trace(_Tiny(), torch.randn(2, 4))
    baseline = _summary(trace)
    assert "purely feedforward, no recurrence" in baseline
    assert "at least" not in baseline and "lower bound" not in baseline
    assert (
        f"Parameters: {trace.num_layers_with_params} parameter operations "
        f"({trace.num_params} params total;"
    ) in baseline
    trace._provenance_verified = True
    assert _summary(trace).split("\n")[3:] == baseline.split("\n")[3:]


def test_not_checked_footer_names_track_nonfinite_and_the_default_stays_false() -> None:
    """forks.7: the default is False and the NOT-CHECKED line teaches the option."""

    assert tl.options.CaptureOptions().track_nonfinite is False
    trace = tl.trace(_Tiny(), torch.randn(2, 4), save=tl.func("relu"))
    assert trace.track_nonfinite is False
    assert trace.nonfinite_verdict == "not_checked"
    health = [line for line in _summary(trace).split("\n") if "Health:" in line]
    assert health == [
        "\tHealth: NOT-CHECKED (payloads unexamined; capture with "
        "CaptureOptions(track_nonfinite=True) to check every op)"
    ]


def test_the_named_option_records_every_op_and_drops_the_hint() -> None:
    """The footer's remedy runs: the opt-in capture checks each op at capture time.

    With the option on, the verdict basis is the capture-time record and the
    hint is gone (it only teaches the option while it is off).
    """

    model, x = _Tiny(), torch.randn(2, 4)
    trace = tl.trace(
        model,
        x,
        save=tl.func("relu"),
        capture=tl.options.CaptureOptions(track_nonfinite=True),
    )
    coverage = trace.nonfinite_coverage
    assert coverage.basis == "capture"
    assert coverage.checked > 0 and coverage.nonfinite == 0
    untracked = tl.trace(model, x, save=tl.func("relu")).nonfinite_coverage
    assert untracked.basis == "saved_payloads"
    assert "track_nonfinite=True" not in _summary(trace)
