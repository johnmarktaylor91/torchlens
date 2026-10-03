"""One-call NaN forensics door (torchnative 6.5 / W3.1).

The pitch is STRUCTURAL: the default capture names the ORIGIN op with
module address and file:line at zero added cost -- detect_anomaly reports
where a NaN surfaced in backward, and the report defers to it BY NAME
inside its own output for the cases this door does not cover. The door
never auto-runs anomaly mode (a second run can mutate state and follow
another stochastic path); clean results are scoped to checked values.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.debug import nan_report
from torchlens.utils._torch_compat import get_cpu_half_kernels_support


class DivideByZero(nn.Module):
    """Injects a divide-by-zero mid-stack (the origin-op fixture)."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(8, 8)
        self.tail = nn.Linear(8, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.head(value)
        poisoned = hidden / torch.zeros_like(hidden)  # origin
        return self.tail(poisoned - poisoned)  # nan propagates


@pytest.mark.smoke
def test_posthoc_report_names_origin_free() -> None:
    """Post-hoc: origin op + source line off the existing capture."""

    log = tl.trace(DivideByZero(), torch.randn(2, 8))
    report = nan_report(log)
    assert report.found
    assert report.kind in ("nan", "inf", "nan+inf")
    assert report.origin_label and "div" in report.origin_label
    assert report.source_line and ":" in report.source_line
    assert report.cost_tier.startswith("free")
    assert report.coverage_basis != "unavailable"
    assert report.checked > 0
    # Defer-by-name INSIDE the output: the authority recipe is verbatim.
    text = report.summary()
    assert "set_detect_anomaly" in text
    assert "origin" in text
    log.cleanup()


def test_clean_report_is_scoped_to_checked_values() -> None:
    """A clean model reports not-found SCOPED, never a blanket 'no NaN'."""

    log = tl.trace(nn.Sequential(nn.Linear(8, 8), nn.ReLU()), torch.randn(2, 8))
    report = nan_report(log)
    assert not report.found
    assert report.scope_note
    text = report.summary()
    assert "CHECKED" in text or "checked" in text
    log.cleanup()


@pytest.mark.smoke
def test_live_tripwire_form() -> None:
    """The live form runs ONE memory-light tripwire forward and discloses it."""

    report = nan_report(DivideByZero(), torch.randn(2, 8))
    assert report.found
    assert report.cost_tier.startswith("diagnostic")
    assert report.coverage_basis == "live_tripwire_first_bad_output"


@pytest.mark.skipif(
    not get_cpu_half_kernels_support(),
    reason="CPU addmm for float16 postdates the torch 2.1 floor",
)
def test_amp_disclosure_on_low_precision() -> None:
    """Half-precision captures carry the AMP/GradScaler scope disclosure."""

    model = nn.Linear(8, 8).half()
    log = tl.trace(model, torch.randn(2, 8).half())
    report = nan_report(log)
    assert report.amp_disclosure and "GradScaler" in report.amp_disclosure
    assert "FORWARD" in report.amp_disclosure
    log.cleanup()


def test_never_auto_runs_anomaly_mode() -> None:
    """The door never arms torch's anomaly mode behind the caller's back."""

    was_enabled = torch.is_anomaly_enabled()
    log = tl.trace(DivideByZero(), torch.randn(2, 8))
    nan_report(log)
    assert torch.is_anomaly_enabled() == was_enabled
    log.cleanup()
