"""Checks kit item 9: range/liveness over the PUBLIC phase seam (memo D14).

The in-tree range check is implemented 100% over the public
``capture.hooks`` seam, and this file IS the seam's CI conformance test:
the probes add ZERO ops to a trace (BatchNorm model included -- the
seam-pollution tripwire), collect honest findings, and the liveness
statistic never utters a dead-unit verdict (``tl.dead`` keeps it).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.checks as tc

pytestmark = pytest.mark.smoke


class _BnNet(nn.Module):
    """BatchNorm model: the historical seam-pollution witness (13/13/13)."""

    def __init__(self) -> None:
        super().__init__()
        self.f1 = nn.Linear(4, 8)
        self.bn = nn.BatchNorm1d(8)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear -> BatchNorm -> ReLU -> head."""

        return self.head(torch.relu(self.bn(self.f1(x))))


def test_range_probe_adds_zero_ops_and_finds_violations() -> None:
    """The conformance gate: hooked and bare traces have IDENTICAL op counts."""

    torch.manual_seed(0)
    model = _BnNet()
    x = torch.randn(8, 4)

    bare = tl.trace(model, x)
    bare_ops = len(bare.ops)

    probe = tc.RangeProbe({tl.func("relu"): (0.0, 0.1)})
    hooked = tl.trace(model, x, capture=tl.options.CaptureOptions(hooks=probe.hook_plan()))

    assert len(hooked.ops) == bare_ops, (
        "the range probe polluted the trace: the public phase seam must add "
        "ZERO ops (memo D14; measured 13/13/13 including BatchNorm)"
    )
    assert probe.observed_fires, "the probe never fired"
    findings = probe.findings
    assert findings, "relu outputs above 0.1 must violate the declared range"
    finding = findings[0]
    assert finding.code == "activation_range_violated"
    assert finding.values["n_above"] > 0
    assert finding.values["n_below"] == 0  # relu output is never below 0
    assert finding.evidence == "capture_hook"


def test_range_probe_within_bounds_collects_nothing() -> None:
    """In-bounds activations produce zero findings, with fires still visible."""

    torch.manual_seed(0)
    model = _BnNet()
    probe = tc.RangeProbe({tl.func("relu"): (0.0, None)})
    tl.trace(model, torch.randn(8, 4), capture=tl.options.CaptureOptions(hooks=probe.hook_plan()))

    assert probe.observed_fires  # coverage: zero fires would be visible
    assert not probe.findings


def test_liveness_probe_records_statistics_never_verdicts() -> None:
    """Liveness facts carry zero-fraction/running-max and never say "dead"."""

    torch.manual_seed(0)
    model = _BnNet()
    probe = tc.LivenessProbe([tl.func("relu")])
    tl.trace(model, torch.randn(8, 4), capture=tl.options.CaptureOptions(hooks=probe.hook_plan()))

    facts = probe.facts()
    assert facts, "the liveness probe never fired"
    for stats in facts.values():
        assert 0.0 <= stats["mean_zero_fraction"] <= 1.0
        assert stats["running_abs_max"] >= 0.0
        assert stats["fires"] >= 1.0
    # The statistic is not a verdict: the probe emits no findings at all and
    # its facts carry no dead-unit claim -- tl.dead keeps the only verdict
    # (D14). Its documented pointer is the multi-sample door.
    assert not hasattr(probe, "findings")
    for stats in facts.values():
        assert "dead" not in {key.lower() for key in stats}
    assert "tl.dead" in (tc.LivenessProbe.facts.__doc__ or "")


def test_probe_refusals_are_typed() -> None:
    """Empty/malformed probe configs refuse typed at construction."""

    with pytest.raises(tc.CheckConfigError) as empty_exc:
        tc.RangeProbe({})
    assert empty_exc.value.fields["code"] == "check_bounds_invalid"

    with pytest.raises(tc.CheckConfigError) as pair_exc:
        tc.RangeProbe({"relu_1": (2.0, 1.0)})
    assert pair_exc.value.fields["code"] == "check_bounds_invalid"

    with pytest.raises(tc.CheckConfigError) as sites_exc:
        tc.LivenessProbe([])
    assert sites_exc.value.fields["code"] == "check_bounds_invalid"
