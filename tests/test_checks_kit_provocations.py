"""Provocation tests for the checks-kit vocabulary (the S-17 provocation ratchet).

Every code the kit declares must be provoked by a real failing input, not
merely documented -- an unprovoked tripwire is indistinguishable from a dead
one. Each test here drives the REAL surface to the exact refusal/finding and
asserts on the machine-readable code, never message text.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.checks as tc

pytestmark = pytest.mark.smoke


def _tiny_trace() -> object:
    """Capture one minimal trace for option-validation provocations."""

    torch.manual_seed(0)
    return tl.trace(nn.Linear(2, 1), torch.ones(1, 2))


def test_grad_basis_invalid_provoked() -> None:
    """An unknown ``basis=`` token refuses typed before any gradient work."""

    trace = _tiny_trace()
    with pytest.raises(Exception) as excinfo:
        tl.debug.gradient_flow_audit(trace, basis="vibes")
    assert excinfo.value.fields["code"] == "grad_basis_invalid"
    assert excinfo.value.fields["remedy"]


def test_grad_scale_conflict_provoked() -> None:
    """Passing both scale spellings refuses typed (one factor XOR a mapping)."""

    trace = _tiny_trace()
    with pytest.raises(Exception) as excinfo:
        tl.debug.gradient_flow_audit(trace, grad_scale=2.0, grad_scales={1: 2.0})
    assert excinfo.value.fields["code"] == "grad_scale_conflict"


def test_grad_magnitude_flagged_provoked() -> None:
    """An armed S-A magnitude pass flags a norm crossing its threshold."""

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    session = tc.ChecksSession(model, optimizer, scaler=None)
    # An absurdly low exploding threshold guarantees the flag on a healthy
    # step -- the provocation targets the finding plumbing, not calibration.
    session.register_magnitude_check(vanishing_threshold=1e-30, exploding_threshold=1e-12)
    with session:
        optimizer.zero_grad()
        (model(torch.ones(2, 4)).sum() * 100).backward()
        optimizer.step()
    flagged = [f for f in session.report().findings if f.code == "grad_magnitude_flagged"]
    assert flagged, "the armed magnitude pass never flagged an over-threshold norm"
    assert all(f.stage is not None and f.scale_provenance is not None for f in flagged)


def test_param_bounds_violated_provoked() -> None:
    """Declared bounds catch an out-of-range parameter value."""

    audit = tc.audit_params({"w": torch.ones(4)}, bounds={"w": (0.0, 0.5)})
    codes = [f.code for f in audit.findings]
    assert "param_bounds_violated" in codes


def test_param_dtype_headroom_provoked() -> None:
    """Magnitudes within max_fraction of the fp16 finite max are flagged."""

    near_max = torch.full((4,), 60000.0, dtype=torch.float16)  # fp16 max is 65504
    audit = tc.audit_params({"w": near_max}, max_fraction=0.9)
    codes = [f.code for f in audit.findings]
    assert "param_dtype_headroom" in codes


def test_param_subnormal_heavy_provoked() -> None:
    """A subnormal-dominated tensor trips the underflow signature."""

    subnormals = torch.full((8,), 1e-40)  # below the ~1.18e-38 fp32 normal floor
    audit = tc.audit_params({"w": subnormals}, subnormal_fraction_threshold=0.1)
    codes = [f.code for f in audit.findings]
    assert "param_subnormal_heavy" in codes
