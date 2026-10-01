"""Checks kit item 4/5: the D6 nonfinite table test (memo test plan row 3).

fp32 x {inf, nan} x {clip, no clip}; scaler x {inf, nan}; bf16 x {inf, nan}.
With a scaler S-B never fires and 0 parameters move; without one S-B fires
and raises BEFORE the write with weights bitwise clean after; the clipping
collateral count is reported; the nan case sets attribution=smeared and
names no innocent; the unchecked control corrupts the model. Parameter-
space nonfinites route to the scheduled scan / audit_params, never to the
gradient check (routing separation).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler

import torchlens.checks as tc

pytestmark = pytest.mark.smoke


def _model(dtype: torch.dtype = torch.float32) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16, dtype=dtype), nn.Linear(16, 2, dtype=dtype))


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in model.state_dict().items()}


def _bitwise_equal(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(snapshot[name], tensor) for name, tensor in model.state_dict().items())


def _poison_backward(model: nn.Module, kind: str, dtype: torch.dtype) -> None:
    """Run one backward, then seed ONE nonfinite into one param gradient."""

    loss = model(torch.randn(4, 8, dtype=dtype)).sum()
    loss.backward()
    with torch.no_grad():
        model[0].weight.grad[0, 0] = float(kind)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["inf", "nan"])
def test_sb_raise_pre_write_without_scaler(dtype: torch.dtype, kind: str) -> None:
    """fp32/bf16 without a scaler: S-B raises pre-write, weights clean."""

    model = _model(dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer)
    session.register_nonfinite_gradient_check()
    session.attach()
    before = _snapshot(model)
    try:
        optimizer.zero_grad()
        _poison_backward(model, kind, dtype)
        with pytest.raises(tc.CheckViolationError) as exc:
            optimizer.step()
    finally:
        session.detach()

    assert exc.value.fields["code"] == "param_grad_nonfinite"
    assert exc.value.fields["names"] == ["0.weight"]
    finding = exc.value.fields["finding"]
    assert finding["attribution"] == "exact"  # one culprit, named
    assert finding["scale_provenance"] == "unscaled"
    assert finding["stage"] == "post_clip_applied"
    assert _bitwise_equal(model, before), "the raise must fire BEFORE the write"
    assert "error_if_nonfinite" in finding["remedy"]


def test_unchecked_control_corrupts_weights() -> None:
    """The control leg: without the check the corrupt gradient is written."""

    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = _snapshot(model)
    optimizer.zero_grad()
    _poison_backward(model, "nan", torch.float32)
    optimizer.step()
    assert not _bitwise_equal(model, before)
    assert torch.isnan(model[0].weight).any()


def test_nan_backward_smear_sets_smeared_attribution() -> None:
    """A NaN seeded mid-forward smears through backward: no innocent named."""

    torch.manual_seed(0)

    class Poisoned(nn.Module):
        """Forward that mints a NaN between the two layers."""

        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(8, 16)
            self.second = nn.Linear(16, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Multiply by NaN mid-stream, poisoning both layers' grads."""

            return self.second(self.first(x) * torch.tensor(float("nan")))

    model = Poisoned()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer)
    session.register_nonfinite_gradient_check()
    session.attach()
    try:
        optimizer.zero_grad()
        model(torch.randn(4, 8)).sum().backward()
        with pytest.raises(tc.CheckViolationError) as exc:
            optimizer.step()
    finally:
        session.detach()

    finding = exc.value.fields["finding"]
    assert finding["attribution"] == "smeared"
    # Every named parameter genuinely carries a nonfinite (no innocents);
    # more than one is exactly why no single culprit can be claimed.
    assert len(exc.value.fields["names"]) > 1
    assert "smeared" in finding["message"]


def test_clip_collateral_counted_and_disclosed() -> None:
    """fp32 + clip + one inf: clip zeroes the healthy grads (collateral).

    ``clip_grad_norm_`` divides by a nonfinite total: every healthy gradient
    becomes zero and the culprit becomes NaN -- the finding reports the
    collateral count instead of a meaningless clip ratio (memo D12
    amendment), and the remedy credits torch's own off-by-default tripwire.
    """

    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer, clip_norm=1.0)
    session.register_nonfinite_gradient_check()
    session.attach()
    before = _snapshot(model)
    try:
        optimizer.zero_grad()
        _poison_backward(model, "inf", torch.float32)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        with pytest.raises(tc.CheckViolationError) as exc:
            optimizer.step()
    finally:
        session.detach()

    finding = exc.value.fields["finding"]
    assert finding["collateral_zeroed"] is not None
    assert finding["collateral_zeroed"] >= 3  # the three healthy param grads
    assert finding["values"]["n_zeroed"] >= 3
    assert _bitwise_equal(model, before)


@pytest.mark.parametrize("kind", ["inf", "nan"])
def test_scaler_makes_the_event_a_harmless_skip(kind: str) -> None:
    """Under a GradScaler S-B is structurally unreachable: skip, 0 moved."""

    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = GradScaler("cpu")
    session = tc.ChecksSession(model, optimizer, scaler=scaler)
    session.register_nonfinite_gradient_check()  # active, never fires
    session.attach()
    before = _snapshot(model)
    try:
        optimizer.zero_grad()
        loss = model(torch.randn(4, 8)).sum() * torch.tensor(float(kind))
        scaler.scale(loss).backward()
        scaler.step(optimizer)  # skips internally; no raise anywhere
        scaler.update()
        report = session.report()
    finally:
        session.detach()

    assert _bitwise_equal(model, before), "a skipped step moves 0 parameters"
    assert report.counters["accepted_steps"] == 0
    assert report.ledgers["scale"]["skipped_attempts"] == 1


def test_param_space_nonfinite_routes_to_scheduled_scan() -> None:
    """D7 + routing separation: corrupt WEIGHTS raise at S-C, named."""

    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer)
    session.register_nonfinite_param_check(every=1)
    session.attach()
    try:
        with torch.no_grad():
            model[1].weight[0, 0] = float("nan")
        optimizer.zero_grad()
        model(torch.randn(4, 8)).sum().backward()
        with pytest.raises(tc.CheckViolationError) as exc:
            optimizer.step()
    finally:
        session.detach()

    assert exc.value.fields["code"] == "param_value_nonfinite"
    assert "1.weight" in exc.value.fields["names"]
    # Safe post-mutation boundary, never mid-step(): the optimizer write
    # completed before the raise (the checkpoint-is-worthless semantics).
    assert exc.value.fields["accepted_step_id"] == 1


def test_actions_demote_per_registration() -> None:
    """The raise stays demotable (D6): action='collect' collects, no raise."""

    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer)
    session.register_nonfinite_gradient_check(action="collect")
    session.attach()
    try:
        optimizer.zero_grad()
        _poison_backward(model, "inf", torch.float32)
        optimizer.step()  # no raise: demoted
        report = session.report()
    finally:
        session.detach()

    collected = [f for f in report.findings if f.code == "param_grad_nonfinite"]
    assert collected and collected[0].action == "collect"
