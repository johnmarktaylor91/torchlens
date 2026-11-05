"""Checks kit item 5/8: decrement skip ledger, watchdog, clip tiers.

The decrement goldens (memo test plan row 4): the naive fire-count
subtraction is asserted WRONG under K=4 accumulation (locking the refuted
method's failure so it cannot return as an optimization), the decrement
ledger is exact on 1-skip / consecutive-skips / growth-events, K is
recovered as an exact integer, and undeclared-dynamic-K coverage says
"inferred"/ambiguous, never "exact".
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens.debug._checks as tc
from torchlens._utils._torch_compat import HAS_AMP_GRADSCALER

if HAS_AMP_GRADSCALER:
    from torch.amp import GradScaler
else:  # torch 2.1-2.2: the device-agnostic GradScaler postdates the floor.
    GradScaler = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(
    not HAS_AMP_GRADSCALER,
    reason="torch.amp.GradScaler (device-agnostic) postdates the torch 2.1 floor",
)


def _loop(
    session: tc.ChecksSession,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    plan: list[tuple[int, bool]],
) -> None:
    """Run (micro_batches, poisoned) attempts through the scaled loop."""

    for micro_batches, poisoned in plan:
        optimizer.zero_grad()
        for micro in range(micro_batches):
            loss = model(torch.randn(2, 4)).sum()
            if poisoned and micro == 0:
                loss = loss * torch.tensor(float("inf"))
            scaler.scale(loss / micro_batches).backward()
        scaler.step(optimizer)
        scaler.update()


def _session(**kwargs: object) -> tuple[tc.ChecksSession, nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer, **kwargs)  # type: ignore[arg-type]
    return session, model, optimizer


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_checks/_ledgers.py.
def test_growth_events_counted_without_fake_skips() -> None:
    """0-skip run with growth events: growths counted, skips stay 0."""

    scaler = GradScaler("cpu", init_scale=1024.0, growth_interval=2)
    session, model, optimizer = _session(scaler=scaler)
    session.attach()
    try:
        plan = [(1, False) for _ in range(6)]
        _loop(session, model, optimizer, scaler, plan)
        ledger = session.report().ledgers["scale"]
    finally:
        session.detach()

    assert ledger["skipped_attempts"] == 0
    assert ledger["growth_events"] == 3
    assert ledger["current_scale"] == 8192.0


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_checks/_ledgers.py.
def test_dynamic_k_disclosed_as_ambiguous() -> None:
    """Undeclared dynamic K: the precondition disclosure says ambiguous."""

    scaler = GradScaler("cpu", init_scale=65536.0)
    session, model, optimizer = _session(scaler=scaler)
    session.attach()
    try:
        plan = [(2, False), (3, True), (2, False)]  # dynamic micro-batching
        with warnings.catch_warnings():
            # The undeclared-K watchdog legitimately warns mid-plan (5
            # backwards without an accepted step at the K=1 default).
            warnings.simplefilter("ignore")
            _loop(session, model, optimizer, scaler, plan)
        ledger = session.report().ledgers["scale"]
    finally:
        session.detach()

    assert ledger["skipped_attempts"] == 1
    assert ledger["inferred_k"] != round(ledger["inferred_k"])
    assert any("dynamic K is ambiguous" in note for note in ledger["preconditions"])


# Kept by the shrink-r4 coverage proof: sole branch coverage of debug/_checks/_session.py.
def test_boundary_mode_is_the_exact_attempt_authority() -> None:
    """D18: the explicit boundary attributes attempts exactly."""

    scaler = GradScaler("cpu", init_scale=65536.0)
    session, model, optimizer = _session(scaler=scaler)
    session.attach()
    try:
        for attempt in range(4):
            with session.step(global_step=attempt, micro_batches=1):
                optimizer.zero_grad()
                loss = model(torch.randn(2, 4)).sum()
                if attempt == 1:
                    loss = loss * torch.tensor(float("inf"))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        report = session.report()
    finally:
        session.detach()

    assert report.counters["explicit_attempts"] == 4
    assert report.counters["explicit_accepted"] == 3
    assert report.counters["explicit_skipped"] == 1


# Kept by the shrink-r4 coverage proof: the only test that provokes error code check_no_accepted_steps.
def test_watchdog_fires_under_forced_total_skip() -> None:
    """D11: 100% skip (no accepted step EVER) trips the watchdog warn."""

    scaler = GradScaler("cpu")
    session, model, optimizer = _session(scaler=scaler, accumulation_steps=1)
    session.attach()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = [(1, True) for _ in range(8)]
            _loop(session, model, optimizer, scaler, plan)
        report = session.report()
    finally:
        session.detach()

    codes = {getattr(w.message, "fields", {}).get("code") for w in caught}
    assert "check_no_accepted_steps" in codes
    assert report.ledgers["watchdog"]["tripped"] is True
    assert report.counters["accepted_steps"] == 0
