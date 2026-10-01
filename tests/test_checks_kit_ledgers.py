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
from torch.amp import GradScaler

import torchlens.checks as tc

pytestmark = pytest.mark.smoke


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


def test_decrement_ledger_exact_at_k4_and_naive_locked_wrong() -> None:
    """K=4, one skip: decrement truth 1; fire-count subtraction says 19."""

    scaler = GradScaler("cpu", init_scale=65536.0)
    session, model, optimizer = _session(scaler=scaler, accumulation_steps=4)
    session.attach()
    try:
        plan = [(4, attempt == 2) for attempt in range(6)]
        _loop(session, model, optimizer, scaler, plan)
        ledger = session.report().ledgers["scale"]
    finally:
        session.detach()

    assert ledger["skipped_attempts"] == 1
    assert ledger["accepted_steps"] == 5
    assert ledger["attempts"] == 6
    assert ledger["backward_reads"] == 24
    assert ledger["inferred_k"] == 4.0  # K recovered as an exact integer
    # THE LOCK: the refuted fire-count subtraction reports 19 skips, not 1.
    naive = ledger["backward_reads"] - ledger["accepted_steps"]
    assert naive == 19
    assert naive != ledger["skipped_attempts"]


def test_consecutive_skips_recovered_from_one_ratio() -> None:
    """65536 -> 32768 -> 16384 -> 8192: three decrements, counted exactly."""

    scaler = GradScaler("cpu", init_scale=65536.0)
    session, model, optimizer = _session(scaler=scaler)
    session.attach()
    try:
        plan = [(1, attempt < 3) for attempt in range(5)]
        _loop(session, model, optimizer, scaler, plan)
        ledger = session.report().ledgers["scale"]
    finally:
        session.detach()

    assert ledger["skipped_attempts"] == 3
    assert ledger["accepted_steps"] == 2
    assert ledger["current_scale"] == 8192.0


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


def test_clip_tier_b_saturation_rate_and_censorship() -> None:
    """D12 tier (b): declared clip_norm yields exact saturation + censorship."""

    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer, clip_norm=0.01)
    session.register_change_check(window=(8, 8), action="collect")  # arm S-B norms
    session.attach()
    try:
        for _ in range(4):
            optimizer.zero_grad()
            (model(torch.randn(8, 4)).sum() * 100).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()
        report = session.report()
    finally:
        session.detach()

    clip = report.ledgers["clip"]
    assert clip["tier"] == "b"
    assert clip["saturated_steps"] == 4
    assert clip["saturation_rate"] == 1.0
    # The censorship disclosure (D8): magnitude unavailable, never clean.
    unavailable = dict(report.unavailable)
    assert "grad_magnitude" in unavailable
    assert "pinned to max_norm" in unavailable["grad_magnitude"]


def test_clip_tier_c_constancy_hint_is_info_only() -> None:
    """D12 tier (c): undeclared clipping surfaces as the constancy hint."""

    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    session = tc.ChecksSession(model, optimizer)  # NO clip_norm declared
    session.register_change_check(window=(9, 9), action="collect")
    session.attach()
    try:
        for _ in range(7):
            optimizer.zero_grad()
            (model(torch.randn(8, 4)).sum() * 100).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()
        report = session.report()
    finally:
        session.detach()

    clip = report.ledgers["clip"]
    assert clip["tier"] == "c"
    assert clip["constancy_hint"] is True
    assert clip["constant_total"] == pytest.approx(0.01, rel=1e-3)


def test_clip_tier_a_applied_factor_series_with_magnitude_armed() -> None:
    """D12 tier (a): with the magnitude pass armed the factor series is exact."""

    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    session = tc.ChecksSession(model, optimizer, clip_norm=0.01)
    session.register_magnitude_check(exploding_threshold=1e9)
    session.register_change_check(window=(9, 9), action="collect")
    session.attach()
    try:
        for _ in range(3):
            optimizer.zero_grad()
            (model(torch.randn(8, 4)).sum() * 100).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()
        report = session.report()
    finally:
        session.detach()

    clip = report.ledgers["clip"]
    assert clip["tier"] == "a"
    assert len(clip["applied_clip_factors"]) == 3
    assert all(factor < 1e-2 for factor in clip["applied_clip_factors"])
