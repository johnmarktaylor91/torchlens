"""Checks kit item 10: the trackers subscription seam (memo 4.7).

Checks publish already-reduced ObserverEvents onto the C06 chassis stream:
``unavailable`` is mandatory vocabulary, every gradient scalar carries
``stage`` + ``scale_provenance``, and the new series this panel delivers --
applied clip factor, skip/scale ledger, watchdog state -- arrive on their
panel. Subscribers never trigger a device sync and never see unknown
rendered as zero; a throwing subscriber cannot corrupt collection (the C06
contract, exercised here through the checks producer).
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens.checks as tc
from torchlens.observability import EventStream, ObserverEvent
from torchlens.utils._torch_compat import HAS_AMP_GRADSCALER

if HAS_AMP_GRADSCALER:
    from torch.amp import GradScaler
else:  # torch 2.1-2.2: the device-agnostic GradScaler postdates the floor.
    GradScaler = None  # type: ignore[assignment,misc]

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not HAS_AMP_GRADSCALER,
        reason="torch.amp.GradScaler (device-agnostic) postdates the torch 2.1 floor",
    ),
]


def _run_scaled_loop(session: tc.ChecksSession, model: nn.Module, optimizer, scaler) -> None:  # noqa: ANN001
    """Drive attempts incl. skips and clipping through an attached session."""

    for attempt in range(6):
        optimizer.zero_grad()
        loss = model(torch.randn(2, 4)).sum()
        if attempt == 1:
            loss = loss * torch.tensor(float("inf"))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()


def test_checks_publish_the_new_series_on_the_shared_stream() -> None:
    """Scale, skipped-attempts, clip-factor, ratio series reach subscribers."""

    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = GradScaler("cpu", init_scale=65536.0)
    stream = EventStream()
    seen: list[ObserverEvent] = []
    stream.subscribe(seen.append)

    session = tc.ChecksSession(model, optimizer, scaler=scaler, clip_norm=0.5, event_stream=stream)
    session.register_update_ratio_check()
    session.register_magnitude_check(exploding_threshold=1e12)
    session.attach()
    try:
        _run_scaled_loop(session, model, optimizer, scaler)
    finally:
        session.detach()

    keys = {event.key for event in seen}
    assert {"checks/scale", "checks/skipped_attempts", "checks/applied_clip_factor"} <= keys

    clip_events = [event for event in seen if event.key == "checks/applied_clip_factor"]
    for event in clip_events:
        # Every gradient scalar carries stage + scale_provenance: without
        # them a dashboard cannot label its own y-axis (memo 4.7).
        assert event.gradient is True
        assert event.stage == "post_clip_applied"
        assert event.scale_provenance == "unscaled"
        assert event.kind == "scalar" and event.value is not None

    skip_events = [event for event in seen if event.key == "checks/skipped_attempts"]
    assert skip_events[-1].value == 1.0
    assert all(event.axis_provenance == "hook_only" for event in skip_events)


def test_watchdog_state_is_a_verdict_series() -> None:
    """The watchdog delivers verdict events, not silence, under total skip."""

    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = GradScaler("cpu")
    stream = EventStream()
    session = tc.ChecksSession(
        model, optimizer, scaler=scaler, accumulation_steps=1, event_stream=stream
    )
    session.attach()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(8):
                optimizer.zero_grad()
                loss = model(torch.randn(2, 4)).sum() * torch.tensor(float("inf"))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
    finally:
        session.detach()

    watchdog_events = [e for e in stream.snapshot() if e.key == "checks/watchdog"]
    assert watchdog_events
    assert watchdog_events[-1].kind == "verdict"
    assert watchdog_events[-1].verdict == "tripped"


def test_unavailable_is_mandatory_vocabulary_on_the_stream() -> None:
    """Censored magnitude arrives as kind='unavailable' with a reason."""

    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    stream = EventStream()
    session = tc.ChecksSession(model, optimizer, clip_norm=0.5, event_stream=stream)
    session.register_change_check(window=(8, 8), action="collect")
    session.attach()
    try:
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
    finally:
        session.detach()

    unavailable = [e for e in stream.snapshot() if e.kind == "unavailable"]
    assert any(e.key == "checks/grad_magnitude" for e in unavailable)
    magnitude_event = next(e for e in unavailable if e.key == "checks/grad_magnitude")
    assert magnitude_event.reason and "max_norm" in magnitude_event.reason


def test_throwing_subscriber_never_corrupts_collection() -> None:
    """The C06 isolation contract holds under the checks producer."""

    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = GradScaler("cpu")
    stream = EventStream()

    def _bomb(event: ObserverEvent) -> None:
        raise RuntimeError("subscriber bug")

    stream.subscribe(_bomb)
    session = tc.ChecksSession(model, optimizer, scaler=scaler, event_stream=stream)
    session.attach()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the loud once-per-subscriber warn
            for _ in range(2):
                optimizer.zero_grad()
                scaler.scale(model(torch.randn(2, 4)).sum()).backward()
                scaler.step(optimizer)
                scaler.update()
        report = session.report()
    finally:
        session.detach()

    assert report.counters["accepted_steps"] == 2
    assert stream.published_total > 0
