"""grind-r8 cluster 4.0.5: capture admission ordering (R54 + R57).

The typed concurrent-capture refusal used to fire only AFTER the loser had
already run process-global side effects: ``tl.trace`` orchestration reset and
overwrote the admitted winner's capture runtime context (clearing its
replay-template flag and intervention plan mid-capture, sol's 2-thread
repro), and the capture runner reseeded the user's global RNG engines before
claiming the slot -- with the refused-reservation settlement path skipping
the R57 restore entirely (the 4th path a51e9b65 did not enumerate). The
reservation claim now precedes ALL side effects on both surfaces, so a
refused loser mutates nothing.
"""

from __future__ import annotations

import threading

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.errors import ReentrantTraceError


def _refused_trace_in_thread(
    model: nn.Module | None = None,
    x: torch.Tensor | None = None,
    **trace_kwargs,
) -> BaseException | None:
    """Run ``tl.trace`` on another thread and return the exception it raised."""

    if model is None:
        model = nn.Linear(3, 3)
    if x is None:
        x = torch.randn(1, 3)
    holder: list[BaseException | None] = [None]

    def _attempt() -> None:
        try:
            tl.trace(model, x, **trace_kwargs)
        except BaseException as exc:  # noqa: BLE001 - the refusal is the datum
            holder[0] = exc

    worker = threading.Thread(target=_attempt)
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive()
    return holder[0]


def test_refused_capture_does_not_reset_winner_runtime_context() -> None:
    """A refused loser must not clear the admitted winner's live context."""

    reservation = _state.capture_reservation()
    reservation.__enter__()
    try:
        _state.reset_capture_runtime_context()
        _state.configure_capture_runtime_context(
            capture_replay_templates=True,
            model_object_id=12345,
            model_class_qualname="Winner",
        )
        refusal = _refused_trace_in_thread()
        assert isinstance(refusal, ReentrantTraceError)
        assert _state._capture_replay_templates is True, (
            "the refused loser reset the winner's replay-template flag"
        )
        assert _state._relationship_model_id == 12345
        assert _state._relationship_model_class == "Winner"
    finally:
        reservation.__exit__(None, None, None)
        _state.reset_capture_runtime_context()


@pytest.mark.smoke
def test_refused_capture_does_not_reseed_global_rng() -> None:
    """The refusal settles BEFORE capture seeding touches the user's engines."""

    model = nn.Linear(3, 3)  # parameter init draws RNG; construct FIRST
    x = torch.randn(1, 3)
    reservation = _state.capture_reservation()
    reservation.__enter__()
    try:
        torch.manual_seed(777)
        before = torch.get_rng_state()
        refusal = _refused_trace_in_thread(
            model,
            x,
            capture=tl.options.CaptureOptions(random_seed=1234),
        )
        assert isinstance(refusal, ReentrantTraceError)
        assert torch.equal(before, torch.get_rng_state()), (
            "the refused loser reseeded the process-global torch RNG"
        )
    finally:
        reservation.__exit__(None, None, None)
        _state.reset_capture_runtime_context()


@pytest.mark.smoke
def test_admitted_capture_still_restores_user_rng() -> None:
    """R57 regression: the winning path keeps its every-settlement restore."""

    model = nn.Linear(3, 3)  # parameter init draws RNG; construct FIRST
    x = torch.randn(1, 3)
    torch.manual_seed(31337)
    before = torch.get_rng_state()
    tl.trace(model, x, capture=tl.options.CaptureOptions(random_seed=99))
    assert torch.equal(before, torch.get_rng_state())


def test_recorder_token_passthrough_still_works() -> None:
    """tl.record's outer reservation continues through the inner claim."""

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU())
    recording = tl.record(model, torch.randn(1, 3), save=tl.func("relu"))
    assert recording is not None


def test_belt_report_is_read_only_from_non_owner_thread_mid_capture(monkeypatch) -> None:
    """doctor()/compat polling mid-capture must not derive (RNG rewind risk)."""

    from torchlens.backends.torch import belt

    tl.trace(nn.Linear(3, 3), torch.randn(1, 3))  # ensure wrappers installed

    def _derive_should_not_run():  # noqa: ANN202 - test shim
        raise AssertionError("belt derivation must not run mid-capture off-owner")

    monkeypatch.setattr(belt, "_derive", _derive_should_not_run)
    monkeypatch.setattr(belt, "_report", None)
    monkeypatch.setattr(belt, "_member_map", None)
    monkeypatch.setattr(_state, "_active_trace", object())
    monkeypatch.setattr(_state, "_active_owner_thread_id", threading.get_ident())
    holder: list[object] = ["unset"]

    def _poll() -> None:
        holder[0] = belt.belt_report()

    worker = threading.Thread(target=_poll)
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert holder[0] is None  # the cached (cleared) report, served read-only
