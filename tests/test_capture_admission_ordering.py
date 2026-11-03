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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _state.py.
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
            capture=tl.options.CaptureOptions(seed=1234),
        )
        assert isinstance(refusal, ReentrantTraceError)
        assert torch.equal(before, torch.get_rng_state()), (
            "the refused loser reseeded the process-global torch RNG"
        )
    finally:
        reservation.__exit__(None, None, None)
        _state.reset_capture_runtime_context()


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/belt.py, campaign mutant kills.
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
