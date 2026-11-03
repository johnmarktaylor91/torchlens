"""Per-fire backward timing pins (L9 memo 1.3, wave-2 subset).

One clock: every duration operand pair is ``time.perf_counter()`` from the
same process, paired at capture time by the per-node keyed LIFO. Measured
pairs live in ``GradFnFired`` event fields only (runtime DROP) and are served
by the live-trace-only ``trace.autograd.grad_fn_fire_timings`` accessor; the persisted
``GradFnCall`` timing fields keep their shipped single-wall-stamp semantics
through all of wave 2 (NOTHING NEW PERSISTS PRE-BUMP). Spellings are
DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._io.prerelease import PRERELEASE_STATE_KEY
from torchlens._ir.events import BackwardCoverageGap, GradFnFired
from torchlens.backends.torch import backward as backward_mod


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _backward_trace() -> tl.Trace:
    torch.manual_seed(0)
    trace = tl.trace(
        _Tiny(),
        torch.randn(3, 4),
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    trace.backward(trace.output_ops[0].out.sum())
    return trace


def _fired_events(trace: tl.Trace) -> list[GradFnFired]:
    return [event for event in trace.backward_events if isinstance(event, GradFnFired)]


# ---------------------------------------------------------------------------
# Timed fires: paired monotonic stamps, positive spans, aligned keys.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Keyed-LIFO degrade set: stale discard, no-prehook, registration failure.
# ---------------------------------------------------------------------------


def test_pop_matching_fire_start_discards_stale_entries() -> None:
    stamps = [(1, 10.0), (2, 20.0), (3, 30.0)]
    assert backward_mod._pop_matching_fire_start(stamps, 2) == 20.0
    # The stale entry above the match (key 3) was discarded, not paired.
    assert stamps == [(1, 10.0)]
    assert backward_mod._pop_matching_fire_start(stamps, 7) is None
    assert stamps == []
    assert backward_mod._pop_matching_fire_start(stamps, 1) is None


def test_timing_registration_failure_degrades_to_untimed_not_coverage_gap() -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        _Tiny(),
        torch.randn(3, 4),
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    original = backward_mod._register_fire_timing_prehook

    def failing(
        inner_trace, grad_fn_handle, grad_fn_object_id, fire_start_stamps, marker_tokens=None
    ):
        # Signature tracks _register_fire_timing_prehook (F27 FLIP-2 added the
        # optional marker-token LIFO); a registration failure still means
        # untimed fires only, never a coverage gap.
        del inner_trace, grad_fn_handle, grad_fn_object_id, fire_start_stamps, marker_tokens
        return None

    try:
        backward_mod._register_fire_timing_prehook = failing
        trace.backward(trace.output_ops[0].out.sum())
    finally:
        backward_mod._register_fire_timing_prehook = original
    fired = _fired_events(trace)
    assert fired
    for event in fired:
        assert event.fire_started_monotonic is None
        assert event.fire_finished_monotonic is None
    timings = trace.autograd.grad_fn_fire_timings
    assert timings and all(span is None for span in timings.values())
    # An optional measurement failure must NEVER convert complete-coverage
    # nodes into coverage gaps.
    gaps = [
        event
        for event in trace.backward_events
        if isinstance(event, BackwardCoverageGap) and event.reason == "registration_error"
    ]
    assert gaps == []
    assert trace.grad_fn_timing_provenance == "unmeasured"


def test_register_fire_timing_prehook_own_try_except() -> None:
    class _RefusesPrehooks:
        def register_prehook(self, hook):
            del hook
            raise RuntimeError("this node refuses prehooks")

    trace = tl.trace(_Tiny(), torch.randn(3, 4))
    handle = backward_mod._register_fire_timing_prehook(trace, _RefusesPrehooks(), 1234, [])
    assert handle is None


# ---------------------------------------------------------------------------
# Wave-2 schema pins: NOTHING NEW PERSISTS PRE-BUMP.
# ---------------------------------------------------------------------------


def test_plain_v8_save_persists_per_fire_timing_semantics(tmp_path) -> None:
    """tlspec v8 (L9 bump-time flip): persisted GradFnCall stamps carry the
    per-fire perf_counter pair, discriminated by the persisted provenance
    field; an untimed fire persists (None, None) and reads a None duration,
    never a false zero. No pre-release marker rides the artifact."""

    trace = _backward_trace()
    path = tmp_path / "timing.tlspec"
    trace.save(str(path))
    loaded = tl.load(str(path))
    assert loaded.grad_fn_timing_provenance == "perf_counter"
    live_spans = trace.autograd.grad_fn_fire_timings
    timed = untimed = 0
    for call in loaded.autograd.grad_fn_calls:
        span = call.backward_duration
        live_span = live_spans[f"{call.grad_fn_label}:{call.call_index}"]
        if span is None:
            untimed += 1
            assert call._time_started is None and call._time_finished is None
            assert live_span is None
        else:
            timed += 1
            assert call._time_finished >= call._time_started
            assert float(span) == pytest.approx(float(live_span))
    assert timed >= 1  # the armed capture must measure at least one fire
    state = trace.__getstate__()
    assert PRERELEASE_STATE_KEY not in state


def test_live_accessor_refuses_typed_on_loaded_trace(tmp_path) -> None:
    trace = _backward_trace()
    path = tmp_path / "timing_loaded.tlspec"
    trace.save(str(path))
    loaded = tl.load(str(path))
    with pytest.raises(InvalidArgumentError) as excinfo:
        _ = loaded.autograd.grad_fn_fire_timings
    assert excinfo.value.fields["code"] == "grad_fn_fire_timing_unavailable"
