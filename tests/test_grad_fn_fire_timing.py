"""Per-fire backward timing pins (L9 memo 1.3, wave-2 subset).

One clock: every duration operand pair is ``time.perf_counter()`` from the
same process, paired at capture time by the per-node keyed LIFO. Measured
pairs live in ``GradFnFired`` event fields only (runtime DROP) and are served
by the live-trace-only ``trace.grad_fn_fire_timings`` accessor; the persisted
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
from torchlens._io import PreReleaseArtifactError
from torchlens._io.prerelease import PRERELEASE_STATE_KEY, activate_prerelease_fields
from torchlens.backends.torch import backward as backward_mod
from torchlens.ir.events import BackwardCoverageGap, GradFnFired


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
        save_mode="reference",
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    trace.log_backward(trace.output_ops[0].out.sum())
    return trace


def _fired_events(trace: tl.Trace) -> list[GradFnFired]:
    return [event for event in trace.backward_events if isinstance(event, GradFnFired)]


# ---------------------------------------------------------------------------
# Timed fires: paired monotonic stamps, positive spans, aligned keys.
# ---------------------------------------------------------------------------


def test_timed_fire_has_positive_span_and_ordered_stamps() -> None:
    trace = _backward_trace()
    fired = _fired_events(trace)
    assert fired, "backward capture produced no GradFnFired events"
    for event in fired:
        assert event.fire_started_monotonic is not None
        assert event.fire_finished_monotonic is not None
        assert event.fire_finished_monotonic > event.fire_started_monotonic
    timings = trace.grad_fn_fire_timings
    assert timings and all(span is not None and float(span) > 0 for span in timings.values())
    assert trace.grad_fn_timing_provenance == "perf_counter"


def test_timing_keys_align_with_grad_fn_calls() -> None:
    trace = _backward_trace()
    timings = trace.grad_fn_fire_timings
    # Timings iterate in fire order, grad_fn_calls in discovery order; the
    # key SETS are identical 1:1.
    assert set(timings.keys()) == {call.call_label for call in trace.grad_fn_calls}


def test_repeat_fires_get_independent_spans_and_monotone_wall_stamps() -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        _Tiny(),
        torch.randn(3, 4),
        save_mode="reference",
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    loss = trace.output_ops[0].out.sum()
    trace.log_backward(loss, retain_graph=True)
    trace.log_backward(loss)
    fired = _fired_events(trace)
    by_object: dict[int, list[GradFnFired]] = {}
    for event in fired:
        by_object.setdefault(event.object_id, []).append(event)
    repeated = [events for events in by_object.values() if len(events) >= 2]
    assert repeated, "no grad_fn fired across both passes"
    for events in repeated:
        wall = [event.timestamp for event in events]
        assert wall == sorted(wall)
        spans = {(event.fire_started_monotonic, event.fire_finished_monotonic) for event in events}
        assert len(spans) == len(events), "repeat fires must carry independent stamp pairs"
    timings = trace.grad_fn_fire_timings
    assert all(span is not None for span in timings.values())


@pytest.mark.smoke
def test_higher_order_backward_pairs_lifo_correctly() -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        _Tiny(),
        torch.randn(3, 4),
        save_mode="reference",
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    loss = trace.output_ops[0].out.pow(2).sum()
    trace.log_backward(loss, create_graph=True)
    for event in _fired_events(trace):
        if event.fire_started_monotonic is None:
            continue
        assert event.fire_finished_monotonic is not None
        assert event.fire_finished_monotonic > event.fire_started_monotonic


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


def test_stale_prehook_entry_never_becomes_a_wrong_positive_span() -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        _Tiny(),
        torch.randn(3, 4),
        save_mode="reference",
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    # Seed retry debris: a stale keyed entry for a call index that will never
    # match, on every node list minted by the walk. The next real fire must
    # pair to ITS OWN stamp and discard the debris.
    original = backward_mod._fire_timing_stamp_list

    def seeded(inner_trace, grad_fn_object_id):
        stamps = original(inner_trace, grad_fn_object_id)
        if not stamps:
            stamps.append((999, -1.0))
        return stamps

    try:
        backward_mod._fire_timing_stamp_list = seeded
        trace.log_backward(trace.output_ops[0].out.sum())
    finally:
        backward_mod._fire_timing_stamp_list = original
    for event in _fired_events(trace):
        assert event.fire_started_monotonic is not None
        assert event.fire_started_monotonic > 0, "stale seeded stamp leaked into a pair"
        assert event.fire_finished_monotonic > event.fire_started_monotonic


def test_timing_registration_failure_degrades_to_untimed_not_coverage_gap() -> None:
    torch.manual_seed(0)
    trace = tl.trace(
        _Tiny(),
        torch.randn(3, 4),
        save_mode="reference",
        capture=tl.options.CaptureOptions(backward_ready=True),
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
        trace.log_backward(trace.output_ops[0].out.sum())
    finally:
        backward_mod._register_fire_timing_prehook = original
    fired = _fired_events(trace)
    assert fired
    for event in fired:
        assert event.fire_started_monotonic is None
        assert event.fire_finished_monotonic is None
    timings = trace.grad_fn_fire_timings
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


@pytest.mark.smoke
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
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    assert loaded.grad_fn_timing_provenance == "perf_counter"
    live_spans = trace.grad_fn_fire_timings
    timed = untimed = 0
    for call in loaded.grad_fn_calls:
        span = call.backward_duration
        live_span = live_spans[f"{call.label}:{call.call_index}"]
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


def test_provenance_field_roundtrips_only_under_prerelease_switch(tmp_path) -> None:
    trace = _backward_trace()
    assert trace.grad_fn_timing_provenance == "perf_counter"
    path = tmp_path / "timing_prerelease.tlspec"
    with activate_prerelease_fields():
        tl.save(trace, str(path))
        loaded = tl.load(str(path))
        assert loaded.grad_fn_timing_provenance == "perf_counter"
    # The switched artifact carries the pre-release marker and REFUSES to
    # load as a real v7 artifact outside the switch.
    with pytest.raises(PreReleaseArtifactError):
        tl.load(str(path))


def test_live_accessor_refuses_typed_on_loaded_trace(tmp_path) -> None:
    trace = _backward_trace()
    path = tmp_path / "timing_loaded.tlspec"
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    with pytest.raises(InvalidArgumentError) as excinfo:
        _ = loaded.grad_fn_fire_timings
    assert excinfo.value.fields["code"] == "grad_fn_fire_timing_unavailable"


def test_fold_sort_key_unchanged_and_projection_count_matches() -> None:
    trace = _backward_trace()
    assert len(trace.grad_fn_fire_timings) == len(trace.grad_fn_calls)
    # tlspec v8: projection writes the per-fire monotonic pair; the wall
    # timestamp stays the separate event-ordering stamp and is never a
    # duration operand.
    for call in trace.grad_fn_calls:
        span = trace.grad_fn_fire_timings[f"{call.label}:{call.call_index}"]
        if span is None:
            assert call._time_started is None and call._time_finished is None
        else:
            assert call._time_finished - call._time_started == pytest.approx(float(span))
            assert call._time_started != call.timestamp  # monotonic, not wall
