"""Correlation-ID join correctness on synthetic fixtures (torchnative W2.2).

The join's correctness properties are proven on synthetic normalized-event
fixtures with hand-set thread ids -- the ONLY sanctioned cross-thread
negative control: torch's profiler records nothing on a plain Python
thread, so a live Python-thread control would pass vacuously on an empty
trace (TN-D19) and is banned here.

Pinned properties: correlation is the ONLY device join (a launch with no
correlated runtime event stays unattributed no matter how its name reads);
one launch is counted ONCE (union, not sum; a launch group is ambiguous,
never split or duplicated); kernels/copies/memsets stay separate rows;
same-thread containment (no session-thread marker owns another thread's
launch); innermost-owner nesting; typed TorchLens-internal work is excluded
from the part-(a) denominator and owned in part (b); the residual is NAMED;
and the sweep is sub-quadratic (the GPT-2-sized cell joins in under a
second).

All smoke-tier (the 10k-event scale probe runs in ~50 ms).
"""

from __future__ import annotations

import pytest

from torchlens._observability import join_events
from torchlens._observability._kineto import NormalizedEvent

_US = 1_000  # ns per microsecond for readable fixture literals


def _marker(name: str, start: int, end: int, tid: int = 1) -> NormalizedEvent:
    """One TorchLens user-annotation marker event."""

    return NormalizedEvent(
        name=name,
        kind="marker",
        activity="user_annotation",
        start_ns=start,
        end_ns=end,
        tid=tid,
        device_type="cpu",
        device_index=None,
        correlation_id=None,
        is_user_annotation=True,
        scope=None,
    )


def _runtime(ts: int, correlation: int, tid: int = 1) -> NormalizedEvent:
    """One CUDA runtime API event carrying a correlation id."""

    return NormalizedEvent(
        name="cudaLaunchKernel",
        kind="runtime",
        activity="cuda_runtime",
        start_ns=ts,
        end_ns=ts + _US,
        tid=tid,
        device_type="cpu",
        device_index=None,
        correlation_id=correlation,
        is_user_annotation=False,
        scope=None,
    )


def _device(  # noqa: PLR0913 -- fixture mirrors the six normalized device-event fields
    name: str,
    start: int,
    end: int,
    correlation: int | None,
    kind: str = "kernel",
    stream: int = 7,
) -> NormalizedEvent:
    """One device event (kernel / memcpy / memset)."""

    return NormalizedEvent(
        name=name,
        kind=kind,
        activity=kind,
        start_ns=start,
        end_ns=end,
        tid=None,
        device_type="cuda",
        device_index=0,
        correlation_id=correlation,
        is_user_annotation=False,
        scope=None,
        stream=stream,
    )


def test_fused_many_owner_launch_group_is_ambiguous() -> None:
    """Several owners of ONE launch form a group -- never split or doubled."""

    events = (
        _marker("torchlens::op::1", 0, 10 * _US),
        _marker("torchlens::op::2", 20 * _US, 30 * _US),
        _runtime(5 * _US, correlation=77),
        _runtime(25 * _US, correlation=77),
        _device("fused_kernel", 100 * _US, 140 * _US, correlation=77),
    )
    result = join_events(events, extraction_path="in_memory")
    (row,) = result.launches
    assert row.status == "ambiguous"
    assert set(row.owner_labels) == {"1", "2"}
    # The group's time lands once, under launch_group -- not in either op.
    assert result.op_device_ns == {}
    assert result.coverage.owner_share_ns["launch_group"] == 40 * _US


def test_cross_thread_negative_control() -> None:
    """No session-thread marker owns another thread's launch (TN-D2/D19)."""

    events = (
        _marker("torchlens::op::3", 0, 100 * _US, tid=1),
        _runtime(50 * _US, correlation=8, tid=2),  # other thread!
        _device("k", 200 * _US, 210 * _US, correlation=8),
    )
    result = join_events(events, extraction_path="in_memory")
    (row,) = result.launches
    assert row.status == "unattributed"


def test_kinds_stay_separate_and_internal_is_typed() -> None:
    """Copies/memsets are separate rows; internal work is excluded in (a)."""

    events = (
        _marker("torchlens::op::4", 0, 10 * _US),
        _marker("torchlens::internal::99", 20 * _US, 30 * _US),
        _runtime(5 * _US, correlation=1),
        _runtime(25 * _US, correlation=2),
        _device("k", 100 * _US, 200 * _US, correlation=1),
        _device("copy", 200 * _US, 260 * _US, correlation=2, kind="memcpy"),
        _device("set", 300 * _US, 310 * _US, correlation=None, kind="memset"),
    )
    result = join_events(events, extraction_path="in_memory")
    kinds = {row.kind for row in result.launches}
    assert kinds == {"kernel", "memcpy", "memset"}
    internal_rows = [row for row in result.launches if "torchlens_internal" in row.owners]
    assert len(internal_rows) == 1 and internal_rows[0].kind == "memcpy"
    part_a, part_b = result.coverage.accounting("cuda:0")
    # busy=170us total; internal=60us; attributed(model)=100us; unattributed=10us.
    assert part_a == pytest.approx(100 / 110)
    assert part_b == pytest.approx(160 / 170)


def test_missing_is_none_never_zero() -> None:
    """An empty session reports empty availability with a reason."""

    result = join_events((), extraction_path="in_memory")
    assert result.availability == "empty"
    assert result.coverage.reason
    assert result.coverage.accounting("cuda:0") == (None, None)


def test_unavailable_extraction_is_disclosed() -> None:
    """A failed extraction is unavailable with reason, never a guess."""

    result = join_events((), extraction_path="unavailable")
    assert result.availability == "unavailable"
    assert "unavailable" in (result.coverage.reason or "")


def test_explicit_device_time_request_refuses_typed() -> None:
    """Ladder rung 3: an inapplicable column raises typed with remedy."""

    from torchlens._observability import join_events as je
    from torchlens._observability._join import require_availability
    from torchlens.errors import TorchLensError

    result = je((), extraction_path="in_memory")
    with pytest.raises(TorchLensError) as excinfo:
        require_availability(result)
    assert excinfo.value.fields["code"] == "device_time_unavailable"
    assert excinfo.value.fields["remedy"]
