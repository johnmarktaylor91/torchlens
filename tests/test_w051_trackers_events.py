"""W051-TRACK / AUD-CODE 2.14c: event-stream verdicts survive every drain.

The session subscribed the bound ``append`` of its INITIAL pending list and
the first drain rebound the attribute to a fresh list, so every verdict after
step 0 landed in an orphaned list and was never serialized.
"""

from __future__ import annotations

import pytest
import torch

import torchlens.trackers as trk
from torchlens.observability import EventStream, ObserverEvent

pytestmark = pytest.mark.smoke


def _mlp() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3)


def _verdict(step: int, verdict: str = "warn") -> ObserverEvent:
    return ObserverEvent(
        key="dead_layer",
        kind="verdict",
        verdict=verdict,
        global_step=step,
        accepted_step_id=step,
        axis_provenance="explicit",
        scope="check",
    )


def _step(session, model, opt, step: int) -> None:  # noqa: ANN001
    with session.step(step):
        opt.zero_grad(set_to_none=True)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()


def test_verdicts_after_the_first_drain_serialize_at_their_step() -> None:
    model, opt = _mlp()
    stream = EventStream()
    sink = trk.MemorySink()
    session = trk.watch(
        model, to=sink, signals=("gradients",), optimizer=opt, every=1, event_stream=stream
    )
    for step in range(3):
        _step(session, model, opt, step)
        stream.publish(_verdict(step, "warn" if step < 2 else "fail"))
    _step(session, model, opt, 3)
    session.close()
    checks = [(p.step, p.value) for p in sink.scalars if p.tag == "torchlens/check/dead_layer"]
    assert checks == [(0, 1.0), (1, 1.0), (2, 2.0)]
    assert session._pending_events == []


def test_verdicts_pending_at_close_are_flushed_not_dropped() -> None:
    """A verdict published after the last step still reaches the sink."""

    model, opt = _mlp()
    stream = EventStream()
    sink = trk.MemorySink()
    session = trk.watch(
        model, to=sink, signals=("gradients",), optimizer=opt, every=1, event_stream=stream
    )
    _step(session, model, opt, 7)
    stream.publish(_verdict(7, "pass"))
    session.close()
    checks = [(p.step, p.value) for p in sink.scalars if p.tag == "torchlens/check/dead_layer"]
    assert checks == [(7, 0.0)]


def test_accepted_step_zero_is_not_treated_as_missing() -> None:
    """``accepted_step_id=0`` is a real coordinate, never a falsy fallback."""

    model, opt = _mlp()
    stream = EventStream()
    sink = trk.MemorySink()
    session = trk.watch(
        model, to=sink, signals=("gradients",), optimizer=opt, every=1, event_stream=stream
    )
    _step(session, model, opt, 0)
    stream.publish(_verdict(0, "warn"))
    _step(session, model, opt, 5)
    session.close()
    checks = [(p.step, p.value) for p in sink.scalars if p.tag == "torchlens/check/dead_layer"]
    assert checks == [(0, 1.0)]
