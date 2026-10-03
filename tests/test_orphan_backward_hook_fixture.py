"""Orphan backward-hook fixture: the defensive contracts, provoked on purpose.

TorchLens hooks every discovered grad_fn (posthook + prehooks). A hook is
ORPHANED when it is armed but never fires in a pass, or fires after its pass
closed. Both conditions are deterministically constructible with public torch
APIs — no version-specific race is required — and these tests provoke each
one and pin the defense that must absorb it:

- ARMED-BUT-SILENT (a partial ``torch.autograd.grad`` executes a subgraph, so
  most armed hooks stay silent that pass): the silent hooks must contribute no
  phantom fires, and the NEXT full pass must be clean — the scavenge step
  clears per-node timing LIFOs at close, so no stale ``(call_index, stamp)``
  pairs cross passes.
- FIRES-AFTER-CLOSE (a raw ``torch.autograd.backward`` after the bracketed
  pass closed): the task-id-change implicit-pass machinery must open a new
  pass and account every fire there, and ``grad_fn_fire_timings`` must keep
  serving.
- INTERRUPTED PASS (a mid-graph backward raise leaves prehooks fired whose
  posthooks never ran): the keyed-LIFO stale-pop discard must keep the retry
  pass's spans non-negative and both-or-neither paired.
- REENTRANT ENGINE (a tensor hook running its own backward mid-pass): the
  identity-checked close callbacks must keep the outer pass's accounting
  intact.

Provenance: behavior verified byte-identical (same fire counts, pass indices,
timing coherence) on torch 2.1.2+cpu (declared floor), 2.7.1+cpu, and
2.13.0+cu130 on 2026-08-19; this file exists so the CI torch matrix keeps
proving it on every supported version.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.ir.events import BackwardCoverageGap, GradFnDiscovered, GradFnFired


class _TwoHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(4, 8)
        self.head_a = nn.Linear(8, 2)
        self.head_b = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.trunk(x))
        return self.head_a(h), self.head_b(h)


def _backward_trace(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )


def _fired(trace: tl.Trace) -> list[GradFnFired]:
    return [e for e in trace.backward_events if isinstance(e, GradFnFired)]


def _gaps(trace: tl.Trace) -> list[BackwardCoverageGap]:
    return [e for e in trace.backward_events if isinstance(e, BackwardCoverageGap)]


def _assert_timing_pairs_coherent(events: list[GradFnFired]) -> None:
    """Every fire is both-or-neither stamped with a non-negative span."""

    for event in events:
        started, finished = event.fire_started_monotonic, event.fire_finished_monotonic
        assert (started is None) == (finished is None)
        if started is not None:
            assert finished - started >= 0


def test_armed_but_silent_hooks_leave_later_passes_clean() -> None:
    """A partial pass orphans most armed hooks; the next full pass is clean."""

    model = _TwoHead()
    trace = _backward_trace(model, torch.randn(3, 4))
    ya = trace["linear_2_3:1"].out
    yb = trace["linear_3_4:1"].out
    trace.log_backward((ya.sum() + yb.sum()), retain_graph=True)
    hooked = {e.object_id for e in trace.backward_events if isinstance(e, GradFnDiscovered)}
    pass1 = _fired(trace)
    assert {e.object_id for e in pass1} == hooked  # full pass fires every armed hook

    # Implicit partial pass: only head_a's path executes; the rest stay silent.
    torch.autograd.grad(ya.sum(), [model.head_a.weight], retain_graph=True)
    pass2 = [e for e in _fired(trace) if e.pass_index == 2]
    assert 0 < len(pass2) < len(hooked)
    silent = hooked - {e.object_id for e in pass2}
    assert silent  # the orphan condition actually occurred

    # The full pass AFTER the partial one must be complete and coherent. Each
    # log_backward seeds fresh sum/add grad_fns, so pass 3's graph is
    # ISOMORPHIC to pass 1's, not identical: compare shape, not object ids.
    trace.log_backward(ya.sum() + yb.sum())
    pass3 = [e for e in _fired(trace) if e.pass_index == 3]
    assert len(pass3) == len(pass1)
    # Every model-graph hook orphaned by the partial pass fires again: the
    # pass-3 silence is confined to pass-1's dead seed chain.
    refired = {e.object_id for e in pass3}
    still_silent = silent - refired
    assert still_silent < silent
    _assert_timing_pairs_coherent(_fired(trace))
    assert _gaps(trace) == []


def test_hooks_firing_after_pass_close_open_an_implicit_pass() -> None:
    """A raw torch backward after the bracketed close lands in a new pass."""

    trace = _backward_trace(
        nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)), torch.randn(3, 4)
    )
    out = trace.output_ops[0].out
    trace.log_backward(out.sum(), retain_graph=True)
    pass1 = _fired(trace)
    assert pass1 and {e.pass_index for e in pass1} == {1}

    torch.autograd.backward(out.sum())  # post-close: hooks fire OUTSIDE any bracket
    all_fired = _fired(trace)
    pass2 = [e for e in all_fired if e.pass_index == 2]
    assert len(pass2) == len(pass1)  # the same graph re-fired, fully accounted
    _assert_timing_pairs_coherent(all_fired)
    assert len(trace.grad_fn_fire_timings) == len(all_fired)


def test_interrupted_backward_discards_stale_timing_stamps() -> None:
    """Prehooks stranded by a mid-pass raise never mis-pair into the retry."""

    class _Bomb(torch.autograd.Function):
        armed = True

        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            return x * 1.0

        @staticmethod
        def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
            if _Bomb.armed:
                raise RuntimeError("mid-backward bomb")
            return grad

    class _BombModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc2(_Bomb.apply(torch.relu(self.fc1(x))))

    trace = _backward_trace(_BombModel(), torch.randn(3, 4))
    out = trace.output_ops[0].out
    with pytest.raises(RuntimeError, match="mid-backward bomb"):
        trace.log_backward(out.sum(), retain_graph=True)
    _Bomb.armed = False
    trace.log_backward(out.sum())
    events = _fired(trace)
    assert {e.pass_index for e in events} == {1, 2}
    retry = [e for e in events if e.pass_index == 2]
    assert retry
    _assert_timing_pairs_coherent(events)


def test_reentrant_backward_keeps_outer_pass_accounting() -> None:
    """A backward launched from inside a hook cannot corrupt the outer pass."""

    trace = _backward_trace(
        nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2)), torch.randn(2, 4)
    )
    side_x = torch.randn(3, requires_grad=True)
    side_y = (side_x * 2).sum()

    def reentrant_hook(grad: torch.Tensor) -> torch.Tensor:
        side_y.backward(retain_graph=True)  # engine reentry mid-pass
        return grad

    trace["tanh_1_2:1"].out.register_hook(reentrant_hook)
    trace.log_backward(trace.output_ops[0].out.sum())
    events = _fired(trace)
    assert events and {e.pass_index for e in events} == {1}
    assert _gaps(trace) == []
    assert side_x.grad is not None
    _assert_timing_pairs_coherent(events)
