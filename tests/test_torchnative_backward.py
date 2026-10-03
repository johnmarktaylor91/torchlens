"""FLIP-2 carrier: grad_fn-fire markers on the node-hook bracket (W2.0).

The backward join's carrier is an S-sized sink on ALREADY-SHIPPED code: the
per-fire timing prehook pushes a ``torchlens::gradfn::<object_id>:<pass>:
<call>`` marker at its start-stamp line, the posthook pops it at its
finish-stamp line, TorchLens's OWN posthook logging runs inside a typed
``torchlens::internal::gradfn_hook`` bracket (TN-D18), and the pass-boundary
cleanup closes anything a raising node leaked (TN-D17, measured).

All smoke-tier: tiny models, CPU-only, one owned profiler session per test.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import observability as obs


def _backward_session(model: nn.Module, x: torch.Tensor, **backward_kwargs):
    """One armed capture + log_backward inside one owned session."""

    with obs.session() as sess:
        log = tl.trace(
            model,
            x,
            capture=tl.options.CaptureOptions(backward_ready=True),
            save_mode="reference",
        )
        loss = log.output_ops[0].out.sum()
        log.log_backward(loss, **backward_kwargs)
    extraction = obs.extract_events(sess.closed_profiler)
    return log, extraction


def test_every_fire_markered() -> None:
    """One marker per grad_fn fire; names carry object/pass/call identity."""

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    log, extraction = _backward_session(model, torch.randn(2, 8, requires_grad=True))
    markers = [e for e in extraction.events if e.name.startswith("torchlens::gradfn::")]
    assert len(markers) == len(log.grad_fn_logs)
    assert log.__dict__.get("_tl_gradfn_marker_leaks", 0) == 0
    for event in markers:
        object_id, pass_index, call_index = event.name.rsplit("::", 1)[-1].split(":")
        assert int(object_id) in log.grad_fn_logs
        assert int(pass_index) == 1
        assert int(call_index) == 1
    log.cleanup()


def test_internal_hook_bodies_are_typed() -> None:
    """TN-D18: our posthook logging is bracketed typed-internal, per fire."""

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    log, extraction = _backward_session(model, torch.randn(2, 8, requires_grad=True))
    markers = [e for e in extraction.events if e.name.startswith("torchlens::gradfn::")]
    internal = [
        e for e in extraction.events if e.name.startswith("torchlens::internal::gradfn_hook")
    ]
    assert len(internal) == len(markers)
    log.cleanup()


def test_second_pass_markers_carry_pass_index() -> None:
    """retain_graph: a second backward's markers say pass 2, never pass 1."""

    model = nn.Linear(8, 4)
    with obs.session() as sess:
        log = tl.trace(
            model,
            torch.randn(2, 8, requires_grad=True),
            capture=tl.options.CaptureOptions(backward_ready=True),
            save_mode="reference",
        )
        loss = log.output_ops[0].out.sum()
        log.log_backward(loss, retain_graph=True)
        log.log_backward(loss)
    extraction = obs.extract_events(sess.closed_profiler)
    markers = [e for e in extraction.events if e.name.startswith("torchlens::gradfn::")]
    pass_indexes = {int(e.name.rsplit("::", 1)[-1].split(":")[1]) for e in markers}
    assert pass_indexes == {1, 2}
    assert log.__dict__.get("_tl_gradfn_marker_leaks", 0) == 0
    log.cleanup()


@pytest.mark.smoke
def test_raising_node_leaks_are_drained_and_disclosed() -> None:
    """TN-D17: a raising node's open marker closes at the pass boundary."""

    class Raiser(torch.autograd.Function):
        """Custom autograd function whose backward raises mid-pass."""

        @staticmethod
        def forward(ctx, value: torch.Tensor) -> torch.Tensor:
            return value * 2

        @staticmethod
        def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("injected backward failure")

    class Model(nn.Module):
        """Linear stack with a raising custom-autograd segment."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return Raiser.apply(self.linear(value)).sum(dim=-1)

    with obs.session() as sess:
        log = tl.trace(
            Model(),
            torch.randn(2, 8, requires_grad=True),
            capture=tl.options.CaptureOptions(backward_ready=True),
            save_mode="reference",
        )
        loss = log.output_ops[0].out.sum()
        with pytest.raises(RuntimeError, match="injected backward failure"):
            log.log_backward(loss)
    extraction = obs.extract_events(sess.closed_profiler)
    # No marker survives the pass boundary: the profiler's marker stack was
    # fully unwound before the session closed (otherwise export would show
    # an unterminated user annotation), and the leak counter disclosed it.
    assert log.__dict__.get("_tl_gradfn_marker_leaks", 0) >= 1
    opened = [e for e in extraction.events if e.name.startswith("torchlens::gradfn::")]
    assert opened, "the pre-raise fires should still have produced markers"
    log.cleanup()


def test_no_marker_work_without_a_session() -> None:
    """Strictly opt-in: plain armed backward pushes no gradfn markers."""

    model = nn.Linear(8, 4)
    log = tl.trace(
        model,
        torch.randn(2, 8, requires_grad=True),
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    loss = log.output_ops[0].out.sum()
    log.log_backward(loss)
    assert log.__dict__.get("_tl_gradfn_marker_leaks", 0) == 0
    assert log.__dict__.get("_tl_gradfn_marker_gaps", 0) == 0
    log.cleanup()
