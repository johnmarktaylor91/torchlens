"""Vizmech wave-2 item 17: backward-view polish (D30).

Pins:
- uniform constant rows are suppressed: ``grad N/A`` on every node (the
  memo's x36), ``order 1`` when no node exceeds order 1, ``bwd 1`` on
  single-pass captures -- ink, not information;
- a mixed render KEEPS its rows (order rows return the moment a second-order
  node exists);
- the in-frame backward key renders by default (AUTO), explains only the
  styles actually painted, and honors ``show_legend=False``;
- accumulation edges carry per-target identity in SVG metadata (the
  twelve-identical-``accum``-labels groundwork);
- single-call grad_fn titles drop the stray ``:1`` display suffix.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl


def _backward_trace() -> tl.Trace:
    """Tiny first-order backward capture."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(1, 4, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    trace.log_backward(trace.output_ops[0].out.sum())
    return trace


def _double_backprop_trace() -> tl.Trace:
    """WGAN-GP-style second-order capture (grad-of-grad in the loss)."""

    critic = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    x = torch.randn(2, 4, requires_grad=True)
    trace = tl.trace(
        critic,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    score = trace.output_ops[0].out.sum()
    (grad_input,) = torch.autograd.grad(score, x, create_graph=True)
    penalty = (grad_input.norm(2, dim=1) - 1.0).pow(2).mean()
    trace.log_backward(score + penalty)
    return trace


@pytest.mark.smoke
def test_double_backprop_keeps_order_rows_and_key_row(tmp_path: Path) -> None:
    """A second-order node un-suppresses order rows and earns its key row."""

    trace = _double_backprop_trace()
    dot = trace.draw(graph="backward", path=str(tmp_path / "bwd2"), file_format="svg").dot
    assert "order 2" in dot
    assert "order 1" in dot  # mixed orders: the row is information again
    assert "grad-of-grad" in dot  # the cream=order-2 key row
