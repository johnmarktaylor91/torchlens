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

import re
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

pytestmark = pytest.mark.smoke


def _backward_trace() -> tl.Trace:
    """Tiny first-order backward capture."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(1, 4, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
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
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    score = trace.output_ops[0].out.sum()
    (grad_input,) = torch.autograd.grad(score, x, create_graph=True)
    penalty = (grad_input.norm(2, dim=1) - 1.0).pow(2).mean()
    trace.log_backward(score + penalty)
    return trace


def test_uniform_rows_suppressed_and_key_present(tmp_path: Path) -> None:
    """First-order single-pass render: no uniform rows, key on by default."""

    trace = _backward_trace()
    dot = trace.draw_backward(
        vis_outpath=str(tmp_path / "bwd"), vis_save_only=True, vis_fileformat="svg"
    )
    assert "grad N/A" not in dot  # uniform on every node -> suppressed
    assert "order 1" not in dot  # no higher-order node -> suppressed
    assert ">bwd 1<" not in dot  # single-pass -> suppressed
    assert "backward key" in dot
    assert "order N = derivative order" in dot
    assert "grad-of-grad" not in dot  # style not painted -> no key row


def test_key_suppressed_with_show_legend_false(tmp_path: Path) -> None:
    """Explicit False is a deliberate act."""

    trace = _backward_trace()
    dot = trace.draw_backward(
        vis_outpath=str(tmp_path / "bwd_nokey"),
        vis_save_only=True,
        vis_fileformat="svg",
        show_legend=False,
    )
    assert "backward key" not in dot


def test_double_backprop_keeps_order_rows_and_key_row(tmp_path: Path) -> None:
    """A second-order node un-suppresses order rows and earns its key row."""

    trace = _double_backprop_trace()
    dot = trace.draw_backward(
        vis_outpath=str(tmp_path / "bwd2"), vis_save_only=True, vis_fileformat="svg"
    )
    assert "order 2" in dot
    assert "order 1" in dot  # mixed orders: the row is information again
    assert "grad-of-grad" in dot  # the cream=order-2 key row


def test_accum_edges_carry_target_identity(tmp_path: Path) -> None:
    """Accum-identity groundwork: per-edge tooltip metadata, not bare labels."""

    trace = _backward_trace()
    dot = trace.draw_backward(
        vis_outpath=str(tmp_path / "bwd_accum"), vis_save_only=True, vis_fileformat="svg"
    )
    tooltips = re.findall(r'tooltip="(accum -> [^"]+)"', dot)
    assert tooltips, "accum edges must carry their target identity"
    assert len(set(tooltips)) == len(tooltips), (
        f"accum tooltips must be per-target distinct, got {tooltips}"
    )


def test_single_call_titles_drop_pass_suffix(tmp_path: Path) -> None:
    """Unrolled backward with one call per grad_fn: no stray ':1' titles."""

    trace = _backward_trace()
    dot = trace.draw_backward(
        vis_outpath=str(tmp_path / "bwd_unrolled"),
        vis_save_only=True,
        vis_fileformat="svg",
        vis_mode="unrolled",
    )
    assert not re.search(r":1</", dot), "single-call grad_fn titles must not carry ':1'"
