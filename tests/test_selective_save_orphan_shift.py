"""Regression tests: selective save/grad selectors under orphan-removed ops.

Postprocess removes internally-sourced dead branches (orphans) from the raw
capture, which renumbers final layer ordinals, type indexes, and labels
relative to capture-time raw indexes. Integer ``layers_to_save`` ordinals,
exact final labels (``relu_1_2``), type-indexed labels (``relu_1``), and
integer ``save_grads`` selectors are all defined against FINAL numbering, so
resolving them against capture-time raw indexes silently saved the wrong
layer's data or nothing at all. These tests lock the fix: final-numbering
selectors resolve on the deferred (post-postprocess) path, in the same index
space the save uses, and save EXACTLY the requested op's data.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class OrphanBranchModel(nn.Module):
    """Forward with a dead internally-sourced branch removed as an orphan."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dead = torch.ones(4) * 3.0
        _dead2 = dead + 5.0
        a = x + 1.0
        b = torch.relu(a)
        return b * 2.0


class SameTypeOrphanModel(nn.Module):
    """Dead branch containing the SAME op type as the requested layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _dead = torch.relu(torch.ones(4) * 3.0)
        a = x + 1.0
        b = torch.relu(a)
        return b * 2.0


class CleanModel(nn.Module):
    """Same live computation with no orphan branch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x + 1.0
        b = torch.relu(a)
        return b * 2.0


def _saved_labels(trace: tl.Trace) -> list[str]:
    """Return final labels of ops with saved activations."""

    return [op.layer_label for op in trace.layer_list if op.has_saved_activation]


@pytest.mark.parametrize("selector", [2, "relu_1_2"], ids=["int_ordinal", "exact_label"])
def test_layers_to_save_final_selectors_survive_orphan_shift(selector) -> None:
    """Int ordinals and exact final labels save the REQUESTED op, orphans or not."""

    x = torch.randn(4)
    trace = tl.trace(
        OrphanBranchModel(),
        x,
        capture=CaptureOptions(layers_to_save=[selector]),
    )
    assert trace.layer_list[2].layer_label == "relu_1_2"
    saved = _saved_labels(trace)
    assert "relu_1_2" in saved, f"requested {selector!r} but saved only {saved}"
    torch.testing.assert_close(trace["relu_1_2"].out, torch.relu(x + 1.0))


def test_layers_to_save_type_indexed_label_survives_same_type_orphan() -> None:
    """A type-indexed label (``relu_1``) resolves against FINAL type numbering."""

    x = torch.randn(4)
    trace = tl.trace(
        SameTypeOrphanModel(),
        x,
        capture=CaptureOptions(layers_to_save=["relu_1"]),
    )
    saved = _saved_labels(trace)
    assert "relu_1_2" in saved, f"requested 'relu_1' but saved only {saved}"
    torch.testing.assert_close(trace["relu_1_2"].out, torch.relu(x + 1.0))


@pytest.mark.parametrize("selector", [2, "relu_1_2"], ids=["int_ordinal", "exact_label"])
def test_layers_to_save_final_selectors_clean_model_unchanged(selector) -> None:
    """The orphan-free path keeps saving exactly the requested op."""

    x = torch.randn(4)
    trace = tl.trace(
        CleanModel(),
        x,
        capture=CaptureOptions(layers_to_save=[selector]),
    )
    saved = _saved_labels(trace)
    assert "relu_1_2" in saved
    torch.testing.assert_close(trace["relu_1_2"].out, torch.relu(x + 1.0))


def test_layers_to_save_substring_stays_exact_under_orphans() -> None:
    """The legacy substring contract keeps matching the surviving relu."""

    x = torch.randn(4)
    trace = tl.trace(
        OrphanBranchModel(),
        x,
        capture=CaptureOptions(layers_to_save=["relu"]),
    )
    assert "relu_1_2" in _saved_labels(trace)
    torch.testing.assert_close(trace["relu_1_2"].out, torch.relu(x + 1.0))


def test_layers_to_save_mixed_live_and_final_selectors() -> None:
    """A mixed list resolves each component in its correct index space."""

    x = torch.randn(4)
    trace = tl.trace(
        OrphanBranchModel(),
        x,
        capture=CaptureOptions(layers_to_save=["add", 2]),
    )
    saved = _saved_labels(trace)
    assert "add_1_1" in saved
    assert "relu_1_2" in saved
    torch.testing.assert_close(trace["relu_1_2"].out, torch.relu(x + 1.0))
    torch.testing.assert_close(trace["add_1_1"].out, x + 1.0)


def test_layers_to_save_positive_int_survives_negative_tail_window() -> None:
    """A mixed ``[ordinal, -1]`` selection keeps the early ordinal's payload.

    The negative tail selector arms a rolling escrow-eviction window; the
    positive ordinal resolves post-postprocess at an arbitrary position, so
    the window must be disabled rather than evict its payload.
    """

    x = torch.randn(4)
    trace = tl.trace(
        OrphanBranchModel(),
        x,
        capture=CaptureOptions(layers_to_save=[1, -1]),
    )
    saved = _saved_labels(trace)
    assert "add_1_1" in saved, f"requested ordinal 1 but saved only {saved}"
    torch.testing.assert_close(trace["add_1_1"].out, x + 1.0)


def test_save_grads_final_selector_survives_negative_tail_window() -> None:
    """A mixed ``[label, -1]`` grad selection keeps the early op's reference."""

    x = torch.randn(4, requires_grad=True)
    trace = tl.trace(
        OrphanBranchModel(),
        x,
        capture=CaptureOptions(save_grads=["relu_1_2", -1], backward_ready=True),
    )
    trace[trace.output_layers[0]].out.sum().backward()
    assert trace["relu_1_2"].grad is not None
    torch.testing.assert_close(trace["relu_1_2"].grad, torch.full((4,), 2.0))


@pytest.mark.smoke
def test_layers_to_save_unmatched_final_label_raises() -> None:
    """A final-shaped label that matches nothing fails closed, never silently."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError):
        tl.trace(
            OrphanBranchModel(),
            torch.randn(4),
            capture=CaptureOptions(layers_to_save=["relu_1_9"]),
        )


@pytest.mark.parametrize("model_cls", [CleanModel, OrphanBranchModel], ids=["clean", "orphan"])
def test_integer_save_grads_survives_orphan_shift(model_cls) -> None:
    """Integer ``save_grads`` ordinals hook the REQUESTED op's gradient."""

    x = torch.randn(4, requires_grad=True)
    trace = tl.trace(
        model_cls(),
        x,
        capture=CaptureOptions(save_grads=[2], backward_ready=True),
    )
    trace[trace.output_layers[0]].out.sum().backward()
    with_grads = [
        op.layer_label for op in trace.layer_list if getattr(op, "grad", None) is not None
    ]
    assert "relu_1_2" in with_grads, f"requested grads for ordinal 2 but got {with_grads}"
    # d/d(relu) of sum(2 * relu) is exactly 2 everywhere.
    torch.testing.assert_close(trace["relu_1_2"].grad, torch.full((4,), 2.0))


def test_string_save_grads_still_works_under_orphans() -> None:
    """The already-correct string grad-selector path stays intact."""

    x = torch.randn(4, requires_grad=True)
    trace = tl.trace(
        OrphanBranchModel(),
        x,
        capture=CaptureOptions(save_grads=["relu_1_2"], backward_ready=True),
    )
    trace[trace.output_layers[0]].out.sum().backward()
    assert trace["relu_1_2"].grad is not None
    torch.testing.assert_close(trace["relu_1_2"].grad, torch.full((4,), 2.0))
