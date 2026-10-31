"""Tests for execution-aligned activation stacking."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.op import Op
from torchlens.intervention.resolver import SiteTable


class _TwoRelus(nn.Module):
    """Two same-shape ReLU operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two ReLUs.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Twice-rectified tensor.
        """

        return torch.relu(torch.relu(x - 1.0) + 0.5)


class _ReusedRelu(nn.Module):
    """ReLU module reused across three calls."""

    def __init__(self) -> None:
        """Initialize the reused module."""

        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the same module repeatedly.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of the final module call.
        """

        x = self.relu(x - 2.0)
        x = self.relu(x + 1.0)
        return self.relu(x + 2.0)


class _DifferentReluShapes(nn.Module):
    """Two ReLUs whose outputs cannot be stacked."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return differently shaped ReLU outputs.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Full-width and sliced ReLU outputs.
        """

        return torch.relu(x), torch.relu(x[:, :2])


def test_stack_matches_individual_outputs_and_labels() -> None:
    """Rows and labels match individually fetched ReLU ops."""

    trace = tl.trace(_TwoRelus(), torch.randn(2, 4))
    expected_ops = sorted(
        trace.find_sites(tl.func("relu"), max_fanout=10), key=lambda op: op.ordinal_index
    )

    stacked = trace.stack(tl.func("relu"))

    assert stacked.labels == tuple(op.layer_label for op in expected_ops)
    assert len(stacked) == len(expected_ops)
    for index, op in enumerate(expected_ops):
        label, row = stacked[index]
        assert label == op.layer_label
        torch.testing.assert_close(row, op.out)
    assert "StackedActivations" in repr(stacked)


def test_stack_sorts_by_recorded_ordinal_even_if_resolver_order_is_adversarial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reused-module rows follow ordinals despite reversed selector iteration."""

    trace = tl.trace(_ReusedRelu(), torch.randn(2, 4))
    original_find_sites = type(trace).find_sites

    def reversed_find_sites(self: Any, query: Any, **kwargs: Any) -> SiteTable:
        """Return the real selector result in deliberately reversed order."""

        resolved = original_find_sites(self, query, **kwargs)
        return SiteTable(tuple(reversed(tuple(resolved))), selector=query)

    monkeypatch.setattr(type(trace), "find_sites", reversed_find_sites)
    expected_ops = sorted(
        original_find_sites(trace, tl.func("relu"), max_fanout=10),
        key=lambda op: op.ordinal_index,
    )

    stacked = trace.stack(tl.func("relu"))

    assert stacked.labels == tuple(op.layer_label for op in expected_ops)
    for row, op in zip(stacked.tensor, expected_ops, strict=True):
        torch.testing.assert_close(row, op.out)


def test_stack_refuses_legacy_op_without_recorded_ordinal() -> None:
    """A restored legacy op with a missing ordinal cannot be ordered honestly."""

    trace = tl.trace(_TwoRelus(), torch.randn(2, 4))
    relu = trace.find_sites(tl.func("relu"), max_fanout=10).first()
    legacy_state = relu.__getstate__()
    legacy_state.pop("ordinal_index")
    restored_relu = Op.__new__(Op)
    restored_relu.__setstate__(legacy_state)
    trace._layer_list = [restored_relu if op is relu else op for op in trace._layer_list]

    with pytest.raises(ValueError, match="requires matched operations with recorded ordinals"):
        trace.stack(tl.func("relu"))


def test_stack_refuses_shape_mismatch_with_both_sites() -> None:
    """Shape mismatch error reports both operation labels and shapes."""

    trace = tl.trace(_DifferentReluShapes(), torch.randn(3, 4))
    relus = tuple(trace.find_sites(tl.func("relu"), max_fanout=10))

    with pytest.raises(ValueError) as exc_info:
        trace.stack(tl.func("relu"))

    message = str(exc_info.value)
    assert relus[0].layer_label in message
    assert relus[1].layer_label in message
    assert "(3, 4)" in message
    assert "(3, 2)" in message


def test_stack_refuses_unsaved_payload() -> None:
    """Matched but unsaved operations fail with payload-specific wording."""

    trace = tl.trace(_TwoRelus(), torch.randn(2, 4), save=tl.func("add"))
    relu = trace.find_sites(tl.func("relu"), max_fanout=10).first()

    with pytest.raises(ValueError, match="has no saved activation payload") as exc_info:
        trace.stack(tl.func("relu"))

    assert relu.layer_label in str(exc_info.value)


def test_stack_refuses_zero_matches_with_selector_repr() -> None:
    """Empty selection is explicit and identifies the requested selector."""

    trace = tl.trace(_TwoRelus(), torch.randn(2, 4))
    selector = tl.func("definitely_absent")

    with pytest.raises(ValueError, match="matched 0 sites") as exc_info:
        trace.stack(selector)

    assert repr(selector) in str(exc_info.value)


def test_stack_tensors_feed_linear_cka() -> None:
    """Flattened activation stacks are valid inputs to linear CKA."""

    model = _TwoRelus()
    stack_a = tl.trace(model, torch.randn(2, 4)).stack(tl.func("relu"))
    stack_b = tl.trace(model, torch.randn(2, 4)).stack(tl.func("relu"))

    result = tl.stats.cka(stack_a.tensor.flatten(1), stack_b.tensor.flatten(1))

    assert isinstance(result, float)
