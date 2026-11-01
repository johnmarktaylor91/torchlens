"""Augmented slice assignment (``x[idx] op= v``) records the write-back's parents per slot.

Python runs ``x[..., 1] *= r`` as ``t = x[..., 1]; t *= r; x[..., 1] = t``. The in-place
multiply writes through the view ``t``, so the capture advances ``x``'s label to it, and
the write-back ``__setitem__`` used to record that one op at slot 0 (the whole tensor)
and at slot 2 (the slice), one label for two different values: argument logging then
failed on every such call (torchvision's keypoint resize, DETR-family box code). The
write-back's slot 0 now binds to the receiver's pre-mutation producer, which still holds
every element the call does not overwrite, and slot 2 to the in-place op.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn

import torchlens as tl


class _ScaleKeypoints(nn.Module):
    """The keypoint-resize call shape: per-column in-place scaling through ``[..., k]``."""

    def forward(self, keypoints: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        resized = keypoints * 1.0
        resized[..., 0] *= ratio
        resized[..., 1] *= ratio * 2
        return resized + 1


class _AddToLeadingSlice(nn.Module):
    """``x[..., :k] += v``: a slice write-back of an in-place add."""

    def forward(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        y = x * 1.0
        y[..., :2] += v
        return y * 2


class _WriteOtherRegion(nn.Module):
    """A view mutated in place, then written into a DIFFERENT region of its base."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 1.0
        w = y[..., :2]
        w.add_(1.0)
        y[..., 2:4] = w
        return y * 2


class _MutateViewThenReadBase(nn.Module):
    """An in-place op on a view, then a plain read of the base."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 1.0
        y[0].add_(100.0)
        return y * 2


def _validate(model: nn.Module, inputs: list[torch.Tensor]) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(tl.validation.validate_forward_pass(model, inputs))


def _setitem_ops(trace: Any) -> list[Any]:
    return [op for op in trace._layer_list if op.func_name == "__setitem__"]


def test_keypoint_scaling_validates() -> None:
    inputs = [torch.rand(1, 17, 3) * 100, torch.tensor(0.8301)]
    assert _validate(_ScaleKeypoints(), inputs)


def test_leading_slice_add_validates() -> None:
    assert _validate(_AddToLeadingSlice(), [torch.randn(3, 4), torch.randn(3, 2)])


def test_write_back_binds_slot_zero_to_the_pre_mutation_producer() -> None:
    model, x, v = _AddToLeadingSlice(), torch.randn(3, 4), torch.randn(3, 2)
    trace = tl.trace(model, [x, v])
    (setitem,) = _setitem_ops(trace)
    positions = setitem.parent_arg_positions["args"]
    assert trace[positions[0]].func_name == "__mul__"
    assert trace[positions[2]].func_name == "__iadd__"
    assert len(setitem.parents) == len(set(setitem.parents)) == 2


def test_consumers_after_the_write_back_bind_to_it() -> None:
    trace = tl.trace(_ScaleKeypoints(), [torch.rand(1, 17, 3), torch.tensor(0.5)])
    first, second = _setitem_ops(trace)
    assert trace[second.parent_arg_positions["args"][0]].label == first.label
    (add,) = [op for op in trace._layer_list if op.func_name == "__add__"]
    assert [trace[parent].label for parent in add.parents] == [second.label]


def test_write_into_another_region_keeps_the_mutation_edge() -> None:
    trace = tl.trace(_WriteOtherRegion(), [torch.randn(3, 4)])
    (setitem,) = _setitem_ops(trace)
    positions = setitem.parent_arg_positions["args"]
    assert trace[positions[0]].func_name == "add_"
    assert positions[0] == positions[2]


def test_read_of_a_view_mutated_base_still_binds_to_the_mutation() -> None:
    trace = tl.trace(_MutateViewThenReadBase(), [torch.randn(3, 4)])
    double = [op for op in trace._layer_list if op.func_name == "__mul__"][-1]
    assert [trace[parent].func_name for parent in double.parents] == ["add_"]
