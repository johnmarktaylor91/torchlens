"""``deform_conv2d`` argument slots: the weight is perturbation-tested again.

TorchLens captures ``torch.ops.torchvision.deform_conv2d(input, weight, offset,
mask, bias, ..., use_mask)``, the op ``torchvision.ops.deform_conv2d`` calls
after reordering its Python arguments. A structural row meant for the sampling
offsets exempted arg 1, which in this op is the WEIGHT, so a dead weight edge
was never caught. The row is gone: the weight, the offset and every other
parent are perturbed, and a dead edge at either slot fails. The dummy mask that
``mask=None`` builds (``use_mask=False``, never read by the kernel) still fails
too; it waits on a ruling and is pinned here so it cannot pass silently.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
from _validation_capture import _capture

from torchlens.validation import validate_forward_pass
from torchlens.validation.core import _check_whether_func_on_saved_parents_yields_saved_tensor
from torchlens.validation.exemptions import STRUCTURAL_ARG_POSITIONS

ops = pytest.importorskip("torchvision.ops")

_WEIGHT_SLOT, _OFFSET_SLOT, _MASK_SLOT = 1, 2, 3


class _DeformBlock(nn.Module):
    """A deformable conv whose weight, offset and (optional) mask are computed tensors."""

    def __init__(self, with_mask: bool) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.with_mask = with_mask
        self.weight = nn.Parameter(torch.randn(4, 3, 3, 3))
        self.offset_conv = nn.Conv2d(3, 18, 3, padding=1)
        self.mask_conv = nn.Conv2d(3, 9, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.sigmoid(self.mask_conv(x)) if self.with_mask else None
        return ops.deform_conv2d(
            x, self.offset_conv(x), self.weight * 1.0, padding=(1, 1), mask=mask
        )


def _deform_op(with_mask: bool):
    trace, _ = _capture(_DeformBlock(with_mask), torch.randn(1, 3, 6, 6))
    op = [o for o in trace._layer_list if o.func_name == "deform_conv2d"][0]
    return trace, op


def _perturb(trace, op, slot: int):
    label = op.parent_arg_positions["args"][slot]
    return _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[label]
    )


def test_deform_conv2d_has_no_structural_slot() -> None:
    """Neither the weight (arg 1) nor the offset (arg 2) is exempted before replay."""

    assert "deform_conv2d" not in STRUCTURAL_ARG_POSITIONS


def test_live_weight_and_offset_edges_validate() -> None:
    """Both slots are perturbed and their real edges register."""

    trace, op = _deform_op(with_mask=True)
    for slot in (_WEIGHT_SLOT, _OFFSET_SLOT):
        result = _perturb(trace, op, slot)
        assert (result.decision, result.reason) == ("validated", "perturbation_changed"), slot


def test_dead_weight_and_offset_edges_fail() -> None:
    """A replay that ignores its inputs fails at the weight (once exempted) and the offset."""

    trace, op = _deform_op(with_mask=True)
    saved = op.out.detach().clone()
    object.__setattr__(op, "func", lambda *args, **kwargs: saved.clone())
    for slot in (_WEIGHT_SLOT, _OFFSET_SLOT):
        result = _perturb(trace, op, slot)
        assert (result.decision, result.reason) == ("failed", "perturbation_insensitive"), slot


def test_masked_deform_block_validates() -> None:
    """End to end, a deformable conv with a real mask validates."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(3):
            model = _DeformBlock(with_mask=True)
            assert validate_forward_pass(model, torch.randn(1, 3, 6, 6), seed=seed) is True


def test_unread_dummy_mask_still_fails() -> None:
    """``mask=None`` passes ``zeros((N, 1))`` with ``use_mask=False``; that edge still fails."""

    trace, op = _deform_op(with_mask=False)
    assert op.saved_args[13] is False
    result = _perturb(trace, op, _MASK_SLOT)
    assert (result.decision, result.reason) == ("failed", "perturbation_insensitive")
