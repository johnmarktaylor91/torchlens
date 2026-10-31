"""``Tensor.resize_as_`` reads its template for its shape only.

``buf.resize_as_(the_template)`` is ``buf.resize_(the_template.size())``: the
template's elements never reach the output, so the perturbation check must not
demand that they do, exactly as for ``view_as`` and ``reshape_as``. The carve-out
is the arg-1 position only: the receiver at arg 0 keeps the element prefix the
output reads, so a receiver-blind replay must still fail perturbation, and the
sibling ``resize_`` gets no exemption at all.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from _validation_capture import _capture, _quiet_validate
from torch import nn

from torchlens.validation.core import _check_perturbation_exemptions, validate_saved_outs
from torchlens.validation.diagnostics import get_validation_failure


class _ResizeAsTemplate(nn.Module):
    """Shrinks a buffer to the shape of a tensor computed from the input."""

    def __init__(self, by_keyword: bool = False) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))
        self.by_keyword = by_keyword

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        template = x[:2] * 3
        with torch.no_grad():
            if self.by_keyword:
                self.b.resize_as_(the_template=template)
            else:
                self.b.resize_as_(template)
        return x + self.b.sum()


class _ResizeAsSelf(nn.Module):
    """``b.resize_as_(b)``: one parent fills the receiver slot and the template slot."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.resize_as_(self.b)
        return x + self.b.sum()


class _ResizeShrink(nn.Module):
    """The sibling ``resize_`` takes its size as integers; its receiver is its only parent."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.resize_(2)
        return x + self.b.sum()


def _op(trace: Any, func_name: str) -> Any:
    return next(op for op in trace.layer_list if op.func_name == func_name)


@pytest.mark.parametrize("by_keyword", [False, True])
def test_resize_as_template_validates(by_keyword: bool) -> None:
    """A template computed from the input is not a value dependency of ``resize_as_``."""

    torch.manual_seed(0)
    assert _quiet_validate(_ResizeAsTemplate(by_keyword), torch.rand(4)) is True


@pytest.mark.parametrize("by_keyword", [False, True])
def test_resize_as_exempts_only_the_template_slot(by_keyword: bool) -> None:
    """Arg 1 (or ``the_template=``) is exempt; the receiver at arg 0 is not."""

    trace, _ = _capture(_ResizeAsTemplate(by_keyword), torch.rand(4))
    try:
        op = _op(trace, "resize_as_")
        receiver = op.parent_arg_positions["args"][0]
        if by_keyword:
            template = op.parent_arg_positions["kwargs"]["the_template"]
        else:
            template = op.parent_arg_positions["args"][1]
        assert receiver != template
        assert _check_perturbation_exemptions(trace, op, [template]) is True
        assert _check_perturbation_exemptions(trace, op, [receiver]) is False
    finally:
        trace.cleanup()


def test_same_parent_in_receiver_and_template_slot_is_not_exempt() -> None:
    """``b.resize_as_(b)`` keeps its receiver value edge tested."""

    trace, _ = _capture(_ResizeAsSelf(), torch.rand(4))
    try:
        op = _op(trace, "resize_as_")
        (receiver,) = set(op.parent_arg_positions["args"].values())
        assert _check_perturbation_exemptions(trace, op, [receiver]) is False
    finally:
        trace.cleanup()


def test_resize_receiver_gets_no_exemption() -> None:
    """The sibling ``resize_`` is untouched: its receiver stays perturbation-tested."""

    trace, _ = _capture(_ResizeShrink(), torch.rand(4))
    try:
        op = _op(trace, "resize_")
        receiver = op.parent_arg_positions["args"][0]
        assert _check_perturbation_exemptions(trace, op, [receiver]) is False
    finally:
        trace.cleanup()


@pytest.mark.parametrize(
    ("model_cls", "func_name"), [(_ResizeAsTemplate, "resize_as_"), (_ResizeShrink, "resize_")]
)
def test_receiver_blind_resize_still_fails_perturbation(
    model_cls: type[nn.Module], func_name: str
) -> None:
    """A replay that ignores the RECEIVER is caught: arg 0 stays perturbation-tested.

    The planted op returns the saved output whatever receiver it is given, so
    plain replay still matches; only the perturbation of the receiver parent can
    expose it, and the template carve-out must not excuse it.
    """

    torch.manual_seed(0)
    baseline, baseline_truth = _capture(model_cls(), torch.rand(4))
    try:
        status = validate_saved_outs(baseline, [baseline_truth], validate_metadata=False)
        assert status.state == "passed"
    finally:
        baseline.cleanup()

    torch.manual_seed(0)
    trace, ground_truth = _capture(model_cls(), torch.rand(4))
    try:
        op = _op(trace, func_name)
        receiver_label = op.parent_arg_positions["args"][0]
        frozen = op.out.detach().clone()

        def receiver_blind(receiver: torch.Tensor, *args: object, **kwargs: object):
            return frozen.clone()

        receiver_blind.__name__ = op.func.__name__
        op.func = receiver_blind
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        assert status.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None
        assert failure.check == "perturbation"
        assert failure.func_name == func_name
        assert receiver_label in failure.extra["perturbed_parents"]
    finally:
        trace.cleanup()
