"""``Tensor.set_`` receiver values never reach its output (AGT-2473).

``buf.set_(source)`` rebinds the receiver to the source's storage, so the
perturbation check must not demand that the receiver (arg 0) influence the
output, exactly as for ``copy_`` and ``fill_``. The carve-out is the arg-0
position only: the SOURCE at arg 1 carries every output value and a dead source
edge must still fail perturbation.
"""

from __future__ import annotations

import torch
from _validation_capture import _capture, _quiet_validate
from torch import nn

from torchlens.validation.core import validate_saved_outs
from torchlens.validation.diagnostics import get_validation_failure


class _SetToSource(nn.Module):
    """Rebinds a buffer to a fresh source tensor, then reads it."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.set_(x * 3)
        return x + self.b


class _SetToNothing(nn.Module):
    """``set_()`` with no source empties the buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.set_()
        return x * 2 + self.b.sum()


class _SetToStorageView(nn.Module):
    """``set_(storage, offset, size, stride)`` views another tensor's storage."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))
        self.register_buffer("src", torch.arange(8, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.set_(self.src.untyped_storage(), 2, (4,), (1,))
        return x + self.b


def test_set_to_source_validates() -> None:
    """The receiver of ``set_(source)`` is not a value dependency."""

    assert _quiet_validate(_SetToSource(), torch.rand(4)) is True


def test_set_with_no_source_validates() -> None:
    """``set_()`` discards the receiver's values: nothing to perturb-test there."""

    assert _quiet_validate(_SetToNothing(), torch.rand(4)) is True


def test_set_to_a_storage_view_validates() -> None:
    """The storage overload replaces the receiver's values the same way."""

    assert _quiet_validate(_SetToStorageView(), torch.rand(4)) is True


def test_dead_set_source_edge_still_fails_perturbation() -> None:
    """A replay that ignores the SOURCE is caught: arg 1 stays perturbation-tested.

    The planted ``set_`` returns the saved output whatever source it is given,
    so plain replay still matches; only the perturbation of the source parent
    can expose it, and the arg-0 carve-out must not excuse it.
    """

    baseline, baseline_truth = _capture(_SetToSource(), torch.rand(4))
    try:
        status = validate_saved_outs(baseline, [baseline_truth], validate_metadata=False)
        assert status.state == "passed"
    finally:
        baseline.cleanup()

    trace, ground_truth = _capture(_SetToSource(), torch.rand(4))
    try:
        set_op = next(op for op in trace.layer_list if op.func_name == "set_")
        source_label = set_op.parent_arg_positions["args"][1]
        frozen = set_op.out.detach().clone()

        def source_blind_set_(receiver: torch.Tensor, *args: object, **kwargs: object):
            return receiver.set_(frozen.clone())

        source_blind_set_.__name__ = set_op.func.__name__
        set_op.func = source_blind_set_
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        assert status.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None
        assert failure.check == "perturbation"
        assert failure.op_label is not None and failure.op_label.startswith("set_")
        assert source_label in failure.extra["perturbed_parents"]
    finally:
        trace.cleanup()
