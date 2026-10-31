"""``Tensor.set_`` receiver values never reach its output (AGT-2473).

``buf.set_(source)`` rebinds the receiver to the source's storage, so the
perturbation check must not demand that the receiver (arg 0) influence the
output, exactly as for ``copy_`` and ``fill_``. The carve-out is the arg-0
position only: the SOURCE at arg 1 carries every output value and a dead source
edge must still fail perturbation.

The other two overloads validate end to end. ``set_()`` empties the buffer and
``set_(storage, offset, size, stride)`` points it into another buffer's
storage; the restore puts each moved tensor back on its own storage before it
restores values, and the storage argument is saved by value at the call, so
replay reads the bytes the call read and a wrong offset or size still fails.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from _validation_capture import _capture, _quiet_validate
from torch import nn

import torchlens as tl
from torchlens._validation_state_snapshot import (
    restore_snapshot_if_changed,
    snapshot_state_for_restore,
)
from torchlens.validation.core import _check_perturbation_exemptions, validate_saved_outs
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


def _receiver_and_set_op(model: nn.Module) -> tuple[Any, Any, str]:
    """Capture ``model`` and return its trace, its ``set_`` op and the receiver label."""

    trace, _ = _capture(model, torch.rand(4))
    set_op = next(op for op in trace.layer_list if op.func_name == "set_")
    return trace, set_op, set_op.parent_arg_positions["args"][0]


@pytest.mark.parametrize("model_cls", [_SetToNothing, _SetToStorageView])
def test_set_overloads_without_a_source_tensor_exempt_only_the_receiver(
    model_cls: type[nn.Module],
) -> None:
    """``set_()`` and ``set_(storage, offset, size, stride)`` get the same arg-0 decision.

    The wrapper logs both overloads as ``set_`` ops whose only tensor parent is
    the receiver (a storage is not a graph tensor), and the receiver's values
    never reach the output.
    """

    trace, set_op, receiver = _receiver_and_set_op(model_cls())
    try:
        assert list(set_op.parents) == [receiver]
        assert _check_perturbation_exemptions(trace, set_op, [receiver]) is True
    finally:
        trace.cleanup()


def test_set_source_tensor_is_never_exempt() -> None:
    """The arg-1 source of ``set_(source)`` stays perturbation-tested."""

    trace, set_op, receiver = _receiver_and_set_op(_SetToSource())
    try:
        source = set_op.parent_arg_positions["args"][1]
        assert _check_perturbation_exemptions(trace, set_op, [receiver]) is True
        assert _check_perturbation_exemptions(trace, set_op, [source]) is False
    finally:
        trace.cleanup()


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


class _SetParameterToNothing(nn.Module):
    """``set_()`` on a parameter empties it."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.w.set_()
        return x * 2 + self.w.sum()


class _GrowBuffer(nn.Module):
    """``set_(source)`` with a larger source grows the buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.set_(torch.cat([x, x]) * 3)
        return x + self.b[:4] + self.b[4:]


class _SetToStorageViewThenOverwrite(_SetToStorageView):
    """The storage overload, then a write to the source storage after the read."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        with torch.no_grad():
            self.src.mul_(-1.0)
        return out


_SHAPE_CHANGING = [_SetToNothing, _SetParameterToNothing, _GrowBuffer, _SetToStorageView]


def _state_by_name(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return every parameter and buffer object by name."""

    return {**dict(model.named_parameters()), **dict(model.named_buffers())}


def _assert_restored(
    model: nn.Module, objects: dict[str, torch.Tensor], values: dict[str, torch.Tensor]
) -> None:
    """Assert the same tensor objects hold their own storage and pre-call values."""

    live = _state_by_name(model)
    assert live.keys() == objects.keys()
    for name, tensor in live.items():
        assert tensor is objects[name], name
        assert tensor.shape == values[name].shape, name
        assert torch.equal(tensor, values[name]), name
    pointers = [tensor.untyped_storage().data_ptr() for tensor in live.values()]
    assert len(set(pointers)) == len(pointers), "a restored tensor still aliases another"


@pytest.fixture
def failing_model_deepcopy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[type]]:
    """Make ``copy.deepcopy`` refuse the listed model classes (validation's live fallback)."""

    failing: list[type] = []
    real_deepcopy = copy.deepcopy

    def deepcopy(obj: Any, memo: Any = None, _nil: Any = None) -> Any:
        if failing and isinstance(obj, tuple(failing)):
            raise RuntimeError("model refuses deepcopy")
        return real_deepcopy(obj, memo)

    monkeypatch.setattr(copy, "deepcopy", deepcopy)
    yield failing


@pytest.mark.parametrize("model_cls", _SHAPE_CHANGING)
@pytest.mark.parametrize("copy_fails", [False, True], ids=["copy", "live-fallback"])
def test_set_overloads_validate_and_leave_the_model_restored(
    model_cls: type[nn.Module], copy_fails: bool, failing_model_deepcopy: list[type]
) -> None:
    """Validation returns True and the caller's tensors keep their objects, shapes and values.

    On the live fallback the forward runs on the caller's model, whose
    rebound tensor must go back on its own storage: a value-only restore
    refuses the new shape, or copies through the alias into another buffer.
    """

    if copy_fails:
        failing_model_deepcopy.append(model_cls)
    torch.manual_seed(0)
    model = model_cls()
    objects = _state_by_name(model)
    values = {name: tensor.detach().clone() for name, tensor in objects.items()}
    assert _quiet_validate(model, torch.rand(4)) is True
    _assert_restored(model, objects, values)


@pytest.mark.parametrize("model_cls", _SHAPE_CHANGING)
def test_restore_rebinds_a_moved_tensor_before_loading_values(model_cls: type[nn.Module]) -> None:
    """The snapshot restore puts a ``set_``-moved tensor back on its own storage."""

    torch.manual_seed(0)
    model = model_cls()
    objects = _state_by_name(model)
    values = {name: tensor.detach().clone() for name, tensor in objects.items()}
    snapshot = snapshot_state_for_restore(model)
    model(torch.rand(4))
    restore_snapshot_if_changed(
        model, snapshot, lambda module, state: module.load_state_dict(state)
    )
    _assert_restored(model, objects, values)


@pytest.mark.parametrize("model_cls", [_SetToNothing, _SetToStorageView])
def test_default_capture_leaves_a_set_buffer_on_its_own_storage(
    model_cls: type[nn.Module],
) -> None:
    """A default capture restores buffers, including one the forward moved with ``set_``."""

    torch.manual_seed(0)
    model = model_cls()
    objects = _state_by_name(model)
    values = {name: tensor.detach().clone() for name, tensor in objects.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, torch.rand(4))
    trace.cleanup()
    _assert_restored(model, objects, values)


def _storage_view_trace(model_cls: type[nn.Module] = _SetToStorageView) -> tuple[Any, Any, Any]:
    """Capture ``model_cls`` and return the trace, its ground truth and its ``set_`` op."""

    trace, ground_truth = _capture(model_cls(), torch.rand(4))
    set_op = next(op for op in trace.layer_list if op.func_name == "set_")
    return trace, ground_truth, set_op


def test_storage_argument_is_saved_by_value_at_the_call() -> None:
    """Replay reads the bytes the storage held at the call, not its later contents."""

    trace, ground_truth, set_op = _storage_view_trace(_SetToStorageViewThenOverwrite)
    try:
        saved_storage = set_op.saved_args[1]
        assert isinstance(saved_storage, torch.UntypedStorage)
        saved_values = torch.frombuffer(bytearray(bytes(saved_storage)), dtype=torch.float32)
        assert torch.equal(saved_values, torch.arange(8, dtype=torch.float32))
        assert torch.equal(set_op.out, torch.arange(2, 6, dtype=torch.float32))
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        assert status.state == "passed"
    finally:
        trace.cleanup()


def test_storage_view_end_to_end_with_a_later_source_write() -> None:
    """The overload validates when the forward rewrites the source after the read."""

    assert _quiet_validate(_SetToStorageViewThenOverwrite(), torch.rand(4)) is True


@pytest.mark.parametrize(
    ("position", "wrong_value"),
    [(2, 3), (3, (3,))],
    ids=["wrong-offset", "wrong-size"],
)
def test_wrong_storage_view_still_fails_replay(position: int, wrong_value: object) -> None:
    """A replay that reads the wrong offset or size of the saved storage fails."""

    trace, ground_truth, set_op = _storage_view_trace()
    try:
        assert validate_saved_outs(trace, [ground_truth], validate_metadata=False).state == (
            "passed"
        )
    finally:
        trace.cleanup()

    trace, ground_truth, set_op = _storage_view_trace()
    try:
        set_op.saved_args[position] = wrong_value
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        assert status.state == "failed"
        failure = get_validation_failure(trace)
        assert failure is not None
        assert failure.check == "forward_replay"
        assert failure.op_label is not None and failure.op_label.startswith("set_")
    finally:
        trace.cleanup()
