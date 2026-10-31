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
from torchlens._user_public_impls import _validate_forward_pass_torch
from torchlens._validation_state_snapshot import (
    restore_snapshot_if_changed,
    snapshot_state_for_restore,
)
from torchlens.options import CaptureOptions
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


class _SetToWholeStorage(nn.Module):
    """``set_(storage)`` views a whole storage from offset 0."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))
        self.register_buffer("src", torch.arange(8, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.set_(self.src.untyped_storage())
        return x + self.b[2:6]


class _ResizeBuffer(nn.Module):
    """``resize_`` grows a buffer in place (same storage, new size)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.resize_(8)
        return x * 2


class _UnsqueezeBuffer(nn.Module):
    """``unsqueeze_`` changes a buffer's shape on its own storage."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.unsqueeze_(0)
        return x + self.b[0]


class _TransposeBuffer(nn.Module):
    """``transpose_`` changes a buffer's shape and strides on its own storage."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.transpose_(0, 1)
        return x + self.b.reshape(4)


class _SetToStorageViewThenOverwrite(_SetToStorageView):
    """The storage overload, then a write to the source storage after the read."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        with torch.no_grad():
            self.src.mul_(-1.0)
        return out


_SHAPE_CHANGING = [
    _SetToNothing,
    _SetParameterToNothing,
    _GrowBuffer,
    _SetToStorageView,
    _SetToWholeStorage,
]
# On the copy path validation also re-traces the copy from its post-forward
# state. When that state already holds the binding ``set_`` applies (an empty
# buffer emptied again, a buffer already viewing the storage), the re-trace
# records the later read of the buffer differently and reports the model as
# stateful; see test_stateful_set_overloads_return_a_verdict_on_the_copy_path.
_RETRACE_STATEFUL = [_SetToNothing, _SetToStorageView, _SetToWholeStorage]

_VALIDATING_CASES = [
    (model_cls, copy_fails)
    for model_cls in _SHAPE_CHANGING
    for copy_fails in (False, True)
    if copy_fails or model_cls not in _RETRACE_STATEFUL
]


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


@pytest.mark.parametrize(("model_cls", "copy_fails"), _VALIDATING_CASES)
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


@pytest.mark.parametrize("model_cls", _RETRACE_STATEFUL)
def test_stateful_set_overloads_return_a_verdict_on_the_copy_path(
    model_cls: type[nn.Module],
) -> None:
    """Validation returns a verdict, never raises, and replay itself passes.

    The only failure allowed is the stateful-model re-trace finding (the
    re-trace runs from the copy's post-forward state; replay of the first
    capture passed before it). The caller's tensors are untouched either way.
    """

    torch.manual_seed(0)
    model = model_cls()
    objects = _state_by_name(model)
    values = {name: tensor.detach().clone() for name, tensor in objects.items()}
    traces: list[Any] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verdict = _validate_forward_pass_torch(model, torch.rand(4), _trace_observer=traces.append)
    assert isinstance(verdict, bool)
    (trace,) = traces
    failure = get_validation_failure(trace)
    if verdict:
        assert failure is None
    else:
        assert failure is not None
        assert failure.check == "retrace_mismatch"
    _assert_restored(model, objects, values)


@pytest.mark.parametrize(
    "model_cls", [*_SHAPE_CHANGING, _ResizeBuffer, _UnsqueezeBuffer, _TransposeBuffer]
)
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


def _storage_view_trace(
    model_cls: type[nn.Module] = _SetToStorageView,
) -> tuple[nn.Module, Any, Any, Any]:
    """Capture a fresh ``model_cls`` and return it, the trace, its ground truth and its ``set_`` op.

    The ground truth runs on a copy, so the capture starts from the model's
    initial state (``_capture`` runs the model itself first).
    """

    model = model_cls()
    x = torch.rand(4)
    ground_truth = copy.deepcopy(model)(x)
    trace = tl.trace(model, x, capture=CaptureOptions(layers_to_save="all", save_arg_values=True))
    set_op = next(op for op in trace.layer_list if op.func_name == "set_")
    return model, trace, ground_truth, set_op


def test_storage_argument_is_saved_by_value_at_the_call() -> None:
    """Replay reads the bytes the storage held at the call, not its later contents."""

    model, trace, ground_truth, set_op = _storage_view_trace(_SetToStorageViewThenOverwrite)
    try:
        saved_storage = set_op.saved_args[1]
        assert isinstance(saved_storage, torch.UntypedStorage)
        assert saved_storage.data_ptr() != model.src.untyped_storage().data_ptr()
        with torch.no_grad():
            model.src.zero_()
        saved_values = torch.frombuffer(bytearray(bytes(saved_storage)), dtype=torch.float32)
        assert torch.equal(saved_values, torch.arange(8, dtype=torch.float32))
        assert torch.equal(set_op.out, torch.arange(2, 6, dtype=torch.float32))
        status = validate_saved_outs(trace, [ground_truth], validate_metadata=False)
        assert status.state == "passed"
    finally:
        trace.cleanup()


def test_storage_view_end_to_end_with_a_later_source_write(
    failing_model_deepcopy: list[type],
) -> None:
    """The overload validates when the forward rewrites the source after the read.

    Run on the live fallback, where the verdict is replay's alone (no re-trace):
    replay must read the storage's bytes from the call, not the rewritten ones.
    """

    failing_model_deepcopy.append(_SetToStorageViewThenOverwrite)
    torch.manual_seed(0)
    model = _SetToStorageViewThenOverwrite()
    objects = _state_by_name(model)
    values = {name: tensor.detach().clone() for name, tensor in objects.items()}
    assert _quiet_validate(model, torch.rand(4)) is True
    _assert_restored(model, objects, values)


@pytest.mark.parametrize(
    ("position", "wrong_value"),
    [(2, 3), (3, (3,))],
    ids=["wrong-offset", "wrong-size"],
)
def test_wrong_storage_view_still_fails_replay(position: int, wrong_value: object) -> None:
    """A replay that reads the wrong offset or size of the saved storage fails."""

    _model, trace, ground_truth, set_op = _storage_view_trace()
    try:
        assert validate_saved_outs(trace, [ground_truth], validate_metadata=False).state == (
            "passed"
        )
    finally:
        trace.cleanup()

    _model, trace, ground_truth, set_op = _storage_view_trace()
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


@pytest.mark.parametrize("model_cls", [_UnsqueezeBuffer, _TransposeBuffer])
def test_in_place_geometry_writes_return_a_verdict_and_restore(
    model_cls: type[nn.Module], failing_model_deepcopy: list[type]
) -> None:
    """Other in-place metadata writes reach the same restore and must not raise it.

    On the live fallback the caller's buffer itself changes shape or strides;
    the restore puts it back on its recorded geometry.
    """

    failing_model_deepcopy.append(model_cls)
    torch.manual_seed(0)
    model = model_cls()
    objects = _state_by_name(model)
    values = {name: tensor.detach().clone() for name, tensor in objects.items()}
    assert isinstance(_quiet_validate(model, torch.rand(4)), bool)
    _assert_restored(model, objects, values)
    assert model.b.stride() == values["b"].stride()
