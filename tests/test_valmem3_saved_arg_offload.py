"""Forward validation keeps its own saved-argument copies on ``output_device``.

Validation captures with ``save_arg_values=True``. With an ``output_device``
other than ``"same"``, its own capture now keeps the retained copy of every
non-parameter tensor argument on that device, tagged with the device the op
consumed it on, and replay holds one layer's tagged arguments back on those
devices while it checks that layer. These tests pin:

* the capture-side copy, tag and save-budget device (CPU, with ``meta`` as a
  stand-in second device for the plumbing only);
* the replay-side swap: exact copies with ``requires_grad`` kept, host copies
  restored on every exit, untagged values untouched;
* with ``output_device="same"``, or a CPU model under ``"cpu"``, nothing moves
  and nothing is tagged; a user's own trace never moves saved arguments;
* on an accelerator: no saved-argument copy beyond parameter aliases stays on
  the device, verdicts equal the ``"same"`` run, a model that moves activations
  between devices mid-forward validates, and corrupting a saved argument (or a
  saved parameter argument, which is the replay copy's live parameter) fails
  validation with the same verdict and failure kind as with ``"same"``.
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest import mock

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.user_funcs as user_funcs
from torchlens import _user_public_impls, _validation_argument_references as arg_references
from torchlens.backends.torch import _ops_retention
from torchlens.options import CaptureOptions
from torchlens.validation import _replay_device
from torchlens.validation.diagnostics import get_validation_diagnostics, get_validation_failure

_TAG = arg_references.CONSUMED_DEVICE_ATTR


def _accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return None


_ACCEL = _accelerator()
_NEEDS_ACCEL = pytest.mark.skipif(_ACCEL is None, reason="needs a CUDA or MPS device")


class _Offsets(nn.Module):
    """Linears, a plain tensor attribute (an unattributed argument) and a buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(6, 5)
        self.out = nn.Linear(5, 3)
        self.offset = torch.linspace(-1.0, 1.0, 5)
        self.register_buffer("scale", torch.full((3,), 0.5))

    def _apply(self, fn: Any, recurse: bool = True) -> nn.Module:
        super()._apply(fn, recurse)
        self.offset = fn(self.offset)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.lin(x)) + self.offset
        return self.out(hidden) * self.scale


class _DtypeHop(nn.Module):
    """An activation changes dtype mid-forward and back (CPU stand-in for a device hop)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide = torch.relu(self.lin(x)).to(torch.float64)
        return (wide * 3.0 + 1.0).to(torch.float32)


class _DeviceHop(nn.Module):
    """Activations move from the accelerator to the CPU and back inside forward."""

    def __init__(self, accelerator: str) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.first = nn.Linear(6, 5).to(accelerator)
        self.second = nn.Linear(5, 4)
        self.offset = torch.linspace(0.0, 1.0, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.first(x)).to("cpu")
        moved = self.second(hidden) + self.offset
        return torch.tanh(moved.to(self.accelerator))


def _validate(model: nn.Module, x: torch.Tensor, output_device: str = "same") -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(
            tl.validate(model, x, scope="forward", random_seed=0, output_device=output_device)
        )


def _validation_capture(
    model: nn.Module, x: torch.Tensor, output_device: str, inspect: Any
) -> tuple[bool, list[Any]]:
    """Validate, calling ``inspect(trace, model)`` on validation's own capture before replay."""

    real_runner = user_funcs._run_model_and_save_specified_outs
    failures: list[Any] = []

    def runner(*args: Any, **kwargs: Any) -> Any:
        trace = real_runner(*args, **kwargs)
        if kwargs["save_arg_values"]:
            inspect(trace, kwargs["model"])
        return trace

    def observe(trace: Any) -> None:
        failure = get_validation_failure(trace) if trace is not None else None
        failures.append(None if failure is None else failure.check)
        failures.extend(
            item.check
            for item in get_validation_diagnostics(trace) or ()
            if item.check == arg_references.FALLBACK_CHECK
        )

    with mock.patch.object(user_funcs, "_run_model_and_save_specified_outs", runner):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            verdict = _user_public_impls._validate_forward_pass_torch(
                model, x, random_seed=0, output_device=output_device, _trace_observer=observe
            )
    return bool(verdict), failures


def _saved_arg_tensors(trace: Any, outputs: bool = True) -> list[torch.Tensor]:
    """Tensor leaves of every op's saved arguments (``outputs=False`` skips output nodes).

    A synthetic output node's ``saved_args`` hold a reference to the forward's
    returned tensor, not a ``save_arg_values`` copy.
    """

    tensors: list[torch.Tensor] = []
    for op in trace.layer_list:
        if not outputs and getattr(op, "is_output", False):
            continue
        values = [*(op.saved_args or ()), *(op.saved_kwargs or {}).values()]
        for value in values:
            items = value if type(value) in (list, tuple) else (value,)
            tensors.extend(item for item in items if isinstance(item, torch.Tensor))
    return tensors


# ---------------------------------------------------------------------------
# Capture side (CPU)
# ---------------------------------------------------------------------------


def test_offload_copies_a_plain_tensor_once_and_tags_its_consumed_device() -> None:
    references = arg_references.ParameterArgumentReferences({}, None, "meta")
    source = torch.randn(3, 4, requires_grad=True) * 1.0
    moved = references.offload(source)
    assert moved is not None and moved.device.type == "meta"
    assert getattr(moved, _TAG) == source.device
    assert moved.requires_grad and moved.shape == source.shape and moved.dtype == source.dtype


def test_offload_declines_parameter_storage_subclasses_and_same_device() -> None:
    model = nn.Linear(3, 2)
    owners = arg_references._parameter_owners(model)
    references = arg_references.ParameterArgumentReferences(owners, None, "meta")
    assert references.offload(model.weight.detach()) is None
    assert references.offload(model.weight) is None
    assert references.offload(torch.randn(2).to_sparse()) is None
    assert (
        arg_references.ParameterArgumentReferences({}, None, "cpu").offload(torch.ones(2)) is None
    )
    assert not arg_references.ParameterArgumentReferences({}, None).offloads


def test_retained_copy_and_budget_device_follow_the_offload_target() -> None:
    references = arg_references.ParameterArgumentReferences({}, None, "meta")
    bare, item, nested = torch.ones(2), torch.ones(3), torch.ones(4)
    t_args = (bare, [item, 2], [[nested]])
    token = arg_references._ACTIVE.set(references)
    try:
        devices = _ops_retention._saved_arg_retention_devices(t_args, {"k": item})
        retained = [_ops_retention._copy_saved_arg_tree(arg, True) for arg in t_args]
        pre_call = _ops_retention._copy_saved_arg_tree(bare)
    finally:
        arg_references._ACTIVE.reset(token)
    assert devices == {id(bare): torch.device("meta"), id(item): torch.device("meta")}
    assert retained[0].device.type == "meta" and retained[1][0].device.type == "meta"
    assert retained[1][1] == 2 and retained[2][0][0].device.type == "cpu"
    assert pre_call.device.type == "cpu" and not hasattr(pre_call, _TAG)


def test_outside_validation_the_retained_copy_is_the_plain_copy() -> None:
    value = torch.ones(3)
    copied = _ops_retention._copy_saved_arg_tree(value, True)
    assert copied.device == value.device and copied is not value and not hasattr(copied, _TAG)
    assert _ops_retention._saved_arg_retention_devices((value,), {}) == {}


def test_offload_is_off_for_same_and_outside_a_validation_session() -> None:
    model = nn.Linear(2, 2)
    with arg_references.parameter_argument_references(model, True, "cpu") as outside:
        assert not outside.offloads and arg_references.active_argument_references() is None
    token = arg_references._SESSION.set(arg_references._ReferenceSession())
    try:
        with arg_references.parameter_argument_references(model, True, "same") as same:
            assert not same.offloads
        with arg_references.parameter_argument_references(model, False, "cpu") as live:
            assert live.offloads and arg_references.active_argument_references() is live
    finally:
        arg_references._SESSION.reset(token)


# ---------------------------------------------------------------------------
# Replay side (CPU): the per-layer swap
# ---------------------------------------------------------------------------


class _FakeOp:
    def __init__(self, saved_args: list[Any], saved_kwargs: dict[str, Any]) -> None:
        self.saved_args = saved_args
        self.saved_kwargs = saved_kwargs


class _FakeTrace:
    def __init__(self, op: _FakeOp) -> None:
        self.layer_logs: dict[str, Any] = {}
        self.layer_dict_all_keys = {"op": op}


def _tagged(value: torch.Tensor) -> torch.Tensor:
    setattr(value, _TAG, value.device)
    return value


def test_swap_holds_exact_copies_and_restores_the_host_copies() -> None:
    host = _tagged(torch.randn(3, requires_grad=True) * 1.0)
    pair = (_tagged(torch.arange(4.0)), 7)
    plain = torch.ones(2)
    op = _FakeOp([host, pair, plain, "x"], {"w": _tagged(torch.full((2,), 2.0))})
    host_kwarg = op.saved_kwargs["w"]
    with _replay_device.saved_args_on_consumed_devices(_FakeTrace(op), "op"):
        held = op.saved_args[0]
        assert held is not host and torch.equal(held, host) and held.requires_grad
        assert type(op.saved_args[1]) is tuple and op.saved_args[1][1] == 7
        assert op.saved_args[1][0] is not pair[0] and torch.equal(op.saved_args[1][0], pair[0])
        assert op.saved_args[2] is plain and op.saved_args[3] == "x"
        assert op.saved_kwargs["w"] is not host_kwarg
    assert op.saved_args[0] is host and op.saved_args[1] is pair
    assert op.saved_kwargs["w"] is host_kwarg


def test_swap_restores_the_host_copies_when_the_check_raises() -> None:
    host = _tagged(torch.ones(2))
    op = _FakeOp([host], {})
    with pytest.raises(RuntimeError, match="check failed"):
        with _replay_device.saved_args_on_consumed_devices(_FakeTrace(op), "op"):
            assert op.saved_args[0] is not host
            raise RuntimeError("check failed")
    assert op.saved_args[0] is host


def test_swap_changes_nothing_without_tags() -> None:
    values = [torch.ones(2), (torch.zeros(1),), None]
    op = _FakeOp(list(values), {"k": torch.ones(1)})
    with _replay_device.saved_args_on_consumed_devices(_FakeTrace(op), "op"):
        assert all(a is b for a, b in zip(op.saved_args, values, strict=True))


# ---------------------------------------------------------------------------
# CPU models: nothing moves, verdicts unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output_device", ["same", "cpu"])
def test_cpu_validation_tags_no_saved_argument(output_device: str) -> None:
    torch.manual_seed(0)
    tagged: list[int] = []

    def inspect(trace: Any, model: Any) -> None:
        tagged.append(sum(hasattr(t, _TAG) for t in _saved_arg_tensors(trace)))

    verdict, failures = _validation_capture(_Offsets(), torch.randn(2, 6), output_device, inspect)
    assert verdict and failures == [None] and tagged == [0]


@pytest.mark.parametrize("output_device", ["same", "cpu"])
def test_a_mid_forward_dtype_hop_validates(output_device: str) -> None:
    torch.manual_seed(0)
    assert _validate(_DtypeHop(), torch.randn(3, 4), output_device)


# ---------------------------------------------------------------------------
# Accelerator
# ---------------------------------------------------------------------------


@_NEEDS_ACCEL
def test_validation_capture_holds_no_saved_argument_copy_on_the_accelerator() -> None:
    torch.manual_seed(0)
    assert _ACCEL is not None
    on_device: list[int] = []
    tagged: list[int] = []

    def inspect(trace: Any, model: Any) -> None:
        aliases = {p.untyped_storage().data_ptr() for p in model.parameters()}
        seen: set[int] = set()
        held = 0
        output_refs = {
            t.untyped_storage().data_ptr()
            for op in trace.layer_list
            if getattr(op, "is_output", False)
            for t in op.saved_args or ()
            if isinstance(t, torch.Tensor)
        }
        assert output_refs
        for tensor in _saved_arg_tensors(trace, outputs=False):
            pointer = tensor.untyped_storage().data_ptr()
            if tensor.device.type == _ACCEL and pointer not in aliases and pointer not in seen:
                seen.add(pointer)
                held += tensor.untyped_storage().nbytes()
        on_device.append(held)
        tagged.append(sum(hasattr(t, _TAG) for t in _saved_arg_tensors(trace)))

    model, x = _Offsets().to(_ACCEL), torch.randn(4, 6, device=_ACCEL)
    verdict, failures = _validation_capture(model, x, "cpu", inspect)
    assert verdict and failures == [None]
    assert on_device == [0] and tagged[0] > 0


@_NEEDS_ACCEL
def test_same_output_device_keeps_saved_arguments_on_the_accelerator() -> None:
    torch.manual_seed(0)
    assert _ACCEL is not None
    devices: list[set[str]] = []

    def inspect(trace: Any, model: Any) -> None:
        devices.append({t.device.type for t in _saved_arg_tensors(trace)})
        assert not any(hasattr(t, _TAG) for t in _saved_arg_tensors(trace))

    model, x = _Offsets().to(_ACCEL), torch.randn(4, 6, device=_ACCEL)
    verdict, _ = _validation_capture(model, x, "same", inspect)
    assert verdict and devices == [{_ACCEL}]


@_NEEDS_ACCEL
def test_a_user_trace_keeps_saved_arguments_on_each_ops_device() -> None:
    torch.manual_seed(0)
    assert _ACCEL is not None
    model, x = _Offsets().to(_ACCEL), torch.randn(4, 6, device=_ACCEL)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", save_arg_values=True, output_device="cpu"),
    )
    saved = _saved_arg_tensors(trace)
    assert saved and {t.device.type for t in saved} == {_ACCEL}
    assert not any(hasattr(t, _TAG) for t in saved)


@_NEEDS_ACCEL
@pytest.mark.parametrize("model_name", ["offsets", "device_hop", "dtype_hop"])
def test_accelerator_verdicts_equal_the_same_device_run(model_name: str) -> None:
    assert _ACCEL is not None
    torch.manual_seed(0)
    if model_name == "offsets":
        model: nn.Module = _Offsets().to(_ACCEL)
        x = torch.randn(4, 6, device=_ACCEL)
    elif model_name == "device_hop":
        model, x = _DeviceHop(_ACCEL), torch.randn(4, 6, device=_ACCEL)
    else:
        model, x = _DtypeHop().to(_ACCEL), torch.randn(3, 4, device=_ACCEL)
    assert _validate(model, x, "same") is True
    assert _validate(model, x, "cpu") is True


def _corrupt_relu_input(trace: Any, model: Any) -> None:
    relu = next(op for op in trace.layer_list if op.func_name == "relu")
    relu.saved_args[0].add_(1.0)


def _corrupt_scale_argument(trace: Any, model: Any) -> None:
    mul = next(op for op in trace.layer_list if op.func_name in {"mul", "__mul__"})
    mul.saved_args[1].add_(1.0)


def _corrupt_saved_weight(trace: Any, model: Any) -> None:
    linear = next(op for op in trace.layer_list if op.func_name == "linear")
    with torch.no_grad():
        linear.saved_args[1].add_(1.0)


@_NEEDS_ACCEL
@pytest.mark.parametrize(
    "corrupt", [_corrupt_relu_input, _corrupt_scale_argument, _corrupt_saved_weight]
)
def test_a_corrupted_saved_argument_fails_as_with_same(corrupt: Any) -> None:
    assert _ACCEL is not None
    results = {}
    for output_device in ("same", "cpu"):
        torch.manual_seed(0)
        model, x = _Offsets().to(_ACCEL), torch.randn(4, 6, device=_ACCEL)
        results[output_device] = _validation_capture(model, x, output_device, corrupt)
    assert results["cpu"] == results["same"]
    assert results["cpu"][0] is False


def test_nested_swaps_move_nothing_twice_and_restore_on_every_exit() -> None:
    host = _tagged(torch.ones(3))
    op = _FakeOp([host], {})
    trace = _FakeTrace(op)
    with _replay_device.saved_args_on_consumed_devices(trace, "op"):
        outer = op.saved_args[0]
        with _replay_device.saved_args_on_consumed_devices(trace, "op"):
            assert op.saved_args[0] is outer
        assert op.saved_args[0] is outer
        with pytest.raises(RuntimeError, match="inner"):
            with _replay_device.saved_args_on_consumed_devices(trace, "op"):
                raise RuntimeError("inner")
        assert op.saved_args[0] is outer
    assert op.saved_args[0] is host


def _grad_models() -> list[tuple[str, Any, tuple[int, ...]]]:
    from example_models import MAMLInnerLoop
    from test_autograd_grad_boundary import (
        _IntermediateInput,
        _SecondOrder,
        _TwoCallsDifferentArity,
        _TwoLosses,
    )

    return [
        ("maml", MAMLInnerLoop, (4, 8)),
        ("two_losses", _TwoLosses, (2, 3)),
        ("intermediate_input", _IntermediateInput, (2, 4)),
        ("second_order", _SecondOrder, (2, 3)),
        ("two_arities", _TwoCallsDifferentArity, (3,)),
    ]


def _no_corruption(trace: Any, model: Any) -> None:
    del trace, model


def _corrupt_matmul_weight(trace: Any, model: Any) -> None:
    matmul = next(op for op in trace.layer_list if op.func_name in {"matmul", "__matmul__"})
    with torch.no_grad():
        matmul.saved_args[1].add_(1.0)


@_NEEDS_ACCEL
@pytest.mark.parametrize(
    "index",
    range(5),
    ids=["maml", "two_losses", "intermediate_input", "second_order", "two_arities"],
)
def test_in_forward_autograd_grad_models_match_the_same_device_run(index: int) -> None:
    assert _ACCEL is not None
    _, factory, shape = _grad_models()[index]
    results = {}
    for output_device in ("same", "cpu"):
        torch.manual_seed(0)
        model, x = factory().to(_ACCEL), torch.randn(*shape, device=_ACCEL)
        results[output_device] = _validation_capture(model, x, output_device, _no_corruption)
    assert results["cpu"] == results["same"]


@_NEEDS_ACCEL
@pytest.mark.parametrize("corrupt", [_corrupt_relu_input, _corrupt_matmul_weight])
def test_a_corrupted_autograd_grad_model_fails_as_with_same(corrupt: Any) -> None:
    from example_models import MAMLInnerLoop

    assert _ACCEL is not None
    results = {}
    for output_device in ("same", "cpu"):
        torch.manual_seed(0)
        model, x = MAMLInnerLoop().to(_ACCEL), torch.randn(4, 8, device=_ACCEL)
        results[output_device] = _validation_capture(model, x, output_device, corrupt)
    assert results["cpu"] == results["same"]
    assert results["cpu"][0] is False
