"""Stored non-tensor arguments keep the value the op saw at call time.

A list, dict or set argument is owned by the caller, who may mutate it after
the call. The op record must hold a snapshot taken at op time, not the
caller's object, or the record (and every consumer that reads it after the
forward pass, such as loop detection's structural argument signature) sees a
value the op never ran with.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import FrameType

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from support.structural_grouper import legacy_structural_grouper  # noqa: F401

import torchlens as tl


def _ops_named(trace: tl.Trace, func_name: str) -> list[object]:
    """Return the trace's ops whose function name is ``func_name``, in order."""
    return [op for op in trace.ops if op.func_name == func_name]


def test_non_tensor_list_arg_is_snapshotted_at_call_time() -> None:
    """A list argument mutated after the call does not rewrite the recorded op."""

    class M(torch.nn.Module):
        """Mutates its shape list after the reshape."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Reshape, then mutate the shape list."""
            shape = [2, 12]
            out = x.reshape(shape)
            shape[0], shape[1] = 4, 6
            return out.sum()

    trace = tl.trace(M(), torch.randn(4, 6))
    (op,) = _ops_named(trace, "reshape")
    assert op.non_tensor_pos_args == [[2, 12]]
    assert op.func_non_tensor_args == [[2, 12]]


def test_list_kwarg_is_snapshotted_at_call_time() -> None:
    """A list passed by keyword keeps its call-time value in ``non_tensor_kwargs``."""

    class M(nn.Module):
        """Pads with a keyword list, then mutates the list."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Pad by keyword, then mutate the pad list."""
            pad = [1, 1]
            out = F.pad(x, pad=pad)
            pad[0], pad[1] = 3, 3
            return out.sum()

    trace = tl.trace(M(), torch.randn(2, 4))
    (op,) = _ops_named(trace, "pad")
    assert op.non_tensor_kwargs["pad"] == [1, 1]
    assert [1, 1] in op.func_non_tensor_args
    assert [3, 3] not in op.func_non_tensor_args


def test_nested_list_and_tuple_of_list_args_are_snapshotted() -> None:
    """Inner lists of a nested list and of an index tuple are copied too."""

    class M(nn.Module):
        """Builds from a nested list and indexes with a tuple holding a list."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Use both containers, then mutate their inner lists."""
            nested = [[1.0, 2.0], [3.0, 4.0]]
            built = x.new_tensor(nested)
            index = [0, 1]
            picked = x[(index,)]
            nested[0][0] = 9.0
            index[0] = 1
            return built.sum() + picked.sum()

    trace = tl.trace(M(), torch.randn(3, 2))
    (built_op,) = _ops_named(trace, "new_tensor")
    assert built_op.non_tensor_pos_args == [[[1.0, 2.0], [3.0, 4.0]]]
    (index_op,) = _ops_named(trace, "__getitem__")
    assert index_op.non_tensor_pos_args == [([0, 1],)]


class _MutatingPadConv(nn.Module):
    """One kernel at one call site; the padding list is mutated between passes."""

    def __init__(self) -> None:
        """Hold one shared kernel."""
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conv twice from one call site, mutating ``padding`` in between."""
        pad = [0, 0]
        total = x.new_zeros(())
        for _ in range(2):
            total = total + F.conv2d(x, self.kernel, padding=pad).sum()
            pad[0], pad[1] = 1, 1
        return total


class _FreshPadConv(nn.Module):
    """The same calls as ``_MutatingPadConv`` with a fresh padding list per pass."""

    def __init__(self) -> None:
        """Hold one shared kernel."""
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conv twice from one call site with immutable-in-practice paddings."""
        total = x.new_zeros(())
        for pad in ([0, 0], [1, 1]):
            total = total + F.conv2d(x, self.kernel, padding=pad).sum()
        return total


def _conv_structure(model: nn.Module) -> list[tuple[str, str, list[int]]]:
    """Return (label, layer label, recorded padding) for each conv op of one capture."""
    torch.manual_seed(0)
    trace = tl.trace(model, torch.randn(1, 1, 8, 8))
    return [
        (op.label, op.layer_label, op.non_tensor_kwargs["padding"])
        for op in _ops_named(trace, "conv2d")
    ]


def _assert_mutation_model_groups_like_control() -> None:
    """The mutating model records and groups exactly like its fresh-list control."""
    mutated = _conv_structure(_MutatingPadConv())
    control = _conv_structure(_FreshPadConv())
    assert [padding for _, _, padding in mutated] == [[0, 0], [1, 1]]
    assert mutated == control


def test_loop_detection_on_mutation_model_matches_control() -> None:
    """Call-site grouping sees the call-time paddings, not the final list value."""
    _assert_mutation_model_groups_like_control()


@pytest.mark.usefixtures("legacy_structural_grouper")
def test_structural_grouper_on_mutation_model_matches_control() -> None:
    """The structural argument signature splits the passes as the control does."""
    _assert_mutation_model_groups_like_control()


def test_snapshot_copies_containers_and_keeps_tensors_and_scalars() -> None:
    """Dicts, sets and nested lists are copied; tensors and scalars pass by reference."""
    from torchlens._capture._arg_snapshot import snapshot_mutable_arg

    tensor = torch.ones(2)
    inner = [1, 2]
    value = {"a": inner, "b": {3, 4}, "c": (inner, tensor), "d": 1.5}
    snap = snapshot_mutable_arg(value)
    inner.append(5)
    value["b"].add(6)
    value["e"] = 0
    assert snap == {"a": [1, 2], "b": {3, 4}, "c": ([1, 2], tensor), "d": 1.5}
    assert snap["c"][1] is tensor
    plain = (1, "x", tensor, None)
    assert snapshot_mutable_arg(plain) is plain
    assert snapshot_mutable_arg(tensor) is tensor


def test_snapshot_handles_self_referencing_list() -> None:
    """A list that contains itself is copied once, keeping the cycle."""
    from torchlens._capture._arg_snapshot import snapshot_mutable_arg

    loop: list[object] = [1]
    loop.append(loop)
    snap = snapshot_mutable_arg(loop)
    assert snap is not loop
    assert snap[1] is snap


def _snapshot_module_calls(run: Callable[[], object]) -> list[str]:
    """Return the names of the snapshot-module functions entered while ``run`` runs."""
    from torchlens._capture import _arg_snapshot

    module_file = _arg_snapshot.__file__
    calls: list[str] = []

    def profile(frame: FrameType, event: str, _arg: object) -> None:
        """Record each Python-level call into the snapshot module."""
        if event == "call" and frame.f_code.co_filename == module_file:
            calls.append(frame.f_code.co_name)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        run()
    finally:
        sys.setprofile(previous)
    return calls


def test_common_arguments_cost_no_snapshot_calls() -> None:
    """Scalars, flat tuples, sizes, dtypes and empty kwargs cost no call past the split."""
    from torchlens._capture._arg_snapshot import split_non_tensor_args

    shape = (2, 3, 4)
    size = torch.Size([2, 3])
    args = (torch.ones(1), shape, 1, 2.5, "mean", None, slice(None), Ellipsis, size, torch.float32)
    result: list[tuple[list[object], dict[str, object]]] = []
    calls = _snapshot_module_calls(
        lambda: result.append(
            split_non_tensor_args(args, {}, lambda arg: isinstance(arg, torch.Tensor))
        )
    )
    assert calls == ["split_non_tensor_args"]
    ((positional, keyword),) = result
    assert positional == [shape, 1, 2.5, "mean", None, slice(None), Ellipsis, size, torch.float32]
    assert positional[0] is shape
    assert positional[7] is size
    assert keyword == {}


def test_capture_of_scalar_and_tuple_args_only_splits_once_per_op() -> None:
    """Ops with no mutable argument and no transform config enter only the split."""

    class M(nn.Module):
        """Reshape, permute and softmax with scalar, tuple and keyword-scalar arguments."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run three ops whose non-tensor arguments are all immutable."""
            return F.softmax(x.reshape(2, 12).permute((1, 0)), dim=-1).sum()

    x = torch.randn(4, 6)
    tl.trace(M(), x)  # warm one-time instrumentation
    calls = _snapshot_module_calls(lambda: tl.trace(M(), x))
    assert calls
    assert set(calls) == {"split_non_tensor_args"}


def test_snapshot_keeps_shared_inner_lists_shared() -> None:
    """One inner list reached twice is copied once and stays shared in the copy."""
    from torchlens._capture._arg_snapshot import snapshot_mutable_arg

    inner = [1, 2]
    outer = [inner, inner, (inner,), {"k": inner}]
    snap = snapshot_mutable_arg(outer)
    inner.append(3)
    assert snap == [[1, 2], [1, 2], ([1, 2],), {"k": [1, 2]}]
    assert snap[0] is snap[1] is snap[2][0] is snap[3]["k"]
    assert snap[0] is not inner
