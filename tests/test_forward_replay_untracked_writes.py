"""Forward replay of call arguments that a later write changes outside any recorded op.

Two capture bugs made isolated replay disagree with the saved out:

* A slice argument holding a tensor bound (``x[i : i + n]`` with a tensor ``i``) was
  saved by reference, so a later ``i += n`` moved the saved bounds and replay indexed
  a different range (an empty one, for the usual running-offset loop).
* The deferred payload (clone-on-write alias) of a tensor handed to a train-mode
  ``batch_norm`` / ``instance_norm`` as running statistics was not copied out before
  the call updated those statistics, so the producer's saved out showed the updated
  value rather than the value it produced.
* The same deferred alias of a buffer that code outside torch dispatch then filled
  through a zero-copy export (``data_ptr``, a storage handle, a NumPy view, DLPack),
  as warp kernels do, showed the later write; exports now copy out pending aliases.

The consumer of a foreign-filled buffer reads a value no recorded op produced, so its
per-child version snapshot differs from the buffer producer's saved out. Forward
validation accepts that snapshot only because capture marked it as a foreign write for
the exact argument slots the buffer was bound to; a planted swap of operands involving
a foreign-filled buffer still fails.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
import torchlens.backends.torch._ops_argument_fields as _argument_fields
import torchlens.backends.torch._wrapper_session as _wrapper_session
from torchlens._utils import tensor_utils as _tu
from torchlens._utils.arg_handling import copy_arg_tree
from torchlens._utils.tensor_utils import tensor_nanequal
from torchlens.validation._core import _op_for_validation_label


@contextlib.contextmanager
def _payload_clone_mode(defer: bool) -> Iterator[None]:
    """Force deferred (clone-on-write) or eager payload clones inside the block."""

    prev_tu, prev_w = _tu._DEFER_ENABLED, _wrapper_session._COW_ENABLED
    _tu._DEFER_ENABLED = defer
    _wrapper_session._COW_ENABLED = defer
    try:
        yield
    finally:
        _tu._DEFER_ENABLED = prev_tu
        _wrapper_session._COW_ENABLED = prev_w


class _RunningOffsetBags(nn.Module):
    """Pool variable-size bags with a running tensor offset, as attention MIL models do."""

    def forward(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        pooled = []
        start = 0
        for n in counts:
            pooled.append(x[start : start + n].sum(0))
            start += n  # an int first, then a tensor updated in place
        return torch.stack(pooled)


class _RunningOffsetLevels(nn.Module):
    """Slice levels off a flattened sequence with a tuple index, as multi-scale DETRs do."""

    def forward(self, memory: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        cursor = 0
        levels = []
        for size in sizes:
            levels.append(memory[:, cursor : cursor + size].mean(1))
            cursor += size
        return torch.cat(levels, dim=-1)


class _AdaptiveNormRepeat(nn.Module):
    """Adaptive instance norm through a train-mode batch_norm on repeated buffers."""

    def __init__(self, functional: str) -> None:
        super().__init__()
        self.functional = functional
        self.weight = nn.Parameter(torch.randn(8))
        self.bias = nn.Parameter(torch.randn(8))
        self.register_buffer("running_mean", torch.zeros(4))
        self.register_buffer("running_var", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        running_mean = self.running_mean.repeat(n)
        running_var = self.running_var.repeat(n)
        flat = x.contiguous().view(1, n * c, h * w)
        if self.functional == "instance_norm":
            normed = F.instance_norm(
                flat, running_mean, running_var, self.weight, self.bias, True, 0.1, 1e-5
            )
        else:
            normed = F.batch_norm(
                flat, running_mean, running_var, self.weight, self.bias, True, 0.1, 1e-5
            )
        return normed.view(n, c, h, w)


def test_copy_arg_tree_snapshots_tensor_slice_bounds() -> None:
    """A saved slice keeps its tensor bounds' call-time values after an in-place update."""

    start = torch.tensor(4)
    index = (slice(None), slice(start, start + 3))
    saved = copy_arg_tree(index)
    start += 10
    assert saved[1].start.item() == 4
    assert saved[1].stop.item() == 7
    plain = slice(1, 5)
    assert copy_arg_tree(plain) is plain


def test_running_offset_slice_validates() -> None:
    """Replay slices the range the call read, not the range after ``start += n``."""

    torch.manual_seed(0)
    model = _RunningOffsetBags().eval()
    inputs = [torch.randn(20, 4), torch.tensor([8, 12])]
    assert tl.validate(model, inputs, seed=0)


def test_running_offset_tuple_index_validates() -> None:
    """The tuple-index path (validation's own capture copies it item by item) is covered."""

    torch.manual_seed(0)
    model = _RunningOffsetLevels().eval()
    inputs = [torch.randn(2, 10, 3), torch.tensor([4, 6])]
    assert tl.validate(model, inputs, seed=0)


@pytest.mark.parametrize("functional", ["batch_norm", "instance_norm"])
def test_train_mode_norm_keeps_producer_payload(functional: str) -> None:
    """A deferred payload passed as running stats is copied out before the stats update."""

    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 3)
    with _payload_clone_mode(defer=True):
        deferred = tl.trace(_AdaptiveNormRepeat(functional).eval(), x)
    with _payload_clone_mode(defer=False):
        eager = tl.trace(_AdaptiveNormRepeat(functional).eval(), x)
    for label in ("repeat_1_1:1", "repeat_2_2:1"):
        assert torch.equal(deferred[label].out, eager[label].out), label
    assert torch.equal(deferred["repeat_1_1:1"].out, torch.zeros(8))


@pytest.mark.parametrize("functional", ["batch_norm", "instance_norm"])
def test_train_mode_norm_on_repeated_buffers_validates(functional: str) -> None:
    """Replaying the repeat reproduces its saved out under deferred payload clones."""

    torch.manual_seed(0)
    with _payload_clone_mode(defer=True):
        assert tl.validate(_AdaptiveNormRepeat(functional).eval(), torch.randn(2, 4, 3, 3), seed=0)


def _foreign_fill(buffer: torch.Tensor, values: torch.Tensor, route: str) -> None:
    """Write ``values`` into ``buffer`` outside torch dispatch, through one export route."""

    src = values.detach().contiguous()
    if route == "data_ptr":
        ctypes.memmove(buffer.data_ptr(), src.data_ptr(), src.numel() * src.element_size())
    elif route == "untyped_storage":
        target = buffer.untyped_storage().data_ptr()
        ctypes.memmove(target, src.data_ptr(), src.numel() * src.element_size())
    elif route == "numpy":
        buffer.numpy()[...] = src.numpy()
    else:
        np.from_dlpack(buffer)[...] = src.numpy()


class _ForeignKernelFill(nn.Module):
    """Allocate an output buffer, detach it and fill it outside torch, as warp ops do."""

    def __init__(self, route: str) -> None:
        super().__init__()
        self.route = route

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        buffer = torch.full((x.shape[0], 4), -1.0)
        handle = buffer.detach()
        _foreign_fill(handle, x[:, None] * torch.arange(4.0), self.route)
        return buffer * 2


_ROUTES = ["data_ptr", "untyped_storage", "numpy", "dlpack"]


@pytest.mark.parametrize("route", _ROUTES)
def test_foreign_fill_keeps_producer_payload(route: str) -> None:
    """The ``full``/``detach`` saved outs keep the bytes they had before the foreign fill."""

    torch.manual_seed(0)
    x = torch.randn(5)
    with _payload_clone_mode(defer=True):
        deferred = tl.trace(_ForeignKernelFill(route), x)
    with _payload_clone_mode(defer=False):
        eager = tl.trace(_ForeignKernelFill(route), x)
    for label in ("full_1_1:1", "detach_1_2:1"):
        assert torch.equal(deferred[label].out, eager[label].out), label
        assert torch.equal(deferred[label].out, torch.full((5, 4), -1.0)), label


@pytest.mark.parametrize("route", _ROUTES)
def test_foreign_fill_validates(route: str) -> None:
    """Replaying the ``detach`` reproduces its saved out under deferred payload clones."""

    torch.manual_seed(0)
    with _payload_clone_mode(defer=True):
        assert tl.validate(_ForeignKernelFill(route), torch.randn(5), seed=0)


class _TwoForeignFills(nn.Module):
    """Subtract two buffers that were each filled outside torch."""

    def __init__(self, route: str) -> None:
        super().__init__()
        self.route = route

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = torch.full((x.shape[0], 4), -1.0)
        second = torch.full((x.shape[0], 4), -1.0)
        _foreign_fill(first.detach(), x[:, None] * torch.arange(4.0), self.route)
        _foreign_fill(second.detach(), x[:, None] + torch.arange(4.0), self.route)
        return torch.sub(first, second)


class _ForeignFillAndProduct(nn.Module):
    """Subtract an ordinary product from a buffer filled outside torch."""

    def __init__(self, route: str) -> None:
        super().__init__()
        self.route = route

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        buffer = torch.full((x.shape[0], 4), -1.0)
        _foreign_fill(buffer.detach(), x[:, None] * torch.arange(4.0), self.route)
        return torch.sub(buffer, x[:, None] * torch.ones(4))


def _swap_sub_operands(trace: tl.Trace) -> None:
    """Swap the parents of ``sub``'s two slots, recording snapshots as capture would."""

    (child,) = [op for op in trace._layer_list if op.func_name == "sub"]
    positions = child.call.parent_arg_positions["args"]
    positions[0], positions[1] = positions[1], positions[0]
    child.parents = type(child.parents)(dict.fromkeys(positions.values()))
    for key, label in positions.items():
        parent = _op_for_validation_label(trace, label)
        versions = parent.out_versions_by_child
        versions.pop(child.label, None)
        versions.pop(child.layer_label, None)
        value = child.call.saved_args[key]
        if not (value.shape == parent.out.shape and tensor_nanequal(value, parent.out)):
            versions[child.layer_label] = value.detach().clone()


def _validate_foreign(model: nn.Module, defer: bool, planted: bool) -> bool:
    """Capture under a payload clone mode, optionally swap ``sub``'s operands, validate."""

    torch.manual_seed(0)
    x = torch.randn(5)
    with _payload_clone_mode(defer=defer):
        trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=True, save_rng=True))
    if planted:
        _swap_sub_operands(trace)
    with _payload_clone_mode(defer=defer):
        return bool(trace.validate_forward_pass(model(x.clone()), validate_metadata=False))


_FOREIGN_MODELS = {
    "two_foreign_fills": _TwoForeignFills,
    "foreign_and_product": _ForeignFillAndProduct,
}


@pytest.mark.parametrize("defer", [True, False], ids=["deferred", "eager"])
@pytest.mark.parametrize("model_name", sorted(_FOREIGN_MODELS))
@pytest.mark.parametrize("route", _ROUTES)
def test_foreign_fill_consumer_validates(route: str, model_name: str, defer: bool) -> None:
    """A two-parent consumer of foreign-filled buffers validates as captured."""

    assert _validate_foreign(_FOREIGN_MODELS[model_name](route), defer, planted=False)


@pytest.mark.parametrize("defer", [True, False], ids=["deferred", "eager"])
@pytest.mark.parametrize("model_name", sorted(_FOREIGN_MODELS))
@pytest.mark.parametrize("route", _ROUTES)
def test_foreign_fill_swapped_operands_fail(route: str, model_name: str, defer: bool) -> None:
    """Swapping operands that involve a foreign-filled buffer fails forward validation.

    With two foreign-filled buffers both swapped snapshots carry a foreign-write mark,
    and replaying them reproduces the saved out, so only the mark's exact-slot rule
    catches the swap.
    """

    assert not _validate_foreign(_FOREIGN_MODELS[model_name](route), defer, planted=True)


#: Set by the planted model just before its ``sub``; the patched slot snapshot swaps that
#: call's two positional parents while capture is running, then clears it.
_PLANT_SWAP = [False]


class _PlantedSwapForeignFills(_TwoForeignFills):
    """``_TwoForeignFills`` that asks the patched slot snapshot to swap ``sub``'s parents."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = torch.full((x.shape[0], 4), -1.0)
        second = torch.full((x.shape[0], 4), -1.0)
        _foreign_fill(first.detach(), x[:, None] * torch.arange(4.0), self.route)
        _foreign_fill(second.detach(), x[:, None] + torch.arange(4.0), self.route)
        _PLANT_SWAP[0] = True
        return torch.sub(first, second)


def _swapping_slot_snapshot(real: Any) -> Any:
    """Wrap the argument-slot snapshot so the planted call's two slots trade their labels.

    Each slot takes the other slot's label and provenance labels together, so the parent
    recorder and the capture witness both read one consistent (wrong) binding.
    """

    def snapshot(trace: Any, call: Any, arg_tensors: Any) -> Any:
        slots = real(trace, call, arg_tensors)
        index = {(slot.kind, slot.key): i for i, slot in enumerate(slots)}
        pair = (index.get(("args", 0)), index.get(("args", 1)))
        if not _PLANT_SWAP[0] or None in pair or slots[pair[0]].label == slots[pair[1]].label:
            return slots
        _PLANT_SWAP[0] = False
        first, second = slots[pair[0]], slots[pair[1]]
        swapped = list(slots)
        swapped[pair[0]] = dataclasses.replace(
            first, label=second.label, session_parent_labels=second.session_parent_labels
        )
        swapped[pair[1]] = dataclasses.replace(
            second, label=first.label, session_parent_labels=first.session_parent_labels
        )
        workspace = _argument_fields._peek_raw_graph_workspace(trace)
        context = workspace._pending_arg_slots
        by_key = {(slot.kind, slot.key): slot for slot in swapped}
        labels = {slot.value_id: slot.label for slot in swapped}
        workspace._pending_arg_slots = type(context)(
            call.args, call.kwargs, tuple(swapped), by_key, labels
        )
        return tuple(swapped)

    return snapshot


@pytest.mark.parametrize("defer", [True, False], ids=["deferred", "eager"])
@pytest.mark.parametrize("route", _ROUTES)
def test_capture_time_swap_of_exported_storages_fails(
    route: str, defer: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong binding made AT capture onto exported storages that are not the parent's fails.

    Both slots hold tensors on exported, foreign-written storages, and capture binds each
    slot to the other buffer's producer, so the marks would be recorded from the wrong
    binding itself. A mark needs the slot to hold the parent's own tensor on that
    exported storage, so neither swapped snapshot is marked and validation fails.
    """

    model = _PlantedSwapForeignFills(route)
    torch.manual_seed(0)
    x = torch.randn(5)
    real = _argument_fields.snapshot_arg_slots
    monkeypatch.setattr(_argument_fields, "snapshot_arg_slots", _swapping_slot_snapshot(real))
    try:
        with _payload_clone_mode(defer=defer):
            trace = tl.trace(
                model, x, capture=tl.options.CaptureOptions(save_args=True, save_rng=True)
            )
    finally:
        monkeypatch.undo()
        _PLANT_SWAP[0] = False
    (child,) = [op for op in trace._layer_list if op.func_name == "sub"]
    order = [op.label for op in trace._layer_list]
    bound = [
        _op_for_validation_label(trace, child.call.parent_arg_positions["args"][k]) for k in (0, 1)
    ]
    assert order.index(bound[0].label) > order.index(bound[1].label), "the plant did not swap"
    assert all(
        child.layer_label in p.out_versions_by_child or child.label in p.out_versions_by_child
        for p in bound
    )
    with _payload_clone_mode(defer=defer):
        ground_truth = model(x.clone())
        _PLANT_SWAP[0] = False
        assert not trace.validate_forward_pass(ground_truth, validate_metadata=False)
