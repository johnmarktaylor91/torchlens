"""Storage-geometry receivers are saved and replayed with their whole storage.

``as_strided_`` reads any bytes of the receiver's storage, and a ``resize_`` that
grows inside a larger shared storage exposes the neighbour's bytes past the
receiver's view. A compact clone of the receiver cannot replay either call, so
capture saves the receiver of ``as_strided_`` / ``resize_`` / ``resize_as_`` as a
view of a private clone of its full storage, the save budget charges that
storage, and validation replays the call on a clone of the saved context with
the parent values under test written into the view.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from _validation_capture import _quiet_validate
from torch import nn

import torchlens as tl
from torchlens._save_budget import SaveBudget
from torchlens._storage_bindings import copy_with_storage_context, views_part_of_its_storage
from torchlens.backends.torch import _ops_retention
from torchlens.options import CaptureOptions

_SRC_BYTES = 16 * 4  # torch.arange(16, dtype=float32)


def _view_into_source() -> tuple[torch.Tensor, torch.Tensor]:
    """A 4-element view at offset 1 of a 16-element source, and the source."""

    src = torch.arange(16, dtype=torch.float32)
    view = torch.empty(0).set_(src.untyped_storage(), 1, (4,), (1,))
    return view, src


class _BudgetTrace:
    """The one attribute the saved-args budget sites read from a trace."""

    def __init__(self, budget_bytes: int | None) -> None:
        self._save_budget_accountant = SaveBudget.from_option(budget_bytes)


class _GeometryOnAliasedBuffer(nn.Module):
    """A buffer viewing a larger source buffer at offset 1, written by ``write``."""

    def __init__(self, write: Any) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))
        self.register_buffer("src", torch.arange(16, dtype=torch.float32))
        self.write = write
        with torch.no_grad():
            self.b.set_(self.src.untyped_storage(), 1, self.b.shape, self.b.stride())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.write(self.b)
        return x + self.b.sum()


def test_views_part_of_its_storage_distinguishes_a_compact_tensor() -> None:
    view, src = _view_into_source()
    assert views_part_of_its_storage(view)
    assert views_part_of_its_storage(src[:8])
    assert not views_part_of_its_storage(src)
    assert not views_part_of_its_storage(src.clone())
    assert not views_part_of_its_storage(torch.empty(0))


def test_copy_with_storage_context_keeps_offset_strides_and_the_whole_storage() -> None:
    view, src = _view_into_source()
    copy = copy_with_storage_context(view)
    assert torch.equal(copy, view)
    assert copy.storage_offset() == 1 and copy.stride() == view.stride()
    assert copy.untyped_storage().nbytes() == _SRC_BYTES
    assert copy.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
    copy.zero_()
    assert torch.equal(src, torch.arange(16, dtype=torch.float32))


def test_copy_with_storage_context_replays_a_grow_and_a_restride() -> None:
    """The copy reproduces what the real receiver reads past its own view."""

    view, src = _view_into_source()
    grown = copy_with_storage_context(view).resize_(8)
    assert torch.equal(grown, src[1:9])
    restrided = copy_with_storage_context(view).as_strided_((2,), (1,), 2)
    assert torch.equal(restrided, src[2:4])
    assert torch.equal(copy_with_storage_context(src.clone()), src)


@pytest.mark.parametrize("func_name", ["as_strided_", "resize_", "resize_as_"])
def test_saved_receiver_of_a_storage_geometry_call_keeps_its_storage(func_name: str) -> None:
    view, src = _view_into_source()
    (receiver, other), _ = _ops_retention._copy_saved_call_args(
        func_name, (view, torch.zeros(2)), {}, True
    )
    assert receiver.storage_offset() == 1
    assert receiver.untyped_storage().nbytes() == _SRC_BYTES
    assert receiver.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
    assert other.untyped_storage().nbytes() == 2 * 4


def test_saved_receiver_of_another_call_stays_compact() -> None:
    view, _ = _view_into_source()
    (receiver,), _ = _ops_retention._copy_saved_call_args("add_", (view,), {}, True)
    assert receiver.storage_offset() == 0
    assert receiver.untyped_storage().nbytes() == 4 * 4


def test_storage_context_copy_is_charged_to_the_save_budget() -> None:
    """Admission projects the whole storage and the commit charges exactly that."""

    view, _ = _view_into_source()
    trace = _BudgetTrace(1 << 20)
    fields: dict[str, Any] = {"_label_raw": "as_strided_"}
    reservations = _ops_retention._admit_saved_args_budget(trace, fields, (view,), {})
    saved_args, saved_kwargs = _ops_retention._copy_saved_call_args(
        "as_strided_", (view,), {}, True
    )
    fields["saved_args"], fields["saved_kwargs"] = list(saved_args), saved_kwargs
    _ops_retention._commit_saved_args_budget(trace, fields, reservations)
    budget = trace._save_budget_accountant
    assert budget is not None
    assert budget.ledgers["cpu"].committed_bytes == _SRC_BYTES


def test_capture_saves_the_aliased_receiver_with_its_source_storage() -> None:
    torch.manual_seed(0)
    model = _GeometryOnAliasedBuffer(lambda b: b.as_strided_((2,), (1,), 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            model,
            torch.rand(4),
            capture=CaptureOptions(layers_to_save="all", save_args=True, seed=0),
        )
    try:
        op = next(layer for layer in trace.layer_list if layer.func_name == "as_strided_")
        saved = op.saved_args[0]
        assert saved.storage_offset() == 1 and saved.untyped_storage().nbytes() == _SRC_BYTES
        assert saved.untyped_storage().data_ptr() != model.src.untyped_storage().data_ptr()
        assert torch.equal(op.out, torch.tensor([2.0, 3.0]))
    finally:
        trace.cleanup()


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda b: b.as_strided_((2,), (1,), 2), id="as_strided_inside_the_view"),
        pytest.param(lambda b: b.as_strided_((2,), (1,), 0), id="as_strided_before_the_view"),
        pytest.param(lambda b: b.resize_(8).zero_(), id="resize_grow_in_storage"),
        pytest.param(
            lambda b: b.resize_as_(torch.zeros(8)).zero_(), id="resize_as_grow_in_storage"
        ),
    ],
)
def test_storage_geometry_on_an_aliased_buffer_validates(write: Any) -> None:
    """Replay reads the neighbour's bytes the real call read, from the aliased state."""

    torch.manual_seed(0)
    assert _quiet_validate(_GeometryOnAliasedBuffer(write), torch.rand(4)) is True


def test_wrong_storage_context_still_fails_replay() -> None:
    """A receiver saved on the wrong context (bytes shifted by one) is caught.

    The call reads storage element 0, before the receiver's view at offset 1,
    so the swapped-in receiver values cannot hide a wrong saved context.
    """

    torch.manual_seed(0)
    model = _GeometryOnAliasedBuffer(lambda b: b.as_strided_((2,), (1,), 0))
    x = torch.rand(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            model,
            x,
            capture=CaptureOptions(layers_to_save="all", save_args=True, seed=0),
        )
    try:
        from torchlens.validation.core import validate_saved_outs

        op = next(layer for layer in trace.layer_list if layer.func_name == "as_strided_")
        saved = op.saved_args[0]
        context = torch.arange(16, dtype=torch.float32) + 1
        context[1:5] = saved
        wrong = torch.empty(0).set_(context.untyped_storage(), 1, (4,), (1,))
        assert torch.equal(wrong, saved)
        op.saved_args[0] = wrong
        status = validate_saved_outs(trace, [model(x)], validate_metadata=False)
        assert status.state == "failed"
    finally:
        trace.cleanup()
