"""``Tensor.share_memory_`` on a buffer is captured as one storage rebind.

``buf.share_memory_()`` moves the buffer's storage to shared memory in place:
the tensor object stays, the storage's data pointer changes, and every tensor
on that storage moves with it. The call is a Python-level tensor method that
reads its storage through ``_typed_storage()``, so the wrapped storage accessor
must not take the call's bottom-level barcode, or the call goes uncaptured and
validation fails closed on the unaccounted allocation.

The move happens once (a storage already in shared memory stays where it is),
so, as for ``set_``, the receiver and every registered buffer sharing its storage
are recorded as one ``data_reassign`` rebind from every entry state, and the
later read links to the ``share_memory_`` output.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from _validation_capture import _quiet_validate
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.utils.hashing import compute_graph_shape_hash


class _ShareOwnStorage(nn.Module):
    """The buffer owns its storage."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.share_memory_()
        return x + self.b.sum()


class _ShareAliasedStorage(nn.Module):
    """The buffer views another buffer's storage, which moves with it."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))
        self.register_buffer("src", torch.arange(16, dtype=torch.float32))
        with torch.no_grad():
            self.b.set_(self.src.untyped_storage(), 1, self.b.shape, self.b.stride())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.share_memory_()
        return x + self.b.sum() + self.src[:2].sum()


def _fresh(model: nn.Module) -> None:
    """Leave the model as constructed: nothing in shared memory yet."""


def _after_one_forward(model: nn.Module) -> None:
    """The state validation's re-trace starts from: the storage already shared."""

    with torch.no_grad():
        model(torch.rand(4))


def _emptied(model: nn.Module) -> None:
    """The buffer keeps its number of dimensions on an empty storage of its own."""

    with torch.no_grad():
        model.b.set_(torch.empty([0] * model.b.dim()))


_STATES: dict[type[nn.Module], dict[str, Callable[[nn.Module], None]]] = {
    _ShareOwnStorage: {
        "fresh": _fresh,
        "after_one_forward": _after_one_forward,
        "emptied": _emptied,
    },
    _ShareAliasedStorage: {"fresh": _fresh, "after_one_forward": _after_one_forward},
}
_CASES = [(model_cls, state) for model_cls, states in _STATES.items() for state in states]


def _model_in_state(model_cls: type[nn.Module], state: str) -> nn.Module:
    torch.manual_seed(0)
    model = model_cls()
    _STATES[model_cls][state](model)
    return model


def _graph(model: nn.Module) -> tuple[str, list[tuple[str, str, tuple[str, ...], Any]]]:
    """Return the graph-shape hash and each layer's func name, parents and buffer write kind."""

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            model,
            torch.rand(4),
            capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=0),
        )
    try:
        rows = [
            (
                layer.layer_label,
                layer.func_name,
                tuple(layer.parents),
                getattr(layer, "buffer_write_kind", None),
            )
            for layer in trace.layer_list
        ]
        return compute_graph_shape_hash(trace, include_module_address=False), rows
    finally:
        trace.cleanup()


def test_share_memory_moves_every_tensor_on_the_storage_once() -> None:
    """The torch contract the capture relies on."""

    src = torch.arange(8, dtype=torch.float32)
    view = src[2:6]
    before = src.untyped_storage().data_ptr()
    view.share_memory_()
    moved = src.untyped_storage().data_ptr()
    assert moved != before and view.untyped_storage().data_ptr() == moved
    assert src.is_shared()
    view.share_memory_()
    assert src.untyped_storage().data_ptr() == moved


@pytest.mark.parametrize(("model_cls", "state"), [c for c in _CASES if c[1] != "fresh"])
def test_share_memory_graph_does_not_depend_on_the_entry_state(
    model_cls: type[nn.Module], state: str
) -> None:
    """The captured graph from any entry state equals the graph from a fresh model."""

    fresh_hash, fresh_rows = _graph(_model_in_state(model_cls, "fresh"))
    state_hash, state_rows = _graph(_model_in_state(model_cls, state))
    assert state_rows == fresh_rows
    assert state_hash == fresh_hash


@pytest.mark.parametrize(("model_cls", "state"), _CASES)
def test_share_memory_is_captured_as_a_storage_rebind(
    model_cls: type[nn.Module], state: str
) -> None:
    """One ``share_memory_`` op reads the buffer; the buffer records a ``data_reassign``.

    The aliased source buffer moves with the receiver's storage and records
    its own rebind; no in-place version node appears, and the later read of the
    receiver links to the ``share_memory_`` output.
    """

    _, rows = _graph(_model_in_state(model_cls, state))
    (share_label,) = [label for label, func, _, _ in rows if func == "share_memory_"]
    share_parents = next(parents for label, _, parents, _ in rows if label == share_label)
    assert len(share_parents) == 1 and share_parents[0].startswith("buffer")
    kinds = [kind for _, _, _, kind in rows if kind is not None]
    expected_rebinds = 2 if model_cls is _ShareAliasedStorage else 1
    assert kinds == ["data_reassign"] * expected_rebinds
    readers = [label for label, _, parents, _ in rows if share_label in parents]
    assert readers, "nothing reads the shared buffer from the share_memory_ output"


@pytest.mark.parametrize(("model_cls", "state"), _CASES)
def test_share_memory_validates_from_every_entry_state(
    model_cls: type[nn.Module], state: str
) -> None:
    """Validation (first trace from the entry state, re-trace from after it) returns True."""

    assert _quiet_validate(_model_in_state(model_cls, state), torch.rand(4)) is True


def test_share_memory_leaves_the_live_buffer_shared_and_unchanged() -> None:
    torch.manual_seed(0)
    model = _ShareOwnStorage()
    expected = model.b.clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, torch.rand(4), capture=CaptureOptions(layers_to_save="all"))
    trace.cleanup()
    assert model.b.is_shared()
    assert torch.equal(model.b, expected)
