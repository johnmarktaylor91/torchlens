"""``Tensor.set_`` on a buffer records the same provenance from every entry state.

``set_`` rebinds its receiver: whether the buffer's storage key changes depends
only on what the buffer held before the call (an emptied buffer emptied again,
a buffer already viewing the storage it is set to). The captured graph must not
depend on that, or validation's re-trace of a copy taken after one forward
records a different graph from the first trace. Every overload is covered:
``set_()``, ``set_(Tensor)``, ``set_(storage)`` and
``set_(storage, offset, size, stride)``.

The storage argument of the storage overloads is cloned for the saved
arguments, so its bytes are charged to the save budget like a tensor
argument's, at admission and at commit.
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
from torchlens._save_budget import SaveBudget
from torchlens._utils.hashing import compute_graph_shape_hash
from torchlens.backends.torch import _ops_retention
from torchlens.options import CaptureOptions


class _SetEmpty(nn.Module):
    """``set_()`` empties the buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.set_()
        return x * 2 + self.b.sum()


class _SetSource(nn.Module):
    """``set_(Tensor)`` rebinds the buffer to a fresh source."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))
        self.register_buffer("src", torch.arange(8, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.set_(x * 3)
        return x + self.b


class _SetWholeStorage(nn.Module):
    """``set_(storage)`` views a whole storage from offset 0."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))
        self.register_buffer("src", torch.arange(8, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.set_(self.src.untyped_storage())
        return x + self.b[2:6]


class _SetStorageView(nn.Module):
    """``set_(storage, offset, size, stride)`` views part of another buffer's storage."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))
        self.register_buffer("src", torch.arange(8, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.b.set_(self.src.untyped_storage(), 2, (4,), (1,))
        return x + self.b


def _fresh(model: nn.Module) -> None:
    """Leave the model as constructed: the buffer on its own storage."""


def _after_one_forward(model: nn.Module) -> None:
    """The state validation's re-trace starts from."""

    with torch.no_grad():
        model(torch.rand(4))


def _emptied(model: nn.Module) -> None:
    """The buffer already has an empty storage."""

    with torch.no_grad():
        model.b.set_()


def _shares_source_storage(model: nn.Module) -> None:
    """The buffer already views the source's storage, at a different offset."""

    with torch.no_grad():
        model.b.set_(model.src.untyped_storage(), 0, (4,), (1,))


_ENTRY_STATES: dict[str, Callable[[nn.Module], None]] = {
    "fresh": _fresh,
    "after_one_forward": _after_one_forward,
    "emptied": _emptied,
    "aliased": _shares_source_storage,
}
_SET_MODELS = [_SetEmpty, _SetSource, _SetWholeStorage, _SetStorageView]


def _model_in_state(model_cls: type[nn.Module], state: str) -> nn.Module:
    torch.manual_seed(0)
    model = model_cls()
    if state == "aliased" and not hasattr(model, "src"):
        pytest.skip("the model has no second buffer to alias")
    _ENTRY_STATES[state](model)
    return model


def _graph(model: nn.Module) -> tuple[str, list[tuple[str, tuple[str, ...], Any]]]:
    """Return the graph-shape hash and each layer's parents and buffer write kind."""

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            model,
            torch.rand(4),
            capture=CaptureOptions(save_args=True, seed=0),
            save="all",
        )
    try:
        rows = [
            (
                layer.layer_label,
                tuple(layer.parents),
                getattr(layer, "buffer_write_kind", None),
            )
            for layer in trace._layer_list
        ]
        return compute_graph_shape_hash(trace, include_module_address=False), rows
    finally:
        trace.cleanup()


@pytest.mark.parametrize("state", [name for name in _ENTRY_STATES if name != "fresh"])
@pytest.mark.parametrize("model_cls", _SET_MODELS)
def test_set_graph_does_not_depend_on_the_entry_storage_state(
    model_cls: type[nn.Module], state: str
) -> None:
    """The captured graph from any entry state equals the graph from a fresh model."""

    fresh_hash, fresh_rows = _graph(_model_in_state(model_cls, "fresh"))
    state_hash, state_rows = _graph(_model_in_state(model_cls, state))
    assert state_rows == fresh_rows
    assert state_hash == fresh_hash


@pytest.mark.parametrize("state", list(_ENTRY_STATES))
@pytest.mark.parametrize("model_cls", _SET_MODELS)
def test_set_is_a_storage_rebind_never_an_in_place_write(
    model_cls: type[nn.Module], state: str
) -> None:
    """``set_`` records one ``data_reassign`` buffer node and no in-place version node.

    The later read of the buffer links to the ``set_`` output itself.
    """

    _, rows = _graph(_model_in_state(model_cls, state))
    kinds = [kind for _, _, kind in rows if kind is not None]
    assert kinds == ["data_reassign"]
    (set_label,) = [label for label, _, _ in rows if label.startswith("set_")]
    readers = [label for label, parents, _ in rows if set_label in parents]
    assert readers, "nothing reads the rebound buffer from the set_ output"


@pytest.mark.parametrize("state", list(_ENTRY_STATES))
@pytest.mark.parametrize("model_cls", _SET_MODELS)
def test_set_validates_on_the_copy_path_from_every_entry_state(
    model_cls: type[nn.Module], state: str
) -> None:
    """Validation (first trace from the entry state, re-trace from after it) returns True."""

    assert _quiet_validate(_model_in_state(model_cls, state), torch.rand(4)) is True


# ---- save budget: the storage argument's clone is charged ----

_STORAGE_BYTES = 8 * 4  # torch.arange(8, dtype=float32)


class _BudgetTrace:
    """The one attribute the saved-args budget sites read from a trace."""

    def __init__(self, budget_bytes: int | None) -> None:
        self._save_budget_accountant = SaveBudget.from_option(budget_bytes)


def test_storage_argument_dedups_with_a_tensor_on_the_same_storage() -> None:
    """A tensor and a storage argument sharing one storage are charged once."""

    src = torch.arange(8, dtype=torch.float32)
    reservations = _ops_retention._admit_saved_args_budget(
        _BudgetTrace(_STORAGE_BYTES),
        {"_label_raw": "set_"},
        (src, src.untyped_storage()),
        {},
    )
    assert reservations is not None and len(reservations) == 1
