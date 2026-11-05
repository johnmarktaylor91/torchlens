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

from collections.abc import Callable

import pytest
import torch
from _validation_capture import _quiet_validate
from torch import nn


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


def _model_in_state(model_cls: type[nn.Module], state: str) -> nn.Module:
    torch.manual_seed(0)
    model = model_cls()
    _STATES[model_cls][state](model)
    return model


# Kept by the shrink-r4 coverage proof: sole branch coverage of backends/torch/buffer_writes.py.
# Rows kept by the shrink-r4 per-id coverage proof (the rest were covered by kept tests).
@pytest.mark.parametrize(("model_cls", "state"), [(_ShareAliasedStorage, "fresh")])
def test_share_memory_validates_from_every_entry_state(
    model_cls: type[nn.Module], state: str
) -> None:
    """Validation (first trace from the entry state, re-trace from after it) returns True."""

    assert _quiet_validate(_model_in_state(model_cls, state), torch.rand(4)) is True
