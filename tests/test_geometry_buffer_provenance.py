"""In-place geometry writes on a buffer record the same provenance from every entry state.

``resize_`` and ``resize_as_`` reallocate the receiver's storage when they grow
past its capacity, change only its shape when they shrink, and change nothing
when the buffer already has the target size. Which of the three happens depends
only on what the buffer held before the forward, so the captured graph must not
depend on it: validation re-traces a copy taken after one forward, and a graph
that differs from the first trace fails it. Every resize of the buffer itself is
journaled as one in-place write produced by the resize op.

The metadata-only in-place views (``unsqueeze_``, ``squeeze_``, ``transpose_``,
``t_``, ``swapdims_``, ``swapaxes_``, ``as_strided_``) already record one graph
from every entry state; they are pinned here so that stays true.

Validation replays ``as_strided_`` and a resize that grows inside a larger
shared storage on the receiver's saved storage context (offset, strides and the
whole storage), so those calls validate from the aliased state too.
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
from torchlens._utils.hashing import compute_graph_shape_hash
from torchlens.options import CaptureOptions

_Write = Callable[[torch.Tensor], Any]

_RESIZE_WRITES: dict[str, tuple[tuple[int, ...], _Write]] = {
    "resize_grow": ((4,), lambda b: b.resize_(8)),
    "resize_shrink": ((4,), lambda b: b.resize_(2)),
    "resize_same": ((4,), lambda b: b.resize_(4)),
    "resize_2d_grow": ((2, 2), lambda b: b.resize_(3, 3)),
    "resize_as_grow": ((4,), lambda b: b.resize_as_(torch.zeros(8))),
    "resize_as_shrink": ((4,), lambda b: b.resize_as_(torch.zeros(2))),
    "resize_grow_then_zero": ((4,), lambda b: b.resize_(8).zero_()),
    "resize_shrink_then_zero": ((4,), lambda b: b.resize_(2).zero_()),
}
_VIEW_WRITES: dict[str, tuple[tuple[int, ...], _Write]] = {
    "unsqueeze_": ((4,), lambda b: b.unsqueeze_(0)),
    "squeeze_all": ((1, 4), lambda b: b.squeeze_()),
    "squeeze_dim": ((1, 4), lambda b: b.squeeze_(0)),
    "squeeze_noop": ((4,), lambda b: b.squeeze_()),
    "transpose_": ((2, 3), lambda b: b.transpose_(0, 1)),
    "t_": ((2, 3), lambda b: b.t_()),
    "swapdims_": ((2, 3), lambda b: b.swapdims_(0, 1)),
    "swapaxes_": ((2, 3), lambda b: b.swapaxes_(0, 1)),
    "as_strided_offset": ((4,), lambda b: b.as_strided_((2,), (1,), 2)),
}
_WRITES = {**_RESIZE_WRITES, **_VIEW_WRITES}

# Calls whose post-call bytes are fully defined from every entry state: no grow
# exposes uninitialized tail bytes to the read (a grow from the emptied state does).
_DEFINED_FROM_EVERY_STATE = [
    "resize_grow_then_zero",
    "resize_shrink_then_zero",
    "unsqueeze_",
    "squeeze_all",
    "squeeze_dim",
    "squeeze_noop",
    "transpose_",
    "t_",
    "swapdims_",
    "swapaxes_",
    "as_strided_offset",
]
_DEFINED_FROM_NONEMPTY_STATES = ["resize_shrink", "resize_same", "resize_as_shrink"]


def _model_class(name: str) -> type[nn.Module]:
    shape, write = _WRITES[name]

    class _GeometryWrite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", torch.rand(*shape))
            self.register_buffer("src", torch.arange(16, dtype=torch.float32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                write(self.b)
            return x + self.b.sum()

    _GeometryWrite.__name__ = f"_GeometryWrite_{name}"
    return _GeometryWrite


def _fresh(model: nn.Module) -> None:
    """Leave the model as constructed: the buffer on its own storage."""


def _after_one_forward(model: nn.Module) -> None:
    """The state validation's re-trace starts from."""

    with torch.no_grad():
        model(torch.rand(4))


def _emptied(model: nn.Module) -> None:
    """The buffer keeps its number of dimensions on an empty storage."""

    with torch.no_grad():
        model.b.set_(torch.empty([0] * model.b.dim()))


def _shares_source_storage(model: nn.Module) -> None:
    """The buffer views the source buffer's storage at offset 1."""

    with torch.no_grad():
        model.b.set_(model.src.untyped_storage(), 1, model.b.shape, model.b.stride())


_ENTRY_STATES: dict[str, Callable[[nn.Module], None]] = {
    "fresh": _fresh,
    "after_one_forward": _after_one_forward,
    "emptied": _emptied,
    "aliased": _shares_source_storage,
}


def _model_in_state(name: str, state: str) -> nn.Module:
    if state == "emptied" and name.startswith("as_strided_"):
        pytest.skip("as_strided_ past an empty storage raises in eager torch")
    torch.manual_seed(0)
    model = _model_class(name)()
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
            capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=0),
        )
    try:
        rows = [
            (
                layer.layer_label,
                tuple(layer.parents),
                getattr(layer, "buffer_write_kind", None),
            )
            for layer in trace.layer_list
        ]
        return compute_graph_shape_hash(trace, include_module_address=False), rows
    finally:
        trace.cleanup()


@pytest.mark.parametrize("state", [name for name in _ENTRY_STATES if name != "fresh"])
@pytest.mark.parametrize("name", list(_WRITES))
def test_geometry_write_graph_does_not_depend_on_the_entry_state(name: str, state: str) -> None:
    """The captured graph from any entry state equals the graph from a fresh model."""

    fresh_hash, fresh_rows = _graph(_model_in_state(name, "fresh"))
    state_hash, state_rows = _graph(_model_in_state(name, state))
    assert state_rows == fresh_rows
    assert state_hash == fresh_hash


@pytest.mark.parametrize("state", list(_ENTRY_STATES))
@pytest.mark.parametrize("name", [name for name in _RESIZE_WRITES if "then_zero" not in name])
def test_resize_is_one_in_place_write_produced_by_the_resize(name: str, state: str) -> None:
    """A buffer resize records one ``inplace`` version whose parent is the resize op.

    No ``data_reassign`` node appears, even when the resize reallocated the
    storage, and the later read links to the written version.
    """

    _, rows = _graph(_model_in_state(name, state))
    written = [(label, parents) for label, parents, kind in rows if kind is not None]
    assert [kind for _, _, kind in rows if kind is not None] == ["inplace"]
    ((version_label, version_parents),) = written
    (resize_label,) = [label for label, _, _ in rows if label.startswith("resize")]
    assert version_parents == (resize_label,)
    assert any(version_label in parents for _, parents, _ in rows)


@pytest.mark.parametrize("state", list(_ENTRY_STATES))
@pytest.mark.parametrize("name", _DEFINED_FROM_EVERY_STATE)
def test_geometry_write_validates_from_every_entry_state(name: str, state: str) -> None:
    """Validation (first trace from the entry state, re-trace from after it) returns True."""

    assert _quiet_validate(_model_in_state(name, state), torch.rand(4)) is True


@pytest.mark.parametrize("state", ["fresh", "after_one_forward", "aliased"])
@pytest.mark.parametrize("name", _DEFINED_FROM_NONEMPTY_STATES)
def test_prefix_preserving_resize_validates(name: str, state: str) -> None:
    """A resize that never grows past the buffer's elements validates from a non-empty buffer."""

    assert _quiet_validate(_model_in_state(name, state), torch.rand(4)) is True
