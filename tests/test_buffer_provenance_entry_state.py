"""Buffer provenance is recorded the same way whatever the entry state (AGT-2415).

A forward that rewrites a buffer builds its graph from recorded facts: the
producer label of the written tensor and the recorded buffer version (tensor
object, storage, version counter). Neither the ``identity`` claim for a buffer
version nor the merge of two buffer nodes may rest on the values the forward
happens to reach, so a capture from different buffer values (including NaN)
has the same graph structure and validation's pristine re-trace matches.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TraceNotReproducibleWarning
from torchlens.validation import validate_forward_pass


class _RewriteInPlace(nn.Module):
    """Reads a buffer, rewrites it in place, reads it again."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.b
        self.b.mul_(1.5)
        return x + self.b


class _Reassign(nn.Module):
    """Reassigns a buffer to an input-dependent tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.b
        self.b = self.b * 1.5 + x.mean()
        return x + self.b


class _RewriteTwice(nn.Module):
    """Reassigns a buffer twice before reading it."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.b
        self.b = x * 2
        self.b = self.b + 1
        return x + self.b


class _SometimesRewrites(nn.Module):
    """Rewrites its buffer only when asked."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor, rewrite: bool) -> torch.Tensor:
        x = x + self.b
        if rewrite:
            self.b = x.detach() * 0.5
        return x


class _RewriteOnSomePasses(nn.Module):
    """Calls one buffer-holding module four times; only two calls rewrite."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _SometimesRewrites()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inner(x, True)
        x = self.inner(x, False)
        x = self.inner(x, True)
        x = self.inner(x, False)
        return x * 2


class _DataSwapToZeros(nn.Module):
    """Swaps a buffer's storage to zeros: equal to the old value from a zeros state."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x + self.b
        self.b.data = torch.zeros(4)
        return a + self.b


class _SetToZeros(nn.Module):
    """``set_``s a buffer to zeros: equal to the old value from a zeros state."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.rand(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x + self.b
        with torch.no_grad():
            self.b.set_(torch.zeros(4))
        return a + self.b


_REWRITERS: dict[str, Callable[[], nn.Module]] = {
    "in_place": _RewriteInPlace,
    "reassigned": _Reassign,
    "rewritten_twice": _RewriteTwice,
    "some_passes": _RewriteOnSomePasses,
    "data_swap": _DataSwapToZeros,
    "set_": _SetToZeros,
}


def _fill_buffers(model: nn.Module, value: float) -> nn.Module:
    """Return ``model`` with every buffer filled with ``value`` in place."""

    with torch.no_grad():
        for buffer in model.buffers():
            buffer.fill_(value)
    return model


_SAVE_ALL = tl.options.CaptureOptions(layers_to_save="all")


def _graph_hash(model: nn.Module, x: torch.Tensor) -> str:
    """Capture ``model`` once, saving every activation, and return its graph shape hash."""

    trace = tl.trace(model, x, capture=_SAVE_ALL)
    try:
        assert trace.graph_shape_hash is not None
        return trace.graph_shape_hash
    finally:
        trace.cleanup()


@pytest.mark.parametrize("name", sorted(_REWRITERS))
def test_rewritten_buffer_graph_is_the_same_from_any_entry_state(name: str) -> None:
    """Fresh, zero-filled and NaN-filled buffers all record one graph structure."""

    factory = _REWRITERS[name]
    x = torch.rand(4)
    torch.manual_seed(0)
    fresh = _graph_hash(factory(), x)
    torch.manual_seed(0)
    zeros = _graph_hash(_fill_buffers(factory(), 0.0), x)
    torch.manual_seed(0)
    nans = _graph_hash(_fill_buffers(factory(), float("nan")), x)
    assert fresh == zeros == nans


@pytest.mark.parametrize("name", sorted(_REWRITERS))
def test_rewritten_buffer_validates_with_no_retrace_warning(name: str) -> None:
    """The pristine re-trace (from the buffers the first forward wrote) matches."""

    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TraceNotReproducibleWarning)
        assert validate_forward_pass(_REWRITERS[name](), torch.rand(4)) is True


def test_nan_reached_by_a_rewritten_buffer_keeps_its_identity_edge() -> None:
    """A buffer version that holds NaN still replays as ``identity`` of its producer."""

    trace = tl.trace(_fill_buffers(_Reassign(), float("nan")), torch.rand(4), capture=_SAVE_ALL)
    try:
        versions = [op for op in trace.layer_list if op.is_buffer and op.buffer_source]
        assert versions, "the reassignment must log a buffer version node"
        for op in versions:
            assert torch.isnan(op.out).all()
            assert op.func_name == "identity"
            assert op.parents == [op.buffer_source]
    finally:
        trace.cleanup()


def test_equal_valued_storage_swap_keeps_its_own_version_node() -> None:
    """A recorded write never merges into the read before it, even when values agree."""

    trace = tl.trace(_fill_buffers(_DataSwapToZeros(), 0.0), torch.rand(4), capture=_SAVE_ALL)
    try:
        buffer_ops = [op for op in trace.layer_list if op.is_buffer]
        kinds = [op.buffer_write_kind for op in buffer_ops]
        assert kinds == [None, "data_reassign"]
    finally:
        trace.cleanup()
