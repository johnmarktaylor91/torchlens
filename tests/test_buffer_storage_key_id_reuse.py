"""Buffer-address resolution must not trust a storage key cached for a dead tensor.

``BufferWriteTracker`` caches per-tensor storage metadata between operation
boundaries. A cache keyed by ``id(tensor)`` alone goes stale when a tensor it
saw is freed mid-capture and a new tensor object reuses the id: the newcomer
then inherits the dead tensor's storage identity, so an unrelated tensor can be
re-rooted as a buffer source (or a real buffer alias missed).
"""

from __future__ import annotations

import gc

import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch.buffer_writes import BufferWriteTracker

_ITERATIONS = 32


class _AliasChurn(nn.Module):
    """Alternate an unlabeled buffer alias with an unlabeled input alias.

    ``.data`` returns a fresh tensor object with no TorchLens label and a fresh
    version counter, so consecutive aliases are freed and reallocated inside one
    capture and CPython hands the next one the freed object's ``id``.
    """

    def __init__(self) -> None:
        """Register the buffer and the id log."""

        super().__init__()
        self.register_buffer("buf", torch.full((4,), 7.0))
        self.alias_ids: list[tuple[str, int]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix buffer aliases (via ``add``) and input aliases (via ``mul``).

        Parameters
        ----------
        x:
            Input tensor with the buffer's shape.

        Returns
        -------
        torch.Tensor
            Accumulated result.
        """

        out = x * 1.0
        for _ in range(_ITERATIONS):
            buffer_alias = self.buf.data
            self.alias_ids.append(("buffer", id(buffer_alias)))
            out = torch.add(out, buffer_alias)
            del buffer_alias
            input_alias = x.data
            self.alias_ids.append(("input", id(input_alias)))
            out = torch.mul(out, input_alias)
            del input_alias
        return out


def _reused_across_kinds(alias_ids: list[tuple[str, int]]) -> int:
    """Count consecutive aliases of different kinds that shared an ``id``."""

    return sum(
        1
        for (kind_a, id_a), (kind_b, id_b) in zip(alias_ids, alias_ids[1:])
        if kind_a != kind_b and id_a == id_b
    )


def test_id_reuse_never_reroots_an_input_alias_as_a_buffer() -> None:
    """An input alias that reuses a buffer alias's id gets no buffer parent, and vice versa."""

    model = _AliasChurn()
    trace = tl.trace(model, torch.ones(4))

    # The scenario must actually happen, or this test proves nothing.
    assert _reused_across_kinds(model.alias_ids) > 0, model.alias_ids[:8]

    adds = [op for op in trace.layer_list if op.func_name == "add"]
    muls = [op for op in trace.layer_list if op.func_name == "mul" and op.parents]
    assert len(adds) == _ITERATIONS
    for op in adds:
        assert any(trace[parent].is_buffer for parent in op.parents), (op.label, op.parents)
    for op in muls:
        assert not any(trace[parent].is_buffer for parent in op.parents), (op.label, op.parents)


def test_storage_metadata_cache_does_not_outlive_its_tensor() -> None:
    """A cached storage key/range is dropped when its tensor dies, not served to an id twin."""

    model = _AliasChurn()
    trace = tl.trace(model, torch.ones(4))
    tracker = BufferWriteTracker(trace, model)

    first = torch.zeros(4)
    first_key = tracker.storage_key(first)
    tracker.storage_range(first)
    first_id = id(first)
    del first
    gc.collect()

    # Allocate until a new tensor object lands on the freed id (bounded).
    keep: list[torch.Tensor] = []
    twin = None
    for _ in range(4096):
        candidate = torch.zeros(8)
        if id(candidate) == first_id:
            twin = candidate
            break
        keep.append(candidate)
    if twin is None:
        # No reuse on this allocator: the cache must still not hold the dead entry.
        assert all(key[0] != first_id for key in tracker._storage_key_cache)
        return
    assert tracker.storage_key(twin) != first_key
    assert tracker.storage_range(twin) == (0, 32)
