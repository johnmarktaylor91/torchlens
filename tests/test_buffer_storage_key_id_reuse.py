"""Buffer-address resolution must not trust a storage key cached for a dead tensor.

``BufferWriteTracker`` caches per-tensor storage metadata between operation
boundaries. A cache keyed by ``id(tensor)`` alone goes stale when a tensor it
saw is freed mid-capture and a new tensor object reuses the id: the newcomer
then inherits the dead tensor's storage identity, so an unrelated tensor can be
re-rooted as a buffer source (or a real buffer alias missed).
"""

from __future__ import annotations

import gc
import warnings
from itertools import pairwise

import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch.buffer_writes import BufferWriteTracker

_ITERATIONS = 32


class _AliasChurn(nn.Module):
    """Alternate a labeled buffer view with an unlabeled newborn tensor.

    Every wrapped call resolves each tensor argument against the registered
    buffers, so the buffer view's storage key is cached. The view is then freed
    and ``Generator.get_state()`` -- a tensor born outside any wrapped torch
    function, hence unlabeled -- is allocated in its place, typically on the
    same ``id`` and with the same (zero) version counter.
    """

    def __init__(self) -> None:
        """Register the buffer, the generator, and the id log."""

        super().__init__()
        self.register_buffer("buf", torch.full((4,), 7.0))
        self.generator = torch.Generator()
        self.alias_ids: list[tuple[str, int]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume buffer views and generator-state tensors in turn.

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
            buffer_view = self.buf.view(4)
            self.alias_ids.append(("buffer", id(buffer_view)))
            out = out + buffer_view
            del buffer_view
            state = self.generator.get_state()
            self.alias_ids.append(("state", id(state)))
            out = out + state[:4].float()
            del state
        return out


def _reused_across_kinds(alias_ids: list[tuple[str, int]]) -> int:
    """Count consecutive tensors of different kinds that shared an ``id``."""

    return sum(
        1
        for (kind_a, id_a), (kind_b, id_b) in pairwise(alias_ids)
        if kind_a != kind_b and id_a == id_b
    )


def test_id_reuse_never_reroots_a_newborn_tensor_as_a_buffer() -> None:
    """A tensor that inherits a dead buffer view's id is not logged as the buffer."""

    model = _AliasChurn()
    with warnings.catch_warnings():
        # The generator state is a genuinely provenance-free argument; its disclosure
        # warning is expected and not what this test is about.
        warnings.filterwarnings("ignore", message=".*no graph/source provenance.*")
        trace = tl.trace(model, torch.ones(4))

    # The scenario must actually happen, or this test proves nothing.
    assert _reused_across_kinds(model.alias_ids) > 0, model.alias_ids[:8]

    buffer_ops = [op for op in trace.layer_list if op.is_buffer]
    assert buffer_ops
    consumer_types = {trace[child].func_name for op in buffer_ops for child in op.children}
    assert consumer_types == {"view"}, consumer_types


def test_storage_metadata_cache_does_not_outlive_its_tensor() -> None:
    """A cached storage key/range is dropped when its tensor dies, not served to an id twin."""

    model = nn.Linear(4, 4)
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
        # No reuse on this allocator: whatever the cache kept for the dead id must not
        # still claim to describe a live object.
        for cache in (tracker._storage_key_cache, tracker._storage_range_cache):
            for (cached_id, _), (ref, _) in cache.items():
                assert cached_id != first_id or ref() is None
        return
    assert tracker.storage_key(twin) != first_key
    assert tracker.storage_range(twin) == (0, 32)
