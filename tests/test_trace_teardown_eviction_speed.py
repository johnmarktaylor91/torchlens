"""Dropping a trace still frees every payload tensor after the eviction walk got cheaper.

A dropped trace dies by reference counting, and its last owning core's death runs
``OpRowStore._evict_tensor_payloads`` right there. The walk used to call itself on
every cell that was not a plain scalar: tuples of scalars (tensor shapes and the
like) and TorchLens's own record objects, none of which can hold a payload the walk
would evict. It now learns, once per eviction, which object types it always returns
unchanged and skips tuples whose members are all such types. The teardown call count
is pinned in ``test_capture_fields_dict_prune.py``; this test pins that every payload
tensor is still freed when the trace is dropped while one of its ops is kept.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from typing import Any

import torch
from _hermetic_counters import quiet_gc

import torchlens as tl


def _tiny_gpt2_generate() -> tuple[Any, Callable[..., Any], torch.Tensor, dict[str, Any]]:
    """A random-weight two-layer GPT-2, its bound ``generate`` and three-token kwargs."""

    import transformers

    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=2, pad_token_id=0
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    kwargs = {"max_new_tokens": 3, "do_sample": False, "pad_token_id": 0}
    return model, model.generate, torch.tensor([[3, 7, 1, 9]]), kwargs


def _tensors_in(value: Any, out: list[torch.Tensor], depth: int = 0) -> None:
    """Append every tensor reachable from ``value`` through builtin containers."""

    if isinstance(value, torch.Tensor):
        out.append(value)
    elif depth < 12 and isinstance(value, dict):
        for item in list(value.values()):
            _tensors_in(item, out, depth + 1)
    elif depth < 12 and isinstance(value, (list, tuple, set, frozenset)):
        for item in list(value):
            _tensors_in(item, out, depth + 1)


def _store_values(store: Any) -> Iterator[Any]:
    """Every cell value the store holds: rows, unpacked columns and the overlay."""

    if store._rows is not None:
        for row in store._rows:
            yield from row
    if store._columns is not None:
        for column in store._columns:
            if not column.packed:
                yield from column.values
    yield from store._overlay.values()


def test_dropping_a_gpt2_trace_frees_every_payload_tensor_while_an_op_is_kept() -> None:
    """With one Op retained, every captured payload tensor dies when the trace is dropped."""

    model, fn, ids, kw = _tiny_gpt2_generate()
    external = {id(t) for t in (*model.parameters(), *model.buffers(), ids)}
    with quiet_gc():
        traced = tl.trace(fn, ids, input_kwargs=kw)
        store = traced._trace_core.ops
        kept_op = traced._layer_list[len(traced._layer_list) // 2]
        payloads: list[torch.Tensor] = []
        for value in _store_values(store):
            _tensors_in(value, payloads)
        refs = [weakref.ref(t) for t in payloads if id(t) not in external]
        del payloads, value
        assert len(refs) > 100, "the capture keeps its payload tensors"
        del traced
        assert store._payload_owner_count == 0
        alive = sum(ref() is not None for ref in refs)
        assert alive == 0, f"{alive} of {len(refs)} payload tensors outlived the dropped trace"
        assert kept_op.layer_label, "the retained Op stays readable"
