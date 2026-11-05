"""Dropping a trace releases every payload tensor, in few Python calls per op.

A dropped trace dies by reference counting, and its last owning core's death runs
``OpRowStore._evict_tensor_payloads`` right there. The walk used to call itself on
every cell that was not a plain scalar: tuples of scalars (tensor shapes and the
like) and TorchLens's own record objects, none of which can hold a payload the walk
would evict. On a tiny GPT-2 ``generate`` that was about 87 Python calls per op. The
walk now learns, once per eviction, which object types it always returns unchanged
and skips tuples whose members are all such types. These tests pin the teardown call
count and that every payload tensor is still freed when the trace is dropped while
one of its ops is kept.
"""

from __future__ import annotations

import os
import sys
import threading
import weakref
from collections.abc import Callable, Iterator
from types import FrameType
from typing import Any

import pytest
import torch
from _hermetic_counters import quiet_gc

import torchlens as tl
import torchlens.backends.torch.ops as torch_ops

# Python calls into TorchLens code per function-call op while a dropped GPT-2
# generate trace is torn down (gc off, so it dies by refcount alone). Measured
# __MEASURED__ on the pinned CI environment (86.93 before the walk skipped opaque
# types and all-scalar tuples).
_MAX_TEARDOWN_CALLS_PER_OP = 0.0
_TORCHLENS_ROOT = os.path.dirname(tl.__file__)


def _tiny_gpt2_generate() -> tuple[Any, Callable[..., Any], torch.Tensor, dict[str, Any]]:
    """A random-weight two-layer GPT-2, its bound ``generate`` and three-token kwargs."""

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=2, pad_token_id=0
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    kwargs = {"max_new_tokens": 3, "do_sample": False, "pad_token_id": 0}
    return model, model.generate, torch.tensor([[3, 7, 1, 9]]), kwargs


def _trace_counting_ops(fn: Callable[..., Any], x: Any, kw: dict[str, Any]) -> tuple[Any, int]:
    """Trace ``fn`` and count the function-call ops frozen (``func_call_id`` set)."""

    real_freeze = torch_ops._op_record_from_log
    ops = 0

    def freeze(trace: Any, fields_dict: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Count one function-call op, then freeze it."""

        nonlocal ops
        if fields_dict.get("func_call_id") is not None:
            ops += 1
        return real_freeze(trace, fields_dict, *args, **kwargs)

    torch_ops._op_record_from_log = freeze
    try:
        traced = tl.trace(fn, x, input_kwargs=kw or None)
    finally:
        torch_ops._op_record_from_log = real_freeze
    return traced, ops


def _teardown_calls(holder: list[Any]) -> int:
    """Python calls into TorchLens code while the trace in ``holder`` is dropped."""

    owner = threading.get_ident()
    calls = 0

    def profile(frame: FrameType, event: str, _arg: Any) -> None:
        """Count Python-level calls whose code lives in the torchlens package."""

        nonlocal calls
        if (
            event == "call"
            and frame.f_code.co_filename.startswith(_TORCHLENS_ROOT)
            and threading.get_ident() == owner
        ):
            calls += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        holder.clear()
    finally:
        sys.setprofile(previous)
    return calls


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


def test_gpt2_generate_trace_teardown_python_calls_per_op() -> None:
    """Dropping a GPT-2 generate trace makes at most the pinned Python calls per op."""

    _model, fn, ids, kw = _tiny_gpt2_generate()
    tl.trace(fn, ids, input_kwargs=kw)  # warm one-time instrumentation
    with quiet_gc():
        traced, ops = _trace_counting_ops(fn, ids, kw)
        store = traced._trace_core.ops
        holder = [traced]
        del traced
        calls = _teardown_calls(holder)
        assert store._payload_owner_count == 0, "the trace died by refcount inside the window"
    assert ops > 100
    per_op = calls / ops
    print(f"[teardown] ops={ops} calls={calls} teardown_calls_per_op={per_op:.2f}")
    assert per_op <= _MAX_TEARDOWN_CALLS_PER_OP


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
