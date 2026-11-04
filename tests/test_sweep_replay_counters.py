"""Deterministic counters and bit equality for a fork + head patch + replay sweep.

The README's patching sweep runs ``italy.fork().do(module & head, patch_from(france))``
once per attention head. Each replay used to rebuild the selector label index
three times (hook validation, the replay's origin pass, its hook-target pass),
build the replay label map and call groups twice (the cone walk and the replay
loop), and copy every digested capture payload into a ``bytes`` object before
hashing it. These tests pin one build of each per replay and no payload copy,
and check that every replay in the sweep still equals a plain forward with the
same head patched by hand, bit for bit.
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl
from torchlens._ir import selector_eval
from torchlens.intervention import _replay_cone, _replay_context, _replay_values

_N_LAYER, _N_HEAD, _N_EMBD = 2, 2, 8
_HEAD_DIM = _N_EMBD // _N_HEAD

_COUNTED = {
    selector_eval._label_site_index.__code__: "label_site_index",
    _replay_cone._label_key_map.__code__: "label_key_map",
    _replay_context._func_call_groups.__code__: "func_call_groups",
}


def _model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=_N_LAYER,
        n_head=_N_HEAD,
        n_embd=_N_EMBD,
        n_positions=16,
        vocab_size=32,
        attn_implementation="eager",
    )
    return GPT2LMHeadModel(config).eval()


@pytest.fixture(scope="module")
def pair() -> Iterator[tuple[GPT2LMHeadModel, torch.Tensor, torch.Tensor, tl.Trace, tl.Trace]]:
    model = _model()
    ids_f = torch.tensor([[1, 5, 7, 3, 9]])
    ids_i = torch.tensor([[1, 5, 7, 4, 9]])
    france = tl.trace(model, ids_f)
    try:
        italy = tl.trace(model, ids_i)
        try:
            yield model, ids_f, ids_i, france, italy
        finally:
            italy.cleanup()
    finally:
        france.cleanup()


def _head(block: int, head: int) -> Any:
    return tl.module(f"transformer.h.{block}.attn") & tl.head(head, "v")


def _patched_logits(italy: tl.Trace, france: tl.Trace, block: int, head: int) -> torch.Tensor:
    patched = italy.fork().do(_head(block, head), tl.patch_from(france))
    return patched["lm_head"].out.detach().clone()


def _counting(fn: Any) -> Counter[str]:
    counts: Counter[str] = Counter()

    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == "call":
            name = _COUNTED.get(frame.f_code)
            if name is not None:
                counts[name] += 1

    sys.setprofile(profile)
    try:
        fn()
    finally:
        sys.setprofile(None)
    return counts


def test_one_head_replay_builds_each_structural_index_once(
    pair: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fork + head patch + replay: one label index, one label map, one call grouping.

    Pinned on the private path (sharing across forks turned off): a clean fork
    that reuses its parent's shared indexes builds none, which
    ``tests/test_fork_shared_label_index.py`` pins.
    """

    from torchlens._trace_core import label_cells

    monkeypatch.setattr(label_cells, "shared_entry", lambda ops: None)
    monkeypatch.setattr(_replay_cone, "shared_entry", lambda ops: None)
    _, _, _, france, italy = pair
    _patched_logits(italy, france, 0, 0)  # first use settles lazy imports and caches
    counts = _counting(lambda: _patched_logits(italy, france, 1, 1))
    assert dict(counts) == {"label_site_index": 1, "label_key_map": 1, "func_call_groups": 1}


def test_replay_digests_capture_payloads_without_copying_them(
    pair: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every capture digest hashes the payload in place: no ``bytes`` copy per site."""

    _, _, _, france, italy = pair
    inputs: Counter[str] = Counter()

    def sha256(data: Any = b"") -> Any:
        inputs[type(data).__name__] += 1
        return hashlib.sha256(data)

    monkeypatch.setattr(_replay_values, "hashlib", SimpleNamespace(sha256=sha256))
    _patched_logits(italy, france, 1, 0)
    assert inputs["memoryview"] > 0, inputs
    assert inputs["bytes"] == 0, inputs


def _hand_patched_logits(
    model: GPT2LMHeadModel, ids_f: torch.Tensor, ids_i: torch.Tensor, block: int, head: int
) -> torch.Tensor:
    """Plain forward on ``ids_i`` with one head's values taken from ``ids_f``."""

    c_attn = model.transformer.h[block].attn.c_attn
    cols = slice(2 * _N_EMBD + head * _HEAD_DIM, 2 * _N_EMBD + (head + 1) * _HEAD_DIM)
    saved: dict[str, torch.Tensor] = {}
    keep = c_attn.register_forward_hook(lambda m, a, out: saved.__setitem__("v", out[..., cols]))
    with torch.no_grad():
        model(ids_f)
    keep.remove()

    def patch(module: Any, args: Any, out: torch.Tensor) -> torch.Tensor:
        out = out.clone()
        out[..., cols] = saved["v"]
        return out

    swap = c_attn.register_forward_hook(patch)
    try:
        with torch.no_grad():
            return model(ids_i).logits.detach().clone()
    finally:
        swap.remove()


def test_head_sweep_replays_equal_hand_patched_forwards(pair: Any) -> None:
    """Every (block, head) replay equals the hand-patched plain forward bit for bit."""

    model, ids_f, ids_i, france, italy = pair
    for block in range(_N_LAYER):
        for head in range(_N_HEAD):
            replayed = _patched_logits(italy, france, block, head)
            expected = _hand_patched_logits(model, ids_f, ids_i, block, head)
            assert torch.equal(replayed, expected), (block, head)
    assert torch.equal(italy["lm_head"].out, model(ids_i).logits.detach())
