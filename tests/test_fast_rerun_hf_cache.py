"""Steered fast rerun on stock Hugging Face forwards whose output carries a KV cache.

A stock ``GPT2LMHeadModel`` or ``Qwen3ForCausalLM`` forward with ``use_cache=True``
returns a ``ModelOutput`` carrying a ``DynamicCache`` of per-layer key/value
tensors. Capture descends into it (one output op per cached tensor, each with a
typed container path), so the guarded fast engine must pair the native output's
leaves with those ops by path instead of by a generic leaf count; otherwise every
steered rerun of a stock forward fell back to the capture engine
(``output_structure_mismatch:fast_live_model_output_structure``).

Pinned here: eight generation steps on each model are exact to 0.0 against a plain
forward hook with no fallback, the explicit ``fast=True`` door agrees, and a planted
control-flow change (an extra torch call once the input grows past a length) still
refuses the fast engine while the legacy door's fallback stays exact.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors.runnable import PathDivergenceError

transformers = pytest.importorskip("transformers")

_MAGNITUDE = 4.0
_PROMPT_LEN = 6
_STEPS = 8


def _gpt2() -> tuple[nn.Module, str, str, int]:
    """Return a tiny stock GPT-2, its steering site, its head address and hidden size."""

    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=128,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_positions=64,
        bos_token_id=0,
        eos_token_id=0,
    )
    config._attn_implementation = "eager"
    return transformers.GPT2LMHeadModel(config).eval(), "transformer.h.1", "lm_head", 32


def _qwen3() -> tuple[nn.Module, str, str, int]:
    """Return a tiny stock Qwen3, its steering site, its head address and hidden size."""

    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        bos_token_id=0,
        eos_token_id=0,
    )
    config._attn_implementation = "eager"
    return transformers.Qwen3ForCausalLM(config).eval(), "model.layers.1", "lm_head", 32


_MODELS: dict[str, Callable[[], tuple[nn.Module, str, str, int]]] = {"gpt2": _gpt2, "qwen3": _qwen3}


def _ids(length: int, seed: int) -> torch.Tensor:
    """Deterministic token ids of the given length."""

    return torch.randint(0, 128, (1, length), generator=torch.Generator().manual_seed(seed))


def _hooked_logits(
    model: nn.Module, site: str, direction: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    """Logits of the stock forward (``use_cache=True``) under a plain steering forward hook."""

    def steer(_module: nn.Module, _args: Any, out: Any) -> Any:
        hidden = out[0] if isinstance(out, tuple) else out
        changed = hidden + direction.to(hidden.dtype) * _MAGNITUDE
        return (changed, *out[1:]) if isinstance(out, tuple) else changed

    handle = model.get_submodule(site).register_forward_hook(steer)
    try:
        with torch.no_grad():
            return model(ids, use_cache=True).logits
    finally:
        handle.remove()


def _steered_trace(
    model: nn.Module, site: str, head: str, direction: torch.Tensor, ids: torch.Tensor
) -> Any:
    """Capture the stock forward (cache returned) with the steering spec staged."""

    spec = tl.when(tl.module(site), tl.steer(direction, magnitude=_MAGNITUDE, feature_axis=-1))
    return tl.trace(model, ids, save=tl.module(site) | tl.module(head), intervene=spec)


def _head_out(trace: Any, head: str) -> torch.Tensor:
    return trace.find_sites(tl.module(head)).first().out


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_stock_forward_with_kv_cache_reruns_fast_over_generation(name: str) -> None:
    """Eight growing-length steered reruns of a stock cached forward: fast, exact, no fallback."""

    model, site, head, hidden = _MODELS[name]()
    direction = torch.randn(hidden, generator=torch.Generator().manual_seed(1))
    ids = _ids(_PROMPT_LEN, seed=2)
    trace = _steered_trace(model, site, head, direction, ids)
    assert len(trace.output_layers) > 1, "capture descends into the returned cache"

    generator = torch.Generator().manual_seed(3)
    for step in range(1, _STEPS + 1):
        ids = torch.cat([ids, torch.randint(0, 128, (1, 1), generator=generator)], dim=1)
        reference = _hooked_logits(model, site, direction, ids)
        trace.run(model, ids)
        assert trace.last_run["engine"] == "guarded_fast", (
            step,
            trace.last_run.get("fast_refused"),
        )
        assert trace.last_run["fast_refused"] is None
        assert trace.last_run["shape_varied"] is True
        assert _max_abs_diff(_head_out(trace, head), reference) == 0.0, step


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_fast_door_accepts_stock_cached_output(name: str) -> None:
    """``run(inputs=..., fast=True)`` pairs the cached output by path at the same and a longer length."""

    model, site, head, hidden = _MODELS[name]()
    direction = torch.randn(hidden, generator=torch.Generator().manual_seed(1))
    ids = _ids(_PROMPT_LEN, seed=2)
    trace = _steered_trace(model, site, head, direction, ids)
    for length in (_PROMPT_LEN, _PROMPT_LEN + 3):
        probe = _ids(length, seed=4)
        reference = _hooked_logits(model, site, direction, probe)
        trace.run(inputs=probe, fast=True)
        assert trace.last_run["engine"] == "guarded_fast"
        assert _max_abs_diff(_head_out(trace, head), reference) == 0.0


class _LengthBranch(nn.Module):
    """A stock model whose forward takes an extra torch call once the input is long."""

    def __init__(self, network: nn.Module, threshold: int) -> None:
        super().__init__()
        self.network = network
        self.threshold = threshold

    def forward(self, ids: torch.Tensor) -> Any:
        out = self.network(ids, use_cache=True)
        if ids.shape[1] > self.threshold:
            out.logits = torch.tanh(out.logits)
        return out


def test_planted_control_flow_change_still_refuses_fast_with_cached_output() -> None:
    """A real structural change refuses the fast engine; the legacy fallback stays exact."""

    network, site, head, hidden = _gpt2()
    model = _LengthBranch(network, threshold=_PROMPT_LEN + 1).eval()
    site, head = f"network.{site}", f"network.{head}"
    direction = torch.randn(hidden, generator=torch.Generator().manual_seed(1))
    ids = _ids(_PROMPT_LEN, seed=2)
    trace = _steered_trace(model, site, head, direction, ids)

    short_ids = _ids(_PROMPT_LEN + 1, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.run(model, short_ids)
    assert trace.last_run["engine"] == "guarded_fast", trace.last_run.get("fast_refused")

    long_ids = _ids(_PROMPT_LEN + 2, seed=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.run(model, long_ids)
    assert trace.last_run["engine"] == "rerun"
    assert str(trace.last_run["fast_refused"]).endswith(":fast_live_call_fingerprint")

    def hooked_head(ids: torch.Tensor) -> torch.Tensor:
        """The head module's output (before the planted tanh) under the plain steering hook."""

        seen: list[torch.Tensor] = []

        def steer(_module: nn.Module, _args: Any, out: Any) -> Any:
            hidden_state = out[0] if isinstance(out, tuple) else out
            changed = hidden_state + direction.to(hidden_state.dtype) * _MAGNITUDE
            return (changed, *out[1:]) if isinstance(out, tuple) else changed

        def record(_module: nn.Module, _args: Any, out: Any) -> None:
            seen.append(out.detach().clone())

        handles = [
            model.get_submodule(site).register_forward_hook(steer),
            model.get_submodule(head).register_forward_hook(record),
        ]
        try:
            with torch.no_grad():
                model(ids)
        finally:
            for handle in handles:
                handle.remove()
        return seen[0]

    assert _max_abs_diff(_head_out(trace, head), hooked_head(long_ids)) == 0.0

    explicit = _steered_trace(model, site, head, direction, ids)
    with pytest.raises(PathDivergenceError):
        explicit.run(inputs=long_ids, fast=True)
