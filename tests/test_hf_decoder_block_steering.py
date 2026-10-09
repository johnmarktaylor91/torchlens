"""Steering a HF decoder block as a MODULE site, KV cache on, agrees with a forward hook.

The common steering case: ``tl.when(tl.module("model.layers.0"), tl.steer(v))`` on a causal
LM whose ``use_cache`` is on. A module site is not an ``in_module`` region, so the region
refusal for exits into returned cache tensors (``region_exit_address_underivable``) must
never reach it. Every door (``tl.trace(intervene=)``, ``tl.record(intervene=,
return_output=True)``, ``spec.bind(model)(ids)`` and ``spec.bind(model).generate(...)``)
must match an eager ``register_forward_hook`` oracle that applies the same edit to the
block's hidden-state output.

The model is a tiny random-init causal LM built from its config (no download): Qwen2 when
this ``transformers`` has it, else Llama, else GPT-2.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

transformers = pytest.importorskip("transformers")

_HIDDEN = 32
_VOCAB = 97  # distinct from every other tensor dimension, so logits are found by shape
_IDS = torch.tensor([[3, 17, 42, 8, 61, 5]])
_NEW_TOKENS = 4


def _tiny_causal_lm() -> tuple[nn.Module, str]:
    """Build a seeded tiny random-init causal LM with the KV cache on.

    Returns
    -------
    tuple[nn.Module, str]
        The eval-mode model and the address of its first decoder block.
    """

    torch.manual_seed(0)
    common = {"vocab_size": _VOCAB, "use_cache": True, "pad_token_id": 0, "eos_token_id": None}
    if hasattr(transformers, "Qwen2ForCausalLM"):
        config = transformers.Qwen2Config(
            hidden_size=_HIDDEN,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            **common,
        )
        return transformers.Qwen2ForCausalLM(config).eval(), "model.layers.0"
    if hasattr(transformers, "LlamaForCausalLM"):
        config = transformers.LlamaConfig(
            hidden_size=_HIDDEN,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            **common,
        )
        return transformers.LlamaForCausalLM(config).eval(), "model.layers.0"
    config = transformers.GPT2Config(n_embd=_HIDDEN, n_layer=2, n_head=4, n_positions=64, **common)
    return transformers.GPT2LMHeadModel(config).eval(), "transformer.h.0"


def _direction() -> torch.Tensor:
    """Return a fixed steering vector over the hidden axis.

    Returns
    -------
    torch.Tensor
        Shape ``(hidden,)``.
    """

    return torch.linspace(-1.0, 1.0, _HIDDEN)


#: ``(edit id, spec helper factory, eager equivalent on the block's hidden state)``.
#: ``tl.steer(v, m, feature_axis=-1)`` adds ``v * m`` broadcast along the last axis
#: (``_align_direction`` reshapes a vector to ``[1, ..., hidden]``); ``tl.add(c)`` adds c.
_EDITS: tuple[tuple[str, Callable[[], Any], Callable[[torch.Tensor], torch.Tensor]], ...] = (
    (
        "steer",
        lambda: tl.steer(_direction(), 2.0, feature_axis=-1),
        lambda h: h + _direction().reshape(1, 1, _HIDDEN).to(h.dtype) * 2.0,
    ),
    ("add", lambda: tl.add(1.0), lambda h: h + 1.0),
)
_EDIT_IDS = [edit[0] for edit in _EDITS]


def _hook_on_hidden(edit: Callable[[torch.Tensor], torch.Tensor]) -> Callable[..., Any]:
    """Wrap an edit as a forward hook that rewrites only the block's hidden state.

    Decoder blocks return the hidden-state tensor (recent ``transformers``) or a tuple whose
    first element is the hidden state, followed by optional attention weights and cache
    (older ``transformers``); the cache is a ``Cache`` object, not a tensor leaf.

    Parameters
    ----------
    edit:
        Function from hidden state to edited hidden state.

    Returns
    -------
    Callable[..., Any]
        A ``register_forward_hook`` callable.
    """

    def _hook(_module: nn.Module, _args: Any, out: Any) -> Any:
        """Return the block output with its hidden state edited."""

        if isinstance(out, tuple):
            return (edit(out[0]), *out[1:])
        return edit(out)

    return _hook


def _oracle_logits(
    model: nn.Module, address: str, edit: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Run the model with the edit applied by an eager forward hook.

    Parameters
    ----------
    model:
        Model under test.
    address:
        Decoder block address.
    edit:
        Hidden-state edit.

    Returns
    -------
    torch.Tensor
        The steered logits.
    """

    handle = model.get_submodule(address).register_forward_hook(_hook_on_hidden(edit))
    try:
        with torch.no_grad():
            return model(_IDS).logits
    finally:
        handle.remove()


def _oracle_generate(
    model: nn.Module, address: str, edit: Callable[[torch.Tensor], torch.Tensor]
) -> Any:
    """Greedy-generate with the edit applied by an eager forward hook.

    Parameters
    ----------
    model:
        Model under test.
    address:
        Decoder block address.
    edit:
        Hidden-state edit.

    Returns
    -------
    Any
        The ``generate`` output with per-step logits.
    """

    handle = model.get_submodule(address).register_forward_hook(_hook_on_hidden(edit))
    try:
        return _generate(model.generate)
    finally:
        handle.remove()


def _generate(generate: Callable[..., Any]) -> Any:
    """Call a ``generate`` for a few greedy tokens, returning sequences and logits.

    Parameters
    ----------
    generate:
        ``model.generate`` or ``spec.bind(model).generate``.

    Returns
    -------
    Any
        The generation output (``sequences`` and per-step ``logits``).
    """

    with torch.no_grad():
        return generate(
            _IDS,
            max_new_tokens=_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            return_dict_in_generate=True,
            output_logits=True,
        )


def _assert_steered(got: torch.Tensor, model: nn.Module, address: str, edit: Callable) -> None:
    """Assert exact agreement with the hook oracle, and that the edit changed the logits.

    Exact equality holds because every door runs the same CPU kernels on the same inputs
    and the edit is the same elementwise add at the same place.

    Parameters
    ----------
    got:
        Door logits.
    model:
        Model under test.
    address:
        Decoder block address.
    edit:
        Hidden-state edit.
    """

    want = _oracle_logits(model, address, edit)
    with torch.no_grad():
        plain = model(_IDS).logits
    assert not torch.equal(want, plain), "the oracle edit must change the logits"
    assert torch.equal(got.detach(), want), (got.detach() - want).abs().max()


@pytest.mark.parametrize(("name", "helper", "edit"), _EDITS, ids=_EDIT_IDS)
def test_trace_intervene_matches_hook(name: str, helper: Callable, edit: Callable) -> None:
    """``tl.trace(intervene=)`` at the block module site matches the hook oracle."""

    model, address = _tiny_causal_lm()
    trace = tl.trace(model, _IDS, intervene=tl.when(tl.module(address), helper()))
    logits = [
        op.out for op in trace.output_ops if tuple(op.out.shape) == (1, _IDS.shape[1], _VOCAB)
    ]
    assert len(logits) == 1, name
    _assert_steered(logits[0], model, address, edit)


@pytest.mark.parametrize(("name", "helper", "edit"), _EDITS, ids=_EDIT_IDS)
def test_record_intervene_matches_hook(name: str, helper: Callable, edit: Callable) -> None:
    """``tl.record(intervene=, return_output=True)`` returns the hook oracle's logits."""

    model, address = _tiny_causal_lm()
    output, _recording = tl.record(
        model,
        _IDS,
        save=tl.module(address),
        intervene=tl.when(tl.module(address), helper()),
        return_output=True,
    )
    _assert_steered(output.logits, model, address, edit)


@pytest.mark.parametrize(("name", "helper", "edit"), _EDITS, ids=_EDIT_IDS)
def test_bind_call_matches_hook(name: str, helper: Callable, edit: Callable) -> None:
    """``spec.bind(model)(ids)`` returns the hook oracle's logits."""

    model, address = _tiny_causal_lm()
    with torch.no_grad():
        output = tl.when(tl.module(address), helper()).bind(model)(_IDS)
    _assert_steered(output.logits, model, address, edit)


@pytest.mark.parametrize(("name", "helper", "edit"), _EDITS, ids=_EDIT_IDS)
def test_bind_generate_matches_hook_steered_generate(
    name: str, helper: Callable, edit: Callable
) -> None:
    """``spec.bind(model).generate`` matches a hook-steered greedy ``model.generate``."""

    model, address = _tiny_causal_lm()
    want = _oracle_generate(model, address, edit)
    plain = _generate(model.generate)
    got = _generate(tl.when(tl.module(address), helper()).bind(model).generate)
    assert torch.equal(got.sequences, want.sequences), (got.sequences, want.sequences)
    assert len(got.logits) == len(want.logits) == _NEW_TOKENS
    for step, (got_step, want_step) in enumerate(zip(got.logits, want.logits, strict=True)):
        assert torch.equal(got_step, want_step), f"{name}: step {step} logits differ"
    assert not all(torch.equal(a, b) for a, b in zip(want.logits, plain.logits, strict=True)), (
        "the oracle edit must change the generation logits"
    )
