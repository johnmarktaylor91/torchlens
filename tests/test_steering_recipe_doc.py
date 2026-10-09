"""The steering recipe in ``docs/agent-reference/common-patterns.md`` runs and is exact.

The recipe is the documented answer to "many forwards with one intervention":
``spec.bind(model)`` for steering, ``tl.record(..., intervene=spec)`` when
activations are needed, and one ``tl.trace`` as the oracle. This test executes the
page's fence as written and checks every path against a plain forward hook that
adds the same direction, the reference a user would otherwise write by hand.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl

PAGE = Path(__file__).resolve().parents[1] / "docs" / "agent-reference" / "common-patterns.md"
HEADING = "### Steering many forwards / generation"
FENCE_RE = re.compile(r"```python\n(?P<code>.*?)\n```", re.DOTALL)


def _recipe_code() -> str:
    text = PAGE.read_text(encoding="utf-8")
    assert HEADING in text, f"{PAGE.name} lost the steering recipe heading"
    match = FENCE_RE.search(text, text.index(HEADING))
    assert match is not None, "the steering recipe has no python fence"
    return match.group("code")


@pytest.fixture(scope="module")
def recipe() -> dict[str, Any]:
    pytest.importorskip("transformers")
    namespace: dict[str, Any] = {"tl": tl}
    exec(compile(_recipe_code(), f"{PAGE.name}:steering-recipe", "exec"), namespace)
    return namespace


def _plain_hook_reference(ns: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits and greedy tokens with a hand-written forward hook on the same module."""

    lm, ids, direction = ns["lm"], ns["ids"], ns["direction"]
    handle = lm.model.layers[1].mlp.register_forward_hook(
        lambda _module, _args, output: output + 4.0 * direction
    )
    try:
        logits = lm(ids).logits
        tokens = lm.generate(ids, max_new_tokens=5, do_sample=False, use_cache=True)
    finally:
        handle.remove()
    return logits, tokens


def test_recipe_bind_matches_plain_hook_exactly(recipe):
    """bind's call and generate (KV cache on) equal the plain hook bit for bit."""

    logits, tokens = _plain_hook_reference(recipe)
    assert torch.equal(recipe["logits"], logits)
    assert torch.equal(recipe["tokens"], tokens)
    unsteered = recipe["lm"].generate(
        recipe["ids"], max_new_tokens=5, do_sample=False, use_cache=True
    )
    assert not torch.equal(recipe["tokens"], unsteered), "the steer must change generation"


def test_recipe_bind_generate_cache_off_agrees(recipe):
    """The bound generate gives the same tokens with the KV cache off."""

    no_cache = recipe["steered"].generate(
        recipe["ids"], max_new_tokens=5, do_sample=False, use_cache=False
    )
    assert torch.equal(no_cache, recipe["tokens"])


def test_recipe_record_and_trace_match_plain_hook(recipe):
    """tl.record's output and the trace oracle's saved site equal the plain hook."""

    logits, _tokens = _plain_hook_reference(recipe)
    assert torch.equal(recipe["out"].logits, logits)
    lm, ids, direction = recipe["lm"], recipe["ids"], recipe["direction"]
    captured: list[torch.Tensor] = []
    handle = lm.model.layers[1].mlp.register_forward_hook(
        lambda _module, _args, output: captured.append(output + 4.0 * direction) or captured[-1]
    )
    try:
        lm(ids)
    finally:
        handle.remove()
    assert torch.equal(recipe["steered_mlp"], captured[0])


def test_recipe_leaves_model_hook_free(recipe):
    """Binding, recording and tracing leave no hooks on the user's model."""

    assert not any(
        module._forward_hooks or module._forward_pre_hooks for module in recipe["lm"].modules()
    )
