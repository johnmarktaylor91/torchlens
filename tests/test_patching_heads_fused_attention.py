"""Per-head activation patching on GPT-2, eager and fused-attention (``sdpa``).

``result`` (``z_h @ W_O_h``) is a computed facet no captured op holds, so the
default ``activation_patch_attention_heads`` used to refuse on EVERY GPT-2; it
now writes the head's ``z``, which replaces exactly its ``result``. Under
``scaled_dot_product_attention`` the ``z`` head facet is itself reconstructed
read-only from the kernel's saved arguments, so
``activation_patch_attention_heads`` and ``patch_heads_grid`` (``z``) cannot
write it. They refuse with the cause and
the remedies named; the eager switch the refusal names works on the same model,
and the writable pre-kernel ``v`` facet patches on the fused model as is. Random
weights, no download.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp  # noqa: F401  (registers tl.mechinterp)
from torchlens.errors import SiteResolutionError

transformers = pytest.importorskip("transformers")


def _gpt2(attn_implementation: str) -> Any:
    config = transformers.GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=32,
        vocab_size=64,
        n_positions=16,
        bos_token_id=0,
        eos_token_id=0,
        use_cache=False,
        attn_implementation=attn_implementation,
    )
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config).eval()


def _prompts() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randint(0, 64, (1, 5), generator=generator),
        torch.randint(0, 64, (1, 5), generator=generator),
    )


def _metric(trace: Any) -> torch.Tensor:
    logits = trace[trace.output_layers[0]].out
    return logits[0, -1, 3] - logits[0, -1, 7]


@pytest.mark.parametrize("facet_name", ["result", "z"])
def test_fused_head_patching_refuses_with_the_cause_and_the_remedies(facet_name: str) -> None:
    model = _gpt2("sdpa")
    clean, corrupt = _prompts()
    with pytest.raises(SiteResolutionError) as excinfo:
        tl.facets.patching.activation_patch_attention_heads(
            model, clean, corrupt, _metric, facet_name=facet_name
        )
    message = str(excinfo.value)
    assert "read-only" in message
    # ``result`` is written through ``z`` (see the eager test below); both are
    # reconstructed read-only under the fused kernel.
    assert "'z' facet is reconstructed read-only" in message
    assert "attn_implementation='eager'" in message
    assert "set_attn_implementation('eager')" in message
    assert "'q'/'k'/'v' facets" in message
    assert "tl.mechinterp.patch_head" in message


def test_patch_heads_grid_refuses_the_same_way_on_fused_attention() -> None:
    model = _gpt2("sdpa")
    clean, corrupt = _prompts()
    with pytest.raises(SiteResolutionError, match="attn_implementation='eager'"):
        tl.mechinterp.patch_heads_grid(model, clean, corrupt, _metric)


def test_the_named_eager_switch_makes_head_patching_work_on_the_same_model() -> None:
    model = _gpt2("sdpa")
    clean, corrupt = _prompts()
    model.set_attn_implementation("eager")
    switched = tl.facets.patching.activation_patch_attention_heads(model, clean, corrupt, _metric)
    eager = tl.facets.patching.activation_patch_attention_heads(
        _gpt2("eager"), clean, corrupt, _metric
    )
    assert switched.shape == (2, 4)
    torch.testing.assert_close(switched, eager)
    assert switched.abs().sum() > 0


def test_the_writable_v_facet_patches_on_the_fused_model() -> None:
    clean, corrupt = _prompts()
    fused = tl.facets.patching.activation_patch_attention_heads(
        _gpt2("sdpa"), clean, corrupt, _metric, facet_name="v"
    )
    eager = tl.facets.patching.activation_patch_attention_heads(
        _gpt2("eager"), clean, corrupt, _metric, facet_name="v"
    )
    assert fused.shape == (2, 4)
    torch.testing.assert_close(fused, eager, atol=1e-5, rtol=1e-4)


def _c_proj_head_oracle(model: Any, clean: torch.Tensor, corrupt: torch.Tensor) -> torch.Tensor:
    """Plain torch: swap one head's slice of each ``c_proj`` input for the clean slice."""

    n_layer, n_head = len(model.transformer.h), model.config.n_head
    d_head = model.config.n_embd // n_head
    values = torch.empty(n_layer, n_head)
    for layer in range(n_layer):
        c_proj = model.transformer.h[layer].attn.c_proj
        captured: dict[str, torch.Tensor] = {}
        grab = c_proj.register_forward_pre_hook(
            lambda _m, args: captured.setdefault("clean", args[0].detach().clone())
        )
        with torch.no_grad():
            model(clean)
        grab.remove()
        for head in range(n_head):
            cols = slice(head * d_head, (head + 1) * d_head)

            def _swap(_m: Any, args: tuple[Any, ...], cols: slice = cols) -> tuple[Any, ...]:
                patched = args[0].clone()
                patched[..., cols] = captured["clean"][..., cols]
                return (patched, *args[1:])

            swap = c_proj.register_forward_pre_hook(_swap)
            try:
                with torch.no_grad():
                    logits = model(corrupt).logits
            finally:
                swap.remove()
            values[layer, head] = logits[0, -1, 3] - logits[0, -1, 7]
    return values


def test_default_result_head_patching_works_on_eager_gpt2_and_matches_the_oracle() -> None:
    model = _gpt2("eager")
    clean, corrupt = _prompts()
    by_result = tl.facets.patching.activation_patch_attention_heads(model, clean, corrupt, _metric)
    by_z = tl.facets.patching.activation_patch_attention_heads(
        model, clean, corrupt, _metric, facet_name="z"
    )
    torch.testing.assert_close(by_result, by_z, rtol=0, atol=0)
    torch.testing.assert_close(
        by_result, _c_proj_head_oracle(model, clean, corrupt), atol=1e-5, rtol=0
    )


def test_a_direct_result_head_write_names_the_computed_cause() -> None:
    """``tl.head(h, "result")`` on eager attention: read-only because computed, not fused."""

    model = _gpt2("eager")
    clean, corrupt = _prompts()
    options = tl.options.CaptureOptions(replayable=True)
    clean_trace = tl.trace(model, clean, capture=options)
    corrupt_trace = tl.trace(model, corrupt, capture=options)
    try:
        with pytest.raises(SiteResolutionError) as excinfo:
            corrupt_trace.fork().do(
                tl.head(0, "result").in_module("transformer.h.0.attn"),
                tl.patch_from(clean_trace),
            )
    finally:
        clean_trace.cleanup()
        corrupt_trace.cleanup()
    message = str(excinfo.value)
    assert "read-only" in message
    assert "computed from other captured values" in message
    assert "under every attention implementation" in message
    assert "fused attention" not in message.lower()
