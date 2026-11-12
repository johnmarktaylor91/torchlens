"""Per-head activation patching when the baselines are traced with forward keyword inputs.

``trace_kwargs={"input_kwargs": {"use_cache": False}}`` is how a Hugging Face LM is
usually traced. The patched reruns used to drop those keyword inputs: the forward took
the cache path, diverged from the corrupted graph (``ControlFlowDivergenceWarning``),
the head hook resolved to no site and ``activation_patch_attention_heads`` raised
``PatchApplicationError`` on every GPT-2 and Qwen2. Every patched rerun now reuses the
baselines' keyword inputs. Tiny configs keep the cache ON so the dropped keyword would
change the path; random weights, no download.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp  # noqa: F401  (registers tl.mechinterp)
from torchlens.errors import ControlFlowDivergenceWarning

transformers = pytest.importorskip("transformers")

# Each test runs one or two full patching campaigns (2 baselines + 8 reruns): 5-20 s.
pytestmark = pytest.mark.heavy

_NO_CACHE = {"input_kwargs": {"use_cache": False}}
_VOCAB = 64


def _gpt2(use_cache: bool = True) -> Any:
    config = transformers.GPT2Config(
        n_layer=2,
        n_head=4,
        n_embd=32,
        vocab_size=_VOCAB,
        n_positions=16,
        use_cache=use_cache,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(config).eval()


def _qwen2(use_cache: bool = True) -> Any:
    config = transformers.Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_cache=use_cache,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(config).eval()


def _gpt2_out_projections(model: Any) -> list[Any]:
    return [block.attn.c_proj for block in model.transformer.h]


def _qwen2_out_projections(model: Any) -> list[Any]:
    return [layer.self_attn.o_proj for layer in model.model.layers]


_MODELS = [
    pytest.param(_gpt2, _gpt2_out_projections, id="gpt2"),
    pytest.param(_qwen2, _qwen2_out_projections, id="qwen2"),
]


def _prompts() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randint(0, _VOCAB, (1, 5), generator=generator),
        torch.randint(0, _VOCAB, (1, 5), generator=generator),
    )


def _logit_diff(logits: torch.Tensor) -> torch.Tensor:
    return logits[0, -1, 3] - logits[0, -1, 7]


def _metric(trace: Any) -> torch.Tensor:
    return _logit_diff(trace[trace.output_layers[0]].out)


def _head_oracle(
    model: Any, projections: list[Any], clean: torch.Tensor, corrupt: torch.Tensor
) -> torch.Tensor:
    """Plain torch: swap one head's slice of each output projection's input for the clean one."""

    n_head = 4
    values = torch.empty(len(projections), n_head)
    for layer, proj in enumerate(projections):
        captured: dict[str, torch.Tensor] = {}
        grab = proj.register_forward_pre_hook(
            lambda _m, args, captured=captured: captured.setdefault(
                "clean", args[0].detach().clone()
            )
        )
        with torch.no_grad():
            model(clean, use_cache=False)
        grab.remove()
        d_head = captured["clean"].shape[-1] // n_head
        for head in range(n_head):
            cols = slice(head * d_head, (head + 1) * d_head)

            def _swap(
                _m: Any,
                args: tuple[Any, ...],
                cols: slice = cols,
                captured: dict[str, torch.Tensor] = captured,
            ) -> tuple[Any, ...]:
                patched = args[0].clone()
                patched[..., cols] = captured["clean"][..., cols]
                return (patched, *args[1:])

            swap = proj.register_forward_pre_hook(_swap)
            try:
                with torch.no_grad():
                    logits = model(corrupt, use_cache=False).logits
            finally:
                swap.remove()
            values[layer, head] = _logit_diff(logits)
    return values


def _without_divergence(run: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Run ``run`` and fail if any rerun diverged from the captured graph."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run()
    diverged = [w for w in caught if issubclass(w.category, ControlFlowDivergenceWarning)]
    assert not diverged, [str(w.message)[:200] for w in diverged]
    return result


@pytest.mark.parametrize(("build", "projections"), _MODELS)
def test_head_patching_with_input_kwargs_matches_the_plain_torch_oracle(
    build: Callable[..., Any], projections: Callable[[Any], list[Any]]
) -> None:
    model = build()
    clean, corrupt = _prompts()
    scores = _without_divergence(
        lambda: tl.facets.patching.activation_patch_attention_heads(
            model, clean, corrupt, _metric, trace_kwargs=_NO_CACHE
        )
    )
    oracle = _head_oracle(model, projections(model), clean, corrupt)
    assert scores.shape == (2, 4)
    torch.testing.assert_close(scores, oracle, atol=1e-5, rtol=0)
    with torch.no_grad():
        corrupted = _logit_diff(model(corrupt, use_cache=False).logits)
    # Patching a head moves the metric off the corrupted baseline, by head-specific amounts.
    assert (scores - corrupted).abs().max() > 1e-4
    assert scores.unique().numel() > 1


@pytest.mark.parametrize(("build", "projections"), _MODELS)
def test_input_kwargs_and_a_cache_off_config_give_the_same_grid(
    build: Callable[..., Any], projections: Callable[[Any], list[Any]]
) -> None:
    del projections
    clean, corrupt = _prompts()
    via_kwargs = tl.facets.patching.activation_patch_attention_heads(
        build(True), clean, corrupt, _metric, trace_kwargs=_NO_CACHE
    )
    via_config = tl.facets.patching.activation_patch_attention_heads(
        build(False), clean, corrupt, _metric
    )
    torch.testing.assert_close(via_kwargs, via_config, atol=0, rtol=0)


def test_attention_output_helper_carries_input_kwargs_on_gpt2() -> None:
    clean, corrupt = _prompts()
    patched = _without_divergence(
        lambda: tl.facets.patching.activation_patch_attention_output(
            _gpt2(), clean, corrupt, _metric, trace_kwargs=_NO_CACHE
        )
    )
    reference = tl.facets.patching.activation_patch_attention_output(
        _gpt2(use_cache=False), clean, corrupt, _metric
    )
    torch.testing.assert_close(patched, reference, atol=0, rtol=0)


def test_residual_stream_helper_carries_input_kwargs_on_gpt2() -> None:
    clean, corrupt = _prompts()
    patched = _without_divergence(
        lambda: tl.facets.patching.activation_patch_residual_stream(
            _gpt2(), clean, corrupt, _metric, patch_positions=False, trace_kwargs=_NO_CACHE
        )
    )
    reference = tl.facets.patching.activation_patch_residual_stream(
        _gpt2(use_cache=False), clean, corrupt, _metric, patch_positions=False
    )
    torch.testing.assert_close(patched, reference, atol=0, rtol=0)


def test_mechinterp_head_grid_carries_input_kwargs_on_gpt2() -> None:
    model = _gpt2()
    clean, corrupt = _prompts()
    grid = _without_divergence(
        lambda: (
            tl.mechinterp.patch_heads_grid(
                model, clean, corrupt, _metric, trace_kwargs=_NO_CACHE
            ).values
        )
    )
    oracle = _head_oracle(model, _gpt2_out_projections(model), clean, corrupt)
    torch.testing.assert_close(grid, oracle.to(grid.dtype), atol=1e-5, rtol=0)
