"""steering-vectors converter against the real package on a tiny cached Llama.

A native ``fit_direction`` direction converted with
``Direction.to_steering_vectors()`` must reproduce the native steer's logits
when the package applies it, for every method. The native methods are also
checked against the package's own aggregators on the same saved rows. Skips
when steering-vectors, transformers, or the checkpoint is unavailable.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from support.hf_cache import skip_unless_hf_checkpoint_cached

import torchlens as tl
from torchlens.intervention import fit_direction

sv = pytest.importorskip("steering_vectors")
transformers = pytest.importorskip("transformers")

pytestmark = [pytest.mark.optional]

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_WORDS = ["cats", "rain", "music", "coffee", "school", "summer", "dogs", "work"]


@pytest.fixture(scope="module")
def stack() -> Iterator[dict[str, Any]]:
    """Load the tiny Llama and trace equal-length positive and negative prompts."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    skip_unless_hf_checkpoint_cached(_TINY)
    tok = transformers.AutoTokenizer.from_pretrained(_TINY)
    # train_steering_vector sets this itself (fix_pad_token); set it first so
    # the traced and direct encodings come from one tokenizer state.
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(_TINY).eval()
    # With use_cache on, "model.layers.1" also names the layer's KV-cache outputs.
    model.config.use_cache = False
    candidates = [(f"I love {w}", f"I hate {w}") for w in _WORDS]
    lengths = [(len(tok(p).input_ids), len(tok(n).input_ids)) for p, n in candidates]
    # One unpadded batch per side: keep the pairs at the most common equal length.
    length = Counter(a for a, b in lengths if a == b).most_common(1)[0][0]
    pairs = [pair for pair, (a, b) in zip(candidates, lengths, strict=True) if a == b == length]
    assert len(pairs) >= 3

    def trace(prompts: list[str]) -> Any:
        # The exact call the package makes per side: same encoding, mask included.
        enc = tok(prompts, return_tensors="pt", padding=True)
        kwargs = {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
        capture = tl.options.CaptureOptions(layers_to_save="all")
        return tl.trace(model, (), input_kwargs=kwargs, capture=capture)

    stack = {
        "model": model,
        "tok": tok,
        "pairs": pairs,
        "log_pos": trace([p for p, _ in pairs]),
        "log_neg": trace([n for _, n in pairs]),
    }
    try:
        yield stack
    finally:
        stack["log_pos"].cleanup()
        stack["log_neg"].cleanup()


_SITE = "model.layers.1"
_MULTIPLIER = 4.0


def _direct(stack: dict[str, Any], **kwargs: Any) -> Any:
    """Train with the package directly, at the traced batch shape.

    The package runs one forward per side per batch; with every pair in one
    batch it runs exactly the two traced forwards (same prompts, encoding and
    mask), so bitwise equality does not depend on the runner's GEMM kernels.
    """

    return sv.train_steering_vector(
        stack["model"],
        stack["tok"],
        stack["pairs"],
        layers=[1],
        read_token_index=-1,
        batch_size=len(stack["pairs"]),
        **kwargs,
    )


def _ids(stack: dict[str, Any]) -> torch.Tensor:
    """Return the positive prompts' input ids (one unpadded batch)."""

    return stack["tok"]([p for p, _ in stack["pairs"]], return_tensors="pt").input_ids


def test_mean_diff_matches_train_steering_vector(stack: dict[str, Any]) -> None:
    """The default native method is bit-identical to the package's own trainer."""

    direct = _direct(stack).layer_activations[1]
    direction = fit_direction(stack["log_pos"], stack["log_neg"], _SITE)
    assert direction.vector.shape == direct.shape == (16,)
    assert torch.equal(direction.vector, direct)


def test_pca_matches_pca_aggregator(stack: dict[str, Any]) -> None:
    """Native ``pca`` agrees with the package's ``pca_aggregator`` on the same rows."""

    direct = _direct(stack, aggregator=sv.pca_aggregator()).layer_activations[1]
    direction = fit_direction(stack["log_pos"], stack["log_neg"], _SITE, "pca")
    torch.testing.assert_close(direction.vector, direct, atol=1e-5, rtol=0)


def test_logistic_matches_logistic_aggregator(stack: dict[str, Any]) -> None:
    """Native ``logistic`` agrees with the package's ``logistic_aggregator`` solved tightly."""

    aggregator = sv.logistic_aggregator(sklearn_kwargs={"tol": 1e-12, "max_iter": 100000})
    direct = _direct(stack, aggregator=aggregator).layer_activations[1]
    direction = fit_direction(stack["log_pos"], stack["log_neg"], _SITE, "logistic")
    torch.testing.assert_close(direction.vector, direct, atol=1e-5, rtol=0)


@pytest.mark.parametrize("method", ["mean_diff", "pca", "logistic"])
def test_converted_vector_reproduces_native_steer_logits(
    stack: dict[str, Any], method: str
) -> None:
    """``SteeringVector.apply`` gives the native steer's logits for every method."""

    model, ids = stack["model"], _ids(stack)
    direction = fit_direction(stack["log_pos"], stack["log_neg"], _SITE, method)
    converted = direction.to_steering_vectors()
    assert isinstance(converted, sv.SteeringVector)
    assert list(converted.layer_activations) == [1]
    with torch.no_grad():
        plain = model(ids).logits
        native = tl.when(tl.module(_SITE), direction.steer(_MULTIPLIER)).bind(model)(ids).logits
        with converted.apply(model, multiplier=_MULTIPLIER):
            library = model(ids).logits
    assert not torch.allclose(native, plain, atol=1e-5, rtol=0)
    torch.testing.assert_close(library, native, atol=1e-5, rtol=0)


def test_planted_wrong_site_fails_the_exact_comparison(stack: dict[str, Any]) -> None:
    """Reading the wrong layer breaks the bitwise match with the package's trainer."""

    direct = _direct(stack).layer_activations[1]
    direction = fit_direction(stack["log_pos"], stack["log_neg"], "model.layers.0")
    assert direction.vector.shape == direct.shape
    assert not torch.equal(direction.vector, direct)


def test_layer_object_site_resolves_in_each_trace(stack: dict[str, Any]) -> None:
    """A Layer-object site is re-resolved by label in the negative trace."""

    site = stack["log_pos"][_SITE]
    assert hasattr(site, "out") and hasattr(site, "layer_label")
    direct = _direct(stack).layer_activations[1]
    direction = fit_direction(stack["log_pos"], stack["log_neg"], site)
    assert torch.equal(direction.vector, direct)


def test_padded_unequal_prompts_match_batched_package(stack: dict[str, Any]) -> None:
    """Right-padded batches read the last real token, as the package does."""

    tok = transformers.AutoTokenizer.from_pretrained(_TINY, padding_side="right")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pairs = [(f"I love {w}", f"I really hate {w}") for w in _WORDS]
    lengths = {len(tok(text).input_ids) for pair in pairs for text in pair}
    assert len(lengths) > 1

    def trace(prompts: list[str]) -> Any:
        enc = tok(prompts, return_tensors="pt", padding=True)
        assert bool((enc.attention_mask == 0).any())
        kwargs = {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
        capture = tl.options.CaptureOptions(layers_to_save="all")
        return tl.trace(stack["model"], (), input_kwargs=kwargs, capture=capture)

    log_pos = trace([p for p, _ in pairs])
    log_neg = trace([n for _, n in pairs])
    try:
        direct = sv.train_steering_vector(
            stack["model"], tok, pairs, layers=[1], read_token_index=-1, batch_size=len(pairs)
        ).layer_activations[1]
        direction = fit_direction(log_pos, log_neg, _SITE)
    finally:
        log_pos.cleanup()
        log_neg.cleanup()
    torch.testing.assert_close(direction.vector, direct, atol=1e-5, rtol=0)
