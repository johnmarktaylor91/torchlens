"""steering-vectors bridge against the real package on a tiny cached Llama.

Each test compares ``tl.bridge.steering_vectors.vector`` on saved activations
with ``steering_vectors.train_steering_vector`` run directly on the same model.
Skips when steering-vectors, transformers, or the checkpoint is unavailable.
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


def test_default_trainer_matches_train_steering_vector(stack: dict[str, Any]) -> None:
    """The documented default path is bit-identical to the package's own trainer."""

    direct = _direct(stack).layer_activations[1]
    payload = tl.bridge.steering_vectors.vector(
        stack["log_pos"], "model.layers.1", negative_log=stack["log_neg"]
    )
    assert payload["vector"].shape == direct.shape == (16,)
    assert torch.equal(payload["vector"], direct)
    assert payload["steering_vector"] is None


def test_layer_returns_real_steering_vector(stack: dict[str, Any]) -> None:
    """``layer=`` wraps the result in a steering_vectors.SteeringVector."""

    direct = _direct(stack)
    payload = tl.bridge.steering_vectors.vector(
        stack["log_pos"], "model.layers.1", negative_log=stack["log_neg"], layer=1
    )
    result = payload["steering_vector"]
    assert isinstance(result, sv.SteeringVector)
    assert result.layer_type == direct.layer_type
    assert result.layer_activations.keys() == direct.layer_activations.keys()
    assert torch.equal(result.layer_activations[1], direct.layer_activations[1])


def test_pca_aggregator_matches_package(stack: dict[str, Any]) -> None:
    """A package aggregator passed as trainer matches the direct run with it."""

    direct = _direct(stack, aggregator=sv.pca_aggregator()).layer_activations[1]
    payload = tl.bridge.steering_vectors.vector(
        stack["log_pos"],
        "model.layers.1",
        negative_log=stack["log_neg"],
        trainer=sv.pca_aggregator(),
    )
    assert torch.equal(payload["vector"], direct)


def test_missing_negative_is_refused(stack: dict[str, Any]) -> None:
    """Without a negative site or trace the bridge refuses with a remedy."""

    with pytest.raises(ValueError, match="negative_log="):
        tl.bridge.steering_vectors.vector(stack["log_pos"], "model.layers.1")


def test_layer_object_site_resolves_in_negative_log(stack: dict[str, Any]) -> None:
    """A Layer-object site is re-resolved in negative_log, not reused from log."""

    site = stack["log_pos"]["model.layers.1"]
    assert hasattr(site, "out") and hasattr(site, "layer_label")
    direct = _direct(stack).layer_activations[1]
    payload = tl.bridge.steering_vectors.vector(
        stack["log_pos"], site, negative_log=stack["log_neg"]
    )
    assert torch.equal(payload["vector"], direct)
    assert not torch.equal(payload["positive"], payload["negative"])


@pytest.mark.parametrize(
    ("positive_site", "negative_site"),
    [("model.layers.0", None), ("model.layers.1", "model.layers.0")],
    ids=["wrong_positive_site", "wrong_negative_site"],
)
def test_planted_wrong_site_fails_the_exact_comparison(
    stack: dict[str, Any], positive_site: str, negative_site: str | None
) -> None:
    """Reading the wrong layer on either side breaks the bitwise match above."""

    direct = _direct(stack).layer_activations[1]
    payload = tl.bridge.steering_vectors.vector(
        stack["log_pos"], positive_site, negative_site, negative_log=stack["log_neg"]
    )
    assert payload["vector"].shape == direct.shape
    assert not torch.equal(payload["vector"], direct)


def test_identical_rows_are_refused(stack: dict[str, Any]) -> None:
    """Identical positive and negative rows (a zero vector) are refused."""

    with pytest.raises(ValueError, match="identical"):
        tl.bridge.steering_vectors.vector(
            stack["log_pos"], "model.layers.1", negative_log=stack["log_pos"]
        )


def test_padded_unequal_prompts_match_batched_package(stack: dict[str, Any]) -> None:
    """Right-padded batches read the last real token, as the package does."""

    tok = transformers.AutoTokenizer.from_pretrained(_TINY, padding_side="right")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pairs = [(f"I love {w}", f"I really hate {w}") for w in _WORDS]
    lengths = {len(tok(text).input_ids) for pair in pairs for text in pair}
    assert len(lengths) > 1

    def trace(prompts: list[str]) -> tuple[Any, torch.Tensor]:
        enc = tok(prompts, return_tensors="pt", padding=True)
        assert bool((enc.attention_mask == 0).any())
        kwargs = {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
        capture = tl.options.CaptureOptions(layers_to_save="all")
        return tl.trace(stack["model"], (), input_kwargs=kwargs, capture=capture), (
            enc.attention_mask
        )

    log_pos, mask_pos = trace([p for p, _ in pairs])
    log_neg, mask_neg = trace([n for _, n in pairs])
    try:
        direct = sv.train_steering_vector(
            stack["model"], tok, pairs, layers=[1], read_token_index=-1, batch_size=len(pairs)
        ).layer_activations[1]
        captured = tl.bridge.steering_vectors.vector(
            log_pos, "model.layers.1", negative_log=log_neg
        )
        explicit = tl.bridge.steering_vectors.vector(
            log_pos,
            "model.layers.1",
            negative_log=log_neg,
            attention_mask=mask_pos,
            negative_attention_mask=mask_neg,
        )
        unmasked = tl.bridge.steering_vectors.vector(
            log_pos,
            "model.layers.1",
            negative_log=log_neg,
            attention_mask=torch.ones_like(mask_pos),
            negative_attention_mask=torch.ones_like(mask_neg),
        )
    finally:
        log_pos.cleanup()
        log_neg.cleanup()
    torch.testing.assert_close(captured["vector"], direct, atol=1e-5, rtol=0)
    assert torch.equal(captured["vector"], explicit["vector"])
    # Reading position -1 of the padded batch picks pad tokens and drifts away.
    assert not torch.allclose(unmasked["vector"], direct, atol=1e-5, rtol=0)
