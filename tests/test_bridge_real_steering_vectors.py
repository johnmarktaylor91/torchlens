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
    try:
        tok = transformers.AutoTokenizer.from_pretrained(_TINY)
        model = transformers.AutoModelForCausalLM.from_pretrained(_TINY).eval()
    except OSError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"checkpoint {_TINY} not cached: {exc}")
    # With use_cache on, "model.layers.1" also names the layer's KV-cache outputs.
    model.config.use_cache = False
    candidates = [(f"I love {w}", f"I hate {w}") for w in _WORDS]
    lengths = [(len(tok(p).input_ids), len(tok(n).input_ids)) for p, n in candidates]
    # One unpadded batch per side: keep the pairs at the most common equal length.
    length = Counter(a for a, b in lengths if a == b).most_common(1)[0][0]
    pairs = [pair for pair, (a, b) in zip(candidates, lengths) if a == b == length]
    assert len(pairs) >= 3

    def trace(prompts: list[str]) -> Any:
        ids = tok(prompts, return_tensors="pt").input_ids
        return tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))

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
    """Train with the package directly, one prompt per forward pass."""

    return sv.train_steering_vector(
        stack["model"],
        stack["tok"],
        stack["pairs"],
        layers=[1],
        read_token_index=-1,
        batch_size=1,
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
