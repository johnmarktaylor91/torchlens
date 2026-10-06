"""dialz bridge against the real package (0.2 and 1.x) on a tiny cached Llama.

Each test compares ``tl.bridge.dialz.vector`` on saved activations with
``dialz.SteeringVector.train`` run directly on the same ``SteeringModel``. Skips
when dialz, transformers, or the checkpoint is unavailable.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import torchlens as tl

dialz = pytest.importorskip("dialz")
transformers = pytest.importorskip("transformers")

pytestmark = [pytest.mark.optional]

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_WORDS = ["cats", "rain", "music", "coffee", "school", "summer", "dogs", "work"]


@pytest.fixture(scope="module")
def stack() -> Iterator[dict[str, Any]]:
    """Build a dialz SteeringModel and trace equal-length prompts through it."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(_TINY)
        steering_model = dialz.SteeringModel(_TINY, layer_ids=[1])
    except OSError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"checkpoint {_TINY} not cached: {exc}")
    model = steering_model.model.eval()
    model.config.use_cache = False
    candidates = [(f"I love {w}", f"I hate {w}") for w in _WORDS]
    lengths = [(len(tok(p).input_ids), len(tok(n).input_ids)) for p, n in candidates]
    # One unpadded batch per side: keep the pairs at the most common equal length.
    length = Counter(a for a, b in lengths if a == b).most_common(1)[0][0]
    pairs = [pair for pair, (a, b) in zip(candidates, lengths) if a == b == length]
    assert len(pairs) >= 3
    dataset = dialz.Dataset()
    for positive, negative in pairs:
        dataset.add_entry(positive, negative)

    def trace(prompts: list[str]) -> Any:
        ids = tok(prompts, return_tensors="pt").input_ids.to(model.device)
        return tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))

    stack = {
        "steering_model": steering_model,
        "dataset": dataset,
        "log_pos": trace([p for p, _ in pairs]),
        "log_neg": trace([n for _, n in pairs]),
    }
    try:
        yield stack
    finally:
        stack["log_pos"].cleanup()
        stack["log_neg"].cleanup()


def _assert_same(bridge: Any, direct: Any) -> None:
    """Assert two dialz steering vectors are identical, bit for bit."""

    assert isinstance(bridge, dialz.SteeringVector)
    assert bridge.model_type == direct.model_type
    assert bridge.directions.keys() == direct.directions.keys()
    for layer, values in direct.directions.items():
        assert bridge.directions[layer].dtype == values.dtype
        assert np.array_equal(bridge.directions[layer], values), (
            layer,
            float(np.abs(bridge.directions[layer] - values).max()),
        )
    assert bridge == direct


@pytest.mark.parametrize("method", [None, "pca_center", "mean_diff"])
def test_vector_matches_steering_vector_train(stack: dict[str, Any], method: str | None) -> None:
    """Decoder layer 0 output gives exactly SteeringVector.train(hidden_layers=[0])."""

    kwargs = {} if method is None else {"method": method}
    direct = dialz.SteeringVector.train(
        stack["steering_model"], stack["dataset"], hidden_layers=[0], batch_size=1, **kwargs
    )
    payload = tl.bridge.dialz.vector(
        stack["log_pos"], "model.layers.0", negative_log=stack["log_neg"], layer=0, method=method
    )
    _assert_same(payload["steering_vector"], direct)


def test_default_last_layer_reads_final_norm(stack: dict[str, Any]) -> None:
    """dialz's default layer list reads the final-norm output at the last layer."""

    direct = dialz.SteeringVector.train(stack["steering_model"], stack["dataset"], batch_size=1)
    assert list(direct.directions) == [1]
    payload = tl.bridge.dialz.vector(
        stack["log_pos"], "model.norm", negative_log=stack["log_neg"], layer=1
    )
    _assert_same(payload["steering_vector"], direct)


def test_layer_object_site_resolves_in_negative_log(stack: dict[str, Any]) -> None:
    """A Layer-object site reads the negative trace's layer, not the positive one."""

    direct = dialz.SteeringVector.train(
        stack["steering_model"], stack["dataset"], hidden_layers=[0], batch_size=1
    )
    # dialz wraps each decoder layer; its output op is one pass of a two-pass layer.
    site = stack["log_pos"]["model.layers.0"]
    payload = tl.bridge.dialz.vector(
        stack["log_pos"],
        site,
        negative_log=stack["log_neg"],
        layer=0,
    )
    _assert_same(payload["steering_vector"], direct)


def test_padded_unequal_prompts_match_batched_train(stack: dict[str, Any]) -> None:
    """Padded unequal prompts read the last real token, as dialz does."""

    # dialz loads its own tokenizer with pad_token_id = 0 and the default side.
    tok = transformers.AutoTokenizer.from_pretrained(_TINY)
    tok.pad_token_id = 0
    model = stack["steering_model"].model
    pairs = [(f"I love {w}", f"I really hate {w}") for w in _WORDS]
    texts = [text for pair in pairs for text in pair]
    longest = max(len(tok(text).input_ids) for text in texts)
    assert len({len(tok(text).input_ids) for text in texts}) > 1
    dataset = dialz.Dataset()
    for positive, negative in pairs:
        dataset.add_entry(positive, negative)

    def trace(prompts: list[str]) -> Any:
        enc = tok(prompts, return_tensors="pt", padding="max_length", max_length=longest)
        kwargs = {
            "input_ids": enc.input_ids.to(model.device),
            "attention_mask": enc.attention_mask.to(model.device),
        }
        capture = tl.options.CaptureOptions(layers_to_save="all")
        return tl.trace(model, (), input_kwargs=kwargs, capture=capture)

    log_pos = trace([p for p, _ in pairs])
    log_neg = trace([n for _, n in pairs])
    try:
        direct = dialz.SteeringVector.train(
            stack["steering_model"], dataset, hidden_layers=[0], batch_size=len(texts)
        )
        payload = tl.bridge.dialz.vector(log_pos, "model.layers.0", negative_log=log_neg, layer=0)
    finally:
        log_pos.cleanup()
        log_neg.cleanup()
    np.testing.assert_allclose(
        payload["steering_vector"].directions[0], direct.directions[0], atol=5e-3, rtol=0
    )
