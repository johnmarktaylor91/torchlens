"""repeng bridge against the real package on a tiny cached Llama.

Each test compares ``tl.bridge.repeng.control_vector`` on saved activations with
``repeng.ControlVector.train`` run directly on the same model. Skips when repeng,
transformers, or the checkpoint is unavailable.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import torchlens as tl

repeng = pytest.importorskip("repeng")
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
    if tok.pad_token_id is None:
        tok.pad_token_id = 0  # repeng pads every batch; one prompt per batch never pads
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
        "dataset": [repeng.DatasetEntry(positive=p, negative=n) for p, n in pairs],
        "log_pos": trace([p for p, _ in pairs]),
        "log_neg": trace([n for _, n in pairs]),
    }
    try:
        yield stack
    finally:
        stack["log_pos"].cleanup()
        stack["log_neg"].cleanup()


def _assert_same(bridge: Any, direct: Any) -> None:
    """Assert two control vectors are identical, bit for bit."""

    assert isinstance(bridge, repeng.ControlVector)
    assert bridge.model_type == direct.model_type
    assert bridge.directions.keys() == direct.directions.keys()
    for layer, values in direct.directions.items():
        assert bridge.directions[layer].dtype == values.dtype
        assert np.array_equal(bridge.directions[layer], values), (
            layer,
            float(np.abs(bridge.directions[layer] - values).max()),
        )


@pytest.mark.parametrize("method", ["pca_diff", "pca_center"])
def test_control_vector_matches_train(stack: dict[str, Any], method: str) -> None:
    """Decoder layer 0 output gives exactly ControlVector.train(hidden_layers=[0])."""

    direct = repeng.ControlVector.train(
        stack["model"],
        stack["tok"],
        stack["dataset"],
        hidden_layers=[0],
        batch_size=1,
        method=method,
    )
    payload = tl.bridge.repeng.control_vector(
        stack["log_pos"],
        "model.layers.0",
        negative_log=stack["log_neg"],
        layer=0,
        method=method,
    )
    _assert_same(payload["control_vector"], direct)


def test_default_last_layer_reads_final_norm(stack: dict[str, Any]) -> None:
    """repeng's default layer list reads the final-norm output at the last layer."""

    direct = repeng.ControlVector.train(
        stack["model"], stack["tok"], stack["dataset"], batch_size=1
    )
    assert list(direct.directions) == [1]
    payload = tl.bridge.repeng.control_vector(
        stack["log_pos"], "model.norm", negative_log=stack["log_neg"], layer=1
    )
    _assert_same(payload["control_vector"], direct)


def test_unknown_method_is_refused(stack: dict[str, Any]) -> None:
    """An unknown method fails like repeng's own ValueError, before any vector."""

    with pytest.raises(ValueError, match="Unknown method"):
        tl.bridge.repeng.control_vector(
            stack["log_pos"],
            "model.layers.0",
            negative_log=stack["log_neg"],
            layer=0,
            method="pca",
        )


def test_layer_object_site_resolves_in_negative_log(stack: dict[str, Any]) -> None:
    """A Layer-object site reads the negative trace's layer, not the positive one."""

    direct = repeng.ControlVector.train(
        stack["model"], stack["tok"], stack["dataset"], hidden_layers=[0], batch_size=1
    )
    payload = tl.bridge.repeng.control_vector(
        stack["log_pos"],
        stack["log_pos"]["model.layers.0"],
        negative_log=stack["log_neg"],
        layer=0,
    )
    _assert_same(payload["control_vector"], direct)


def test_padded_unequal_prompts_match_batched_train(stack: dict[str, Any]) -> None:
    """Right-padded unequal prompts read the last real token, as repeng does."""

    tok = transformers.AutoTokenizer.from_pretrained(_TINY, padding_side="right")
    tok.pad_token_id = 0
    pairs = [(f"I love {w}", f"I really hate {w}") for w in _WORDS]
    texts = [text for pair in pairs for text in pair]
    longest = max(len(tok(text).input_ids) for text in texts)
    assert len({len(tok(text).input_ids) for text in texts}) > 1

    def trace(prompts: list[str]) -> Any:
        # Pad to repeng's one-batch length so every row sees the same tokens.
        enc = tok(prompts, return_tensors="pt", padding="max_length", max_length=longest)
        kwargs = {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}
        capture = tl.options.CaptureOptions(layers_to_save="all")
        return tl.trace(stack["model"], (), input_kwargs=kwargs, capture=capture)

    dataset = [repeng.DatasetEntry(positive=p, negative=n) for p, n in pairs]
    log_pos = trace([p for p, _ in pairs])
    log_neg = trace([n for _, n in pairs])
    try:
        direct = repeng.ControlVector.train(
            stack["model"], tok, dataset, hidden_layers=[0], batch_size=len(texts)
        )
        payload = tl.bridge.repeng.control_vector(
            log_pos, "model.layers.0", negative_log=log_neg, layer=0
        )
    finally:
        log_pos.cleanup()
        log_neg.cleanup()
    np.testing.assert_allclose(
        payload["control_vector"].directions[0], direct.directions[0], atol=1e-5, rtol=0
    )
