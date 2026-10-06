"""repeng bridge against the real package on a tiny cached Llama.

Each test compares ``tl.bridge.repeng.control_vector`` on saved activations with
``repeng.ControlVector.train`` run directly on the same model. Skips when repeng,
transformers, or the checkpoint is unavailable.
"""

from __future__ import annotations

import os
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
def stack() -> dict[str, Any]:
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
    pairs = [(f"I love {w}", f"I hate {w}") for w in _WORDS]
    length = len(tok(pairs[0][0]).input_ids)
    pairs = [
        (p, n)
        for p, n in pairs
        if len(tok(p).input_ids) == length and len(tok(n).input_ids) == length
    ]
    assert len(pairs) >= 3

    def trace(prompts: list[str]) -> Any:
        ids = tok(prompts, return_tensors="pt").input_ids
        return tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))

    return {
        "model": model,
        "tok": tok,
        "dataset": [repeng.DatasetEntry(positive=p, negative=n) for p, n in pairs],
        "log_pos": trace([p for p, _ in pairs]),
        "log_neg": trace([n for _, n in pairs]),
    }


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
