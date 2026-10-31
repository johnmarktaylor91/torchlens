"""dialz converter against the real package on a tiny cached Llama.

A native ``direction_from_pairs`` direction converted with
``Direction.to_dialz()`` must reproduce the native steer's logits when
``dialz.SteeringModel`` applies it, for every method. Skips when dialz,
transformers, or the checkpoint is unavailable.
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

dialz = pytest.importorskip("dialz")
transformers = pytest.importorskip("transformers")

pytestmark = [pytest.mark.optional]

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_WORDS = ["cats", "rain", "music", "coffee", "school", "summer", "dogs", "work"]
_SITE = "model.layers.1"
_COEFF = 4.0


@pytest.fixture(scope="module")
def stack() -> Iterator[dict[str, Any]]:
    """Wrap the tiny Llama in a SteeringModel and trace equal-length prompts."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    skip_unless_hf_checkpoint_cached(_TINY)
    # The tokenizer SteeringVector.train builds: pad id 0, default padding side.
    tok = transformers.AutoTokenizer.from_pretrained(_TINY)
    tok.pad_token_id = 0
    # SteeringModel loads its own float16 copy and wraps decoder layer 1 in
    # place; every run below (trace, native steer, library steer) uses that
    # one wrapped model, cast to float32.
    steering_model = dialz.SteeringModel(_TINY, layer_ids=[1])
    model = steering_model.model.to(device="cpu", dtype=torch.float32).eval()
    model.config.use_cache = False
    candidates = [(f"I love {w}", f"I hate {w}") for w in _WORDS]
    lengths = [(len(tok(p).input_ids), len(tok(n).input_ids)) for p, n in candidates]
    # One unpadded batch per side: keep the pairs at the most common equal length.
    length = Counter(a for a, b in lengths if a == b).most_common(1)[0][0]
    pairs = [pair for pair, (a, b) in zip(candidates, lengths, strict=True) if a == b == length]
    assert len(pairs) >= 3
    dataset = dialz.Dataset()
    for positive, negative in pairs:
        dataset.add_entry(positive, negative)

    def trace(prompts: list[str]) -> Any:
        ids = tok(prompts, return_tensors="pt").input_ids
        return tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))

    stack = {
        "steering_model": steering_model,
        "model": model,
        "dataset": dataset,
        "ids": tok([p for p, _ in pairs], return_tensors="pt").input_ids,
        "log_pos": trace([p for p, _ in pairs]),
        "log_neg": trace([n for _, n in pairs]),
    }
    try:
        yield stack
    finally:
        stack["log_pos"].cleanup()
        stack["log_neg"].cleanup()


def _per_prompt_logits(call: Any, ids: torch.Tensor) -> torch.Tensor:
    """Run one prompt per forward and stack the logits.

    dialz 1.1's ``SteeringModule`` builds its padding mask from the decoder
    layer's ``position_ids``, which this transformers passes as ``[1, seq]``,
    so a steered batch of several prompts fails its reshape; one prompt per
    forward is the case it supports.
    """

    with torch.no_grad():
        return torch.cat([call(ids[i : i + 1]).logits for i in range(ids.shape[0])])


@pytest.mark.parametrize("method", ["mean_diff", "pca", "logistic"])
def test_converted_steering_vector_reproduces_native_steer_logits(
    stack: dict[str, Any], method: str
) -> None:
    """``SteeringModel.set_control`` gives the native steer's logits for every method."""

    model, ids, steering_model = stack["model"], stack["ids"], stack["steering_model"]
    direction = fit_direction(stack["log_pos"], stack["log_neg"], _SITE, method)
    converted = direction.to_dialz()
    assert isinstance(converted, dialz.SteeringVector)
    assert converted.model_type == "llama" and list(converted.directions) == [1]
    plain = _per_prompt_logits(model, ids)
    native = _per_prompt_logits(tl.when(tl.module(_SITE), direction.steer(_COEFF)).bind(model), ids)
    steering_model.set_control(converted, scalar=_COEFF)
    try:
        library = _per_prompt_logits(steering_model, ids)
    finally:
        steering_model.reset()
    assert not torch.allclose(native, plain, atol=1e-5, rtol=0)
    torch.testing.assert_close(library, native, atol=1e-5, rtol=0)
    torch.testing.assert_close(_per_prompt_logits(model, ids), plain, atol=0, rtol=0)


def test_native_mean_diff_matches_steering_vector_train(stack: dict[str, Any]) -> None:
    """Native ``mean_diff`` equals dialz's own ``mean_diff`` training on the same prompts.

    dialz layer ``i`` reads ``hidden_states[i + 1]``, the output of
    ``model.layers.<i>`` for every layer but the last, so layer 0 pairs with
    site ``"model.layers.0"``.
    """

    direct = dialz.SteeringVector.train(
        stack["steering_model"], stack["dataset"], method="mean_diff", hidden_layers=[0]
    )
    direction = fit_direction(stack["log_pos"], stack["log_neg"], "model.layers.0")
    converted = direction.to_dialz()
    assert list(direct.directions) == list(converted.directions) == [0]
    assert converted.directions[0].dtype == direct.directions[0].dtype
    torch.testing.assert_close(
        torch.from_numpy(converted.directions[0]),
        torch.from_numpy(direct.directions[0]),
        atol=1e-5,
        rtol=0,
    )


def test_explicit_layer_and_model_type(stack: dict[str, Any]) -> None:
    """``layer=`` and ``model_type=`` are honoured; a site naming no layer needs ``layer=``."""

    direction = fit_direction(stack["log_pos"], stack["log_neg"], "model.norm")
    with pytest.raises(ValueError, match="decoder layer index"):
        direction.to_dialz()
    converted = direction.to_dialz(layer=1, model_type="custom")
    assert converted.model_type == "custom" and list(converted.directions) == [1]
