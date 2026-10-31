"""repeng converter against the real package on a tiny cached Llama.

A native ``fit_direction`` direction converted with
``Direction.to_repeng()`` must reproduce the native steer's logits when
``repeng.ControlModel`` applies it, for every method. Skips when repeng,
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
    skip_unless_hf_checkpoint_cached(_TINY)
    tok = transformers.AutoTokenizer.from_pretrained(_TINY)
    model = transformers.AutoModelForCausalLM.from_pretrained(_TINY).eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = 0  # repeng pads every batch; one prompt per batch never pads
    model.config.use_cache = False
    candidates = [(f"I love {w}", f"I hate {w}") for w in _WORDS]
    lengths = [(len(tok(p).input_ids), len(tok(n).input_ids)) for p, n in candidates]
    # One unpadded batch per side: keep the pairs at the most common equal length.
    length = Counter(a for a, b in lengths if a == b).most_common(1)[0][0]
    pairs = [pair for pair, (a, b) in zip(candidates, lengths, strict=True) if a == b == length]
    assert len(pairs) >= 3

    def trace(prompts: list[str]) -> Any:
        ids = tok(prompts, return_tensors="pt").input_ids
        return tl.trace(model, ids, capture=tl.options.CaptureOptions(layers_to_save="all"))

    stack = {
        "model": model,
        "tok": tok,
        "ids": tok([p for p, _ in pairs], return_tensors="pt").input_ids,
        "log_pos": trace([p for p, _ in pairs]),
        "log_neg": trace([n for _, n in pairs]),
    }
    try:
        yield stack
    finally:
        stack["log_pos"].cleanup()
        stack["log_neg"].cleanup()


_SITE = "model.layers.1"
_COEFF = 4.0


def _per_prompt_logits(call: Any, ids: torch.Tensor) -> torch.Tensor:
    """Run one prompt per forward and stack the logits.

    repeng 0.4's ``ControlModule`` builds its padding mask from the decoder
    layer's ``position_ids``, which this transformers passes as ``[1, seq]``,
    so a batch of several prompts fails its reshape; one prompt per forward
    is the case it supports.
    """

    with torch.no_grad():
        return torch.cat([call(ids[i : i + 1]).logits for i in range(ids.shape[0])])


@pytest.mark.parametrize("method", ["mean_diff", "pca", "logistic"])
def test_converted_control_vector_reproduces_native_steer_logits(
    stack: dict[str, Any], method: str
) -> None:
    """``ControlModel.set_control`` gives the native steer's logits for every method."""

    model, ids = stack["model"], stack["ids"]
    direction = fit_direction(stack["log_pos"], stack["log_neg"], _SITE, method)
    converted = direction.to_repeng()
    assert isinstance(converted, repeng.ControlVector)
    assert converted.model_type == "llama" and list(converted.directions) == [1]
    plain = _per_prompt_logits(model, ids)
    native = _per_prompt_logits(tl.when(tl.module(_SITE), direction.steer(_COEFF)).bind(model), ids)
    control = repeng.ControlModel(model, layer_ids=[1])
    try:
        control.set_control(converted, coeff=_COEFF)
        library = _per_prompt_logits(control, ids)
    finally:
        control.reset()
        control.unwrap()
    assert not torch.allclose(native, plain, atol=1e-5, rtol=0)
    torch.testing.assert_close(library, native, atol=1e-5, rtol=0)
    torch.testing.assert_close(_per_prompt_logits(model, ids), plain, atol=0, rtol=0)


def test_explicit_layer_and_model_type(stack: dict[str, Any]) -> None:
    """``layer=`` and ``model_type=`` are honoured; a site naming no layer needs ``layer=``."""

    direction = fit_direction(stack["log_pos"], stack["log_neg"], "model.norm")
    with pytest.raises(ValueError, match="decoder layer index"):
        direction.to_repeng()
    converted = direction.to_repeng(layer=1, model_type="custom")
    assert converted.model_type == "custom" and list(converted.directions) == [1]
