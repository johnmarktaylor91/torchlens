"""A real protein language model checkpoint: ESM-2 8M, real weights, capture, validation.

The biology row of the real-model suite. The test first proves the registry checkpoint's
REAL weights loaded: an exact SHA-256 of the token embedding, then the checkpoint's known
predictions on human ubiquitin (the native residue ranks first at 72 of 76 positions, and a
masked start position is filled with methionine). A config-built random-weight twin fails
both. Only then does it check that the six attention softmax ops are captured and equal the
model's returned attentions, that the traced output equals the direct forward, and that
forward replay validation passes.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

# Measured in the pinned CPU venv (torch 2.13.0, transformers 5.18.0): load 3 s, capture 3 s,
# forward validation 11 s; slow-tier with the other real checkpoints.
pytestmark = [pytest.mark.slow, pytest.mark.real_model, pytest.mark.real_checkpoint]

ARTIFACT_ID = "r1-esm2-t6-8m-ur50d"
EMBEDDINGS_SHA256 = (
    "7ea1245a2e2a99b85bb4437ef244430fe1140a15411784c32687aeb1a8991f69"  # pragma: allowlist secret
)
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
NATIVE_TOP1 = 72
NUM_LAYERS, NUM_HEADS = 6, 20


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def test_esm2_8m_real_weights_attention_capture_and_validate(r1_loader):
    import torchlens as tl

    loaded = r1_loader(ARTIFACT_ID)
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    embeddings = model.get_parameter("esm.embeddings.word_embeddings.weight")
    assert _tensor_sha256(embeddings) == EMBEDDINGS_SHA256, "not the pinned real weights"

    encoded = tokenizer(UBIQUITIN, return_tensors="pt")
    with torch.no_grad():
        direct = model(**encoded, output_attentions=True)
    native = encoded["input_ids"][0, 1:-1]
    assert int((direct.logits[0, 1:-1].argmax(-1) == native).sum()) == NATIVE_TOP1

    masked = encoded["input_ids"].clone()
    masked[0, 1] = tokenizer.mask_token_id
    with torch.no_grad():
        filled = model(input_ids=masked, attention_mask=encoded["attention_mask"]).logits
    assert tokenizer.convert_ids_to_tokens(int(filled[0, 1].argmax())) == "M"

    trace = tl.trace(model, (), input_kwargs=dict(encoded))
    try:
        assert trace.outcome.status.name == "COMPLETE"
        maps = [site.out for site in trace.find_sites(tl.func("softmax"))]
        assert len(maps) == NUM_LAYERS
        tokens = encoded["input_ids"].shape[1]
        for captured, returned in zip(maps, direct.attentions, strict=True):
            assert tuple(captured.shape) == (1, NUM_HEADS, tokens, tokens)
            assert torch.allclose(captured, returned)
        assert any(
            isinstance(op.out, torch.Tensor)
            and op.out.shape == direct.logits.shape
            and torch.equal(op.out, direct.logits)
            for op in trace.output_ops
        ), "traced output != direct forward on the real checkpoint"
    finally:
        trace.cleanup()
    assert tl.validate(model, (), dict(encoded), scope="forward") is True
