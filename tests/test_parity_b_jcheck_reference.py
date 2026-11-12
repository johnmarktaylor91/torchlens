"""The JAX ecosystem rows match the PyTorch Transformers models of the same revision (G3).

The independent reference: each pinned repository revision run through its
PyTorch Transformers model (``BertModel``, ``ViTModel``) with the repository's
own PyTorch weights, tokenizer and image processor. The JAX rows share none of
that code: their architecture and weight mapping are
``tests/conformance/_jax_checkpoints.py``. Runs only under
``TORCHLENS_CONFORMANCE_ECOSYSTEM=1`` (checkpoint download).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

jax = pytest.importorskip("jax")
transformers = pytest.importorskip("transformers")

import torch  # noqa: E402

from tests.conformance import _jax_checkpoints as checkpoints  # noqa: E402

pytestmark = [
    pytest.mark.backend_jax,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("TORCHLENS_CONFORMANCE_ECOSYSTEM") != "1",
        reason="ecosystem rows need TORCHLENS_CONFORMANCE_ECOSYSTEM=1 (checkpoint download)",
    ),
    pytest.mark.rare,
]

#: Float32 agreement bound between two independent implementations of the same
#: weights (different kernels, summation orders and LayerNorm/GELU code).
_ATOL = 1e-4
_RTOL = 1e-4


def _reference_model(pin: checkpoints.FlaxCheckpoint, cls: Any) -> Any:
    """The Transformers model at the pinned revision, its weights file digest checked."""
    weights = checkpoints.fetch(pin, pin.reference_filename)
    assert checkpoints.sha256_of(weights) == pin.reference_sha256
    model = cls.from_pretrained(pin.repo, revision=pin.revision)
    return model.eval()


def _assert_outputs_match(jax_outputs: Any, torch_outputs: Any) -> None:
    """Both the sequence output and the pooled output agree within the bound."""
    for name, mine, theirs in zip(
        ("last_hidden_state", "pooler_output"),
        jax_outputs,
        (torch_outputs.last_hidden_state, torch_outputs.pooler_output),
        strict=True,
    ):
        mine_np = np.asarray(mine)
        theirs_np = theirs.detach().numpy()
        assert mine_np.shape == theirs_np.shape, name
        gap = float(np.max(np.abs(mine_np - theirs_np)))
        np.testing.assert_allclose(
            mine_np, theirs_np, atol=_ATOL, rtol=_RTOL, err_msg=f"{name} max abs gap {gap:.3g}"
        )


def test_tokenizer_reproduces_the_pinned_ids() -> None:
    """The repository's tokenizer at the pinned revision gives the row's ids."""
    pin = checkpoints.BERT_L2_H256
    tokenizer = transformers.BertTokenizer.from_pretrained(pin.repo, revision=pin.revision)
    ids = tokenizer(checkpoints.BERT_PROMPT)["input_ids"]
    assert tuple(ids) == checkpoints.BERT_INPUT_IDS


def test_image_processor_reproduces_the_pinned_pixels() -> None:
    """The repository's image processor gives the row's pixels (NCHW there, NHWC here)."""
    from PIL import Image

    pin = checkpoints.VIT_B16_IN21K
    processor = transformers.AutoImageProcessor.from_pretrained(pin.repo, revision=pin.revision)
    image = Image.fromarray(checkpoints.vit_image_bytes())
    pixels = np.asarray(processor(images=image, return_tensors="np")["pixel_values"])
    np.testing.assert_allclose(
        pixels.transpose(0, 2, 3, 1), checkpoints.vit_pixels(), atol=1e-6, rtol=0
    )


def test_bert_matches_transformers() -> None:
    """The JAX BERT row's outputs equal ``BertModel``'s on the same ids."""
    model, ids = checkpoints.build_bert()
    reference = _reference_model(checkpoints.BERT_L2_H256, transformers.BertModel)
    tokens = torch.tensor([checkpoints.BERT_INPUT_IDS])
    with torch.no_grad():
        expected = reference(input_ids=tokens, attention_mask=torch.ones_like(tokens))
    _assert_outputs_match(model(ids), expected)


def test_vit_matches_transformers() -> None:
    """The JAX ViT row's outputs equal ``ViTModel``'s on the same pixels."""
    model, pixels = checkpoints.build_vit()
    reference = _reference_model(checkpoints.VIT_B16_IN21K, transformers.ViTModel)
    nchw = torch.from_numpy(np.ascontiguousarray(checkpoints.vit_pixels().transpose(0, 3, 1, 2)))
    with torch.no_grad():
        expected = reference(pixel_values=nchw)
    _assert_outputs_match(model(pixels), expected)
