"""F06 B8 real-checkpoint rows for text() (attrib memo section 4).

The gpt2 Eiffel headline (residual < 1% asserted end to end), the distilgpt2
fast row, and the bert [MASK] callable-target row on the auto pad+scaffold
baseline -- the only measured converging encoder baseline; NO residual bound
is asserted for the zeros baseline on bert because none exists at any
measured grid. All rows run offline against the cached checkpoints.
"""

from __future__ import annotations

import pytest
import torch

import torchlens.attribution as attribution

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def gpt2() -> tuple[object, object]:
    """Cached gpt2 + tokenizer."""

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return model, tokenizer


@pytest.mark.heavy
def test_gpt2_eiffel_headline_row(gpt2: tuple[object, object]) -> None:
    """The funded two-liner: residual < 1% end to end on real gpt2."""

    model, tokenizer = gpt2
    result = attribution.text(
        model, tokenizer, "The Eiffel Tower is located in the city of", target=" Paris"
    )
    assert result.completeness["residual_rel"] < 0.01
    assert result.baseline["policy"] == "zeros_all"
    assert "decoder" in result.provenance["task_family"]
    assert "' Paris'" in result.target_repr
    assert result.n_steps == 128
    assert result.cost["steps_per_batch"] == 8
    audit = result.cost["step_audit"]
    assert audit["mode"] == "per_call"
    assert audit["worst_deviation"] <= audit["rtol"]
    payload = result.show()
    assert len(payload.scores) == len(payload.raw_tokens)
    html = payload.to_html()
    assert "residual_rel" in html


@pytest.mark.heavy
def test_gpt2_batched_matches_sequential(gpt2: tuple[object, object]) -> None:
    """steps_per_batch=8 under the audit matches sequential at rtol 1e-3.

    Real float32 transformer gradients carry ~1e-4 infinity-norm-relative
    kernel noise between batched and unbatched matmuls, so the cross-check
    tolerance is the low-precision audit tier, not exact equality.
    """

    model, tokenizer = gpt2
    prompt = "The cat sat on the"
    batched = attribution.text(model, tokenizer, prompt, n_steps=16, steps_per_batch=8)
    sequential = attribution.text(model, tokenizer, prompt, n_steps=16, steps_per_batch=1)
    torch.testing.assert_close(batched.scores, sequential.scores, rtol=1e-3, atol=1e-5)


@pytest.mark.heavy
def test_distilgpt2_fast_row() -> None:
    """The distilgpt2 fast row: decoder auto baseline, sub-percent residual."""

    tokenizer = transformers.AutoTokenizer.from_pretrained("distilgpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()
    result = attribution.text(model, tokenizer, "The capital of France is")
    assert result.completeness["residual_rel"] < 0.01
    assert result.baseline["policy"] == "zeros_all"


@pytest.mark.heavy
def test_bert_mask_row_auto_pad_scaffold() -> None:
    """bert [MASK] callable-target row: auto resolves pad + scaffolding.

    The encoder cell of baseline='auto' is pad content + special-token
    scaffolding (the only measured converging baseline, monotone to 0.097%
    at n=1024; 0.643% at n=128 on the panel's prompt). NO residual bound is
    asserted for the zeros baseline -- none exists at any measured grid.
    """

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    model = transformers.AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()
    prompt = "The capital of France is [MASK]."
    encoding = tokenizer(prompt, return_tensors="pt")
    mask_position = int((encoding["input_ids"][0] == tokenizer.mask_token_id).nonzero().item())
    with torch.no_grad():
        logits = model(**encoding).logits
    predicted = int(logits[0, mask_position].argmax())

    def mask_scorer(out: torch.Tensor) -> torch.Tensor:
        """Score the predicted token at the [MASK] position."""

        return out[0, mask_position, predicted]

    result = attribution.text(model, tokenizer, prompt, target=mask_scorer)
    assert result.baseline["policy"] == "pad_content_plus_scaffold"
    assert "encoder" in result.provenance["task_family"]
    # Special tokens ([CLS]/[SEP]) are scaffolded at their true embeddings.
    assert result.baseline["scaffolded_positions"], "bert specials must scaffold"
    assert result.completeness["residual_rel"] < 0.02
    # The zeros baseline carries NO bound; assert only that it runs and
    # discloses its explicit policy.
    zeros = attribution.text(
        model, tokenizer, prompt, target=mask_scorer, baseline="zeros", n_steps=16
    )
    assert zeros.baseline["policy"] == "zeros_all"


@pytest.mark.slow
def test_qwen_gqa_row() -> None:
    """Qwen2.5-0.5B-Instruct: the GQA-architecture row (slow tier)."""

    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32
    )
    model.eval()
    result = attribution.text(model, tokenizer, "The capital of France is", n_steps=32)
    assert result.baseline["policy"] == "zeros_all"
    assert torch.isfinite(result.scores).all()
