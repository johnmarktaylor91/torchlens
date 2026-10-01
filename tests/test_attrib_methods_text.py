"""F06 B8: the token-attribution two-liner (attrib memo D22-D29) -- toy tier.

A tiny HF-shaped causal model + word-level fake tokenizer exercise the whole
contract without network: target resolution incl. the bare-int warn band,
task-aware baseline resolution and its named fallback, explicit baseline
spellings and refusals, the dual-criterion auto ladder (incl. the V17/T1
named regression on the stopping predicate), truncation disclosure, and the
escaped-table payload. Real-checkpoint rows live in
``test_attrib_methods_text_real.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens.attribution as attribution
from torchlens.attribution import AttributionError
from torchlens.attribution._result import AttributionWarning
from torchlens.attribution._text import _dual_criterion_met

pytestmark = pytest.mark.smoke

_VOCAB = ["<pad>", "the", "eiffel", "tower", "is", "in", "paris", "france", "cat", "sat"]


class _FakeTokenizer:
    """Deterministic word-level tokenizer with the HF surface text() reads."""

    def __init__(self, pad_token_id: int | None = 0) -> None:
        """Build the vocab maps."""

        self.vocab = {token: index for index, token in enumerate(_VOCAB)}
        self.pad_token_id = pad_token_id
        self.name_or_path = "fake-word-tokenizer"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        """Tokenize by whitespace; honor truncation/max_length; emit masks."""

        ids = [self.vocab[word] for word in text.lower().split()]
        if kwargs.get("truncation") and kwargs.get("max_length") is not None:
            ids = ids[: kwargs["max_length"]]
        encoding: dict[str, Any] = {"input_ids": ids}
        if kwargs.get("return_tensors") == "pt":
            encoding = {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
            if kwargs.get("return_special_tokens_mask"):
                encoding["special_tokens_mask"] = torch.zeros(1, len(ids), dtype=torch.long)
            if kwargs.get("return_offsets_mapping"):
                encoding["offset_mapping"] = torch.tensor([[[0, 1]] * len(ids)], dtype=torch.long)
        return encoding

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode without specials (this tokenizer has none)."""

        del add_special_tokens
        return [self.vocab[word] for word in text.lower().split()]

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to words."""

        return " ".join(_VOCAB[index] for index in ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Convert ids to raw tokens."""

        return [_VOCAB[index] for index in ids]


class _TinyForCausalLM(nn.Module):
    """HF-shaped tiny causal LM (class name drives family detection)."""

    class _Config:
        """Minimal config carrying the identity fields text() reads."""

        _name_or_path = "tiny-causal"
        is_decoder = True

    def __init__(self, curvature: float = 1.0) -> None:
        """Build deterministic embeddings and head.

        Parameters
        ----------
        curvature
            Frequency of the sine nonlinearity; large values make the IG
            integral genuinely hard so the auto ladder honestly caps out.
        """

        super().__init__()
        self.config = self._Config()
        self.curvature = curvature
        self.wte = nn.Embedding(len(_VOCAB), 8, dtype=torch.float64)
        self.head = nn.Linear(8, len(_VOCAB), dtype=torch.float64)
        with torch.no_grad():
            weights = torch.arange(len(_VOCAB) * 8, dtype=torch.float64)
            self.wte.weight.copy_(weights.reshape(len(_VOCAB), 8) / 40.0 - 0.9)
            self.head.weight.copy_(
                torch.arange(len(_VOCAB) * 8, dtype=torch.float64).reshape(len(_VOCAB), 8) / 30.0
                - 1.1
            )
            self.head.bias.zero_()

    def get_input_embeddings(self) -> nn.Module:
        """Return the token embedding module."""

        return self.wte

    def forward(
        self,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        **_ignored: Any,
    ) -> tuple[Tensor]:
        """Per-position logits over a smooth (or curved) nonlinearity."""

        assert inputs_embeds is not None
        assert attention_mask is not None
        assert attention_mask.shape[0] == inputs_embeds.shape[0], (
            "fixed integer inputs must be tiled to the stacked batch"
        )
        hidden = torch.sin(self.curvature * inputs_embeds)
        return (self.head(hidden),)


def test_two_liner_runs_and_disches_everything() -> None:
    """The two lines work; the result carries the full disclosure record."""

    model = _TinyForCausalLM()
    tokenizer = _FakeTokenizer()
    result = attribution.text(model, tokenizer, "the eiffel tower is in", n_steps=64)
    assert result.n_steps == 64
    assert len(result.scores) == 5
    assert result.values.shape == (5, 8)
    assert result.baseline["policy"] == "zeros_all"
    assert "decoder family" in result.baseline["reason"]
    assert result.target_repr.startswith("default: model argmax next token")
    assert result.provenance["task_family"] == "decoder"
    assert result.provenance["model_name_or_path"] == "tiny-causal"
    # Per-token scores retain completeness accounting: sum(scores) is the
    # attribution sum the residual is certified against.
    torch.testing.assert_close(
        result.scores.sum(),
        result.completeness["attribution_sum"],
        rtol=1e-9,
        atol=1e-12,
    )
    payload = result.show()
    assert payload.score_domain == "zero_centered_diverging"
    assert any(line.startswith("target:") for line in payload.footer_lines)
    assert any("|target_delta|" in line for line in payload.footer_lines)
    assert any("residual_rel" in line for line in payload.footer_lines)
    assert any("n_steps='auto'" in line for line in payload.footer_lines)


def test_auto_ladder_certifies_on_smooth_model() -> None:
    """The dual criterion certifies a smooth model at grid pair (64, 128)."""

    model = _TinyForCausalLM(curvature=0.5)
    tokenizer = _FakeTokenizer()
    result = attribution.text(model, tokenizer, "the cat sat", n_steps="auto")
    assert result.converged is True
    assert result.convergence["grid_pair"] == (64, 128)
    assert result.convergence["grids_evaluated"] == [64, 128]
    payload = result.show()
    assert any("met criteria at grid pair" in line for line in payload.footer_lines)
    assert any("detector, not a proof" in line for line in payload.footer_lines)


def test_auto_ladder_cap_hit_is_first_class(recwarn: pytest.WarningsRecorder) -> None:
    """D26: a genuinely hard integral caps out with converged=False, visibly."""

    model = _TinyForCausalLM(curvature=3000.0)
    tokenizer = _FakeTokenizer()
    result = attribution.text(model, tokenizer, "the cat sat is in", n_steps="auto")
    assert result.converged is False
    assert result.convergence["grids_evaluated"] == [64, 128, 256, 512]
    payload = result.show()
    assert any("NOT CERTIFIED" in line for line in payload.footer_lines)
    assert any("under-resolved integral" in line for line in payload.footer_lines)


def test_v17_t1_regression_stopping_predicate() -> None:
    """The V17/T1 named regression: measured false-pass cases must NOT certify.

    bert/zeros at n=16 read residual 1.80% (sign-cancellation luck);
    bert/pad-plain at n=128 read residual 0.584% with 3.33% instability.
    Neither may certify under any shipped rule; one grid alone never
    certifies; and 5%/5% thresholds are measured non-discriminating so the
    tolerances may tighten but never loosen past 1%.
    """

    assert not _dual_criterion_met(0.018, None)  # bert/zeros n=16, single grid
    assert not _dual_criterion_met(0.018, 0.001)  # residual above 1%
    assert not _dual_criterion_met(0.00584, 0.0333)  # saved ONLY by stability
    assert _dual_criterion_met(0.00584, 0.009)
    from torchlens.attribution._text import (
        _AUTO_LADDER,
        _RESIDUAL_TOLERANCE,
        _STABILITY_TOLERANCE,
    )

    assert _AUTO_LADDER[0] == 64, "starting below 64 readmits the measured false pass"
    assert _RESIDUAL_TOLERANCE <= 0.01, "thresholds may tighten, never loosen"
    assert _STABILITY_TOLERANCE <= 0.01, "thresholds may tighten, never loosen"


def test_bare_int_warn_band() -> None:
    """D29: int < sequence_length warns; int >= sequence_length does not."""

    model = _TinyForCausalLM()
    tokenizer = _FakeTokenizer()
    prompt = "the cat sat"  # 3 tokens; vocab has 10 entries
    with pytest.warns(AttributionWarning) as captured:
        ambiguous = attribution.text(model, tokenizer, prompt, target=2, n_steps=8)
    assert any(
        warning.message.fields.get("code") == "text_bare_int_target_ambiguous"
        for warning in captured
    )
    assert "logits[0, 2, 2]" in ambiguous.target_repr
    import warnings as warnings_module

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error", AttributionWarning)
        unambiguous = attribution.text(model, tokenizer, prompt, target=7, n_steps=8)
    assert "logits[0, 2, 7]" in unambiguous.target_repr


def test_target_spellings() -> None:
    """Tuple, token string, contrastive pair, and callable targets resolve."""

    model = _TinyForCausalLM()
    tokenizer = _FakeTokenizer()
    prompt = "the eiffel tower is in"
    tuple_result = attribution.text(model, tokenizer, prompt, target=(4, 6), n_steps=8)
    assert "logits[0, 4, 6]" in tuple_result.target_repr
    string_result = attribution.text(model, tokenizer, prompt, target="paris", n_steps=8)
    assert "logits[0, 4, 6]" in string_result.target_repr
    torch.testing.assert_close(string_result.scores, tuple_result.scores)
    contrastive = attribution.text(
        model, tokenizer, prompt, target={"target": "paris", "foil": "france"}, n_steps=8
    )
    assert "contrastive" in contrastive.target_repr

    def scorer(logits: Tensor) -> Tensor:
        """Custom scalarizer."""

        return logits[0, -1, 6] - logits[0, -1, 7]

    callable_result = attribution.text(model, tokenizer, prompt, target=scorer, n_steps=8)
    assert callable_result.target_repr == "scorer"
    with pytest.raises(AttributionError) as multi:
        attribution.text(model, tokenizer, prompt, target="the cat", n_steps=8)
    assert multi.value.fields["code"] == "text_target_unresolvable"


def test_explicit_baseline_spellings_and_refusals() -> None:
    """zeros / pad_token / ids / tensor baselines resolve; misuse refuses."""

    model = _TinyForCausalLM()
    tokenizer = _FakeTokenizer()
    prompt = "the cat sat"
    zeros = attribution.text(model, tokenizer, prompt, baseline="zeros", n_steps=16)
    assert zeros.baseline["policy"] == "zeros_all"
    pad = attribution.text(model, tokenizer, prompt, baseline="pad_token", n_steps=16)
    assert pad.baseline["policy"] == "pad_content_plus_scaffold"
    assert pad.baseline["baseline_ids"] == [0, 0, 0]
    ids = attribution.text(model, tokenizer, prompt, baseline=[0, 0, 0], n_steps=16)
    torch.testing.assert_close(ids.scores, pad.scores)
    tensor_baseline = torch.zeros(1, 3, 8, dtype=torch.float64)
    tensor_result = attribution.text(model, tokenizer, prompt, baseline=tensor_baseline, n_steps=16)
    torch.testing.assert_close(tensor_result.scores, zeros.scores)

    padless = _FakeTokenizer(pad_token_id=None)
    with pytest.raises(AttributionError) as no_pad:
        attribution.text(model, padless, prompt, baseline="pad_token", n_steps=8)
    assert no_pad.value.fields["code"] == "text_baseline_unresolvable"
    assert "never guessed" in str(no_pad.value)
    with pytest.raises(AttributionError) as bad_ids:
        attribution.text(model, tokenizer, prompt, baseline=[0, 0], n_steps=8)
    assert bad_ids.value.fields["code"] == "text_baseline_unresolvable"
    with pytest.raises(AttributionError) as bad_spelling:
        attribution.text(model, tokenizer, prompt, baseline=3.5, n_steps=8)
    assert bad_spelling.value.fields["code"] == "text_baseline_unresolvable"


def test_explicit_baseline_residual_warning() -> None:
    """Explicit spellings warn when the achieved residual exceeds tolerance."""

    model = _TinyForCausalLM(curvature=60.0)
    tokenizer = _FakeTokenizer()
    with pytest.warns(AttributionWarning) as captured:
        attribution.text(model, tokenizer, "the cat sat is in", baseline="zeros", n_steps=8)
    assert any(
        warning.message.fields.get("code") == "text_residual_above_tolerance"
        for warning in captured
    )


def test_truncation_is_explicit_and_disclosed() -> None:
    """max_length truncation records the span and warns; never silent."""

    model = _TinyForCausalLM()
    tokenizer = _FakeTokenizer()
    with pytest.warns(AttributionWarning) as captured:
        result = attribution.text(
            model, tokenizer, "the eiffel tower is in paris", max_length=4, n_steps=8
        )
    assert any(warning.message.fields.get("code") == "text_input_truncated" for warning in captured)
    assert result.truncation == {
        "original_length": 6,
        "retained_length": 4,
        "retained_span": [0, 4],
    }
    assert any("truncation" in line for line in result.footer_lines())


def test_payload_escapes_hostile_tokens() -> None:
    """The fallback table escapes token text (D28 escaping rule)."""

    payload = attribution.TokenAttributionPayload(
        raw_tokens=["<script>", "ok"],
        display_tokens=["<script>alert(1)</script>", "ok"],
        scores=[1.0, -0.5],
        tooltips=["evil", "fine"],
        footer_lines=["target: <b>bold</b>"],
    )
    rendered = payload.to_html()
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;b&gt;" in rendered
    text_fallback = payload.to_text()
    assert "ok" in text_fallback and "target:" in text_fallback


def test_zero_curvature_ig_is_exact() -> None:
    """Sanity: near-linear model reaches ~0 residual at tiny grids."""

    model = _TinyForCausalLM(curvature=0.01)
    tokenizer = _FakeTokenizer()
    result = attribution.text(model, tokenizer, "the cat sat", n_steps=16)
    assert result.completeness["residual_rel"] < 1e-4
