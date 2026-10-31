"""U38.1: structure-only text capture with token ids plus an all-ones mask.

Ruling (adopt only if discharge passes): the text bridge moves the tokenized
ids to the meta device with a meta all-ones mask. Measured on distilgpt2,
Qwen2.5-0.5B and pythia-160m: CORROBORATED against both a real ids+mask trace
and the real text-route trace under eager and the TorchLens attention backend;
the sdpa mask builder reads mask values and refuses typed. These tests pin the
same three outcomes offline on a tiny config-built GPT-2 with an in-memory
tokenizer.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

import tokenizers  # noqa: E402  (a hard dependency of transformers)

from tests.real_model.r0.families import FAMILY_BY_NAME  # noqa: E402
from torchlens._errors import InvalidArgumentError  # noqa: E402
from torchlens.bridge import transformers as tl_hf  # noqa: E402
from torchlens.capture.structure_only import ValueDependentBranchError  # noqa: E402
from torchlens.options import CaptureOptions  # noqa: E402

SO = CaptureOptions(structure_only=True)
WORDS = ("[UNK]", "[PAD]", "hello", "world", "again")


def _tokenizer():
    vocab = {word: index for index, word in enumerate(WORDS)}
    model = tokenizers.models.WordLevel(vocab, unk_token="[UNK]")
    backend = tokenizers.Tokenizer(model)
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )


def _twins(impl: str):
    """Real and meta GPT-2 twins, both built before any capture."""

    real = FAMILY_BY_NAME["gpt2"].build(impl)
    with torch.device("meta"):
        meta = FAMILY_BY_NAME["gpt2"].build(impl)
    return real, meta.eval()


def test_eager_text_twin_discharges_corroborated() -> None:
    """ids + all-ones meta mask traces and corroborates against the real text route."""

    tok = _tokenizer()
    real, meta = _twins("eager")
    structure = tl_hf.trace_text(meta, "hello world again", tokenizer=tok, capture=SO)
    real_trace = tl_hf.trace_text(real, "hello world again", tokenizer=tok)
    discharge = structure.discharge_against(real_trace)
    assert discharge.verdict.value == "corroborated"
    assert len(discharge.claims) > 0
    assert len(structure._layer_list) == len(real_trace._layer_list)


def test_sdpa_text_capture_refuses_typed_with_the_attention_remedy() -> None:
    """sdpa reads the mask's values: typed refusal naming eager or the backend."""

    _real, meta = _twins("sdpa")
    with pytest.raises(ValueDependentBranchError) as excinfo:
        tl_hf.trace_text(meta, "hello world", tokenizer=_tokenizer(), capture=SO)
    assert excinfo.value.fields["code"] == "value_dependent_branch_unsupported"
    assert 'attn_implementation="eager"' in excinfo.value.fields["remedy"]
    assert "register_attention" in str(excinfo.value)


def test_padded_prompts_refuse_typed() -> None:
    """A padded batch has no all-ones mask: refuse instead of losing the pattern."""

    _real, meta = _twins("eager")
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl_hf.trace_text(meta, ["hello world again", "hello"], tokenizer=_tokenizer(), capture=SO)
    assert excinfo.value.fields["code"] == "structure_only_text_mask_not_all_ones"
