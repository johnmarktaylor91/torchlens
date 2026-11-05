"""Text input to a bound-method root resolves through the method's owner.

``tl.trace(model.generate, "text")`` tokenizes the way ``tl.trace(model,
"text")`` does: an attached ``model.tokenizer`` or the owner's
``name_or_path`` (auto-resolved through ``AutoTokenizer``). Both cases run
offline on a tiny GPT-2 with a word-level tokenizer.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

import tokenizers  # noqa: E402  (a hard dependency of transformers)

import torchlens as tl  # noqa: E402
from torchlens._input_route import _owner_of  # noqa: E402
from torchlens.backends.torch.bound_root import TLBoundMethodRoot  # noqa: E402

WORDS = ("[UNK]", "[PAD]", "the", "capital", "of", "france", "is")
PROMPT = "the capital of france is"
GENERATE = {"max_new_tokens": 2, "do_sample": False}


def _tokenizer():
    """A word-level fast tokenizer over ``WORDS``; no download."""

    vocab = {word: index for index, word in enumerate(WORDS)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]"
    )


def _model():
    """A one-layer GPT-2 with random weights and no ``name_or_path``."""

    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=len(WORDS),
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=1,
        attn_implementation="eager",
    )
    return transformers.GPT2LMHeadModel(config).eval()


def test_input_owner_resolves_bound_methods_and_their_root() -> None:
    model = _model()
    assert _owner_of(model) is model
    assert _owner_of(model.generate) is model
    assert _owner_of(TLBoundMethodRoot(model.generate)) is model


def test_generate_text_uses_the_owners_attached_tokenizer() -> None:
    model = _model()
    model.tokenizer = _tokenizer()
    ids = model.tokenizer(PROMPT, return_tensors="pt").input_ids

    from_text = tl.trace(model.generate, PROMPT, input_kwargs=GENERATE)
    from_ids = tl.trace(model.generate, ids, input_kwargs=GENERATE)

    assert from_text.provenance.root_entry_point.endswith(".generate")
    assert len(from_text.ops) == len(from_ids.ops)


def test_generate_text_auto_resolves_the_owners_tokenizer(tmp_path) -> None:
    _tokenizer().save_pretrained(tmp_path)
    model = _model()
    model.config._name_or_path = str(tmp_path)
    assert not hasattr(model, "tokenizer")

    from_text = tl.trace(model.generate, PROMPT, input_kwargs=GENERATE)
    from_module = tl.trace(model, PROMPT)

    assert from_text.provenance.root_entry_point.endswith(".generate")
    # The prefill forward inside generate sees the same tokens the plain
    # module trace sees, so generate captures at least that many ops.
    assert len(from_text.ops) > len(from_module.ops)


def test_generate_text_without_any_tokenizer_still_refuses() -> None:
    model = _model()
    with pytest.raises(TypeError, match="model.tokenizer"):
        tl.trace(model.generate, PROMPT, input_kwargs=GENERATE)


@pytest.mark.real_model
def test_gpt2_generate_text_as_the_readme_spells_it() -> None:
    """The README call: pretrained GPT-2, text straight into ``generate``."""

    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()
    text = "The capital of France is"
    from_text = tl.trace(model.generate, text, input_kwargs={"max_new_tokens": 4})
    model.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    with_attached = tl.trace(model.generate, text, input_kwargs={"max_new_tokens": 4})
    ids = model.tokenizer(text, return_tensors="pt").input_ids
    from_ids = tl.trace(model.generate, ids, input_kwargs={"max_new_tokens": 4})

    assert from_text.provenance.root_entry_point.endswith(".generate")
    assert len(from_text.ops) == len(with_attached.ops)
    assert len(from_ids.ops) > 0
