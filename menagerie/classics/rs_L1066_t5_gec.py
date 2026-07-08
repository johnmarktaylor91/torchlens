# RUNG 1: real library model, no architectural modification.
#
# T5-based Grammatical Error Correction (gec-t5, https://github.com/gotutiyan/gec-t5).
# A reproduction of "A Simple Recipe for Multilingual Grammatical Error
# Correction" (Rothe et al., ACL 2021) that fine-tunes stock
# `transformers.T5ForConditionalGeneration` on cLang-8 + W&I GEC data; the
# repo's own `src/gec_t5/generate.py` API constructs the model via
# `AutoModelForSeq2SeqLM.from_pretrained(...)` with zero custom layers/heads
# -- the contribution is the training data/objective (source->corrected-target
# seq2seq), not the architecture. Staged as a module (not a recipe) because
# T5 is an encoder-decoder that needs two input tensors (`input_ids` and
# `decoder_input_ids`), which the single-tensor recipe format cannot express.

import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration

MENAGERIE_ZOO = "vendored-pytorch"

_VOCAB_SIZE = 100
_SEQ_LEN = 8


class _T5GECWrapper(nn.Module):
    """Thin wrapper so the traced model takes the two positional tensors T5's
    seq2seq forward needs (`input_ids`, `decoder_input_ids`) via a fixed
    keyword call -- calling T5ForConditionalGeneration.forward() with purely
    positional args trips an unrelated `transformers`-internal name-mangled
    warning-message bug (`__HEAD_MASK_WARNING_MSG`) on this version; routing
    through explicit kwargs avoids that path without touching T5 itself."""

    def __init__(self, model: T5ForConditionalGeneration):
        super().__init__()
        self.model = model

    def forward(self, input_ids, decoder_input_ids):
        return self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)


def build_t5_gec():
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=_VOCAB_SIZE,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_kv=8,
    )
    model = T5ForConditionalGeneration(config)
    model.eval()
    return _T5GECWrapper(model)


def example_input_t5_gec():
    torch.manual_seed(0)
    input_ids = torch.randint(0, _VOCAB_SIZE, (1, _SEQ_LEN))
    decoder_input_ids = torch.randint(0, _VOCAB_SIZE, (1, _SEQ_LEN))
    return (input_ids, decoder_input_ids)


MENAGERIE_ENTRIES = [
    ("T5-GEC", "build_t5_gec", "example_input_t5_gec", 2021, MENAGERIE_ZOO),
]
