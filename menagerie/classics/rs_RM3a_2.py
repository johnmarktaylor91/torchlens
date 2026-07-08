# SOURCE: rung-1 real library model. The generative-ABSA approach fine-tunes an unmodified
# transformers T5ForConditionalGeneration to emit linearized aspect-sentiment triplets/labels as
# free text -- no architectural change of any kind, only the text-generation task formulation
# (annotation-style / extraction-style target sequences) and the fine-tuning data/objective.
# Confirmed against the real training code: IsakZhang/Generative-ABSA @ main, main.py
# (`class T5FineTuner(pl.LightningModule): ... self.model =
# T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)`), the official code for
# Zhang, Li, Deng, Bing & Lam, "Towards Generative Aspect-Based Sentiment Analysis," ACL 2021
# (GAS). Because the real model is a seq2seq encoder-decoder needing both `input_ids` and
# `decoder_input_ids`, this is staged as a module (not a single-tensor recipe row) per the
# multi-input carve-out for rung-1 real-library models.
from __future__ import annotations

import torch
from torch import Tensor
from transformers import T5Config, T5ForConditionalGeneration

MENAGERIE_ZOO = "vendored-pytorch"


def build_absa_t5() -> T5ForConditionalGeneration:
    """Build a tiny, traceable T5ForConditionalGeneration for generative ABSA fine-tuning."""
    cfg = T5Config(vocab_size=256, d_model=32, d_ff=64, num_layers=2, num_heads=2, d_kv=16)
    model = T5ForConditionalGeneration(cfg)
    model.eval()
    return model


def example_input_absa_t5() -> tuple[Tensor, Tensor, Tensor]:
    """Return (input_ids, attention_mask, decoder_input_ids) for the generative-ABSA T5 model."""
    input_ids = torch.randint(0, 256, (1, 16))
    attention_mask = torch.ones_like(input_ids)
    decoder_input_ids = torch.randint(0, 256, (1, 8))
    return input_ids, attention_mask, decoder_input_ids


MENAGERIE_ENTRIES = [
    (
        "ABSA Conditional Generation (Seq2Seq ABSA)",
        build_absa_t5,
        example_input_absa_t5,
        2021,
        "RM3a-absa-t5",
    ),
]
