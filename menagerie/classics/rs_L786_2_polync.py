# SOURCE: real library model (transformers.T5ForConditionalGeneration)
#
# PolyNC ("PolyNC: a natural and chemical language model for the prediction of unified
# polymer properties", Chem. Sci. 2024, HKQiu/Unified_ML4Polymers): the repo's own
# `src/model.py::T5Model` wraps `transformers.T5ForConditionalGeneration` directly with
# NO architectural modification -- PolyNC's contribution is fine-tuning data (paired
# natural-language prompts + polymer SMILES) and a resized vocabulary/tokenizer, not a new
# architecture. The published checkpoint (huggingface.co/hkqiu/PolyNC, config.json
# `"architectures": ["T5ForConditionalGeneration"]`, initialized from
# `GT4SD/multitask-text-and-chemistry-t5-base-standard`) confirms this is stock T5-base
# (12+12 layers, d_model=768, d_ff=3072, 12 heads). This is a rung-1 real-library-model
# case (same class as DialoGPT=GPT2): construct the real `T5ForConditionalGeneration`
# class at tiny size with random init. Staged as a MODULE (not a recipe row) because T5 is
# an encoder-decoder that needs two concrete-tensor inputs (input_ids AND
# decoder_input_ids), which a single-input recipe row cannot express.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
from transformers import T5Config, T5ForConditionalGeneration


def build_polync():
    cfg = T5Config(
        vocab_size=1000,
        d_model=64,
        d_ff=128,
        d_kv=16,
        num_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        dropout_rate=0.1,
        dense_act_fn="relu",
        feed_forward_proj="relu",
        is_gated_act=False,
        layer_norm_epsilon=1e-6,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(cfg)
    model.eval()
    return model


def example_input_polync():
    input_ids = torch.randint(2, 1000, (1, 20))
    decoder_input_ids = torch.randint(2, 1000, (1, 6))
    # T5ForConditionalGeneration.forward signature: (input_ids, attention_mask,
    # decoder_input_ids, ...) -- None-fill attention_mask to hit decoder_input_ids
    # positionally, matching the established multi-input staging convention.
    return (input_ids, None, decoder_input_ids)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("PolyNC", "build_polync", "example_input_polync", 2024, "vendored"),
]
