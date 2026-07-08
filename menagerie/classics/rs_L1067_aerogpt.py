# RUNG 1 -- real library model, no architectural modification.
#
# AeroGPT (Liu, Peng, Wang, Liu, Li, Xie -- IEEE Trans. Cybernetics, accepted;
# arXiv:2506.16225) "Leveraging Large-Scale Audio Model for Aero-Engine Bearing
# Fault Diagnosis". Per the paper (Sec. IV-B, footnote 39, Table results): "The
# foundation model used in the experiments was initialized with the weights of
# Qwen2-Audio [39], which comprises a pre-trained audio encoder with 124 million
# parameters and a large language model with 7 billion parameters." AeroGPT's
# contribution is (a) treating bearing vibration signals as audio-like waveforms
# fed through the SAME Qwen2-Audio audio encoder + LLM pipeline, (b) LoRA-based
# domain adaptation (Vibration Signal Alignment) applied to Qwen2-Audio's
# existing linear layers, and (c) a generative-fault-classification training
# objective. No new nn.Module architecture is introduced -- Qwen2-Audio itself
# (audio_tower: 32-layer / 20-head Whisper-style conv+transformer encoder,
# multi_modal_projector, and a Qwen2 causal-LM decoder) is used unmodified, as
# stated explicitly in the paper. This is architecturally analogous to
# DialoGPT=GPT2 / TOD-BERT=BertModel: the real transformers library class,
# `Qwen2AudioForConditionalGeneration`, IS the AeroGPT architecture.
#
# MENAGERIE_ZOO = "vendored-pytorch" (staging module because Qwen2-Audio needs
# TWO inputs -- input_features (audio) and input_ids (text) -- so it cannot be
# expressed as a single-tensor recipe row; the module wraps the real
# transformers class directly with no modification.)

import torch

from transformers import Qwen2AudioForConditionalGeneration
from transformers.models.qwen2_audio.configuration_qwen2_audio import (
    Qwen2AudioConfig,
    Qwen2AudioEncoderConfig,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

MENAGERIE_ZOO = "vendored-pytorch"

# ---------------------------------------------------------------------------
# Menagerie staging glue: tiny Qwen2-Audio config (real HF architecture,
# shrunk dims only) + example (input_features, input_ids) pair sized so the
# audio-token count in input_ids matches the audio encoder's real output
# length formula (see Qwen2AudioEncoder._get_feat_extract_output_lengths:
# out_len = ((mel_len - 1) // 2 + 1 - 2) // 2 + 1).
# ---------------------------------------------------------------------------
_NUM_MEL_BINS = 8
_MEL_SEQ_LEN = 16  # -> audio_feat_lengths=8 -> audio_output_lengths=4
_N_AUDIO_TOKENS = 4
_VOCAB_SIZE = 64
_AUDIO_TOKEN_ID = 5
_TEXT_LEN = 6  # total input_ids length including the 4 audio placeholder tokens
_BATCH = 1


def build_aerogpt():
    torch.manual_seed(0)
    audio_config = Qwen2AudioEncoderConfig(
        num_mel_bins=_NUM_MEL_BINS,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        d_model=8,
        max_source_positions=8,
    )
    text_config = Qwen2Config(
        vocab_size=_VOCAB_SIZE,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config = Qwen2AudioConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_index=_AUDIO_TOKEN_ID,
    )
    model = Qwen2AudioForConditionalGeneration(config)
    model.eval()
    return model


def example_input_aerogpt():
    torch.manual_seed(0)
    input_features = torch.randn(_BATCH, _NUM_MEL_BINS, _MEL_SEQ_LEN)
    feature_attention_mask = torch.ones(_BATCH, _MEL_SEQ_LEN, dtype=torch.long)

    # Build input_ids with exactly _N_AUDIO_TOKENS consecutive audio-placeholder
    # tokens (non-legacy processing path) followed by text tokens, all < vocab_size
    # and != audio_token_id.
    text_tokens = torch.randint(
        _N_AUDIO_TOKENS + 1, _VOCAB_SIZE, (_BATCH, _TEXT_LEN - _N_AUDIO_TOKENS)
    )
    audio_tokens = torch.full((_BATCH, _N_AUDIO_TOKENS), _AUDIO_TOKEN_ID, dtype=torch.long)
    input_ids = torch.cat([audio_tokens, text_tokens], dim=1)
    attention_mask = torch.ones_like(input_ids)

    return (input_ids, input_features, attention_mask, feature_attention_mask)


MENAGERIE_ENTRIES = [
    ("AeroGPT", "build_aerogpt", "example_input_aerogpt", 2026, MENAGERIE_ZOO),
]
