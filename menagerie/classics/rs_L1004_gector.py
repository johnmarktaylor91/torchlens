# FAITHFUL PORT of grammarly/gector @ master (original framework: torch + allennlp==0.9.0)
#
# GECToR (Omelianchuk et al., BEA 2020, "GECToR -- Grammatical Error Correction: Tag, Not
# Rewrite") is a sequence-tagging grammatical-error-correction model: a pretrained
# transformer encoder (BERT/RoBERTa/XLNet via `AutoModel`, see
# gector/bert_token_embedder.py::PretrainedBertEmbedder) feeds token-level representations
# into TWO parallel linear classification heads (gector/seq2labels_model.py::Seq2Labels):
#   - `tag_labels_projection_layer`: predicts one of ~5000 edit-tag classes per token
#     (KEEP / DELETE / APPEND_x / REPLACE_x / g-transforms).
#   - `tag_detect_projection_layer`: a binary CORRECT/INCORRECT detection head used to
#     gate inference-time confidence (`min_error_probability`).
# Both heads are `TimeDistributed(Linear(hidden_size, num_classes))` applied over every
# token position (`predictor_dropout` before the labels head only). At inference, the
# labels head's argmax logit for the KEEP class and the DELETE class are boosted by a
# fixed `confidence` / `del_confidence` bias before argmax (see the `probability_change`
# vector added to `class_probabilities_labels` in `Seq2Labels.forward`) -- this bias trick
# is architectural behavior of the real model, not training-only, so it is preserved here.
#
# The real repo's `Seq2Labels`/`BertEmbedder` are AllenNLP `Model`/`TokenEmbedder`
# subclasses (gector/seq2labels_model.py, gector/bert_token_embedder.py) that depend on
# `allennlp` (EOL since ~2022, pinned to old torch/transformers, not installed and not
# reasonably installable alongside the current torch/transformers stack here). Only the
# AllenNLP scaffolding (Vocabulary, TimeDistributed wrapper, span-F1 metrics, windowed
# wordpiece-splitting for >512-token inputs) is dropped/replaced with plain torch
# equivalents; every architectural computation -- the transformer backbone via
# `transformers.AutoModel` (the same class the original code uses through
# `PretrainedBertModel.load` -> `AutoModel.from_pretrained`), the two linear heads, the
# dropout placement, and the confidence-bias trick -- is transcribed verbatim from
# gector/seq2labels_model.py::Seq2Labels.forward and
# gector/bert_token_embedder.py::BertEmbedder.forward (the no-window-split path, since the
# windowed-wordpiece-splitting logic only activates for sequences over `max_pieces=512`).

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

MENAGERIE_ZOO = "ported-pytorch"


class GECToR(nn.Module):
    """Seq2Labels: transformer encoder (AutoModel) -> two TimeDistributed-equivalent
    linear heads (edit-tag labels, correct/incorrect detection), with the inference-time
    confidence-bias trick from the real `Seq2Labels.forward`."""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels_classes: int,
        num_detect_classes: int,
        incorr_index: int,
        predictor_dropout: float = 0.0,
        confidence: float = 0.0,
        del_confidence: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.num_labels_classes = num_labels_classes
        self.num_detect_classes = num_detect_classes
        self.incorr_index = incorr_index
        self.confidence = confidence
        self.del_conf = del_confidence

        self.predictor_dropout = nn.Dropout(predictor_dropout)
        self.tag_labels_projection_layer = nn.Linear(hidden_size, num_labels_classes)
        self.tag_detect_projection_layer = nn.Linear(hidden_size, num_detect_classes)

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> dict:
        mask = attention_mask
        encoded_text = self.encoder(input_ids=input_ids, attention_mask=mask).last_hidden_state
        batch_size, sequence_length, _ = encoded_text.size()

        logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(encoded_text))
        logits_d = self.tag_detect_projection_layer(encoded_text)

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, sequence_length, self.num_labels_classes]
        )
        class_probabilities_d = F.softmax(logits_d, dim=-1).view(
            [batch_size, sequence_length, self.num_detect_classes]
        )
        error_probs = class_probabilities_d[:, :, self.incorr_index] * mask
        incorr_prob = torch.max(error_probs, dim=-1)[0]

        probability_change = [self.confidence, self.del_conf] + [0.0] * (
            self.num_labels_classes - 2
        )
        bias = torch.tensor(
            probability_change,
            dtype=class_probabilities_labels.dtype,
            device=class_probabilities_labels.device,
        )
        class_probabilities_labels = class_probabilities_labels + bias.repeat(
            (batch_size, sequence_length, 1)
        )

        return {
            "logits_labels": logits_labels,
            "logits_d_tags": logits_d,
            "class_probabilities_labels": class_probabilities_labels,
            "class_probabilities_d_tags": class_probabilities_d,
            "max_error_probability": incorr_prob,
        }


def build_gector():
    """Tiny GECToR (BERT-backbone GEC tagger, 2 parallel linear heads over the encoder's
    token representations) for tracing. Architecture is unmodified from the ported
    grammarly/gector source (allennlp scaffolding replaced with plain torch equivalents;
    computation is verbatim)."""
    bert_config = AutoConfig.for_model(
        "bert",
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        pad_token_id=0,
    )
    encoder = AutoModel.from_config(bert_config)
    model = GECToR(
        encoder=encoder,
        num_labels_classes=12,
        num_detect_classes=2,
        incorr_index=1,
        predictor_dropout=0.0,
        confidence=0.0,
        del_confidence=0.0,
    )
    model.eval()
    return model


def example_input_gector():
    batch, seq_len = 2, 9
    input_ids = torch.randint(1, 128, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    return (input_ids, attention_mask)


MENAGERIE_ENTRIES = [
    ("GECToR", build_gector, example_input_gector, 2020, "ported-pytorch"),
]
