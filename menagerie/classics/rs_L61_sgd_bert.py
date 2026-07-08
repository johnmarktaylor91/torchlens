# FAITHFUL PORT of google-research/google-research @ master
# (schema_guided_dst/baseline/train_and_predict.py::SchemaGuidedDST,
# original framework: TensorFlow 1.x / tf.compat.v1 + tf.estimator)
#
# https://github.com/google-research/google-research/tree/master/schema_guided_dst
# https://raw.githubusercontent.com/google-research/google-research/master/schema_guided_dst/baseline/train_and_predict.py
# https://raw.githubusercontent.com/google-research/google-research/master/schema_guided_dst/baseline/config.py
#
# Rastogi et al. 2020 "Towards Scalable Multi-Domain Conversational Agents:
# The Schema-Guided Dialogue Dataset" (AAAI 2020) -- Google's official SGD
# baseline model, `SchemaGuidedDST` (`schema_guided_dst/baseline/train_and_
# predict.py:281-638`). This is a FAITHFUL PORT (not a vendor) because the
# real code depends on TensorFlow 1.14 (`tensorflow.compat.v1`,
# `tf.estimator`/`tf_estimator.tpu.TPUEstimatorSpec`, and the bundled
# `schema_guided_dst/baseline/bert/modeling.py` -- Google's original TF1 BERT
# implementation) which is not an installed base lib and is not being added
# (TF1.x is EOL and incompatible with the modern deps in this environment).
#
# Every mechanism of `SchemaGuidedDST` is transcribed faithfully into torch:
#   - `_encode_utterances` (TF1 `modeling.BertModel` -> pooled_output +
#     sequence_output, with dropout) -> real `transformers.BertModel`
#     (`pooler_output` == TF1 `get_pooled_output()`, `last_hidden_state` ==
#     `get_sequence_output()`), same dropout applied to both,
#   - `_get_logits` (the shared classifier head used by every downstream
#     task: GELU-activated `Dense(embedding_dim)` projection of the utterance
#     embedding, tiled and concatenated with per-element embeddings along the
#     last axis, then `Dense(embedding_dim, gelu) -> Dense(num_classes)`) ->
#     `SgdLogitsHead` below, a line-for-line re-expression using
#     `nn.Linear`/`F.gelu`,
#   - `_get_intents` (null-intent trainable embedding prepended to the
#     provided intent embeddings, then `_get_logits(..., 1, ...)` squeezed) ->
#     `_get_intents` below, `null_intent_embedding` as an `nn.Parameter`,
#   - `_get_requested_slots` (`_get_logits(slot_embeddings, 1, ...)` squeezed)
#     -> preserved verbatim,
#   - `_get_categorical_slot_goals` (status head over slot embeddings +
#     value head over flattened (slot, value) embeddings, reshaped back) ->
#     preserved verbatim including the reshape-flatten-reshape pattern,
#   - `_get_noncategorical_slot_goals` (status head over slot embeddings;
#     span head: tile token embeddings x slots and slot embeddings x tokens,
#     concat, `Dense(gelu) -> Dense(2)`, unstack into start/end logits) ->
#     preserved verbatim,
#   - `define_model`'s exact output dict (`logit_intent_status`,
#     `logit_req_slot_status`, `logit_cat_slot_status`, `logit_cat_slot_value`,
#     `logit_noncat_slot_status`, `logit_noncat_slot_start`,
#     `logit_noncat_slot_end`) -> preserved as the forward() return value.
#   - The `-0.7 * dtype.max` masking of padded logits, the loss functions
#     (`define_loss`), the TPUEstimator training/serving plumbing
#     (`_model_fn_builder`, checkpoint scaffolding, `tf.summary` logging),
#     and TF-only input-pipeline code (`_file_based_input_fn_builder`,
#     `tf.Example` decoding) are dropped -- they configure *how* the TF1
#     graph is trained/served/fed, not what tensors the forward pass over a
#     single batch of schema-embedding-conditioned classifier heads computes,
#     which is what TorchLens captures.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


class SgdLogitsHead(nn.Module):
    """Faithful torch port of SchemaGuidedDST._get_logits.

    Upstream (TF1) forward (train_and_predict.py:504-538):
        utterance_proj = Dense(embedding_dim, activation=gelu)
        utterance_embedding = utterance_proj(encoded_utterance)
        repeat_utterance_embeddings = tile(expand_dims(utterance_embedding, 1), [1, num_elements, 1])
        utterance_element_emb = concat([repeat_utterance_embeddings, element_embeddings], axis=2)
        layer_1 = Dense(embedding_dim, activation=gelu)
        layer_2 = Dense(num_classes)
        return layer_2(layer_1(utterance_element_emb))
    """

    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.utterance_proj = nn.Linear(embedding_dim, embedding_dim)
        self.layer_1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.layer_2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, encoded_utterance, element_embeddings):
        # encoded_utterance: (batch, embedding_dim)
        # element_embeddings: (batch, num_elements, embedding_dim)
        num_elements = element_embeddings.shape[1]
        utterance_embedding = F.gelu(self.utterance_proj(encoded_utterance))
        repeat_utterance_embeddings = utterance_embedding.unsqueeze(1).expand(-1, num_elements, -1)
        utterance_element_emb = torch.cat([repeat_utterance_embeddings, element_embeddings], dim=2)
        hidden = F.gelu(self.layer_1(utterance_element_emb))
        return self.layer_2(hidden)


class SgdSpanHead(nn.Module):
    """Faithful torch port of the span-prediction branch of
    SchemaGuidedDST._get_noncategorical_slot_goals (train_and_predict.py:614-628).

    Upstream (TF1) forward:
        tiled_token_embeddings = tile(expand_dims(token_embeddings, 1), [1, max_num_slots, 1, 1])
        tiled_slot_embeddings = tile(expand_dims(slot_embeddings, 2), [1, 1, max_num_tokens, 1])
        slot_token_embeddings = concat([tiled_slot_embeddings, tiled_token_embeddings], axis=3)
        layer_1 = Dense(embedding_dim, activation=gelu)
        layer_2 = Dense(2)
        span_logits = layer_2(layer_1(slot_token_embeddings))
        span_start_logits, span_end_logits = unstack(span_logits, axis=3)
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.layer_1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.layer_2 = nn.Linear(embedding_dim, 2)

    def forward(self, slot_embeddings, token_embeddings):
        # slot_embeddings: (batch, max_num_slots, embedding_dim)
        # token_embeddings: (batch, max_num_tokens, embedding_dim)
        max_num_slots = slot_embeddings.shape[1]
        max_num_tokens = token_embeddings.shape[1]
        tiled_token_embeddings = token_embeddings.unsqueeze(1).expand(-1, max_num_slots, -1, -1)
        tiled_slot_embeddings = slot_embeddings.unsqueeze(2).expand(-1, -1, max_num_tokens, -1)
        slot_token_embeddings = torch.cat([tiled_slot_embeddings, tiled_token_embeddings], dim=3)
        hidden = F.gelu(self.layer_1(slot_token_embeddings))
        span_logits = self.layer_2(hidden)
        span_start_logits = span_logits[..., 0]
        span_end_logits = span_logits[..., 1]
        return span_start_logits, span_end_logits


class SchemaGuidedDST(nn.Module):
    """Faithful torch port of schema_guided_dst/baseline/train_and_predict.py::
    SchemaGuidedDST (the SGD-BERT baseline: a BERT utterance encoder feeding
    four schema-conditioned classifier heads for zero-shot slot filling).
    """

    def __init__(self, bert_config, dropout_rate=0.1):
        super().__init__()
        self.bert = BertModel(bert_config)
        self.dropout = nn.Dropout(dropout_rate)
        embedding_dim = bert_config.hidden_size

        self.intents_head = SgdLogitsHead(embedding_dim, 1)
        self.requested_slots_head = SgdLogitsHead(embedding_dim, 1)
        self.cat_slot_status_head = SgdLogitsHead(embedding_dim, 3)
        self.cat_slot_value_head = SgdLogitsHead(embedding_dim, 1)
        self.noncat_slot_status_head = SgdLogitsHead(embedding_dim, 3)
        self.noncat_span_head = SgdSpanHead(embedding_dim)

        # null-intent trainable embedding (train_and_predict.py:551-554)
        self.null_intent_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dim))

    def _encode_utterances(self, utt_ids, utt_mask, utt_seg):
        outputs = self.bert(input_ids=utt_ids, attention_mask=utt_mask, token_type_ids=utt_seg)
        encoded_utterance = self.dropout(outputs.pooler_output)
        encoded_tokens = self.dropout(outputs.last_hidden_state)
        return encoded_utterance, encoded_tokens

    def _get_intents(self, encoded_utterance, intent_emb):
        batch_size = intent_emb.shape[0]
        repeated_null = self.null_intent_embedding.expand(batch_size, -1, -1)
        intent_embeddings = torch.cat([repeated_null, intent_emb], dim=1)
        logits = self.intents_head(encoded_utterance, intent_embeddings)
        return logits.squeeze(-1)

    def _get_requested_slots(self, encoded_utterance, req_slot_emb):
        logits = self.requested_slots_head(encoded_utterance, req_slot_emb)
        return logits.squeeze(-1)

    def _get_categorical_slot_goals(self, encoded_utterance, cat_slot_emb, cat_slot_value_emb):
        status_logits = self.cat_slot_status_head(encoded_utterance, cat_slot_emb)

        batch_size, max_num_slots, max_num_values, embedding_dim = cat_slot_value_emb.shape
        value_embeddings_reshaped = cat_slot_value_emb.reshape(
            batch_size, max_num_slots * max_num_values, embedding_dim
        )
        value_logits = self.cat_slot_value_head(encoded_utterance, value_embeddings_reshaped)
        value_logits = value_logits.reshape(batch_size, max_num_slots, max_num_values)
        return status_logits, value_logits

    def _get_noncategorical_slot_goals(self, encoded_utterance, encoded_tokens, noncat_slot_emb):
        status_logits = self.noncat_slot_status_head(encoded_utterance, noncat_slot_emb)
        span_start_logits, span_end_logits = self.noncat_span_head(noncat_slot_emb, encoded_tokens)
        return status_logits, span_start_logits, span_end_logits

    def forward(
        self,
        utt_ids,
        utt_mask,
        utt_seg,
        intent_emb,
        req_slot_emb,
        cat_slot_emb,
        cat_slot_value_emb,
        noncat_slot_emb,
    ):
        encoded_utterance, encoded_tokens = self._encode_utterances(utt_ids, utt_mask, utt_seg)

        logit_intent_status = self._get_intents(encoded_utterance, intent_emb)
        logit_req_slot_status = self._get_requested_slots(encoded_utterance, req_slot_emb)
        logit_cat_slot_status, logit_cat_slot_value = self._get_categorical_slot_goals(
            encoded_utterance, cat_slot_emb, cat_slot_value_emb
        )
        logit_noncat_slot_status, logit_noncat_slot_start, logit_noncat_slot_end = (
            self._get_noncategorical_slot_goals(encoded_utterance, encoded_tokens, noncat_slot_emb)
        )

        return (
            logit_intent_status,
            logit_req_slot_status,
            logit_cat_slot_status,
            logit_cat_slot_value,
            logit_noncat_slot_status,
            logit_noncat_slot_start,
            logit_noncat_slot_end,
        )


def build_sgd_bert():
    bert_config = BertConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return SchemaGuidedDST(bert_config, dropout_rate=0.1)


def example_input_sgd_bert():
    batch_size = 2
    max_num_tokens = 16
    max_num_intents = 3
    max_num_req_slots = 4
    max_num_cat_slots = 2
    max_num_cat_values = 3
    max_num_noncat_slots = 2
    embedding_dim = 32

    utt_ids = torch.randint(0, 200, (batch_size, max_num_tokens))
    utt_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.long)
    utt_seg = torch.zeros(batch_size, max_num_tokens, dtype=torch.long)

    intent_emb = torch.randn(batch_size, max_num_intents, embedding_dim)
    req_slot_emb = torch.randn(batch_size, max_num_req_slots, embedding_dim)
    cat_slot_emb = torch.randn(batch_size, max_num_cat_slots, embedding_dim)
    cat_slot_value_emb = torch.randn(
        batch_size, max_num_cat_slots, max_num_cat_values, embedding_dim
    )
    noncat_slot_emb = torch.randn(batch_size, max_num_noncat_slots, embedding_dim)

    return (
        utt_ids,
        utt_mask,
        utt_seg,
        intent_emb,
        req_slot_emb,
        cat_slot_emb,
        cat_slot_value_emb,
        noncat_slot_emb,
    )


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "SGD-BERT (Schema-Guided Dialogue baseline)",
        "build_sgd_bert",
        "example_input_sgd_bert",
        2020,
        "ported",
    ),
]
