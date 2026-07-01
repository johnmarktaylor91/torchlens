# SOURCE: vendored from guanlinchao/bert-dst @ master (main.py: create_model)
#
# BERT-DST (Chao & Lane, INTERSPEECH 2019, "BERT for Joint Intent Classification
# and Slot Filling" -- dialogue state tracking variant: "A Simple but Effective
# BERT Model for Dialog State Tracking on Resource-Limited Systems"). Real
# architecture per `create_model()` in the official repo: a BERT encoder (the
# original uses google-research/bert's `modeling.BertModel`, the same
# architecture as `transformers.BertModel` used here) feeds two per-slot heads
# built directly off BERT's outputs --
#   (1) a "class" head: linear stack over `pooled_output` predicting one of
#       {none, dontcare, copy_value, unpointable} for the slot, and
#   (2) a "token" head: linear stack over `sequence_output` predicting
#       per-token start/end span logits (for `copy_value` span extraction) --
# with one independent pair of heads instantiated per dialogue slot
# (`slot_scope_name = "slot_%s" % slot`). Default hyperparameters
# (`num_class_hidden_layer=0`, `num_token_hidden_layer=0`) give single-linear-
# layer heads directly from `hidden_size`; `fully_connect_layers` (util.py)
# is transcribed faithfully as `_fully_connect_layers` below (ReLU on every
# intermediate layer, none on the last).
#
# Vendoring notes:
#   - The real repo's `modeling.BertModel` (google-research/bert, TF1) is
#     replaced 1:1 with `transformers.BertModel` -- same encoder architecture
#     (embeddings + N transformer blocks + pooler), no modification; this is
#     the only "framework swap", not an architectural change.
#   - `create_model`'s TF `tf.get_variable` head-weight lists are transcribed
#     into `nn.Linear` layers wired through the identical `fully_connect_layers`
#     control flow (single layer when `num_*_hidden_layer == 0`, ReLU-gated
#     stack otherwise).
#   - Only the forward/logit computation is ported (`class_logits`,
#     `start_logits`, `end_logits` per slot); the loss/training-step code in
#     `create_model` and `model_fn_builder` is training-only and not part of
#     the traceable forward architecture.
#   - `slot_list` here uses a small subset (from the repo's `Dstc2Processor`)
#     to keep the recipe tiny; the head-construction logic is unchanged.

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


def _fully_connect_layers(input_layer, linears):
    """Faithful transcription of util.py's fully_connect_layers: input layer
    unchanged, ReLU on every layer except the last."""
    if len(linears) == 1:
        return linears[0](input_layer)
    logits = torch.relu(linears[0](input_layer))
    if len(linears) > 2:
        for layer in linears[1:-1]:
            logits = torch.relu(layer(logits))
    return linears[-1](logits)


class BertDSTSlotHead(nn.Module):
    """Per-slot class + span heads, ported from create_model()'s per-slot
    `tf.variable_scope(slot_scope_name)` block."""

    def __init__(
        self, hidden_size, num_class_labels, num_class_hidden_layer=0, num_token_hidden_layer=0
    ):
        super().__init__()
        class_dims = [hidden_size] + [64] * num_class_hidden_layer + [num_class_labels]
        token_dims = [hidden_size] + [64] * num_token_hidden_layer + [2]
        self.class_linears = nn.ModuleList(
            [nn.Linear(class_dims[i], class_dims[i + 1]) for i in range(len(class_dims) - 1)]
        )
        self.token_linears = nn.ModuleList(
            [nn.Linear(token_dims[i], token_dims[i + 1]) for i in range(len(token_dims) - 1)]
        )

    def forward(self, class_output_layer, token_output_layer):
        class_logits = _fully_connect_layers(class_output_layer, self.class_linears)
        batch_size, seq_length, hidden_size = token_output_layer.shape
        flat_token = token_output_layer.reshape(batch_size * seq_length, hidden_size)
        token_logits = _fully_connect_layers(flat_token, self.token_linears)
        token_logits = token_logits.reshape(batch_size, seq_length, 2)
        start_logits, end_logits = token_logits.unbind(dim=-1)
        return class_logits, start_logits, end_logits


class BertDST(nn.Module):
    """BERT-DST joint dialogue-state-tracking model: one BERT trunk shared
    across slots, one class+span head pair per slot (create_model())."""

    def __init__(
        self,
        bert_config,
        slot_list,
        num_class_labels=4,
        num_class_hidden_layer=0,
        num_token_hidden_layer=0,
    ):
        super().__init__()
        self.slot_list = slot_list
        self.bert = BertModel(bert_config)
        self.heads = nn.ModuleDict(
            {
                slot: BertDSTSlotHead(
                    bert_config.hidden_size,
                    num_class_labels,
                    num_class_hidden_layer,
                    num_token_hidden_layer,
                )
                for slot in slot_list
            }
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        class_output_layer = outputs.pooler_output
        token_output_layer = outputs.last_hidden_state

        per_slot_class_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        for slot in self.slot_list:
            class_logits, start_logits, end_logits = self.heads[slot](
                class_output_layer, token_output_layer
            )
            per_slot_class_logits[slot] = class_logits
            per_slot_start_logits[slot] = start_logits
            per_slot_end_logits[slot] = end_logits
        return per_slot_class_logits, per_slot_start_logits, per_slot_end_logits


MENAGERIE_ZOO = "vendored-pytorch"

_SLOT_LIST = ["area", "food", "price range"]


def build_bert_dst():
    bert_config = BertConfig(
        vocab_size=200,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    model = BertDST(bert_config, _SLOT_LIST, num_class_labels=4)
    model.eval()
    return model


def example_input_bert_dst():
    input_ids = torch.randint(0, 200, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    token_type_ids = torch.zeros(2, 16, dtype=torch.long)
    return (input_ids, attention_mask, token_type_ids)


MENAGERIE_ENTRIES = [
    ("BERT-DST", build_bert_dst, example_input_bert_dst, 2019, "vendored-pytorch"),
]
