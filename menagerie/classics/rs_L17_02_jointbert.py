# SOURCE: vendored from monologg/JointBERT @ master (model/modeling_jointbert.py, model/module.py)
#
# JointBERT (Chen, Zhuo, Wang, "BERT for Joint Intent Classification and
# Slot Filling", arXiv 2019; this canonical unofficial PyTorch
# implementation is the widely-cited reference, ACL-recognized). Real
# architecture: a stock `BertModel` encoder feeding TWO task heads off its
# two pooled representations -- an `IntentClassifier` (dropout + linear on
# the `[CLS]` pooled output, joint utterance-level intent softmax) and a
# `SlotClassifier` (dropout + linear on the full per-token sequence output,
# per-token BIO slot-tag softmax), trained jointly end-to-end. This adds a
# real architectural component (the dual-head joint classification design)
# on top of `BertModel`, so it is vendored (rung 2) rather than treated as
# a bare `BertModel` recipe.
#
# Vendoring notes (imports/config fixes only, architecture untouched):
#   - `from transformers.modeling_bert import BertPreTrainedModel,
#     BertModel, BertConfig` (an old sub-module import path removed in
#     modern `transformers`) is replaced with the equivalent modern
#     top-level import `from transformers import BertPreTrainedModel,
#     BertModel, BertConfig` -- same classes, same behavior.
#   - `from torchcrf import CRF` (the optional CRF slot-decoding layer,
#     gated behind `args.use_crf`) is dropped; `torchcrf` is a non-base
#     package not installed in this environment. The traced entry point
#     below sets `use_crf=False` (an existing, real, non-CRF execution
#     path already present in the original `forward()`'s
#     `if self.args.use_crf:` branch -- not a new branch added here), so
#     no architecture is invented; the CRF path is simply not exercised.
#   - `IntentClassifier`/`SlotClassifier` (`module.py`) and `JointBERT`
#     (`modeling_jointbert.py`) are copied verbatim (unchanged compute;
#     only whitespace/formatting cleanup and the CRF branch removed since
#     it is unreachable with `use_crf=False`).
#   - The `args` namespace (originally an argparse Namespace with
#     `dropout_rate`/`use_crf`/`ignore_index`/`slot_loss_coef`) is
#     replaced with a tiny local dataclass carrying the same fields, since
#     the real code only ever reads attributes off it.

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.0):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.0):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


@dataclass
class JointBERTArgs:
    dropout_rate: float = 0.1
    use_crf: bool = False
    ignore_index: int = 0
    slot_loss_coef: float = 1.0


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(
            config.hidden_size, self.num_intent_labels, args.dropout_rate
        )
        self.slot_classifier = SlotClassifier(
            config.hidden_size, self.num_slot_labels, args.dropout_rate
        )

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )
            total_loss += intent_loss

        # 2. Slot Softmax (use_crf=False path -- see vendoring notes)
        if slot_labels_ids is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss = slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss = slot_loss_fct(
                    slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1)
                )
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[
            2:
        ]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits


def build_jointbert():
    config = BertConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    args = JointBERTArgs()
    intent_label_lst = list(range(7))  # e.g. ATIS-style intent label set
    slot_label_lst = list(range(11))  # e.g. BIO slot label set
    model = JointBERT(config, args, intent_label_lst, slot_label_lst)
    model.eval()
    return model


def example_input_jointbert():
    torch.manual_seed(0)
    batch, seq_len = 2, 12
    input_ids = torch.randint(1, 200, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    token_type_ids = torch.zeros(batch, seq_len, dtype=torch.long)
    intent_label_ids = torch.randint(0, 7, (batch,))
    slot_labels_ids = torch.randint(0, 11, (batch, seq_len))
    return (input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "JointBERT (Joint Intent Classification + Slot Filling)",
        build_jointbert,
        example_input_jointbert,
        2019,
        "vendored-pytorch",
    ),
]
