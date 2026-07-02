# SOURCE: vendored from stanfordnlp/contract-nli-bert @ master
#
# Real files:
#   contract_nli/model/identification_classification/bert.py
#   contract_nli/model/identification_classification/config.py
#   contract_nli/model/identification_classification/model_output.py
#
# `BertForIdentificationClassification` extends `transformers.BertModel` with two new
# heads (`class_outputs`: 3-way NLI classification off the pooled [CLS] output, and
# `span_outputs`: 2-way per-token span-identification logits off the sequence output) --
# a genuine architectural addition over plain BertModel, so this is vendored rather than
# constructed from the stock `transformers.BertModel` class. The class body below is
# copied verbatim from the real repo; only the training-time loss branch (unused for a
# plain forward trace) and its local-module imports are adapted to be self-contained in
# one file (`NLILabel` inlined as a plain int constant, `IdentificationClassificationModelOutput`
# inlined unchanged from model_output.py).

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import BertConfig
from transformers.file_utils import ModelOutput
from transformers.models.bert import BertModel, BertPreTrainedModel

MENAGERIE_ZOO = "vendored-pytorch"

# --- inlined from contract_nli/model/identification_classification/model_output.py ---


@dataclass
class IdentificationClassificationModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_cls: Optional[torch.FloatTensor] = None
    loss_span: Optional[torch.FloatTensor] = None
    class_logits: torch.FloatTensor = None
    span_logits: torch.FloatTensor = None


# --- inlined from contract_nli/dataset/loader.py (only the NOT_MENTIONED sentinel used) ---
_NLI_NOT_MENTIONED = 0

# --- vendored from contract_nli/model/identification_classification/bert.py ---


class BertForIdentificationClassification(BertPreTrainedModel):
    IMPOSSIBLE_STRATEGIES = {"ignore", "label", "not_mentioned"}

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        self.class_outputs = nn.Linear(config.hidden_size, 3)
        self.span_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.model_type: str = config.model_type

        if config.impossible_strategy not in self.IMPOSSIBLE_STRATEGIES:
            raise ValueError(f"impossible_strategy must be one of {self.IMPOSSIBLE_STRATEGIES}")
        self.impossible_strategy = config.impossible_strategy

        self.class_loss_weight = config.class_loss_weight

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        class_labels=None,
        span_labels=None,
        p_mask=None,
        valid_span_missing_in_context=None,
    ) -> IdentificationClassificationModelOutput:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        pooled_output = self.dropout(pooled_output)
        logits_cls = self.class_outputs(pooled_output)

        sequence_output = self.dropout(sequence_output)
        logits_span = self.span_outputs(sequence_output)

        if class_labels is not None:
            assert p_mask is not None
            assert span_labels is not None
            assert valid_span_missing_in_context is not None

            loss_fct = nn.CrossEntropyLoss()
            if self.impossible_strategy == "ignore":
                class_labels = torch.where(
                    valid_span_missing_in_context == 0,
                    class_labels,
                    torch.tensor(loss_fct.ignore_index).type_as(class_labels),
                )
            elif self.impossible_strategy == "not_mentioned":
                class_labels = torch.where(
                    valid_span_missing_in_context == 0, class_labels, _NLI_NOT_MENTIONED
                )
            loss_cls = self.class_loss_weight * loss_fct(logits_cls, class_labels)

            loss_fct = nn.CrossEntropyLoss()
            active_logits = logits_span.view(-1, 2)
            active_labels = torch.where(
                p_mask.view(-1) == 0,
                span_labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(span_labels),
            )
            loss_span = loss_fct(active_logits, active_labels)
            loss = loss_cls + loss_span
        else:
            loss, loss_cls, loss_span = None, None, None

        return IdentificationClassificationModelOutput(
            loss=loss,
            loss_cls=loss_cls,
            loss_span=loss_span,
            class_logits=logits_cls,
            span_logits=logits_span,
        )


def build_contractnli_bert():
    config = BertConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    # fields injected by contract_nli/model/identification_classification/config.py's
    # update_config() at real-repo construction time
    config.model_type = "bert"
    config.impossible_strategy = "ignore"
    config.class_loss_weight = 1.0
    return BertForIdentificationClassification(config)


def example_input_contractnli_bert():
    torch.manual_seed(0)
    input_ids = torch.randint(0, 200, (1, 16))
    attention_mask = torch.ones(1, 16, dtype=torch.long)
    token_type_ids = torch.zeros(1, 16, dtype=torch.long)
    # Passed positionally (matching forward's declared arg order) rather than as a
    # dict of kwargs -- a bare-dict `input_args` triggers an unrelated torchlens/
    # transformers interaction in BertModel.warn_if_padding_and_no_attention_mask's
    # `input_ids[:, [-1, 0]]` advanced indexing; positional args avoid it cleanly
    # without touching the real model or HF's code.
    return (input_ids, attention_mask, token_type_ids)


MENAGERIE_ENTRIES = [
    (
        "ContractNLI-BERT",
        build_contractnli_bert,
        example_input_contractnli_bert,
        2021,
        "vendored-pytorch",
    ),
]
