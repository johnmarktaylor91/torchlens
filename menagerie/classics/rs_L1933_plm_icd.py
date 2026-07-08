# SOURCE: vendored from https://github.com/MiuLab/PLM-ICD @ master
# (src/modeling_bert.py, BertForMultilabelClassification)
"""PLM-ICD: automatic ICD coding via pretrained language models + LAAT attention head.

Vendored verbatim (imports/formatting only adjusted) from PLM-ICD's
``src/modeling_bert.py``. The class wraps a real ``transformers.BertModel`` with a
label-attention (LAAT) classification head operating over chunked long documents
(batch, num_chunks, chunk_size). This is a genuine architectural addition on top of
BertModel (chunk reshaping + LAAT attention), so it is vendored rather than treated
as a bare library model.
"""

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss

from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput

MENAGERIE_ZOO = "vendored-pytorch"


class BertForMultilabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model_mode = config.model_mode

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if "cls" in self.model_mode:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        elif "laat" in self.model_mode:
            self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
            self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.bert(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size),
            token_type_ids=token_type_ids.view(-1, chunk_size),
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if "cls" in self.model_mode:
            pooled_output = outputs[1].view(batch_size, num_chunks, -1)
            if self.model_mode == "cls-sum":
                pooled_output = pooled_output.sum(dim=1)
            elif self.model_mode == "cls-max":
                pooled_output = pooled_output.max(dim=1).values
            else:
                raise ValueError(f"model_mode {self.model_mode} not recognized")
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        elif "laat" in self.model_mode:
            if self.model_mode == "laat":
                hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
            elif self.model_mode == "laat-split":
                hidden_output = outputs[0].view(batch_size * num_chunks, chunk_size, -1)
            weights = torch.tanh(self.first_linear(hidden_output))
            att_weights = self.second_linear(weights)
            att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
            weighted_output = att_weights @ hidden_output
            logits = (
                self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)
            )
            if self.model_mode == "laat-split":
                logits = logits.view(batch_size, num_chunks, -1).max(dim=1).values
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        loss = None
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def build_plm_icd():
    config = BertConfig(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=20,
    )
    config.model_mode = "laat"
    return BertForMultilabelClassification(config)


def example_input_plm_icd():
    batch_size, num_chunks, chunk_size = 1, 2, 16
    input_ids = torch.randint(0, 256, (batch_size, num_chunks, chunk_size))
    attention_mask = torch.ones(batch_size, num_chunks, chunk_size, dtype=torch.long)
    token_type_ids = torch.zeros(batch_size, num_chunks, chunk_size, dtype=torch.long)
    return (input_ids, attention_mask, token_type_ids)


MENAGERIE_ENTRIES = [
    ("PLM-ICD", build_plm_icd, example_input_plm_icd, 2022, "vendored-pytorch"),
]
