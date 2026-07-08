# SOURCE: vendored from facebookresearch/ketod @ fc7182c5c62eb5033901e64550d8b4b1aaa3d8df
# https://raw.githubusercontent.com/facebookresearch/ketod/fc7182c5c62eb5033901e64550d8b4b1aaa3d8df/code/kg_selection/Model.py
#
# Chen et al. 2022 "KETOD: Knowledge-Enriched Task-Oriented Dialogue" (NAACL 2022
# Findings). KETOD couples a standard `GPT2LMHeadModel` generator (`code/simpletodplus`,
# an unmodified transformers class -- rung-1 territory) with a real added architecture
# piece: `Bert_model`, the "knowledge-selection" classifier that decides, at each dialog
# turn, whether to inject a retrieved knowledge snippet into the response (the paper's
# core knowledge-enrichment-decision contribution). It wraps a real
# `transformers.BertModel`/`RobertaModel` backbone with a genuinely new 2-layer
# classification head (`cls_prj` Linear -> Dropout -> `cls_final` Linear over the
# pooled [CLS] hidden state) absent from the base HF class -- a real, if small, added
# module, which is why this is vendored (rung 2) rather than treated as an unmodified
# rung-1 `BertModel`. Real code taken verbatim (only reformatted; no computation
# changed).
#
# Minimal, non-architectural changes made:
#   - The real code reads `conf.pretrained_model`/`conf.model_size`/`conf.cache_dir`
#     from `code/kg_selection/config.py` (an argparse-free but path-hardcoded module
#     that defaults to downloading `bert-base-cased` from the HF hub and references
#     `/data/users/...` cache paths). Replaced with a tiny in-memory `BertConfig`
#     (random init, no download) constructed directly, keeping the real
#     `Bert_model.__init__`/`forward` code exactly as written -- a config-loading
#     concern, not an architecture change.

import torch
from torch import nn
from transformers import BertConfig, BertModel


class Bert_model(nn.Module):
    def __init__(self, hidden_size, dropout_rate, bert_config):
        super(Bert_model, self).__init__()

        self.hidden_size = hidden_size

        self.bert = BertModel(bert_config)

        self.cls_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.cls_dropout = nn.Dropout(dropout_rate)

        self.cls_final = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, is_training, input_ids, input_mask, segment_ids, device):
        bert_outputs = self.bert(
            input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids
        )

        bert_sequence_output = bert_outputs.last_hidden_state

        bert_pooled_output = bert_sequence_output[:, 0, :]

        pooled_output = self.cls_prj(bert_pooled_output)
        pooled_output = self.cls_dropout(pooled_output)

        logits = self.cls_final(pooled_output)

        return logits


def build_ketod_bert_model():
    hidden_size = 32
    bert_config = BertConfig(
        vocab_size=200,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=2,
    )
    model = Bert_model(hidden_size=hidden_size, dropout_rate=0.0, bert_config=bert_config)
    model.eval()
    return model


def example_input_ketod_bert_model():
    batch_size, seq_len = 2, 12
    input_ids = torch.randint(2, 200, (batch_size, seq_len))
    input_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    return (False, input_ids, input_mask, segment_ids, torch.device("cpu"))


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "KETOD (Knowledge-Enhanced TOD, knowledge-selection classifier)",
        "build_ketod_bert_model",
        "example_input_ketod_bert_model",
        2022,
        "vendored",
    ),
]
