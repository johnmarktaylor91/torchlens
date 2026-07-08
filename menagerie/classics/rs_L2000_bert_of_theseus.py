# SOURCE: vendored from JetRunner/BERT-of-Theseus @ 09de324e3dd856405d21197587ae6638c246c62d
# (bert_of_theseus/modeling_bert_of_theseus.py) -- the novel BertEncoder module-replacement
# mechanism (Bernoulli-gated "Theseus" successor-layer swap) is preserved verbatim; only the
# sub-layer imports are updated from the pre-4.x `transformers.modeling_bert` internal layout
# (which no longer exists in modern transformers) to the current
# `transformers.models.bert.modeling_bert` module path. BertLayer's modern forward signature
# needs `_attn_implementation="eager"` and returns a plain tuple as before; call sites are
# adjusted minimally to match (no architectural change).
"""BERT-of-Theseus: progressive module replacement for BERT compression.

Xu, Canwen, et al. "BERT-of-Theseus: Compressing BERT by Progressive Module Replacing."
EMNLP 2020. The compressed "successor" model (`scc_layer`, `scc_n_layer` shallow BertLayers)
is trained by stochastically substituting predecessor (`layer`) blocks with successor blocks
via a per-module Bernoulli draw; at inference time only the successor stack runs.
"""

import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli

from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertLayer,
    BertPooler,
    BertPreTrainedModel,
)

MENAGERIE_ZOO = "vendored-pytorch"

__all__ = ["BertOfTheseusModel"]


class BertEncoder(nn.Module):
    def __init__(self, config, scc_n_layer=3):
        super(BertEncoder, self).__init__()
        self.prd_n_layer = config.num_hidden_layers
        self.scc_n_layer = scc_n_layer
        assert self.prd_n_layer % self.scc_n_layer == 0
        self.compress_ratio = self.prd_n_layer // self.scc_n_layer
        self.bernoulli = None
        self.layer = nn.ModuleList(
            [BertLayer(config, layer_idx=i) for i in range(self.prd_n_layer)]
        )
        self.scc_layer = nn.ModuleList(
            [BertLayer(config, layer_idx=i) for i in range(self.scc_n_layer)]
        )

    def set_replacing_rate(self, replacing_rate):
        if not 0 < replacing_rate <= 1:
            raise Exception("Replace rate must be in the range (0, 1]!")
        self.bernoulli = Bernoulli(torch.tensor([replacing_rate]))

    def forward(self, hidden_states, attention_mask=None):
        all_hidden_states = ()
        if self.training:
            inference_layers = []
            for i in range(self.scc_n_layer):
                if self.bernoulli.sample() == 1:  # REPLACE
                    inference_layers.append(self.scc_layer[i])
                else:  # KEEP the original
                    for offset in range(self.compress_ratio):
                        inference_layers.append(self.layer[i * self.compress_ratio + offset])
        else:  # inference with compressed model
            inference_layers = self.scc_layer

        for layer_module in inference_layers:
            all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

        all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = (hidden_states, all_hidden_states)
        return outputs  # last-layer hidden state, (all hidden states)


class BertOfTheseusModel(BertPreTrainedModel):
    """Predecessor+successor BERT encoder with progressive module replacement."""

    def __init__(self, config, scc_n_layer=3):
        super(BertOfTheseusModel, self).__init__(config)
        self.config = config
        config._attn_implementation = "eager"

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config, scc_n_layer=scc_n_layer)
        self.pooler = BertPooler(config)

        self.encoder.set_replacing_rate(0.5)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs


def _tiny_config():
    return BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=2,
    )


def build_bert_of_theseus_train():
    model = BertOfTheseusModel(_tiny_config(), scc_n_layer=3)
    model.train()
    return model


def example_input_bert_of_theseus_train():
    return torch.randint(0, 128, (2, 12))


def build_bert_of_theseus_compressed():
    model = BertOfTheseusModel(_tiny_config(), scc_n_layer=3)
    model.eval()
    return model


def example_input_bert_of_theseus_compressed():
    return torch.randint(0, 128, (2, 12))


MENAGERIE_ENTRIES = [
    (
        "BERT-of-Theseus (training, stochastic replace)",
        build_bert_of_theseus_train,
        example_input_bert_of_theseus_train,
        2020,
        "vendored",
    ),
    (
        "BERT-of-Theseus (compressed successor)",
        build_bert_of_theseus_compressed,
        example_input_bert_of_theseus_compressed,
        2020,
        "vendored",
    ),
]
