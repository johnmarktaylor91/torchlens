# FAITHFUL PORT of deepmedicine/BEHRT @ master (original dependency: pytorch_pretrained_bert)
#
# Source model code: https://github.com/deepmedicine/BEHRT/blob/master/model/MLM.py
# Source config (task/MLM.ipynb `model_config`):
#   https://github.com/deepmedicine/BEHRT/blob/master/task/MLM.ipynb
#
# BEHRT (Li et al. 2020, "BEHRT: Transformer for Electronic Health Records")
# applies a BERT-style transformer encoder to structured EHR event sequences,
# with a custom embedding layer that additively combines FOUR embeddings per
# token (not vanilla BERT's word+segment+position): disease/event code, visit
# segment, patient AGE-at-visit, and a fixed sinusoidal position embedding.
# This is an architectural modification over stock BertModel, so it is not a
# rung-1 "real library class" case.
#
# The real `model/MLM.py` imports `pytorch_pretrained_bert as Bert` and builds
# on `Bert.modeling.BertPreTrainedModel` / `BertEncoder` / `BertPooler` /
# `BertLayerNorm` / `BertOnlyMLMHead`. `pytorch_pretrained_bert` is an archived
# package not in this repo's base-lib env, so this is transcribed as a faithful
# port: BEHRT's own `BertEmbeddings`/`BertModel`/`BertForMaskedLM` classes are
# reproduced verbatim (same forward-pass math), rewired onto the modern
# `transformers` library's equivalent internal building blocks
# (`transformers.models.bert.modeling_bert.BertEncoder`/`BertPooler`/
# `BertOnlyMLMHead`), which are drop-in architectural equivalents of the
# original `pytorch_pretrained_bert.modeling` classes BEHRT depended on.
#
# Config values (vocab_size substituted with a small placeholder; everything
# else verbatim) are taken from the paper's own published `model_config` in
# `task/MLM.ipynb`:
#   hidden_size=288, seg_vocab_size=2, num_hidden_layers=6,
#   num_attention_heads=12, intermediate_size=512, hidden_act='gelu',
#   hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
#   initializer_range=0.02, max_position_embedding=max_len_seq(=64 in paper).

import numpy as np
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder, BertOnlyMLMHead, BertPooler
from transformers.models.bert.configuration_bert import BertConfig as _HFBertConfig


class BehrtConfig(_HFBertConfig):
    """Port of BEHRT's `BertConfig(Bert.modeling.BertConfig)` wrapper from
    task/MLM.ipynb: same fields as vanilla BertConfig plus `seg_vocab_size`
    and `age_vocab_size` for the two extra embedding tables."""

    def __init__(self, config):
        super().__init__(
            vocab_size=config.get("vocab_size"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=config.get("num_hidden_layers"),
            num_attention_heads=config.get("num_attention_heads"),
            intermediate_size=config.get("intermediate_size"),
            hidden_act=config.get("hidden_act"),
            hidden_dropout_prob=config.get("hidden_dropout_prob"),
            attention_probs_dropout_prob=config.get("attention_probs_dropout_prob"),
            max_position_embeddings=config.get("max_position_embedding"),
            initializer_range=config.get("initializer_range"),
        )
        self.seg_vocab_size = config.get("seg_vocab_size")
        self.age_vocab_size = config.get("age_vocab_size")
        # menagerie: modern `transformers` resolves the attention backend via
        # `PreTrainedModel.__init__`; since BehrtEncoder wires `BertEncoder`
        # directly (no `PreTrainedModel` base), set the eager implementation
        # explicitly so `BertAttention` can look it up.
        self._attn_implementation = "eager"


class BertEmbeddings(nn.Module):
    """Port of BEHRT's `BertEmbeddings` (model/MLM.py): word + segment + age +
    fixed sinusoidal position embeddings, summed then LayerNorm'd."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.seg_vocab_size, config.hidden_size)
        self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size)
        posi_table = self._init_posi_embedding(config.max_position_embeddings, config.hidden_size)
        self.posi_embeddings = nn.Embedding.from_pretrained(posi_table)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_ids, age_ids=None, seg_ids=None, posi_ids=None, age=True):
        if seg_ids is None:
            seg_ids = torch.zeros_like(word_ids)
        if age_ids is None:
            age_ids = torch.zeros_like(word_ids)
        if posi_ids is None:
            posi_ids = torch.zeros_like(word_ids)

        word_embed = self.word_embeddings(word_ids)
        segment_embed = self.segment_embeddings(seg_ids)
        age_embed = self.age_embeddings(age_ids)
        posi_embeddings = self.posi_embeddings(posi_ids)

        if age:
            embeddings = word_embed + segment_embed + age_embed + posi_embeddings
        else:
            embeddings = word_embed + segment_embed + posi_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    @staticmethod
    def _init_posi_embedding(max_position_embedding, hidden_size):
        def even_code(pos, idx):
            return np.sin(pos / (10000 ** (2 * idx / hidden_size)))

        def odd_code(pos, idx):
            return np.cos(pos / (10000 ** (2 * idx / hidden_size)))

        lookup_table = np.zeros((max_position_embedding, hidden_size), dtype=np.float32)
        for pos in range(max_position_embedding):
            for idx in np.arange(0, hidden_size, step=2):
                lookup_table[pos, idx] = even_code(pos, idx)
        for pos in range(max_position_embedding):
            for idx in np.arange(1, hidden_size, step=2):
                lookup_table[pos, idx] = odd_code(pos, idx)
        return torch.tensor(lookup_table, dtype=torch.float32)


class BehrtBertModel(nn.Module):
    """Port of BEHRT's `BertModel(Bert.modeling.BertPreTrainedModel)`
    (model/MLM.py): the four-way embedding above feeding a standard BERT
    transformer encoder + pooler."""

    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, age_ids, seg_ids, posi_ids)
        encoder_out = self.encoder(embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_out[0]
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class BehrtForMaskedLM(nn.Module):
    """Port of BEHRT's `BertForMaskedLM` (model/MLM.py): the MLM pretraining
    head used for the BEHRT masked-EHR-code pretraining task."""

    def __init__(self, config):
        super().__init__()
        self.bert = BehrtBertModel(config)
        self.cls = BertOnlyMLMHead(config)

    def forward(self, input_ids, age_ids=None, seg_ids=None, posi_ids=None, attention_mask=None):
        sequence_output, _ = self.bert(input_ids, age_ids, seg_ids, posi_ids, attention_mask)
        prediction_scores = self.cls(sequence_output)
        return prediction_scores


MENAGERIE_ZOO = "ported-pytorch"

_MODEL_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 288,
    "seg_vocab_size": 2,
    "age_vocab_size": 128,
    "max_position_embedding": 64,
    "hidden_dropout_prob": 0.1,
    "num_hidden_layers": 6,
    "num_attention_heads": 12,
    "attention_probs_dropout_prob": 0.1,
    "intermediate_size": 512,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
}
_SEQ_LEN = 16


def build_behrt():
    config = BehrtConfig(_MODEL_CONFIG)
    return BehrtForMaskedLM(config)


def example_input_behrt():
    input_ids = torch.randint(0, _MODEL_CONFIG["vocab_size"], (1, _SEQ_LEN))
    age_ids = torch.randint(0, _MODEL_CONFIG["age_vocab_size"], (1, _SEQ_LEN))
    seg_ids = torch.randint(0, _MODEL_CONFIG["seg_vocab_size"], (1, _SEQ_LEN))
    posi_ids = torch.arange(_SEQ_LEN).unsqueeze(0)
    attention_mask = torch.ones(1, _SEQ_LEN, dtype=torch.long)
    return (input_ids, age_ids, seg_ids, posi_ids, attention_mask)


MENAGERIE_ENTRIES = [
    (
        "BEHRT",
        build_behrt,
        example_input_behrt,
        2020,
        MENAGERIE_ZOO,
    ),
]
