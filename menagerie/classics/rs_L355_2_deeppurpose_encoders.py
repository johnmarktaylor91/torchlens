# SOURCE: vendored from kexinhuang12345/DeepPurpose @ master
# https://raw.githubusercontent.com/kexinhuang12345/DeepPurpose/master/DeepPurpose/encoders.py
# https://raw.githubusercontent.com/kexinhuang12345/DeepPurpose/master/DeepPurpose/model_helper.py
#
# Huang, Fu, Glass, Zitnik, Xiao, Sun 2020 (Bioinformatics) "DeepPurpose: a deep learning
# library for drug-target interaction prediction". `DeepPurpose.encoders.transformer` is the
# package's Transformer-encoder branch for drug SMILES / protein-sequence embedding: a
# learned token+position embedding (`Embeddings`) feeding a BERT-style multi-head
# self-attention encoder stack (`Encoder_MultipleLayers` of `Encoder` blocks built from
# `SelfAttention` -> `SelfOutput` -> `Intermediate` -> `Output`, each pre/post `LayerNorm`),
# pooling the CLS-position (`encoded_layers[:, 0]`) as the drug/protein embedding.
#
# `LayerNorm`, `Embeddings`, `SelfAttention`, `SelfOutput`, `Attention`, `Intermediate`,
# `Output`, `Encoder`, `Encoder_MultipleLayers` are copied verbatim from
# `DeepPurpose/model_helper.py` (pure torch/numpy/math/copy -- no DeepPurpose-internal
# dependency). `transformer` is copied verbatim from `DeepPurpose/encoders.py`, encoding
# fixed to `"drug"` (the `"protein"` branch is architecturally identical, only differing in
# `config` hyperparameter values). The only edit: `encoders.py` does
# `from DeepPurpose.utils import *` at module scope purely to bring `torch`/`np` names into
# scope for *other* classes in that file (`CNN`, `MPNN`, ...); `transformer.__init__`/
# `.forward` never call anything from `DeepPurpose.utils`, and importing the real
# `utils.py` pulls in `rdkit`/`subword_nmt`/`wget`, which are not installed and are
# unrelated to this encoder's architecture -- that unused wildcard import is dropped.
# `device` is resolved locally (`cuda` if available else `cpu`) exactly as the real module
# does at import time.

import copy
import math

import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- from DeepPurpose/model_helper.py (verbatim) ----


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings."""

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(
        self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob
    ):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super(Encoder, self).__init__()
        self.attention = Attention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob
        )
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(
        self,
        n_layer,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(
            hidden_size,
            intermediate_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


# ---- from DeepPurpose/encoders.py (verbatim, encoding="drug" specialization) ----


class transformer(nn.Sequential):
    def __init__(self, encoding, **config):
        super(transformer, self).__init__()
        if encoding == "drug":
            self.emb = Embeddings(
                config["input_dim_drug"],
                config["transformer_emb_size_drug"],
                50,
                config["transformer_dropout_rate"],
            )
            self.encoder = Encoder_MultipleLayers(
                config["transformer_n_layer_drug"],
                config["transformer_emb_size_drug"],
                config["transformer_intermediate_size_drug"],
                config["transformer_num_attention_heads_drug"],
                config["transformer_attention_probs_dropout"],
                config["transformer_hidden_dropout_rate"],
            )
        elif encoding == "protein":
            self.emb = Embeddings(
                config["input_dim_protein"],
                config["transformer_emb_size_target"],
                545,
                config["transformer_dropout_rate"],
            )
            self.encoder = Encoder_MultipleLayers(
                config["transformer_n_layer_target"],
                config["transformer_emb_size_target"],
                config["transformer_intermediate_size_target"],
                config["transformer_num_attention_heads_target"],
                config["transformer_attention_probs_dropout"],
                config["transformer_hidden_dropout_rate"],
            )

    def forward(self, v):
        e = v[0].long().to(device)
        e_mask = v[1].long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]


MENAGERIE_ZOO = "vendored-pytorch"

# Matches DeepPurpose.utils.generate_config's drug-transformer defaults, shrunk for a tiny
# traced example (real defaults: emb_size 128, intermediate 512, 8 layers, 8 heads).
_DRUG_TRANSFORMER_CONFIG = dict(
    input_dim_drug=2586,  # real ESPF drug-vocab size (DeepPurpose/ESPF/subword_units_map_chembl_freq_1500.csv)
    transformer_emb_size_drug=16,
    transformer_n_layer_drug=2,
    transformer_intermediate_size_drug=32,
    transformer_num_attention_heads_drug=2,
    transformer_dropout_rate=0.1,
    transformer_attention_probs_dropout=0.1,
    transformer_hidden_dropout_rate=0.1,
)


def build_deeppurpose_transformer():
    # DeepPurpose.DTI.model_initialize does `self.model = self.model.to(self.device)`
    # right after construction; the encoder's own forward() moves its *inputs* to
    # `device` but not its parameters, so the model must be placed on `device` too.
    model = transformer("drug", **_DRUG_TRANSFORMER_CONFIG).to(device)
    model.eval()
    return model


def example_input_deeppurpose_transformer():
    # matches DeepPurpose.utils.drug2emb_encoder's fixed max_d=50 sequence length
    seq_len = 50
    batch = 1
    token_ids = torch.randint(0, _DRUG_TRANSFORMER_CONFIG["input_dim_drug"], (batch, seq_len))
    mask = torch.ones(batch, seq_len)
    return ((token_ids, mask),)


MENAGERIE_ENTRIES = [
    (
        "DeepPurpose Transformer Encoder",
        build_deeppurpose_transformer,
        example_input_deeppurpose_transformer,
        2020,
        MENAGERIE_ZOO,
    ),
]
