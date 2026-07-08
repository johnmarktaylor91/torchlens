# SOURCE: vendored from smartyfh/DST-STAR @ main (commit 0ce04f5b77a8c853c5de625a1efb755f1854c1e4)
# https://raw.githubusercontent.com/smartyfh/DST-STAR/main/models/ModelBERT.py
#
# Feng, Wang & Chen 2021 (WWW 2021) "Sequential Slot Attention with State Tracking
# for Multi-Domain Dialogue State Tracking" (STAR) -- a BERT-encoder dialogue state
# tracker with a slot-attention decoder: `UtteranceEncoding` wraps `BertModel` as the
# turn/context encoder, `UtteranceAttention`/`MultiHeadAttention` cross-attend each
# slot query against the encoded utterance, `SlotSelfAttention` (a stack of
# `SlotAttentionLayer`s, each self-attn + `PositionwiseFeedForward` with residual
# `SublayerConnection`) lets slot representations exchange information with each
# other (the paper's "sequential slot self-attention" for inter-slot correlation),
# and `Decoder.slot_value_matching` scores each slot's hidden state against a fixed
# value-embedding lookup table via cosine/euclidean distance to pick a value.
# `BeliefTracker` composes `UtteranceEncoding` (BERT) + `Decoder` (the slot-attention
# stack above) end to end.
#
# `UtteranceEncoding`, `MultiHeadAttention`, `MultiHeadAttentionTanh`, `clones`,
# `SlotSelfAttention`, `SublayerConnection`, `SlotAttentionLayer`,
# `PositionwiseFeedForward`, `UtteranceAttention`, `Decoder`, `BeliefTracker` are
# copied verbatim (module bodies unchanged) from `models/ModelBERT.py`. No
# architectural code was rewritten; only these mechanical, import-isolation changes
# were made:
#   - The real training script builds `args` via argparse and `slot_lookup`/
#     `value_lookup`/`num_labels`/`slot_value_pos` by tokenizing a domain-ontology
#     file (`data/mwz2.4/ontology.json`) with the BERT tokenizer at startup. That is
#     replaced here by a tiny, fully-synthetic ontology built directly with a random
#     `nn.Embedding` for both lookups (`_build_lookups()`) -- same shapes/semantics
#     the model code reads (`slot_lookup.weight`, `value_lookup.weight`,
#     `slot_value_pos[slot_id] -> (start, end)` row ranges into `value_lookup`), just
#     supplied directly instead of via ontology-file tokenization.
#   - `args` is a plain `SimpleNamespace` carrying exactly the fields
#     `BeliefTracker`/`Decoder` read (`pretrained_model` [unused when constructing
#     `UtteranceEncoding` via `BertConfig` random init instead of
#     `from_pretrained`], `attn_head`, `dropout_prob`, `attn_type`,
#     `num_self_attention_layer`, `distance_metric`) -- same values/semantics as the
#     upstream argparse defaults (`attn_head=4`, `attn_type="softmax"`,
#     `num_self_attention_layer=6`, `distance_metric="euclidean"`), just supplied
#     directly instead of via CLI flags.
#   - `UtteranceEncoding.from_pretrained(self.args.pretrained_model)` (which
#     downloads a real BERT checkpoint) is swapped for
#     `UtteranceEncoding(BertConfig(...tiny...))` (random init) so the module is
#     runnable offline at tiny size; `UtteranceEncoding` itself (the class body) is
#     unchanged -- it is still the exact same `BertPreTrainedModel` wrapper around
#     `BertModel`.

import math
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, BertPreTrainedModel


class UtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(UtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        output_attentions=False,
        output_hidden_states=False,
    ):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


class MultiHeadAttentionTanh(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.tanh(scores)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, 0.0)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class SlotSelfAttention(nn.Module):
    "A stack of N layers"

    def __init__(self, layer, N):
        super(SlotSelfAttention, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x = self.norm(x)
        return x + self.dropout(sublayer(x))


class SlotAttentionLayer(nn.Module):
    "SlotAttentionLayer is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(SlotAttentionLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.ReLU()  # use gelu or relu

    def forward(self, x):
        return self.w_2(self.dropout(self.gelu(self.w_1(x))))


class UtteranceAttention(nn.Module):
    def __init__(self, attn_head, model_output_dim, dropout=0.0, attn_type="softmax"):
        super(UtteranceAttention, self).__init__()
        self.attn_head = attn_head
        self.model_output_dim = model_output_dim
        self.dropout = dropout
        if attn_type == "tanh":
            self.attn_fun = MultiHeadAttentionTanh(
                self.attn_head, self.model_output_dim, dropout=0.0
            )
        else:
            self.attn_fun = MultiHeadAttention(self.attn_head, self.model_output_dim, dropout=0.0)

    def forward(self, query, value, attention_mask=None):
        num_query = query.size(0)
        batch_size = value.size(0)
        seq_length = value.size(1)

        expanded_query = query.unsqueeze(0).expand(batch_size, *query.shape)
        if attention_mask is not None:
            expanded_attention_mask = (
                attention_mask.view(-1, seq_length, 1).expand(value.size()).float()
            )
            new_value = torch.mul(value, expanded_attention_mask)
            attn_mask = attention_mask.unsqueeze(1).expand(batch_size, num_query, seq_length)
        else:
            new_value = value
            attn_mask = None

        attended_embedding = self.attn_fun(expanded_query, new_value, new_value, mask=attn_mask)

        return attended_embedding


class Decoder(nn.Module):
    def __init__(self, args, model_output_dim, num_labels, slot_value_pos, device):
        super(Decoder, self).__init__()
        self.model_output_dim = model_output_dim
        self.num_slots = len(num_labels)
        self.num_total_labels = sum(num_labels)
        self.num_labels = num_labels
        self.slot_value_pos = slot_value_pos
        self.attn_head = args.attn_head
        self.device = device
        self.args = args
        self.dropout_prob = self.args.dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)
        self.attn_type = self.args.attn_type

        ### slot utterance attention
        self.slot_utter_attn = UtteranceAttention(
            self.attn_head, self.model_output_dim, dropout=0.0, attn_type=self.attn_type
        )

        ### MLP
        self.SlotMLP = nn.Sequential(
            nn.Linear(self.model_output_dim * 2, self.model_output_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.model_output_dim, self.model_output_dim),
        )

        ### basic modues, attention dropout is 0.1 by default
        attn = MultiHeadAttention(self.attn_head, self.model_output_dim)
        ffn = PositionwiseFeedForward(
            self.model_output_dim, self.model_output_dim, self.dropout_prob
        )

        ### attention layer, multiple self attention layers
        self.slot_self_attn = SlotSelfAttention(
            SlotAttentionLayer(
                self.model_output_dim, deepcopy(attn), deepcopy(ffn), self.dropout_prob
            ),
            self.args.num_self_attention_layer,
        )

        ### prediction
        self.pred = nn.Sequential(
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(self.model_output_dim, self.model_output_dim),
            nn.LayerNorm(self.model_output_dim),
        )

        ### measure
        self.distance_metric = args.distance_metric
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        self.nll = CrossEntropyLoss(ignore_index=-1)

    def slot_value_matching(self, value_lookup, hidden, target_slots, labels):
        loss = 0.0
        loss_slot = []
        pred_slot = []

        batch_size = hidden.size(0)
        value_emb = value_lookup.weight[0 : self.num_total_labels, :]

        for s, slot_id in enumerate(target_slots):  # note: target_slots are successive
            hidden_label = value_emb[
                self.slot_value_pos[slot_id][0] : self.slot_value_pos[slot_id][1], :
            ]
            num_slot_labels = hidden_label.size(0)  # number of value choices for each slot

            _hidden_label = (
                hidden_label.unsqueeze(0)
                .repeat(batch_size, 1, 1)
                .reshape(batch_size * num_slot_labels, -1)
            )
            _hidden = (
                hidden[:, s, :]
                .unsqueeze(1)
                .repeat(1, num_slot_labels, 1)
                .reshape(batch_size * num_slot_labels, -1)
            )
            _dist = self.metric(_hidden_label, _hidden).view(batch_size, num_slot_labels)

            if self.distance_metric == "euclidean":
                _dist = -_dist

            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(batch_size, 1))

            _loss = self.nll(_dist, labels[:, s])

            loss += _loss
            loss_slot.append(_loss.item())

        pred_slot = torch.cat(pred_slot, 1)  # [batch_size, num_slots]

        return loss, loss_slot, pred_slot

    def forward(
        self, sequence_output, attention_mask, labels, slot_lookup, value_lookup, eval_type="train"
    ):
        batch_size = sequence_output.size(0)
        target_slots = list(range(0, self.num_slots))

        # slot utterance attention
        slot_embedding = slot_lookup.weight[target_slots, :]  # select target slots' embeddings
        slot_utter_emb = self.slot_utter_attn(slot_embedding, sequence_output, attention_mask)

        # concatenate with slot_embedding
        slot_utter_embedding = torch.cat(
            (slot_embedding.unsqueeze(0).repeat(batch_size, 1, 1), slot_utter_emb), 2
        )

        # MLP
        slot_utter_embedding2 = self.SlotMLP(slot_utter_embedding)

        # slot self attention
        hidden_slot = self.slot_self_attn(slot_utter_embedding2)

        # prediction
        hidden = self.pred(hidden_slot)

        # slot value matching
        loss, loss_slot, pred_slot = self.slot_value_matching(
            value_lookup, hidden, target_slots, labels
        )

        return loss, loss_slot, pred_slot


class BeliefTracker(nn.Module):
    def __init__(self, args, slot_lookup, value_lookup, num_labels, slot_value_pos, device):
        super(BeliefTracker, self).__init__()

        self.num_slots = len(num_labels)
        self.num_labels = num_labels
        self.slot_value_pos = slot_value_pos
        self.args = args
        self.device = device
        self.slot_lookup = slot_lookup
        self.value_lookup = value_lookup

        # NOTE: upstream calls `UtteranceEncoding.from_pretrained(args.pretrained_model)`
        # to load a real BERT checkpoint. We build the identical class body directly
        # from a tiny random-init `BertConfig` instead, so this runs offline.
        bert_config = BertConfig(
            vocab_size=200,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            type_vocab_size=2,
        )
        self.encoder = UtteranceEncoding(bert_config)
        self.model_output_dim = self.encoder.config.hidden_size
        self.decoder = Decoder(
            args, self.model_output_dim, self.num_labels, self.slot_value_pos, device
        )

    def forward(self, input_ids, attention_mask, token_type_ids, labels, eval_type="train"):
        batch_size = input_ids.size(0)
        num_slots = self.num_slots

        # encoder, a pretrained model, output is a tuple
        sequence_output = self.encoder(input_ids, attention_mask, token_type_ids)[0]

        # decoder
        loss, loss_slot, pred_slot = self.decoder(
            sequence_output, attention_mask, labels, self.slot_lookup, self.value_lookup, eval_type
        )

        # calculate accuracy
        accuracy = pred_slot == labels
        acc_slot = (
            torch.true_divide(torch.sum(accuracy, 0).float(), batch_size).cpu().detach().numpy()
        )  # slot accuracy
        acc = (
            torch.sum(torch.floor_divide(torch.sum(accuracy, 1), num_slots)).float().item()
            / batch_size
        )  # joint accuracy

        return loss, loss_slot, acc, acc_slot, pred_slot


def _build_args():
    return SimpleNamespace(
        pretrained_model="bert-base-uncased",  # unused: BeliefTracker builds a tiny random BertConfig instead
        attn_head=4,
        dropout_prob=0.0,
        attn_type="softmax",
        num_self_attention_layer=2,
        distance_metric="euclidean",
    )


def _build_lookups(num_slots=3, values_per_slot=4, hidden_size=32):
    slot_lookup = nn.Embedding(num_slots, hidden_size)
    total_values = num_slots * values_per_slot
    value_lookup = nn.Embedding(total_values, hidden_size)
    num_labels = [values_per_slot] * num_slots
    slot_value_pos = [(i * values_per_slot, (i + 1) * values_per_slot) for i in range(num_slots)]
    return slot_lookup, value_lookup, num_labels, slot_value_pos


def build_star_dst():
    args = _build_args()
    slot_lookup, value_lookup, num_labels, slot_value_pos = _build_lookups()
    device = torch.device("cpu")
    return BeliefTracker(args, slot_lookup, value_lookup, num_labels, slot_value_pos, device)


def example_input_star_dst():
    batch_size = 2
    seq_len = 12
    vocab_size = 200
    num_slots = 3
    values_per_slot = 4

    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int64)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.int64)
    labels = torch.randint(0, values_per_slot, (batch_size, num_slots))

    return (input_ids, attention_mask, token_type_ids, labels)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("STAR for DST", "build_star_dst", "example_input_star_dst", 2021, "vendored"),
]
