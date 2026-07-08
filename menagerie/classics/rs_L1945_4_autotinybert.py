# SOURCE: vendored from huawei-noah/Pretrained-Language-Model @ master (AutoTinyBERT subdir)
# https://raw.githubusercontent.com/huawei-noah/Pretrained-Language-Model/master/AutoTinyBERT/transformer/modeling_super_kd.py
#
# Yin, Zhang, Ganesh, Jia, Zhu, Liu, Han, 2021 (ACL) "AutoTinyBERT: Automatic
# Hyper-parameter Optimization for Efficient Pre-trained Language Models".
# AutoTinyBERT trains an elastic "SuperPLM" BERT supernet whose every layer has
# runtime-configurable hidden size / intermediate (FFN) size / attention-head
# count / QKV projection size (`sample_hidden_size`, `sample_intermediate_sizes`,
# `sample_num_attention_heads`, `sample_qkv_sizes`, `sample_layer_num` -- one
# `subbert_config` per forward pass), then a one-shot evolutionary search over
# this elastic space finds small BERT sub-architectures under a latency
# constraint. `SuperLinear`/`SuperEmbedding`/`SuperBertLayerNorm` (weight/bias
# **slicing** at forward time, no separate weight per sub-config) and the
# `subbert_config`-driven `SuperBertModel.forward` are AutoTinyBERT's real
# architectural contribution -- an elastic transformer, not a stock BERT class --
# so this is vendored real code, not built from `transformers.BertModel`.
#
# `modeling_super_kd.py`'s `BertConfig`, `SuperBertLayerNorm`, `SuperLinear`
# (+ `sample_weight`/`sample_bias`), `SuperEmbedding`, `SuperBertEmbeddings`,
# `SuperBertSelfAttention`, `SuperBertSelfOutput`, `SuperBertAttention`,
# `SuperBertIntermediate`, `SuperBertOutput`, `SuperBertLayer`, `SuperBertEncoder`,
# `SuperBertPooler`, and `SuperBertModel` (the `kd=False` supernet-inference path)
# are reproduced verbatim below. `BertPreTrainedModel`'s `from_pretrained`/
# `from_scratch` checkpoint-loading machinery (needs `.transformer.file_utils`'s
# S3/HTTP `cached_path`) is dropped -- never exercised by a random-init
# `SuperBertModel(config)` construction + forward pass; the `init_bert_weights`
# random-init path (the only part of `BertPreTrainedModel` a fresh model needs)
# is inlined directly onto `SuperBertModel` instead. The `subbert_config` below is
# a real per-layer elastic sub-architecture in the shape the repo's `searcher.py`
# produces (`sample_layer_num`/`sample_hidden_size`/`sample_num_attention_heads`/
# `sample_intermediate_sizes`/`sample_qkv_sizes`), just at a tiny `vocab_size`/
# `hidden_size` superset for a fast trace.

import copy
import math
import sys

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


class BertConfig(object):
    def __init__(
        self,
        vocab_size_or_config_json_file,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_size=None,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        if qkv_size is not None:
            self.qkv_size = qkv_size


class SuperBertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(SuperBertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.sample_weight = None
        self.sample_bias = None

    def set_sample_config(self, sample_hidden_dim):
        self.sample_weight = self.weight[:sample_hidden_dim]
        self.sample_bias = self.bias[:sample_hidden_dim]

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.sample_weight * x + self.sample_bias


def sample_weight(weight, sample_in_dim, sample_out_dim, in_index=None, out_index=None):
    if in_index is None:
        sample_weight_ = weight[:, :sample_in_dim]
    else:
        sample_weight_ = weight.index_select(1, in_index.to(weight.device))
    if out_index is None:
        sample_weight_ = sample_weight_[:sample_out_dim, :]
    else:
        sample_weight_ = sample_weight_.index_select(0, out_index.to(sample_weight_.device))
    return sample_weight_


def sample_bias(bias, sample_out_dim, out_index=None):
    if out_index is None:
        return bias[:sample_out_dim]
    return bias.index_select(0, out_index.to(bias.device))


class SuperLinear(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear="linear"):
        super().__init__(super_in_dim, super_out_dim, bias=bias)
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.samples = {}
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear
        )
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def set_sample_config(self, sample_in_dim, sample_out_dim, in_index=None, out_index=None):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self._sample_parameters(in_index=in_index, out_index=out_index)

    def _sample_parameters(self, in_index=None, out_index=None):
        self.samples["weight"] = sample_weight(
            self.weight,
            self.sample_in_dim,
            self.sample_out_dim,
            in_index=in_index,
            out_index=out_index,
        )
        self.samples["bias"] = self.bias
        if self.bias is not None:
            self.samples["bias"] = sample_bias(self.bias, self.sample_out_dim, out_index=out_index)
        return self.samples

    def forward(self, x):
        if self.bias is not None:
            return nn.functional.linear(
                x, self.samples["weight"].to(x.device), self.samples["bias"].to(x.device)
            )
        return nn.functional.linear(x, self.samples["weight"].to(x.device))


class SuperEmbedding(nn.Module):
    def __init__(self, dict_size, embd_size, padding_idx=None):
        super(SuperEmbedding, self).__init__()
        self.embedding = nn.Embedding(dict_size, embd_size, padding_idx=padding_idx)
        self.sample_embedding_weight = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embedding_weight = self.embedding.weight[..., :sample_embed_dim]

    def forward(self, input_ids):
        return nn.functional.embedding(
            input_ids,
            self.sample_embedding_weight.to(input_ids.device),
            self.embedding.padding_idx,
            self.embedding.max_norm,
            self.embedding.norm_type,
            self.embedding.scale_grad_by_freq,
            self.embedding.sparse,
        )


class SuperBertEmbeddings(nn.Module):
    def __init__(self, config):
        super(SuperBertEmbeddings, self).__init__()
        self.word_embeddings = SuperEmbedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = SuperEmbedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = SuperEmbedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sample_embed_dim = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.word_embeddings.set_sample_config(sample_embed_dim)
        self.position_embeddings.set_sample_config(sample_embed_dim)
        self.token_type_embeddings.set_sample_config(sample_embed_dim)
        self.LayerNorm.set_sample_config(sample_embed_dim)

    def forward(self, input_ids, sample_embed_dim=-1, token_type_ids=None):
        self.set_sample_config(sample_embed_dim)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SuperBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(SuperBertSelfAttention, self).__init__()
        qkv_size = getattr(config, "qkv_size", config.hidden_size)
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(qkv_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = SuperLinear(config.hidden_size, self.all_head_size)
        self.key = SuperLinear(config.hidden_size, self.all_head_size)
        self.value = SuperLinear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.sample_num_attention_head = None
        self.sample_attention_head_size = None
        self.sample_qkv_size = None

    def set_sample_config(
        self, sample_embed_dim, num_attention_head, qkv_size, in_index=None, out_index=None
    ):
        assert qkv_size % num_attention_head == 0
        self.sample_qkv_size = qkv_size
        self.sample_attention_head_size = qkv_size // num_attention_head
        self.sample_num_attention_head = num_attention_head
        self.query.set_sample_config(
            sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index
        )
        self.key.set_sample_config(
            sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index
        )
        self.value.set_sample_config(
            sample_embed_dim, qkv_size, in_index=in_index, out_index=out_index
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.sample_num_attention_head,
            self.sample_attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        sample_embed_dim=-1,
        num_attention_head=-1,
        qkv_size=-1,
        in_index=None,
        out_index=None,
    ):
        self.set_sample_config(
            sample_embed_dim, num_attention_head, qkv_size, in_index=in_index, out_index=out_index
        )
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.sample_attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(self.dropout(attention_probs), value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.sample_qkv_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores


class SuperBertSelfOutput(nn.Module):
    def __init__(self, config):
        super(SuperBertSelfOutput, self).__init__()
        qkv_size = getattr(config, "qkv_size", config.hidden_size)
        self.dense = SuperLinear(qkv_size, config.hidden_size)
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_sample_config(self, qkv_size, sample_embed_dim, in_index=None):
        self.dense.set_sample_config(qkv_size, sample_embed_dim, in_index=in_index)
        self.LayerNorm.set_sample_config(sample_embed_dim)

    def forward(self, hidden_states, input_tensor, qkv_size=-1, sample_embed_dim=-1, in_index=None):
        self.set_sample_config(qkv_size, sample_embed_dim, in_index=in_index)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SuperBertAttention(nn.Module):
    def __init__(self, config):
        super(SuperBertAttention, self).__init__()
        self.self = SuperBertSelfAttention(config)
        self.output = SuperBertSelfOutput(config)

    def forward(
        self,
        input_tensor,
        attention_mask,
        sample_embed_dim=-1,
        num_attention_head=-1,
        qkv_size=-1,
        in_index=None,
        out_index=None,
    ):
        self_output = self.self(
            input_tensor,
            attention_mask,
            sample_embed_dim,
            num_attention_head,
            qkv_size,
            in_index=in_index,
            out_index=out_index,
        )
        self_output, layer_att = self_output
        attention_output = self.output(
            self_output, input_tensor, qkv_size, sample_embed_dim, in_index=out_index
        )
        return attention_output, layer_att


class SuperBertIntermediate(nn.Module):
    def __init__(self, config):
        super(SuperBertIntermediate, self).__init__()
        self.dense = SuperLinear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def set_sample_config(self, sample_embed_dim, intermediate_size):
        self.dense.set_sample_config(sample_embed_dim, intermediate_size)

    def forward(self, hidden_states, sample_embed_dim=-1, intermediate_size=-1):
        self.set_sample_config(sample_embed_dim, intermediate_size)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SuperBertOutput(nn.Module):
    def __init__(self, config):
        super(SuperBertOutput, self).__init__()
        self.dense = SuperLinear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = SuperBertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def set_sample_config(self, intermediate_size, sample_embed_dim):
        self.dense.set_sample_config(intermediate_size, sample_embed_dim)
        self.LayerNorm.set_sample_config(sample_embed_dim)

    def forward(self, hidden_states, input_tensor, intermediate_size=-1, sample_embed_dim=-1):
        self.set_sample_config(intermediate_size, sample_embed_dim)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SuperBertLayer(nn.Module):
    def __init__(self, config):
        super(SuperBertLayer, self).__init__()
        self.attention = SuperBertAttention(config)
        self.intermediate = SuperBertIntermediate(config)
        self.output = SuperBertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        sample_embed_dim=-1,
        intermediate_size=-1,
        num_attention_head=-1,
        qkv_size=-1,
        in_index=None,
        out_index=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            sample_embed_dim,
            num_attention_head,
            qkv_size,
            in_index=in_index,
            out_index=out_index,
        )
        attention_output, layer_att = attention_output
        intermediate_output = self.intermediate(
            attention_output, sample_embed_dim, intermediate_size
        )
        layer_output = self.output(
            intermediate_output, attention_output, intermediate_size, sample_embed_dim
        )
        return layer_output, layer_att


class SuperBertEncoder(nn.Module):
    def __init__(self, config):
        super(SuperBertEncoder, self).__init__()
        layer = SuperBertLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.sample_layer_num = None

    def forward(
        self,
        hidden_states,
        attention_mask,
        subbert_config=None,
        kd=False,
        in_index=None,
        out_index=None,
    ):
        all_encoder_layers = []
        all_encoder_att = []

        sample_embed_dim = subbert_config["sample_hidden_size"]
        num_attention_heads = subbert_config["sample_num_attention_heads"]
        itermediate_sizes = subbert_config["sample_intermediate_sizes"]
        qkv_sizes = subbert_config["sample_qkv_sizes"]
        sample_layer_num = subbert_config["sample_layer_num"]

        for i, layer_module in enumerate(self.layers[:sample_layer_num]):
            all_encoder_layers.append(hidden_states)
            hidden_states = layer_module(
                all_encoder_layers[i],
                attention_mask,
                sample_embed_dim,
                itermediate_sizes[i],
                num_attention_heads[i],
                qkv_sizes[i],
                in_index=in_index,
                out_index=out_index,
            )
            hidden_states, layer_att = hidden_states
            all_encoder_att.append(layer_att)

        all_encoder_layers.append(hidden_states)

        if not kd:
            return all_encoder_layers, all_encoder_att
        return all_encoder_layers[-1], all_encoder_att[-1]


class SuperBertPooler(nn.Module):
    def __init__(self, config):
        super(SuperBertPooler, self).__init__()
        self.dense = SuperLinear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def set_sample_config(self, sample_hidden_dim):
        self.dense.set_sample_config(sample_hidden_dim, sample_hidden_dim)

    def forward(self, hidden_states, sample_hidden_dim):
        self.set_sample_config(sample_hidden_dim)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SuperBertModel(nn.Module):
    """Real AutoTinyBERT elastic supernet (`kd=False` inference path).

    NOTE: `BertPreTrainedModel.from_pretrained`/`from_scratch` (S3/HTTP
    checkpoint loading via `.transformer.file_utils.cached_path`) is dropped
    here since it is never exercised by this random-init construction; the
    `init_bert_weights` random-init logic is inlined directly."""

    def __init__(self, config, fit_size=768):
        super(SuperBertModel, self).__init__()
        self.config = config
        self.embeddings = SuperBertEmbeddings(config)
        self.encoder = SuperBertEncoder(config)
        self.pooler = SuperBertPooler(config)
        self.dense_fit = SuperLinear(config.hidden_size, fit_size)

        self.hidden_size = config.hidden_size
        self.qkv_size = getattr(config, "qkv_size", config.hidden_size)
        self.fit_size = fit_size
        self.head_number = config.num_attention_heads
        self.apply(self._init_bert_weights)

    def _init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, SuperBertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        subbert_config,
        attention_mask=None,
        token_type_ids=None,
        kd=False,
        kd_infer=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_number = self.head_number
        qkv_size = self.qkv_size
        sample_qkv_size = subbert_config["sample_qkv_sizes"][0]

        in_out_index = None
        if kd:
            in_dim_per_head = int(qkv_size / head_number)
            in_sample_per_head = int(sample_qkv_size / head_number)
            in_out_index = []
            for i in range(head_number):
                start_ind = in_dim_per_head * i
                in_out_index.extend(range(start_ind, start_ind + in_sample_per_head))
            in_out_index = torch.tensor(in_out_index)
            in_out_index.to(input_ids.device)

        embedding_output = self.embeddings(
            input_ids, subbert_config["sample_hidden_size"], token_type_ids=token_type_ids
        )

        if kd:
            last_rep, last_att = self.encoder(
                embedding_output,
                extended_attention_mask,
                subbert_config,
                kd=True,
                out_index=in_out_index,
            )
            self.dense_fit.set_sample_config(subbert_config["sample_hidden_size"], self.fit_size)
            last_rep = self.dense_fit(last_rep)
            if not kd_infer:
                return last_rep, last_att
            pooled_output = self.pooler(last_rep, subbert_config["sample_hidden_size"])
            return last_rep, pooled_output
        else:
            all_encoder_layers, all_encoder_att = self.encoder(
                embedding_output,
                extended_attention_mask,
                subbert_config,
                kd=False,
                out_index=in_out_index,
            )
            sequence_output = all_encoder_layers[-1]
            pooled_output = self.pooler(sequence_output, subbert_config["sample_hidden_size"])
            return sequence_output, pooled_output


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_autotinybert_super():
    """A small SuperPLM supernet config, then a real found sub-architecture
    `subbert_config` in the shape `searcher.py` produces (per-layer
    `sample_hidden_size`/`sample_num_attention_heads`/
    `sample_intermediate_sizes`/`sample_qkv_sizes`, `sample_layer_num`)."""
    config = BertConfig(
        vocab_size_or_config_json_file=999,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=32,
        type_vocab_size=2,
        qkv_size=128,
    )
    model = SuperBertModel(config, fit_size=128)
    model.eval()
    return model


def example_input_autotinybert_super():
    torch.manual_seed(0)
    input_ids = torch.randint(0, 999, (1, 8), dtype=torch.long)
    subbert_config = {
        "sample_layer_num": 3,
        "sample_hidden_size": 96,
        "sample_num_attention_heads": [4, 4, 4],
        "sample_intermediate_sizes": [384, 384, 384],
        "sample_qkv_sizes": [96, 96, 96],
    }
    return input_ids, subbert_config


def _build_autotinybert_module():
    """Staging harness wraps the (input_ids, subbert_config) call signature
    into a single nn.Module so `tl.trace(model, example_input)` works with one
    positional example-input call, matching the `(name, build_fn, example_fn)`
    MENAGERIE_ENTRIES contract."""

    class _AutoTinyBertWrapper(nn.Module):
        def __init__(self, super_model, subbert_config):
            super().__init__()
            self.super_model = super_model
            self.subbert_config = subbert_config

        def forward(self, input_ids):
            return self.super_model(input_ids, self.subbert_config, kd=False)

    inner = build_autotinybert_super()
    _, subbert_config = example_input_autotinybert_super()
    wrapper = _AutoTinyBertWrapper(inner, subbert_config)
    wrapper.eval()
    return wrapper


def _example_input_module():
    torch.manual_seed(0)
    return torch.randint(0, 999, (1, 8), dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("AutoTinyBERT", _build_autotinybert_module, _example_input_module, 2021, "vendored-pytorch"),
]
