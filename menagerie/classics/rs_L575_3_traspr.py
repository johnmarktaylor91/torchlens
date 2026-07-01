# SOURCE: vendored from biociphers/traspr @ 2a371817142f9a88af42ce73e5935a460d1fd0a7
#   (bitbucket.org/biociphers/traspr)
#   src/transformers/modeling_bert.py: BertLayerNorm, BertEmbeddings, BertSelfAttention,
#     BertSelfOutput, BertAttention, BertIntermediate, BertOutput, BertLayer, BertEncoder,
#     BertPooler, BertPreTrainedModel, BertModel, BertForSequenceMultiClassificationMultiTransformer
#   src/transformers/configuration_bert.py: BertConfig
#   src/transformers/configuration_utils.py: PretrainedConfig (trimmed: from_pretrained/
#     save_pretrained dropped -- not needed for fresh random-init construction)
#   src/transformers/modeling_utils.py: PreTrainedModel, ModuleUtilsMixin (trimmed:
#     from_pretrained/save_pretrained/resize_token_embeddings dropped, same reason)
#   src/transformers/activations.py: gelu, gelu_new, swish, ACT2FN
#   src/transformers/file_utils.py: add_start_docstrings, add_start_docstrings_to_callable
#     (the two pure-docstring decorators actually used by modeling_bert.py; the rest of
#     file_utils.py pulls in boto3/botocore/filelock for the HTTP model-hub cache, which
#     this fork's own eLife-paper training scripts (examples/finetune_psi.py) never need
#     for local random-init construction and this staged module does not exercise)
#
# TRASPr (Xiong-Hunt, Xiao, et al., eLife; biociphers/traspr) predicts pre-mRNA splicing
# percent-spliced-in (PSI) and body-of-splice-site (BOS) scores from surrounding genomic
# sequence. The paper's repo is a fork of the pre-refactor (2019-era) HuggingFace
# `transformers` package with an added DNA tokenizer (`tokenization_dna.py`) and, in
# `modeling_bert.py`, novel splice-prediction head classes on top of `BertModel`:
# `BertForSequenceMultiClassification`, `BertForSequenceMultiClassificationMultiTransformer`,
# and further `...Alt`/`...AltFull` variants (real usage: examples/finetune_psi.py
# `MODEL_CLASSES` dict, "dnamulti"/"dnamultitrans" keys). The core architectural
# contribution vendored here is `BertForSequenceMultiClassificationMultiTransformer`
# ("multi_tf_two_mlp" mode): it runs FOUR independent `BertModel` transformer encoders in
# parallel over four sequence inputs (real usage: 4 DNA windows around a splice site, each
# with its own attention_mask + `consval` per-base conservation-score channel that is
# embedded into `BertEmbeddings` alongside token/position embeddings -- this fork's one
# architectural change inside `BertModel` itself), concatenates the four `[CLS]` pooled
# outputs (+ optional hand-crafted `features` + optional `tissue_rep` "pokedex"
# representation), and passes the fused vector through Dense(hidden*4[+extra], hidden) ->
# LeakyReLU(0.1) -> Dropout -> Dense(hidden, num_labels) to predict PSI/BOS/dPSI labels.
# Every layer/mechanism is reproduced verbatim from the fork below (BertSelfAttention's
# unused-in-this-config `rope`/rotary-embedding branch included, since it's real code
# gated by `config.embedding_method`); the only changes are import-path fixes (relative
# `.foo` imports flattened into this single file) and dropping the from_pretrained/
# save_pretrained/HTTP-cache machinery that requires boto3/botocore/filelock (never
# invoked by fresh `Config(...)` + real-class `__init__` + `init_weights()` construction).

import logging
import math

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

logger = logging.getLogger(__name__)


# ---- src/transformers/activations.py (verbatim) ---------------------------------------
def swish(x):
    return x * torch.sigmoid(x)


def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


gelu = getattr(torch.nn.functional, "gelu", _gelu_python)


def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}


# ---- src/transformers/file_utils.py (the two docstring decorators actually used) ------
def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


# ---- src/transformers/configuration_utils.py: PretrainedConfig (trimmed) --------------
class PretrainedConfig(object):
    pretrained_config_archive_map = {}
    model_type = ""

    def __init__(self, **kwargs):
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_past = kwargs.pop("output_past", True)
        self.torchscript = kwargs.pop("torchscript", False)
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.id2label = {i: "LABEL_{}".format(i) for i in range(self.num_labels)}
        self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

        for key, value in kwargs.items():
            setattr(self, key, value)


# ---- src/transformers/configuration_bert.py: BertConfig (verbatim, minus pretrained map) ---
class BertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
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
        split=10,
        num_rnn_layer=1,
        rnn_dropout=0.0,
        rnn_hidden=768,
        rnn="lstm",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
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
        self.split = split
        self.num_rnn_layer = num_rnn_layer
        self.rnn = rnn
        self.rnn_dropout = rnn_dropout
        self.rnn_hidden = rnn_hidden


# ---- src/transformers/modeling_utils.py: ModuleUtilsMixin, PreTrainedModel (trimmed) ---
class ModuleUtilsMixin:
    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask


class PreTrainedModel(nn.Module, ModuleUtilsMixin):
    config_class = None
    base_model_prefix = ""

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    def get_output_embeddings(self):
        return None

    def tie_weights(self):
        output_embeddings = self.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.weight = self.get_input_embeddings().weight

    def init_weights(self):
        self.apply(self._init_weights)
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)
        self.tie_weights()


# ---- src/transformers/modeling_bert.py (verbatim except relative-import flattening) ----
BertLayerNorm = torch.nn.LayerNorm


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out):
        import numpy as np

        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape, past_key_values_length=0):
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position, token_type, and (fork-specific) consval embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.embedding_method = config.embedding_method
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.has_consval = config.consval
        self.hidden_size = config.hidden_size
        if config.consval:
            self.consval_embeddings = nn.Embedding(11, config.hidden_size, padding_idx=0)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        consval=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.has_consval:
            consval_embeddings = self.consval_embeddings(consval)
            embeddings += consval_embeddings
        if self.embedding_method is None or self.embedding_method.lower() != "rope":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.embedding_method = config.embedding_method

        if self.embedding_method is not None and self.embedding_method.lower() == "rope":
            self.embed_positions = SinusoidalPositionalEmbedding(
                config.max_position_embeddings, config.hidden_size // config.num_attention_heads
            )
            self.rotary_value = config.rotary_value

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.embedding_method is not None and self.embedding_method.lower() == "rope":
            sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1], 0)[None, None, :, :]
            if self.rotary_value:
                query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, key_layer, value_layer
                )
            else:
                query_layer, key_layer = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_layer, key_layer
                )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        rotate_half_query_layer = torch.stack(
            [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
        ).reshape_as(query_layer)
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        rotate_half_key_layer = torch.stack(
            [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
        ).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            rotate_half_value_layer = torch.stack(
                [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
            ).reshape_as(value_layer)
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization."""

    config_class = BertConfig
    base_model_prefix = "bert"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for param in module.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.xavier_normal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """The bare Bert Model transformer outputting raw hidden-states, with the fork's `consval` conservation channel."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        consval=None,
        tissue_rep=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            consval=consval,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs


class BertForSequenceMultiClassificationMultiTransformer(BertPreTrainedModel):
    """4-transformer PSI/BOS splicing-prediction head (real class: src/transformers/modeling_bert.py)."""

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.relu = nn.LeakyReLU(0.1)
        self.tf_mode = config.tf_mode
        self.multi_weight_dpsi = config.multi_weight_dpsi
        self.use_features = config.use_features
        self.pokedex_rep = config.pokedex_rep
        if self.use_features:
            self.feature_length = 8
        else:
            self.feature_length = 0
        if self.pokedex_rep:
            self.feature_length += 100

        if not config.tf_mode or config.tf_mode == "multi_tf_two_mlp":
            self.bert = nn.ModuleList([BertModel(config) for _ in range(4)])
            self.fc1 = nn.Linear(config.hidden_size * 4 + self.feature_length, config.hidden_size)
            self.fc2 = nn.Linear(config.hidden_size, self.num_labels)
        elif config.tf_mode == "multi_tf_multiFC":
            self.bert = nn.ModuleList([BertModel(config) for _ in range(4)])
            self.fcs = nn.ModuleList([nn.Linear(config.hidden_size, 200) for _ in range(4)])
            self.fc1 = nn.Linear(200 * 4, 200)
            self.fc2 = nn.Linear(200, self.num_labels)
        else:
            raise ValueError(
                "tf_mode can only be multi_tf_two_mlp or multi_tf_multiFC for MultiTransformer"
            )

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
        consval=None,
        features=None,
        tissue_rep=None,
    ):
        output = []
        input_ids = torch.transpose(input_ids, 0, 1)
        attention_mask = torch.transpose(attention_mask, 0, 1)
        consval = torch.transpose(consval, 0, 1)

        if not self.tf_mode or self.tf_mode == "multi_tf_two_mlp":
            if inputs_embeds is None:
                for i, layer_model in enumerate(self.bert):
                    output.append(
                        layer_model(
                            input_ids[i],
                            attention_mask=attention_mask[i],
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            consval=consval[i],
                        )[1]
                    )
                if self.use_features:
                    output.append(features)
                if self.pokedex_rep:
                    output.append(tissue_rep)
                pooled_output = torch.cat(output, dim=-1)
            else:
                pooled_output = inputs_embeds
            hidden = self.relu(self.fc1(self.dropout(pooled_output)))
            hidden = self.dropout(hidden)
            logits = self.fc2(hidden)

        elif self.tf_mode == "multi_tf_multiFC":
            for i, layer_model in enumerate(self.bert):
                temp_output = layer_model(
                    input_ids[i],
                    attention_mask=attention_mask[i],
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    consval=consval[i],
                )[1]
                temp_output = self.dropout(temp_output)
                temp_output = self.relu(self.fcs[i](temp_output))
                output.append(temp_output)
            pooled_output = torch.cat(output, dim=-1)
            pooled_output = self.dropout(pooled_output)
            hidden = self.relu(self.fc1(pooled_output))
            hidden = self.dropout(hidden)
            logits = self.fc2(hidden)

        outputs = (logits, pooled_output)
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels == 3 or self.num_labels == 2:
                if self.multi_weight_dpsi and self.multi_weight_dpsi != 1:
                    zero_weight = 0.3
                    weight = labels.clone().detach()
                    weight[:, 1:] = torch.max(weight[:, 1:], dim=1, keepdim=True)[0]
                    weight[:, 0] -= zero_weight
                    weight = (weight + zero_weight).to(logits.device).view(-1)
                    loss_fct = BCEWithLogitsLoss(weight=weight)
                else:
                    loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs


def build_traspr():
    # Real usage (examples/finetune_psi.py "dnamultitrans" MODEL_CLASSES entry): 4-mer DNA
    # vocab (4^4+special tokens ~ 69, real DNATokenizer vocab), small BERT config, PSI+BOS
    # multi-label regression (num_labels=3 in the real dPSI/PSI/BOS setup). Shrunk hidden
    # size/layers/heads for tiny tracing; multi-transformer fusion + consval embedding
    # architecture kept exactly as in the real class.
    config = BertConfig(
        vocab_size=69,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=3,
        tf_mode="multi_tf_two_mlp",
        multi_weight_dpsi=None,
        consval=True,
        use_features=True,
        pokedex_rep=False,
        embedding_method=None,
    )
    return BertForSequenceMultiClassificationMultiTransformer(config)


def example_input_traspr():
    # Real usage: 4 DNA sequence windows (each tokenized to input_ids/attention_mask) around
    # a splice site, each with a per-base `consval` conservation-score bucket (0-10) and a
    # shared 8-dim hand-crafted `features` vector (real feature_length=8, use_features=True).
    # Returned as a positional tuple matching forward()'s
    # (input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
    # labels, consval, features) argument order, so a plain `model(*example_input())` call
    # (as used by the tiny-tensor recipe/module trace harness) routes correctly.
    batch, n_transformers, seq_len = 2, 4, 12
    input_ids = torch.randint(0, 69, (batch, n_transformers, seq_len))
    attention_mask = torch.ones(batch, n_transformers, seq_len)
    consval = torch.randint(0, 11, (batch, n_transformers, seq_len))
    features = torch.rand(batch, 8)
    return (input_ids, attention_mask, None, None, None, None, None, consval, features)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("TRASPr", "build_traspr", "example_input_traspr", 2024, MENAGERIE_ZOO),
]
