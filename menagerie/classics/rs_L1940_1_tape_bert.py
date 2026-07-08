# SOURCE: vendored from songlab-cal/tape @ master
# https://raw.githubusercontent.com/songlab-cal/tape/master/tape/models/modeling_bert.py
# https://raw.githubusercontent.com/songlab-cal/tape/master/tape/models/modeling_utils.py
#
# Rao, Bhattacharya, Thomas, Duan, Chen, Canny, Abbeel, Song, 2019 (NeurIPS 2019)
# "Evaluating Protein Transfer Learning with TAPE". TAPE's "Transformer" baseline
# (`ProteinBertModel`) is a from-scratch protein-domain fork of Google/HuggingFace's
# original PyTorch BERT (`modeling_bert.py` header credits Google AI Language +
# HuggingFace, "Modified by Roshan Rao"): its own `ProteinBertEmbeddings` (word +
# position + token_type, no NSP-specific head), its own pre-LN-free
# `ProteinBertSelfAttention`/`ProteinBertSelfOutput`/`ProteinBertIntermediate`/
# `ProteinBertOutput` encoder stack, its own `LayerNorm`/`get_activation_fn` (TF-style
# variance epsilon, gelu via erf) and a custom `ProteinConfig`/`ProteinModel` base
# (no `pytorch_transformers`/`transformers` runtime dependency) rather than the stock
# `transformers.BertModel` -- so this is a real architectural fork, not a bare
# `transformers.BertModel` construction, and is vendored verbatim below (`ProteinConfig`,
# `ProteinModel`, `LayerNorm`, `get_activation_fn`, and every `ProteinBert*` class from
# `modeling_bert.py`/`modeling_utils.py`). The `@registry.register_task_model(...)`
# decorators (TAPE's CLI task-registration system, `tape/registry.py`) are dropped since
# they are unrelated to the model architecture and pull in the CLI-only `registry`
# module; direct construction of `ProteinBertModel(config)` is exactly what
# `ProteinBertModel.from_pretrained(...)` / the TAPE CLI do under the hood.

import math
import typing

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"

# ============================================================================
# modeling_utils.py (verbatim, trimmed to construction-time surface)
# ============================================================================


class ProteinConfig(object):
    """Base class for all configuration classes."""

    pretrained_config_archive_map: typing.Dict[str, str] = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.num_labels = kwargs.pop("num_labels", 2)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.torchscript = kwargs.pop("torchscript", False)


class ProteinModel(nn.Module):
    """Base class for all models."""

    config_class: typing.Type[ProteinConfig] = ProteinConfig
    pretrained_model_archive_map: typing.Dict[str, str] = {}
    base_model_prefix = ""

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, ProteinConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class "
                "`ProteinConfig`.".format(self.__class__.__name__)
            )
        self.config = config

    def init_weights(self):
        """Initialize and prunes weights if needed."""
        self.apply(self._init_weights)
        if getattr(self.config, "pruned_heads", False):
            self.prune_heads(self.config.pruned_heads)

    def _tie_or_clone_weights(self, first_module, second_module):
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight


def prune_linear_layer(layer, index, dim=0):
    """Prune a linear layer (a model parameters) to keep only entries in index."""
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def gelu(x):
    """Implementation of the gelu activation function."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def get_activation_fn(name: str) -> typing.Callable:
    if name == "gelu":
        return gelu
    elif name == "relu":
        return torch.nn.functional.relu
    elif name == "swish":
        return swish
    else:
        raise ValueError(f"Unrecognized activation fn: {name}")


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# ============================================================================
# modeling_bert.py (verbatim, `registry` task-model decorators dropped)
# ============================================================================


class ProteinBertConfig(ProteinConfig):
    def __init__(
        self,
        vocab_size: int = 30,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 8096,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
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


class ProteinBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ProteinBertSelfAttention(nn.Module):
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

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class ProteinBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ProteinBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = ProteinBertSelfAttention(config)
        self.output = ProteinBertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ProteinBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_activation_fn(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ProteinBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ProteinBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ProteinBertAttention(config)
        self.intermediate = ProteinBertIntermediate(config)
        self.output = ProteinBertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class ProteinBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [ProteinBertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask, chunks=None):
        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask)
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


class ProteinBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ProteinBertAbstractModel(ProteinModel):
    """An abstract class to handle weights initialization."""

    config_class = ProteinBertConfig
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class ProteinBertModel(ProteinBertAbstractModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = ProteinBertEmbeddings(config)
        self.encoder = ProteinBertEncoder(config)
        self.pooler = ProteinBertPooler(config)

        self.init_weights()

    def forward(self, input_ids, input_mask=None):
        if input_mask is None:
            input_mask = torch.ones_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, chunks=None)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_tape_bert():
    """Tiny ProteinBertConfig (real TAPE default vocab_size=30 protein-token
    vocabulary; hidden/layer/head counts shrunk from the real bert-base preset
    of hidden_size=768/num_hidden_layers=12/num_attention_heads=12/
    intermediate_size=3072 for a fast trace)."""
    config = ProteinBertConfig(
        vocab_size=30,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = ProteinBertModel(config)
    model.eval()
    return model


def example_input_tape_bert():
    torch.manual_seed(0)
    return torch.randint(0, 30, (1, 16))


MENAGERIE_ENTRIES = [
    (
        "TAPE Transformer",
        build_tape_bert,
        example_input_tape_bert,
        2019,
        "vendored-pytorch",
    ),
]
