# SOURCE: vendored from OATML-Markslab/Tranception @ main
# https://raw.githubusercontent.com/OATML-Markslab/Tranception/main/tranception/model_pytorch.py
# https://raw.githubusercontent.com/OATML-Markslab/Tranception/main/tranception/activations.py
# https://raw.githubusercontent.com/OATML-Markslab/Tranception/main/tranception/outputs.py
# https://raw.githubusercontent.com/OATML-Markslab/Tranception/main/tranception/config.py
#
# Notin, Dias, Frazer, Marchena-Hurtado, Gomez, Marks, Gal 2022 (ICML) "Tranception:
# protein fitness prediction with autoregressive transformers and inference-time
# retrieval". `TranceptionLMHeadModel` (`tranception/model_pytorch.py`) is the real
# PyTorch `nn.Module`: a GPT2-style autoregressive transformer (`TranceptionModel` +
# `TranceptionBlock`) with two architectural departures from stock GPT2 that are
# copied verbatim here: (1) `TranceptionBlockAttention` splits attention heads into
# four groups and applies grouped depthwise 1D convolutions (`SpatialDepthWiseConvolution`,
# causal-padded kernel sizes 1/3/5/7) to the query/key/value projections before the
# dot-product attention -- multi-scale local context mixed into the global-attention
# heads; (2) `TranceptionModel.position_embedding="grouped_alibi"` replaces learned
# positional embeddings with a grouped ALiBi linear-bias scheme (`get_slopes`, adapted
# from the original ALiBi codebase) baked into causal attention scores instead of the
# input embeddings. `TranceptionBlockMLP`/`TranceptionBlock` reuse GPT2's `Conv1D`
# feedforward shape (`transformers.modeling_utils.Conv1D`) unmodified. This trace
# exercises the plain autoregressive forward path only (`retrieval_aggregation_mode=None`,
# the real code's own no-op branch that skips MSA-retrieval log-prior fusion entirely
# and prints "Model only uses autoregressive inference" -- see `TranceptionLMHeadModel.__init__`);
# retrieval-augmented scoring (the "TranceptEVE"/EVE-prior fusion variant referenced in
# the same repo) only changes the loss/scoring computation at inference time via
# `retrieval_aggregation_mode="aggregate_substitution"|"aggregate_indel"` and MSA prior
# data files, not the traced forward network topology, so it is not a separate
# architecture and is not staged separately. No other architectural changes were made;
# only unused imports (pandas-based MSA/scoring utilities not needed for a bare forward
# pass) were dropped and config defaults were shrunk to tiny sizes for a fast trace.

import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2PreTrainedModel
from transformers.pytorch_utils import (
    Conv1D,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


class TranceptionConfig(GPT2Config):
    """
    Config subclass for Tranception model architecture.
    """

    def __init__(
        self,
        attention_mode="tranception",
        position_embedding="grouped_alibi",
        tokenizer=None,
        retrieval_aggregation_mode=None,
        retrieval_inference_weight=0.6,
        MSA_filename=None,
        MSA_weight_file_name=None,
        MSA_start=None,
        MSA_end=None,
        full_protein_length=None,
        clustal_omega_location=None,
        scoring_window=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_type = "tranception"
        self.attention_mode = attention_mode
        self.position_embedding = position_embedding
        self.tokenizer = tokenizer
        self.retrieval_aggregation_mode = retrieval_aggregation_mode
        self.retrieval_inference_weight = retrieval_inference_weight
        self.MSA_filename = MSA_filename
        self.MSA_weight_file_name = MSA_weight_file_name
        self.MSA_start = MSA_start
        self.MSA_end = MSA_end
        self.full_protein_length = full_protein_length
        self.clustal_omega_location = clustal_omega_location
        self.scoring_window = scoring_window


tranception_ACT2FN = {
    "relu": nn.functional.relu,
    "silu": nn.functional.silu,
    "swish": nn.functional.silu,
    "gelu": nn.functional.gelu,
    "tanh": torch.tanh,
    "gelu_new": lambda x: (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    ),
    "sigmoid": torch.sigmoid,
}


@dataclass
class TranceptionCausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    fused_shift_log_probas: Optional[torch.FloatTensor] = None


def get_slopes(n, mode="standard_alibi", verbose=False):
    """
    Function to compute the m constant for each attention head. Code has been adapted from
    the official ALiBi codebase at:
    https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py
    """

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if mode == "grouped_alibi":
        n = n // 4
    if math.log2(n).is_integer():
        result = get_slopes_power_of_2(n)
    else:
        # Workaround when the number of heads is not a power of 2
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        result = (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )
    if mode == "grouped_alibi":
        result = result * 4
        if verbose:
            print("ALiBi slopes: {}".format(result))
    return result


class SpatialDepthWiseConvolution(nn.Module):
    def __init__(self, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=(kernel_size,),
            padding=(kernel_size - 1,),
            groups=head_dim,
        )

    def forward(self, x: torch.Tensor):
        batch_size, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size * heads, head_dim, seq_len)
        x = self.conv(x)
        if self.kernel_size > 1:
            x = x[:, :, : -(self.kernel_size - 1)]
        x = x.view(batch_size, heads, head_dim, seq_len)
        x = x.permute(0, 1, 3, 2)
        return x


class TranceptionBlockAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, SDWC_kernel_size=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and "
                f"`num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

        self.attention_mode = config.attention_mode

        if self.attention_mode == "tranception":
            assert self.num_heads % 4 == 0, (
                "Invalid number of heads. Tranception requires the number of heads to be a multiple of 4."
            )
            self.num_heads_per_kernel_size = self.num_heads // 4
            self.query_depthwiseconv = nn.ModuleDict()
            self.key_depthwiseconv = nn.ModuleDict()
            self.value_depthwiseconv = nn.ModuleDict()
            for kernel_idx, kernel in enumerate([3, 5, 7]):
                self.query_depthwiseconv[str(kernel_idx)] = SpatialDepthWiseConvolution(
                    self.head_dim, kernel
                )
                self.key_depthwiseconv[str(kernel_idx)] = SpatialDepthWiseConvolution(
                    self.head_dim, kernel
                )
                self.value_depthwiseconv[str(kernel_idx)] = SpatialDepthWiseConvolution(
                    self.head_dim, kernel
                )

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_heads, self.head_dim, self.pruned_heads
        )
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, alibi_bias=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[
                :, :, key_length - query_length : key_length, :key_length
            ].bool()
            attn_weights = torch.where(
                causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
            )

        if alibi_bias is not None:
            attn_weights = attn_weights + alibi_bias[:, :, : attn_weights.size(-1)]

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        alibi_bias=None,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.attention_mode == "tranception":
            query_list = [query[:, : self.num_heads_per_kernel_size, :, :]]
            key_list = [key[:, : self.num_heads_per_kernel_size, :, :]]
            value_list = [value[:, : self.num_heads_per_kernel_size, :, :]]
            for kernel_idx in range(3):
                query_list.append(
                    self.query_depthwiseconv[str(kernel_idx)](
                        query[
                            :,
                            (kernel_idx + 1) * self.num_heads_per_kernel_size : (kernel_idx + 2)
                            * self.num_heads_per_kernel_size,
                            :,
                            :,
                        ]
                    )
                )
                key_list.append(
                    self.key_depthwiseconv[str(kernel_idx)](
                        key[
                            :,
                            (kernel_idx + 1) * self.num_heads_per_kernel_size : (kernel_idx + 2)
                            * self.num_heads_per_kernel_size,
                            :,
                            :,
                        ]
                    )
                )
                value_list.append(
                    self.value_depthwiseconv[str(kernel_idx)](
                        value[
                            :,
                            (kernel_idx + 1) * self.num_heads_per_kernel_size : (kernel_idx + 2)
                            * self.num_heads_per_kernel_size,
                            :,
                            :,
                        ]
                    )
                )
            query = torch.cat(query_list, dim=1)
            key = torch.cat(key_list, dim=1)
            value = torch.cat(value_list, dim=1)

        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask, alibi_bias=alibi_bias
        )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class TranceptionBlockMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = tranception_ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TranceptionBlock(nn.Module):
    def __init__(self, config, SDWC_kernel_size=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = TranceptionBlockAttention(config, SDWC_kernel_size=SDWC_kernel_size)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = TranceptionBlockAttention(
                config, is_cross_attention=True, SDWC_kernel_size=SDWC_kernel_size
            )
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = TranceptionBlockMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        alibi_bias=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            alibi_bias=alibi_bias,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)

        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class TranceptionModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.position_embedding = (
            config.position_embedding if hasattr(config, "position_embedding") else "learned"
        )
        if self.position_embedding == "learned":
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
            self.alibi = None
        elif self.position_embedding == "grouped_alibi":
            maxpos = config.n_positions
            attn_heads = config.n_head
            self.slopes = torch.Tensor(get_slopes(attn_heads, mode=self.position_embedding))
            alibi = self.slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(
                0
            ).unsqueeze(0).expand(attn_heads, -1, -1)
            alibi = alibi.view(attn_heads, 1, maxpos)
            self.register_buffer("alibi", alibi)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([TranceptionBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()

        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if self.position_embedding == "learned":
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
            alibi_bias = None
        else:
            hidden_states = inputs_embeds
            alibi_bias = self.alibi.unsqueeze(0).to(hidden_states.device)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi_bias=alibi_bias,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class TranceptionLMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = TranceptionModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config

        self.init_weights()

        self.default_model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_parallel = False
        self.device_map = None

        # retrieval_aggregation_mode=None (the real repo's default for plain
        # autoregressive scoring) takes the no-op branch here -- no MSA prior
        # loading, no TranceptEVE retrieval fusion. See module docstring above.
        self.retrieval_aggregation_mode = (
            config.retrieval_aggregation_mode
            if hasattr(config, "retrieval_aggregation_mode")
            else None
        )

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        flip=None,
        start_slice=None,
        end_slice=None,
        mutated_sequence=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return output

        return TranceptionCausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            fused_shift_log_probas=None,
        )


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_tranception():
    # Tiny protein-alphabet GPT2Config: 25-token vocab (20 amino acids + a handful
    # of special/ambiguity tokens, matching the real repo's small protein tokenizer
    # scale), num_attention_heads must be a multiple of 4 for the tranception
    # attention_mode's 4-way kernel-size grouping (1/3/5/7).
    config = TranceptionConfig(
        vocab_size=25,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        attention_mode="tranception",
        position_embedding="grouped_alibi",
        retrieval_aggregation_mode=None,
    )
    return TranceptionLMHeadModel(config)


def example_input_tranception():
    # The real `_attn` alibi-bias broadcast (`alibi_bias[:,:,:attn_weights.size(-1)]`,
    # copied verbatim above) slices alibi_bias's singleton dim 2, not its maxpos-sized
    # last dim -- so the real code's implicit contract is that the traced sequence
    # length must equal config.n_positions (here 64) for the addition to broadcast.
    batch = 2
    seq_len = 64
    input_ids = torch.randint(0, 25, (batch, seq_len))
    return (input_ids,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Tranception", "build_tranception", "example_input_tranception", 2022, "vendored"),
]
