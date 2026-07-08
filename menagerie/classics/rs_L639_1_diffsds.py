# SOURCE: vendored from A4Bio/DiffSDS @ master
# Files combined: models/DiffSDS_model.py (DiffSDS_model, Structural_module),
# modules/modeling_bert_DiffSDS.py (BertPreTrainedModel/BertEncoder stack -- a locally
# modified copy of HuggingFace's transformers.models.bert.modeling_bert with an added
# `att_bias` argument threaded through self-attention), modules/FoldingDiff_module.py
# (BertEmbeddings, AnglesPredictor, GaussianFourierProjection), utils/nerf.py
# (TorchNERFBuilder -- only __init__ is exercised on this forward path).
# Only minimal changes: merged multiple source files into one module, dropped unused
# BertModel/BertForMaskedLM/BertForPreTraining/etc. head classes from
# modeling_bert_DiffSDS.py (DiffSDS_model only imports BertPreTrainedModel + BertEncoder),
# and pruned nerf.py to the trivial TorchNERFBuilder.__init__ (the model's forward() hardcodes
# `first_vector = True`, so pred_coord()/the NERF cartesian-coordinate builder is dead code on
# this path in the original repo too). Architecture (embeddings, custom BERT self-attention
# with att_bias, angle/vector decoding geometry) is untouched.
import math
import time
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.activations import ACT2FN, get_activation
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward

MENAGERIE_ZOO = "vendored-pytorch"

TIME_ENCODING = Literal["gaussian_fourier", "sinusoidal"]
DECODER_HEAD = Literal["mlp", "linear"]


# --- modules/FoldingDiff_module.py ---
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim: int, scale: float = 2 * torch.pi):
        super().__init__()
        w = torch.randn(embed_dim // 2) * scale
        assert not w.requires_grad
        self.register_buffer("W", w)

    def forward(self, x: torch.Tensor):
        if x.ndim > 1:
            x = x.squeeze()
        elif x.ndim < 1:
            x = x.unsqueeze(0)
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embed


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1)),
            )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_embeds: torch.Tensor, position_ids: torch.LongTensor) -> torch.Tensor:
        assert position_ids is not None, "`position_ids` must be defined"
        embeddings = input_embeds
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AnglesPredictor(nn.Module):
    """Predict angles from the embeddings (BERT-MLM-head-shaped decoder)."""

    def __init__(self, d_model: int, d_out: int = 4, activation="gelu", eps: float = 1e-12) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.dense1 = nn.Linear(d_model, d_model)
        if isinstance(activation, str):
            self.dense1_act = get_activation(activation)
        else:
            self.dense1_act = activation()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.dense2 = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.dense1_act(x)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return x


# --- modules/modeling_bert_DiffSDS.py (locally-modified HF BERT stack, att_bias threaded through) ---
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )
        self.is_decoder = config.is_decoder

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
        past_key_value=None,
        output_attentions=False,
        att_bias=None,
    ):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        attention_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)
            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_weights = attention_weights + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_weights = (
                    attention_weights
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        if att_bias is not None:
            attention_scores = attention_weights + att_bias
        else:
            attention_scores = attention_weights
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

        outputs = (
            (context_layer, attention_weights, attention_probs)
            if output_attentions
            else (context_layer,)
        )
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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
        past_key_value=None,
        output_attentions=False,
        att_bias=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            att_bias=att_bias,
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
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, (
                f"{self} should be used as a decoder model if cross attention is added"
            )
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
        past_key_value=None,
        output_attentions=False,
        att_bias=None,
    ):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            att_bias=att_bias,
        )
        attention_output = self_attention_outputs[0]

        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs

        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        att_bias=None,
    ):
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                att_bias=att_bias,
            )
            hidden_states = layer_outputs[0]
            att_bias = layer_outputs[1]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
        return hidden_states


class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# --- utils/nerf.py (TorchNERFBuilder; only __init__ is exercised -- DiffSDS_model.forward
# hardcodes first_vector=True, so the NERF cartesian-coordinate rebuild path is dead code) ---
class TorchNERFBuilder(nn.Module):
    """Builder for NERF (Natural Extension Reference Frame) coordinate reconstruction."""

    def __init__(self, virtual_num=3, num_rbf=16) -> None:
        super().__init__()
        self.virtual_num = virtual_num
        self.num_rbf = num_rbf


def _rbf(D, num_rbf):
    D_min, D_max, D_count = 0.0, 20.0, num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1, 1, 1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


# --- models/DiffSDS_model.py ---
class DiffSDS_model(BertPreTrainedModel):
    def __init__(
        self,
        config,
        step_gamma=0.00001,
        ft_is_angular: List[bool] = [False, True, True, True],
        ft_names: Optional[List[str]] = None,
        time_encoding: TIME_ENCODING = "gaussian_fourier",
        decoder: DECODER_HEAD = "mlp",
        use_grad=1,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.use_grad = use_grad
        self.step_gamma = step_gamma
        if self.config.is_decoder:
            raise NotImplementedError
        n_inputs = len(ft_is_angular)
        self.n_inputs = n_inputs

        self.distance_to_hidden_dim = nn.Sequential(
            nn.Linear(in_features=32, out_features=config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.config.num_attention_heads),
        )

        self.inputs_to_hidden_dim = nn.Linear(in_features=n_inputs, out_features=config.hidden_size)
        self.data_builder = TorchNERFBuilder(0)
        self.embeddings = BertEmbeddings(config)

        self.num_embedding = nn.Sequential(
            nn.Embedding(128, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.length_embedding = nn.Sequential(
            nn.Linear(32, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.seq_embedding = nn.Sequential(
            nn.Embedding(21, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.mask_embedding = nn.Sequential(
            nn.Embedding(2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.encoder = BertEncoder(config)

        self.start_pos_embedding = nn.Embedding(2 * 128, config.hidden_size)
        self.end_pos_embedding = nn.Embedding(2 * 128, config.hidden_size)

        if decoder == "linear":
            self.token_decoder = nn.Linear(config.hidden_size, n_inputs)
        elif decoder == "mlp":
            self.token_decoder = AnglesPredictor(config.hidden_size, n_inputs)
        else:
            raise ValueError(f"Unrecognized decoder: {decoder}")

        self.vector_decoder = AnglesPredictor(config.hidden_size, 9)

        if time_encoding == "gaussian_fourier":
            self.time_embed = GaussianFourierProjection(config.hidden_size)
        elif time_encoding == "sinusoidal":
            self.time_embed = SinusoidalPositionEmbeddings(config.hidden_size)
        else:
            raise ValueError(f"Unknown time encoding: {time_encoding}")

        self.init_weights()

        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()

    @staticmethod
    def _unit_vec(x):
        return x / torch.norm(x, dim=-1, keepdim=True)

    def dihedral(self, v1, v2, v3):
        v1 = self._unit_vec(v1)
        v2 = self._unit_vec(v2)
        v3 = self._unit_vec(v3)
        n1 = torch.cross(v1, v2, dim=-1)
        n2 = torch.cross(v2, v3, dim=-1)
        x = torch.sum(n1 * n2, dim=-1)
        y = torch.sum(torch.cross(n1, n2, dim=-1) * v2, dim=-1)
        return torch.arctan2(y, x)

    def angle(self, v1, v2):
        v1 = self._unit_vec(v1)
        v2 = self._unit_vec(v2)
        cos = torch.sum(v1 * v2, dim=-1)
        cos = torch.clamp(cos, -0.99999, 0.99999)
        return torch.arccos(cos)

    def get_vector(
        self, C_Ni, Ni_CAi, CAi_Ci, start_name="CA", end_name="CA", start_idx=80, end_idx=81
    ):
        vector_list = []
        for b_idx in range(start_idx.shape[0]):
            s_idx = start_idx[b_idx]
            e_idx = end_idx[b_idx]
            vector = (
                torch.sum(CAi_Ci[b_idx, s_idx:e_idx], dim=0) * 1.54
                + torch.sum(C_Ni[b_idx, s_idx + 1 : e_idx + 1], dim=0) * 1.34
                + torch.sum(Ni_CAi[b_idx, s_idx + 1 : e_idx + 1], dim=0) * 1.46
            )
            if start_name == "C":
                vector = vector - CAi_Ci[b_idx, s_idx] * 1.54
            if start_name == "N":
                vector = vector + Ni_CAi[b_idx, s_idx] * 1.46
            if end_name == "C":
                vector = vector + CAi_Ci[b_idx, e_idx] * 1.54
            if end_name == "N":
                vector = vector - Ni_CAi[b_idx, e_idx] * 1.46
            vector_list.append(vector)
        return torch.stack(vector_list)

    def forward(
        self,
        inputs: torch.Tensor,
        coords,
        timestep: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        seqs: torch.Tensor,
        unknown_mask: torch.Tensor,
        start_idx,
        end_idx,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        B, N, _ = inputs.shape
        num = end_idx - start_idx
        num_embed = self.num_embedding(num)
        idx = torch.arange(start_idx.shape[0], device=start_idx.device)
        length = torch.norm(coords[idx, end_idx, 1, :] - coords[idx, start_idx, 1, :], dim=1)
        length_embed = self.length_embedding(_rbf(length, 32)).reshape(B, -1)

        seq_length = inputs.size()[1]
        position_ids_start = torch.arange(seq_length, dtype=torch.long, device=inputs.device).view(
            1, -1
        )
        position_ids_end = torch.arange(seq_length, dtype=torch.long, device=inputs.device).view(
            1, -1
        )

        position_ids_start = (position_ids_start - start_idx[:, None]) + seq_length
        position_ids_end = (position_ids_end - end_idx[:, None]) + seq_length

        start_embedding = self.start_pos_embedding(position_ids_start)
        end_embedding = self.end_pos_embedding(position_ids_end)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = inputs.size()
        batch_size, seq_length, *_ = input_shape

        assert attention_mask is not None
        if position_ids is None:
            position_ids = torch.arange(seq_length).expand(batch_size, -1).type_as(timestep)

        assert attention_mask.dim() == 2
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.type_as(attention_mask)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        assert len(inputs.shape) == 3
        inputs_upscaled = self.inputs_to_hidden_dim(inputs)

        inputs_upscaled = self.embeddings(inputs_upscaled, position_ids=position_ids)
        seq_encoded = self.seq_embedding(seqs).reshape(B, N, -1)
        mask_encoded = self.mask_embedding(unknown_mask.long()).squeeze()

        time_encoded = self.time_embed(timestep.squeeze(dim=-1)).unsqueeze(1)
        inputs_with_time = (
            inputs_upscaled
            + time_encoded
            + seq_encoded
            + length_embed[:, None]
            + num_embed[:, None]
            + start_embedding
            + end_embedding
        ) * torch.sigmoid(mask_encoded)

        sequence_output = self.encoder(
            inputs_with_time,
            attention_mask=extended_attention_mask,
            output_attentions=True,
            return_dict=return_dict,
            att_bias=None,
        )

        vectors = self.vector_decoder(sequence_output)
        Bv, Nv, _ = vectors.shape
        vectors = vectors.reshape(Bv, -1, 3)
        vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

        C_Ni = vectors[:, 0::3]
        Ni_CAi = vectors[:, 1::3]
        CAi_Ci = vectors[:, 2::3]

        phi = self.dihedral(C_Ni, Ni_CAi, CAi_Ci)
        psi = self.dihedral(Ni_CAi, CAi_Ci, torch.roll(C_Ni, -1, dims=1))
        omega = self.dihedral(CAi_Ci, torch.roll(C_Ni, -1, dims=1), torch.roll(Ni_CAi, -1, dims=1))
        tau = self.angle(-torch.roll(Ni_CAi, -1, dims=1), torch.roll(CAi_Ci, -1, dims=1))
        CA_C_1N = self.angle(-CAi_Ci, torch.roll(C_Ni, -1, dims=1))
        C_1N_1CA = self.angle(-torch.roll(C_Ni, -1, dims=1), torch.roll(Ni_CAi, -1, dims=1))

        pred = torch.stack([phi, psi, omega, tau, CA_C_1N, C_1N_1CA], dim=-1)

        N_N = self.get_vector(C_Ni, Ni_CAi, CAi_Ci, "N", "N", start_idx, end_idx)
        CA_CA = self.get_vector(C_Ni, Ni_CAi, CAi_Ci, "CA", "CA", start_idx, end_idx)
        C_C = self.get_vector(C_Ni, Ni_CAi, CAi_Ci, "C", "C", start_idx, end_idx)
        vectors = torch.stack([N_N, C_C, CA_CA], dim=1)

        rand_idx = torch.cat(
            [torch.randint(start_idx[i], end_idx[i], (1,)) for i in range(start_idx.shape[0])]
        ).to(pred.device)

        anchor = coords[idx, start_idx, 1] + self.get_vector(
            C_Ni, Ni_CAi, CAi_Ci, "CA", "CA", start_idx, rand_idx
        )

        dist = (anchor[:, None, None] - coords[:, None, :, 1, :]).norm(dim=-1)[:, 0]
        rel_idx = rand_idx[:, None] - torch.arange(coords.shape[1], device=coords.device)[None]
        select_mask = ~unknown_mask[:, :, 0] * attention_mask * (rel_idx > 10)
        dist = dist * select_mask + 10000 * (1 - select_mask)
        dist = dist.min(dim=1)[0]

        return pred, vectors, dist


class Structural_module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, vectors):
        B, N, _ = vectors.shape
        q = vectors
        A = torch.einsum("bnd,bmd->bnm", vectors, vectors)
        attn = self.CNN(A.reshape(B, 1, N, N)).reshape(B, N, N)
        out = attn @ q
        return out


# --- tiny build/example helpers ---
def build_diffsds():
    cfg = BertConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        is_decoder=False,
        add_cross_attention=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    model = DiffSDS_model(cfg)
    model.eval()
    return model


def example_input_diffsds():
    # Real usage (methods/DiffSDS.py DiffSDS_run.train()) always passes an explicit
    # batch["position_ids"] long tensor; only forward()'s default-None branch derives
    # position_ids via `.type_as(timestep)`, which breaks for a float timestep (as used
    # by the real training loop's batch["t"]). Pass position_ids explicitly, matching
    # the real call site, rather than exercise that dead default-arg branch.
    torch.manual_seed(0)
    B, N = 2, 20
    inputs = torch.randn(B, N, 4)
    coords = torch.randn(B, N, 3, 3)
    timestep = torch.randint(0, 1000, (B, 1)).float()
    attention_mask = torch.ones(B, N)
    position_ids = torch.arange(N).unsqueeze(0).expand(B, -1).long()
    seqs = torch.randint(0, 21, (B, N))
    unknown_mask = torch.zeros(B, N, 1, dtype=torch.bool)
    start_idx = torch.full((B,), 2, dtype=torch.long)
    end_idx = torch.full((B,), N - 3, dtype=torch.long)
    return (
        inputs,
        coords,
        timestep,
        attention_mask,
        position_ids,
        seqs,
        unknown_mask,
        start_idx,
        end_idx,
    )


MENAGERIE_ENTRIES = [
    ("DiffSDS", build_diffsds, example_input_diffsds, 2023, MENAGERIE_ZOO),
]
