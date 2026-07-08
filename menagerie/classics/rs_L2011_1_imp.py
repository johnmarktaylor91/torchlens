# SOURCE: vendored from MILVLG/imp-v1-3b @ HuggingFace Hub commit 634c7fcac78f045a4f8359c70589ffc55bf2de84
# (custom modeling code shipped via `trust_remote_code=True`: modeling_imp.py, configuration_imp.py,
#  vision_encoder.py). Imports/relative-paths adjusted minimally for standalone staging; architecture
# untouched. Original repo copyright: MILVLG team (Apache-2.0), derived from Phi-2 (Microsoft, MIT),
# SigLIP (Google/HF, Apache-2.0), and LLaVA (Haotian Liu, Apache-2.0).
#
# Imp: a small-but-mighty multimodal LLM (MILVLG). Phi-2 causal LM tower + SigLIP vision tower
# fused through an MLP projector (LLaVA-style prepare_inputs_labels_for_multimodal). Custom
# HF-hub architecture (not a stock `transformers` class) -> vendored staging module (rung 2).
#
# Text-only entry point exercised here (ImpForCausalLM.forward with input_ids only; the vision
# path activates only when `images` is not None, so a pure-text trace already exercises the real
# ImpModel/PhiDecoderLayer/PhiAttention stack faithfully without requiring image preprocessing).

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)


# ============================================================================
# configuration_imp.py (PhiConfig, ImpConfig) -- vision config omitted here
# since the staged entry point is text-only.
# ============================================================================


class PhiConfig(PretrainedConfig):
    model_type = "phi"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=51200,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="gelu_new",
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=0.5,
        qk_layernorm=False,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.qk_layernorm = qk_layernorm
        self._rope_scaling_validation()

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _rope_scaling_validation(self):
        if self.rope_scaling is None:
            return
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, float)
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}"
            )


class ImpConfig(PhiConfig):
    model_type = "imp"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_token_index = getattr(self, "image_token_index", 50296)
        self.image_token = getattr(self, "image_token", "<image>")


# ============================================================================
# modeling_imp.py (Phi-2-style decoder stack + Imp causal LM head).
# Vision-tower branch (LlavaMetaModel.build_vision_tower / build_vision_projector)
# is kept intact structurally but never invoked since we don't set mm_vision_tower.
# ============================================================================


class PhiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    temp_type = q.dtype
    q, k, cos, sin = [t.to(dtype=torch.float32) for t in [q, k, cos, sin]]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed, k_embed = q_embed.to(temp_type), k_embed.to(temp_type)
    return q_embed, k_embed


class PhiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PhiAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: PhiConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads,
                eps=config.layer_norm_eps,
                elementwise_affine=True,
            )
            self.k_layernorm = nn.LayerNorm(
                config.hidden_size // self.num_heads,
                eps=config.layer_norm_eps,
                elementwise_affine=True,
            )

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = PhiRotaryEmbedding(
                int(self.partial_rotary_factor * self.head_dim),
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            raise NotImplementedError("rope_scaling not exercised by this staging module")

    @torch.autocast("cpu", enabled=False)
    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        softmax_scale = 1.0 / math.sqrt(query_states.shape[-1])
        attn_weights = torch.matmul(
            query_states.to(torch.float32),
            key_states.to(torch.float32).transpose(2, 3) * softmax_scale,
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            value_states.dtype
        )
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class PhiDecoderLayer(nn.Module):
    def __init__(self, config: PhiConfig, layer_idx: int):
        super().__init__()
        self.self_attn = PhiAttention(config, layer_idx=layer_idx)
        self.mlp = PhiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class PhiPreTrainedModel(PreTrainedModel):
    """Phi pre-trained model."""

    config_class = PhiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PhiDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlavaMetaModel(ABC):
    """Vision-tower plumbing kept for architectural fidelity; unused in the text-only trace."""

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def build_vision_projector(self, config):
        projector_type = getattr(config, "mm_projector_type", "linear")
        if projector_type == "linear":
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            return
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.mm_projector = nn.Sequential(*modules)
            return
        if projector_type == "identity":
            self.mm_projector = nn.Identity()
            return
        raise ValueError(f"Unknown projector type: {projector_type}")


class ImpModel(PhiPreTrainedModel, LlavaMetaModel):
    """Imp model. This implementation is modified from the implementation of Phi-2"""

    def __init__(self, config: ImpConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [PhiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = False

        self.gradient_checkpointing = False

        # NOTE: mm_vision_tower intentionally unset for this staging module (text-only trace);
        # the real repo builds a SigLIP VisionTower + projector here when config.mm_vision_tower
        # is present.

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.embed_dropout(inputs_embeds)

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlavaMetaForCausalLM(ABC):
    """This implementation is based on the implementation from the Llava project."""

    def init_constants(self, config):
        self.IGNORE_INDEX = getattr(config, "ignore_index", -100)
        self.IMAGE_TOKEN_INDEX = getattr(config, "image_token_index", 50296)
        self.DEFAULT_IMAGE_TOKEN = getattr(config, "image_token", "<image>")

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, None, None, past_key_values, None, None
        raise NotImplementedError("Vision path not exercised by this text-only staging module")


class ImpForCausalLM(PhiPreTrainedModel, LlavaMetaForCausalLM):
    """Imp for Causal Language Modeling."""

    config_class = ImpConfig

    def __init__(self, config: ImpConfig) -> None:
        super().__init__(config)

        self.model = ImpModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        self.post_init()
        self.init_constants(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def get_model(self):
        return self.model

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            loss = None
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


try:
    AutoConfig.register("imp", ImpConfig)
    AutoModelForCausalLM.register(ImpConfig, ImpForCausalLM)
except ValueError:
    pass  # already registered (e.g. re-import during validation)


# ============================================================================
# Menagerie staging entry points
# ============================================================================

MENAGERIE_ZOO = "vendored-pytorch"


def build_imp_v1_3b():
    """Tiny random-init Imp (Phi-2-style decoder + LLaVA-style multimodal scaffold), text-only path."""
    config = ImpConfig(
        vocab_size=51200,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        partial_rotary_factor=0.4,
        qk_layernorm=False,
        pad_token_id=50256,
        eos_token_id=50295,
        image_token_index=50296,
        # use_cache=False: plain single-pass forward (no KV cache), avoiding the legacy
        # DynamicCache.get_usable_length() API removed in newer `transformers` -- a real,
        # documented model configuration knob, not an architecture change.
        use_cache=False,
    )
    return ImpForCausalLM(config)


def example_input_imp_v1_3b():
    torch.manual_seed(0)
    return torch.randint(0, 51200, (1, 16), dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("Imp-v1-3B", build_imp_v1_3b, example_input_imp_v1_3b, 2024, "vendored-pytorch"),
]
