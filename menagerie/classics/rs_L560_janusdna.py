# SOURCE: vendored from https://github.com/Qihao-Duan/JanusDNA @ main
# Files: janusdna/configuration_janusdna.py + janusdna/modeling_janusdna.py
# JanusDNA is a hybrid bidirectional Mamba-SSM / grouped-query-attention / MoE genomic
# language model (DNA foundation model). Architecture is a real, non-trivial modification
# of the Jamba hybrid-SSM design (added: bidirectional Mamba wrappers, mid/final attention
# fusion, DNA-specific token vocab) -- not a class from any installed base library, so it is
# vendored here verbatim (only the unused `import wandb` and the relative
# `from .configuration_janusdna import` are adjusted since both files are merged into one
# module). The mamba_ssm/causal_conv1d fast kernels are optional CUDA extensions gated by
# `is_fast_path_available` / `config.use_mamba_kernels`; when unavailable (as in this base
# env) the real `slow_forward` pure-PyTorch SSM recurrence path is used instead -- this is
# the model's own documented fallback, not an approximation we wrote.

MENAGERIE_ZOO = "vendored-pytorch"

# This code is based on the original work by AI21 Labs Ltd. and HuggingFace Inc.,
# which is derived from EleutherAI's GPT-NeoX library and the GPT-NeoX/OPT implementations.
# Significant modifications have been made by AILS Lab for the purpose of developing the JanusDNA model.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" JanusDNA model configuration"""
import math

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from typing import List, Union
# from genfoundry.crab.modeling_crab import CrabFlexAttention

logger = logging.get_logger(__name__)


class JanusDNAConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the JanusDNA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`JanusDNAModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
            logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
            sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
            significantly.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss. See [here]() for more details
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to `None`.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            This value doesn't have any real effect. The maximum sequence length that this model is intended to be
            used with. It can be used with longer sequences, but performance may degrade.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            The number of experts to root per-token, can be also interpreted as the `top-p` routing
            parameter
        num_experts (`int`, *optional*, defaults to 16):
            Number of experts per Sparse MLP layer.
        expert_layer_period (`int`, *optional*, defaults to 2):
            Once in this many layers, we will have an expert layer
        expert_layer_offset (`int`, *optional*, defaults to 1):
            The first layer index that contains an expert mlp layer
        attn_layer_period (`int`, *optional*, defaults to 8):
            Once in this many layers, we will have a vanilla attention layer
        attn_layer_offset (`int`, *optional*, defaults to 4):
            The first layer index that contains a vanilla attention mlp layer
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
            `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
            `True` and kernels are not available
        mamba_d_state (`int`, *optional*, defaults to 16):
            The dimension the mamba state space latents
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
        mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block

    """

    model_type = "JanusDNA"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=6,  # 6
        phenotypes_size=2,  # 6
        tie_word_embeddings=False,
        hidden_size=1024,
        intermediate_size=1024,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        num_logits_to_keep=1,
        output_router_logits=False,
        router_aux_loss_coef=0.2,
        pad_token_id=0,
        # bos_token_id=1,
        # eos_token_id=2,
        sliding_window=None,
        max_position_embeddings=262144,
        attention_dropout=0.0,
        num_experts_per_tok=2,
        num_experts=16,
        expert_layer_period=2,
        expert_layer_offset=1,
        attn_layer_period=8,
        attn_layer_offset=4,
        use_mamba_kernels=True,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        bidirectional: bool = True,
        bidirectional_strategy: Union[
            str, None
        ] = "add",  # todo: # should be "add", "concat", "ew_multiply" and "final layer transformer"
        bidirectional_weight_tie: bool = True,
        layer_fusion: bool = False,
        attention_mask: bool = False,
        final_attention: bool = False,
        layer_fusion_strategy: Union[
            str, None
        ] = "pool",  # should be "pool" and "half", but frankly, half is not reasonable, for only fwd info without bwd info
        mid_single_direction_attention: bool = False,
        final_attention_class: str = "flex_attention",
        bidirectional_attn_tie: bool = False,
        is_causal: bool = True,
        gradient_checkpointing: bool = False,
        flex_attn_n_embd: int = 0,
        pad_vocab_size_multiple: int = 8,
        flexattn_fullattn: bool = False,
        **kwargs,
    ):
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.layer_fusion = layer_fusion
        self.attention_mask = attention_mask
        self.final_attention = final_attention
        self.layer_fusion_strategy = layer_fusion_strategy
        self.mid_single_direction_attention = mid_single_direction_attention
        self.final_attention_class = final_attention_class
        self.bidirectional_attn_tie = bidirectional_attn_tie
        self.is_causal = is_causal
        self.gradient_checkpointing = gradient_checkpointing
        self.flex_attn_n_embd = flex_attn_n_embd
        self.pad_vocab_size_multiple = pad_vocab_size_multiple

        self.vocab_size = vocab_size
        self.phenotypes_size = phenotypes_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.expert_layer_period = expert_layer_period
        self.expert_layer_offset = expert_layer_offset
        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset

        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = (
            math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        )
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        self.flexattn_fullattn = flexattn_fullattn

        super().__init__(
            pad_token_id=pad_token_id,
            # bos_token_id=bos_token_id,
            # eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        return [
            "attention" if i % self.attn_layer_period == self.attn_layer_offset else "mamba"
            for i in range(self.num_hidden_layers)
        ]

    @property
    def layers_num_experts(self):
        if self.expert_layer_period == 0:
            return 1
        else:
            return [
                self.num_experts if i % self.expert_layer_period == self.expert_layer_offset else 1
                for i in range(self.num_hidden_layers)
            ]


# This code is based on the original work by AI21 Labs Ltd. and HuggingFace Inc.,
# which is derived from EleutherAI's GPT-NeoX library and the GPT-NeoX/OPT implementations.
# Significant modifications have been made by AILS Lab for the purpose of developing the JanusDNA model.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch JanusDNA model."""
import os
import inspect
from typing import Any, Dict, Optional, Tuple, Union
import time

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch._dynamo.config.optimize_ddp = False
flex_attention = torch.compile(flex_attention, dynamic=False)
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache  # we need __iter__ and __len__ of pkv
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_attn_2_available,
    is_mamba_ssm_available,
)

# (JanusDNAConfig defined above, vendored in same file)

# try except block so it'll work with trust_remote_code.
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters,
    )
except ImportError:
    pass


# try except block so it'll work with trust_remote_code.
try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

# try except block so it'll work with trust_remote_code.
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_update, causal_conv1d_fn = None, None


is_fast_path_available = all(
    (
        selective_state_update,
        selective_scan_fn,
        causal_conv1d_fn,
        causal_conv1d_update,
        mamba_inner_fn,
    ),
)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "JanusDNAConfig"

""" --------------------------------- tools for model--------------------------------- """


def calculate_unactivated_params(config, model):
    if isinstance(model, JanusDNASparseMoeBlock):
        redundant_params = sum(
            p.numel()
            for expert in model.experts[config.num_experts_per_tok :]
            for p in expert.parameters()
        )
        return redundant_params
    else:
        return 0


""" --------------------------------- tools --------------------------------- """


def compute_expert_counts(selected_experts, num_experts, device):
    """count the number of times each expert is selected."""
    flat_experts = selected_experts.view(-1)
    counter = torch.bincount(flat_experts, minlength=num_experts).to(device)
    return counter


# Copied from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func with gate->router
def load_balancing_loss_func(
    router_logits: torch.Tensor,
    num_experts: torch.Tensor = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `router`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if router_logits is None or not isinstance(router_logits, tuple):
        return 0

    if isinstance(router_logits, tuple):
        compute_device = router_logits[0].device
        concatenated_router_logits = torch.cat(
            [layer_router.to(compute_device) for layer_router in router_logits],
            dim=0,
        )

    routing_weights = torch.nn.functional.softmax(concatenated_router_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_router_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask,
            dim=0,
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask,
            dim=0,
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from Jamba-v0.1 with Jamba->JanusDNA
class JanusDNARMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        JanusDNARMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        slen,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HybridMambaAttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.layers_block_type = config.layers_block_type
        self.has_previous_state = False  # only used by mamba
        intermediate_size = config.mamba_expand * config.hidden_size
        ssm_state_size = config.mamba_d_state
        conv_kernel_size = config.mamba_d_conv
        self.conv_states = []
        self.ssm_states = []
        for i in range(config.num_hidden_layers):
            if self.layers_block_type[i] == "mamba":
                self.conv_states += [
                    torch.zeros(
                        batch_size,
                        intermediate_size,
                        conv_kernel_size,
                        device=device,
                        dtype=dtype,
                    ),
                ]
                self.ssm_states += [
                    torch.zeros(
                        batch_size,
                        intermediate_size,
                        ssm_state_size,
                        device=device,
                        dtype=dtype,
                    ),
                ]
            else:
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]

        self.key_cache = [
            torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)
        ]
        self.value_cache = [
            torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)
        ]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states],
                dim=2,
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(
                0,
                beam_idx.to(device),
            )
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(
                0,
                beam_idx.to(device),
            )

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(
                0,
                beam_idx.to(device),
            )
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(
                0,
                beam_idx.to(device),
            )

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError(
            "HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.",
        )

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> "DynamicCache":
        raise NotImplementedError(
            "HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.",
        )


# Copied from Jamba-v0.1 with Jamba->JanusDNA
class JanusDNAAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: JanusDNAConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class.",
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = config.is_causal
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).",
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1,
            2,
        )
        value_states = value_states.view(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim,
        )

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype,
        )
        attn_weights = nn.functional.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training,
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}",
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class JanusDNAFlexAttention(nn.Module):
    def __init__(self, config: JanusDNAConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.last_q_len = None
        self.block_mask = None

        """ ------------ original attention config --------------"""
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class.",
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = config.is_causal
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).",
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )

        # self.o_projs = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.o_projs = nn.ModuleList(
            [nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        q_seq = hidden_states
        kv_seq = hidden_states

        bsz, q_len, q_hidden_size = q_seq.size()

        query_states = self.q_proj(q_seq)
        key_states = self.k_proj(kv_seq)
        value_states = self.v_proj(kv_seq)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1,
            2,
        )
        value_states = value_states.view(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        orig_seq_len = q_len // 2

        def create_janus_mask(b, h, q_idx, kv_idx):
            mask1 = (kv_idx < orig_seq_len) & (q_idx < orig_seq_len) & (q_idx >= kv_idx)
            mask2 = (
                (kv_idx >= orig_seq_len)
                & (q_idx < orig_seq_len)
                & (kv_idx >= orig_seq_len + q_idx + 2 * 1)
            )
            mask3 = (
                (kv_idx < orig_seq_len)
                & (q_idx >= orig_seq_len)
                & (q_idx >= kv_idx + orig_seq_len + 2 * 1)
            )
            mask4 = (kv_idx >= orig_seq_len) & (q_idx >= orig_seq_len) & (q_idx <= kv_idx)

            # Combine all mask conditions
            mask = mask1 | mask2 | mask3 | mask4
            return mask

        if self.last_q_len is None or self.block_mask is None or self.last_q_len != q_len:
            if not self.config.flexattn_fullattn:
                self.last_q_len = q_len
                self.block_mask = create_block_mask(
                    create_janus_mask,
                    bsz,
                    self.num_heads,
                    q_len,
                    q_len,
                    device=query_states.device,
                    _compile=True,
                )
                print("--- janus block mask constructed ---")

        batch, head, q_len, head_dim = query_states.shape
        expected_head_dim = self.config.flex_attn_n_embd // self.num_heads
        if head_dim != expected_head_dim:
            query_states = F.pad(
                query_states, (0, expected_head_dim - head_dim), mode="constant", value=0
            )
            key_states = F.pad(
                key_states, (0, expected_head_dim - head_dim), mode="constant", value=0
            )
            value_states = F.pad(
                value_states, (0, expected_head_dim - head_dim), mode="constant", value=0
            )

        attn_output = flex_attention(
            query_states,
            key_states,
            value_states,
            block_mask=self.block_mask,
        )

        attn_output = attn_output[:, :, :, :head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_projs[0](attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Adapted from Jamba-v0.1 with Jamba->JanusDNA
class JanusDNAFlashAttention2(JanusDNAAttention):
    """
    JanusDNA flash attention module. This module inherits from `JanusDNAAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1,
            2,
        )
        value_states = value_states.view(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        kv_seq_len = cache_position[-1]

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library.",
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = cache_position[0] > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}",
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])],
                        dim=-1,
                    )

            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}.",
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
                self._upad_input(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    query_length,
                )
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    # Copied from transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
            indices_k,
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1,
                dtype=torch.int32,
                device=query_layer.device,
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer,
                attention_mask,
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Adapted from Jamba-v0.1 with Jamba->JanusDNA
class JanusDNASdpaAttention(JanusDNAAttention):
    """
    JanusDNA attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `JanusDNAAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from JanusDNAAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "JanusDNAModel is using JanusDNASdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.',
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1,
            2,
        )
        value_states = value_states.view(
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# Adapted from caduceus.
class BiJanusDNAAttentionWrapper(nn.Module):
    """
    A wrapper for JanusDNA attention that supports separated bidirectional attention.
    """

    def __init__(self, config: JanusDNAConfig, layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx]

        self.self_attn_fwd = JANUSDNA_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx
        )

        ffn_layer_class = JanusDNASparseMoeBlock if num_experts > 1 else JanusDNAMLP
        self.feed_forward_fwd = ffn_layer_class(config)
        self.input_layernorm_fwd = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm_fwd = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.bidirectional_attn_tie = config.bidirectional_attn_tie

        self.self_attn_bwd = JANUSDNA_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx
        )
        self.feed_forward_bwd = ffn_layer_class(config)
        self.input_layernorm_bwd = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm_bwd = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.bidirectional_attn_tie:
            # tie qkv_proj
            self.self_attn_bwd.q_proj = self.self_attn_fwd.q_proj
            self.self_attn_bwd.k_proj = self.self_attn_fwd.k_proj
            self.self_attn_bwd.v_proj = self.self_attn_fwd.v_proj
            # tie out_proj
            self.self_attn_bwd.o_proj = self.self_attn_fwd.o_proj
            print("attention_weight_tied")

        if hasattr(self, "feed_forward_fwd") and hasattr(self, "feed_forward_bwd"):
            self.redundant_expert_params = (
                calculate_unactivated_params(config, self.feed_forward_fwd) / 1e6
                + calculate_unactivated_params(config, self.feed_forward_bwd) / 1e6
            )
        else:
            self.redundant_expert_params = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`HybridMambaAttentionDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """
        hidden_states_fwd = hidden_states[:, : hidden_states.shape[1] // 2, :]
        hidden_states_bwd = hidden_states[:, hidden_states.shape[1] // 2 :, :]

        residual_fwd = hidden_states_fwd
        residual_bwd = hidden_states_bwd

        hidden_states_fwd = self.input_layernorm_fwd(hidden_states_fwd)
        hidden_states_bwd = self.input_layernorm_bwd(hidden_states_bwd)

        hidden_states_fwd, self_attn_weights_fwd, present_key_value_fwd = self.self_attn_fwd(
            hidden_states=hidden_states_fwd,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states_bwd, self_attn_weights_bwd, present_key_value_bwd = self.self_attn_bwd(
            hidden_states=hidden_states_bwd,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states_fwd = residual_fwd + hidden_states_fwd
        hidden_states_bwd = residual_bwd + hidden_states_bwd

        # feed-forward (experts/MLP)
        residual_fwd = hidden_states_fwd
        residual_bwd = hidden_states_bwd
        hidden_states_fwd = self.pre_ff_layernorm_fwd(hidden_states_fwd)
        hidden_states_bwd = self.pre_ff_layernorm_bwd(hidden_states_bwd)

        ff_outputs_fwd = self.feed_forward_fwd(hidden_states_fwd)
        ff_outputs_bwd = self.feed_forward_bwd(hidden_states_bwd)

        if isinstance(ff_outputs_fwd, tuple):
            hidden_states_fwd, router_logits_fwd = ff_outputs_fwd
        else:
            hidden_states_fwd, router_logits_fwd = ff_outputs_fwd, None

        if isinstance(ff_outputs_bwd, tuple):
            hidden_states_bwd, router_logits_bwd = ff_outputs_bwd
        else:
            hidden_states_bwd, router_logits_bwd = ff_outputs_bwd, None

        hidden_states_fwd = residual_fwd + hidden_states_fwd
        hidden_states_bwd = residual_bwd + hidden_states_bwd

        hidden_states = torch.cat([hidden_states_fwd, hidden_states_bwd], dim=1)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights_fwd, self_attn_weights_bwd)

        if use_cache:
            outputs += (present_key_value_fwd, present_key_value_bwd)

        if output_router_logits:
            outputs += (router_logits_fwd, router_logits_bwd)

        return outputs


JANUSDNA_ATTENTION_CLASSES = {
    "eager": JanusDNAAttention,
    "flash_attention_2": JanusDNAFlashAttention2,
    "sdpa": JanusDNASdpaAttention,
    "flex_attention": JanusDNAFlexAttention,
}


class SingleJanusDNAAttentionWrapper(nn.Module):
    """
    A wrapper for JanusDNAAttention that processes the input sequence in a single direction. no actual use here.
    """

    def __init__(self, config: JanusDNAConfig, layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx]
        self.self_attn = JANUSDNA_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        ffn_layer_class = JanusDNASparseMoeBlock if num_experts > 1 else JanusDNAMLP
        self.feed_forward = ffn_layer_class(config)
        self.input_layernorm = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.redundant_expert_params = calculate_unactivated_params(config, self.feed_forward) / 1e6

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`HybridMambaAttentionDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        # residual connection after attention
        hidden_states = residual + hidden_states

        # feed-forward (experts/MLP)
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        ff_outputs = self.feed_forward(hidden_states)
        if isinstance(ff_outputs, tuple):
            hidden_states, router_logits = ff_outputs
        else:
            hidden_states, router_logits = ff_outputs, None
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class FinalAttention(nn.Module):
    """
    The final attention layer for Janus Modeling.
    """

    def __init__(self, config: JanusDNAConfig, layer_idx: int):
        super().__init__()
        num_experts = 0
        self.self_attn = JANUSDNA_ATTENTION_CLASSES[config.final_attention_class](config, layer_idx)

        ffn_layer_class = JanusDNASparseMoeBlock if num_experts > 1 else JanusDNAMLP
        self.feed_forward = ffn_layer_class(config)
        self.input_layernorm = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`HybridMambaAttentionDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """

        residual = hidden_states

        # hidden_states = self.in_real_layernorm(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        def process_hidden_states(self, hidden_states, residual):
            # residual connection after attention
            hidden_states = residual + hidden_states

            # feed-forward (experts/MLP)
            residual = hidden_states
            hidden_states = self.pre_ff_layernorm(hidden_states)
            ff_outputs = self.feed_forward(hidden_states)
            if isinstance(ff_outputs, tuple):
                hidden_states, router_logits = ff_outputs
            else:
                hidden_states, router_logits = ff_outputs, None
            hidden_states = residual + hidden_states

            return hidden_states, router_logits

        if isinstance(hidden_states, list):
            results = [process_hidden_states(self, hs, residual) for hs in hidden_states]
            hidden_states, router_logits = zip(*results)
        else:
            hidden_states, router_logits = process_hidden_states(self, hidden_states, residual)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


# Adapted from Jamba-v0.1 with Jamba->JanusDNA
class JanusDNAMambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: JanusDNAConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.use_fast_kernels = config.use_mamba_kernels

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
        )
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        self.dt_layernorm = JanusDNARMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        self.b_layernorm = JanusDNARMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.c_layernorm = JanusDNARMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d. If you want to use the naive implementation, set `use_mamba_kernels=False` in the model config",
            )

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: HybridMambaAttentionDynamicCache = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
        )

        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        # We can't use `mamba_inner_fn` even if in training and without cache params because we have the
        # inner layernorms which isn't supported by this fused kernel
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0),
            self.conv1d.weight.size(2),
        )
        if use_precomputed_states:
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache_params.conv_states[self.layer_idx],
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0),
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_states)

            hidden_states = causal_conv1d_fn(
                hidden_states,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
            )

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )

        time_step = self.dt_layernorm(time_step)
        original_type = B.dtype
        B = self.b_layernorm(B).to(original_type)
        C = self.c_layernorm(C).to(original_type)

        # Here we need to apply dt_proj without the bias, as the bias is added in the selective scan kernel.
        # This is a hack to apply dt_proj while still using the forward pass of `torch.nn.Linear`, which is needed
        # in order to make quantization work. Quantization code replaces `torch.nn.Linear` layers with quantized
        # linear layers, and requires to call the forward pass directly.
        # The original code here was: ```discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)```
        time_proj_bias = self.dt_proj.bias
        self.dt_proj.bias = None
        discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
        self.dt_proj.bias = time_proj_bias

        A = -torch.exp(self.A_log.float())
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None
        if use_precomputed_states:
            scan_outputs = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))

        return contextualized_states

    # fmt: off；no mamba kernel version, which is a navie implementation
    def slow_forward(self, input_states, cache_params: HybridMambaAttentionDynamicCache = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(
            1, 2
        )  # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        use_cache = isinstance(cache_params, HybridMambaAttentionDynamicCache)
        # 2. Convolution sequence transformation
        if use_cache and cache_params.ssm_states[self.layer_idx].shape[0] == batch_size:
            if self.training:
                # In training mode, we don't want to perform in-place operations on ssm_state so we can compute the backwards pass
                ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            else:
                ssm_state = cache_params.ssm_states[self.layer_idx]

            if (
                cache_params.has_previous_state
                and seq_len == 1
                and cache_params.conv_states[self.layer_idx].shape[0] == batch_size
            ):
                conv_state = cache_params.conv_states[
                    self.layer_idx
                ]  # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = (
                    self.act(hidden_states).to(dtype).unsqueeze(-1)
                )  # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0),
                )
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = self.act(
                    self.conv1d(hidden_states)[..., :seq_len]
                )  # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device,
                dtype=dtype,
            )
            hidden_states = self.act(
                self.conv1d(hidden_states)[..., :seq_len]
            )  # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )

        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(
            1, 2
        )  # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(
            A[None, :, None, :] * discrete_time_step[:, :, :, None]
        )  # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = (
            discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        )  # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = (
                discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            )  # [batch, intermediade_size, ssm_state]
            scan_output = torch.matmul(
                ssm_state.to(dtype), C[:, i, :].unsqueeze(-1)
            )  # [batch, intermediade_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)  # [batch, intermediade_size, seq_len]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = scan_output * self.act(gate)

        if use_cache:
            cache_params.ssm_states[self.layer_idx] = ssm_state

        # 4. Final linear projection
        contextualized_states = self.out_proj(
            scan_output.transpose(1, 2)
        )  # [batch, seq_len, hidden_size]
        return contextualized_states

    # fmt: on

    def forward(self, hidden_states, cache_params: HybridMambaAttentionDynamicCache = None):
        if self.use_fast_kernels:
            if not is_fast_path_available or "cuda" not in self.x_proj.weight.device.type:
                raise ValueError(
                    "Fast Mamba kernels are not available. Make sure to they are installed and that the mamba module is on a CUDA device",
                )
            return self.cuda_kernels_forward(hidden_states, cache_params)
        return self.slow_forward(hidden_states, cache_params)


# Adapted from Jamba-v0.1 with Jamba->JanusDNA
class JanusDNAMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Adapted from Jamba-v0.1 with Jamba->JanusDNA
class JanusDNASparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config: JanusDNAConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([JanusDNAMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts,
            num_classes=self.num_experts,
        ).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class JanusDNAAttentionDecoderLayer(nn.Module):
    """
    only decide single direction attention or bidirectional attention
    """

    def __init__(self, config: JanusDNAConfig, layer_idx: int):
        super().__init__()
        self.attn = (
            BiJanusDNAAttentionWrapper(config, layer_idx)
            if config.mid_single_direction_attention and not config.layer_fusion
            else SingleJanusDNAAttentionWrapper(config, layer_idx)
        )
        self.redundant_expert_params = self.attn.redundant_expert_params

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`HybridMambaAttentionDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """

        hidden_states = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        return hidden_states


class BiJanusDNAMambaWrapper(nn.Module):
    def __init__(self, config: JanusDNAConfig, layer_idx: int):
        super().__init__()

        self.bidirectional = config.bidirectional
        self.bidirectional_strategy = config.bidirectional_strategy
        self.bidirectional_weight_tie = config.bidirectional_weight_tie

        self.mamba_fwd = JanusDNAMambaMixer(config=config, layer_idx=layer_idx)
        if self.bidirectional:
            self.mamba_rev = JanusDNAMambaMixer(config=config, layer_idx=layer_idx)
            if self.bidirectional_weight_tie:  # most of parameters are in and out projection
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

        """ ---------add ffn module start -------- """
        num_experts = config.layers_num_experts[layer_idx]
        ffn_layer_class = JanusDNASparseMoeBlock if num_experts > 1 else JanusDNAMLP
        self.feed_forward = ffn_layer_class(config)
        self.input_layernorm = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.redundant_expert_params = calculate_unactivated_params(config, self.feed_forward) / 1e6
        """ ---------add ffn module end -------- """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        residual = hidden_states
        # input layernorm
        hidden_states = self.input_layernorm(hidden_states)

        out = self.mamba_fwd(hidden_states, cache_params=past_key_value)

        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                cache_params=past_key_value,
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(
                    f"`{self.bidirectional_strategy}` for bi-directionality not implemented!"
                )

        self_attn_weights = None
        # residual
        hidden_states = residual + hidden_states

        # feed-forward (experts/MLP)
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        ff_outputs = self.feed_forward(hidden_states)
        if isinstance(ff_outputs, tuple):
            hidden_states, router_logits = ff_outputs
        else:
            hidden_states, router_logits = ff_outputs, None
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class BiJanusDNAMambaSeperateWrapper(nn.Module):
    """
    seperate unidirectional JanusDNA Mamba MoE Mixer for bidirectional processing.
    """

    def __init__(self, config: JanusDNAConfig, layer_idx: int):
        super().__init__()

        self.bidirectional = config.bidirectional
        # todo: # should be "add", "concat", "ew_multiply" and "final layer transformer"
        self.bidirectional_strategy = config.bidirectional_strategy
        self.bidirectional_weight_tie = config.bidirectional_weight_tie

        self.mamba_fwd = JanusDNAMambaMixer(
            config=config, layer_idx=layer_idx
        )  # todo: if the layer id is suitable?
        if self.bidirectional:
            self.mamba_rev = JanusDNAMambaMixer(config=config, layer_idx=layer_idx)
            if self.bidirectional_weight_tie:  # most of parameters are in and out projection
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

        """ ---------add ffn module start -------- """
        num_experts = config.layers_num_experts[layer_idx]
        ffn_layer_class = JanusDNASparseMoeBlock if num_experts > 1 else JanusDNAMLP
        # fwd
        self.input_layernorm_fwd = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm_fwd = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward_fwd = ffn_layer_class(config)

        # bwd
        self.input_layernorm_bwd = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm_bwd = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward_bwd = ffn_layer_class(config)

        # calculate the redundant expert params
        if hasattr(self, "feed_forward_fwd") and hasattr(self, "feed_forward_bwd"):
            self.redundant_expert_params = (
                calculate_unactivated_params(config, self.feed_forward_fwd) / 1e6
                + calculate_unactivated_params(config, self.feed_forward_bwd) / 1e6
            )
        else:
            self.redundant_expert_params = 0
        """ ---------add ffn module end -------- """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Wrap the jambaMamba and Caduceus together

        Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # todo: this is the first version of add after each layer
        # just for bidirectional mode
        assert self.bidirectional, (
            "BiJanusDNAMambaSeperateWrapper requires 'self.bidirectional' to be set to True."
        )

        residual_fwd = hidden_states[:, : hidden_states.shape[1] // 2, :]
        residual_bwd = hidden_states[:, hidden_states.shape[1] // 2 :, :]
        # input layernorm
        hidden_states_fwd = self.input_layernorm_fwd(residual_fwd)
        hidden_states_bwd = self.input_layernorm_bwd(residual_bwd)
        # bi-mamba
        hidden_states_fwd = self.mamba_fwd(hidden_states_fwd, cache_params=past_key_value)
        hidden_states_bwd = self.mamba_rev(hidden_states_bwd, cache_params=past_key_value)

        self_attn_weights = None
        # residual
        hidden_states_fwd = hidden_states_fwd + residual_fwd
        hidden_states_bwd = hidden_states_bwd + residual_bwd

        # feed-forward (experts/MLP)
        residual_fwd = hidden_states_fwd
        residual_bwd = hidden_states_bwd
        hidden_states_fwd = self.pre_ff_layernorm_fwd(hidden_states_fwd)
        hidden_states_bwd = self.pre_ff_layernorm_bwd(hidden_states_bwd)

        ff_outputs_fwd = self.feed_forward_fwd(hidden_states_fwd)
        ff_outputs_bwd = self.feed_forward_bwd(hidden_states_bwd)
        if isinstance(ff_outputs_fwd, tuple):
            hidden_states_fwd, router_logits_fwd = ff_outputs_fwd
            hidden_states_bwd, router_logits_bwd = ff_outputs_bwd
        else:
            hidden_states_fwd, router_logits_fwd = ff_outputs_fwd, None
            hidden_states_bwd, router_logits_bwd = ff_outputs_bwd, None
        # residual
        hidden_states_fwd = residual_fwd + hidden_states_fwd
        hidden_states_bwd = residual_bwd + hidden_states_bwd

        hidden_states = torch.cat([hidden_states_fwd, hidden_states_bwd], dim=1)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if output_router_logits:
            outputs += (router_logits_fwd, router_logits_bwd)

        return outputs


class JanusDNAMambaDecoderLayer(nn.Module):
    def __init__(self, config: JanusDNAConfig, layer_idx: int):
        super().__init__()

        self.mamba_module = (
            BiJanusDNAMambaWrapper(config=config, layer_idx=layer_idx)
            if config.layer_fusion
            else BiJanusDNAMambaSeperateWrapper(config=config, layer_idx=layer_idx)
        )

        self.redundant_expert_params = self.mamba_module.redundant_expert_params

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`HybridMambaAttentionDynamicCache`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
        """
        return self.mamba_module(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            use_cache=use_cache,
            cache_position=cache_position,
        )


JANUSDNA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`JanusDNAConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Janus DNA Model outputting raw hidden-states without any specific head on top.",
    JANUSDNA_START_DOCSTRING,
)
# Adapted from Jamba-v0.1 with Jamba->JanusDNA
class JanusDNAPreTrainedModel(PreTrainedModel):
    config_class = JanusDNAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["JanusDNAAttentionDecoderLayer", "JanusDNAMambaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        n_residuals_per_layer = 1
        n_layer = self.config.num_hidden_layers

        # do one more xavier_norm for attn qkv
        for name, param in module.named_parameters():
            if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                nn.init.xavier_uniform_(param)


JANUSDNA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`HybridMambaAttentionDynamicCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            A HybridMambaAttentionDynamicCache object containing pre-computed hidden-states (keys and values in the
            self-attention blocks and convolution and ssm states in the mamba blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
            Key and value cache tensors have shape `(batch_size, num_heads, seq_len, head_dim)`.
            Convolution and ssm states tensors have shape `(batch_size, d_inner, d_conv)` and
            `(batch_size, d_inner, d_state)` respectively.
            See the `HybridMambaAttentionDynamicCache` class for more details.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
            should not be returned during inference.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

ALL_DECODER_LAYER_TYPES = {
    "attention": JanusDNAAttentionDecoderLayer,
    "mamba": JanusDNAMambaDecoderLayer,
}


@add_start_docstrings(
    "The bare Janus DNA Model outputting raw hidden-states without any specific head on top.",
    JANUSDNA_START_DOCSTRING,
)
# Adapted from Jamba-v0.1 with Jamba->JanusDNA
class JanusDNAModel(JanusDNAPreTrainedModel):
    """
    Args:
        config: JanusDNAConfig
    """

    def __init__(self, config: JanusDNAConfig):
        super().__init__(config)
        # self.padding_idx = config.pad_token_id

        # make vocab size even
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        decoder_layers = []
        for i in range(config.num_hidden_layers):
            layer_class = ALL_DECODER_LAYER_TYPES[config.layers_block_type[i]]
            layer = layer_class(config, layer_idx=i)
            decoder_layers.append(layer)

        self.layers = nn.ModuleList(decoder_layers)
        self.redundant_expert_params_all_layers = sum(
            layer.redundant_expert_params for layer in self.layers
        )

        self._attn_implementation = config._attn_implementation
        self.final_layernorm = JanusDNARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_fusion = config.layer_fusion  # if layer_fusion, bi-directional output of each layer would be fused. If not, will just concat and fuse at last decode layer.
        self.attention_mask = config.attention_mask

        self.gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights and apply final processing
        self.post_init()

        # last layer attention
        if config.final_attention:
            self.final_attention = FinalAttention(config, layer_idx=config.num_hidden_layers)

        self.final_fusion = (
            self.final_fusion_repos_formasked
            if config.layer_fusion_strategy == "pool"
            else self.final_fusion_first_half
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            ACT2FN[config.hidden_act],
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            ACT2FN[config.hidden_act],
        )

    def final_fusion_pool(self, hidden_states):
        mid = hidden_states.shape[1] // 2
        first_half = hidden_states[:, :mid, :]

        # # 1. for autoregressive dataloader way to combine, i.e. forward_token_0 and backward_token_2 to predict pred_tokne_1
        second_half = (
            hidden_states[:, mid + 2 :, :]
            if hidden_states.shape[1] > mid + 2
            else torch.zeros_like(first_half)
        )

        if second_half.shape[1] < first_half.shape[1]:
            padding = torch.zeros(
                hidden_states.shape[0],
                first_half.shape[1] - second_half.shape[1],
                hidden_states.shape[2],
                device=hidden_states.device,
            )
            second_half = torch.cat([second_half, padding], dim=1)

        return first_half + second_half

    def final_fusion_repos_formasked(self, hidden_states):
        """
        Work for repose the predicted token position regarding the the final fused bidirectional hidden states.
        """
        b, l, h = hidden_states.shape
        mid_len = l // 2
        assert mid_len > 2 * 1, (
            "sequence length must be larger than 2 * 1 for JanusDNA final fusion."
        )

        fused_hidden_states = torch.zeros(
            b, mid_len, h, dtype=hidden_states.dtype, device=hidden_states.device
        )

        head_indices = torch.arange(1, device=hidden_states.device)
        fused_hidden_states[:, head_indices, :] = hidden_states[:, mid_len + 1 + head_indices, :]

        middle_indices = torch.arange(mid_len - (2 * 1), device=hidden_states.device)
        fused_hidden_states[:, 1 + middle_indices, :] = (
            hidden_states[:, middle_indices, :]
            + hidden_states[:, mid_len + 1 + middle_indices + 1, :]
        )

        tail_indices = torch.arange(1, device=hidden_states.device)
        fused_hidden_states[:, mid_len - 1 - tail_indices, :] = hidden_states[
            :, mid_len - 1 - 1 - tail_indices, :
        ]

        return fused_hidden_states

    def final_fusion_first_half(self, hidden_states):
        return hidden_states[:, : hidden_states.shape[1] // 2, :]

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(JANUSDNA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.",
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = (
            inputs_embeds
            if self.layer_fusion
            else torch.concat([inputs_embeds, inputs_embeds.flip(dims=(1,))], dim=1)
        )

        if use_cache and past_key_values is None:
            logger.warning_once(
                "JanusDNA requires an initialized `HybridMambaAttentionDynamicCache` to return a cache. None was "
                "provided, so no cache will be returned.",
            )

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is not None:
            pass
        elif self.attention_mask:
            seq_len = hidden_states.shape[1]

            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=0
            )

        causal_mask = self._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                pass

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                if layer_outputs[1] is not None:
                    all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                if layer_outputs[-1] is not None and layer_outputs[-2] is not None:
                    all_router_logits += (layer_outputs[-2],)  # first forward moe loss
                    all_router_logits += (layer_outputs[-1],)  # second backward moe loss

        if self.config.final_attention:
            if not self.layer_fusion:
                hidden_states = torch.cat(
                    [
                        hidden_states[:, : hidden_states.shape[1] // 2, :],
                        hidden_states[:, hidden_states.shape[1] // 2 :, :].flip(dims=[1]),
                    ],
                    dim=1,
                )

            layer_outputs = self.final_attention(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                if layer_outputs[1] is not None:
                    all_self_attns += (layer_outputs[1],)

        if not self.layer_fusion:
            hidden_states = self.final_fusion(hidden_states)

        hidden_states = self.final_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.final_mlp(hidden_states)
        hidden_states = residual + hidden_states  # residual connection

        # add hidden states from the last decoder layer
        if output_hidden_states:
            pass
            # all_hidden_states += (hidden_states.cpu(),)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        next_cache = None if not use_cache else past_key_values

        """ ---------------------check the moe router output-------------------------- """
        if output_router_logits:
            if isinstance(all_router_logits, tuple):
                if not all_router_logits:
                    raise ValueError("all_router_logits is empty")

                compute_device = all_router_logits[0].device
                num_experts = 16
                top_k = 2

                all_on_device = all(logits.device == compute_device for logits in all_router_logits)

                fwd_logits = all_router_logits[0::2]
                bwd_logits = all_router_logits[1::2]

                if all_on_device:
                    concatenated_router_logits_fwd = torch.cat(fwd_logits, dim=0)
                    concatenated_router_logits_bwd = torch.cat(bwd_logits, dim=0)
                else:
                    concatenated_router_logits_fwd = torch.cat(
                        [logits.to(compute_device) for logits in fwd_logits], dim=0
                    )
                    concatenated_router_logits_bwd = torch.cat(
                        [logits.to(compute_device) for logits in bwd_logits], dim=0
                    )

                # calculate routing weights
                routing_weights_fwd = F.softmax(concatenated_router_logits_fwd, dim=-1)
                routing_weights_bwd = F.softmax(concatenated_router_logits_bwd, dim=-1)

                # pick top-k experts
                _, selected_experts_fwd = torch.topk(routing_weights_fwd, top_k, dim=-1)
                _, selected_experts_bwd = torch.topk(routing_weights_bwd, top_k, dim=-1)

                # compute expert counts
                counter_fwd = compute_expert_counts(
                    selected_experts_fwd, num_experts, compute_device
                )
                counter_bwd = compute_expert_counts(
                    selected_experts_bwd, num_experts, compute_device
                )
                log_dict = {}
                for i in range(num_experts):
                    log_dict[f"train/fwd_{i}"] = counter_fwd[i].item()  # record counter_fwd
                    log_dict[f"train/bwd_{i}"] = counter_bwd[i].item()  # record counter_bwd
        """ ---------------------check the moe router output-------------------------- """

        """ ---------------------   add moe_loss output   -------------------------- """
        device = (
            hidden_states[0].device
            if isinstance(hidden_states, list) or isinstance(hidden_states, tuple)
            else hidden_states.device
        )
        if output_router_logits:
            fwd_logits = all_router_logits[0::2]
            bwd_logits = all_router_logits[1::2]

            aux_loss_fwd = load_balancing_loss_func(
                fwd_logits,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                attention_mask,
            )
            aux_loss_bwd = load_balancing_loss_func(
                bwd_logits,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                attention_mask,
            )
            moe_loss = self.config.router_aux_loss_coef * (
                aux_loss_fwd.to(device) + aux_loss_bwd.to(device)
            )  # make sure to reside in the same device
        else:
            moe_loss = 0
        """ ---------------------   add moe_loss output   -------------------------- """

        if not return_dict:
            return hidden_states, moe_loss

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=moe_loss,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if (
            self.config._attn_implementation == "flash_attention_2"
            or self.config._attn_implementation == "flex_attention"
        ):
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1

        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[
                    :,
                    None,
                    None,
                    :,
                ].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                    padding_mask,
                    min_dtype,
                )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


def cross_entropy(logits, y, ignore_index=-100):
    """Cross entropy loss."""
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y, ignore_index=ignore_index)


def weighted_cross_entropy(logits, y, loss_weights, ignore_index=-100):
    """Weighted cross entropy loss (discounts certain tokens, e.g., repeated base pairs in genome)."""
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction="none")
    loss_weights = loss_weights.view(-1)
    loss_weights[y == ignore_index] = 0.0
    # TODO: Follows GPN implementation, but should we remove weight normalization?
    return (ce * (loss_weights / loss_weights.sum())).sum()


# Adapted from Jamba-v0.1 with Jamba->JanusDNA
class JanusDNAForCausalLM(JanusDNAPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: JanusDNAConfig):
        super().__init__(config)
        self.model = JanusDNAModel(config)
        if self.model.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            print("enable checkpoint training")
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(JANUSDNA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # Ignore copy
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: Optional[Union[int, None]] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        The forward pass of the JanusDNA model.

        Returns:
        """

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        def process_hidden_states(self, hidden_states):
            if num_logits_to_keep is None:
                logits = self.lm_head(hidden_states)
            else:
                logits = self.lm_head(hidden_states[..., -num_logits_to_keep:, :])
            logits = logits.float()

            return logits

        if isinstance(hidden_states, tuple) or isinstance(hidden_states, list):
            logits = [process_hidden_states(self, hs) for hs in hidden_states]
        else:
            logits = process_hidden_states(self, hidden_states)

        loss = None
        if labels is not None:
            if isinstance(logits, list):
                loss = sum(
                    cross_entropy(
                        logit.view(-1, logit.size(-1)),
                        labels.view(-1),
                        ignore_index=self.config.pad_token_id,
                    )
                    for logit in logits
                )
            else:
                loss = cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.config.pad_token_id,
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output  # noqa: F821 (vendored verbatim; latent upstream bug in unreached non-return_dict router-logits branch)
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=outputs.router_logits,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


def build_janusdna():
    """Tiny-scale real JanusDNA hybrid bidirectional Mamba/attention/MoE DNA LM."""
    config = JanusDNAConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        attn_layer_period=4,
        attn_layer_offset=3,
        expert_layer_period=2,
        expert_layer_offset=1,
        num_experts=2,
        num_experts_per_tok=1,
        use_mamba_kernels=False,
        mamba_d_state=4,
        mamba_d_conv=2,
        mamba_expand=2,
        bidirectional=True,
        bidirectional_strategy="add",
        layer_fusion=False,
        final_attention=False,
        pad_vocab_size_multiple=8,
        use_cache=False,
        gradient_checkpointing=False,
    )
    return JanusDNAForCausalLM(config)


def example_input_janusdna():
    torch.manual_seed(0)
    return torch.randint(0, 16, (1, 24), dtype=torch.long)


MENAGERIE_ENTRIES = [
    ("JanusDNA", build_janusdna, example_input_janusdna, 2026, MENAGERIE_ZOO),
]
