# SOURCE: vendored from ace-step/ACE-Step @ 1bee4c9f5b43e30995f8d4d33b3919197ce1bd68
# https://raw.githubusercontent.com/ace-step/ACE-Step/main/acestep/models/ace_step_transformer.py
# https://raw.githubusercontent.com/ace-step/ACE-Step/main/acestep/models/attention.py
# https://raw.githubusercontent.com/ace-step/ACE-Step/main/acestep/models/customer_attention_processor.py
# https://raw.githubusercontent.com/ace-step/ACE-Step/main/acestep/models/lyrics_utils/lyric_encoder.py
#
# ACE-Step (StepFun/ACE Studio, 2025): a full-song streaming Diffusion Transformer (DiT)
# for text-to-music generation, jointly conditioned on genre/text embeddings, a speaker
# embedding, and a Conformer-encoded lyric token sequence, with rotary-position linear
# attention (self) + standard SDPA cross-attention and an SSL-distillation projector head
# (MERT / m-hubert). This staging module vendors the transformer backbone
# (`ACEStepTransformer2DModel` + its `Qwen2RotaryEmbedding` / `T2IFinalLayer` / `PatchEmbed`
# helpers), the Sana-style linear-attention transformer block (`LinearTransformerBlock` +
# `GLUMBConv` + `ConvLayer` helpers), the two custom diffusers `Attention` processors
# (`CustomLiteLAProcessor2_0` linear self-attention, `CustomerAttnProcessor2_0` SDPA
# cross-attention), and the lyric Conformer encoder (`ConformerEncoder` + its
# `ConformerEncoderLayer` / `RelPositionMultiHeadedAttention` / `ConvolutionModule` /
# `EspnetRelPositionalEncoding` / `LinearEmbed` building blocks, adapted from the CosyVoice/
# WeNet Conformer lineage per the repo's own comment). This is the real DiT architecture
# used by ACE-Step's `text2music` pipeline (`acestep/pipeline_ace_step.py` constructs and
# calls exactly this `ACEStepTransformer2DModel`); the music VAE/DCAE codec, vocoder,
# language-segmentation, and diffusion-scheduler modules are separate pipeline stages not
# part of this transformer's own forward graph and are out of scope for a module-level
# architecture capture.
#
# All classes/functions below are copied verbatim from the four real repo files listed
# above (only the per-file license headers are de-duplicated into this one header, module-
# relative imports are flattened to a single top-of-file import block with no behavioral
# change, and the `try/except ImportError` dual-path import of `customer_attention_processor`
# in `attention.py` is resolved to the single flattened case since both now live in one
# module). `torch.onnx.is_in_onnx_export()`-style ONNX-export branches are always False for
# a plain eager forward and exercise the real (non-export) code path, exactly as they would
# for the official repo's own eager training/inference forward.
"""ACE-Step: full-song streaming text-to-music Diffusion Transformer -- Qwen2-RoPE linear-
attention DiT backbone (Sana-style LinearTransformerBlock/GLUMBConv) + genre/speaker/lyric
conditioning (lyric Conformer encoder) + SSL-distillation projector heads, vendored from
the official ace-step/ACE-Step PyTorch repo (`acestep/models/ace_step_transformer.py`)."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils import BaseOutput, is_torch_version, logging

MENAGERIE_ZOO = "vendored-pytorch"

logger = logging.get_logger(__name__)


# =========================================================================================
# acestep/models/lyrics_utils/lyric_encoder.py (verbatim)
# =========================================================================================


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: nn.Module = nn.ReLU(),
        norm: str = "batch_norm",
        causal: bool = False,
        bias: bool = True,
    ):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ["batch_norm", "layer_norm"]
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x = nn.functional.pad(x, (self.lorder, 0), "constant", 0.0)
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert x.size(2) > self.lorder
            new_cache = x[:, :, -self.lorder :]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)

        if mask.size(2) > 0:  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, : scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)

        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                CosyVoice.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros(
            (x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # NOTE(Xiang Lyu): Keep rel_shift since espnet rel_pos_emb is used
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache


def subsequent_mask(
    size: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(
    xs: torch.Tensor,
    masks: torch.Tensor,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
    enable_full_context: bool = True,
):
    """Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1,)).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks, (1,)).item()
        chunk_masks = subsequent_chunk_mask(
            xs.size(1), chunk_size, num_left_chunks, xs.device
        )  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(
            xs.size(1), static_chunk_size, num_left_chunks, xs.device
        )  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-5)  # for the CNN module
            self.norm_final = nn.LayerNorm(size, eps=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


class EspnetRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Construct an PositionalEncoding object."""
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(
        self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        """For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - size + 1 : self.pe.size(1) // 2 + size,
        ]
        return pos_emb


class LinearEmbed(torch.nn.Module):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class  # rel_pos_espnet

    def position_encoding(self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)

    def forward(
        self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb


ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}

ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


# https://github.com/FunAudioLLM/CosyVoice/blob/main/examples/magicdata-read/cosyvoice/conf/cosyvoice.yaml
class ConformerEncoder(torch.nn.Module):
    """Conformer encoder module."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 1024,
        attention_heads: int = 16,
        linear_units: int = 4096,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "linear",
        pos_enc_layer_type: str = "rel_pos_espnet",
        normalize_before: bool = True,
        static_chunk_size: int = 1,  # 1: causal_mask; 0: full_mask
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = False,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = False,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility. #'rel_selfattn'
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """
        super().__init__()
        self.output_size = output_size
        self.embed = LinearEmbed(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.gradient_checkpointing = gradient_checkpointing
        self.use_dynamic_chunk = use_dynamic_chunk

        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        activation = ACTIVATION_CLASSES[activation_type]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer_args = (
            output_size,
            cnn_module_kernel,
            activation,
            cnn_module_norm,
            causal,
        )

        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    (PositionwiseFeedForward(*positionwise_layer_args) if macaron_style else None),
                    (ConvolutionModule(*convolution_layer_args) if use_cnn_module else None),
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward_layers(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    @torch.jit.unused
    def forward_layers_checkpointed(
        self,
        xs: torch.Tensor,
        chunk_masks: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = torch.utils.checkpoint.checkpoint(
                layer.__call__, xs, chunk_masks, pos_emb, mask_pad, use_reentrant=False
            )
        return xs

    def forward(
        self,
        xs: torch.Tensor,
        pad_mask: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.size(1)  # noqa: F841 (unused in real source too; kept verbatim)
        masks = pad_mask.to(torch.bool).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb = self.embed(xs)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(
            xs,
            masks,
            self.use_dynamic_chunk,
            self.use_dynamic_left_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
            num_decoding_left_chunks,
        )
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, chunk_masks, pos_emb, mask_pad)
        else:
            xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks


# =========================================================================================
# acestep/models/customer_attention_processor.py (verbatim)
# =========================================================================================

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CustomLiteLAProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections. add rms norm for query and key and apply RoPE"""

    def __init__(self):
        self.kernel_func = nn.ReLU(inplace=False)
        self.eps = 1e-15
        self.pad_val = 1.0

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states_len = hidden_states.shape[1]

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        if encoder_hidden_states is not None:
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        dtype = hidden_states.dtype
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        has_encoder_hidden_state_proj = (
            hasattr(attn, "add_q_proj")
            and hasattr(attn, "add_k_proj")
            and hasattr(attn, "add_v_proj")
        )
        if encoder_hidden_states is not None and has_encoder_hidden_state_proj:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # attention
            if not attn.is_cross_attention:
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
            else:
                query = hidden_states
                key = encoder_hidden_states
                value = encoder_hidden_states

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1)
        key = key.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1).transpose(-1, -2)
        value = value.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1)

        # RoPE需要 [B, H, S, D] 输入
        # 此时 query是 [B, H, D, S], 需要转成 [B, H, S, D] 才能应用RoPE
        query = query.permute(0, 1, 3, 2)  # [B, H, S, D]  (从 [B, H, D, S])

        # Apply query and key normalization if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_freqs_cis is not None:
            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, rotary_freqs_cis)
            elif rotary_freqs_cis_cross is not None and has_encoder_hidden_state_proj:
                key = self.apply_rotary_emb(key, rotary_freqs_cis_cross)

        # 此时 query是 [B, H, S, D]，需要还原成 [B, H, D, S]
        query = query.permute(0, 1, 3, 2)  # [B, H, D, S]

        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, S, 1]
            attention_mask = attention_mask[:, None, :, None].to(key.dtype)  # [B, 1, S, 1]
            query = query * attention_mask.permute(0, 1, 3, 2)  # [B, H, S, D] * [B, 1, S, 1]
            if not attn.is_cross_attention:
                key = key * attention_mask  # key: [B, h, S, D] 与 mask [B, 1, S, 1] 相乘
                value = value * attention_mask.permute(
                    0, 1, 3, 2
                )  # 如果 value 是 [B, h, D, S]，那么需调整mask以匹配S维度

        if (
            attn.is_cross_attention
            and encoder_attention_mask is not None
            and has_encoder_hidden_state_proj
        ):
            encoder_attention_mask = encoder_attention_mask[:, None, :, None].to(
                key.dtype
            )  # [B, 1, S_enc, 1]
            # 此时 key: [B, h, S_enc, D], value: [B, h, D, S_enc]
            key = key * encoder_attention_mask  # [B, h, S_enc, D] * [B, 1, S_enc, 1]
            value = value * encoder_attention_mask.permute(
                0, 1, 3, 2
            )  # [B, h, D, S_enc] * [B, 1, 1, S_enc]

        query = self.kernel_func(query)
        key = self.kernel_func(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=self.pad_val)

        vk = torch.matmul(value, key)

        hidden_states = torch.matmul(vk, query)

        if hidden_states.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.float()

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)

        hidden_states = hidden_states.view(batch_size, attn.heads * head_dim, -1).permute(0, 2, 1)

        hidden_states = hidden_states.to(dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype)

        # Split the attention outputs.
        if (
            encoder_hidden_states is not None
            and not attn.is_cross_attention
            and has_encoder_hidden_state_proj
        ):
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :hidden_states_len],
                hidden_states[:, hidden_states_len:],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if (
            encoder_hidden_states is not None
            and not attn.context_pre_only
            and not attn.is_cross_attention
            and hasattr(attn, "to_add_out")
        ):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if encoder_hidden_states is not None and context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if torch.get_autocast_gpu_dtype() == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return hidden_states, encoder_hidden_states


class CustomerAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        has_encoder_hidden_state_proj = (
            hasattr(attn, "add_q_proj")
            and hasattr(attn, "add_k_proj")
            and hasattr(attn, "add_v_proj")
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_freqs_cis is not None:
            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, rotary_freqs_cis)
            elif rotary_freqs_cis_cross is not None and has_encoder_hidden_state_proj:
                key = self.apply_rotary_emb(key, rotary_freqs_cis_cross)

        if (
            attn.is_cross_attention
            and encoder_attention_mask is not None
            and has_encoder_hidden_state_proj
        ):
            # attention_mask: N x S1
            # encoder_attention_mask: N x S2
            # cross attention 整合attention_mask和encoder_attention_mask
            combined_mask = attention_mask[:, :, None] * encoder_attention_mask[:, None, :]
            attention_mask = torch.where(combined_mask == 1, 0.0, -torch.inf)
            attention_mask = (
                attention_mask[:, None, :, :].expand(-1, attn.heads, -1, -1).to(query.dtype)
            )

        elif not attn.is_cross_attention and attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# =========================================================================================
# acestep/models/attention.py (verbatim; the real file's try/except dual import of
# CustomLiteLAProcessor2_0/CustomerAttnProcessor2_0/Attention from
# customer_attention_processor.py collapses to nothing since both live in this module now)
# =========================================================================================

logger = logging.get_logger(__name__)


def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def get_same_padding(
    kernel_size: Union[int, Tuple[int, ...]],
) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: Union[int, None] = None,
        use_bias=False,
        norm=None,
        act=None,
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )
        if norm is not None:
            self.norm = RMSNorm(out_dim, elementwise_affine=False)
        else:
            self.norm = None
        if act is not None:
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: Union[int, None] = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        out_feature = out_feature or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.glu_act = nn.SiLU(inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.transpose(1, 2)

        return x


class LinearTransformerBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        use_adaln_single=True,
        cross_attention_dim=None,
        added_kv_proj_dim=None,
        context_pre_only=False,
        mlp_ratio=4.0,
        add_cross_attention=False,
        add_cross_attention_dim=None,
        qk_norm=None,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            qk_norm=qk_norm,
            processor=CustomLiteLAProcessor2_0(),
        )

        self.add_cross_attention = add_cross_attention
        self.context_pre_only = context_pre_only

        if add_cross_attention and add_cross_attention_dim is not None:
            self.cross_attn = Attention(
                query_dim=dim,
                cross_attention_dim=add_cross_attention_dim,
                added_kv_proj_dim=add_cross_attention_dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                context_pre_only=context_pre_only,
                bias=True,
                qk_norm=qk_norm,
                processor=CustomerAttnProcessor2_0(),
            )

        self.norm2 = RMSNorm(dim, 1e-06, elementwise_affine=False)

        self.ff = GLUMBConv(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            use_bias=(True, True, False),
            norm=(None, None, None),
            act=("silu", "silu", None),
        )
        self.use_adaln_single = use_adaln_single
        if use_adaln_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        temb: torch.FloatTensor = None,
    ):
        N = hidden_states.shape[0]

        # step 1: AdaLN single
        if self.use_adaln_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + temb.reshape(N, 6, -1)
            ).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states)
        if self.use_adaln_single:
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        # step 2: attention
        if not self.add_cross_attention:
            attn_output, encoder_hidden_states = self.attn(
                hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=rotary_freqs_cis_cross,
            )
        else:
            attn_output, _ = self.attn(
                hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=None,
            )

        if self.use_adaln_single:
            attn_output = gate_msa * attn_output
        hidden_states = attn_output + hidden_states

        if self.add_cross_attention:
            attn_output = self.cross_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=rotary_freqs_cis_cross,
            )
            hidden_states = attn_output + hidden_states

        # step 3: add norm
        norm_hidden_states = self.norm2(hidden_states)
        if self.use_adaln_single:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # step 4: feed forward
        ff_output = self.ff(norm_hidden_states)
        if self.use_adaln_single:
            ff_output = gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        return hidden_states


# =========================================================================================
# acestep/models/ace_step_transformer.py (verbatim)
# =========================================================================================


def cross_norm(hidden_states, controlnet_input):
    # input N x T x c
    mean_hidden_states, std_hidden_states = (
        hidden_states.mean(dim=(1, 2), keepdim=True),
        hidden_states.std(dim=(1, 2), keepdim=True),
    )
    mean_controlnet_input, std_controlnet_input = (
        controlnet_input.mean(dim=(1, 2), keepdim=True),
        controlnet_input.std(dim=(1, 2), keepdim=True),
    )
    controlnet_input = (controlnet_input - mean_controlnet_input) * (
        std_hidden_states / (std_controlnet_input + 1e-12)
    ) + mean_hidden_states
    return controlnet_input


# Copied from transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding with Mixtral->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(
            self.inv_freq
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class T2IFinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(self, hidden_size, patch_size=[16, 1], out_channels=256):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True
        )
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels
        self.patch_size = patch_size

    def unpatchfy(
        self,
        hidden_states: torch.Tensor,
        width: int,
    ):
        # 4 unpatchify
        new_height, new_width = 1, hidden_states.size(1)
        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                new_height,
                new_width,
                self.patch_size[0],
                self.patch_size[1],
                self.out_channels,
            )
        ).contiguous()
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                new_height * self.patch_size[0],
                new_width * self.patch_size[1],
            )
        ).contiguous()
        if width > new_width:
            output = torch.nn.functional.pad(output, (0, width - new_width, 0, 0), "constant", 0)
        elif width < new_width:
            output = output[:, :, :, :width]
        return output

    def forward(self, x, t, output_length):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        # unpatchify
        output = self.unpatchfy(x, output_length)
        return output


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        height=16,
        width=4096,
        patch_size=(16, 1),
        in_channels=8,
        embed_dim=1152,
        bias=True,
    ):
        super().__init__()
        patch_size_h, patch_size_w = patch_size
        self.early_conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 256,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=bias,
            ),
            torch.nn.GroupNorm(
                num_groups=32, num_channels=in_channels * 256, eps=1e-6, affine=True
            ),
            nn.Conv2d(
                in_channels * 256,
                embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
        )
        self.patch_size = patch_size
        self.height, self.width = height // patch_size_h, width // patch_size_w
        self.base_size = self.width

    def forward(self, latent):
        # early convolutions, N x C x H x W -> N x 256 * sqrt(patch_size) x H/patch_size x W/patch_size
        latent = self.early_conv_layers(latent)
        latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return latent


# Real repo: `from .lyrics_utils.lyric_encoder import ConformerEncoder as LyricEncoder`
LyricEncoder = ConformerEncoder


@dataclass
class Transformer2DModelOutput(BaseOutput):
    sample: torch.FloatTensor
    proj_losses: Optional[Tuple[Tuple[str, torch.Tensor]]] = None


class ACEStepTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: Optional[int] = 8,
        num_layers: int = 28,
        inner_dim: int = 1536,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        mlp_ratio: float = 4.0,
        out_channels: int = 8,
        max_position: int = 32768,
        rope_theta: float = 1000000.0,
        speaker_embedding_dim: int = 512,
        text_embedding_dim: int = 768,
        ssl_encoder_depths: List[int] = [9, 9],
        ssl_names: List[str] = ["mert", "m-hubert"],
        ssl_latent_dims: List[int] = [1024, 768],
        lyric_encoder_vocab_size: int = 6681,
        lyric_hidden_size: int = 1024,
        patch_size: List[int] = [16, 1],
        max_height: int = 16,
        max_width: int = 4096,
        **kwargs,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.out_channels = out_channels
        self.max_position = max_position
        self.patch_size = patch_size

        self.rope_theta = rope_theta

        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=self.attention_head_dim,
            max_position_embeddings=self.max_position,
            base=self.rope_theta,
        )

        # 2. Define input layers
        self.in_channels = in_channels

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                LinearTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    add_cross_attention=True,
                    add_cross_attention_dim=self.inner_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )
        self.num_layers = num_layers

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)
        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(self.inner_dim, 6 * self.inner_dim, bias=True)
        )

        # speaker
        self.speaker_embedder = nn.Linear(speaker_embedding_dim, self.inner_dim)

        # genre
        self.genre_embedder = nn.Linear(text_embedding_dim, self.inner_dim)

        # lyric
        self.lyric_embs = nn.Embedding(lyric_encoder_vocab_size, lyric_hidden_size)
        self.lyric_encoder = LyricEncoder(input_size=lyric_hidden_size, static_chunk_size=0)
        self.lyric_proj = nn.Linear(lyric_hidden_size, self.inner_dim)

        projector_dim = 2 * self.inner_dim

        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.inner_dim, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, ssl_dim),
                )
                for ssl_dim in ssl_latent_dims
            ]
        )

        self.ssl_latent_dims = ssl_latent_dims
        self.ssl_encoder_depths = ssl_encoder_depths

        self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction="mean")
        self.ssl_names = ssl_names

        self.proj_in = PatchEmbed(
            height=max_height,
            width=max_width,
            patch_size=patch_size,
            embed_dim=self.inner_dim,
            bias=True,
        )

        self.final_layer = T2IFinalLayer(
            self.inner_dim, patch_size=patch_size, out_channels=out_channels
        )
        self.gradient_checkpointing = False

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward_lyric_encoder(
        self,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
    ):
        # N x T x D
        lyric_embs = self.lyric_embs(lyric_token_idx)
        prompt_prenet_out, _mask = self.lyric_encoder(
            lyric_embs, lyric_mask, decoding_chunk_size=1, num_decoding_left_chunks=-1
        )
        prompt_prenet_out = self.lyric_proj(prompt_prenet_out)
        return prompt_prenet_out

    def encode(
        self,
        encoder_text_hidden_states: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
    ):
        bs = encoder_text_hidden_states.shape[0]
        device = encoder_text_hidden_states.device

        # speaker embedding
        encoder_spk_hidden_states = self.speaker_embedder(speaker_embeds).unsqueeze(1)
        speaker_mask = torch.ones(bs, 1, device=device)

        # genre embedding
        encoder_text_hidden_states = self.genre_embedder(encoder_text_hidden_states)

        # lyric
        encoder_lyric_hidden_states = self.forward_lyric_encoder(
            lyric_token_idx=lyric_token_idx,
            lyric_mask=lyric_mask,
        )

        encoder_hidden_states = torch.cat(
            [
                encoder_spk_hidden_states,
                encoder_text_hidden_states,
                encoder_lyric_hidden_states,
            ],
            dim=1,
        )
        encoder_hidden_mask = torch.cat([speaker_mask, text_attention_mask, lyric_mask], dim=1)
        return encoder_hidden_states, encoder_hidden_mask

    def decode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_mask: torch.Tensor,
        timestep: Optional[torch.Tensor],
        ssl_hidden_states: Optional[List[torch.Tensor]] = None,
        output_length: int = 0,
        block_controlnet_hidden_states: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        controlnet_scale: Union[float, torch.Tensor] = 1.0,
        return_dict: bool = True,
    ):
        embedded_timestep = self.timestep_embedder(
            self.time_proj(timestep).to(dtype=hidden_states.dtype)
        )
        temb = self.t_block(embedded_timestep)

        hidden_states = self.proj_in(hidden_states)

        # controlnet logic
        if block_controlnet_hidden_states is not None:
            control_condi = cross_norm(hidden_states, block_controlnet_hidden_states)
            hidden_states = hidden_states + control_condi * controlnet_scale

        inner_hidden_states = []

        rotary_freqs_cis = self.rotary_emb(hidden_states, seq_len=hidden_states.shape[1])
        encoder_rotary_freqs_cis = self.rotary_emb(
            encoder_hidden_states, seq_len=encoder_hidden_states.shape[1]
        )

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_hidden_mask,
                    rotary_freqs_cis=rotary_freqs_cis,
                    rotary_freqs_cis_cross=encoder_rotary_freqs_cis,
                    temb=temb,
                    use_reentrant=False,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_hidden_mask,
                    rotary_freqs_cis=rotary_freqs_cis,
                    rotary_freqs_cis_cross=encoder_rotary_freqs_cis,
                    temb=temb,
                )

            for ssl_encoder_depth in self.ssl_encoder_depths:
                if index_block == ssl_encoder_depth:
                    inner_hidden_states.append(hidden_states)

        proj_losses = []
        if (
            len(inner_hidden_states) > 0
            and ssl_hidden_states is not None
            and len(ssl_hidden_states) > 0
        ):
            for inner_hidden_state, projector, ssl_hidden_state, ssl_name in zip(
                inner_hidden_states, self.projectors, ssl_hidden_states, self.ssl_names
            ):
                if ssl_hidden_state is None:
                    continue
                # 1. N x T x D1 -> N x D x D2
                est_ssl_hidden_state = projector(inner_hidden_state)
                # 3. projection loss
                bs = inner_hidden_state.shape[0]
                proj_loss = 0.0
                for i, (z, z_tilde) in enumerate(zip(ssl_hidden_state, est_ssl_hidden_state)):
                    # 2. interpolate
                    z_tilde = (
                        F.interpolate(
                            z_tilde.unsqueeze(0).transpose(1, 2),
                            size=len(z),
                            mode="linear",
                            align_corners=False,
                        )
                        .transpose(1, 2)
                        .squeeze(0)
                    )

                    z_tilde = torch.nn.functional.normalize(z_tilde, dim=-1)
                    z = torch.nn.functional.normalize(z, dim=-1)
                    # T x d -> T x 1 -> 1
                    target = torch.ones(z.shape[0], device=z.device)
                    proj_loss += self.cosine_loss(z, z_tilde, target)
                proj_losses.append((ssl_name, proj_loss / bs))

        output = self.final_layer(hidden_states, embedded_timestep, output_length)
        if not return_dict:
            return (output, proj_losses)

        return Transformer2DModelOutput(sample=output, proj_losses=proj_losses)

    # @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_text_hidden_states: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
        timestep: Optional[torch.Tensor] = None,
        ssl_hidden_states: Optional[List[torch.Tensor]] = None,
        block_controlnet_hidden_states: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        controlnet_scale: Union[float, torch.Tensor] = 1.0,
        return_dict: bool = True,
    ):
        encoder_hidden_states, encoder_hidden_mask = self.encode(
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=lyric_token_idx,
            lyric_mask=lyric_mask,
        )

        output_length = hidden_states.shape[-1]

        output = self.decode(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_hidden_mask,
            timestep=timestep,
            ssl_hidden_states=ssl_hidden_states,
            output_length=output_length,
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            controlnet_scale=controlnet_scale,
            return_dict=return_dict,
        )

        return output


def build_ace_step_transformer():
    # Tiny random-init config (real defaults: num_layers=28, inner_dim=1536,
    # lyric_encoder_vocab_size=6681, lyric ConformerEncoder num_blocks=6/output_size=1024 --
    # shrunk here purely as *construction parameters* passed to the real, unmodified
    # ACEStepTransformer2DModel/ConformerEncoder classes; no architecture code changed).
    torch.manual_seed(0)
    model = ACEStepTransformer2DModel(
        in_channels=8,
        num_layers=2,
        attention_head_dim=32,
        num_attention_heads=4,
        mlp_ratio=2.0,
        out_channels=8,
        max_position=512,
        speaker_embedding_dim=32,
        text_embedding_dim=32,
        ssl_encoder_depths=[1],
        ssl_names=["mert"],
        ssl_latent_dims=[32],
        lyric_encoder_vocab_size=64,
        lyric_hidden_size=32,
        patch_size=[16, 1],
        max_height=16,
        max_width=256,
    )
    # Shrink the lyric ConformerEncoder to a tiny config too (real default: 6 blocks,
    # 1024-dim, 16 heads, 4096 FFN -- constructed directly here with the same real
    # ConformerEncoder class, just smaller hyperparameters).
    model.lyric_encoder = ConformerEncoder(
        input_size=32,
        output_size=32,
        attention_heads=2,
        linear_units=64,
        num_blocks=1,
        static_chunk_size=0,
    )
    model.lyric_proj = nn.Linear(32, model.inner_dim)
    model.eval()
    return model


def example_input_ace_step_transformer():
    torch.manual_seed(0)
    batch = 1
    latent_len = 32
    text_len = 6
    lyric_len = 8
    hidden_states = torch.randn(batch, 8, 16, latent_len)
    attention_mask = torch.ones(batch, latent_len)
    encoder_text_hidden_states = torch.randn(batch, text_len, 32)
    text_attention_mask = torch.ones(batch, text_len)
    speaker_embeds = torch.randn(batch, 32)
    lyric_token_idx = torch.randint(0, 64, (batch, lyric_len))
    lyric_mask = torch.ones(batch, lyric_len)
    timestep = torch.tensor([500.0])
    return (
        hidden_states,
        attention_mask,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embeds,
        lyric_token_idx,
        lyric_mask,
        timestep,
    )


MENAGERIE_ENTRIES = [
    (
        "ACE-Step",
        "build_ace_step_transformer",
        "example_input_ace_step_transformer",
        2025,
        "vendored",
    ),
]
